"""Fetches each corpus record's Europe PMC identity, preprint server and data-links summary in one
batched `core` search pass -- the layer everything else keys on.

Per record it captures, from the `resultType=core` search response:

- `source` and `id` -> `source.epmc_source` / `identifiers.epmc_id` (schema v1.3.0). This is the
  `(source, id)` pair every per-record Europe PMC endpoint (`/datalinks`, the annotations API) is
  addressed by, which is why this pass runs before either;
- `bookOrReportDetails.publisher` -> `publication_metadata.preprint_server` (v1.3.0);
- `hasData`, `dataLinksTagsList`, `tmAccessionTypeList`, `dbCrossReferenceList`,
  `hasTMAccessionNumbers`, `hasDbCrossReferences`, `hasSuppl` -> the `data_links` summary
  (v1.4.0) and the pre-filters for the link fetches. `hasData` and `dataLinksTagsList` exist only in
  `core` (verified live, 2026-09-14), which is why this is not a `lite` fetch.

Batching and keys are `fetch_citations.py`'s: up to 300 PMIDs / 120 DOIs / 200 PMCIDs per request,
three passes, `pmid -> doi -> pmcid`. **Preprints are the exception and the point.** A preprint
later published in a journal resolves by DOI to both a PPR and a MED record, and `pick_best()`
prefers MED -- the right answer for a citation count and exactly the wrong one here
(docs/preprint.md §1). So a corpus document whose `pub_types` says Preprint is keyed by its DOI
under `AND SRC:PPR`, in its own pass, and the PPR record is taken.

**DOIs Europe PMC answers without echoing them.** A DOI-keyed chunk often comes back as a PMC-source
record carrying no `doi` field (measured 2026-09-14: 90% of the plain doi pass), which cannot be
attributed to one of 120 keys. Every key a chunk leaves unanswered is asked again: first, where our
corpus row also holds a pmcid, through that pmcid in batches of 200 (a PMC record always echoes its
pmcid, and the pmcid pass answers ~100%); then the rest one key per request, where the only record
returned is unambiguously its answer. A single quoted-DOI `core` query is slow on Europe PMC's side
(~3 calls/s at 64 workers, measured), which is why the batched route goes first.
`--resolve-misses` does the same for the misses already in an output file.

Output: `output/epmc_metadata.csv`, keyed like the citation file (`key_type, key`) so
`build_data_links.py` joins it to document ids from OUR corpus, never from Europe PMC's echo.
Resumable: keys already in the file are skipped unless `--max-age-days` says they are stale. A key
Europe PMC did not return is written as a row with a blank `epmc_id` and `has_data` "N"
("looked up, nothing"), so it is not re-fetched forever.

    python3 fetch_epmc_metadata.py --input ../output/corpus_keys.csv --limit 3000   # sample + rate
    python3 fetch_epmc_metadata.py --input ../output/corpus_keys.csv               # the corpus
    python3 fetch_epmc_metadata.py --input ../output/incoming_new.csv              # a batch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from fetch_citations import (
    CHUNK_SIZES,
    EPMC_SEARCH_URL,
    _key_of,
    _session,
    assign_key,
    build_query,
    pick_best,
)

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_INPUT = FOLDER_DIR / "output" / "corpus_keys.csv"
DEFAULT_OUTPUT = FOLDER_DIR / "output" / "epmc_metadata.csv"

# The preprint pass: DOI-keyed, source-restricted. Same URI ceiling as the doi pass.
PPR_KEY_TYPE = "ppr_doi"
PASS_ORDER = (PPR_KEY_TYPE, "pmid", "doi", "pmcid")
PASS_SPECS = {
    PPR_KEY_TYPE: ("doi", "PPR", CHUNK_SIZES["doi"]),
    "pmid": ("pmid", None, CHUNK_SIZES["pmid"]),
    "doi": ("doi", None, CHUNK_SIZES["doi"]),
    "pmcid": ("pmcid", None, CHUNK_SIZES["pmcid"]),
}
DEFAULT_MAX_WORKERS = 64
REQUEST_TIMEOUT = 60  # core responses for 300 records are ~1 MB; batched calls get more patience
PREPRINT_PUB_TYPES = {"Preprint", "preprint"}

OUTPUT_COLUMNS = [
    "key_type", "key", "epmc_source", "epmc_id", "pmid", "pmcid", "doi", "preprint_server",
    "has_data", "data_links_tags", "accession_types", "db_cross_references",
    "has_tm_accessions", "has_db_xrefs", "has_suppl", "fetched_at",
]


def is_preprint_row(row: dict) -> bool:
    """A corpus export says `is_preprint`; a staging CSV carries `pub_types` as a JSON list."""
    flag = (row.get("is_preprint") or "").strip().lower()
    if flag in ("true", "false"):
        return flag == "true"
    raw = (row.get("pub_types") or "").strip()
    if raw:
        try:
            return bool(PREPRINT_PUB_TYPES & set(json.loads(raw)))
        except (ValueError, TypeError):
            return False
    return False


def metadata_key(row: dict) -> tuple[str, str] | None:
    """Which single key this record is fetched (and later joined) by. A preprint with a DOI goes
    through the PPR pass; everything else follows `assign_key`. THE one rule, imported by
    `build_data_links.py` rather than reimplemented, so fetch and join cannot disagree."""
    pmid = (row.get("pmid") or "").strip()
    pmcid = (row.get("pmcid") or "").strip()
    doi = (row.get("doi") or "").strip()
    if is_preprint_row(row) and doi:
        return (PPR_KEY_TYPE, doi.lower())
    return assign_key(pmid, pmcid, doi)


def select_record(key_type: str, records: list[dict]) -> dict:
    """The PPR pass takes the preprint's own record, deterministically; every other pass keeps
    `pick_best()`'s MED-first rule."""
    if key_type == PPR_KEY_TYPE:
        pprs = [r for r in records if r.get("source") == "PPR"]
        if pprs:
            return sorted(pprs, key=lambda r: (-(r.get("citedByCount") or 0),
                                               str(r.get("id") or "")))[0]
    return pick_best(records)


def _json_list(value) -> str:
    return json.dumps(list(value or []))


SUMMARY_COLUMNS = ("has_data", "data_links_tags", "accession_types", "db_cross_references",
                   "has_tm_accessions", "has_db_xrefs", "has_suppl")


def identity_fields(record: dict) -> dict:
    """The v1.3.0 capture from one `core` search record: Europe PMC's own source and id, and the
    preprint server. Shared with `build_incoming_documents.py` so a new record and a backfilled
    one are read the same way.

    `bookOrReportDetails.publisher` is the preprint server ONLY on a PPR record. Europe PMC fills
    it for other sources too (a thesis's university, a book's publisher: 344 non-PPR corpus records
    on 2026-09-14), and writing those into `preprint_server` would label them preprints."""
    source = record.get("source") or ""
    publisher = (record.get("bookOrReportDetails") or {}).get("publisher") or ""
    return {
        "epmc_source": source,
        "epmc_id": str(record.get("id") or ""),
        "preprint_server": publisher if source == "PPR" else "",
    }


def summary_fields(record: dict) -> dict:
    """The data_links summary from one `core` search record (the seven SUMMARY_COLUMNS). A record
    Europe PMC returned without a flag has no data: "N" is the honest answer, and it keeps the
    key from being re-fetched forever."""
    flags = record.get
    return {
        "has_data": "Y" if flags("hasData") == "Y" else "N",
        "data_links_tags": _json_list((record.get("dataLinksTagsList") or {}).get("dataLinkstag")),
        "accession_types": _json_list((record.get("tmAccessionTypeList") or {}).get("accessionType")),
        "db_cross_references": _json_list((record.get("dbCrossReferenceList") or {}).get("dbName")),
        "has_tm_accessions": "Y" if flags("hasTMAccessionNumbers") == "Y" else "N",
        "has_db_xrefs": "Y" if flags("hasDbCrossReferences") == "Y" else "N",
        "has_suppl": "Y" if flags("hasSuppl") == "Y" else "N",
    }


def record_to_row(key_type: str, key: str, record: dict, fetched_at: str) -> dict:
    return {
        "key_type": key_type,
        "key": key,
        **identity_fields(record),
        "pmid": record.get("pmid") or "",
        "pmcid": record.get("pmcid") or "",
        "doi": record.get("doi") or "",
        **summary_fields(record),
        "fetched_at": fetched_at,
    }


def miss_row(key_type: str, key: str, fetched_at: str) -> dict:
    """Europe PMC was asked and returned nothing attributable: recorded, not dropped."""
    row = {column: "" for column in OUTPUT_COLUMNS}
    row.update({"key_type": key_type, "key": key, "has_data": "N", "data_links_tags": "[]",
                "accession_types": "[]", "db_cross_references": "[]", "has_tm_accessions": "N",
                "has_db_xrefs": "N", "has_suppl": "N", "fetched_at": fetched_at})
    return row


def fetch_chunk(session: requests.Session, key_type: str, keys: list[str],
                fetched_at: str) -> list[dict]:
    clause_type, src, _ = PASS_SPECS[key_type]
    resp = session.get(
        EPMC_SEARCH_URL,
        params={"query": build_query(clause_type, keys, src), "pageSize": 1000,
                "format": "json", "resultType": "core"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("resultList", {}).get("result", [])

    wanted = {k.lower() if clause_type == "doi" else k for k in keys}
    grouped: dict[str, list[dict]] = {}
    for record in results:
        key = _key_of(record, clause_type)
        if key in wanted:
            grouped.setdefault(key, []).append(record)

    rows = [record_to_row(key_type, key, select_record(key_type, records), fetched_at)
            for key, records in grouped.items()]
    answered = {r["key"] for r in rows}
    for key in keys:
        normalised = key.lower() if clause_type == "doi" else key
        if normalised not in answered:
            rows.append(miss_row(key_type, normalised, fetched_at))
    return rows


def fetch_single(session: requests.Session, key_type: str, key: str, fetched_at: str) -> dict:
    """One key per request. Every record returned is an answer for this key unless it echoes a
    DIFFERENT identifier; an un-echoed record (the PMC-source case) is kept."""
    clause_type, src, _ = PASS_SPECS[key_type]
    resp = session.get(
        EPMC_SEARCH_URL,
        params={"query": build_query(clause_type, [key], src), "pageSize": 25,
                "format": "json", "resultType": "core"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    wanted = key.lower() if clause_type == "doi" else key
    results = [r for r in resp.json().get("resultList", {}).get("result", [])
               if _key_of(r, clause_type) in (wanted, "")]
    if not results:
        return miss_row(key_type, wanted, fetched_at)
    return record_to_row(key_type, wanted, select_record(key_type, results), fetched_at)


def load_pmcid_alternates(input_path: Path) -> dict[tuple[str, str], str]:
    """(key_type, key) -> the record's pmcid, for plain-doi keys whose input row also carries one.
    Preprint keys are left out on purpose: the PPR record is wanted there, never the PMC one."""
    alternates: dict[tuple[str, str], str] = {}
    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            key = metadata_key(row)
            pmcid = (row.get("pmcid") or "").strip()
            if key is not None and key[0] == "doi" and pmcid:
                alternates[key] = pmcid
    return alternates


def fetch_via_pmcid(session: requests.Session, pairs: list[tuple[tuple[str, str], str]],
                    fetched_at: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """Answers doi keys through their pmcids, one request for up to 200. The row stays keyed by
    the doi (the join key); only the lookup used the pmcid. Returns (answered rows, keys left)."""
    resp = session.get(
        EPMC_SEARCH_URL,
        params={"query": build_query("pmcid", [pmcid for _, pmcid in pairs]), "pageSize": 1000,
                "format": "json", "resultType": "core"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    grouped: dict[str, list[dict]] = {}
    for record in resp.json().get("resultList", {}).get("result", []):
        grouped.setdefault(record.get("pmcid") or "", []).append(record)
    rows, left = [], []
    for (key_type, key), pmcid in pairs:
        records = grouped.get(pmcid)
        if records:
            rows.append(record_to_row(key_type, key, pick_best(records), fetched_at))
        else:
            left.append((key_type, key))
    return rows, left


def resolve(session: requests.Session, pairs: list[tuple[str, str]],
            alternates: dict[tuple[str, str], str], fetched_at: str, writer: csv.DictWriter, sink,
            max_workers: int) -> tuple[int, int, int]:
    """Batched pmcid route for the keys that have one, then one key per request for the rest.
    Returns (calls, resolved, errors)."""
    via = [(key, alternates[key]) for key in pairs if key in alternates]
    rest = [key for key in pairs if key not in alternates]
    size = CHUNK_SIZES["pmcid"]
    chunks = [via[i:i + size] for i in range(0, len(via), size)]
    n_calls = n_resolved = n_errors = 0
    if chunks:
        print(f"fetch_epmc_metadata: {len(via):,} keys through their pmcid in {len(chunks):,} "
              f"calls; {len(rest):,} have none")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_via_pmcid, session, chunk, fetched_at): chunk
                       for chunk in chunks}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="metadata[via pmcid]", unit="chunk"):
                n_calls += 1
                chunk = futures[future]
                try:
                    rows, left = future.result()
                except Exception as exc:  # noqa: BLE001 -- fall through to one by one
                    n_errors += 1
                    tqdm.write(f"  pmcid chunk FAILED: {exc!r}"[:200])
                    rest.extend(key for key, _ in chunk)
                    continue
                for row in rows:
                    writer.writerow(row)
                n_resolved += len(rows)
                rest.extend(left)
                sink.flush()
    if rest:
        print(f"fetch_epmc_metadata: {len(rest):,} keys asked one by one")
        calls, resolved, errors = resolve_one_by_one(session, rest, fetched_at, writer, sink,
                                                     max_workers)
        n_calls += calls
        n_resolved += resolved
        n_errors += errors
    return n_calls, n_resolved, n_errors


def resolve_one_by_one(session: requests.Session, pairs: list[tuple[str, str]], fetched_at: str,
                       writer: csv.DictWriter, sink, max_workers: int) -> tuple[int, int, int]:
    """Asks each (key_type, key) on its own; appends every answer. Returns (calls, resolved, errors)."""
    n_calls = n_resolved = n_errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_single, session, kt, k, fetched_at): (kt, k) for kt, k in pairs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="metadata[single]",
                           unit="call"):
            n_calls += 1
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001 -- re-run with --resolve-misses retries them
                n_errors += 1
                if n_errors <= 20:
                    tqdm.write(f"  {futures[future]} FAILED: {exc!r}"[:200])
                continue
            writer.writerow(row)
            n_resolved += bool(row["epmc_id"])
            if n_calls % 500 == 0:
                sink.flush()
    sink.flush()
    return n_calls, n_resolved, n_errors


def load_misses(output_path: Path, targets: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Keys of this input whose LATEST row in the output file has no Europe PMC identity."""
    wanted = {(kt, k) for kt, keys in targets.items() for k in keys}
    latest: dict[tuple[str, str], str] = {}
    if output_path.exists():
        with output_path.open(newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                key = (row.get("key_type") or "", row.get("key") or "")
                if key in wanted:
                    latest[key] = row.get("epmc_id") or ""
    return sorted(k for k, epmc_id in latest.items() if not epmc_id)


def load_targets(input_path: Path) -> dict[str, list[str]]:
    targets: dict[str, set[str]] = {kt: set() for kt in PASS_ORDER}
    n = 0
    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            n += 1
            assigned = metadata_key(row)
            if assigned is None:
                continue
            targets[assigned[0]].add(assigned[1])
    print(f"fetch_epmc_metadata: {input_path.name}: {n:,} rows -> " +
          ", ".join(f"{kt} {len(v):,}" for kt, v in targets.items()))
    return {kt: sorted(v) for kt, v in targets.items()}


def load_already_fetched(output_path: Path, max_age_days: int | None) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    frame = pd.read_csv(output_path, dtype=str, usecols=["key_type", "key", "fetched_at"]).fillna("")
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        fetched = pd.to_datetime(frame["fetched_at"], errors="coerce", utc=True)
        frame = frame[fetched >= cutoff]
        print(f"fetch_epmc_metadata: --max-age-days {max_age_days} -> {len(frame):,} rows fresh")
    return set(zip(frame["key_type"], frame["key"]))


def run(input_path: Path, output_path: Path, max_workers: int, limit: int | None,
        max_age_days: int | None, resolve_misses: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = load_targets(input_path)
    if resolve_misses:
        pairs = load_misses(output_path, targets)
        if limit is not None:
            pairs = sorted(random.Random(42).sample(pairs, min(limit, len(pairs))))
        print(f"fetch_epmc_metadata: --resolve-misses -> {len(pairs):,} keys asked one by one")
        if not pairs:
            return
        session = _session(max_workers)
        started = time.time()
        alternates = load_pmcid_alternates(input_path)
        with output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            calls, resolved, errors = resolve(
                session, pairs, alternates, datetime.now(timezone.utc).isoformat(), writer, f,
                max_workers)
        session.close()
        elapsed = time.time() - started
        print(f"fetch_epmc_metadata: resolved {resolved:,} of {len(pairs):,} keys with {calls:,} "
              f"calls in {elapsed:.1f}s ({calls / elapsed if elapsed else 0:.1f} calls/s), "
              f"{errors} failed call(s)")
        return
    already = load_already_fetched(output_path, max_age_days)
    remaining = {kt: [k for k in keys if (kt, k) not in already] for kt, keys in targets.items()}
    total_remaining = sum(len(v) for v in remaining.values())
    print(f"fetch_epmc_metadata: {len(already):,} already fetched | {total_remaining:,} remaining")
    if limit is not None:
        rng = random.Random(42)  # a seeded sample, never a lexical prefix (fetch_citations.py)
        for kt in remaining:
            remaining[kt] = sorted(rng.sample(remaining[kt], min(limit, len(remaining[kt]))))
        total_remaining = sum(len(v) for v in remaining.values())
        print(f"fetch_epmc_metadata: --limit {limit} -> {total_remaining:,} keys this run")
    if total_remaining == 0:
        print("fetch_epmc_metadata: nothing to do.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    header_needed = not output_path.exists()
    n_written = n_hits = n_calls = n_errors = 0
    missed: list[tuple[str, str]] = []
    started = time.time()

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if header_needed:
            writer.writeheader()
        f.flush()
        for key_type in PASS_ORDER:
            keys = remaining[key_type]
            if not keys:
                continue
            size = PASS_SPECS[key_type][2]
            chunks = [keys[i:i + size] for i in range(0, len(keys), size)]
            print(f"\nfetch_epmc_metadata: pass '{key_type}' -- {len(keys):,} keys in "
                  f"{len(chunks):,} chunks of {size}, {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(fetch_chunk, session, key_type, chunk, fetched_at): chunk
                           for chunk in chunks}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"metadata[{key_type}]", unit="chunk"):
                    chunk = futures[future]
                    n_calls += 1
                    try:
                        rows = future.result()
                    except Exception as exc:  # noqa: BLE001 -- network errors; keep going
                        n_errors += 1
                        tqdm.write(f"  chunk starting {chunk[0]} FAILED: {exc!r}"[:200])
                        continue
                    for row in rows:
                        writer.writerow(row)
                        if not row["epmc_id"]:
                            missed.append((key_type, row["key"]))
                    n_written += len(rows)
                    n_hits += sum(1 for r in rows if r["epmc_id"])
                    f.flush()

        if missed:
            print(f"\nfetch_epmc_metadata: {len(missed):,} keys unanswered in their chunk -- asking "
                  f"again (a chunk cannot attribute a record that does not echo its key)")
            calls, resolved, errors = resolve(session, missed, load_pmcid_alternates(input_path),
                                              fetched_at, writer, f, max_workers)
            n_calls += calls
            n_errors += errors
            n_hits += resolved

    session.close()
    elapsed = time.time() - started
    print(f"\nfetch_epmc_metadata: done -- {n_calls:,} calls in {elapsed:.1f}s "
          f"({n_calls / elapsed if elapsed else 0:.1f} calls/s, "
          f"{n_written / elapsed if elapsed else 0:.0f} records/s), {n_errors} failed chunk(s) "
          f"({n_errors / n_calls * 100 if n_calls else 0:.1f}%). {n_hits:,} of {n_written:,} keys "
          f"answered. Re-run the same command to retry failed chunks; answered keys are skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="corpus_keys.csv (export_corpus_keys.py) or a staged incoming CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="At most N keys per pass, a seeded random sample. Prints calls/s.")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Re-fetch keys whose last fetch is older than this.")
    parser.add_argument("--resolve-misses", action="store_true",
                        help="Ask every key whose latest row in --output has no identity again, "
                             "one key per request, and append the answers.")
    args = parser.parse_args()
    run(args.input, args.output, args.max_workers, args.limit, args.max_age_days,
        args.resolve_misses)


if __name__ == "__main__":
    main()
