"""Fetches real citation counts from Europe PMC for every record in the corpus.

`publication_metadata.citation_count` has been a required-but-null field on all 827,061 documents
since the collection was created -- reserved, plumbed through to the API's `citations_desc` sort
and the UI's "Most cited" option, and never populated. This closes that.

Modelled directly on `epmc_licensing/fetch_licensing.py` (same retrying session, same
stream-and-flush-per-batch resumability, same tqdm), with three differences that were all measured
live against the real API on 2026-09-03 rather than assumed:

1. **`resultType=lite` is enough.** The licence fetch needs `core` because `lite` carries no
   `license` field at all; `citedByCount` *is* in `lite` (verified: AlphaFold 2 = 34,984), and
   `lite` responses are a fraction of the size.
2. **Three keyed passes, not one.** 67,985 records have no PMID, so a PMID-only fetch would leave
   8.2% of the corpus null forever. Coverage is pmid 91.78% / doi 99.15% / pmcid 75.45%, with 0
   records carrying no usable identifier at all, so `pmid -> doi -> pmcid` reaches all of it.
3. **Per-pass chunk ceilings, binary-searched live.** The wall is a URI *byte* limit, not a count,
   so it differs per key type:

   | pass | clause | measured ceiling | used |
   |---|---|---|---|
   | pmid | `EXT_ID:34265844` (unquoted -- numeric) | 360 ok / 370 -> 400 / 390+ -> 414 | 300 |
   | doi | `DOI:"10.1038/..."` (quoted) | 150 ok (6.4KB URI) / 200 -> 414 (8.5KB) | 120 |
   | pmcid | `PMCID:PMC8371605` | not pushed -- population is tiny | 200 |

Measured yield per pass (2026-09-03, real populations): **pmid 100%**, **pmcid 100%**,
**doi 80.0%** on a seeded random sample of the real no-pmid population. The doi shortfall is
genuine EPMC coverage, not a query bug -- case was ruled out (94/120 identical for original-case
and lowercased spellings, and EPMC's DOI index is demonstrably case-insensitive). Projected
end-state coverage is therefore ~98.4% of the 835,500 distinct keys.

**Multi-source hits are real and must be resolved.** A DOI query for 50 DOIs returned 51 records:
one DOI existed as both a MED article and a PPR preprint. Preference is
`MED > PMC > PPR > AGR > PAT`, applied deterministically, and the winning source is recorded per
row so the choice is auditable rather than implicit.

Resumable and refresh-aware: `--max-age-days N` skips any key already fetched within N days, which
is what makes the monthly refresh cheap instead of re-fetching 830k counts every time.

    python3 fetch_citations.py --limit 2000          # smoke check
    python3 fetch_citations.py                       # full corpus + the curated set
    python3 fetch_citations.py --max-age-days 30     # monthly refresh: only stale entries
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

csv.field_size_limit(sys.maxsize)

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
REPO_DIR = FOLDER_DIR.parent

# Both populations, so the curated records that are about to be merged get counts in the same run.
DEFAULT_INPUTS = (
    REPO_DIR / "mongo_landscape_export" / "ai_ml_landscape_classified_usable.csv",
    REPO_DIR / "data" / "processed" / "canonical_dataset.csv",
)
DEFAULT_OUTPUT = FOLDER_DIR / "output" / "epmc_citations.csv"

CITATION_SOURCE = "europepmc"
OUTPUT_COLUMNS = [
    "key_type", "key", "epmc_source", "pmid", "pmcid", "doi", "citation_count",
    "citation_source", "fetched_at",
    # Only populated under --with-licence, which switches resultType to "core". Appended rather
    # than inserted so a reader of an older file is unaffected.
    "license", "epmc_is_open_access",
]

# See the module docstring's table -- each of these is a measured ceiling with margin, not a guess.
CHUNK_SIZES = {"pmid": 300, "doi": 120, "pmcid": 200}
DEFAULT_MAX_WORKERS = 35

# Which EPMC record wins when one identifier resolves to several. MED is the published article of
# record; PPR is a preprint of (often) the same work and carries its own, lower, count.
SOURCE_PREFERENCE = ("MED", "PMC", "PPR", "AGR", "PAT")
_SOURCE_RANK = {src: i for i, src in enumerate(SOURCE_PREFERENCE)}


def _session(max_workers: int) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retry, pool_connections=max_workers, pool_maxsize=max_workers * 2
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def build_clause(key_type: str, key: str) -> str:
    """PMIDs are purely numeric and must NOT be quoted -- a quoted single-clause chunk combined
    with `AND SRC:MED` silently returns 0 hits on EPMC's Lucene parser (documented in
    `dome_triage/ingest/epmc_client.py` after a real incident). DOIs contain `/` and `.` and must
    be quoted."""
    if key_type == "pmid":
        return f"EXT_ID:{key}"
    if key_type == "doi":
        return f'DOI:"{key}"'
    if key_type == "pmcid":
        return f"PMCID:{key}"
    raise ValueError(f"unknown key type {key_type!r}")


def build_query(key_type: str, keys: list[str], src: str | None = None) -> str:
    """`src` overrides the source restriction: `fetch_epmc_metadata.py` passes "PPR" to fetch a
    preprint's own record by DOI, because for a preprint later published in a journal the DOI
    resolves to both a PPR and a MED record and `pick_best()` would take the MED one
    (docs/preprint.md, "The one trap"). Left None, the default rule below applies unchanged."""
    clauses = " OR ".join(build_clause(key_type, k) for k in keys)
    if src is not None:
        return f"({clauses}) AND SRC:{src}"
    # SRC:MED only on the pmid pass: a PMID *is* a MEDLINE identifier, so the restriction is free
    # and keeps the result set to one record per key. Restricting the doi/pmcid passes would
    # exclude exactly the preprint and PMC-only records those passes exist to reach.
    return f"({clauses}) AND SRC:MED" if key_type == "pmid" else f"({clauses})"


def pick_best(records: list[dict]) -> dict:
    """Deterministic winner among several EPMC records for one identifier: preferred source first,
    then the higher count, then the id, so the result never depends on response ordering."""
    return sorted(
        records,
        key=lambda r: (
            _SOURCE_RANK.get(r.get("source"), len(SOURCE_PREFERENCE)),
            -(r.get("citedByCount") or 0),
            str(r.get("id") or ""),
        ),
    )[0]


def _key_of(record: dict, key_type: str) -> str:
    value = record.get(key_type) or ""
    return value.lower() if key_type == "doi" else value


def fetch_chunk(
    session: requests.Session, key_type: str, keys: list[str], fetched_at: str,
    with_licence: bool = False,
) -> list[dict]:
    """One HTTP request for up to CHUNK_SIZES[key_type] keys. Keys EPMC does not know are simply
    absent from the result -- never an error, same convention as `EpmcClient.get_by_ids`.

    `with_licence` switches `resultType` from "lite" to "core", which is required rather than
    preferred: confirmed live that `lite` carries no `license` field at all, only `isOpenAccess`.
    `core` carries both that and `citedByCount`, so one pass answers both questions."""
    resp = session.get(
        EPMC_SEARCH_URL,
        params={
            "query": build_query(key_type, keys),
            "pageSize": 1000,
            "format": "json",
            "resultType": "core" if with_licence else "lite",
        },
        timeout=90,
    )
    resp.raise_for_status()
    results = resp.json().get("resultList", {}).get("result", [])

    wanted = {k.lower() if key_type == "doi" else k for k in keys}
    grouped: dict[str, list[dict]] = {}
    for record in results:
        key = _key_of(record, key_type)
        if key in wanted:
            grouped.setdefault(key, []).append(record)

    rows = []
    for key, records in grouped.items():
        best = pick_best(records)
        count = best.get("citedByCount")
        row = {
            "key_type": key_type,
            "key": key,
            "epmc_source": best.get("source") or "",
            "pmid": best.get("pmid") or "",
            "pmcid": best.get("pmcid") or "",
            "doi": best.get("doi") or "",
            "citation_count": "" if count is None else int(count),
            "citation_source": CITATION_SOURCE,
            "fetched_at": fetched_at,
            "license": "", "epmc_is_open_access": "",
        }
        if with_licence:
            # "" is a real answer here, not a blank: it means EPMC was asked and disclosed no
            # licence, which the schema distinguishes from null ("never looked up"). A record
            # that reached this line WAS looked up.
            row["license"] = best.get("license") or ""
            row["epmc_is_open_access"] = best.get("isOpenAccess") or ""
        rows.append(row)

    if with_licence:
        # Every key that produced no attributable row still got a real answer: we asked, and no
        # licence came back. Recording that as "" rather than dropping the key is what stops the
        # document staying null and being re-fetched on every future backfill, forever.
        #
        # Two distinct causes, and "" is the honest answer to both. Some keys EPMC genuinely does
        # not have. Others it *does* have but returns without echoing the identifier they were
        # fetched by -- measured on 12,980 keys, typically a PMC-source record for a DOI-keyed
        # lookup, which carries neither `doi` nor `license` in either resultType (checked both).
        # With up to 120 keys in a chunk an un-echoed record cannot be attributed to one of them,
        # so it is not usable as a count; but "EPMC disclosed no licence for this key" is true
        # either way, and that is exactly what `source.access.license: ""` means in this schema.
        answered = {r["key"] for r in rows}
        for key in keys:
            normalised = key.lower() if key_type == "doi" else key
            if normalised in answered:
                continue
            rows.append({
                "key_type": key_type, "key": normalised, "epmc_source": "",
                "pmid": "", "pmcid": "", "doi": "",
                "citation_count": "", "citation_source": "", "fetched_at": fetched_at,
                "license": "", "epmc_is_open_access": "",
            })
    return rows


def assign_key(pmid: str, pmcid: str, doi: str) -> tuple[str, str] | None:
    """Which single EPMC key this record is looked up by: pmid if it has one, else doi, else
    pmcid. Returns None for a record with no usable identifier (measured: 0 in the corpus).

    THE one source of truth for this rule -- `join_citations.py` imports it rather than
    reimplementing it, because a divergence would join the right counts to the wrong records and
    nothing downstream would notice.

    DOIs are lowercased because EPMC returns them in mixed case; the API itself is
    case-insensitive (verified live), so this is purely about making our own join key stable.
    """
    pmid, pmcid, doi = pmid.strip(), pmcid.strip(), doi.strip()
    if pmid:
        return ("pmid", pmid)
    if doi:
        return ("doi", doi.lower())
    if pmcid:
        return ("pmcid", pmcid)
    return None


MOROS_MISSING_LICENCE = "moros:missing-licence"


def load_targets_from_moros_missing_licence() -> dict[str, list[str]]:
    """Every document whose licence was NEVER looked up, keyed the same way `load_targets` keys a
    CSV, so the two are interchangeable downstream.

    `source.access.license: null` means "never looked up" and is distinct from `""`, which means
    "looked up, EPMC disclosed none". Only the former needs fetching -- and it must be targeted
    from the collection rather than a hand-kept list, because the gap has two unrelated causes and
    a list would capture only whichever one someone had in mind.

    Measured 2026-09-03: 74,472 documents, of which 68,103 have no pmid at all. The original
    `epmc_licensing/fetch_licensing.py` builds `EXT_ID:` clauses under `SRC:MED`, so those 68,103
    were structurally unfetchable from the day it ran -- the summary recorded them as unmatched,
    which reads like an EPMC coverage limit rather than a key-choice one."""
    from moros_client import Moros

    pmids: set[str] = set()
    dois: set[str] = set()
    pmcids: set[str] = set()
    with Moros.from_env() as moros:
        print(f"fetch_citations: targets from {moros.describe()}")
        cursor = moros.collection.find(
            {"source.access.license": None},
            {"identifiers.pmid": 1, "identifiers.pmcid": 1, "identifiers.doi": 1},
        )
        for doc in tqdm(cursor, desc="targets", unit="doc"):
            ids = doc.get("identifiers") or {}
            assigned = assign_key(ids.get("pmid") or "", ids.get("pmcid") or "",
                                  ids.get("doi") or "")
            if assigned is None:
                continue
            key_type, key = assigned
            {"pmid": pmids, "doi": dois, "pmcid": pmcids}[key_type].add(key)
    return {"pmid": sorted(pmids), "doi": sorted(dois), "pmcid": sorted(pmcids)}


def load_targets(input_paths: tuple[Path, ...]) -> dict[str, list[str]]:
    """One pass per key type, assigned by `assign_key` so no record is fetched twice. The
    ordering (pmid, then doi, then pmcid) is the inverse of `pid.py`'s identifier priority: pmid
    is the cheapest and by far the highest-yield key for this particular API (99.6% vs 80.7%)."""
    pmids: set[str] = set()
    dois: set[str] = set()
    pmcids: set[str] = set()

    for path in input_paths:
        if not path.exists():
            print(f"fetch_citations: WARNING {path} does not exist -- skipping it")
            continue
        frame = pd.read_csv(path, dtype=str, usecols=lambda c: c in ("pmid", "pmcid", "doi"))
        frame = frame.fillna("")
        for col in ("pmid", "pmcid", "doi"):
            if col not in frame.columns:
                frame[col] = ""
        n_before = len(pmids) + len(dois) + len(pmcids)
        for pmid, pmcid, doi in zip(frame["pmid"], frame["pmcid"], frame["doi"]):
            assigned = assign_key(pmid, pmcid, doi)
            if assigned is None:
                continue
            key_type, key = assigned
            {"pmid": pmids, "doi": dois, "pmcid": pmcids}[key_type].add(key)
        print(f"fetch_citations: {path.name}: {len(frame):,} rows "
              f"(+{len(pmids) + len(dois) + len(pmcids) - n_before:,} new keys)")

    return {"pmid": sorted(pmids), "doi": sorted(dois), "pmcid": sorted(pmcids)}


def load_already_fetched(output_path: Path, max_age_days: int | None) -> set[tuple[str, str]]:
    """(key_type, key) pairs that do not need fetching again. With `--max-age-days`, a row older
    than the cutoff is treated as absent so the refresh re-fetches only stale counts."""
    if not output_path.exists():
        return set()
    frame = pd.read_csv(output_path, dtype=str).fillna("")
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        fetched = pd.to_datetime(frame["fetched_at"], errors="coerce", utc=True)
        frame = frame[fetched >= cutoff]
        print(f"fetch_citations: --max-age-days {max_age_days} -> {len(frame):,} rows still fresh")
    return set(zip(frame["key_type"], frame["key"]))


def run(
    input_paths: tuple[Path, ...],
    output_path: Path,
    max_workers: int,
    limit: int | None,
    max_age_days: int | None,
    with_licence: bool = False,
    targets_spec: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if targets_spec == MOROS_MISSING_LICENCE:
        if not with_licence:
            raise SystemExit(
                f"--targets {MOROS_MISSING_LICENCE} selects documents with no licence, so it only "
                f"makes sense with --with-licence (which switches resultType to 'core'; 'lite' "
                f"carries no license field at all)."
            )
        targets = load_targets_from_moros_missing_licence()
    elif targets_spec:
        raise SystemExit(f"unknown --targets {targets_spec!r} -- expected {MOROS_MISSING_LICENCE}")
    else:
        targets = load_targets(input_paths)
    if with_licence:
        print("fetch_citations: --with-licence -- resultType=core, capturing license and "
              "isOpenAccess alongside the count")
    already = load_already_fetched(output_path, max_age_days)

    remaining = {
        key_type: [k for k in keys if (key_type, k) not in already]
        for key_type, keys in targets.items()
    }
    total_remaining = sum(len(v) for v in remaining.values())
    print(
        "fetch_citations: targets " +
        ", ".join(f"{kt} {len(targets[kt]):,}" for kt in ("pmid", "doi", "pmcid")) +
        f" | {len(already):,} already fetched | {total_remaining:,} remaining"
    )
    if limit is not None:
        # A seeded random sample, NOT the first N. Learned the hard way: the lexically-first 480
        # no-pmid DOIs are a single cluster of Cochrane reviews (10.1002/14651858.cd*) that EPMC
        # does not index by DOI, so a prefix smoke check reported a 10.2% hit rate where the real
        # population yields 80.0% (both measured, 2026-09-03). A prefix is a cluster, not a
        # sample, and a smoke check that lies about coverage is worse than none. seed=42 matches
        # this project's convention everywhere else, so the sample is reproducible.
        rng = random.Random(42)
        for key_type in remaining:
            keys = remaining[key_type]
            remaining[key_type] = sorted(rng.sample(keys, min(limit, len(keys))))
        total_remaining = sum(len(v) for v in remaining.values())
        print(f"fetch_citations: --limit {limit} -> {total_remaining:,} keys this run "
              f"(seeded random sample, not a lexical prefix -- see the code comment)")
    if total_remaining == 0:
        print("fetch_citations: nothing to do.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    header_needed = not output_path.exists()
    n_written = 0
    n_errors = 0
    started = time.time()

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if header_needed:
            writer.writeheader()
        f.flush()

        for key_type in ("pmid", "doi", "pmcid"):
            keys = remaining[key_type]
            if not keys:
                continue
            size = CHUNK_SIZES[key_type]
            chunks = [keys[i : i + size] for i in range(0, len(keys), size)]
            print(f"\nfetch_citations: pass '{key_type}' -- {len(keys):,} keys in "
                  f"{len(chunks):,} chunks of {size}")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(fetch_chunk, session, key_type, chunk, fetched_at,
                                with_licence): chunk
                    for chunk in chunks
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=f"citations[{key_type}]", unit="chunk",
                ):
                    chunk = futures[future]
                    try:
                        rows = future.result()
                    except Exception as exc:  # noqa: BLE001 -- real network errors; keep going
                        n_errors += 1
                        tqdm.write(f"  chunk starting {chunk[0]} FAILED: {exc!r}"[:200])
                        continue
                    for row in rows:
                        writer.writerow(row)
                    n_written += len(rows)
                    f.flush()

    session.close()
    elapsed = time.time() - started
    rate = n_written / elapsed if elapsed else 0
    print(
        f"\nfetch_citations: done -- {n_written:,} counts fetched in {elapsed:.1f}s "
        f"({rate:.1f} records/sec), {n_errors} chunk-level error(s). Re-run the same command to "
        f"retry: already-fetched keys are skipped, so nothing is paid for twice in API load."
    )
    missed = total_remaining - n_written
    if missed > 0:
        print(
            f"fetch_citations: {missed:,} of {total_remaining:,} requested keys were not returned "
            f"by EPMC (genuine misses, or an errored chunk above). A genuine miss recurs on every "
            f"re-run -- that is expected, not a bug."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Fetch at most N keys per pass -- a cheap real-API smoke check.")
    parser.add_argument("--max-age-days", type=int, default=None,
                        help="Re-fetch only keys whose last fetch is older than this.")
    parser.add_argument("--with-licence", action="store_true",
                        help="Also capture license and isOpenAccess, via resultType=core. Off by "
                             "default because licences do not change and citations do: a routine "
                             "citation refresh should stay on the lighter 'lite' response.")
    parser.add_argument("--targets", default=None,
                        help=f"'{MOROS_MISSING_LICENCE}' to target every document whose licence "
                             f"was never looked up, instead of reading --input.")
    args = parser.parse_args()
    run(tuple(args.input), args.output, args.max_workers, args.limit, args.max_age_days,
        args.with_licence, args.targets)


if __name__ == "__main__":
    main()
