"""Fetches every text-mined accession number Europe PMC has for a record, through the batched
annotations API -- the primary source of `data_links.links`.

Identifiers are written here exactly as Europe PMC returned them, punctuation and all: this is the
raw record. `build_data_links.py` (with `link_identifiers.py`) is the only place they are cleaned,
so a better normaliser never needs a re-fetch.

`GET https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds
     ?articleIds=MED:34265844,MED:33024307,...&type=Accession Numbers&format=JSON`

Eight article ids per call is the hard limit (a ninth is a 400; verified 2026-09-14). Each call
returns, per article, every accession Europe PMC's text mining found in the full text and, with
`provider: "Biostudies"`, in the supplementary files -- with an identifiers.org URI, the accession
type (`subType`: PDBe, ENA, GEO, UniProt, DOI, ...), the section it was found in and, for
supplementary hits, the file name and frequency. This is the `tm_accession` half of the
`/datalinks` endpoint, batched eight times denser and served by a different backend (SciLite),
which is why it is the primary route and `/datalinks` is the residual one (`fetch_datalinks.py`).

Targets come from `fetch_epmc_metadata.py`'s output: records whose `has_tm_accessions` is Y
(about half the corpus), addressed by their Europe PMC `(source, id)`. An id the API does not
return is recorded as `status: absent` with no annotations -- a real answer, never re-asked
unless `--max-age-days` says it has aged.

Concurrency is the point: 64 workers by default, 15 s per call, retries only on 429/5xx. Every run
prints achieved calls/s and the error rate; raise `--max-workers` between runs until either moves.

    python3 fetch_annotations.py --limit 3000                  # sample: measure the rate first
    python3 fetch_annotations.py --max-workers 128             # the corpus
    python3 fetch_annotations.py --input ../output/incoming_new.csv   # a batch (after Phase 4)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from tqdm import tqdm

from datalinks_resources import scheme_from_uri
from fetch_citations import _session

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_INPUT = FOLDER_DIR / "output" / "epmc_metadata.csv"
DEFAULT_OUTPUT = FOLDER_DIR / "output" / "epmc_annotations.jsonl"

ANNOTATIONS_URL = "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
ANNOTATION_TYPE = "Accession Numbers"
IDS_PER_CALL = 8          # the API's hard limit
DEFAULT_MAX_WORKERS = 64
REQUEST_TIMEOUT = 15      # a sick backend must not park a worker for 30 s

_SECTION_URI_RE = re.compile(r"\s*\(https?://[^)]*\)\s*$")


def short_section(section: str | None) -> str | None:
    """`"Article (http://semanticscience.org/resource/SIO_001029)"` -> `"Article"`."""
    if not section:
        return None
    return _SECTION_URI_RE.sub("", section).strip() or None


def reduce_annotation(annotation: dict) -> dict:
    """The seven fields a link needs, from one annotations-API entry. `sub_type` is absent on an
    accession mined from a supplementary file, so it is recovered from the identifiers.org URI."""
    tags = annotation.get("tags") or []
    uri = (tags[0].get("uri") if tags and isinstance(tags[0], dict) else None) or None
    return {
        "exact": (annotation.get("exact") or "").strip(),
        "uri": uri,
        "sub_type": annotation.get("subType") or scheme_from_uri(uri),
        "provider": annotation.get("provider"),
        "section": short_section(annotation.get("section")),
        "frequency": annotation.get("frequency"),
        "file_name": annotation.get("fileName"),
    }


def reduce_response(payload: list[dict], requested: list[tuple[str, str]],
                    fetched_at: str) -> list[dict]:
    """One JSONL record per requested id: `ok` with its annotations, or `absent` when the API did
    not mention it (an unknown id, or one it has not mined). Both are answers."""
    by_id: dict[tuple[str, str], dict] = {}
    for article in payload or []:
        key = (str(article.get("source") or ""), str(article.get("extId") or ""))
        by_id[key] = article
    records = []
    for source, ext_id in requested:
        article = by_id.get((source, ext_id))
        if article is None:
            records.append({"source": source, "id": ext_id, "fetched_at": fetched_at,
                            "status": "absent", "annotations": []})
            continue
        records.append({
            "source": source, "id": ext_id, "fetched_at": fetched_at, "status": "ok",
            "pmcid": article.get("pmcid"),
            "annotations": [reduce_annotation(a) for a in article.get("annotations") or []],
        })
    return records


def fetch_batch(session: requests.Session, batch: list[tuple[str, str]],
                fetched_at: str) -> list[dict]:
    resp = session.get(
        ANNOTATIONS_URL,
        params={"articleIds": ",".join(f"{s}:{i}" for s, i in batch),
                "type": ANNOTATION_TYPE, "format": "JSON"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return reduce_response(resp.json(), batch, fetched_at)


def load_targets(input_path: Path, everything: bool) -> list[tuple[str, str]]:
    """`(source, id)` pairs with text-mined accessions, from a metadata CSV or a staged batch CSV
    carrying the same columns. `--all` ignores the flag (a re-check of the whole set)."""
    seen: set[tuple[str, str]] = set()
    targets: list[tuple[str, str]] = []
    n = 0
    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            n += 1
            source = (row.get("epmc_source") or "").strip()
            ext_id = (row.get("epmc_id") or "").strip()
            if not source or not ext_id:
                continue
            if not everything and (row.get("has_tm_accessions") or "").strip().upper() != "Y":
                continue
            key = (source, ext_id)
            if key in seen:
                continue
            seen.add(key)
            targets.append(key)
    print(f"fetch_annotations: {input_path.name}: {n:,} rows -> {len(targets):,} targets"
          + ("" if everything else " with text-mined accessions"))
    return targets


def load_already_fetched(output_path: Path, max_age_days: int | None) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    cutoff = None
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    done: set[tuple[str, str]] = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if cutoff is not None:
                try:
                    if datetime.fromisoformat(rec["fetched_at"]) < cutoff:
                        continue
                except (KeyError, ValueError):
                    continue
            done.add((rec["source"], rec["id"]))
    return done


def run(input_path: Path, output_path: Path, max_workers: int, limit: int | None,
        max_age_days: int | None, everything: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = load_targets(input_path, everything)
    already = load_already_fetched(output_path, max_age_days)
    remaining = [t for t in targets if t not in already]
    print(f"fetch_annotations: {len(already):,} already fetched | {len(remaining):,} remaining")
    if limit is not None:
        rng = random.Random(42)
        remaining = sorted(rng.sample(remaining, min(limit, len(remaining))))
        print(f"fetch_annotations: --limit {limit} -> {len(remaining):,} records this run")
    if not remaining:
        print("fetch_annotations: nothing to do.")
        return

    batches = [remaining[i:i + IDS_PER_CALL] for i in range(0, len(remaining), IDS_PER_CALL)]
    print(f"fetch_annotations: {len(batches):,} calls of {IDS_PER_CALL}, {max_workers} workers")
    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    n_calls = n_errors = n_records = n_absent = n_annotations = 0
    started = time.time()
    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_batch, session, batch, fetched_at): batch
                       for batch in batches}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="annotations", unit="call"):
                n_calls += 1
                try:
                    records = future.result()
                except Exception as exc:  # noqa: BLE001 -- keep going; re-run retries them
                    n_errors += 1
                    if n_errors <= 20:
                        tqdm.write(f"  call FAILED: {exc!r}"[:200])
                    continue
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_records += 1
                    n_absent += rec["status"] == "absent"
                    n_annotations += len(rec["annotations"])
                f.flush()
    session.close()
    elapsed = time.time() - started
    print(f"\nfetch_annotations: done -- {n_calls:,} calls in {elapsed:.1f}s "
          f"({n_calls / elapsed if elapsed else 0:.1f} calls/s, "
          f"{n_records / elapsed if elapsed else 0:.0f} records/s), "
          f"{n_errors} failed call(s) ({n_errors / n_calls * 100 if n_calls else 0:.1f}%). "
          f"{n_records:,} records written, {n_absent:,} absent, {n_annotations:,} accessions. "
          f"Re-run the same command to retry failures; fetched ids are skipped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="epmc_metadata.csv, or a staged CSV carrying epmc_source / epmc_id "
                             "/ has_tm_accessions")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="At most N records, a seeded random sample. Prints calls/s.")
    parser.add_argument("--max-age-days", type=int, default=None)
    parser.add_argument("--all", action="store_true",
                        help="Ignore has_tm_accessions and ask for every record.")
    args = parser.parse_args()
    run(args.input, args.output, args.max_workers, args.limit, args.max_age_days, args.all)


if __name__ == "__main__":
    main()
