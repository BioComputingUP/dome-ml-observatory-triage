"""Fetches Europe PMC's Scholix data links for the records text mining cannot cover -- curated
database cross-references, data citations from DataCite / Crossref, external links -- one GET
per record to `/{source}/{id}/datalinks`.

Identifiers are written here exactly as Europe PMC returned them, punctuation and all: this is the
raw record. `build_data_links.py` (with `link_identifiers.py`) is the only place they are cleaned,
so a better normaliser never needs a re-fetch.

    GET https://www.ebi.ac.uk/europepmc/webservices/rest/MED/33024307/datalinks?format=json

The response nests Category -> Section -> Linklist.Link, each link a Scholix pair with a
`Target.Identifier {ID, IDScheme, IDURL}`, `Target.Publisher.Name`, `RelationshipType.Name`
and `ObtainedBy` (`tm_accession`, `ext_links`, ...). `reduce_datalinks()` flattens that to one row
per link; `build_data_links.py` merges the rows with the annotations-API links and dedupes them.

This endpoint cannot be batched, so it is kept to the residual: by default only records whose
metadata says `has_db_xrefs` Y or carries the `related_data` tag (~3% of the corpus).
`--all-supporting` widens to every `supporting_data` record (a completeness sweep, run only if a
sample shows it adds links the annotations API lacked); `--all` to every `has_data` Y record.

Concurrency is adaptive by hand: start at 64 workers, read the calls/s and error rate the run
prints, raise until either moves. Timeouts are 15 s and retries happen only on 429/5xx, so a
degraded backend (2026-09-14: HTTP 500 on every id for a day) costs a fast failed run, not a
stalled one; failed ids are simply re-asked on the next run. A 404 is an answer (no links).

    python3 fetch_datalinks.py --limit 3000          # sample: rate, error rate, resource mix
    python3 fetch_datalinks.py --max-workers 128     # the residual set
    python3 fetch_datalinks.py --input ../output/incoming_new.csv   # a batch
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

import requests
from tqdm import tqdm

from fetch_annotations import load_already_fetched
from fetch_citations import _session

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_INPUT = FOLDER_DIR / "output" / "epmc_metadata.csv"
DEFAULT_OUTPUT = FOLDER_DIR / "output" / "epmc_datalinks.jsonl"

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"
DEFAULT_MAX_WORKERS = 64
REQUEST_TIMEOUT = 15
RESIDUAL_TAGS = {"related_data"}
SUPPORTING_TAGS = {"related_data", "supporting_data"}


def _name(node) -> str | None:
    if isinstance(node, dict):
        return node.get("Name") or None
    return node or None


def reduce_link(category: str | None, section_obtained_by: str | None, link: dict) -> dict:
    target = link.get("Target") or {}
    ident = target.get("Identifier") or {}
    return {
        "category": category,
        "obtained_by": link.get("ObtainedBy") or section_obtained_by,
        "id": str(ident.get("ID") or "").strip(),
        "id_scheme": ident.get("IDScheme"),
        "url": ident.get("IDURL") or None,
        "title": target.get("Title") or None,
        "publisher": _name(target.get("Publisher")),
        "target_type": _name(target.get("Type")),
        "relationship": _name(link.get("RelationshipType")),
        "link_provider": _name(link.get("LinkProvider")),
        "publication_date": link.get("PublicationDate") or None,
    }


def reduce_datalinks(payload: dict) -> list[dict]:
    """Category -> Section -> Link, flattened. Tolerates every level being absent."""
    links: list[dict] = []
    for category in ((payload or {}).get("dataLinkList") or {}).get("Category") or []:
        name = category.get("Name")
        for section in category.get("Section") or []:
            obtained_by = section.get("ObtainedBy")
            for link in (section.get("Linklist") or {}).get("Link") or []:
                reduced = reduce_link(name, obtained_by, link)
                if reduced["id"]:
                    links.append(reduced)
    return links


def fetch_one(session: requests.Session, source: str, ext_id: str, fetched_at: str) -> dict:
    resp = session.get(f"{BASE_URL}/{source}/{ext_id}/datalinks",
                       params={"format": "json"}, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return {"source": source, "id": ext_id, "fetched_at": fetched_at, "http_status": 404,
                "hit_count": 0, "links": []}
    resp.raise_for_status()
    payload = resp.json()
    return {"source": source, "id": ext_id, "fetched_at": fetched_at, "http_status": 200,
            "hit_count": payload.get("hitCount"), "links": reduce_datalinks(payload)}


def wanted(row: dict, scope: str) -> bool:
    if scope == "all":
        return (row.get("has_data") or "").strip().upper() == "Y"
    tags = set()
    raw = (row.get("data_links_tags") or "").strip()
    if raw:
        try:
            tags = set(json.loads(raw))
        except ValueError:
            tags = set()
    if (row.get("has_db_xrefs") or "").strip().upper() == "Y":
        return True
    return bool(tags & (SUPPORTING_TAGS if scope == "supporting" else RESIDUAL_TAGS))


def load_targets(input_path: Path, scope: str) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    targets: list[tuple[str, str]] = []
    n = 0
    with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            n += 1
            source = (row.get("epmc_source") or "").strip()
            ext_id = (row.get("epmc_id") or "").strip()
            if not source or not ext_id or not wanted(row, scope):
                continue
            key = (source, ext_id)
            if key not in seen:
                seen.add(key)
                targets.append(key)
    print(f"fetch_datalinks: {input_path.name}: {n:,} rows -> {len(targets):,} targets "
          f"(scope: {scope})")
    return targets


def run(input_path: Path, output_path: Path, max_workers: int, limit: int | None,
        max_age_days: int | None, scope: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    targets = load_targets(input_path, scope)
    already = load_already_fetched(output_path, max_age_days)
    remaining = [t for t in targets if t not in already]
    print(f"fetch_datalinks: {len(already):,} already fetched | {len(remaining):,} remaining")
    if limit is not None:
        rng = random.Random(42)
        remaining = sorted(rng.sample(remaining, min(limit, len(remaining))))
        print(f"fetch_datalinks: --limit {limit} -> {len(remaining):,} records this run")
    if not remaining:
        print("fetch_datalinks: nothing to do.")
        return

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    n_calls = n_errors = n_written = n_links = 0
    started = time.time()
    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(fetch_one, session, s, i, fetched_at): (s, i)
                       for s, i in remaining}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="datalinks", unit="call"):
                n_calls += 1
                try:
                    rec = future.result()
                except Exception as exc:  # noqa: BLE001 -- re-run retries them
                    n_errors += 1
                    if n_errors <= 20:
                        tqdm.write(f"  {futures[future]} FAILED: {exc!r}"[:200])
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1
                n_links += len(rec["links"])
                if n_written % 500 == 0:
                    f.flush()
    session.close()
    elapsed = time.time() - started
    print(f"\nfetch_datalinks: done -- {n_calls:,} calls in {elapsed:.1f}s "
          f"({n_calls / elapsed if elapsed else 0:.1f} calls/s), {n_errors} failed "
          f"({n_errors / n_calls * 100 if n_calls else 0:.1f}%). {n_written:,} records written, "
          f"{n_links:,} links. Re-run the same command to retry failures.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-age-days", type=int, default=None)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--all-supporting", action="store_true",
                       help="Every supporting_data record too (the completeness sweep).")
    scope.add_argument("--all", action="store_true", help="Every has_data record.")
    args = parser.parse_args()
    run(args.input, args.output, args.max_workers, args.limit, args.max_age_days,
        "all" if args.all else "supporting" if args.all_supporting else "residual")


if __name__ == "__main__":
    main()
