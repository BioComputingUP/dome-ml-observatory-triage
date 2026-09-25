"""Re-derives `source.access.fulltext_available` from Europe PMC, for the documents whose `false`
is known to go stale. Writes `output/pid_fulltext.csv` for `load_fields.py --mode fulltext`.

The flag means Europe PMC holds the full text (`inEPMC`) or PMC does (`inPMC`) -- the rule
`dome_triage/ingest/bulk_match.py` set it by at fetch time. Measured on 2026-09-25, two ways it
ends up wrong:

1. **The curated merge never derived it.** `build_curated_documents.py` carried the curated
   sources' own column, which held the triage model's default (`fulltext_available: bool = False`),
   not a lookup. 2,098 of its 6,179 documents read false; a sample of 150 of those with a PMCID
   was 150 / 150 `inEPMC=Y inPMC=Y` live. AlphaFold 2 (PMC8371605, CC BY) was one.
2. **An embargo lifts after the fetch.** A paper fetched while its PMC full text was embargoed has
   a PMCID and, correctly at the time, `inPMC=N`. 15,231 bulk-classified documents had a PMCID and
   false; of a sample of 150, 104 were `Y/Y` live and 95 were 2026 papers.

So the default targets are the false values either cause can explain -- any document with a PMCID,
and every curated-merge document -- about 17,300. `--query` widens it (every false is ~218,600 and
still cheap). Keys are chosen by `fetch_citations.assign_key`, pmid > doi > pmcid, and queried with
its chunk sizes and source preference, so this looks each paper up exactly as the citation fetch
does. A key Europe PMC does not answer gets no row: `load_fields.py` then leaves the value alone.

    python3 fetch_fulltext.py --limit 200      # seeded random sample -- a real-API smoke check
    python3 fetch_fulltext.py                  # the default targets
    python3 load_fields.py --mode fulltext     # dry run: reports how many values would change
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

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
from moros_client import Moros

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR.parent / "output" / "pid_fulltext.csv"
OUTPUT_COLUMNS = ["pid", "key_type", "key", "epmc_source", "in_epmc", "in_pmc", "fetched_at"]
DEFAULT_MAX_WORKERS = 8

# The false values that can be stale: a PMCID (PMC has the paper, so an embargo may have lifted)
# or the curated merge (never looked up). See the module docstring for the measurements.
DEFAULT_QUERY = {
    "source.access.fulltext_available": False,
    "$or": [
        {"identifiers.pmcid": {"$nin": [None, ""]}},
        {"llm_classification.batch_id": {"$regex": "^curated_merge_"}},
    ],
}


def fulltext_flags(record: dict) -> tuple[str, str]:
    """Europe PMC's two flags, verbatim ("Y" / "N"), or "" where the record omits one."""
    return (record.get("inEPMC") or "").strip(), (record.get("inPMC") or "").strip()


def load_targets(moros: Moros, query: dict) -> dict[tuple[str, str], list[str]]:
    """(key_type, key) -> the pids looked up by it. Usually one pid per key; a list so two
    documents sharing an identifier cannot silently lose one of their answers."""
    targets: dict[tuple[str, str], list[str]] = {}
    cursor = moros.collection.find(
        query, {"identifiers.pmid": 1, "identifiers.pmcid": 1, "identifiers.doi": 1}
    )
    for doc in tqdm(cursor, desc="targets", unit="doc"):
        ids = doc.get("identifiers") or {}
        assigned = assign_key(ids.get("pmid") or "", ids.get("pmcid") or "", ids.get("doi") or "")
        if assigned is not None:
            targets.setdefault(assigned, []).append(doc["_id"])
    return targets


def fetch_chunk(session, key_type: str, keys: list[str]) -> dict[str, dict]:
    """key -> the winning Europe PMC record, for the keys Europe PMC answered."""
    resp = session.get(
        EPMC_SEARCH_URL,
        params={"query": build_query(key_type, keys), "pageSize": 1000, "format": "json",
                "resultType": "lite"},
        timeout=90,
    )
    resp.raise_for_status()
    wanted = set(keys)
    grouped: dict[str, list[dict]] = {}
    for record in resp.json().get("resultList", {}).get("result", []):
        key = _key_of(record, key_type)
        if key in wanted:
            grouped.setdefault(key, []).append(record)
    return {key: pick_best(records) for key, records in grouped.items()}


def run(query: dict, output_path: Path, max_workers: int, limit: int | None) -> None:
    with Moros.from_env() as moros:
        print(f"fetch_fulltext: targets from {moros.describe()}")
        targets = load_targets(moros, query)
    keys_by_type: dict[str, list[str]] = {"pmid": [], "doi": [], "pmcid": []}
    for key_type, key in targets:
        keys_by_type[key_type].append(key)
    print("fetch_fulltext: " + ", ".join(f"{kt} {len(v):,}" for kt, v in keys_by_type.items())
          + f" keys for {sum(len(p) for p in targets.values()):,} documents")
    if limit is not None:
        # A seeded random sample, never a prefix -- fetch_citations.py says why (seed 42 there too).
        rng = random.Random(42)
        keys_by_type = {kt: sorted(rng.sample(v, min(limit, len(v)))) for kt, v in keys_by_type.items()}
        print(f"fetch_fulltext: --limit {limit} -> {sum(len(v) for v in keys_by_type.values()):,} keys")

    fetched_at = datetime.now(timezone.utc).isoformat()
    session = _session(max_workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started, answered, true_count, errors = time.time(), 0, 0, 0
    # Overwritten, not appended: this is one load's staging file, and a stale row from an older
    # run would write a flag Europe PMC no longer gives.
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for key_type, keys in keys_by_type.items():
            if not keys:
                continue
            size = CHUNK_SIZES[key_type]
            chunks = [keys[i:i + size] for i in range(0, len(keys), size)]
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(fetch_chunk, session, key_type, c): c for c in chunks}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"fulltext[{key_type}]", unit="chunk"):
                    try:
                        found = future.result()
                    except Exception as exc:  # noqa: BLE001 -- a real network error; keep going
                        errors += 1
                        tqdm.write(f"  chunk starting {futures[future][0]} FAILED: {exc!r}"[:200])
                        continue
                    for key, record in found.items():
                        in_epmc, in_pmc = fulltext_flags(record)
                        if not (in_epmc or in_pmc):
                            continue
                        for pid in targets[(key_type, key)]:
                            writer.writerow({"pid": pid, "key_type": key_type, "key": key,
                                             "epmc_source": record.get("source") or "",
                                             "in_epmc": in_epmc, "in_pmc": in_pmc,
                                             "fetched_at": fetched_at})
                            answered += 1
                            true_count += in_epmc == "Y" or in_pmc == "Y"
                    f.flush()
    session.close()
    print(f"\nfetch_fulltext: {answered:,} documents answered in {time.time() - started:.0f}s, "
          f"{true_count:,} with full text in Europe PMC or PMC, {errors} chunk error(s) -> "
          f"{output_path}\nNext: python3 load_fields.py --mode fulltext   (a dry run)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", type=json.loads, default=None,
                        help="A Mongo filter (JSON) replacing the default targets.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=None,
                        help="At most N keys per key type, seeded random -- a smoke check.")
    args = parser.parse_args()
    run(args.query or DEFAULT_QUERY, args.output, args.max_workers, args.limit)


if __name__ == "__main__":
    main()
