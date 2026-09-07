"""Standalone Europe PMC licensing fetch -- deliberately NOT part of the dome_triage package and
NOT run through Docker (per Gavin's explicit request, 2026-08-24). Self-contained: only
requests/pandas/tqdm, no dome_triage import, runs directly on the host with plain `python3`.

Purpose: for every PMID in the real AI/ML EPMC search-space pool (data/interim/bulk_candidates.csv,
~745k records -- the same pool Step 22/23 in STEPS_Progress.md plan to triage), fetch the real
license string (e.g. "cc by", "cc by-nc") plus the isOpenAccess flag, ahead of ever building a
download/redistribution feature for the future landscape database (STEPS_Progress.md's Phase 8 --
"licensing research required before any download feature ships").

Real-calibration numbers behind the design choices below (measured live against the real EPMC API,
2026-08-24, not assumed):
- `resultType=core` is REQUIRED -- confirmed live that the lighter `resultType=lite` does not
  include a `license` field at all, only `isOpenAccess`.
- Batch size 300 PMIDs/query, NOT the 40 `dome_triage/ingest/epmc_client.py::EpmcClient.get_by_ids`
  uses for the main pipeline's smaller ad-hoc lookups -- that number was never pushed to its real
  ceiling because it never needed to be. This script's job is a ~745k-PMID bulk fetch, where
  request count is the real cost, so the real ceiling was binary-searched live: 360 PMIDs/query
  succeeds, 370 returns HTTP 400, 390+ returns a clean 414 "Request-URI Too Large" from EPMC's
  nginx front end. 300 keeps a real safety margin below that wall (PMID digit-length varies record
  to record, so a fixed COUNT isn't a fixed byte length).
- 35 concurrent chunk requests, measured live at 685 records/sec, 0 errors, over two real 6,000-
  record test batches at chunk_size=300 -- projects to ~18 minutes for the full ~745k pool. Not
  pushed further than this without re-testing; EPMC is a free public service, not our
  infrastructure to hammer, and this is already a meaningfully heavier load than a typical single-ID
  lookup.
- PMIDs are always purely numeric -- EXT_ID clauses are built WITHOUT quotes, matching the fix
  epmc_client.py's own docstring documents (a quoted single-clause chunk combined with
  "AND SRC:MED" silently returns 0 hits on EPMC's Lucene parser for exactly this field).

Resumable by design: every completed batch is written to the output CSV immediately (streamed, one
batch's rows appended at a time, matching this whole project's established "never buffer a
long-running fetch in memory only" rule from a real prior data-loss incident) -- a crash or
Ctrl+C only loses the one in-flight batch's worth of PMIDs, not the whole run. Re-running the exact
same command skips every PMID already present in the output file.
"""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
# Real-measured ceiling (2026-08-24, binary-searched live against this exact query shape): 360
# PMIDs/query succeeds, 370 returns HTTP 400, 390+ returns a clear 414 "Request-URI Too Large" from
# EPMC's nginx front end. 300 keeps a real safety margin below that hard wall -- PMID digit-length
# varies record to record, so the exact byte length of a same-COUNT chunk isn't fixed. This is
# deliberately larger than epmc_client.py's own CHUNK_SIZE=40 (that client was built for the main
# pipeline's smaller, ad-hoc ID lookups and was never pushed to find its own real ceiling; this
# script's whole job is a ~745k-PMID bulk fetch, where request count is the real cost).
CHUNK_SIZE = 300
DEFAULT_MAX_WORKERS = 35
OUTPUT_COLUMNS = ["pmid", "license", "is_open_access"]

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = THIS_DIR.parent / "data" / "interim" / "bulk_candidates.csv"
DEFAULT_OUTPUT_PATH = THIS_DIR / "output" / "epmc_pmid_licensing.csv"


def _session(max_retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=DEFAULT_MAX_WORKERS, pool_maxsize=DEFAULT_MAX_WORKERS * 2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_licensing_chunk(session: requests.Session, pmids: list[str]) -> list[dict]:
    """One real HTTP request for up to CHUNK_SIZE PMIDs. Returns a row per PMID actually found in
    EPMC (missing/unmatched PMIDs are simply absent -- never raises for a partial miss, same
    convention as `EpmcClient.get_by_ids`)."""
    clauses = " OR ".join(f"EXT_ID:{pmid}" for pmid in pmids)
    query = f"({clauses}) AND SRC:MED"
    resp = session.get(
        EPMC_SEARCH_URL,
        params={"query": query, "pageSize": len(pmids), "format": "json", "resultType": "core"},
        timeout=60,
    )
    resp.raise_for_status()
    results = resp.json().get("resultList", {}).get("result", [])
    return [
        {
            "pmid": result.get("pmid"),
            "license": result.get("license") or "",
            "is_open_access": result.get("isOpenAccess") or "",
        }
        for result in results
        if result.get("pmid")
    ]


def load_target_pmids(input_path: Path) -> list[str]:
    df = pd.read_csv(input_path, dtype=str, usecols=["pmid"])
    return sorted(df["pmid"].dropna().unique().tolist())


def load_already_fetched(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    existing = pd.read_csv(output_path, dtype=str, usecols=["pmid"])
    return set(existing["pmid"].dropna())


def run(input_path: Path, output_path: Path, max_workers: int, chunk_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_pmids = load_target_pmids(input_path)
    already_fetched = load_already_fetched(output_path)
    remaining = [p for p in all_pmids if p not in already_fetched]

    print(
        f"fetch_licensing: {len(all_pmids):,} total PMIDs in {input_path.name}, "
        f"{len(already_fetched):,} already fetched, {len(remaining):,} remaining."
    )
    if not remaining:
        print("fetch_licensing: nothing to do -- every PMID is already in the output file.")
        return

    chunks = [remaining[i : i + chunk_size] for i in range(0, len(remaining), chunk_size)]
    session = _session()
    header_needed = not output_path.exists()
    n_written = 0
    n_errors = 0
    started_at = time.time()

    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if header_needed:
            writer.writeheader()
        f.flush()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_licensing_chunk, session, chunk): chunk for chunk in chunks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="fetch_licensing", unit="chunk"):
                chunk = futures[future]
                try:
                    rows = future.result()
                except Exception as exc:  # noqa: BLE001 -- real network/API errors, keep going
                    n_errors += 1
                    tqdm.write(f"fetch_licensing: chunk starting {chunk[0]} FAILED: {exc}")
                    continue
                for row in rows:
                    writer.writerow(row)
                n_written += len(rows)
                f.flush()

    elapsed = time.time() - started_at
    session.close()
    print(
        f"fetch_licensing: done -- {n_written:,} PMIDs fetched in {elapsed:.1f}s "
        f"({n_written / elapsed:.1f} records/sec), {n_errors} chunk-level errors "
        f"(re-run this exact command to retry -- already-fetched PMIDs are skipped, not re-paid "
        f"for in API load)."
    )
    n_missing = len(remaining) - n_written
    if n_missing > 0:
        print(
            f"fetch_licensing: {n_missing:,} of the {len(remaining):,} requested PMIDs were not "
            f"returned by EPMC at all (not found under SRC:MED, or a chunk-level error above) -- "
            f"re-run this command to retry the errored chunks; a genuine EPMC miss will keep "
            f"recurring on every re-run, which is expected, not a bug."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="CSV with a 'pmid' column.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output pmid,license,is_open_access CSV.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()
    run(args.input, args.output, args.max_workers, args.chunk_size)
