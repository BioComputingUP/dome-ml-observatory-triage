"""Fetches only the time windows of the search space that are not already covered.

The point of this script is what it *doesn't* do: it does not re-fetch 842,271 records every month
and rely on dedupe to discard them. It asks the ledger which windows have been fetched, asks moros
which are actually loaded, and fetches the difference.

Two authorities, checked against each other every run:

- the **ledger** (`output/coverage_ledger.json`) knows what was fetched, including windows that
  legitimately returned nothing;
- **moros** knows what was loaded.

The ledger may legitimately be ahead (fetched but not yet classified and loaded). moros being
ahead -- documents in years the ledger has never recorded -- means the ledger has lost history,
and the run **stops** rather than reconciling silently. That is the same class of mistake the
whole finalisation exists to correct: quietly deciding a population is already handled.

The current year is always re-fetched, because it is still filling up. Everything else is fetched
once, checkpointed per year with a `.done` marker exactly as `ingest/bulk_match.py` does, so an
interrupted run resumes rather than restarting.

    python3 fetch_search_space.py --show-query          # what would be searched, and its hash
    python3 fetch_search_space.py --dry-run             # which windows are missing
    python3 fetch_search_space.py --up-to today

Re-fetching the current year returns every paper of the year again (192,432 on 2026-09-15) to find
the few thousand moros lacks, and it never returns a paper Europe PMC indexes late for an earlier
year. `--indexed-since` fetches instead what Europe PMC first indexed since the last fetch, whatever
its publication date (about 5,000 a week), into its own folder:

    python3 fetch_search_space.py --indexed-since last --up-to today --dry-run
    python3 fetch_search_space.py --indexed-since last --up-to today
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from coverage_ledger import (
    DEFAULT_CONFIG,
    DEFAULT_LEDGER,
    CoverageLedger,
    SearchSpace,
    plan_index_window,
    missing_years,
)
from epmc_search import EpmcSearch
from moros_client import Moros

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_INCOMING = FOLDER_DIR / "output" / "incoming"


def moros_year_histogram(moros: Moros) -> dict[int, int]:
    raw = moros.histogram("publication_metadata.year")
    return {int(k): v for k, v in raw.items() if isinstance(k, (int, float)) and k}


def cross_check(space: SearchSpace, ledger: CoverageLedger, moros: Moros) -> list[int]:
    """Years where moros holds documents the ledger has no record of fetching."""
    covered = ledger.covered_years(space)
    loaded = moros_year_histogram(moros)
    first = int(space.coverage_start[:4])
    return sorted(y for y, n in loaded.items() if n and first <= y <= date.today().year
                  and y not in covered)


def window_for(year: int, up_to: date) -> tuple[str, str]:
    start = f"{year}-01-01"
    end = up_to.isoformat() if year == up_to.year else f"{year}-12-31"
    return start, end


def _ledger_path(out: Path) -> str:
    """How a fetched window's file is recorded in the ledger: relative to this folder.

    The ledger is committed, so an absolute path in it names one machine and, worse, survives a
    move -- the 62 windows recorded before 2026-09-07 still pointed into a repository this one
    replaced. The field is written here and read nowhere, so it is provenance rather than a
    lookup, but provenance that names a directory nobody else has is not much use.
    """
    try:
        return str(out.resolve().relative_to(FOLDER_DIR))
    except ValueError:
        # --incoming pointed outside the folder; an absolute path is then the honest record.
        return str(out)


# A window that returns fewer records than Europe PMC's own hitCount stopped paging early. The run of
# 2026-09-03 recorded 148,815 records for 2026-01-01..09-03 while the same query returned 191,536 on
# 2026-09-15, most of them indexed and unrevised before 09-03, and nothing noticed. Records indexed
# while a fetch runs can push the count up slightly, never down, so only a shortfall fails.
COMPLETENESS_TOLERANCE = 0.002


def check_complete(fetched: int, expected: int, label: str) -> None:
    """SystemExit when a window returned fewer records than Europe PMC reported for it."""
    if fetched < expected * (1 - COMPLETENESS_TOLERANCE):
        raise SystemExit(
            f"\nINCOMPLETE: {label} returned {fetched:,} records but Europe PMC reports {expected:,}. "
            f"Paging stopped early; the window is NOT recorded as fetched and no .done marker is "
            f"written. Re-run the same command.")


def run_indexed(space: SearchSpace, ledger: CoverageLedger, incoming_dir: Path, up_to: date,
                indexed_since: str, dry_run: bool) -> None:
    """One index window: what Europe PMC first indexed from `since` to `up_to`, plus, the first time,
    the papers dated after the last year window. Written to its own folder so the batch builder reads
    only this window, not the year files beside it."""
    since, future_from = plan_index_window(space, ledger, up_to, indexed_since)
    query = space.index_query(since, up_to.isoformat(), future_from)
    client = EpmcSearch()
    try:
        expected = client.count(query)
        print(f"\nindex window: first indexed {since} .. {up_to.isoformat()}"
              + (f", plus papers dated from {future_from} (first index window)" if future_from else ""))
        print(f"  query   : {query}")
        print(f"  records : {expected:,}")
        if dry_run:
            print("\nDRY RUN -- nothing fetched. Re-run without --dry-run to fetch this window.")
            return
        out_dir = incoming_dir / space.sha256()[:12] / f"indexed_{since}_{up_to.isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "records.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        n = 0
        with tmp.open("w", encoding="utf-8") as f:
            for record in client.search(query):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
        check_complete(n, expected, f"index window {since}..{up_to.isoformat()}")
        tmp.replace(out)
    finally:
        client.close()
    ledger.record_index_window(space, since, up_to.isoformat(), fetched=n, path=_ledger_path(out),
                               future_dated_from=future_from)
    ledger.save()
    print(f"{n:,} records -> {out}")
    print(f"\nledger -> {ledger.path}")
    print(f"next: python3 build_incoming_documents.py --incoming {out_dir}")


def run(config_path: Path, ledger_path: Path, incoming_dir: Path, up_to: date,
        dry_run: bool, show_query: bool, ignore_cross_check: bool,
        indexed_since: str | None = None) -> None:
    space = SearchSpace.load(config_path)
    ledger = CoverageLedger(ledger_path)

    print(f"search space '{space.name}'  sha256 {space.sha256()[:16]}...")
    print(f"  terms   : {space.term_clause()}")
    print(f"  sources : {space.sources or 'ALL (MED, PPR, PMC, AGR, PAT)'}")
    print(f"  example : {space.query(space.coverage_start, up_to.isoformat())}")
    if show_query:
        return

    covered = ledger.covered_years(space)
    print(f"\nledger: {len(covered)} year(s) already fetched for this exact query"
          + (f" ({min(covered)}-{max(covered)})" if covered else " -- nothing yet"))

    with Moros.from_env() as moros:
        print(f"moros : {moros.describe()}")
        unrecorded = cross_check(space, ledger, moros)
        if unrecorded and not ignore_cross_check:
            loaded = moros_year_histogram(moros)
            raise SystemExit(
                f"\nSTOPPING: moros holds documents for {len(unrecorded)} year(s) the ledger has "
                f"no record of fetching -- e.g. "
                f"{ {y: loaded[y] for y in unrecorded[:5]} }.\n"
                f"The ledger is the record of what was fetched; moros being ahead of it means "
                f"that history was lost (a fresh ledger against an existing corpus does exactly "
                f"this). Reconciling automatically would mean guessing which windows are safe to "
                f"skip, and guessing wrong silently re-creates the gap this pipeline exists to "
                f"close.\n\n"
                f"Either restore the ledger, or -- if you are deliberately adopting an existing "
                f"corpus -- re-run with --ignore-cross-check to record those years as covered."
            )
        if unrecorded and ignore_cross_check:
            print(f"\n--ignore-cross-check: recording {len(unrecorded)} pre-existing year(s) as "
                  f"covered without fetching them")
            for year in unrecorded:
                start, end = window_for(year, up_to)
                ledger.record_window(space, start, end, fetched=0, path="(pre-existing corpus)")
                ledger.record_loaded(space, start, end, moros_year_histogram(moros).get(year, 0))
            ledger.save()
            covered = ledger.covered_years(space)

    if indexed_since is not None:
        run_indexed(space, ledger, incoming_dir, up_to, indexed_since, dry_run)
        return

    years = missing_years(space, ledger, up_to)
    print(f"\nto fetch, up to {up_to.isoformat()}: {len(years)} window(s)")
    for year in years:
        start, end = window_for(year, up_to)
        marker = " (current year -- always re-fetched)" if year == up_to.year else ""
        print(f"  {start} .. {end}{marker}")
    if not years:
        print("  nothing -- coverage is complete for this query.")
        return

    if dry_run:
        print("\nDRY RUN -- nothing fetched. Re-run without --dry-run to fetch these windows.")
        return

    incoming_dir = incoming_dir / space.sha256()[:12]
    incoming_dir.mkdir(parents=True, exist_ok=True)
    client = EpmcSearch()
    try:
        for year in years:
            start, end = window_for(year, up_to)
            out = incoming_dir / f"{year}.jsonl"
            done = out.with_suffix(".done")
            if done.exists() and year != up_to.year:
                print(f"{year}: .done marker present, skipping")
                continue
            query = space.query(start, end)
            expected = client.count(query)
            tmp = out.with_suffix(".jsonl.tmp")
            n = 0
            with tmp.open("w", encoding="utf-8") as f:
                for record in client.search(query):
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n += 1
            check_complete(n, expected, f"{year} ({start}..{end})")
            tmp.replace(out)
            done.write_text(f"{n}\n", encoding="utf-8")
            ledger.record_window(space, start, end, fetched=n, path=_ledger_path(out))
            ledger.save()  # after every window, so an interrupted run keeps what it finished
            print(f"{year}: {n:,} records -> {out}")
    finally:
        client.close()

    print(f"\nledger -> {ledger.path}")
    print(f"next: python3 build_incoming_documents.py --incoming {incoming_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--incoming", type=Path, default=DEFAULT_INCOMING)
    parser.add_argument("--up-to", default="today", help="ISO date, or 'today'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-query", action="store_true")
    parser.add_argument("--ignore-cross-check", action="store_true",
                        help="Adopt an existing corpus: record its years as covered, unfetched.")
    parser.add_argument("--indexed-since", default=None,
                        help="YYYY-MM-DD, or 'last' (the last day already covered): fetch what Europe "
                             "PMC first indexed from then to --up-to, whatever its publication date, "
                             "instead of re-fetching the current year.")
    args = parser.parse_args()
    up_to = date.today() if args.up_to == "today" else date.fromisoformat(args.up_to)
    run(args.config, args.ledger, args.incoming, up_to, args.dry_run, args.show_query,
        args.ignore_cross_check, args.indexed_since)


if __name__ == "__main__":
    main()
