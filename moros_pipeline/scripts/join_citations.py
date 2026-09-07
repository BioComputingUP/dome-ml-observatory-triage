"""Joins the fetched EPMC citation counts onto document `_id`s, producing `pid_citations.csv`.

`epmc_citations.csv` is keyed by the *identifier used to fetch it* (pmid / doi / pmcid). Mongo is
keyed by `_id`, the deterministic UUID5 that `mongo_landscape_export/scripts/pid.py` mints from
`pmcid > doi > pmid`. This closes that gap, and it is the only place the two key spaces meet.

**The join key is taken from our own corpus, never from EPMC's response.** EPMC will happily return
a record whose `doi` differs in case, or whose `pmid` is populated when ours was blank. Minting a
pid from EPMC's identifier triple would produce a *different* `_id` for the same paper and the
`$set` would match nothing (or worse, something else). So the mapping is built from the corpus CSV,
which already carries both `pid` and the identifiers, and `assign_key` -- imported from
`fetch_citations.py`, not reimplemented -- guarantees both sides agree on which key was used.

**Deliberate deviation from `join_license.py`.** That script rewrites the 2.13GB staging CSV to add
its columns, because the licence data had to be in the CSV before the JSONL was ever written. The
citation situation is different: all 827,061 documents are already *in* Mongo, and they take their
counts through a field-level `$set` rather than a reload. Rewriting 2.13GB to add three columns
would buy nothing. So this writes one small mapping file instead, which both consumers read:
`load_fields.py --mode citations` (for documents already in Mongo) and the small
new-document builders (curated merge, incremental runs), which have only thousands of rows each.

    python3 join_citations.py                    # -> output/pid_citations.csv
    python3 join_citations.py --report-only      # coverage numbers, writes nothing
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fetch_citations import CITATION_SOURCE, assign_key

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
REPO_DIR = FOLDER_DIR.parent

DEFAULT_CITATIONS = FOLDER_DIR / "output" / "epmc_citations.csv"
DEFAULT_CORPUS = REPO_DIR / "mongo_landscape_export" / "ai_ml_landscape_classified_usable.csv"
DEFAULT_OUTPUT = FOLDER_DIR / "output" / "pid_citations.csv"

OUTPUT_COLUMNS = ["pid", "citation_count", "citation_count_updated", "citation_source"]


def load_citation_index(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(key_type, key) -> (citation_count, fetched_at). Last row wins, so a refresh run appended
    to the same file supersedes the older count for that key."""
    index: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in tqdm(csv.DictReader(f), desc="citations", unit="row"):
            count = (row.get("citation_count") or "").strip()
            if not count:
                continue  # EPMC returned the record but no count -- treat as not available
            index[(row["key_type"], row["key"])] = (count, row.get("fetched_at") or "")
    return index


def load_licence_index(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(key_type, key) -> (license, epmc_is_open_access), from a --with-licence fetch.

    Unlike the citation index this keeps rows whose `license` is `""`. That empty string is a real
    answer -- "EPMC was asked and disclosed no licence" -- and the schema distinguishes it from
    null, "never looked up". Dropping it would leave the document null forever and every future
    run would re-fetch it."""
    index: dict[tuple[str, str], tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in tqdm(csv.DictReader(f), desc="licences", unit="row"):
            if "license" not in row:
                continue  # a citations-only file, fetched without --with-licence
            index[(row["key_type"], row["key"])] = (
                (row.get("license") or "").strip(),
                (row.get("epmc_is_open_access") or "").strip(),
            )
    return index


def iter_corpus_from_moros():
    """`pid` + identifiers straight from the live collection.

    The corpus CSV covers only the original landscape load. Every document added since -- the
    curated merge, and each incremental batch -- exists solely in Mongo, so a join that reads the
    CSV silently cannot reach them. For the licence backfill that was 13,476 of the documents that
    needed it most, being the newest.

    Still obeys this module's rule that the join key comes from OUR corpus, never from EPMC's
    response: these identifiers are the ones the pid was minted from."""
    from moros_client import Moros

    with Moros.from_env() as moros:
        print(f"join_citations: corpus from {moros.describe()}")
        cursor = moros.collection.find(
            {}, {"identifiers.pmid": 1, "identifiers.pmcid": 1, "identifiers.doi": 1},
        )
        for doc in cursor:
            ids = doc.get("identifiers") or {}
            yield {
                "pid": doc["_id"],
                "pmid": ids.get("pmid") or "",
                "pmcid": ids.get("pmcid") or "",
                "doi": ids.get("doi") or "",
            }


def run(citations_path: Path, corpus_path: Path, output_path: Path, report_only: bool,
        with_licence: bool = False, corpus_from_moros: bool = False) -> None:
    index = load_citation_index(citations_path)
    print(f"join_citations: {len(index):,} usable counts in {citations_path.name}")
    licences = load_licence_index(citations_path) if with_licence else {}
    if with_licence:
        print(f"join_citations: {len(licences):,} licence answers "
              f"({sum(1 for v in licences.values() if v[0]):,} disclosed a licence, the rest are "
              f"a real 'none disclosed')")
        # A licence-only backfill has no counts for records it did not also refresh, so the
        # citation index must not gate which rows are written.
        index = index or {}

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_matched = n_unmatched = n_no_key = n_licence = 0
    by_key_type: dict[str, int] = {"pmid": 0, "doi": 0, "pmcid": 0}
    seen_pids: set[str] = set()
    dupe_pids = 0

    sink = None if report_only else tmp_path.open("w", newline="", encoding="utf-8")
    try:
        writer = None
        if sink is not None:
            columns = OUTPUT_COLUMNS + (["license", "epmc_is_open_access"] if with_licence else [])
            writer = csv.DictWriter(sink, fieldnames=columns)
            writer.writeheader()

        corpus_iter = (
            iter_corpus_from_moros() if corpus_from_moros
            else csv.DictReader(corpus_path.open(newline="", encoding="utf-8", errors="replace"))
        )
        if True:
            for row in tqdm(corpus_iter, desc="corpus", unit="row"):
                n_rows += 1
                assigned = assign_key(row.get("pmid") or "", row.get("pmcid") or "",
                                      row.get("doi") or "")
                if assigned is None:
                    n_no_key += 1
                    continue
                hit = index.get(assigned)
                licence_hit = licences.get(assigned) if with_licence else None
                if hit is None and licence_hit is None:
                    n_unmatched += 1
                    continue
                pid = (row.get("pid") or "").strip()
                if not pid:
                    raise ValueError(
                        f"row {n_rows} has no pid -- run add_pid_column.py first; this join cannot "
                        f"invent a document _id"
                    )
                if pid in seen_pids:
                    dupe_pids += 1
                    continue  # one count per document; first occurrence wins
                seen_pids.add(pid)
                n_matched += 1
                by_key_type[assigned[0]] += 1
                if writer is not None:
                    count, fetched_at = hit if hit is not None else ("", "")
                    out = {
                        "pid": pid,
                        "citation_count": count,
                        "citation_count_updated": fetched_at,
                        "citation_source": CITATION_SOURCE if count else "",
                    }
                    if with_licence:
                        licence, epmc_oa = licence_hit if licence_hit is not None else ("", "")
                        out["license"] = licence
                        out["epmc_is_open_access"] = epmc_oa
                        if licence_hit is not None:
                            n_licence += 1
                    writer.writerow(out)
    finally:
        if sink is not None:
            sink.close()

    print(f"\njoin_citations: {n_rows:,} corpus rows")
    print(f"  matched a count      : {n_matched:,} ({n_matched / n_rows * 100:.2f}%)")
    print(f"    by pmid            : {by_key_type['pmid']:,}")
    print(f"    by doi             : {by_key_type['doi']:,}")
    print(f"    by pmcid           : {by_key_type['pmcid']:,}")
    print(f"  no count available   : {n_unmatched:,} ({n_unmatched / n_rows * 100:.2f}%)")
    print(f"  no usable identifier : {n_no_key:,}")
    print(f"  duplicate pid skipped: {dupe_pids:,}")
    if with_licence:
        print(f"  carried a licence    : {n_licence:,}")

    if report_only:
        print("\njoin_citations: --report-only, nothing written.")
        return

    os.replace(tmp_path, output_path)  # atomic -- a crash leaves the old file, never a partial one
    print(f"\njoin_citations: wrote {n_matched:,} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--with-licence", action="store_true",
                        help="Also carry license / epmc_is_open_access through, from a fetch run "
                             "made with --with-licence.")
    parser.add_argument("--corpus-from-moros", action="store_true",
                        help="Take pid and identifiers from the live collection instead of the "
                             "corpus CSV. Required for anything loaded after the original "
                             "landscape -- the curated merge and every incremental batch exist "
                             "only in Mongo.")
    args = parser.parse_args()
    run(args.citations, args.corpus, args.output, args.report_only,
        args.with_licence, args.corpus_from_moros)


if __name__ == "__main__":
    main()
