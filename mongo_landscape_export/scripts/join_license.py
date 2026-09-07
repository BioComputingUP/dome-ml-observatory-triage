"""Step 24 phase 1, Step 3d(ii): joins the real Europe PMC license data
(`epmc_licensing/output/epmc_pmid_licensing.csv`, freshly re-fetched to close the coverage gap --
see Database_STEPS_Progress.md) onto both CSVs in this folder, keyed on `pmid`.

Adds three new columns rather than overwriting the existing `is_open_access` column, so no raw
data is silently lost:
  - `license`             -- the EPMC-disclosed license string (e.g. "cc by-nc-nd"), "" when EPMC
                              was checked and disclosed none, "" (same as never-checked at the raw
                              CSV level -- see `license_checked` to tell them apart)
  - `license_checked`     -- "True"/"False": whether this pmid was actually found in the licensing
                              file at all. False covers both "no pmid to look up" and "pmid present
                              but EPMC has no SRC:MED record for it" (a confirmed, stable, genuine
                              miss -- 3,041 such pmids, re-verified via a second fetch run finding 0
                              new results).
  - `epmc_is_open_access` -- EPMC's own freshly-fetched Y/N/"" flag, kept separate from the
                              existing `is_open_access` column (which disagrees with this on 395
                              rows -- confirmed live). Resolution policy (EPMC's value wins when
                              checked, else fall back to the original column) is applied later in
                              schema.py's build_document, not mutated here -- this script's only
                              job is attaching the raw fetched data.

Streams both CSVs in place via a `.tmp` + atomic `os.replace`, same shape as the rest of this
folder's tooling.

Usage:
    python3 join_license.py [--usable PATH] [--removed PATH] [--licensing PATH]
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent  # mongo_landscape_export/ -- scripts live one level below the data
REPO_ROOT = FOLDER_DIR.parent
DEFAULT_USABLE = FOLDER_DIR / "ai_ml_landscape_classified_usable.csv"
DEFAULT_REMOVED = FOLDER_DIR / "missingness_removed_ai_ml_landscape_classified.csv"
DEFAULT_LICENSING = REPO_ROOT / "epmc_licensing" / "output" / "epmc_pmid_licensing.csv"

NEW_COLUMNS = ("license", "license_checked", "epmc_is_open_access")


def load_licensing(path: Path) -> dict[str, tuple[str, str]]:
    """pmid -> (license, is_open_access), both exactly as EPMC returned them."""
    table: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[row["pmid"].strip()] = (row["license"], row["is_open_access"])
    return table


def join_license(path: Path, licensing: dict[str, tuple[str, str]]) -> dict:
    """Rewrites `path` in place with the three new columns appended. Returns a report dict:
    rows_total, rows_checked, rows_not_checked. Raises if any of NEW_COLUMNS is already present."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rows_total = 0
    rows_checked = 0

    with path.open(newline="", encoding="utf-8") as f_in, tmp_path.open(
        "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.reader(f_in)
        header = next(reader)
        already_present = [c for c in NEW_COLUMNS if c in header]
        if already_present:
            raise ValueError(f"{path}: {already_present} already present -- refusing to join twice")

        pmid_idx = header.index("pmid")
        writer = csv.writer(f_out)
        writer.writerow(header + list(NEW_COLUMNS))

        for row in reader:
            rows_total += 1
            pmid = row[pmid_idx].strip()
            entry = licensing.get(pmid)
            if entry is None:
                writer.writerow(row + ["", "False", ""])
            else:
                rows_checked += 1
                license_, epmc_oa = entry
                writer.writerow(row + [license_, "True", epmc_oa])

    os.replace(tmp_path, path)
    return {
        "rows_total": rows_total,
        "rows_checked": rows_checked,
        "rows_not_checked": rows_total - rows_checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usable", type=Path, default=DEFAULT_USABLE)
    parser.add_argument("--removed", type=Path, default=DEFAULT_REMOVED)
    parser.add_argument("--licensing", type=Path, default=DEFAULT_LICENSING)
    args = parser.parse_args()

    print(f"loading licensing table from {args.licensing} ...")
    licensing = load_licensing(args.licensing)
    print(f"  {len(licensing):,} pmids with a license lookup available")

    for label, path in (("usable", args.usable), ("removed", args.removed)):
        print(f"joining license data onto {path} ...")
        report = join_license(path, licensing)
        print(
            f"  {label}: {report['rows_total']:,} rows, "
            f"{report['rows_checked']:,} matched a license lookup, "
            f"{report['rows_not_checked']:,} did not (no pmid, or a genuine EPMC miss)"
        )


if __name__ == "__main__":
    main()
