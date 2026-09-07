"""Turns a fetched search-space window into a classify-ready CSV of *genuinely new* records only.

The middle step of the incremental loop: `fetch_search_space.py` fetches windows that were never
fetched, and this decides which of those records moros has never seen. Both filters are needed --
a window can be new while most of its records are not (EPMC back-fills and re-indexes older
records continuously, so a 2026 window legitimately returns papers already in the corpus).

The check is a batched `$in` against the `_id` index using the deterministic UUID5 from `pid.py`,
so "already in the corpus" costs three index lookups per thousand records rather than a scan, and
means the same thing here as everywhere else in the project.

Output columns match what `llm-classify classify` needs plus the metadata the staging chain will
want later, so nothing has to be re-fetched between classification and loading.

    python3 build_incoming_documents.py --incoming ../output/incoming/<hash>
    python3 build_incoming_documents.py --incoming ... --out ../output/incoming_new.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from moros_client import Moros

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                       / "mongo_landscape_export" / "scripts"))
from pid import mint_landscape_pid  # noqa: E402  -- see the sys.path note below

# `pid.py` is imported from mongo_landscape_export/scripts rather than copied. The UUID5 minting
# rule is the single property that makes every upsert in this project idempotent; a second
# implementation of it that drifted by one character would mint different _ids for the same papers
# and quietly duplicate the corpus. One direction only: this folder depends on that one, never the
# reverse.

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent

OUTPUT_COLUMNS = [
    "pid", "pmid", "pmcid", "doi", "title", "abstract", "authors", "year", "journal",
    "mesh_headings", "pub_types", "keywords_author", "is_open_access", "fulltext_available",
    "abstract_source", "epmc_source", "first_publication_date",
]


def epmc_record_to_row(record: dict) -> dict | None:
    """One EPMC `resultType=core` result -> one staging row. Returns None when the record has no
    usable identifier at all (measured 0 across 827,061 records, but a future window need not be
    so tidy, and minting a placeholder pid would be far worse than skipping)."""
    pmid = (record.get("pmid") or "").strip()
    pmcid = (record.get("pmcid") or "").strip()
    doi = (record.get("doi") or "").strip()
    try:
        pid = mint_landscape_pid(pmcid, doi, pmid)
    except ValueError:
        return None

    abstract = record.get("abstractText") or ""
    mesh = [
        term.get("descriptorName")
        for term in (record.get("meshHeadingList") or {}).get("meshHeading", [])
        if term.get("descriptorName")
    ]
    pub_types = list((record.get("pubTypeList") or {}).get("pubType", []) or [])
    keywords = list((record.get("keywordList") or {}).get("keyword", []) or [])

    return {
        "pid": pid,
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": doi,
        "title": record.get("title") or "",
        "abstract": abstract,
        "authors": record.get("authorString") or "",
        "year": record.get("pubYear") or "",
        "journal": (record.get("journalInfo") or {}).get("journal", {}).get("title", "")
                   or record.get("journalTitle") or "",
        "mesh_headings": json.dumps(mesh),
        "pub_types": json.dumps(pub_types),
        "keywords_author": json.dumps(keywords),
        "is_open_access": "True" if record.get("isOpenAccess") == "Y" else "False",
        "fulltext_available": "True" if record.get("inEPMC") == "Y" else "False",
        "abstract_source": "europepmc" if abstract else "",
        "epmc_source": record.get("source") or "",
        "first_publication_date": record.get("firstPublicationDate") or "",
    }


def iter_incoming(incoming_dir: Path):
    files = sorted(incoming_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no .jsonl files in {incoming_dir} -- run fetch_search_space.py first")
    print(f"reading {len(files)} window file(s) from {incoming_dir}")
    for path in files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def run(incoming_dir: Path, out_path: Path) -> None:
    rows: dict[str, dict] = {}
    n_read = n_no_id = 0
    for record in tqdm(iter_incoming(incoming_dir), desc="staging", unit="rec"):
        n_read += 1
        row = epmc_record_to_row(record)
        if row is None:
            n_no_id += 1
            continue
        # Within a fetch, a paper can appear in more than one window (a record re-indexed across
        # a year boundary). First occurrence wins, matching dedupe_bulk_match_batch's convention.
        rows.setdefault(row["pid"], row)

    print(f"\n{n_read:,} records read, {n_no_id:,} skipped for having no usable identifier, "
          f"{len(rows):,} distinct papers")

    with Moros.from_env() as moros:
        print(f"checking against {moros.describe()}")
        existing = moros.existing_ids(tqdm(list(rows), desc="dedupe vs moros", unit="id"))

    new_rows = [row for pid, row in rows.items() if pid not in existing]
    print(f"\n{len(existing):,} already in the corpus, {len(new_rows):,} genuinely new")
    if not new_rows:
        print("nothing new to classify -- the corpus is already current for these windows.")
        return

    with_abstract = sum(1 for r in new_rows if r["abstract"])
    print(f"  of the new records, {with_abstract:,} have an abstract "
          f"({len(new_rows) - with_abstract:,} do not and cannot be classified on content)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(new_rows, key=lambda r: r["pid"]))
    os.replace(tmp, out_path)
    print(f"\nwrote {len(new_rows):,} new records -> {out_path}")
    print(f"""
next:
  1. classify them (paid -- project the cost first):
       docker compose run --rm pipeline dome-triage llm-classify classify \\
           --scope staged_file --input /app/{out_path.relative_to(FOLDER_DIR.parent)} \\
           --tier flash --estimated-usd <X> --confirm
  2. fetch citations for them:  python3 fetch_citations.py --input {out_path}
  3. build documents, then:     python3 load_documents.py --input <jsonl> --confirm
  4. python3 ensure_indexes.py && python3 verify_corpus.py""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=FOLDER_DIR / "output" / "incoming_new.csv")
    args = parser.parse_args()
    run(args.incoming, args.out)


if __name__ == "__main__":
    main()
