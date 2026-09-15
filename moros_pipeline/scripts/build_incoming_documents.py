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

from fetch_epmc_metadata import identity_fields, summary_fields  # noqa: E402

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
    # Schema v1.3.0 / v1.4.0, read straight off the same `core` search record -- zero extra HTTP.
    # `epmc_source` above predates them and is kept in place; the identity and the data-links
    # summary use the exact readers the retrospective backfill uses (fetch_epmc_metadata.py).
    "epmc_id", "preprint_server",
    "has_data", "data_links_tags", "accession_types", "db_cross_references",
    "has_tm_accessions", "has_db_xrefs", "has_suppl",
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
        "epmc_id": identity_fields(record)["epmc_id"],
        "preprint_server": identity_fields(record)["preprint_server"],
        **summary_fields(record),
    }


IDENTITY_FIELDS = ("pmcid", "doi", "pmid")


def _norm_id(field: str, value) -> str:
    value = str(value or "").strip()
    return value.lower() if field in ("doi", "pmcid") else value


def known_identifiers(moros: Moros, rows: list[dict], batch_size: int = 5_000) -> dict[str, dict[str, str]]:
    """For each identity field, which of these rows' values moros already holds, mapped to the
    `_id` holding it. Batched `$in` on `identifiers.<field>`, read-only."""
    known: dict[str, dict[str, str]] = {f: {} for f in IDENTITY_FIELDS}
    for field in IDENTITY_FIELDS:
        values = sorted({str(r.get(field) or "").strip() for r in rows if str(r.get(field) or "").strip()})
        for i in range(0, len(values), batch_size):
            chunk = values[i:i + batch_size]
            if field in ("doi", "pmcid"):
                chunk = sorted(set(chunk) | {v.lower() for v in chunk} | {v.upper() for v in chunk})
            cursor = moros.collection.find({f"identifiers.{field}": {"$in": chunk}},
                                           {"_id": 1, f"identifiers.{field}": 1}, max_time_ms=300_000)
            for doc in cursor:
                value = (doc.get("identifiers") or {}).get(field)
                if value:
                    known[field][_norm_id(field, value)] = doc["_id"]
    return known


def drop_known_identifiers(rows: list[dict], known: dict[str, dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """Splits rows new by `_id` into (genuinely new, already in the corpus under another `_id`).

    The `_id` is minted from pmcid > doi > pmid, so a paper that gains a PMCID after it was loaded
    mints a different `_id` and passes the `_id` check as new. On 2026-09-15 that was 2,361 of 45,271
    "new" records. Loading them would duplicate the paper, so a row whose pmcid, doi or pmid is
    already held is set aside with the `_id` that holds it."""
    new, already = [], []
    for row in rows:
        holder = next((known[f][_norm_id(f, row.get(f))] for f in IDENTITY_FIELDS
                       if _norm_id(f, row.get(f)) and _norm_id(f, row.get(f)) in known[f]), None)
        if holder:
            already.append({"pid": row["pid"], "existing_id": holder,
                            **{f: row.get(f) or "" for f in IDENTITY_FIELDS}})
        else:
            new.append(row)
    return new, already


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
        new_by_id = [row for pid, row in rows.items() if pid not in existing]
        new_rows, under_older_id = drop_known_identifiers(new_by_id, known_identifiers(moros, new_by_id))

    print(f"\n{len(existing):,} already in the corpus, {len(under_older_id):,} already in it under an "
          f"older _id (matched by pmcid, doi or pmid), {len(new_rows):,} genuinely new")
    if under_older_id:
        side = out_path.with_name(out_path.stem + "_known_under_older_id.csv")
        side.parent.mkdir(parents=True, exist_ok=True)
        with side.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pid", "existing_id", *IDENTITY_FIELDS])
            writer.writeheader()
            writer.writerows(under_older_id)
        print(f"  set aside, not staged: {side}")
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
    try:
        container_input = f"/app/{out_path.resolve().relative_to(FOLDER_DIR.parent)}"
    except ValueError:  # --out outside the repository: the container cannot see it
        container_input = "<copy it under moros_pipeline/output/ first; the container sees only the repository>"
    print(f"""
next:
  1. classify them (paid -- project the cost first):
       docker compose run --rm pipeline dome-triage llm-classify classify \\
           --scope staged_file --input {container_input} \\
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
