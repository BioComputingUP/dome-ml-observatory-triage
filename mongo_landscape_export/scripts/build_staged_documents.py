"""Step 24 phase 2: turns a classified incremental batch into loadable MongoDB documents.

This is the last missing link in the continual triage loop. The chain is:

    fetch_search_space.py        only the time windows never covered
    build_incoming_documents.py  only the records moros has never seen  -> staged CSV
    llm-classify classify        --scope staged_file --input <staged CSV>  -> events CSV
    build_staged_documents.py    <- THIS, staged CSV + events -> documents JSONL
    load_documents.py            upsert into moros
    ensure_indexes.py / verify_corpus.py

`convert_to_jsonl.py` cannot do this job: it reads the 28-column landscape CSV that Step 23a
produced, and the staged shape is narrower and differently named. But the *document* must come out
identical, so this reuses **`schema.build_document()` unchanged** -- the same builder the landscape
and the curated merge both use. `test_both_builders_produce_the_same_shape` is what keeps all three
paths honest; if this file drifted, that test would not catch it, so it deliberately owns no
document shape of its own and only assembles the row `build_document` expects.

Two joins fill what neither input carries, both reusing existing machinery rather than
reimplementing it: licences from `epmc_licensing/output/epmc_pmid_licensing.csv` on the same
`load_licensing` contract `join_license.py` defines, and citation counts from
`citations_index.lookup_citation`, which tries pmid -> doi -> pmcid so a count fetched under any
key is found.

    python3 build_staged_documents.py --staged <csv> --events <csv> --report-only
    python3 build_staged_documents.py --staged <csv> --events <csv>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from citations_index import LOOKUP_ORDER, load_citation_index, lookup_citation
from schema import CITATION_SOURCE_EPMC, VALID_CLASSIFICATIONS, build_document

csv.field_size_limit(sys.maxsize)

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
REPO_DIR = FOLDER_DIR.parent

DEFAULT_LICENSING = REPO_DIR / "epmc_licensing" / "output" / "epmc_pmid_licensing.csv"
# A --with-licence fetch, keyed the same way citations are. Preferred over the static pmid table
# above, which was built once for the original corpus and is pmid-only: on the 2026-09-03
# incremental batch it matched just 10,043 of 13,476 records, and the 3,433 misses landed with
# `license: null` -- "never looked up" -- rather than a real answer.
DEFAULT_LICENCE_FETCH = REPO_DIR / "moros_pipeline" / "output" / "epmc_licence_backfill.csv"
DEFAULT_CITATIONS = REPO_DIR / "moros_pipeline" / "output" / "epmc_citations.csv"
# The batch's merged data links (build_data_links.py --keys <staged> --metadata <staged>), keyed on
# pid. Optional: a batch built without it gets `data_links.fetched_at: null` and is picked up by
# the next data-links refresh, exactly like a citation that was not fetched in time.
DEFAULT_DATA_LINKS = REPO_DIR / "moros_pipeline" / "output" / "pid_data_links.csv"
DATA_LINKS_COLUMNS = ("has_data", "data_links_tags", "accession_types", "db_cross_references",
                      "data_links_json")
# The batch's identifiers from the same build (`--out-identifiers`), keyed on pid: the DOME Registry
# entry naming the paper, or "" when it was looked up and none does. Optional, like the data links.
DEFAULT_IDENTIFIERS = REPO_DIR / "moros_pipeline" / "output" / "pid_identifiers.csv"

# What `build_document` reads off a row but the staged CSV has no source for.
_STAGED_BLANKS = ("metadata_repair_sources",)


def _blank(value: object) -> str:
    return "" if value is None else str(value).strip()


def load_licensing(path: Path) -> dict[str, tuple[str, str]]:
    """pmid -> (license, is_open_access). Same contract as `join_license.py::load_licensing`."""
    if not Path(path).exists():
        print(f"  WARNING {path} missing -- staged documents will have no licence")
        return {}
    table: dict[str, tuple[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[_blank(row.get("pmid"))] = (row.get("license") or "", row.get("is_open_access") or "")
    return table


def load_classifications(path: Path) -> dict[str, dict]:
    """record_id -> the latest non-error classification event. The event log is append-only and a
    re-run retries parse errors as new rows, so the last usable row for a record is the verdict."""
    latest: dict[str, dict] = {}
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rid = _blank(row.get("record_id"))
            classification = _blank(row.get("classification"))
            if not rid or classification not in VALID_CLASSIFICATIONS:
                continue  # parse_error rows carry no usable verdict
            latest[rid] = row
    return latest


def load_licence_fetch(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """(key_type, key) -> (license, epmc_is_open_access) from a `--with-licence` fetch.

    Keyed by identifier exactly as the citation index is, so `citations_index.LOOKUP_ORDER`
    resolves both and a record cannot take its licence from one key and its count from another."""
    if not Path(path).exists():
        return {}
    index: dict[tuple[str, str], tuple[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "license" not in row:
                continue
            index[(row["key_type"], row["key"])] = (
                (row.get("license") or "").strip(),
                (row.get("epmc_is_open_access") or "").strip(),
            )
    return index


def resolve_licence(pmid: str, pmcid: str, doi: str, licensing: dict,
                    licence_fetch: dict) -> tuple[str, str, str]:
    """(license, epmc_is_open_access, license_checked).

    Tries the multi-key fetch first, then the pmid-only static table. `license_checked == "True"`
    is what makes `schema.py` write `""` rather than `None`, so the distinction between "EPMC
    disclosed none" and "never looked up" survives into the document."""
    values = {"pmid": (pmid or "").strip(), "pmcid": (pmcid or "").strip(),
              "doi": (doi or "").strip().lower()}
    for key_type in LOOKUP_ORDER:
        value = values[key_type]
        if not value:
            continue
        hit = licence_fetch.get((key_type, value))
        if hit is not None:
            return hit[0], hit[1], "True"
    entry = licensing.get(pmid)
    if entry is not None:
        return entry[0], entry[1], "True"
    return "", "", "False"


def load_data_links(path: Path) -> dict[str, dict[str, str]]:
    """pid -> the data_links staging columns, from `build_data_links.py`'s output."""
    if not Path(path).exists():
        return {}
    index: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = _blank(row.get("pid"))
            if pid:
                index[pid] = {k: row.get(k) or "" for k in DATA_LINKS_COLUMNS if k in row}
    return index


def load_identifiers(path: Path) -> dict[str, str]:
    """pid -> dome_registry ("" kept: looked up, none), from `build_data_links.py`'s output."""
    if not Path(path).exists():
        return {}
    index: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = _blank(row.get("pid"))
            if pid and "dome_registry" in row:
                index[pid] = (row.get("dome_registry") or "").strip()
    return index


def build_row(staged: dict, event: dict, licensing: dict, citations: dict,
              licence_fetch: dict | None = None, data_links: dict | None = None,
              identifiers: dict | None = None) -> dict:
    """Assembles exactly the keys `schema.build_document` reads. Owns no shape of its own."""
    pmid, pmcid, doi = _blank(staged.get("pmid")), _blank(staged.get("pmcid")), _blank(staged.get("doi"))

    licence, epmc_oa, checked = resolve_licence(pmid, pmcid, doi, licensing,
                                                licence_fetch or {})

    citation = lookup_citation(citations, pmid=pmid, pmcid=pmcid, doi=doi)
    count, updated, source = citation if citation else ("", "", "")

    row = {**{k: "" for k in _STAGED_BLANKS}, **staged}
    row.update({
        "citation_count": count,
        "citation_count_updated": updated,
        "citation_source": CITATION_SOURCE_EPMC if count else "",
        "license": licence,
        "license_checked": checked,
        "epmc_is_open_access": epmc_oa,
        "classification": _blank(event.get("classification")),
        "rationale": _blank(event.get("rationale")),
        "model_tier": _blank(event.get("model_tier")),
        "mode": _blank(event.get("mode")),
        "prompt_version": _blank(event.get("prompt_version")),
        "criteria_sha256": _blank(event.get("criteria_sha256")),
        "batch_id": _blank(event.get("batch_id")),
        "timestamp": _blank(event.get("timestamp")),
    })
    # The staged row already carries the data-links summary off the search record; the merged
    # file adds the link detail (and repeats the summary, identically) for the pids it covers.
    row.update((data_links or {}).get(_blank(staged.get("pid")), {}))
    # A pid the identifiers file lists was looked up; `schema.py` then writes its value, "" included.
    pid = _blank(staged.get("pid"))
    if identifiers and pid in identifiers:
        row["dome_registry"] = identifiers[pid]
        row["dome_registry_checked"] = "True"
    return row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(staged_path: Path, events_path: Path, licensing_path: Path, citations_path: Path,
        out_jsonl: Path, out_report: Path, report_only: bool,
        licence_fetch_path: Path = DEFAULT_LICENCE_FETCH,
        data_links_path: Path = DEFAULT_DATA_LINKS,
        identifiers_path: Path = DEFAULT_IDENTIFIERS) -> None:
    print(f"1. reading {staged_path.name} and {events_path.name} ...")
    with staged_path.open(newline="", encoding="utf-8", errors="replace") as f:
        staged_rows = list(csv.DictReader(f))
    events = load_classifications(events_path)
    print(f"   {len(staged_rows):,} staged records, {len(events):,} with a usable classification")

    licensing = load_licensing(licensing_path)
    licence_fetch = load_licence_fetch(licence_fetch_path)
    if licence_fetch:
        print(f"   {len(licence_fetch):,} multi-key licence answers (preferred over the "
              f"pmid-only table)")
    citations = load_citation_index(citations_path) if Path(citations_path).exists() else {}
    print(f"   {len(licensing):,} pmids with a licence, {len(citations):,} citation keys")
    data_links = load_data_links(data_links_path)
    print(f"   {len(data_links):,} pids with merged data links"
          + ("" if data_links else " (none: run build_data_links.py on the batch, or let the "
                                   "next data-links refresh pick them up)"))
    identifiers = load_identifiers(identifiers_path)
    print(f"   {len(identifiers):,} pids with an identifiers row")

    print("\n2. building documents ...")
    documents, errors = [], []
    stats = Counter()
    seen: set[str] = set()
    for staged in tqdm(staged_rows, desc="documents", unit="rec"):
        pid = _blank(staged.get("pid"))
        if not pid:
            stats["no_pid"] += 1
            continue
        event = events.get(pid)
        if event is None:
            stats["unclassified"] += 1
            continue
        if pid in seen:
            stats["duplicate_pid"] += 1
            continue
        seen.add(pid)
        row = build_row(staged, event, licensing, citations, licence_fetch, data_links, identifiers)
        if row["citation_count"]:
            stats["with_citation"] += 1
        if row["license_checked"] == "True":
            stats["with_licence"] += 1
        if row.get("data_links_json"):
            stats["with_data_links"] += 1
        if row.get("dome_registry"):
            stats["with_dome_registry"] += 1
        try:
            documents.append(build_document(row))
            stats[row["classification"]] += 1
        except Exception as exc:  # noqa: BLE001 -- collect, never write a partial file
            errors.append({"pid": pid, "error": repr(exc)})

    print(f"   {len(documents):,} built, {len(errors)} error(s)")
    for key in ("positive", "negative", "undeterminable"):
        if stats[key]:
            print(f"     {key:<16} {stats[key]:,}")
    print(f"     with a citation  {stats['with_citation']:,}")
    print(f"     with a licence   {stats['with_licence']:,}")
    print(f"     with data links  {stats['with_data_links']:,}")
    print(f"     with a DOME entry {stats['with_dome_registry']:,}")
    if stats["unclassified"]:
        print(f"     NOT classified   {stats['unclassified']:,} (left out -- classify them first)")
    if stats["duplicate_pid"] or stats["no_pid"]:
        print(f"     skipped          {stats['duplicate_pid']:,} duplicate pid, "
              f"{stats['no_pid']:,} with no pid")
    for err in errors[:5]:
        print(f"     {err}")
    if errors:
        raise SystemExit("refusing to write with build errors -- fix them first")

    if report_only:
        print("\nbuild_staged_documents: --report-only, nothing written.")
        return

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    os.replace(tmp, out_jsonl)
    print(f"\n3. wrote {len(documents):,} documents -> {out_jsonl}")

    out_report.write_text(json.dumps({
        "inputs": {
            "staged": {"path": str(staged_path), "rows": len(staged_rows),
                       "sha256": sha256_file(staged_path)},
            "events": {"path": str(events_path), "classified": len(events)},
        },
        "stats": dict(stats),
        "documents_written": len(documents),
        "errors": errors,
        "output": {"path": str(out_jsonl), "sha256": sha256_file(out_jsonl)},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"   report -> {out_report}")
    print(f"\nnext: python3 moros_pipeline/scripts/load_documents.py --input {out_jsonl} --dry-run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", type=Path, required=True,
                        help="build_incoming_documents.py's output CSV.")
    parser.add_argument("--events", type=Path, required=True,
                        help="The classification event log for that staged batch.")
    parser.add_argument("--licensing", type=Path, default=DEFAULT_LICENSING)
    parser.add_argument("--licence-fetch", type=Path, default=DEFAULT_LICENCE_FETCH,
                        help="A --with-licence fetch output; preferred over the pmid-only table.")
    parser.add_argument("--citations", type=Path, default=DEFAULT_CITATIONS)
    parser.add_argument("--data-links", type=Path, default=DEFAULT_DATA_LINKS,
                        help="build_data_links.py's pid_data_links.csv for this batch.")
    parser.add_argument("--identifiers", type=Path, default=DEFAULT_IDENTIFIERS,
                        help="build_data_links.py's --out-identifiers file for this batch.")
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    out_jsonl = args.out_jsonl or (FOLDER_DIR / "output" / f"{args.staged.stem}_documents.jsonl")
    out_report = args.out_report or out_jsonl.with_suffix(".report.json")
    run(args.staged, args.events, args.licensing, args.citations, out_jsonl, out_report,
        args.report_only, args.licence_fetch, args.data_links, args.identifiers)


if __name__ == "__main__":
    main()
