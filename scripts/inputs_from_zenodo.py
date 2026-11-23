"""Builds enrichment input CSVs from the public Zenodo corpus dump, so the EBI laptop needs no moros access.

Two modes:

  --cohort NAME   production inputs. Selects positives (`llm_classification.classification == "positive"`)
                  that are not enriched (`llm_enrichment.provider` null) and have a non-empty title and
                  abstract, in the cohort's journals (configs/enrichment_cohorts.yaml, matched exactly).
                  Writes <out-dir>/<cohort>/enrich_input_<cohort>_zenodo-<version>_cNN.csv in chunks of
                  --chunk-size, in the dump's `_id` order (or, for a cohort with `order: journals`, by the
                  journal's position in the cohort's list, then `_id`, so the first-listed journals run
                  first), plus SOURCE.md (record id, version, file sha256,
                  counts, selection), then rewrites the runs repository's SHA256SUMS.

  --ids-from CSV  test inputs. Every record whose `_id` is listed in CSV (column `record_id`, optionally
                  filtered by `--ids-cohort` on column `cohort`), whatever its classification or enrichment,
                  to --out FILE. Refuses if any listed id is missing from the dump.

Both write the columns `export_journal_for_enrichment.py` writes (`record_id, title, abstract, journal,
year`), with the same values, so a run cannot tell which one built its input.

Runs in the `tools` container, e.g.:
    docker compose run --rm tools python scripts/inputs_from_zenodo.py \\
        --dump ../zenodo_corpus/dome-observatory-corpus-2026-09-25.jsonl.gz \\
        --cohort core_bioinformatics --out-dir ../dome-ml-observatory-enrichment-runs/inputs
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_repo import sha256_file, write_sha256sums  # noqa: E402

OUTPUT_COLUMNS = ["record_id", "title", "abstract", "journal", "year"]
REPO = Path(__file__).resolve().parents[1]
COHORTS_FILE = REPO / "configs" / "enrichment_cohorts.yaml"


def load_cohort(name: str, path: Path = COHORTS_FILE) -> list[str]:
    cohorts = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cohorts", {})
    if name not in cohorts:
        raise SystemExit(f"unknown cohort {name!r}; known: {sorted(cohorts)}")
    journals = cohorts[name]["journals"]
    if len(set(journals)) != len(journals):
        raise SystemExit(f"cohort {name!r} lists a journal twice")
    return journals


ORDERS = ("dump", "journals", "reruns_then_journal_size")


@dataclass(frozen=True)
class Scope:
    """What a cohort covers beyond its journal list. All off by default (core and batch2 rebuild unchanged).

    all_journals     `scope: all` -- every journal, including records with no journal name
    rerun            `rerun: [e1]` -- also documents whose current enrichment is from these prompts; loading
                     the new events moves the old enrichment into `enrichment_history` (schema 1.8.0)
    exclude_cohorts  `exclude_cohorts: [...]` -- never repeat a record listed in those cohorts' inputs
    """
    all_journals: bool = False
    rerun: tuple[str, ...] = ()
    exclude_cohorts: tuple[str, ...] = ()


def load_cohort_scope(name: str, path: Path = COHORTS_FILE) -> Scope:
    spec = ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cohorts", {})).get(name) or {}
    scope = spec.get("scope", "listed")
    if scope not in ("listed", "all"):
        raise SystemExit(f"cohort {name!r}: unknown scope {scope!r}; known: ('listed', 'all')")
    if scope == "all" and spec.get("journals"):
        raise SystemExit(f"cohort {name!r}: `scope: all` takes no journal list")
    return Scope(scope == "all", tuple(spec.get("rerun") or ()), tuple(spec.get("exclude_cohorts") or ()))


def excluded_ids(out_dir: Path, cohorts: Iterable[str]) -> set[str]:
    """Every record id in the named cohorts' input CSVs. Refuses if a cohort's inputs are missing: silently
    excluding nothing would re-enrich a whole cohort."""
    ids: set[str] = set()
    for cohort in cohorts:
        files = sorted((out_dir / cohort).glob("*.csv"))
        if not files:
            raise SystemExit(f"refusing: exclude_cohorts names {cohort!r} but {out_dir / cohort} has no inputs")
        for f in files:
            with f.open(newline="", encoding="utf-8") as fh:
                ids.update(r["record_id"] for r in csv.DictReader(fh))
    return ids


def load_cohort_order(name: str, path: Path = COHORTS_FILE) -> str:
    """`dump` (the default: the dump's `_id` order) or `journals` (the cohort's journal order, then `_id`)."""
    cohorts = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cohorts", {})
    order = (cohorts.get(name) or {}).get("order", "dump")
    if order not in ORDERS:
        raise SystemExit(f"cohort {name!r}: unknown order {order!r}; known: {ORDERS}")
    return order


def iter_docs(dump: Path) -> Iterator[dict]:
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def to_row(doc: dict) -> dict:
    pm = doc.get("publication_metadata") or {}
    year = pm.get("year")
    return {"record_id": doc["_id"], "title": pm.get("title") or "", "abstract": pm.get("abstract") or "",
            "journal": pm.get("journal") or "", "year": "" if year is None else str(year)}


def is_rerun(doc: dict, rerun: Iterable[str] = ()) -> bool:
    """The document already holds an enrichment from one of the `rerun` prompts (e.g. e1), to be redone."""
    enr = doc.get("llm_enrichment") or {}
    return bool(enr.get("provider")) and enr.get("prompt_version") in set(rerun)


def selected(doc: dict, journals: set[str], all_journals: bool = False, rerun: Iterable[str] = ()) -> bool:
    pm = doc.get("publication_metadata") or {}
    return ((doc.get("llm_classification") or {}).get("classification") == "positive"
            and (not (doc.get("llm_enrichment") or {}).get("provider") or is_rerun(doc, rerun))
            and bool((pm.get("title") or "").strip()) and bool((pm.get("abstract") or "").strip())
            and (all_journals or (pm.get("journal") or "") in journals))


def chunked(rows: list[dict], size: int) -> Iterable[list[dict]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def dump_version(dump: Path) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", dump.name)
    if not m:
        raise SystemExit(f"cannot read the dump's version (a date) from {dump.name!r}")
    return m.group(1)


def journal_table(per_journal: dict[str, int], limit: int | None = None) -> list[str]:
    """One row per journal; with a limit, the largest `limit` journals and one row for the rest."""
    items = list(per_journal.items()) if limit is None else sorted(per_journal.items(), key=lambda kv: -kv[1])
    rows = [f"| {j or '(no journal name)'} | {n:,} |" for j, n in items[:limit]]
    if limit is not None and len(items) > limit:
        rest = items[limit:]
        rows.append(f"| {len(rest):,} other journals | {sum(n for _, n in rest):,} |")
    return rows


def build_cohort(dump: Path, cohort: str, out_dir: Path, chunk_size: int, zenodo_record: str) -> list[Path]:
    journals = load_cohort(cohort)
    wanted = set(journals)
    scope = load_cohort_scope(cohort)
    skip = excluded_ids(out_dir, scope.exclude_cohorts)
    rows, reruns = [], set()
    for d in iter_docs(dump):
        if d["_id"] in skip or not selected(d, wanted, scope.all_journals, scope.rerun):
            continue
        rows.append(to_row(d))
        if is_rerun(d, scope.rerun):
            reruns.add(d["_id"])
    order = load_cohort_order(cohort)
    if order == "journals":
        position = {j: i for i, j in enumerate(journals)}
        rows.sort(key=lambda r: position[r["journal"]])  # stable: `_id` order within a journal
    elif order == "reruns_then_journal_size":
        size: dict[str, int] = {}
        for r in rows:
            size[r["journal"]] = size.get(r["journal"], 0) + 1
        # re-runs first, then the largest journals, records with no journal name last; `_id` order within
        rows.sort(key=lambda r: (r["record_id"] not in reruns, r["journal"] == "", -size[r["journal"]], r["journal"]))
    per_journal = {j: 0 for j in journals}
    for r in rows:
        per_journal[r["journal"]] = per_journal.get(r["journal"], 0) + 1
    version = dump_version(dump)
    target = out_dir / cohort
    paths = []
    for n, chunk in enumerate(chunked(rows, chunk_size), start=1):
        path = target / f"enrich_input_{cohort}_zenodo-{version}_c{n:02d}.csv"
        write_csv(path, chunk)
        paths.append(path)
    source = [
        f"# Inputs: cohort `{cohort}`",
        "",
        f"Built by `dome-ml-observatory-triage/scripts/inputs_from_zenodo.py --cohort {cohort}`.",
        "",
        f"- Zenodo record: {zenodo_record or 'not given'}; dump version {version}",
        f"- Dump file: `{dump.name}`, sha256 `{sha256_file(dump)}`",
        *(["- Selection: classification `positive`, `llm_enrichment.provider` null, non-empty title and abstract,",
            "  journal exactly one of the cohort's (configs/enrichment_cohorts.yaml)."] if scope == Scope() else
          ["- Selection: classification `positive`, non-empty title and abstract, and",
           f"  - journals: {'every journal, including records with no journal name' if scope.all_journals else 'exactly the cohort list'};",
           "  - enrichment: none yet" + (f", or a current enrichment from prompt {', '.join(scope.rerun)} to redo "
                                         f"({len(reruns):,} records; loading moves the old one to `enrichment_history`)"
                                         if scope.rerun else "") + ";",
           f"  - excluded: every record in the inputs of {', '.join(scope.exclude_cohorts) or 'no other cohort'}"
           f" ({len(skip):,} ids)."]),
        "- Order: " + {"journals": "the cohort's journal order, then the dump's `_id` order.",
                       "reruns_then_journal_size": "re-runs first, then journals largest first, records with no "
                                                   "journal name last; the dump's `_id` order within each.",
                       }.get(order, "the dump's `_id` order."),
        f"- Records: {len(rows):,} in {len(paths)} chunk(s) of at most {chunk_size:,}",
        "",
        "| Journal | Records |",
        "|---|---:|",
        *journal_table(per_journal, limit=60 if scope.all_journals else None),
        "",
        "| Chunk | Records | sha256 |",
        "|---|---:|---|",
        *[f"| `{p.name}` | {sum(1 for _ in csv.DictReader(p.open(encoding='utf-8'))):,} | `{sha256_file(p)}` |"
          for p in paths],
        "",
    ]
    (target / "SOURCE.md").write_text("\n".join(source), encoding="utf-8")
    missing = [j for j, n in per_journal.items() if n == 0 and j in wanted]
    if missing:
        print(f"note: no records for {missing} (already enriched, or the name does not match the dump)")
    print(f"{cohort}: {len(rows):,} records -> {len(paths)} chunk(s) in {target}")
    return paths


def build_ids(dump: Path, ids_csv: Path, ids_cohort: str | None, out: Path) -> int:
    with ids_csv.open(newline="", encoding="utf-8") as f:
        ids = [r["record_id"] for r in csv.DictReader(f) if ids_cohort is None or r.get("cohort") == ids_cohort]
    if not ids:
        raise SystemExit(f"no ids selected from {ids_csv} (cohort {ids_cohort!r})")
    wanted = set(ids)
    found = {d["_id"]: to_row(d) for d in iter_docs(dump) if d["_id"] in wanted}
    missing = wanted - set(found)
    if missing:
        raise SystemExit(f"refusing: {len(missing)} id(s) are not in the dump, e.g. {sorted(missing)[:3]}")
    write_csv(out, [found[i] for i in ids])
    print(f"{len(ids)} records -> {out}")
    return len(ids)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cohort")
    mode.add_argument("--ids-from", type=Path)
    ap.add_argument("--out-dir", type=Path, help="--cohort: the runs repository's inputs/ folder")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--zenodo-record", default="", help="recorded in SOURCE.md, e.g. 22967974")
    ap.add_argument("--ids-cohort", help="--ids-from: keep only rows whose `cohort` column equals this")
    ap.add_argument("--out", type=Path, help="--ids-from: the output CSV")
    args = ap.parse_args(argv)
    if args.cohort:
        if not args.out_dir:
            ap.error("--cohort needs --out-dir")
        build_cohort(args.dump, args.cohort, args.out_dir, args.chunk_size, args.zenodo_record)
        repo = args.out_dir.resolve().parent
        if (repo / "SHA256SUMS").exists() or (repo / ".git").exists():
            print(f"rewrote {write_sha256sums(repo)}")
    else:
        if not args.out:
            ap.error("--ids-from needs --out")
        build_ids(args.dump, args.ids_from, args.ids_cohort, args.out)


if __name__ == "__main__":
    main()
