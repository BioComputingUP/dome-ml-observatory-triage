"""Exports a cohort of positive records out of moros as an enrichment-ready input CSV.

This is the reusable half of "enrich a journal" or "enrich this batch": `--journal` and
`--batch-id` are arguments, so the next cohort is one command rather than a new script. It reads
from **moros, not the staging CSV**, which is what makes it work on the whole corpus including the
curated records that were only just merged -- a CSV-based export would miss exactly the records
this project cares most about.

The output carries only `record_id, title, abstract, journal, year`. That is the identical field
set `llm_classify/sampling.py::strip_for_api` sends to the API -- the blinding boundary -- so no
label, provenance or classification can reach the model even by accident. `record_id` is the Mongo
`_id`, which is also the merge key `load_enrichment.py` writes back on.

⚠️ Always pass the resulting CSV to `enrich` together with a dedicated `--events-out`. The
`record_id` here is a UUID5; the Step 20j trial log's is a sha1 from `canonical_dataset.csv`.
Appending one to the other conflates two identifier spaces in one event log and breaks resumption
for both. (`enrich --events-out` was declared but silently dropped until 2026-09-03 --
`tests/test_enrich_events_out.py` exists so it cannot regress.)

    python3 export_journal_for_enrichment.py --journal "Bioinformatics (Oxford, England)"
    python3 export_journal_for_enrichment.py --journal "Nature" --classification positive --limit 50
    python3 export_journal_for_enrichment.py --batch-id classify_flash_staged_file_primary_20260903T201216 --limit 200 --max-usd 3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from moros_client import ID_FIELD, Moros

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
DEFAULT_OUT_DIR = FOLDER_DIR / "output"

OUTPUT_COLUMNS = ["record_id", "title", "abstract", "journal", "year"]

# What an export is costed at before its input file exists. **$1.80 per 1,000 records**: the higher of
# the two V4.1 runs billed on 2026-09-15 ($1.80 per 1,000 for 100 Bioinformatics records, $1.00 for 200
# records of a refresh batch). V4-Flash was billed about $10 per 1,000, and the token model's 4.07 was
# tokens x list price from 100 paired Bioinformatics records (5,947 output tokens per record, 98%
# of them reasoning), never checked against the balance, and it let this gate admit about 2.5x the
# spend its --max-usd said. Replace it only with a balance delta from
# data/processed/cost_estimates/deepseek_real_cost_log.csv, never with a token-model figure.
#
# Enrichment cost is strongly population-dependent: Bioinformatics records think about three times
# as long as the Step 20j trial's (median 5,627 output tokens vs 2,163).
#
# Attempts to reduce it, all measured and all rejected (thinking_ablation/run_effort_ablation.py,
# 2026-09-03): `reasoning_effort: low` cost MORE than the default with 6x the violations; a
# hierarchical domain rendering saved nothing; disabling thinking is 88% cheaper but agrees with the
# production configuration on all six fields for 0% of records.
USD_PER_1000_RECORDS = 1.80


def slugify(value: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in value]
    return "".join(keep).strip("_").replace("__", "_")[:60]


def build_query(journals: list[str] | None, classification: str | None,
                include_enriched: bool, batch_ids: list[str] | None = None) -> dict:
    """`$in` over the journal names and/or the classification batch ids, so several journals or
    batches are one export and one events file rather than one run to babysit each. Journal names
    are matched VERBATIM and are fuller than expected -- Science is stored as "Science (New York,
    N.Y.)". A batch id is the `llm_classification.batch_id` a load stamped, as `verify_corpus.py`
    lists under `batch_ids`."""
    if not journals and not batch_ids:
        raise ValueError("a cohort needs at least one journal or classification batch id")
    query: dict = {}
    if journals:
        query["publication_metadata.journal"] = {"$in": journals}
    if batch_ids:
        query["llm_classification.batch_id"] = {"$in": batch_ids}
    if classification:
        query["llm_classification.classification"] = classification
    if not include_enriched:
        # Already-enriched documents are skipped so a re-run after a partial merge does not pay
        # to redo them. `enrich` also resumes from its own event log; this is the cheaper filter.
        query["llm_enrichment.provider"] = None
    return query


def projected_usd(n_match: int, limit: int | None) -> float:
    """What the export will cost once enriched: the records it will actually write, which is the
    `--limit` when one is given, at the billed rate."""
    n = n_match if limit is None else min(n_match, limit)
    return n / 1000 * USD_PER_1000_RECORDS


def run(journals: list[str] | None, classification: str | None, out_path: Path, limit: int | None,
        include_enriched: bool, max_usd: float | None, batch_ids: list[str] | None = None) -> None:
    query = build_query(journals, classification, include_enriched, batch_ids)
    out_path = Path(out_path).resolve()

    projection = {
        ID_FIELD: 1,
        "publication_metadata.title": 1,
        "publication_metadata.abstract": 1,
        "publication_metadata.journal": 1,
        "publication_metadata.year": 1,
    }

    with Moros.from_env() as moros:
        print(f"target {moros.describe()}\n")
        cohorts = ([("journal", "publication_metadata.journal", name) for name in journals or []]
                   + [("batch", "llm_classification.batch_id", bid) for bid in batch_ids or []])
        print(f"{'cohort':<60}{'documents':>11}{'to enrich':>11}")
        missing = []
        for kind, field, value in cohorts:
            total = moros.count({field: value})
            if total == 0:
                missing.append((kind, value))
            one = build_query([value] if kind == "journal" else None, classification,
                              include_enriched, [value] if kind == "batch" else None)
            print(f"{value[:58]:<60}{total:>11,}{moros.count(one):>11,}")
        if missing:
            raise SystemExit(
                f"\nno documents for {missing}. Journal names are matched verbatim and are often "
                f"fuller than expected -- Bioinformatics is stored as 'Bioinformatics (Oxford, "
                f"England)' and Science as 'Science (New York, N.Y.)'. Check /api/facets/journal "
                f"for the exact string. Batch ids are listed by verify_corpus.py under batch_ids."
            )
        n_match = moros.count(query)
        print(f"{'TOTAL':<60}{'':>11}{n_match:>11,}\n")

        # The budget gate. `enrich` deliberately has no --estimated-usd/--confirm gate (Gavin's
        # 2026-08-27 call), so this is the one place a runaway is stopped -- before the input file
        # exists, rather than after the money is spent.
        projected = projected_usd(n_match, limit)
        if max_usd is not None and projected > max_usd:
            raise SystemExit(
                f"REFUSING to write: {min(n_match, limit or n_match):,} records project to "
                f"${projected:.2f} at ${USD_PER_1000_RECORDS:.2f}/1k, over the ${max_usd:.2f} "
                f"limit. Narrow the cohort, pass a smaller --limit, or raise --max-usd deliberately."
            )

        rows = []
        for doc in tqdm(moros.iter_documents(query, projection), total=n_match,
                        desc="export", unit="doc"):
            pub = doc.get("publication_metadata", {})
            abstract = pub.get("abstract")
            if not (pub.get("title") and abstract):
                continue  # nothing to enrich from -- the prompt needs title+abstract
            rows.append({
                "record_id": doc[ID_FIELD],
                "title": pub.get("title") or "",
                "abstract": abstract,
                "journal": pub.get("journal") or "",
                "year": pub.get("year") if pub.get("year") is not None else "",
            })
            if limit is not None and len(rows) >= limit:
                break

    skipped = n_match - len(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_path)

    cost = len(rows) / 1000 * USD_PER_1000_RECORDS
    print(f"\nwrote {len(rows):,} records -> {out_path}")
    if skipped and limit is None:
        print(f"  {skipped:,} skipped for having no title or abstract to enrich from")
    print(f"\nprojected cost: {len(rows):,} records x ${USD_PER_1000_RECORDS:.2f}/1k = "
          f"**${cost:.2f}**"
          + (f" (limit ${max_usd:.2f})" if max_usd is not None else ""))
    print("  At the billed rate. Read the DeepSeek balance before and after the run and log the "
          "delta; ~3% of records are expected to hit the 16,000-token cap (finish_reason=length).")
    events = out_path.with_name(out_path.stem.replace("enrich_input_", "enrichment_") + "_events.csv")
    print(f"""
next -- smoke check 25 first, inspect, then run the rest:

  docker compose run --rm pipeline dome-triage llm-classify enrich \\
      --tier flash --concurrency 20 --limit 25 \\
      --input /app/{out_path.relative_to(FOLDER_DIR.parent)} \\
      --events-out /app/{events.relative_to(FOLDER_DIR.parent)}

then the same command without --limit, then:

  python3 load_enrichment.py --events {events}""")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    cohort = parser.add_mutually_exclusive_group(required=True)
    cohort.add_argument("--journal", action="append", dest="journals",
                        help="Exact journal string, matched verbatim. Repeatable.")
    cohort.add_argument("--batch-id", action="append", dest="batch_ids",
                        help="A classification batch id (llm_classification.batch_id), e.g. the "
                             "batch a refresh just loaded. Repeatable.")
    parser.add_argument("--classification", default="positive",
                        choices=["positive", "negative", "undeterminable", "any"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-enriched", action="store_true",
                        help="Also export documents that already carry an enrichment.")
    parser.add_argument("--max-usd", type=float, default=20.0,
                        help="Refuse to write an input whose projected cost exceeds this. "
                             "Pass 0 to disable the gate.")
    return parser


def default_name(journals: list[str] | None, batch_ids: list[str] | None) -> str:
    if journals:
        return slugify(journals[0]) if len(journals) == 1 else f"{len(journals)}journals"
    return f"batch_{slugify(batch_ids[0])}" if len(batch_ids) == 1 else f"{len(batch_ids)}batches"


def main() -> None:
    args = build_parser().parse_args()
    out = args.out or DEFAULT_OUT_DIR / f"enrich_input_{default_name(args.journals, args.batch_ids)}.csv"
    run(args.journals, None if args.classification == "any" else args.classification,
        out, args.limit, args.include_enriched, args.max_usd or None, args.batch_ids)


if __name__ == "__main__":
    main()
