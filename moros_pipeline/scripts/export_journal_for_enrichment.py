"""Exports one journal's positive records out of moros as an enrichment-ready input CSV.

This is the reusable half of "enrich a journal": `--journal` is an argument, so the next journal
is one command rather than a new script. It reads from **moros, not the staging CSV**, which is
what makes it work on the whole corpus including the curated records that were only just merged --
a CSV-based export would miss exactly the records this project cares most about.

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

# Measured on THIS population, not the Step 20j trial: 100 paired Bioinformatics records at the
# production configuration (thinking on, provider-default effort, flat domain rendering) cost
# **$4.07 per 1,000 records** -- 5,947 output tokens per record, 98% of them reasoning.
#
# The old constant here was $1.76/1k, taken from the Step 20j trial population, and it
# under-projected this journal by 2.3x. Enrichment cost is strongly population-dependent: these
# records think about three times as long as the trial's (median 5,627 output tokens vs 2,163).
# Treat this number as a Bioinformatics-calibrated estimate, not a universal rate, and re-measure
# from a real run's own token totals when projecting a different journal.
#
# Attempts to reduce it, all measured and all rejected (thinking_ablation/run_effort_ablation.py,
# 2026-09-03): `reasoning_effort: low` cost MORE than the default ($4.15/1k) with 6x the
# violations; a hierarchical domain rendering saved nothing ($4.09/1k); disabling thinking is 88%
# cheaper but agrees with the production configuration on all six fields for 0% of records.
USD_PER_1000_RECORDS = 4.07


def slugify(value: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in value]
    return "".join(keep).strip("_").replace("__", "_")[:60]


def build_query(journals: list[str], classification: str | None,
                include_enriched: bool) -> dict:
    """`$in` over the journal names, so several journals are one export and one events file
    rather than one run to babysit per journal. Names are matched VERBATIM and are fuller than
    expected -- Science is stored as "Science (New York, N.Y.)"."""
    query: dict = {"publication_metadata.journal": {"$in": journals}}
    if classification:
        query["llm_classification.classification"] = classification
    if not include_enriched:
        # Already-enriched documents are skipped so a re-run after a partial merge does not pay
        # to redo them. `enrich` also resumes from its own event log; this is the cheaper filter.
        query["llm_enrichment.provider"] = None
    return query


def run(journals: list[str], classification: str | None, out_path: Path, limit: int | None,
        include_enriched: bool, max_usd: float | None) -> None:
    query = build_query(journals, classification, include_enriched)
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
        print(f"{'journal':<36}{'documents':>11}{'to enrich':>11}")
        missing = []
        for name in journals:
            total = moros.count({"publication_metadata.journal": name})
            if total == 0:
                missing.append(name)
            per_journal = moros.count(build_query([name], classification, include_enriched))
            print(f"{name:<36}{total:>11,}{per_journal:>11,}")
        if missing:
            raise SystemExit(
                f"\nno documents for {missing}. Journal names are matched verbatim and are often "
                f"fuller than expected -- Bioinformatics is stored as 'Bioinformatics (Oxford, "
                f"England)' and Science as 'Science (New York, N.Y.)'. Check /api/facets/journal "
                f"for the exact string."
            )
        n_match = moros.count(query)
        print(f"{'TOTAL':<36}{'':>11}{n_match:>11,}\n")

        # The budget gate. `enrich` deliberately has no --estimated-usd/--confirm gate (Gavin's
        # 2026-08-27 call), so this is the one place a runaway is stopped -- before the input file
        # exists, rather than after the money is spent.
        projected = n_match / 1000 * USD_PER_1000_RECORDS
        if max_usd is not None and projected > max_usd:
            raise SystemExit(
                f"REFUSING to write: {n_match:,} records project to ${projected:.2f}, over the "
                f"${max_usd:.2f} limit. Narrow the journal list, or raise --max-usd deliberately."
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
    if skipped:
        print(f"  {skipped:,} skipped for having no title or abstract to enrich from")
    print(f"\nprojected cost: {len(rows):,} records x ${USD_PER_1000_RECORDS}/1k = "
          f"**${cost:.2f}**"
          + (f" (limit ${max_usd:.2f})" if max_usd is not None else ""))
    print("  Rate measured on this exact configuration; ~3% of records are expected to hit the "
          "16,000-token cap and be skipped (recorded as finish_reason=length).")
    events = out_path.with_name(out_path.stem.replace("enrich_input_", "enrichment_") + "_events.csv")
    print(f"""
next -- smoke check 25 first, inspect, then run the rest:

  docker compose run --rm pipeline dome-triage llm-classify enrich \\
      --tier flash --concurrency 20 --limit 25 \\
      --input /app/{out_path.relative_to(FOLDER_DIR.parent)} \\
      --events-out /app/{events.relative_to(FOLDER_DIR.parent)}

then the same command without --limit, then:

  python3 load_enrichment.py --events {events}""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", required=True, action="append", dest="journals",
                        help="Exact journal string, matched verbatim. Repeatable.")
    parser.add_argument("--classification", default="positive",
                        choices=["positive", "negative", "undeterminable", "any"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-enriched", action="store_true",
                        help="Also export documents that already carry an enrichment.")
    parser.add_argument("--max-usd", type=float, default=20.0,
                        help="Refuse to write an input whose projected cost exceeds this. "
                             "Pass 0 to disable the gate.")
    args = parser.parse_args()
    default_name = (slugify(args.journals[0]) if len(args.journals) == 1
                    else f"{len(args.journals)}journals")
    out = args.out or DEFAULT_OUT_DIR / f"enrich_input_{default_name}.csv"
    run(args.journals, None if args.classification == "any" else args.classification,
        out, args.limit, args.include_enriched, args.max_usd or None)


if __name__ == "__main__":
    main()
