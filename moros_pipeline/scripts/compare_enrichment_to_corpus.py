#!/usr/bin/env python3
"""Read-only: how far an enrichment events file agrees with the enrichment already stored in moros
for the same records.

Written for the 2026-09-15 model check. DeepSeek now answers `deepseek-v4-flash` with
DeepSeek-V4.1-Flash, so already-enriched positives were re-enriched into a separate events file and
compared, field by field, with the V4-Flash values in moros. **Never merge that events file**:
`load_enrichment.py` would overwrite the stored values it is being compared with.

    python3 compare_enrichment_to_corpus.py --events ../output/enrichment_revalidation_v41_events.csv
    python3 compare_enrichment_to_corpus.py --events <csv> --json --examples 5

Per field: exact agreement (the same set of terms), mean Jaccard overlap, and how often both sides
are empty; then the share of records agreeing on all six fields. `domain_tier1` is compared the way
`load_enrichment.py` stores it, as the first tier-1 term only. Terms are compared case-folded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from moros_client import ID_FIELD, Moros

FIELDS = ("domain_tier1", "domain_tier2", "domain_tier3", "learning_paradigm", "model_family", "model_type")
BATCH = 1_000


def json_list(cell) -> list:
    if isinstance(cell, list):
        return cell
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        value = json.loads(cell)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def latest_ok(rows: list[dict]) -> dict[str, dict]:
    """Last parsed event per record, the last-event-wins convention every log here uses."""
    out: dict[str, dict] = {}
    for row in sorted(rows, key=lambda r: r.get("timestamp") or ""):
        if (row.get("parse_status") or "").strip() == "ok":
            out[row["record_id"]] = row
    return out


def _norm(values) -> set[str]:
    return {str(v).strip().casefold() for v in values if v is not None and str(v).strip()}


def event_terms(row: dict, field: str) -> set[str]:
    values = json_list(row.get(field, ""))
    if field == "domain_tier1":
        values = values[:1]
    return _norm(values)


def stored_terms(doc: dict, field: str) -> set[str]:
    value = (doc.get("content_filters") or {}).get(field)
    if field == "domain_tier1":
        return _norm([value])
    return _norm(value or [])


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compare(pairs: list[tuple[dict, dict]]) -> dict:
    """pairs: (event row, stored document) for the same record."""
    n = len(pairs)
    result: dict = {"n": n, "fields": {}, "all_six": 0}
    if n == 0:
        return result
    all_six = [True] * n
    for field in FIELDS:
        exact = both_empty = 0
        jac = 0.0
        for i, (row, doc) in enumerate(pairs):
            a, b = event_terms(row, field), stored_terms(doc, field)
            if a == b:
                exact += 1
                if not a:
                    both_empty += 1
            else:
                all_six[i] = False
            jac += jaccard(a, b)
        result["fields"][field] = {"exact": exact / n, "mean_jaccard": jac / n, "both_empty": both_empty / n}
    result["all_six"] = sum(all_six) / n
    return result


def disagreements(pairs: list[tuple[dict, dict]], field: str, limit: int) -> list[dict]:
    out = []
    for row, doc in pairs:
        a, b = event_terms(row, field), stored_terms(doc, field)
        if a != b:
            out.append({"record_id": row["record_id"], "events": sorted(a), "stored": sorted(b)})
            if len(out) >= limit:
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--examples", type=int, default=0, help="Show N disagreements per field.")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    with args.events.open(newline="", encoding="utf-8") as f:
        events = latest_ok(list(csv.DictReader(f)))
    ids = list(events)

    projection = {ID_FIELD: 1, "llm_enrichment.provider": 1, "llm_enrichment.model_id": 1,
                  "llm_enrichment.batch_id": 1, **{f"content_filters.{k}": 1 for k in FIELDS}}
    docs: dict[str, dict] = {}
    with Moros.from_env() as moros:
        target = moros.describe()
        for i in range(0, len(ids), BATCH):
            for doc in moros.collection.find({ID_FIELD: {"$in": ids[i:i + BATCH]}}, projection):
                docs[doc[ID_FIELD]] = doc

    pairs = [(events[r], docs[r]) for r in ids
             if r in docs and (docs[r].get("llm_enrichment") or {}).get("provider")]
    result = compare(pairs)
    result.update({
        "events_file": str(args.events),
        "target": target,
        "records_in_events": len(ids),
        "not_in_moros": sum(1 for r in ids if r not in docs),
        "not_enriched_in_moros": sum(1 for r in ids if r in docs and not (docs[r].get("llm_enrichment") or {}).get("provider")),
    })
    if args.examples:
        result["examples"] = {field: disagreements(pairs, field, args.examples) for field in FIELDS}

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"target {target}")
    print(f"{result['records_in_events']:,} records with a parsed event; {result['n']:,} compared "
          f"({result['not_in_moros']} not in moros, {result['not_enriched_in_moros']} not enriched there)\n")
    print(f"{'field':<20}{'exact':>8}{'jaccard':>9}{'both empty':>12}")
    for field, s in result["fields"].items():
        print(f"{field:<20}{s['exact'] * 100:>7.1f}%{s['mean_jaccard']:>9.3f}{s['both_empty'] * 100:>11.1f}%")
    print(f"\nall six fields agree: {result['all_six'] * 100:.1f}% of records")
    for field, rows in (result.get("examples") or {}).items():
        for row in rows:
            print(f"  {field}  {row['record_id']}  events={row['events']}  stored={row['stored']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
