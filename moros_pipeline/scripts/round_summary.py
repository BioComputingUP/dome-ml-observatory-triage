#!/usr/bin/env python3
"""Read-only: what a processing round did, read off moros itself.

For the classification batches named: documents by verdict, the first and last classification
timestamp, and the model ids and prompt versions they carry. For the enrichment batches named:
documents, the first and last enrichment timestamp, model ids and prompt versions. The
`processing-log` skill copies these figures onto dome-ml-observatory's /about/processing page, so
they come from the database, never from a log line or a typed estimate.

    python3 round_summary.py --classification-batch classify_flash_staged_file_primary_20260903T201216
    python3 round_summary.py --classification-prefix classify_flash_bulk_pool_excluding_curated --json
    python3 round_summary.py --enrichment-prefix enrich_flash_20260903 --json

A prefix matches every batch id that starts with it (a round that ran as several batches).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from moros_client import Moros

VERDICTS = ("positive", "negative", "undeterminable")
MAX_TIME_MS = 540_000


def batch_filter(field: str, ids: list[str], prefixes: list[str]) -> dict:
    """One `$or` over exact ids and `^prefix` regexes on a batch-id field."""
    clauses = []
    if ids:
        clauses.append({field: {"$in": ids}})
    for prefix in prefixes:
        clauses.append({field: {"$regex": f"^{re.escape(prefix)}"}})
    if not clauses:
        raise ValueError("name at least one batch id or prefix")
    return clauses[0] if len(clauses) == 1 else {"$or": clauses}


def pipeline(group: str, ids: list[str], prefixes: list[str]) -> list[dict]:
    return [
        {"$match": batch_filter(f"{group}.batch_id", ids, prefixes)},
        {"$group": {
            "_id": {"batch": f"${group}.batch_id",
                    "verdict": "$llm_classification.classification",
                    "model_id": f"${group}.model_id",
                    "prompt_version": f"${group}.prompt_version"},
            "n": {"$sum": 1},
            "first": {"$min": f"${group}.timestamp"},
            "last": {"$max": f"${group}.timestamp"},
        }},
    ]


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="seconds")
    return str(value)


def summarise(rows: list[dict]) -> dict:
    """Shapes grouped aggregate rows ({_id: {batch, verdict, model_id, prompt_version}, n, first,
    last}) into one round: totals by verdict, the time span, and what produced it."""
    out = {"batch_ids": set(), "total": 0, "first": None, "last": None,
           "model_ids": set(), "prompt_versions": set(), **{v: 0 for v in VERDICTS}}
    for row in rows:
        key = row["_id"]
        out["batch_ids"].add(key["batch"])
        out["total"] += row["n"]
        verdict = key.get("verdict")
        if verdict in VERDICTS:
            out[verdict] += row["n"]
        out["model_ids"].add(key.get("model_id"))
        out["prompt_versions"].add(key.get("prompt_version"))
        first, last = _iso(row.get("first")), _iso(row.get("last"))
        if first and (out["first"] is None or first < out["first"]):
            out["first"] = first
        if last and (out["last"] is None or last > out["last"]):
            out["last"] = last
    for key in ("batch_ids", "model_ids", "prompt_versions"):
        out[key] = sorted(out[key], key=lambda v: (v is None, v or ""))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classification-batch", action="append", default=[])
    parser.add_argument("--classification-prefix", action="append", default=[])
    parser.add_argument("--enrichment-batch", action="append", default=[])
    parser.add_argument("--enrichment-prefix", action="append", default=[])
    parser.add_argument("--json", action="store_true", help="Print one JSON object, nothing else.")
    args = parser.parse_args()
    wanted = {
        "classification": (args.classification_batch, args.classification_prefix),
        "enrichment": (args.enrichment_batch, args.enrichment_prefix),
    }
    if not any(ids or prefixes for ids, prefixes in wanted.values()):
        parser.error("name at least one --classification-* or --enrichment-* batch")

    result: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with Moros.from_env() as moros:
        result["target"] = moros.describe()
        result["corpus_total"] = moros.collection.estimated_document_count()
        for group, (ids, prefixes) in wanted.items():
            if not (ids or prefixes):
                continue
            rows = list(moros.collection.aggregate(
                pipeline(f"llm_{group}", ids, prefixes), allowDiskUse=True, maxTimeMS=MAX_TIME_MS))
            result[group] = summarise(rows)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"target {result['target']}  (corpus {result['corpus_total']:,})")
    for group in ("classification", "enrichment"):
        if group not in result:
            continue
        s = result[group]
        print(f"\n{group}: {s['total']:,} documents in {len(s['batch_ids'])} batch(es), "
              f"{s['first']} -> {s['last']}")
        if group == "classification":
            print("  " + " / ".join(f"{v} {s[v]:,}" for v in VERDICTS))
        print(f"  model_id {s['model_ids']}  prompt_version {s['prompt_versions']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
