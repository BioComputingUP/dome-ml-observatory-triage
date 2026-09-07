"""Idempotently ensures the corpus's indexes exist, and reports which already did.

Three indexes matter on `dome_observatory.Content`:

- `_id_` -- automatic, never absent.
- `class_year_id` -- `{classification: 1, year: -1, _id: 1}`. Cheap (~4s, ~37MB).
- `positives_text` -- the weighted text index over title/abstract/authors, **partial** on
  `llm_classification.classification: "positive"`. Expensive: ~3 minutes of tokenising plus a
  1.5GB collection read, and ~390MB plus ~1GB of transient sort files.

The reason this script exists rather than a note in a runbook: **when `positives_text` is missing,
nothing breaks.** `observatory-ws` detects its absence at boot and silently falls back to a regex
scan, so search still returns correct results, just slowly, with no error logged anywhere. "Search
works" is therefore not evidence the index is there. This checks, and says.

The partial filter is also why the index survives the v1.2.0 change: `decision_provenance` was
added as a separate discriminator precisely so `llm_classification.classification` stays the one
field the index and `canUseTextIndex` in `records.query.ts` key on. A curated positive is
`classification: "positive"` like any other, so it enters `positives_text` normally and becomes
searchable with no index change at all.

    python3 ensure_indexes.py                      # report only
    python3 ensure_indexes.py --confirm            # create anything missing
    python3 ensure_indexes.py --measure-citation-sort   # is a citation index warranted yet?
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pymongo import ASCENDING, DESCENDING, TEXT

from moros_client import Moros
from moros_write import REPORT_DIR, new_run_id, utc_now_iso

# Reproduced verbatim from ../dome-ml-observatory/ROADMAP.md lines 80-92, which is the
# authoritative definition. Do not "improve" these without changing that file first.
REQUIRED_INDEXES = {
    "positives_text": {
        "keys": [
            ("publication_metadata.title", TEXT),
            ("publication_metadata.abstract", TEXT),
            ("publication_metadata.authors", TEXT),
        ],
        "options": {
            "name": "positives_text",
            "partialFilterExpression": {"llm_classification.classification": "positive"},
            "weights": {
                "publication_metadata.title": 10,
                "publication_metadata.authors": 5,
                "publication_metadata.abstract": 1,
            },
            "background": True,
        },
        "cost": "~3 minutes, ~390MB, plus ~1GB transient sort files",
    },
    "class_year_id": {
        "keys": [
            ("llm_classification.classification", ASCENDING),
            ("publication_metadata.year", DESCENDING),
            ("_id", ASCENDING),
        ],
        "options": {"name": "class_year_id", "background": True},
        "cost": "~4 seconds, ~37MB",
    },
}

# Not created by default. `citations_desc`/`citations_asc` are existing API sort options that only
# started doing real work once citation_count was populated, and the field is unindexed -- but an
# index nobody's queries reach is pure write amplification, so this is measured before it is added.
CITATION_INDEX = {
    "keys": [
        ("llm_classification.classification", ASCENDING),
        ("publication_metadata.citation_count", DESCENDING),
        ("_id", ASCENDING),
    ],
    "options": {"name": "class_citations_id", "background": True},
}


def report(moros: Moros) -> dict[str, dict]:
    present = moros.indexes()
    print(f"target {moros.describe()}\n")
    print(f"{'index':<20}{'state':<12}{'note'}")
    status: dict[str, dict] = {}
    for name, spec in {**REQUIRED_INDEXES, "class_citations_id": CITATION_INDEX}.items():
        exists = name in present
        required = name in REQUIRED_INDEXES
        state = "present" if exists else ("MISSING" if required else "not created")
        note = spec.get("cost", "") if not exists and required else ""
        if name == "class_citations_id" and not exists:
            note = "optional -- run --measure-citation-sort to decide"
        print(f"{name:<20}{state:<12}{note}")
        status[name] = {"present": exists, "required": required}
    for name in present:
        if name not in status and name != "_id_":
            print(f"{name:<20}{'present':<12}not managed by this script")
    print(f"\n_id_                present     automatic")
    return status


def create_missing(moros: Moros, status: dict[str, dict], confirm: bool) -> list[str]:
    missing = [n for n, s in status.items() if s["required"] and not s["present"]]
    if not missing:
        print("\nnothing to create -- every required index is present.")
        return []
    print(f"\nmissing: {missing}")
    if not confirm:
        print("DRY RUN -- re-run with --confirm to create them.")
        return []
    created = []
    for name in missing:
        spec = REQUIRED_INDEXES[name]
        print(f"creating {name} ({spec['cost']}) ... background=True, the collection stays "
              f"readable throughout")
        started = time.time()
        moros.collection.create_index(spec["keys"], **spec["options"])
        print(f"  created in {time.time() - started:.1f}s")
        created.append(name)
    return created


def measure_citation_sort(moros: Moros) -> dict:
    """Real timings for the two queries the UI's "Most cited" option issues, so the decision to
    add an index is made on numbers rather than instinct."""
    print("\nmeasuring the citation sort as the API actually issues it ...")
    results = {}
    for label, skip in (("first page", 0), ("deep page (skip 9000)", 9000)):
        pipeline = [
            {"$match": {"llm_classification.classification": "positive"}},
            {"$sort": {"publication_metadata.citation_count": -1, "_id": 1}},
            {"$skip": skip},
            {"$limit": 20},
            {"$project": {"publication_metadata.citation_count": 1}},
        ]
        started = time.time()
        docs = list(moros.collection.aggregate(pipeline, allowDiskUse=True, maxTimeMS=120_000))
        elapsed = time.time() - started
        results[label] = round(elapsed, 2)
        top = docs[0]["publication_metadata"]["citation_count"] if docs else None
        print(f"  {label:<24} {elapsed:>6.2f}s   top count on the page: {top}")
    print("\n  Guidance: the API caps paging at page*pageSize > 10,000, so the deep page above is")
    print("  the worst case a user can reach. If it is comfortably inside the 20s search budget")
    print("  (MONGO_SEARCH_MAX_TIME_MS), the index is not yet worth ~400MB of write amplification.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Create any missing required index.")
    parser.add_argument("--measure-citation-sort", action="store_true")
    parser.add_argument("--create-citation-index", action="store_true",
                        help="Create class_citations_id. Only after measuring.")
    args = parser.parse_args()

    run_id = new_run_id("ensure_indexes")
    with Moros.from_env() as moros:
        status = report(moros)
        created = create_missing(moros, status, args.confirm)
        measurements = measure_citation_sort(moros) if args.measure_citation_sort else {}

        if args.create_citation_index:
            if not args.confirm:
                print("\n--create-citation-index needs --confirm too.")
            elif CITATION_INDEX["options"]["name"] in moros.indexes():
                print("\nclass_citations_id already exists.")
            else:
                print(f"\ncreating {CITATION_INDEX['options']['name']} ...")
                started = time.time()
                moros.collection.create_index(CITATION_INDEX["keys"], **CITATION_INDEX["options"])
                print(f"  created in {time.time() - started:.1f}s")
                created.append(CITATION_INDEX["options"]["name"])

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{run_id}.report.json"
        path.write_text(json.dumps({
            "run_id": run_id,
            "target": moros.describe(),
            "finished_at": utc_now_iso(),
            "status": status,
            "created": created,
            "citation_sort_measurements": measurements,
            "indexes_after": sorted(moros.indexes()),
        }, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
