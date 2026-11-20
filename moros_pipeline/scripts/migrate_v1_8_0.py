"""Migrates every existing document from schema v1.7.0 to v1.8.0, in place.

v1.8.0 adds one top-level field, `enrichment_history`: the enrichments a later load superseded,
oldest first. `load_enrichment.py` appends a document's current `llm_enrichment` group and its
`content_filters` enrichment values (plus `superseded_at` / `superseded_by`) before it sets a new
enrichment from a different prompt or batch, so re-enriching a record (e1 -> e3, or e3 -> a later
prompt) never loses the earlier one. The observatory reads only the current enrichment.

**Every existing document gets the same constant: `enrichment_history: []`** -- nothing has been
superseded yet. Not a `record_modified` stamp: an empty new field changes nothing the observatory's
metadata exposes.

**Rollback is a constant**, as in `migrate_v1_7_0.py`: `--reverse` sets the version back to 1.7.0
and removes `enrichment_history`. It refuses once any history entry exists, because that would
discard a superseded enrichment. Roll the enrichment load back first (`moros_write.py --rollback`).

Order (schema/README.md): publish v1.8.0 in dome-ml-observatory and run check_alignment.py first;
then `migrate_v1_7_0.py` if moros is still at 1.6.0; then this; then `load_enrichment.py`, which
refuses until this has run.

    python3 migrate_v1_8_0.py                     # dry run: pre-flight checks, writes nothing
    python3 migrate_v1_8_0.py --confirm           # migrate
    python3 migrate_v1_8_0.py --reverse --confirm # undo (refused once history has been written)
"""

from __future__ import annotations

import argparse
import json

from moros_client import Moros
from moros_write import ENRICHMENT_HISTORY_PATH, REPORT_DIR, ROLLBACK_DIR, SafeWriter, new_run_id, utc_now_iso

MODE = "migrate_v1_8_0"
FROM_VERSIONS = ("1.7.0",)
TO_VERSION = "1.8.0"
REVERSE_VERSION = "1.7.0"

FORWARD_SET = {"schema_version": TO_VERSION, ENRICHMENT_HISTORY_PATH: []}
REVERSE_SET = {"schema_version": REVERSE_VERSION}
REVERSE_UNSET = (ENRICHMENT_HISTORY_PATH,)

# Any of these matching a document means a superseded enrichment is stored and --reverse would lose it.
POPULATED_MARKERS: tuple[dict, ...] = (
    {f"{ENRICHMENT_HISTORY_PATH}.0": {"$exists": True}},
)


def reverse_update() -> dict:
    return {"$set": REVERSE_SET, "$unset": {path: "" for path in REVERSE_UNSET}}


def preflight(moros: Moros) -> dict:
    """Refuses to guess about a collection that is not in the state this migration expects."""
    versions = moros.histogram("schema_version")
    total = sum(versions.values())

    print(f"target: {moros.describe()}")
    print(f"schema_version histogram : {versions}")

    unexpected = {v: n for v, n in versions.items() if v not in (*FROM_VERSIONS, TO_VERSION)}
    if unexpected:
        raise SystemExit(
            f"refusing to migrate: {unexpected} documents are at a schema version this migration "
            f"does not know how to reason about (run migrate_v1_7_0.py first if they are 1.6.0). "
            f"Investigate before writing."
        )

    at_from = sum(versions.get(v, 0) for v in FROM_VERSIONS)
    at_to = versions.get(TO_VERSION, 0)
    if at_to and at_from:
        print(f"NOTE: mixed state -- {at_to:,} already at {TO_VERSION}, {at_from:,} still at "
              f"{FROM_VERSIONS}. This migration is idempotent; it will finish the remainder.")
    return {"total": total, "at_from": at_from, "at_to": at_to, "versions": versions}


def reverse_preflight(moros: Moros) -> None:
    landed = {json.dumps(marker): moros.count(marker) for marker in POPULATED_MARKERS}
    if any(landed.values()):
        raise SystemExit(
            f"refusing to reverse: superseded enrichments are stored in {ENRICHMENT_HISTORY_PATH} -- "
            f"{landed}. Roll the enrichment load back first (moros_write.py --rollback)."
        )


def run(confirm: bool, reverse: bool) -> None:
    run_id = new_run_id(f"{MODE}_reverse" if reverse else MODE)
    with Moros.from_env() as moros:
        writer = SafeWriter(moros, mode=MODE, run_id=run_id, dry_run=not confirm)
        writer.validate(FORWARD_SET)
        writer.validate(REVERSE_SET)
        writer.validate({path: None for path in REVERSE_UNSET})

        if reverse:
            reverse_preflight(moros)
            query = {"schema_version": TO_VERSION}
            update = reverse_update()
            n_target = moros.count(query)
            print(f"reverse: {n_target:,} documents at {TO_VERSION} -> {REVERSE_VERSION}, "
                  f"{ENRICHMENT_HISTORY_PATH} removed")
        else:
            stats = preflight(moros)
            query = {"schema_version": {"$in": list(FROM_VERSIONS)}}
            update = {"$set": FORWARD_SET}
            n_target = stats["at_from"]
            print(f"forward: {n_target:,} documents at {'/'.join(FROM_VERSIONS)} -> {TO_VERSION}, "
                  f"{ENRICHMENT_HISTORY_PATH} = []")

        if n_target == 0:
            print("nothing to do -- already in the target state.")
            return

        if not confirm:
            sample = moros.find_one(query, {"schema_version": 1, ENRICHMENT_HISTORY_PATH: 1})
            print("\nDRY RUN -- nothing written. One document as it stands now:")
            print(f"  {json.dumps(sample, indent=2, default=str)}")
            print(f"  update: {json.dumps(update)}")
            print(f"\nre-run with --confirm to write. Inverse afterwards: python3 {MODE}.py --reverse --confirm")
            return

        ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
        rollback_path = ROLLBACK_DIR / f"{run_id}.rollback.json"
        rollback_path.write_text(json.dumps({
            "run_id": run_id,
            "kind": "constant-inverse",
            "why": "this migration sets one identical constant on every document, so the inverse is "
                   "a single updateMany rather than a per-document snapshot",
            "target": moros.describe(),
            "taken_at": utc_now_iso(),
            "documents": n_target,
            "reverse_command": f"python3 {MODE}.py --reverse --confirm",
            "reverse_filter": {"schema_version": TO_VERSION},
            "reverse_update": reverse_update(),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"rollback spec -> {rollback_path}")

        result = moros.collection.update_many(query, update)
        print(f"matched {result.matched_count:,}, modified {result.modified_count:,}")

        after = moros.histogram("schema_version")
        print(f"\nschema_version now : {after}")
        missing = 0 if reverse else moros.count(
            {"schema_version": TO_VERSION, ENRICHMENT_HISTORY_PATH: {"$exists": False}})

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORT_DIR / f"{run_id}.report.json"
        report.write_text(json.dumps({
            "run_id": run_id,
            "direction": "reverse" if reverse else "forward",
            "target": moros.describe(),
            "finished_at": utc_now_iso(),
            "matched": result.matched_count,
            "modified": result.modified_count,
            "schema_version_after": {str(k): v for k, v in after.items()},
            "at_to_version_without_enrichment_history": missing,
            "rollback": str(rollback_path),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {report}")

        expected = TO_VERSION if not reverse else REVERSE_VERSION
        if list(after) != [expected]:
            raise SystemExit(f"POST-FLIGHT FAILED: expected every document at {expected}, got {after}")
        if missing:
            raise SystemExit(f"POST-FLIGHT FAILED: {missing:,} documents at {TO_VERSION} have no "
                             f"{ENRICHMENT_HISTORY_PATH}")
        print("\npost-flight checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Actually write. Off by default.")
    parser.add_argument("--reverse", action="store_true", help="Apply the documented inverse.")
    args = parser.parse_args()
    run(args.confirm, args.reverse)


if __name__ == "__main__":
    main()
