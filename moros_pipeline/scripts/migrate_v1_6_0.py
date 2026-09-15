"""Migrates every existing document from schema v1.5.1 to v1.6.0, in place.

v1.6.0 adds one top-level field, `record_modified`: when the record last changed in a field the
observatory's metadata exposes, UTC to the second (`YYYY-MM-DDThh:mm:ssZ`). OAI-PMH harvests by it
(`from` / `until`) and the sitemap reports it as `lastmod`; the `record_modified_positive` index
(`ensure_indexes.py`) serves both.

**Every existing document gets the same stamp: the time this migration runs.** Deriving a
per-document value from the group timestamps was considered and not done. The restamp is itself a
change to every record, a harvester's first harvest is a full one whatever the dates say, and the
group timestamps (`llm_classification.timestamp`, `data_links.fetched_at`, ...) stay where they are
as provenance. From here on the field moves only when a write changes a harvested value
(`moros_write.STAMPS_RECORD_MODIFIED`, `load_documents.py`).

**Rollback is a constant**, as in `migrate_v1_5_1.py`: `--reverse` sets the version back to 1.5.1
and removes `record_modified`, which 1.5.1 does not have. Nothing else lives in that field, so
reversing is always safe; any later stamps go with it, by design.

    python3 migrate_v1_6_0.py                     # dry run: pre-flight checks, writes nothing
    python3 migrate_v1_6_0.py --confirm           # migrate
    python3 migrate_v1_6_0.py --reverse --confirm # undo
"""

from __future__ import annotations

import argparse
import json

from moros_client import Moros
from moros_write import (
    RECORD_MODIFIED_PATH,
    REPORT_DIR,
    ROLLBACK_DIR,
    SafeWriter,
    new_run_id,
    record_modified_stamp,
    utc_now_iso,
)

MODE = "migrate_v1_6_0"
FROM_VERSIONS = ("1.5.1",)
TO_VERSION = "1.6.0"
REVERSE_VERSION = "1.5.1"

REVERSE_SET = {"schema_version": REVERSE_VERSION}
REVERSE_UNSET = (RECORD_MODIFIED_PATH,)

# record_modified is v1.6.0's own field and nothing else depends on it, so --reverse is always safe.
POPULATED_MARKERS: tuple[dict, ...] = ()


def forward_set(stamp: str) -> dict:
    """The constant applied to every document still at a FROM version; SafeWriter.validate checks it
    against WRITE_MODES[MODE]."""
    return {"schema_version": TO_VERSION, RECORD_MODIFIED_PATH: stamp}


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
            f"does not know how to reason about. Investigate before writing."
        )

    at_from = sum(versions.get(v, 0) for v in FROM_VERSIONS)
    at_to = versions.get(TO_VERSION, 0)
    if at_to and at_from:
        print(f"NOTE: mixed state -- {at_to:,} already at {TO_VERSION}, {at_from:,} still at "
              f"{FROM_VERSIONS}. This migration is idempotent; it will finish the remainder and "
              f"leave the stamps already written alone.")
    return {"total": total, "at_from": at_from, "at_to": at_to, "versions": versions}


def reverse_preflight(moros: Moros) -> None:
    """`--reverse` stamps 1.5.1 back and removes record_modified; always safe (see module docstring)."""
    landed = {json.dumps(marker): moros.count(marker) for marker in POPULATED_MARKERS}
    if any(landed.values()):
        raise SystemExit(
            f"refusing to reverse: a populated marker matched -- {landed}."
        )


def run(confirm: bool, reverse: bool) -> None:
    run_id = new_run_id(f"{MODE}_reverse" if reverse else MODE)
    stamp = record_modified_stamp()
    with Moros.from_env() as moros:
        # Reuse the mode allowlist rather than trusting the constants above -- one source of truth.
        writer = SafeWriter(moros, mode=MODE, run_id=run_id, dry_run=not confirm)
        writer.validate(forward_set(stamp))
        writer.validate(REVERSE_SET)
        writer.validate({path: None for path in REVERSE_UNSET})

        if reverse:
            reverse_preflight(moros)
            query = {"schema_version": TO_VERSION}
            update = reverse_update()
            n_target = moros.count(query)
            print(f"reverse: {n_target:,} documents at {TO_VERSION} -> {REVERSE_VERSION}, "
                  f"{RECORD_MODIFIED_PATH} removed")
        else:
            stats = preflight(moros)
            query = {"schema_version": {"$in": list(FROM_VERSIONS)}}
            update = {"$set": forward_set(stamp)}
            n_target = stats["at_from"]
            print(f"forward: {n_target:,} documents at {'/'.join(FROM_VERSIONS)} -> {TO_VERSION}, "
                  f"{RECORD_MODIFIED_PATH} = {stamp}")

        if n_target == 0:
            print("nothing to do -- already in the target state.")
            return

        if not confirm:
            sample = moros.find_one(query, {"schema_version": 1, RECORD_MODIFIED_PATH: 1})
            print("\nDRY RUN -- nothing written. One document as it stands now:")
            print(f"  {json.dumps(sample, indent=2, default=str)}")
            print(f"  update: {json.dumps(update)}")
            print(f"\nre-run with --confirm to write. Inverse afterwards: "
                  f"python3 {MODE}.py --reverse --confirm")
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
            "record_modified": stamp if not reverse else None,
            "reverse_command": f"python3 {MODE}.py --reverse --confirm",
            "reverse_filter": {"schema_version": TO_VERSION},
            "reverse_update": reverse_update(),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"rollback spec -> {rollback_path}")

        result = moros.collection.update_many(query, update)
        print(f"matched {result.matched_count:,}, modified {result.modified_count:,}")

        after = moros.histogram("schema_version")
        print(f"\nschema_version now : {after}")
        unstamped = moros.count({"schema_version": TO_VERSION, RECORD_MODIFIED_PATH: None})

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report = REPORT_DIR / f"{run_id}.report.json"
        report.write_text(json.dumps({
            "run_id": run_id,
            "direction": "reverse" if reverse else "forward",
            "target": moros.describe(),
            "finished_at": utc_now_iso(),
            "matched": result.matched_count,
            "modified": result.modified_count,
            "record_modified": stamp if not reverse else None,
            "schema_version_after": {str(k): v for k, v in after.items()},
            "at_to_version_without_record_modified": unstamped,
            "rollback": str(rollback_path),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {report}")

        expected = TO_VERSION if not reverse else REVERSE_VERSION
        if list(after) != [expected]:
            raise SystemExit(
                f"POST-FLIGHT FAILED: expected every document at {expected}, got {after}"
            )
        if unstamped:
            raise SystemExit(
                f"POST-FLIGHT FAILED: {unstamped:,} documents at {TO_VERSION} have no {RECORD_MODIFIED_PATH}"
            )
        print("\npost-flight checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Actually write. Off by default.")
    parser.add_argument("--reverse", action="store_true", help="Apply the documented inverse.")
    args = parser.parse_args()
    run(args.confirm, args.reverse)


if __name__ == "__main__":
    main()
