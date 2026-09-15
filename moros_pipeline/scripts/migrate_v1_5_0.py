"""Migrates every existing document from schema v1.4.0 to v1.5.0, in place.

v1.5.0 adds no leaf path. Its change lives inside the `data_links.resources[]` / `links[]` arrays
(`routes`, `browse_url`, `matched_by`, `source_domain`) and in values (`data_links.sources` may
carry "ebisearch"; `identifiers.dome_registry` is filled). Those arrive per document through
`load_fields.py --mode data_links` and `--mode identifiers`, which never write `schema_version`, so
this one `updateMany` moves the version stamp and nothing else. Run it BEFORE those loads:
`verify_corpus.py` fails on a document carrying v1.5.0 link keys under an older version.

**Rollback is a constant**, as in `migrate_v1_4_0.py`: the prior version was "1.4.0" and nothing
else changed, so `--reverse` sets it back. It refuses once EBI Search links or a DOME Registry
identifier have landed, since a 1.4.0 stamp on documents carrying 1.5.0 content would be false.
Roll those loads back first (`moros_write.py --rollback`).

    python3 migrate_v1_5_0.py                     # dry run: pre-flight checks, writes nothing
    python3 migrate_v1_5_0.py --confirm           # migrate
    python3 migrate_v1_5_0.py --reverse --confirm # undo, if nothing has been loaded since
"""

from __future__ import annotations

import argparse
import json

from moros_client import Moros
from moros_write import REPORT_DIR, ROLLBACK_DIR, SafeWriter, new_run_id, utc_now_iso

MODE = "migrate_v1_5_0"
FROM_VERSIONS = ("1.4.0",)
TO_VERSION = "1.5.0"
REVERSE_VERSION = "1.4.0"

# The constant applied to every document; SafeWriter.validate checks it against WRITE_MODES[MODE].
FORWARD_SET = {"schema_version": TO_VERSION}
REVERSE_SET = {"schema_version": REVERSE_VERSION}

# Any of these matching a document means v1.5.0 content has landed and --reverse would mislabel it.
POPULATED_MARKERS = (
    {"data_links.sources": "ebisearch"},
    {"identifiers.dome_registry": {"$ne": None}},
    {"data_links.links.matched_by": {"$exists": True}},
)


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
              f"{FROM_VERSIONS}. This migration is idempotent; it will finish the remainder.")
    return {"total": total, "at_from": at_from, "at_to": at_to, "versions": versions}


def reverse_preflight(moros: Moros) -> None:
    """`--reverse` stamps 1.4.0 back. Once v1.5.0 content has landed that stamp would be false."""
    landed = {json.dumps(marker): moros.count(marker) for marker in POPULATED_MARKERS}
    if any(landed.values()):
        raise SystemExit(
            f"refusing to reverse: v1.5.0 content has landed since the migration -- {landed}. "
            f"Roll the data_links / identifiers loads back first (moros_write.py --rollback)."
        )


def run(confirm: bool, reverse: bool) -> None:
    run_id = new_run_id(f"{MODE}_reverse" if reverse else MODE)
    with Moros.from_env() as moros:
        # Reuse the mode allowlist rather than trusting the constants above -- one source of truth.
        writer = SafeWriter(moros, mode=MODE, run_id=run_id, dry_run=not confirm)
        writer.validate(FORWARD_SET)
        writer.validate(REVERSE_SET)

        if reverse:
            reverse_preflight(moros)
            query = {"schema_version": TO_VERSION}
            update = {"$set": REVERSE_SET}
            n_target = moros.count(query)
            print(f"reverse: {n_target:,} documents at {TO_VERSION} -> {REVERSE_VERSION}")
        else:
            stats = preflight(moros)
            query = {"schema_version": {"$in": list(FROM_VERSIONS)}}
            update = {"$set": FORWARD_SET}
            n_target = stats["at_from"]
            print(f"forward: {n_target:,} documents at {'/'.join(FROM_VERSIONS)} -> {TO_VERSION}")

        if n_target == 0:
            print("nothing to do -- already in the target state.")
            return

        if not confirm:
            sample = moros.find_one(query, {"schema_version": 1})
            print("\nDRY RUN -- nothing written. One document as it stands now:")
            print(f"  {json.dumps(sample, indent=2, default=str)}")
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
            "reverse_command": f"python3 {MODE}.py --reverse --confirm",
            "reverse_filter": {"schema_version": TO_VERSION},
            "reverse_update": {"$set": REVERSE_SET},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"rollback spec -> {rollback_path}")

        result = moros.collection.update_many(query, update)
        print(f"matched {result.matched_count:,}, modified {result.modified_count:,}")

        after = moros.histogram("schema_version")
        print(f"\nschema_version now : {after}")

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
            "rollback": str(rollback_path),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {report}")

        expected = TO_VERSION if not reverse else REVERSE_VERSION
        if list(after) != [expected]:
            raise SystemExit(
                f"POST-FLIGHT FAILED: expected every document at {expected}, got {after}"
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
