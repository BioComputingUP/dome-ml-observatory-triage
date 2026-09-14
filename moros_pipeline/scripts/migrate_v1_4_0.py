"""Migrates every existing document from schema v1.2.0 (or v1.3.0) to v1.4.0, in place.

Both bumps are additive. v1.3.0 adds `identifiers.epmc_id`, `source.epmc_source` and
`publication_metadata.preprint_server` (docs/preprint.md); v1.4.0 adds the `data_links` group
(schema.py's changelog). The corpus never carried 1.3.0 -- it was published by dome-ml-observatory
before it was authored here -- so this one `updateMany` takes 1.2.0 straight to 1.4.0, and accepts
1.3.0 as a source purely so a partially-migrated state is never a state it refuses to reason about.
Re-importing the ~3GB JSONL to add thirteen fields would be absurd, and a drop-and-reload would
destroy `positives_text` for ~3 minutes while search silently fell back to the regex path.

Every new leaf is set to its "never looked up" value (null, or an empty array). The real data
arrives afterwards, per document, through `load_fields.py --mode preprints` and
`--mode data_links`, which is why those modes exist separately from this one.

**Rollback is a constant, not a snapshot**, exactly as in `migrate_v1_2_0.py`: the prior state is
known (`schema_version` was "1.2.0", the thirteen paths were absent), so `--reverse` applies the
documented inverse. It refuses to run once any real preprint or data-links data has landed, since
the `$unset` would then be data loss rather than a reversal.

    python3 migrate_v1_4_0.py                    # dry run: pre-flight checks, writes nothing
    python3 migrate_v1_4_0.py --confirm          # migrate
    python3 migrate_v1_4_0.py --reverse --confirm # undo, if nothing has been written since
"""

from __future__ import annotations

import argparse
import json

from moros_client import Moros
from moros_write import (
    DATA_LINKS_LINK_FIELDS,
    DATA_LINKS_SUMMARY_FIELDS,
    PREPRINT_FIELDS,
    REPORT_DIR,
    ROLLBACK_DIR,
    SafeWriter,
    new_run_id,
    utc_now_iso,
)

FROM_VERSIONS = ("1.2.0", "1.3.0")
TO_VERSION = "1.4.0"
# What the corpus actually held before this migration; 1.3.0 never reached moros.
REVERSE_VERSION = "1.2.0"

# The constant applied to every document. Keys are the leaf paths; SafeWriter.validate checks them
# against WRITE_MODES["migrate_v1_4_0"] so this and the allowlist cannot drift apart.
FORWARD_SET: dict = {
    "schema_version": TO_VERSION,
    **{path: None for path in PREPRINT_FIELDS},
    "data_links.has_data": None,
    "data_links.tags": [],
    "data_links.accession_types": [],
    "data_links.db_cross_references": [],
    "data_links.fetched_at": None,
    "data_links.sources": [],
    "data_links.link_count": None,
    "data_links.truncated": None,
    "data_links.resources": [],
    "data_links.links": [],
}
assert set(FORWARD_SET) == {"schema_version", *PREPRINT_FIELDS} | {
    f"data_links.{f}" for f in DATA_LINKS_SUMMARY_FIELDS + DATA_LINKS_LINK_FIELDS
}

# The documented inverse: the version the corpus held, and every new path unset.
REVERSE_SET = {"schema_version": REVERSE_VERSION}
REVERSE_UNSET = [path for path in FORWARD_SET if path != "schema_version"]

# Any of these being populated means a real backfill has landed and --reverse would destroy it.
POPULATED_MARKERS = (
    "identifiers.epmc_id",
    "publication_metadata.preprint_server",
    "data_links.has_data",
    "data_links.fetched_at",
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
    """`--reverse` unsets thirteen paths. If a backfill has already populated any of them, that is
    no longer a reversal -- it is data loss. Refuse."""
    populated = {marker: moros.count({marker: {"$ne": None}}) for marker in POPULATED_MARKERS}
    if any(populated.values()):
        raise SystemExit(
            f"refusing to reverse: real data has landed since the migration -- {populated}. "
            f"Unsetting those paths would destroy fetched data, not undo this migration. Roll the "
            f"preprints / data_links loads back first (moros_write.py --rollback)."
        )


def run(confirm: bool, reverse: bool) -> None:
    run_id = new_run_id("migrate_v1_4_0_reverse" if reverse else "migrate_v1_4_0")
    with Moros.from_env() as moros:
        # Reuse the mode allowlist rather than trusting the constants above -- one source of truth.
        writer = SafeWriter(moros, mode="migrate_v1_4_0", run_id=run_id, dry_run=not confirm)
        writer.validate(FORWARD_SET)
        writer.validate({**REVERSE_SET, **{p: None for p in REVERSE_UNSET}})

        if reverse:
            reverse_preflight(moros)
            query = {"schema_version": TO_VERSION}
            update = {"$set": REVERSE_SET, "$unset": {p: "" for p in REVERSE_UNSET}}
            n_target = moros.count(query)
            print(f"reverse: {n_target:,} documents at {TO_VERSION} -> {REVERSE_VERSION}")
        else:
            stats = preflight(moros)
            query = {"schema_version": {"$in": list(FROM_VERSIONS)}}
            update = {"$set": FORWARD_SET}
            n_target = stats["at_from"]
            print(f"forward: {n_target:,} documents at {'/'.join(FROM_VERSIONS)} -> {TO_VERSION}")
            print(f"         setting {json.dumps(FORWARD_SET)}")

        if n_target == 0:
            print("nothing to do -- already in the target state.")
            return

        if not confirm:
            sample = moros.find_one(query, {p: 1 for p in FORWARD_SET})
            print("\nDRY RUN -- nothing written. One document as it stands now:")
            print(f"  {json.dumps(sample, indent=2, default=str)}")
            print(f"\nre-run with --confirm to write. Inverse afterwards: "
                  f"python3 migrate_v1_4_0.py --reverse --confirm")
            return

        ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
        rollback_path = ROLLBACK_DIR / f"{run_id}.rollback.json"
        rollback_path.write_text(json.dumps({
            "run_id": run_id,
            "kind": "constant-inverse",
            "why": "this migration sets identical constants on every document, so the inverse is "
                   "a single updateMany rather than a per-document snapshot",
            "target": moros.describe(),
            "taken_at": utc_now_iso(),
            "documents": n_target,
            "reverse_command": "python3 migrate_v1_4_0.py --reverse --confirm",
            "reverse_filter": {"schema_version": TO_VERSION},
            "reverse_update": {"$set": REVERSE_SET, "$unset": {p: "" for p in REVERSE_UNSET}},
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
