"""Migrates every existing document from schema v1.1.0 to v1.2.0, in place.

v1.2.0 is additive: `source.decision_provenance`, `publication_metadata.citation_count_updated`
and `publication_metadata.citation_source`. Re-importing the 2.98GB JSONL to add three fields --
two of them null -- would be absurd, and a drop-and-reload would destroy `positives_text` for
~3 minutes while search silently fell back to the regex path. So this is one `updateMany`.

Every document currently in the collection was produced by Step 23a's LLM classification run --
verified, not assumed: every `llm_classification.batch_id` matches
`classify_flash_bulk_pool_excluding_curated_*`. That is why a single constant
`decision_provenance: "llm"` is correct for all of them. The human-curated and registry-confirmed
records arrive separately, already carrying their own provenance, through
`build_curated_documents.py` and `load_documents.py`.

**Rollback is a constant, not a snapshot.** Unlike every other write in this folder, this one sets
the same values on every document and the prior state is known exactly: `schema_version` was
"1.1.0" and the three new paths were absent. Recording 827,061 lines of "this field was absent"
would be a ~165MB file carrying no information the two lines below do not. So `--reverse` applies
the documented inverse instead, and refuses to run if any real citation data has landed in the
meantime (which would make the `$unset` destructive rather than reversing).

    python3 migrate_v1_2_0.py                    # dry run: pre-flight checks, writes nothing
    python3 migrate_v1_2_0.py --confirm          # migrate
    python3 migrate_v1_2_0.py --reverse --confirm # undo, if nothing has been written since
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from moros_client import Moros
from moros_write import (
    REPORT_DIR,
    ROLLBACK_DIR,
    SafeWriter,
    new_run_id,
    utc_now_iso,
)

FROM_VERSION = "1.1.0"
TO_VERSION = "1.2.0"

# The constant applied to every document. Keys are the leaf paths; SafeWriter.validate checks them
# against WRITE_MODES["migrate_v1_2_0"] so this and the allowlist cannot drift apart.
FORWARD_SET = {
    "schema_version": TO_VERSION,
    "source.decision_provenance": "llm",
    "publication_metadata.citation_count_updated": None,
    "publication_metadata.citation_source": None,
}

# The documented inverse. `citation_count` is deliberately NOT touched in either direction: it
# existed in v1.1.0 (null everywhere) and this migration does not populate it.
REVERSE_SET = {"schema_version": FROM_VERSION}
REVERSE_UNSET = [
    "source.decision_provenance",
    "publication_metadata.citation_count_updated",
    "publication_metadata.citation_source",
]

CITATION_PATHS = (
    "publication_metadata.citation_count",
    "publication_metadata.citation_count_updated",
    "publication_metadata.citation_source",
)


def preflight(moros: Moros) -> dict:
    """Refuses to guess about a collection that is not in the state this migration expects."""
    versions = moros.histogram("schema_version")
    provenance = moros.histogram("source.decision_provenance")
    total = sum(versions.values())

    print(f"target: {moros.describe()}")
    print(f"schema_version histogram : {versions}")
    print(f"decision_provenance      : {provenance}")

    unexpected = {v: n for v, n in versions.items() if v not in (FROM_VERSION, TO_VERSION)}
    if unexpected:
        raise SystemExit(
            f"refusing to migrate: {unexpected} documents are at a schema version this migration "
            f"does not know how to reason about. Investigate before writing."
        )

    at_from = versions.get(FROM_VERSION, 0)
    at_to = versions.get(TO_VERSION, 0)
    if at_to and at_from:
        print(f"NOTE: mixed state -- {at_to:,} already at {TO_VERSION}, {at_from:,} still at "
              f"{FROM_VERSION}. This migration is idempotent; it will finish the remainder.")
    return {"total": total, "at_from": at_from, "at_to": at_to, "versions": versions,
            "provenance": provenance}


def reverse_preflight(moros: Moros) -> None:
    """`--reverse` unsets two citation fields. If a citation backfill has already populated them,
    that is no longer a reversal -- it is data loss. Refuse."""
    populated = moros.count({"publication_metadata.citation_source": {"$ne": None}})
    counted = moros.count({"publication_metadata.citation_count": {"$ne": None}})
    if populated or counted:
        raise SystemExit(
            f"refusing to reverse: {populated:,} documents carry a citation_source and "
            f"{counted:,} carry a citation_count. Unsetting those fields would destroy real "
            f"fetched data, not undo this migration. Reverse the citation load first."
        )


def run(confirm: bool, reverse: bool) -> None:
    run_id = new_run_id("migrate_v1_2_0_reverse" if reverse else "migrate_v1_2_0")
    with Moros.from_env() as moros:
        # Reuse the mode allowlist rather than trusting the constants above -- one source of truth.
        writer = SafeWriter(moros, mode="migrate_v1_2_0", run_id=run_id, dry_run=not confirm)
        writer.validate(FORWARD_SET)
        writer.validate({**REVERSE_SET, **{p: None for p in REVERSE_UNSET}})

        if reverse:
            reverse_preflight(moros)
            query = {"schema_version": TO_VERSION}
            update = {"$set": REVERSE_SET, "$unset": {p: "" for p in REVERSE_UNSET}}
            n_target = moros.count(query)
            print(f"reverse: {n_target:,} documents at {TO_VERSION} -> {FROM_VERSION}")
        else:
            stats = preflight(moros)
            query = {"schema_version": FROM_VERSION}
            update = {"$set": FORWARD_SET}
            n_target = stats["at_from"]
            print(f"forward: {n_target:,} documents at {FROM_VERSION} -> {TO_VERSION}")
            print(f"         setting {json.dumps(FORWARD_SET)}")

        if n_target == 0:
            print("nothing to do -- already in the target state.")
            return

        if not confirm:
            sample = moros.find_one(query, {p: 1 for p in FORWARD_SET})
            print("\nDRY RUN -- nothing written. One document as it stands now:")
            print(f"  {json.dumps(sample, indent=2, default=str)}")
            print(f"\nre-run with --confirm to write. Inverse afterwards: "
                  f"python3 migrate_v1_2_0.py --reverse --confirm")
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
            "reverse_command": "python3 migrate_v1_2_0.py --reverse --confirm",
            "reverse_filter": {"schema_version": TO_VERSION},
            "reverse_update": {"$set": REVERSE_SET, "$unset": {p: "" for p in REVERSE_UNSET}},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"rollback spec -> {rollback_path}")

        result = moros.collection.update_many(query, update)
        print(f"matched {result.matched_count:,}, modified {result.modified_count:,}")

        after = moros.histogram("schema_version")
        prov = moros.histogram("source.decision_provenance")
        print(f"\nschema_version now : {after}")
        print(f"decision_provenance: {prov}")

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
            "decision_provenance_after": {str(k): v for k, v in prov.items()},
            "rollback": str(rollback_path),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {report}")

        expected = TO_VERSION if not reverse else FROM_VERSION
        if list(after) != [expected]:
            raise SystemExit(
                f"POST-FLIGHT FAILED: expected every document at {expected}, got {after}"
            )
        if not reverse and list(prov) != ["llm"]:
            raise SystemExit(
                f"POST-FLIGHT FAILED: expected decision_provenance 'llm' on every document, "
                f"got {prov}"
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
