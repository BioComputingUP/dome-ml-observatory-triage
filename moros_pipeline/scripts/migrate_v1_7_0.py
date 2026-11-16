"""Migrates every existing document from schema v1.6.0 to v1.7.0, in place.

v1.7.0 adds one field, `content_filters.model_use`: how the paper obtains the models it reports
(`trained in this study`, `fine-tuned or adapted`, `pretrained, applied as-is`,
`prompted (zero/few-shot)`; at most 2). Only enrichment prompt e2 (GLM-5.3-Flash,
`curation_criteria/e2/PROVENANCE.md`) fills it. The release also adopts e2's modelling
vocabularies, a strict superset of e1's, so no stored value becomes invalid.

**Every existing document gets the same constant: `model_use: []`**, the value of a document no e2
enrichment has reached. e1-enriched documents keep `[]` because e1 never asked the question;
`llm_enrichment.prompt_version` says which prompt a document was enriched with. Not a
`record_modified` stamp: an empty new field changes nothing the observatory's metadata exposes.

**Rollback is a constant**, as in `migrate_v1_6_0.py`: `--reverse` sets the version back to 1.6.0
and removes `content_filters.model_use`. It refuses once an e2 enrichment has landed, because that
would discard real values and leave e2 vocabulary terms under a version whose vocabulary lacks
them. Roll the enrichment load back first (`moros_write.py --rollback`).

Order (schema/README.md): publish v1.7.0 in dome-ml-observatory and run check_alignment.py first;
then this migration; then `load_enrichment.py` for e2 events, which refuses until this has run.

    python3 migrate_v1_7_0.py                     # dry run: pre-flight checks, writes nothing
    python3 migrate_v1_7_0.py --confirm           # migrate
    python3 migrate_v1_7_0.py --reverse --confirm # undo (refused once e2 values have landed)
"""

from __future__ import annotations

import argparse
import json

from moros_client import Moros
from moros_write import REPORT_DIR, ROLLBACK_DIR, SafeWriter, new_run_id, utc_now_iso

MODE = "migrate_v1_7_0"
FROM_VERSIONS = ("1.6.0",)
TO_VERSION = "1.7.0"
REVERSE_VERSION = "1.6.0"

MODEL_USE_PATH = "content_filters.model_use"
FORWARD_SET = {"schema_version": TO_VERSION, MODEL_USE_PATH: []}
REVERSE_SET = {"schema_version": REVERSE_VERSION}
REVERSE_UNSET = (MODEL_USE_PATH,)

# Any of these matching a document means e2/e3 content has landed and --reverse would destroy it.
POPULATED_MARKERS: tuple[dict, ...] = (
    {"content_filters.model_use.0": {"$exists": True}},
    {"llm_enrichment.prompt_version": {"$in": ["e2", "e3"]}},
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
            f"does not know how to reason about. Investigate before writing."
        )

    at_from = sum(versions.get(v, 0) for v in FROM_VERSIONS)
    at_to = versions.get(TO_VERSION, 0)
    if at_to and at_from:
        print(f"NOTE: mixed state -- {at_to:,} already at {TO_VERSION}, {at_from:,} still at "
              f"{FROM_VERSIONS}. This migration is idempotent; it will finish the remainder.")
    return {"total": total, "at_from": at_from, "at_to": at_to, "versions": versions}


def reverse_preflight(moros: Moros) -> None:
    """`--reverse` stamps 1.6.0 back. Once e2 content has landed that would discard it."""
    landed = {json.dumps(marker): moros.count(marker) for marker in POPULATED_MARKERS}
    if any(landed.values()):
        raise SystemExit(
            f"refusing to reverse: e2 enrichment has landed since the migration -- {landed}. "
            f"Roll the enrichment load back first (moros_write.py --rollback)."
        )


def run(confirm: bool, reverse: bool) -> None:
    run_id = new_run_id(f"{MODE}_reverse" if reverse else MODE)
    with Moros.from_env() as moros:
        # Reuse the mode allowlist rather than trusting the constants above -- one source of truth.
        writer = SafeWriter(moros, mode=MODE, run_id=run_id, dry_run=not confirm)
        writer.validate(FORWARD_SET)
        writer.validate(REVERSE_SET)
        writer.validate({path: None for path in REVERSE_UNSET})

        if reverse:
            reverse_preflight(moros)
            query = {"schema_version": TO_VERSION}
            update = reverse_update()
            n_target = moros.count(query)
            print(f"reverse: {n_target:,} documents at {TO_VERSION} -> {REVERSE_VERSION}, {MODEL_USE_PATH} removed")
        else:
            stats = preflight(moros)
            query = {"schema_version": {"$in": list(FROM_VERSIONS)}}
            update = {"$set": FORWARD_SET}
            n_target = stats["at_from"]
            print(f"forward: {n_target:,} documents at {'/'.join(FROM_VERSIONS)} -> {TO_VERSION}, "
                  f"{MODEL_USE_PATH} = []")

        if n_target == 0:
            print("nothing to do -- already in the target state.")
            return

        if not confirm:
            sample = moros.find_one(query, {"schema_version": 1, MODEL_USE_PATH: 1})
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
        missing = 0 if reverse else moros.count({"schema_version": TO_VERSION, MODEL_USE_PATH: {"$exists": False}})

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
            "at_to_version_without_model_use": missing,
            "rollback": str(rollback_path),
        }, indent=2) + "\n", encoding="utf-8")
        print(f"report -> {report}")

        expected = TO_VERSION if not reverse else REVERSE_VERSION
        if list(after) != [expected]:
            raise SystemExit(f"POST-FLIGHT FAILED: expected every document at {expected}, got {after}")
        if missing:
            raise SystemExit(f"POST-FLIGHT FAILED: {missing:,} documents at {TO_VERSION} have no {MODEL_USE_PATH}")
        print("\npost-flight checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Actually write. Off by default.")
    parser.add_argument("--reverse", action="store_true", help="Apply the documented inverse.")
    args = parser.parse_args()
    run(args.confirm, args.reverse)


if __name__ == "__main__":
    main()
