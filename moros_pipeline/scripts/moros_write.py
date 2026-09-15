"""The safe writer: every write to moros in this folder goes through here.

This exists because the curated merge, the citation backfill and the enrichment merge all write
fields that no code path has ever written, to a production collection with no authentication, no
replica set and therefore no transactions. The sibling repo's roadmap states the hazard plainly:
"a classification refresh must not blank an `llm_enrichment` group a later enrichment run has
populated. `$set` of a whole document will decide that by accident if nobody decides it on
purpose." This module is that decision, made mechanical.

Four properties, all enforced in code rather than documented and hoped for:

1. **Dry run is the default.** `dry_run=False` has to be asked for, and every CLI that uses this
   requires an explicit `--confirm`.
2. **An allowlist of leaf field paths per mode.** A path outside the mode's set raises before a
   single write is issued. Every allowlist entry is a *leaf*: `$set` on a group path such as
   `source` would replace the entire subdocument and silently drop its siblings, so group paths
   are refused outright.
3. **A rollback snapshot before the first batch.** The current values of exactly the paths about
   to change, for exactly the `_id`s about to change, streamed to `output/rollback/<run_id>.jsonl`
   *before* anything is written. Replay it with `python3 moros_write.py --rollback <file>`.
   Absent-vs-null is recorded separately, so a rollback restores "this field did not exist" rather
   than turning it into an explicit null.
4. **Never a delete, a drop, or a whole-document replace.** Only `$set` of allowlisted leaves. In
   particular the collection is never dropped: that would destroy `positives_text` (~3 minutes of
   tokenising plus a 1.5GB read to rebuild) and search would silently degrade to the regex path
   with no error anywhere while it was gone.

Usage as a library:
    with Moros.from_env() as moros:
        writer = SafeWriter(moros, mode="citations", dry_run=False)
        result = writer.apply(updates, total=n)

Usage as a CLI (rollback only -- there is no way to write from this file's command line):
    python3 moros_write.py --rollback ../output/rollback/<run_id>.jsonl --confirm
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from pymongo import UpdateOne
from tqdm import tqdm

from moros_client import ID_FIELD, Moros

THIS_DIR = Path(__file__).resolve().parent
FOLDER_DIR = THIS_DIR.parent
ROLLBACK_DIR = FOLDER_DIR / "output" / "rollback"
REPORT_DIR = FOLDER_DIR / "output"

BATCH_SIZE = 1_000

# Written by every mode: a document whose shape changed must say so, or a later migration cannot
# find it. Included in each allowlist below rather than special-cased.
_SCHEMA_VERSION_PATH = "schema_version"

_ENRICHMENT_FIELDS = (
    "provider", "model_tier", "model_id", "mode", "rationale", "prompt_version",
    "ruleset_sha256", "batch_id", "timestamp", "vocab_violations", "parse_status",
    "input_tokens", "output_tokens", "cache_hit_tokens", "parse_fallback_used",
)
_ENRICHED_FILTER_FIELDS = (
    "domain_tier1", "domain_tier2", "domain_tier3", "learning_paradigm", "model_family",
    "model_type",
)
# v1.3.0: the Europe PMC identity of the record and, for a preprint, its server.
PREPRINT_FIELDS = (
    "identifiers.epmc_id",
    "source.epmc_source",
    "publication_metadata.preprint_server",
)
# v1.4.0: the data_links group. The four summary leaves come from the search record; the six link
# leaves (DATA_LINKS_LINK_FIELDS, mirrored in schema.py) from the link fetch, written together.
DATA_LINKS_SUMMARY_FIELDS = ("has_data", "tags", "accession_types", "db_cross_references")
DATA_LINKS_LINK_FIELDS = ("fetched_at", "sources", "link_count", "truncated", "resources", "links")
_DATA_LINKS_PATHS = frozenset(
    f"data_links.{f}" for f in DATA_LINKS_SUMMARY_FIELDS + DATA_LINKS_LINK_FIELDS
)
# v1.5.0: the reserved external cross-reference identifiers, written by the `identifiers` mode.
IDENTIFIER_FIELDS = (
    "identifiers.dome_registry",
    "identifiers.bioai_repo",
    "identifiers.huggingface",
    "identifiers.kaggle",
    "identifiers.zenodo",
)

# Mode -> the exact leaf paths that mode may write. Adding a field to a document means adding it
# here first, on purpose, in a diff someone reviews.
WRITE_MODES: dict[str, frozenset[str]] = {
    # The in-place 1.1.0 -> 1.2.0 shape bump. Constant values, no per-document data.
    "migrate_v1_2_0": frozenset({
        _SCHEMA_VERSION_PATH,
        "source.decision_provenance",
        "publication_metadata.citation_count_updated",
        "publication_metadata.citation_source",
    }),
    # The citation backfill. Deliberately cannot reach `decision_provenance`: a refresh of a
    # number must not be able to relabel who decided the record.
    "citations": frozenset({
        _SCHEMA_VERSION_PATH,
        "publication_metadata.citation_count",
        "publication_metadata.citation_count_updated",
        "publication_metadata.citation_source",
    }),
    # The licence backfill. Two fields, and the second is required rather than convenient:
    # `schema.py::_resolve_open_access` fixes the rule that EPMC's freshly-fetched flag wins
    # wherever a real lookup happened, so writing a licence while leaving `open_access` on the
    # stale pipeline value would leave the document in a state the schema says cannot exist.
    # Deliberately cannot reach `fulltext_available`, which is ours, not EPMC's.
    "licences": frozenset({
        _SCHEMA_VERSION_PATH,
        "source.access.license",
        "source.access.open_access",
    }),
    # The enrichment merge. Cannot reach `llm_classification` at all, which is the whole point:
    # enrichment is additive by construction and must not be able to revise a verdict.
    "enrichment": frozenset(
        {_SCHEMA_VERSION_PATH}
        | {f"content_filters.{f}" for f in _ENRICHED_FILTER_FIELDS}
        | {f"llm_enrichment.{f}" for f in _ENRICHMENT_FIELDS}
    ),
    # The in-place 1.2.0 -> 1.4.0 shape bump: the three v1.3.0 leaves and the ten data_links
    # leaves, every one set to its "never looked up" value. Constant values, no per-document data.
    "migrate_v1_4_0": frozenset({_SCHEMA_VERSION_PATH, *PREPRINT_FIELDS} | _DATA_LINKS_PATHS),
    # The Europe PMC identity / preprint backfill (docs/preprint.md). Exactly the three v1.3.0
    # leaves: it cannot reach `journal`, `pub_types` or anything that decides what a record is.
    "preprints": frozenset({_SCHEMA_VERSION_PATH, *PREPRINT_FIELDS}),
    # The data-links fetch. `resources` and `links` are arrays of objects and are leaves here on
    # purpose: `$set` replaces each whole and the rollback snapshot restores each whole (dig()
    # does not walk into arrays). Cannot reach `identifiers.*`: those go through their own mode.
    "data_links": frozenset({_SCHEMA_VERSION_PATH} | _DATA_LINKS_PATHS),
    # The in-place 1.4.0 -> 1.5.0 version stamp. v1.5.0's change lives inside the data_links arrays
    # and in identifier values, which the `data_links` and `identifiers` modes write per document,
    # so the migration sets nothing but the version.
    "migrate_v1_5_0": frozenset({_SCHEMA_VERSION_PATH}),
    # 1.5.0 -> 1.5.1: the version stamp only; v1.5.1 changed vocabulary metadata, not documents.
    "migrate_v1_5_1": frozenset({_SCHEMA_VERSION_PATH}),
    # The external cross-reference identifiers (cross_links/README.md). The data-links build fills
    # `dome_registry` from EBI Search's DOME Registry entries; the other four arrive with their own
    # passes. Cannot reach data_links, the Europe PMC identity (`identifiers.epmc_id` is the
    # `preprints` mode's) or anything that decides what a record is.
    "identifiers": frozenset({_SCHEMA_VERSION_PATH, *IDENTIFIER_FIELDS}),
}

_ABSENT = object()


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dig(doc: dict, path: str) -> Any:
    """Value at a dotted path, or the `_ABSENT` sentinel. Absent and `None` are different things
    here: a rollback has to restore "this field did not exist", not an explicit null."""
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


class WriteResult:
    def __init__(self) -> None:
        self.matched = 0
        self.modified = 0
        self.batches = 0
        self.errors: list[str] = []

    def as_dict(self) -> dict:
        return {
            "matched": self.matched,
            "modified": self.modified,
            "batches": self.batches,
            "errors": self.errors,
        }


class SafeWriter:
    def __init__(
        self,
        moros: Moros,
        mode: str,
        run_id: Optional[str] = None,
        dry_run: bool = True,
        rollback_dir: Path = ROLLBACK_DIR,
    ) -> None:
        if mode not in WRITE_MODES:
            raise ValueError(f"unknown write mode {mode!r} -- known modes: {sorted(WRITE_MODES)}")
        self.moros = moros
        self.mode = mode
        self.allowed = WRITE_MODES[mode]
        self.run_id = run_id or new_run_id(mode)
        self.dry_run = dry_run
        self.rollback_dir = Path(rollback_dir)
        self.rollback_path = self.rollback_dir / f"{self.run_id}.jsonl"

    # -- validation --------------------------------------------------------------

    def validate(self, fields: dict[str, Any]) -> None:
        """Raises on anything the mode may not write. Called for every single update, not just a
        sample -- a per-document guard is only worth having if it runs per document."""
        if not fields:
            raise ValueError("empty update -- refusing to issue a no-op write")
        for path in fields:
            if path == ID_FIELD:
                raise ValueError("refusing to write _id: it is the merge key, not a field")
            if path.startswith("$"):
                raise ValueError(f"refusing operator-shaped field name {path!r}")
            if path not in self.allowed:
                raise ValueError(
                    f"mode {self.mode!r} may not write {path!r}. Allowed paths: "
                    f"{sorted(self.allowed)}. If this field genuinely belongs to this mode, add "
                    f"it to WRITE_MODES on purpose rather than widening the check."
                )

    # -- rollback ----------------------------------------------------------------

    def snapshot(self, updates: list[tuple[str, dict[str, Any]]]) -> Path:
        """Records the pre-write state of exactly the paths about to change. Written and flushed
        before `apply` issues anything, so an interrupted write is still reversible."""
        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        paths = sorted({p for _, fields in updates for p in fields})
        projection = {p: 1 for p in paths}
        ids = [doc_id for doc_id, _ in updates]

        n = 0
        with self.rollback_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({
                "_meta": {
                    "run_id": self.run_id,
                    "mode": self.mode,
                    "target": self.moros.describe(),
                    "paths": paths,
                    "documents": len(ids),
                    "taken_at": utc_now_iso(),
                },
            }) + "\n")
            for batch_start in range(0, len(ids), BATCH_SIZE):
                batch = ids[batch_start : batch_start + BATCH_SIZE]
                cursor = self.moros.collection.find({ID_FIELD: {"$in": batch}}, projection)
                for doc in cursor:
                    restore: dict[str, Any] = {}
                    unset: list[str] = []
                    for path in paths:
                        value = dig(doc, path)
                        if value is _ABSENT:
                            unset.append(path)
                        else:
                            restore[path] = value
                    f.write(json.dumps({
                        ID_FIELD: doc[ID_FIELD], "set": restore, "unset": unset,
                    }) + "\n")
                    n += 1
                f.flush()
        print(f"[{self.run_id}] rollback snapshot: {n:,} documents -> {self.rollback_path}")
        return self.rollback_path

    # -- the write ---------------------------------------------------------------

    def apply(
        self,
        updates: Iterable[tuple[str, dict[str, Any]]],
        total: Optional[int] = None,
        take_snapshot: bool = True,
    ) -> WriteResult:
        """`updates` is an iterable of `(_id, {leaf_path: value})`. Validated in full, snapshotted,
        then applied as unordered `$set` batches so one bad document does not abort the rest."""
        materialised = list(updates)
        for _, fields in materialised:
            self.validate(fields)

        result = WriteResult()
        target = self.moros.describe()
        print(f"[{self.run_id}] mode={self.mode} target={target}")
        print(f"[{self.run_id}] {len(materialised):,} documents to update, "
              f"paths: {sorted({p for _, fs in materialised for p in fs})}")

        if self.dry_run:
            print(f"[{self.run_id}] DRY RUN -- nothing written. Sample of what would change:")
            for doc_id, fields in materialised[:5]:
                current = self.moros.get(doc_id, {p: 1 for p in fields})
                if current is None:
                    print(f"    {doc_id}  (NOT PRESENT in the collection -- would not match)")
                    continue
                for path, value in fields.items():
                    was = dig(current, path)
                    shown = "<absent>" if was is _ABSENT else repr(was)
                    print(f"    {doc_id}  {path}: {shown} -> {value!r}")
            print(f"[{self.run_id}] re-run with --confirm to write.")
            return result

        if take_snapshot:
            self.snapshot(materialised)

        for batch in _batched(materialised, BATCH_SIZE):
            ops = [UpdateOne({ID_FIELD: doc_id}, {"$set": fields}) for doc_id, fields in batch]
            try:
                res = self.moros.collection.bulk_write(ops, ordered=False)
                result.matched += res.matched_count
                result.modified += res.modified_count
            except Exception as exc:  # noqa: BLE001 -- keep going, record it, report at the end
                result.errors.append(f"batch starting {batch[0][0]}: {exc!r}"[:400])
            result.batches += 1

        print(f"[{self.run_id}] matched {result.matched:,}, modified {result.modified:,}, "
              f"{len(result.errors)} batch error(s)")
        for err in result.errors[:5]:
            print(f"    {err}")
        return result

    def apply_streaming(
        self,
        updates: Iterable[tuple[str, dict[str, Any]]],
        total: Optional[int] = None,
        desc: str = "write",
    ) -> WriteResult:
        """Same guarantees as `apply`, for loads too large to hold in memory.

        `apply` validates and snapshots everything before writing anything, which is the stronger
        property and right for the thousands-of-rows loads. At 811k rows that means a ~290MB list
        and a single 100MB snapshot write before the first document moves, so this variant works
        batch by batch instead: **validate the batch, snapshot the batch, then write the batch.**

        The ordering is what matters. An interrupted run leaves a snapshot covering everything that
        was written, plus at most one batch that was snapshotted but never written -- and replaying
        a snapshot entry for an unchanged document is a no-op, so the rollback stays correct either
        way. The failure this rules out is the one that matters: a document written with no record
        of what it held before.
        """
        result = WriteResult()
        print(f"[{self.run_id}] mode={self.mode} target={self.moros.describe()}")

        if self.dry_run:
            # Materialise only the head -- enough to show what would change without the memory.
            head = []
            for item in updates:
                head.append(item)
                if len(head) >= 5:
                    break
            for _, fields in head:
                self.validate(fields)
            print(f"[{self.run_id}] DRY RUN -- nothing written. Sample of what would change:")
            for doc_id, fields in head:
                current = self.moros.get(doc_id, {p: 1 for p in fields})
                if current is None:
                    print(f"    {doc_id}  (NOT PRESENT -- would not match)")
                    continue
                for path, value in fields.items():
                    was = dig(current, path)
                    shown = "<absent>" if was is _ABSENT else repr(was)
                    print(f"    {doc_id}  {path}: {shown} -> {value!r}")
            print(f"[{self.run_id}] re-run with --confirm to write.")
            return result

        self.rollback_dir.mkdir(parents=True, exist_ok=True)
        n_snapshotted = 0
        wrote_meta = False

        with self.rollback_path.open("w", encoding="utf-8") as snap:
            progress = tqdm(total=total, desc=desc, unit="doc")
            for batch in _iter_batches(updates, BATCH_SIZE):
                for _, fields in batch:
                    self.validate(fields)
                paths = sorted({p for _, fields in batch for p in fields})

                if not wrote_meta:
                    snap.write(json.dumps({"_meta": {
                        "run_id": self.run_id, "mode": self.mode,
                        "target": self.moros.describe(), "paths": paths,
                        "documents": total, "taken_at": utc_now_iso(),
                        "streaming": True,
                    }}) + "\n")
                    wrote_meta = True

                ids = [doc_id for doc_id, _ in batch]
                cursor = self.moros.collection.find({ID_FIELD: {"$in": ids}}, {p: 1 for p in paths})
                for doc in cursor:
                    restore: dict[str, Any] = {}
                    unset: list[str] = []
                    for path in paths:
                        value = dig(doc, path)
                        if value is _ABSENT:
                            unset.append(path)
                        else:
                            restore[path] = value
                    snap.write(json.dumps({
                        ID_FIELD: doc[ID_FIELD], "set": restore, "unset": unset,
                    }) + "\n")
                    n_snapshotted += 1
                snap.flush()  # the snapshot is on disk BEFORE the write below is issued

                ops = [UpdateOne({ID_FIELD: doc_id}, {"$set": fields}) for doc_id, fields in batch]
                try:
                    res = self.moros.collection.bulk_write(ops, ordered=False)
                    result.matched += res.matched_count
                    result.modified += res.modified_count
                except Exception as exc:  # noqa: BLE001 -- record and keep going
                    result.errors.append(f"batch starting {batch[0][0]}: {exc!r}"[:400])
                result.batches += 1
                progress.update(len(batch))
            progress.close()

        print(f"[{self.run_id}] snapshot: {n_snapshotted:,} documents -> {self.rollback_path}")
        print(f"[{self.run_id}] matched {result.matched:,}, modified {result.modified:,}, "
              f"{len(result.errors)} batch error(s)")
        for err in result.errors[:5]:
            print(f"    {err}")
        return result

    # -- reporting ---------------------------------------------------------------

    def write_report(self, extra: dict) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"{self.run_id}.report.json"
        payload = {
            "run_id": self.run_id,
            "mode": self.mode,
            "dry_run": self.dry_run,
            "target": self.moros.describe(),
            "finished_at": utc_now_iso(),
            "rollback": str(self.rollback_path) if self.rollback_path.exists() else None,
            **extra,
        }
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"[{self.run_id}] report -> {path}")
        return path


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _iter_batches(items: Iterable, size: int) -> Iterator[list]:
    """Same batching over an iterator, so a streaming load never materialises the whole input."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Rollback replay -- the only thing this file does from the command line.
# ---------------------------------------------------------------------------


def replay_rollback(moros: Moros, path: Path, confirm: bool) -> WriteResult:
    """Restores the paths a prior run changed, using its snapshot. `set` values are written back;
    `unset` paths are removed, so a field that did not exist before does not come back as null."""
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0]).get("_meta", {})
    entries = [json.loads(line) for line in lines[1:] if line.strip()]
    print(f"rollback {path.name}: mode={meta.get('mode')} run={meta.get('run_id')} "
          f"paths={meta.get('paths')} documents={len(entries):,}")
    print(f"target: {moros.describe()}")

    if not confirm:
        print("DRY RUN -- pass --confirm to restore. First 5 entries:")
        for entry in entries[:5]:
            print(f"    {entry[ID_FIELD]}  set={entry['set']} unset={entry['unset']}")
        return WriteResult()

    result = WriteResult()
    for batch in _batched(entries, BATCH_SIZE):
        ops = []
        for entry in batch:
            update: dict[str, Any] = {}
            if entry["set"]:
                update["$set"] = entry["set"]
            if entry["unset"]:
                update["$unset"] = {p: "" for p in entry["unset"]}
            if update:
                ops.append(UpdateOne({ID_FIELD: entry[ID_FIELD]}, update))
        if not ops:
            continue
        try:
            res = moros.collection.bulk_write(ops, ordered=False)
            result.matched += res.matched_count
            result.modified += res.modified_count
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"batch starting {batch[0][ID_FIELD]}: {exc!r}"[:400])
        result.batches += 1
    print(f"rollback: matched {result.matched:,}, modified {result.modified:,}, "
          f"{len(result.errors)} error(s)")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", type=Path, required=True, help="A rollback snapshot JSONL.")
    parser.add_argument("--confirm", action="store_true", help="Actually restore. Off by default.")
    args = parser.parse_args()
    with Moros.from_env() as moros:
        replay_rollback(moros, args.rollback, args.confirm)


if __name__ == "__main__":
    main()
