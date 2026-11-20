"""Merges an enrichment event log back into moros, in place, by `_id`.

Enrichment fills `content_filters`' six reserved fields (seven from v1.7.0: prompt e2 adds
`model_use`) and the whole `llm_enrichment` group on documents that **already exist**. Both groups
were fleshed out in schema v1.0.0 precisely so this would be an update rather than a migration.

Safety comes from `moros_write.WRITE_MODES["enrichment"]`, whose allowlist contains those leaf
paths and nothing else. `llm_classification` is not reachable from this code path at all, which is
what makes "enrichment is additive and cannot revise a verdict" a structural property rather than
a claim about the prompt. A rollback snapshot is taken before the first batch, as for every write.

Two shape details the event log does not share with the document:

- `domain_tier1` is a **list of at most one** in the event (the parser treats every vocab field
  uniformly) but a **scalar** in the schema. It is unwrapped here.
- The event has no `mode`; enrichment has no forced-choice variant the way classification does, so
  it stays null rather than being given an invented value.
- `provider` and `model_id` come from the event's `model_tier`: DeepSeek for `flash`/`pro` (prompt
  e1), Z.ai for `glm-5.3-flash`/`glm-5.3` (prompt e2), EMBL-EBI for `ebi-glm-5.3-flash` (prompt e3).
  When the event carries its own `model_id` (prompt e3: the id the server reported, verbatim, e.g.
  `zai-org/GLM-5.3-Flash`), that is stored instead of the tier's mirror.
- Prompt e2's and e3's events also carry `model_use` (written to `content_filters.model_use`) and
  `family_backfilled` (not stored: the parser's own audit trail, kept in the event log). An e2
  events file is refused until every document is at schema v1.7.0 (`migrate_v1_7_0.py`).

Two things happen on the way in, both deterministic and both recorded on the document (schema v1.8.0):

- **Vocabulary cleaning** (`enrichment_clean.py`, rule set `c1`): a closed-vocabulary value the parser
  flagged but kept is moved to its right field, trimmed to the cap or dropped, and each change is
  appended to `llm_enrichment.vocab_violations` as `cleaned:c1:...`. The events file is never edited.
- **History**: a document that already holds an enrichment from a different prompt or batch keeps it.
  The old `llm_enrichment` group and its `content_filters` enrichment values are appended to
  `enrichment_history` before the new values are set; the observatory reads only the current one.
  Re-loading the same batch adds nothing.

`--clean-existing` applies the same cleaning to the enrichment already stored in moros (no events
file), writing only documents whose values change.

`parse_error` events are skipped entirely. Re-running `enrich` retries them, and a half-parsed
enrichment is worse than none.

    python3 load_enrichment.py --events ../output/enrichment_<journal>_events.csv
    python3 load_enrichment.py --events ... --limit 25 --confirm
    python3 load_enrichment.py --events ... --confirm
    python3 load_enrichment.py --events a_events.csv b_events.csv ... --confirm   # one run per file
    python3 load_enrichment.py --clean-existing [--confirm]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from enrichment_clean import RULESET, Vocab, clean_fields, clean_values, load_vocab
from moros_client import ID_FIELD, Moros
from moros_write import REPORT_DIR, SafeWriter, new_run_id, record_modified_stamp

csv.field_size_limit(sys.maxsize)

# Mirrors mongo_landscape_export/scripts/schema.py's TIER_MODEL_IDS, which itself mirrors
# llm_classify/deepseek_client.py and llm_classify/zai_client.py. Same documented mirror, same reason:
# this folder cannot import the dome_triage package.
TIER_MODEL_IDS = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro",
                  "glm-5.3-flash": "glm-5.3-flash", "glm-5.3": "glm-5.3"}
TIER_PROVIDERS = {"flash": "deepseek", "pro": "deepseek", "glm-5.3-flash": "zai", "glm-5.3": "zai",
                  "ebi-glm-5.3-flash": "embl-ebi"}
PROVIDER = "deepseek"  # the provider of every e1 event, and of an event with no tier
PARSE_ERROR = "parse_error"

LIST_FIELDS = ("domain_tier2", "domain_tier3", "learning_paradigm", "model_family", "model_type")
# v1.7.0: only prompt e2's events carry it. An e1 event leaves the document's value alone ([] since
# migrate_v1_7_0.py), because e1 never asked the question.
E2_LIST_FIELDS = ("model_use",)
E2_MIN_SCHEMA_VERSION = "1.7.0"
# v1.8.0: every load may append to enrichment_history, so every load needs the whole corpus at 1.8.0.
MIN_SCHEMA_VERSION = "1.8.0"
HISTORY_PATH = "enrichment_history"
ENRICHED_FILTER_FIELDS = ("domain_tier1", "domain_tier2", "domain_tier3", "learning_paradigm",
                          "model_family", "model_type", "model_use")
CURRENT_PROJECTION = {"llm_enrichment": 1, HISTORY_PATH: 1,
                      **{f"content_filters.{f}": 1 for f in ENRICHED_FILTER_FIELDS}}


def _json_list(value: str) -> list:
    value = (value or "").strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _int(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _bool(value: str) -> bool | None:
    value = (value or "").strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    return None


def event_to_update(row: dict[str, str]) -> tuple[str, dict[str, Any]] | None:
    """One enrichment event -> `(_id, {leaf_path: value})`, or None to skip it."""
    record_id = (row.get("record_id") or "").strip()
    if not record_id:
        return None
    if (row.get("parse_status") or "").strip() == PARSE_ERROR:
        return None

    tier = (row.get("model_tier") or "").strip() or None
    tier1 = _json_list(row.get("domain_tier1", ""))

    fields: dict[str, Any] = {
        # domain_tier1 is max_tags=1 in the vocabulary, so the event's single-element list becomes
        # the schema's scalar. An empty list means the model named no tier-1 domain.
        "content_filters.domain_tier1": (tier1[0] if tier1 else None),
        "llm_enrichment.provider": TIER_PROVIDERS.get(tier or "", PROVIDER),
        "llm_enrichment.model_tier": tier,
        "llm_enrichment.model_id": (row.get("model_id") or "").strip() or TIER_MODEL_IDS.get(tier or ""),
        "llm_enrichment.mode": None,
        "llm_enrichment.rationale": (row.get("rationale") or "").strip() or None,
        "llm_enrichment.prompt_version": (row.get("prompt_version") or "").strip() or None,
        "llm_enrichment.ruleset_sha256": (row.get("vocab_sha256") or "").strip() or None,
        "llm_enrichment.batch_id": (row.get("batch_id") or "").strip() or None,
        "llm_enrichment.timestamp": (row.get("timestamp") or "").strip() or None,
        "llm_enrichment.vocab_violations": _json_list(row.get("vocab_violations", "")),
        "llm_enrichment.parse_status": (row.get("parse_status") or "").strip() or None,
        "llm_enrichment.input_tokens": _int(row.get("input_tokens", "")),
        "llm_enrichment.output_tokens": _int(row.get("output_tokens", "")),
        "llm_enrichment.cache_hit_tokens": _int(row.get("cache_hit_tokens", "")),
        "llm_enrichment.parse_fallback_used": _bool(row.get("parse_fallback_used", "")),
    }
    for field in LIST_FIELDS:
        fields[f"content_filters.{field}"] = _json_list(row.get(field, ""))
    for field in E2_LIST_FIELDS:
        if field in row:
            fields[f"content_filters.{field}"] = _json_list(row.get(field, ""))
    return record_id, fields


def _version(value) -> tuple:
    """'1.10.0' > '1.7.0' numerically; an unparseable or missing version sorts lowest (refused)."""
    try:
        return tuple(int(x) for x in str(value).split("."))
    except ValueError:
        return (-1,)


def carries_e2_fields(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), [])
    return any(field in header for field in E2_LIST_FIELDS)


def schema_problem(versions: dict) -> str | None:
    """Every load may write `enrichment_history` (v1.8.0). Refuse while any document is below it."""
    behind = {v: n for v, n in versions.items() if _version(v) < _version(MIN_SCHEMA_VERSION)}
    if behind:
        return (f"refusing: enrichment loads write v{MIN_SCHEMA_VERSION} fields but "
                f"{sum(behind.values()):,} documents are below it ({behind}). Publish v{MIN_SCHEMA_VERSION} in "
                f"dome-ml-observatory, run check_alignment.py, then the migrations up to "
                f"migrate_v{MIN_SCHEMA_VERSION.replace('.', '_')}.py --confirm first.")
    return None


def e2_schema_problem(versions: dict) -> str | None:
    """e2 events write a v1.7.0 field. Refuse while any document is below 1.7.0, or the values would
    land under a version whose schema and vocabulary do not have them."""
    behind = {v: n for v, n in versions.items() if _version(v) < _version(E2_MIN_SCHEMA_VERSION)}
    if behind:
        return (f"refusing: these events carry model_use (prompt e2, schema {E2_MIN_SCHEMA_VERSION}) but "
                f"{sum(behind.values()):,} documents are below it ({behind}). Publish v{E2_MIN_SCHEMA_VERSION} in "
                f"dome-ml-observatory, run check_alignment.py, then migrate_v1_7_0.py --confirm first.")
    return None


def iter_updates(path: Path, limit: int | None) -> Iterator[tuple[str, dict[str, Any]]]:
    """The event log is append-only: a retried record appends a new row rather than replacing the
    old one, so a file can hold several events per record. The **last** successful one is the
    verdict, so the file is read fully and only then emitted -- a single forward pass keeping the
    first occurrence would silently prefer the oldest attempt, which is the opposite of what the
    resumability contract promises.

    (Real case: a run that accidentally spawned overlapping containers produced 3,935 event rows
    for 1,424 records. Every duplicate was a valid enrichment of the same record, so the choice was
    harmless there -- but only by luck, and it would not be after a genuine retry of a parse
    error.)"""
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            update = event_to_update(row)
            if update is None:
                continue
            record_id, fields = update
            if record_id not in latest:
                order.append(record_id)
            latest[record_id] = fields
    for n, record_id in enumerate(order, start=1):
        yield record_id, latest[record_id]
        if limit is not None and n >= limit:
            return


def summarise(path: Path) -> dict:
    total = ok = errors = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if (row.get("parse_status") or "").strip() == PARSE_ERROR:
                errors += 1
            else:
                ok += 1
    return {"events": total, "ok": ok, "parse_errors": errors}


def history_entry(current: dict, new_fields: dict[str, Any], stamp: str) -> dict:
    """The enrichment a document holds now, as one `enrichment_history` element."""
    filters = current.get("content_filters") or {}
    return {
        "llm_enrichment": dict(current.get("llm_enrichment") or {}),
        "content_filters": {f: filters.get(f) for f in ENRICHED_FILTER_FIELDS},
        "superseded_at": stamp,
        "superseded_by": {
            "prompt_version": new_fields.get("llm_enrichment.prompt_version"),
            "batch_id": new_fields.get("llm_enrichment.batch_id"),
        },
    }


def supersedes(current: dict | None, new_fields: dict[str, Any]) -> bool:
    """True when the document already holds an enrichment from another prompt or batch. Re-loading
    the batch it already holds is not a supersession (a resumed or repeated load stays idempotent)."""
    enrichment = (current or {}).get("llm_enrichment") or {}
    if enrichment.get("provider") is None:
        return False
    return ((enrichment.get("prompt_version"), enrichment.get("batch_id"))
            != (new_fields.get("llm_enrichment.prompt_version"), new_fields.get("llm_enrichment.batch_id")))


def prepare(updates: list[tuple[str, dict[str, Any]]], current: dict[str, dict], vocab: Vocab,
            stamp: str) -> tuple[list[tuple[str, dict[str, Any]]], dict, list[tuple[str, str]]]:
    """Pure: cleans every update and adds `enrichment_history` where a load supersedes an enrichment.
    `current` maps `_id` -> the document's present enrichment (CURRENT_PROJECTION). Returns
    `(updates, stats, cleaning rows)`."""
    out: list[tuple[str, dict[str, Any]]] = []
    rows: list[tuple[str, str]] = []
    stats = {"cleaned_documents": 0, "cleaning_changes": 0, "superseded": 0}
    for doc_id, fields in updates:
        fields, changes = clean_fields(fields, vocab)
        if changes:
            stats["cleaned_documents"] += 1
            stats["cleaning_changes"] += len(changes)
            rows.extend((doc_id, change) for change in changes)
        doc = current.get(doc_id)
        if supersedes(doc, fields):
            history = list((doc or {}).get(HISTORY_PATH) or [])
            fields = {**fields, HISTORY_PATH: history + [history_entry(doc, fields, stamp)]}
            stats["superseded"] += 1
        out.append((doc_id, fields))
    return out, stats, rows


def read_current(moros: Moros, ids: list[str]) -> dict[str, dict]:
    current: dict[str, dict] = {}
    for start in range(0, len(ids), 5_000):
        for doc in moros.collection.find({ID_FIELD: {"$in": ids[start:start + 5_000]}}, CURRENT_PROJECTION):
            current[doc[ID_FIELD]] = doc
    return current


def write_cleaning_rows(run_id: str, rows: list[tuple[str, str]]) -> Path | None:
    if not rows:
        return None
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{run_id}.cleaning.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "change"])
        writer.writerows(rows)
    return path


def _gate(moros: Moros, run_id: str) -> None:
    problem = schema_problem(moros.histogram("schema_version"))
    if problem:
        raise SystemExit(f"[{run_id}] {problem}")


def run(events_path: Path, confirm: bool, limit: int | None) -> None:
    if not events_path.exists():
        raise SystemExit(f"{events_path} does not exist")
    run_id = new_run_id("load_enrichment")
    stats = summarise(events_path)
    updates = list(iter_updates(events_path, limit))
    print(f"[{run_id}] {events_path.name}: {stats['events']:,} events "
          f"({stats['ok']:,} ok, {stats['parse_errors']:,} parse_error skipped)")
    print(f"[{run_id}] {len(updates):,} distinct documents to enrich"
          + (f" (--limit {limit})" if limit else ""))
    if not updates:
        print(f"[{run_id}] nothing to do.")
        return

    vocab = load_vocab()
    with Moros.from_env() as moros:
        _gate(moros, run_id)
        writer = SafeWriter(moros, mode="enrichment", run_id=run_id, dry_run=not confirm)
        before = moros.count({"llm_enrichment.provider": {"$ne": None}})
        ids = [doc_id for doc_id, _ in updates]
        current = read_current(moros, ids)
        absent = [doc_id for doc_id in ids if doc_id not in current]
        if absent:
            print(f"[{run_id}] !! {len(absent):,} record_ids are not in the collection and will "
                  f"not match. If these came from a non-Mongo export, their record_id is a sha1 "
                  f"rather than the document _id. First few: {absent[:3]}")

        updates, prep, rows = prepare(updates, current, vocab, writer.stamp or record_modified_stamp())
        cleaning_path = write_cleaning_rows(run_id, rows)
        print(f"[{run_id}] cleaning {RULESET} (vocab {vocab.sha256[:12]}): {prep['cleaned_documents']:,} "
              f"documents, {prep['cleaning_changes']:,} changes"
              + (f" -> {cleaning_path}" if cleaning_path else ""))
        print(f"[{run_id}] superseding an existing enrichment (kept in {HISTORY_PATH}): {prep['superseded']:,}")

        result = writer.apply_streaming(iter(updates), total=len(updates), desc="enrich")
        after = moros.count({"llm_enrichment.provider": {"$ne": None}}) if confirm else before
        if confirm:
            print(f"[{run_id}] enriched documents: {before:,} -> {after:,}")
            violations = moros.count({"llm_enrichment.vocab_violations": {"$nin": [None, []]}})
            print(f"[{run_id}] documents with at least one vocab violation or cleaning note: {violations:,}")
        writer.write_report({
            "events_file": str(events_path), "event_stats": stats,
            "documents_updated": len(updates), "ids_not_found": len(absent),
            "cleaning": {"ruleset": RULESET, "vocab_sha256": vocab.sha256, **prep,
                         "rows": str(cleaning_path) if cleaning_path else None},
            "result": result.as_dict(), "enriched_before": before, "enriched_after": after,
        })
        if confirm:
            print(f"\n[{run_id}] next: python3 verify_corpus.py, then restart observatory-ws so "
                  f"the enrichment-coverage banner and facets pick this up.")


def clean_existing(confirm: bool, limit: int | None) -> None:
    """Applies cleaning `c1` to the enrichment already stored in moros. Writes only documents whose
    closed-vocabulary values change, through the same `enrichment` allowlist."""
    run_id = new_run_id("clean_existing_enrichment")
    vocab = load_vocab()
    with Moros.from_env() as moros:
        _gate(moros, run_id)
        writer = SafeWriter(moros, mode="enrichment", run_id=run_id, dry_run=not confirm)
        projection = {"llm_enrichment.vocab_violations": 1,
                      **{f"content_filters.{f}": 1 for f in ENRICHED_FILTER_FIELDS}}
        updates: list[tuple[str, dict[str, Any]]] = []
        rows: list[tuple[str, str]] = []
        scanned = 0
        for doc in moros.collection.find({"llm_enrichment.provider": {"$ne": None}}, projection):
            scanned += 1
            filters = doc.get("content_filters") or {}
            values = {f: filters[f] for f in ENRICHED_FILTER_FIELDS if f in filters and f != "model_type"}
            cleaned, changes = clean_values(values, vocab)
            if not changes:
                continue
            fields: dict[str, Any] = {f"content_filters.{f}": v for f, v in cleaned.items() if v != values[f]}
            existing = (doc.get("llm_enrichment") or {}).get("vocab_violations") or []
            fields["llm_enrichment.vocab_violations"] = list(existing) + changes
            updates.append((doc[ID_FIELD], fields))
            rows.extend((doc[ID_FIELD], c) for c in changes)
            if limit is not None and len(updates) >= limit:
                break
        cleaning_path = write_cleaning_rows(run_id, rows)
        print(f"[{run_id}] scanned {scanned:,} enriched documents; {len(updates):,} change under cleaning "
              f"{RULESET} ({len(rows):,} changes)" + (f" -> {cleaning_path}" if cleaning_path else ""))
        if not updates:
            print(f"[{run_id}] nothing to do.")
            return
        result = writer.apply(updates)
        writer.write_report({"mode": "clean_existing", "ruleset": RULESET, "vocab_sha256": vocab.sha256,
                             "scanned": scanned, "documents_updated": len(updates),
                             "rows": str(cleaning_path) if cleaning_path else None,
                             "result": result.as_dict()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--events", type=Path, nargs="+", help="One or more events files; one run each.")
    target.add_argument("--clean-existing", action="store_true",
                        help=f"Apply cleaning {RULESET} to the enrichment already in moros.")
    parser.add_argument("--confirm", action="store_true", help="Actually write. Off by default.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.clean_existing:
        clean_existing(args.confirm, args.limit)
        return
    for events in args.events:
        run(events, args.confirm, args.limit)


if __name__ == "__main__":
    main()
