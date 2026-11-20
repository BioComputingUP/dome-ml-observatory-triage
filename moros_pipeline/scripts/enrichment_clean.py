"""Deterministic vocabulary cleaning of enrichment values at ingestion (rule set `c1`).

The enrichment parsers (e1, e2, e3) **flag** a value outside the controlled vocabulary but never drop
it ("nothing is ever silently dropped"), so a flagged value would otherwise land in moros and surface
as a facet value no published vocabulary defines. This module is the one place those values are
resolved, on the ingestion side, whichever laptop or provider produced the events:

    events (raw, never edited) -> load_enrichment.py -> clean_fields() -> moros

It is pure and idempotent: cleaning a cleaned record changes nothing. Its only inputs are the
vocabularies of the **current published release** -- `curation_criteria/domain_vocab.json` and
`curation_criteria/e3/modelling_branch_vocab.json`, the files `schema/check_alignment.py` ties to
`schema/releases/<CURRENT>/vocab/` in dome-ml-observatory. e3's modelling vocabularies are a strict
superset of e2's and e1's, so every value valid under an older prompt stays valid here.

Rules, applied per closed field, in this order:

1. **case**    -- a value matching a label case-insensitively (after trimming) takes the label.
2. **dedup**   -- a repeated value is kept once.
3. **trimmed** -- a field over its `max_tags` keeps its first N values, in the model's order.
4. **moved**   -- a value that is not a term of its field but is a term of a sibling field of the same
   vocabulary (domain tiers 1-3; learning_paradigm / model_family / model_use) moves there, if that
   field has room under its cap and does not already hold it. Otherwise it is dropped
   (`dropped_full` / `dropped_duplicate`).
5. **dropped_unknown** -- a value that is a term of no field is dropped.

`model_type` is an open vocabulary (seed list + verbatim names, aliases already normalised by the
parser) and is never touched; nor are the parser's heuristic `model_type:excluded_kind:*` flags.

Every change is appended to `llm_enrichment.vocab_violations` as
`cleaned:c1:<field>:<action>:<value>` (moves as `cleaned:c1:<field>:moved_to_<target>:<value>`), so
the stored document says exactly what ingestion did and the raw event log still says what the model
returned.

Read-only audit over events files (no database), also writing the vocabulary-gap list:

    python3 enrichment_clean.py --events ../../../dome-ml-observatory-enrichment-runs/events/*/*.csv \\
        --gaps-md ../../docs/enrichment_vocab_gaps.md
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

csv.field_size_limit(sys.maxsize)

RULESET = "c1"
REPO = Path(__file__).resolve().parents[2]
DOMAIN_VOCAB = REPO / "curation_criteria" / "domain_vocab.json"
MODELLING_VOCAB = REPO / "curation_criteria" / "e3" / "modelling_branch_vocab.json"

DOMAIN_FIELDS = ("domain_tier1", "domain_tier2", "domain_tier3")
MODELLING_FIELDS = ("learning_paradigm", "model_family", "model_use")
GROUPS = (DOMAIN_FIELDS, MODELLING_FIELDS)
CLOSED_FIELDS = DOMAIN_FIELDS + MODELLING_FIELDS
# domain_tier1 is a scalar on the document (max_tags 1); every other closed field is a list.
SCALAR_FIELDS = ("domain_tier1",)
VIOLATIONS_PATH = "llm_enrichment.vocab_violations"
CLEANED_PREFIX = f"cleaned:{RULESET}:"


@dataclass(frozen=True)
class Vocab:
    labels: dict[str, dict[str, str]]   # field -> {lowercased label: label}
    caps: dict[str, int]
    sha256: str                         # of the two vocabulary files, in order

    def canonical(self, field: str, value: str) -> str | None:
        return self.labels[field].get(value.strip().lower())


def load_vocab(domain_path: Path = DOMAIN_VOCAB, modelling_path: Path = MODELLING_VOCAB) -> Vocab:
    labels: dict[str, dict[str, str]] = {}
    caps: dict[str, int] = {}
    digest = hashlib.sha256()
    for path in (domain_path, modelling_path):
        raw = path.read_bytes()
        digest.update(raw)
        fields = json.loads(raw)["fields"]
        for name, spec in fields.items():
            if name not in CLOSED_FIELDS:
                continue
            labels[name] = {t["label"].strip().lower(): t["label"] for t in spec["terms"]}
            caps[name] = int(spec["max_tags"])
    missing = [f for f in CLOSED_FIELDS if f not in labels]
    if missing:
        raise SystemExit(f"vocabulary files lack closed fields {missing}: {domain_path}, {modelling_path}")
    return Vocab(labels=labels, caps=caps, sha256=digest.hexdigest())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [v for v in value if isinstance(v, str) and v.strip()]


def clean_values(values: dict[str, Any], vocab: Vocab) -> tuple[dict[str, Any], list[str]]:
    """`values` maps closed field names (bare, e.g. `domain_tier2`) to their stored values; fields
    absent from `values` are neither read nor written. Returns `(cleaned, changes)`; `cleaned` has
    the same keys and shapes as `values`."""
    present = [f for f in CLOSED_FIELDS if f in values]
    kept: dict[str, list[str]] = {}
    misplaced: list[tuple[str, str, str]] = []   # (from field, value, to field)
    changes: list[str] = []

    for field in present:
        out: list[str] = []
        for raw in _as_list(values[field]):
            label = vocab.canonical(field, raw)
            if label is not None:
                if label != raw:
                    changes.append(f"{CLEANED_PREFIX}{field}:case:{raw}->{label}")
                if label in out:
                    changes.append(f"{CLEANED_PREFIX}{field}:dedup:{label}")
                    continue
                out.append(label)
                continue
            group = next(g for g in GROUPS if field in g)
            target = next((g for g in group if g != field and vocab.canonical(g, raw) is not None), None)
            if target is None:
                changes.append(f"{CLEANED_PREFIX}{field}:dropped_unknown:{raw}")
            else:
                misplaced.append((field, vocab.canonical(target, raw), target))
        cap = vocab.caps[field]
        for extra in out[cap:]:
            changes.append(f"{CLEANED_PREFIX}{field}:trimmed:{extra}")
        kept[field] = out[:cap]

    for source, label, target in misplaced:
        if target not in values:
            # The record does not carry the target field at all (e.g. an e1 event and model_use):
            # writing it would invent a value the prompt never produced for that field's question.
            changes.append(f"{CLEANED_PREFIX}{source}:dropped_unknown:{label}")
            continue
        bucket = kept.setdefault(target, [])
        if label in bucket:
            changes.append(f"{CLEANED_PREFIX}{source}:dropped_duplicate:{label}")
        elif len(bucket) >= vocab.caps[target]:
            changes.append(f"{CLEANED_PREFIX}{source}:dropped_full:{label}")
        else:
            bucket.append(label)
            changes.append(f"{CLEANED_PREFIX}{source}:moved_to_{target}:{label}")

    cleaned: dict[str, Any] = {}
    for field in present:
        out = kept.get(field, [])
        cleaned[field] = (out[0] if out else None) if field in SCALAR_FIELDS else out
    return cleaned, changes


def clean_fields(fields: dict[str, Any], vocab: Vocab) -> tuple[dict[str, Any], list[str]]:
    """The loader's form: `fields` is a `{leaf_path: value}` update (as `load_enrichment.event_to_update`
    builds it). Returns a new dict with the closed `content_filters.*` values cleaned and every change
    appended to `llm_enrichment.vocab_violations` (if that path is in the update)."""
    bare = {path.split(".", 1)[1]: value for path, value in fields.items()
            if path.startswith("content_filters.") and path.split(".", 1)[1] in CLOSED_FIELDS}
    if not bare:
        return dict(fields), []
    cleaned, changes = clean_values(bare, vocab)
    if not changes:
        return dict(fields), []
    out = dict(fields)
    for field, value in cleaned.items():
        out[f"content_filters.{field}"] = value
    if VIOLATIONS_PATH in out:
        out[VIOLATIONS_PATH] = list(out[VIOLATIONS_PATH] or []) + changes
    return out, changes


# --- read-only audit over events files ---------------------------------------------------------------

def _event_values(row: dict[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field in CLOSED_FIELDS:
        if field not in row:
            continue
        try:
            parsed = json.loads(row[field] or "[]")
        except json.JSONDecodeError:
            parsed = []
        values[field] = parsed if isinstance(parsed, list) else []
    return values


def audit(paths: Iterable[Path], vocab: Vocab) -> dict:
    """Counts what ingestion will do to these events files, plus the parser's own flags. Only the last
    ok event per record counts, as in load_enrichment.iter_updates."""
    latest: dict[str, dict[str, str]] = {}
    files = []
    for path in paths:
        files.append(str(path))
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("parse_status") or "").strip() == "parse_error":
                    continue
                if (row.get("record_id") or "").strip():
                    latest[row["record_id"].strip()] = row
    actions: collections.Counter = collections.Counter()
    flags: collections.Counter = collections.Counter()
    records_cleaned = 0
    for row in latest.values():
        _, changes = clean_values(_event_values(row), vocab)
        records_cleaned += bool(changes)
        for change in changes:
            actions[change[len(CLEANED_PREFIX):]] += 1
        try:
            for flag in json.loads(row.get("vocab_violations") or "[]"):
                if isinstance(flag, str) and flag.startswith("model_type:"):
                    flags[flag] += 1
        except json.JSONDecodeError:
            pass
    return {"files": files, "records": len(latest), "records_cleaned": records_cleaned,
            "actions": actions, "model_type_flags": flags, "vocab_sha256": vocab.sha256}


def gaps_markdown(result: dict) -> str:
    lines = [
        "# Enrichment vocabulary gaps",
        "",
        "Generated by `moros_pipeline/scripts/enrichment_clean.py --gaps-md` (read-only, deterministic) "
        "from the events files listed at the end. Do not edit by hand; re-run it.",
        "",
        f"- Records (last ok event per record): {result['records']:,}",
        f"- Records changed by ingestion cleaning `{RULESET}`: {result['records_cleaned']:,}",
        f"- Vocabularies (domain + e3 modelling) sha256: `{result['vocab_sha256'][:16]}…`",
        "",
        "## What ingestion cleaning does",
        "",
        "`dropped_unknown` values are candidates for a later vocabulary version; `moved_to_*` values were "
        "real terms the model put in the wrong field.",
        "",
        "| Count | Field | Action | Value |",
        "|---:|---|---|---|",
    ]
    for key, n in sorted(result["actions"].items(), key=lambda kv: (-kv[1], kv[0])):
        field, action, value = key.split(":", 2)
        lines.append(f"| {n:,} | {field} | {action} | {value.replace('|', '/')} |")
    lines += [
        "",
        "## model_type flags (kept, not cleaned)",
        "",
        "`model_type` is an open vocabulary. These are the parser's heuristic flags for items the prompt's "
        "rules exclude (optimisers that only tune another model, simulations, tasks, components). They stay "
        "in the record; the counts are here for the next prompt version. Top 60:",
        "",
        "| Count | Flag |",
        "|---:|---|",
    ]
    for flag, n in result["model_type_flags"].most_common(60):
        lines.append(f"| {n:,} | {flag.replace('|', '/')} |")
    lines += ["", "## Events files", ""] + [f"- `{Path(p).as_posix().split('/events/')[-1]}`" for p in result["files"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--gaps-md", type=Path, default=None, help="Write the vocabulary-gap list here.")
    args = parser.parse_args()
    vocab = load_vocab()
    result = audit(sorted(args.events), vocab)
    print(f"{len(result['files'])} file(s), {result['records']:,} records, "
          f"{result['records_cleaned']:,} changed by cleaning {RULESET}")
    by_action = collections.Counter()
    for key, n in result["actions"].items():
        field, action, _ = key.split(":", 2)
        by_action[(field, action)] += n
    for (field, action), n in sorted(by_action.items()):
        print(f"  {n:6,}  {field}:{action}")
    if args.gaps_md:
        args.gaps_md.write_text(gaps_markdown(result), encoding="utf-8")
        print(f"wrote {args.gaps_md}")


if __name__ == "__main__":
    main()
