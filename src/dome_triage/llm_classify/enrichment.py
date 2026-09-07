"""Step 20j: the positive-set enrichment pass -- domain (3 EDAM tiers), learning paradigm, model
family, and open-vocabulary model type, per the three OFFICIAL vocabularies in
`curation_criteria/` (domain_vocab.json / modelling_branch_vocab.json / model_type_seed_vocab.json,
Steps 20g-20i).

Additive-only by construction: this prompt never asks the positive/negative/undeterminable
question, so it cannot conflict with the finalized Step 20 classification no matter what it
returns. It reads ONLY title/abstract/journal/year off each record -- same explicit-key blinding
boundary as `prompts.py` -- so the provenance columns the trial CSV carries for audit
(label/label_confidence/cross_curate_*) can never reach the model.

Token efficiency (real mechanism, not hand-waving): the static system message -- instructions plus
all three vocabularies -- is built ONCE per run and byte-identical across every call, with all
record-specific content isolated in the user message. DeepSeek's API applies automatic prefix
caching on repeated prompt prefixes and reports it back per-call as
`usage.prompt_cache_hit_tokens`; this module records that field on every event and the runner shows
a live cache-hit percentage, so the saving is observable in real numbers rather than assumed.
(Already confirmed live on this project: a list-price estimate over 16 calibration calls came out
~5x the dashboard's actual billed delta, largely because of exactly this prefix caching -- see
deepseek_client.py's pricing-table comment.)

Crash-safety is the caller's job (`pipeline/steps.py` streams each yielded event to disk
immediately, same as the classify path -- a machine/network death loses at most the in-flight
calls); resumability is this module's job (`_already_enriched_ids` keys on record_id +
prompt_version + vocab_sha256, so re-running the same command skips everything already done and
retries parse_errors, and a vocabulary edit automatically makes a fresh, non-conflated batch).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from dome_triage.llm_classify.deepseek_client import DeepSeekClient
from dome_triage.llm_classify.response_parser import PARSE_ERROR, _BRACE_BLOCK_RE, _THINK_BLOCK_RE, _try_json_dict

ENRICHMENT_PROMPT_VERSION = "e1"

# Enrichment needs a larger output budget than classification, which is why this is set here
# rather than left at `DeepSeekClient.chat_completion`'s 6000 default.
#
# Measured on a real 25-record Bioinformatics smoke run (2026-09-03): 12 of 25 came back as
# parse_error, and every single failing row had `output_tokens` of **exactly 6000** -- the cap --
# while the 13 successes averaged 3,566 and peaked at 5,544. The responses were being truncated
# mid-JSON, not failing on content. That is the same failure `deepseek_client.py` documents from
# the classification path, where the cap went 400 -> 2000 -> 6000 for exactly this reason.
#
# Enrichment simply emits more than classification does: six vocabulary-constrained fields plus a
# rationale, on top of flash's default thinking mode. Billing is per token actually generated, not
# per the ceiling, so a generous cap removes the truncation risk without inflating the cost of the
# large majority of calls that finish well under it.
ENRICHMENT_MAX_TOKENS = 16000

# Thinking-effort control. `deepseek-v4-flash` runs thinking mode at effort "high" by default, and
# that default is what this step's cost actually is: measured on the real Bioinformatics smoke run
# (2026-09-03), output tokens are 94% of the bill and ~98% of the output is the reasoning trace --
# the visible JSON answer is ~112 tokens out of ~5,770.
#
# The API exposes `thinking: {type, reasoning_effort}` with effort in {low, high, max}; the
# `extra_body` hook in `deepseek_client.py` is how it reaches the request body, unmodelled, so the
# finalised classification path is untouched. "none" maps to `{"type": "disabled"}`, which the
# client's own comment records as confirmed-live (reasoning_tokens -> 0).
#
# Default is None -- send nothing, keep the provider default -- so this changes no behaviour until
# a level is chosen from the paired experiment in `thinking_ablation/run_effort_ablation.py`.
# Turning thinking off entirely was already measured and REJECTED (thinking_ablation/README.md):
# it is 18x cheaper but pushes vocabulary violations from 6% to 46.5% of records, because a
# 259-term closed-vocabulary lookup is exactly what the reasoning is being spent on.
REASONING_EFFORT_LEVELS = ("none", "low", "high", "max")


def thinking_extra_body(reasoning_effort: Optional[str]) -> Optional[dict]:
    """The `extra_body` payload for a given effort level, or None to send nothing at all.

    Kept a module-level function rather than an inline dict so the exact wire shape is one
    testable thing -- the request body is the only place this can go wrong silently, and the
    client raises on a 4xx rather than ignoring an unknown key, so a wrong shape fails loudly.
    """
    if reasoning_effort is None:
        return None
    if reasoning_effort not in REASONING_EFFORT_LEVELS:
        raise ValueError(
            f"reasoning_effort must be one of {list(REASONING_EFFORT_LEVELS)}, "
            f"got {reasoning_effort!r}"
        )
    if reasoning_effort == "none":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "enabled", "reasoning_effort": reasoning_effort}}

# The five vocab-constrained fields' caps, straight from the official vocab files' own
# max_tags -- re-read at runtime, never duplicated here (single source of truth).
LIST_FIELDS = ("domain_tier1", "domain_tier2", "domain_tier3", "learning_paradigm", "model_family", "model_type")

EVENT_COLUMNS = [
    "record_id",
    *LIST_FIELDS,
    "rationale",
    "vocab_violations",
    "parse_status",  # "ok" or PARSE_ERROR -- enrichment has no single "classification" field
    "model_tier",
    "prompt_version",
    "vocab_sha256",
    "batch_id",
    "input_tokens",
    "output_tokens",
    # Of `output_tokens`, how many were the reasoning trace. Recorded from 2026-09-03: both
    # truncation incidents on this step had to be diagnosed indirectly from `output_tokens ==
    # cap`, because the API reports this and nothing was keeping it. It is the numerator of this
    # step's entire cost problem, so it belongs in the event log, not just in a one-off probe.
    "reasoning_tokens",
    # "stop" (finished normally) or "length" (hit max_tokens mid-generation -- the real cause of
    # every parse_error this step has produced). Captured on the response since the client was
    # written, discarded until now.
    "finish_reason",
    "cache_hit_tokens",
    "parse_fallback_used",
    "timestamp",
]


# ---------------------------------------------------------------------------
# Vocabulary loading + the static (cache-friendly) system message
# ---------------------------------------------------------------------------


def load_vocabularies(criteria_dir: Path) -> dict:
    """The three OFFICIAL vocab files, loaded once per run."""
    criteria_dir = Path(criteria_dir)
    return {
        "domain": json.loads((criteria_dir / "domain_vocab.json").read_text(encoding="utf-8")),
        "modelling_branch": json.loads((criteria_dir / "modelling_branch_vocab.json").read_text(encoding="utf-8")),
        "model_type_seed": json.loads((criteria_dir / "model_type_seed_vocab.json").read_text(encoding="utf-8")),
    }


def _domain_field_block(name: str, field: dict) -> str:
    labels = ", ".join(term["label"] for term in field["terms"])
    return f"{name} (choose at most {field['max_tags']}, or none if genuinely not applicable):\n{labels}"


# The three domain tiers are slices of one EDAM subtree, not three unrelated lists -- every
# tier-2 term has a tier-1 ancestor and almost every tier-3 term has a tier-2 one. The flat
# rendering below throws that structure away and asks the model to scan 259 comma-separated labels;
# the tree rendering gives it back. Two reasons to try it, both grounded in measured behaviour:
# the reasoning trace is what a flat 259-term lookup is being spent on, and wrong-TIER placement
# (a real term put in the wrong field) was 46% of all violations in the thinking-OFF ablation --
# a mistake that is structurally harder to make when the term is printed under its own parent.
_DOMAIN_TIERS = ("domain_tier1", "domain_tier2", "domain_tier3")


def _tier_of(edam_id: str, domain_fields: dict) -> Optional[str]:
    for tier in _DOMAIN_TIERS:
        if any(t["edam_id"] == edam_id for t in domain_fields[tier]["terms"]):
            return tier
    return None


def _nearest_ancestor_in(term: dict, target_tier: str, domain_fields: dict,
                         by_id: dict) -> Optional[str]:
    """Walk up `parent_ids` until a term in `target_tier` is reached. EDAM is a DAG -- 51 of the
    259 terms have more than one parent -- so the first match in document order wins, which is
    deterministic and keeps the rendering byte-stable across runs (the prefix-cache boundary)."""
    seen: set[str] = set()
    frontier = list(term.get("parent_ids") or [])
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if _tier_of(current, domain_fields) == target_tier:
            return current
        parent = by_id.get(current)
        if parent is not None:
            frontier.extend(parent.get("parent_ids") or [])
    return None


def _domain_tree_block(domain_fields: dict) -> str:
    """All three tiers as one indented tree. Every term appears exactly once -- terms whose
    ancestor is outside the vocabulary go under an explicit 'other' heading rather than being
    dropped, which `test_domain_tree_renders_every_term_exactly_once` enforces."""
    by_id = {t["edam_id"]: t for tier in _DOMAIN_TIERS for t in domain_fields[tier]["terms"]}

    children_of: dict[str, list[dict]] = {}
    orphan_t2: list[dict] = []
    for term in domain_fields["domain_tier2"]["terms"]:
        anchor = _nearest_ancestor_in(term, "domain_tier1", domain_fields, by_id)
        (children_of.setdefault(anchor, []) if anchor else orphan_t2).append(term) \
            if anchor else orphan_t2.append(term)

    grandchildren_of: dict[str, list[dict]] = {}
    orphan_t3: list[dict] = []
    for term in domain_fields["domain_tier3"]["terms"]:
        anchor = _nearest_ancestor_in(term, "domain_tier2", domain_fields, by_id)
        if anchor:
            grandchildren_of.setdefault(anchor, []).append(term)
        else:
            orphan_t3.append(term)

    lines: list[str] = []

    def _emit_tier2(term: dict) -> None:
        lines.append(f"  - {term['label']}")
        for child in grandchildren_of.get(term["edam_id"], []):
            lines.append(f"      * {child['label']}")

    for tier1 in domain_fields["domain_tier1"]["terms"]:
        lines.append(tier1["label"])
        for tier2 in children_of.get(tier1["edam_id"], []):
            _emit_tier2(tier2)
    if orphan_t2:
        lines.append("(tier-2 terms with no tier-1 parent in this vocabulary)")
        for term in orphan_t2:
            _emit_tier2(term)
    if orphan_t3:
        lines.append("(tier-3 terms with no tier-2 parent in this vocabulary)")
        for term in orphan_t3:
            lines.append(f"      * {term['label']}")

    caps = {tier: domain_fields[tier]["max_tags"] for tier in _DOMAIN_TIERS}
    header = (
        "DOMAIN VOCABULARY -- one EDAM topic tree, three tiers of depth. Unindented lines are "
        f"domain_tier1 (choose at most {caps['domain_tier1']}), '-' lines beneath them are "
        f"domain_tier2 (at most {caps['domain_tier2']}), '*' lines beneath those are domain_tier3 "
        f"(at most {caps['domain_tier3']}). A term belongs to the tier it is printed at and to no "
        "other -- never move a term into a different tier's field. Prefer terms that sit under the "
        "tier-1 domain you choose. Use [] for any tier where nothing genuinely applies."
    )
    return header + "\n\n" + "\n".join(lines)


def _branch_field_block(name: str, field: dict) -> str:
    lines = []
    for term in field["terms"]:
        definition = term.get("definition")
        lines.append(f"- {term['label']}" + (f": {definition}" if definition else ""))
    return f"{name} (choose at most {field['max_tags']}):\n" + "\n".join(lines)


def _seed_block(seed: dict) -> str:
    lines = []
    for entry in seed["terms"]:
        aliases = ", ".join(entry.get("aliases", []))
        lines.append(f"- {entry['canonical']}" + (f" (aliases: {aliases})" if aliases else ""))
    return (
        "Canonical model-type spellings (when the paper's method matches one of these, by any "
        "listed alias or spelling variant, output the canonical spelling shown; methods NOT listed "
        "here are still valid -- tag them verbatim as the paper names them):\n" + "\n".join(lines)
    )


_ENRICHMENT_PREAMBLE = (
    "You are a metadata curator enriching a biomedical-literature record that has ALREADY been "
    "confirmed to describe the application or development of an AI/ML method. Do not re-judge "
    "that decision -- your only task is to tag the paper with the controlled vocabularies below, "
    "based ONLY on the title, abstract, journal, and year given to you. You have no access to any "
    "external tool, search engine, or database."
)

_ENRICHMENT_FORMAT_SPEC = (
    "Respond with a single JSON object and nothing else, in exactly this shape:\n"
    '{"domain_tier1": [...], "domain_tier2": [...], "domain_tier3": [...], '
    '"learning_paradigm": [...], "model_family": [...], "model_type": [...], '
    '"rationale": "<at most 2 sentences>"}\n'
    "Every field is a JSON array of strings (use [] when nothing applies). For the domain, "
    "learning_paradigm, and model_family fields use ONLY terms from the vocabularies above, "
    "spelled exactly as shown, and respect each field's maximum. model_type is open: prefer the "
    "canonical spellings listed above when applicable, otherwise tag the method verbatim; list "
    "every distinct model type the paper genuinely uses or develops, with no maximum."
)


DOMAIN_RENDERINGS = ("flat", "tree")


def build_static_system_text(vocabs: dict, domain_rendering: str = "flat") -> str:
    """Byte-identical across every call in a run -- the DeepSeek prefix-cache boundary. Field
    order and rendering must stay stable; any change here (or in the vocab files) changes
    `vocab_sha256` and correctly starts a fresh batch.

    `domain_rendering` selects how the 259 domain terms are laid out: "flat" is three
    comma-separated lists (the rendering every run to date has used, and the default, so existing
    hashes are unchanged); "tree" prints them as the single EDAM subtree they actually are. The
    two are compared in `thinking_ablation/run_effort_ablation.py`."""
    if domain_rendering not in DOMAIN_RENDERINGS:
        raise ValueError(
            f"domain_rendering must be one of {list(DOMAIN_RENDERINGS)}, got {domain_rendering!r}"
        )
    domain_fields = vocabs["domain"]["fields"]
    branch_fields = vocabs["modelling_branch"]["fields"]
    if domain_rendering == "tree":
        domain_blocks = [_domain_tree_block(domain_fields)]
    else:
        domain_blocks = [
            "DOMAIN VOCABULARIES (EDAM topic branch, three depth tiers -- tag the paper's subject "
            "domain(s) at each tier where applicable):",
            _domain_field_block("domain_tier1", domain_fields["domain_tier1"]),
            _domain_field_block("domain_tier2", domain_fields["domain_tier2"]),
            _domain_field_block("domain_tier3", domain_fields["domain_tier3"]),
        ]
    blocks = [
        _ENRICHMENT_PREAMBLE,
        *domain_blocks,
        "MODELLING VOCABULARIES:",
        _branch_field_block("learning_paradigm", branch_fields["learning_paradigm"]),
        _branch_field_block("model_family", branch_fields["model_family"]),
        _seed_block(vocabs["model_type_seed"]),
        _ENRICHMENT_FORMAT_SPEC,
    ]
    return "\n\n---\n\n".join(blocks)


def vocab_sha256(static_system_text: str) -> str:
    """Same staleness role as `prompts.criteria_sha256`: every event records the hash of the exact
    vocab/prompt text it was tagged under, and resumability keys on it."""
    return hashlib.sha256(static_system_text.encode("utf-8")).hexdigest()


def build_enrichment_prompt(record, static_system_text: str) -> list[dict]:
    """Record fields read explicitly by key -- title/abstract/journal/year ONLY -- so the trial
    CSV's provenance columns (label, cross_curate_notes, ...) can never leak into the prompt even
    if a caller passes a full row. Same blinding pattern `prompts.py` documents and tests."""
    title = record.get("title") or "(no title)"
    journal = record.get("journal") or "(unknown journal)"
    year = record.get("year") or "(unknown year)"
    abstract = record.get("abstract") or "(no abstract available)"
    user = f"Title: {title}\nJournal: {journal}\nYear: {year}\n\nAbstract:\n{abstract}"
    return [
        {"role": "system", "content": static_system_text},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Parsing + validation (tolerate-and-log, never coerce, never raise)
# ---------------------------------------------------------------------------


def _clean_str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _build_lookup(vocabs: dict) -> dict:
    """Per-field {lowercased valid label -> canonical label}, plus the seed's canonical+alias map
    for model_type normalization."""
    domain_fields = vocabs["domain"]["fields"]
    branch_fields = vocabs["modelling_branch"]["fields"]
    lookup: dict = {}
    for name in ("domain_tier1", "domain_tier2", "domain_tier3"):
        lookup[name] = {t["label"].lower(): t["label"] for t in domain_fields[name]["terms"]}
    for name in ("learning_paradigm", "model_family"):
        lookup[name] = {t["label"].lower(): t["label"] for t in branch_fields[name]["terms"]}
    seed_map: dict = {}
    for entry in vocabs["model_type_seed"]["terms"]:
        seed_map[entry["canonical"].lower()] = entry["canonical"]
        for alias in entry.get("aliases", []):
            seed_map[alias.lower()] = entry["canonical"]
    lookup["model_type_seed"] = seed_map
    caps = {
        name: domain_fields[name]["max_tags"] for name in ("domain_tier1", "domain_tier2", "domain_tier3")
    }
    caps["learning_paradigm"] = branch_fields["learning_paradigm"]["max_tags"]
    caps["model_family"] = branch_fields["model_family"]["max_tags"]
    lookup["_caps"] = caps
    return lookup


def parse_enrichment(raw_content: str, lookup: dict) -> dict:
    """Returns {field: list, "rationale": str, "vocab_violations": list, "parse_status": str,
    "parse_fallback_used": bool}. Tolerate-and-log: an unknown term or an over-cap list is kept as
    returned but recorded in vocab_violations (real signal for the trial's success stats), never
    silently dropped or coerced. A response with no parsable JSON object at all is PARSE_ERROR --
    excluded from "already done" on re-run, exactly like the classify path."""
    text = (raw_content or "").strip()
    data = _try_json_dict(text)
    fallback = False
    if data is None:
        stripped = _THINK_BLOCK_RE.sub("", text).strip()
        data = _try_json_dict(stripped)
        if data is None:
            brace = _BRACE_BLOCK_RE.search(stripped)
            data = _try_json_dict(brace.group(0)) if brace else None
        fallback = True
    if data is None:
        return {
            **{field: [] for field in LIST_FIELDS},
            "rationale": text[:300],
            "vocab_violations": [],
            "parse_status": PARSE_ERROR,
            "parse_fallback_used": True,
        }

    violations: list[str] = []
    result: dict = {}

    # --- the three domain tiers, with wrong-tier repair ---------------------
    #
    # A "violation" is not one failure mode but two, and they deserve different treatment. In the
    # thinking-OFF ablation, 46% of all violations were a REAL vocabulary term placed in the wrong
    # tier field (41 of them tier-2 terms returned under domain_tier3) -- the model knew the term,
    # it just misfiled it. Those are deterministically repairable at zero cost, and the ablation
    # write-up recommended doing so twice without it being built. The other kind -- a term that is
    # in no tier at all -- is a genuine miss and is still kept and flagged, never coerced.
    #
    # A repaired term is recorded as `repaired:` rather than `unknown:` so the two stay countable
    # apart: repair must never be able to flatter a quality metric into looking better than the
    # model actually was.
    domain_tiers = ("domain_tier1", "domain_tier2", "domain_tier3")
    resolved: dict[str, list[str]] = {tier: [] for tier in domain_tiers}
    for tier in domain_tiers:
        for value in _clean_str_list(data.get(tier)):
            match = lookup[tier].get(value.lower())
            if match is not None:
                resolved[tier].append(match)
                continue
            home = next(
                (other for other in domain_tiers
                 if other != tier and lookup[other].get(value.lower()) is not None),
                None,
            )
            if home is None:
                violations.append(f"{tier}:unknown:{value}")
                resolved[tier].append(value)
                continue
            canonical = lookup[home][value.lower()]
            if canonical in resolved[home] or len(resolved[home]) >= lookup["_caps"][home]:
                # Its real tier is already full (or already has it) -- dropping it would be a
                # silent coercion, so it stays where the model put it and is flagged as unknown
                # for that field, which is exactly what it is.
                violations.append(f"{tier}:unknown:{value}")
                resolved[tier].append(value)
            else:
                violations.append(f"{tier}:repaired_to_{home}:{canonical}")
                resolved[home].append(canonical)
    for tier in domain_tiers:
        cap = lookup["_caps"][tier]
        if len(resolved[tier]) > cap:
            violations.append(f"{tier}:cap_exceeded:{len(resolved[tier])}>{cap}")
        result[tier] = resolved[tier]

    # --- the closed modelling fields, unchanged -----------------------------
    for field in ("learning_paradigm", "model_family"):
        values = _clean_str_list(data.get(field))
        canonical_values = []
        for value in values:
            match = lookup[field].get(value.lower())
            if match is None:
                violations.append(f"{field}:unknown:{value}")
                canonical_values.append(value)
            else:
                canonical_values.append(match)
        cap = lookup["_caps"][field]
        if len(canonical_values) > cap:
            violations.append(f"{field}:cap_exceeded:{len(canonical_values)}>{cap}")
        result[field] = canonical_values

    model_types = []
    for value in _clean_str_list(data.get("model_type")):
        model_types.append(lookup["model_type_seed"].get(value.lower(), value))
    result["model_type"] = model_types

    result["rationale"] = str(data.get("rationale", "")).strip()
    result["vocab_violations"] = violations
    result["parse_status"] = "ok"
    result["parse_fallback_used"] = fallback
    return result


# ---------------------------------------------------------------------------
# The concurrent, resumable enrichment loop
# ---------------------------------------------------------------------------


def _already_enriched_ids(
    events: pd.DataFrame, tier: str, vocab_hash: str, retry_truncated: bool = False
) -> set:
    """A PARSE_ERROR row does not count as done -- re-running the same command retries it for
    free, same proven mechanism as `runner._already_classified_ids`.

    **Except a truncation, which is not free.** A record whose response hit `max_tokens` comes back
    `finish_reason == "length"`, and retrying it at the same cap re-truncates: the model wants more
    tokens than it is allowed, deterministically, so every retry burns another full
    `ENRICHMENT_MAX_TOKENS` of output and fails identically. Measured on the 100-record ablation
    (2026-09-03): 3 of 100 records at the default effort truncate at 16,000 tokens, and a resumed
    run spent ten minutes and ~48,000 output tokens re-failing on exactly those three.

    So a truncated record is treated as done-for-now and skipped. It is still visible -- it carries
    `parse_status=parse_error` and `finish_reason=length` in the log, and the run summary counts it
    -- and `retry_truncated=True` forces a retry, which is worth doing only after raising
    `ENRICHMENT_MAX_TOKENS`, because nothing else about the call will have changed.
    """
    if events.empty:
        return set()
    matches = events[
        (events["model_tier"] == tier)
        & (events["prompt_version"] == ENRICHMENT_PROMPT_VERSION)
        & (events["vocab_sha256"] == vocab_hash)
    ]
    done = set(matches[matches["parse_status"] != PARSE_ERROR]["record_id"])
    if not retry_truncated and "finish_reason" in matches.columns:
        done |= set(matches[matches["finish_reason"] == "length"]["record_id"])
    return done


def enrich_records(
    records_df: pd.DataFrame,
    tier: str,
    client: DeepSeekClient,
    static_system_text: str,
    lookup: dict,
    batch_id: str,
    existing_events: Optional[pd.DataFrame] = None,
    max_workers: int = 100,
    reasoning_effort: Optional[str] = None,
    retry_truncated: bool = False,
) -> Iterator[dict]:
    """Yields one event dict per newly-enriched record as soon as it completes (never in input
    order -- up to `max_workers` calls race concurrently; callers must not assume order). The live
    progress bar shows a running tally: ok / parse errors / vocab violations / prefix-cache hit
    rate -- the same at-a-glance readout style as the classify runs, with the cache column added
    because it is this design's whole token-efficiency claim, made observable.

    If one call ultimately fails (after the client's own network retries), every other in-flight
    call still completes and is still yielded -- the first failure re-raises only after the pool
    drains, so a single bad call loses at most itself (same contract as `classify_records`)."""
    existing = existing_events if existing_events is not None else pd.DataFrame(columns=EVENT_COLUMNS)
    vocab_hash = vocab_sha256(static_system_text)
    done_ids = _already_enriched_ids(existing, tier, vocab_hash, retry_truncated)
    to_do = records_df[~records_df["record_id"].isin(done_ids)]
    records = [record for _, record in to_do.iterrows()]

    extra_body = thinking_extra_body(reasoning_effort)

    def _enrich_one(record) -> dict:
        response = client.chat_completion(
            build_enrichment_prompt(record, static_system_text),
            tier=tier,
            max_tokens=ENRICHMENT_MAX_TOKENS,
            extra_body=extra_body,
        )
        parsed = parse_enrichment(response.content, lookup)
        usage = response.raw.get("usage", {}) if isinstance(response.raw, dict) else {}
        details = usage.get("completion_tokens_details") or {}
        return {
            "record_id": record["record_id"],
            **{field: json.dumps(parsed[field]) for field in LIST_FIELDS},
            "rationale": parsed["rationale"],
            "vocab_violations": json.dumps(parsed["vocab_violations"]),
            "parse_status": parsed["parse_status"],
            "model_tier": tier,
            "prompt_version": ENRICHMENT_PROMPT_VERSION,
            "vocab_sha256": vocab_hash,
            "batch_id": batch_id,
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.completion_tokens,
            "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
            "finish_reason": response.finish_reason or "",
            "cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
            "parse_fallback_used": str(parsed["parse_fallback_used"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    n_ok = n_err = n_violations = 0
    tokens_in = tokens_cached = tokens_out = tokens_reasoning = 0
    first_exception: Optional[BaseException] = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_enrich_one, record) for record in records]
        with tqdm(total=len(futures), desc=f"enrich:{tier}", unit="rec") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    event = future.result()
                except Exception as exc:  # noqa: BLE001 -- drained + re-raised below
                    if first_exception is None:
                        first_exception = exc
                    pbar.update(1)
                    continue
                if event["parse_status"] == PARSE_ERROR:
                    n_err += 1
                else:
                    n_ok += 1
                n_violations += len(json.loads(event["vocab_violations"]))
                tokens_in += event["input_tokens"]
                tokens_cached += event["cache_hit_tokens"]
                tokens_out += event["output_tokens"]
                tokens_reasoning += event["reasoning_tokens"]
                cache_pct = (tokens_cached / tokens_in * 100) if tokens_in else 0.0
                # `think` is the share of output tokens spent reasoning -- this step's cost driver,
                # made visible while a run is happening rather than only in the post-hoc analysis.
                think_pct = (tokens_reasoning / tokens_out * 100) if tokens_out else 0.0
                pbar.set_postfix(ok=n_ok, err=n_err, viol=n_violations, cache=f"{cache_pct:.0f}%",
                                 think=f"{think_pct:.0f}%")
                pbar.update(1)
                yield event

    if first_exception is not None:
        raise first_exception
