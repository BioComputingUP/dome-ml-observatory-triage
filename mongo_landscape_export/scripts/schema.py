"""Step 24 phase 1, Step 4: converts one CSV row into one grouped, properly-typed MongoDB
document.

Two public builders, deliberately sharing every group helper so their shapes cannot drift:

- `build_document(row)` -- the LLM landscape path, one row of
  `ai_ml_landscape_classified_usable.csv` (Step 23a's classified output). Provenance `"llm"`.
- `build_curated_document(row)` -- the human/registry path (Step 24 phase 2), one row of the
  normalised curated set built by `build_curated_documents.py`. Provenance `"human_curated"` or
  `"registry_confirmed"`.

`test_schema.py::test_both_builders_produce_the_same_shape` is what keeps that promise; if you add
a field to one, the test fails until it is in both.

Schema designed together with Gavin (2026-08-28) -- six groups:

- `identifiers`: pmid / pmcid / doi, plus dome_registry / bioai_repo / huggingface / kaggle /
  zenodo -- external registry/repo cross-references, not in the source CSV at all, always null
  for now (reserved for a future linking pass)
- `publication_metadata`: title / abstract / authors / year / journal / citation_count /
  citation_count_updated / citation_source
- `source`: abstract_source / metadata_repair_sources / decision_provenance, plus a nested
  `access` group (open_access / license / fulltext_available)
- `content_filters`: mesh_headings / pub_types / keywords_author, plus six fields reserved for
  Step 23b's enrichment pass (domain_tier1-3 / learning_paradigm / model_family / model_type) --
  present but null/empty until an enrichment run actually populates them
- `llm_classification` / `llm_enrichment`: deliberately PARALLEL groups (same core field shape:
  provider/model_tier/model_id/mode/rationale/prompt_version/ruleset_sha256/batch_id/timestamp),
  renamed from "deepseek_processed" to be provider-generic. `llm_enrichment` is fully fleshed out
  (every field present, all null) rather than a single null placeholder, so no schema migration is
  needed once an enrichment run lands.

`match_metadata` and `fulltext_source_root` were dropped entirely upstream (Step 3c, confirmed
100% blank) -- no placeholder for them here at all.

Every document also carries a top-level `schema_version` (see `SCHEMA_VERSION` below) -- it
describes the shape of the document itself, not the paper's content or any processing run, so it
doesn't belong inside any of the content groups above; it sits at the top level next to `_id`
instead. Bump it whenever this module's document shape changes, so a future migration can query
`{schema_version: "1.1.0"}` to find documents built under an older shape. The checked-in reference
copy of an empty document (all fields present, no real data) lives at
`../schema/ai_ml_landscape.schema.json` -- regenerate it with `write_schema_template.py` after any
shape change so it never drifts from what this module actually produces.

Self-contained -- no `dome_triage` import, matching this folder's established "runs outside
Docker" pattern. `TIER_MODEL_IDS` below is a deliberate, documented mirror of
`src/dome_triage/llm_classify/deepseek_client.py:33-36`, not a new data source: every row in the
landscape file was classified by DeepSeek's "flash" tier, which that module maps to the API model
id `deepseek-v4-flash`. The more precise dated snapshot (e.g. `-0731`) is not recoverable -- it was
never persisted per-row anywhere in the pipeline.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Optional

# Mirrors deepseek_client.py's TIER_MODEL_IDS -- see module docstring.
TIER_MODEL_IDS = {"flash": "deepseek-v4-flash", "pro": "deepseek-v4-pro"}

# Bump on any change to build_document()'s output shape -- see module docstring.
# 1.1.0 (2026-08-28): added identifiers.dome_registry/bioai_repo/huggingface/kaggle/zenodo
# (additive, always null for now -- no existing field changed or removed).
# 1.2.0 (2026-09-03): added source.decision_provenance (so a human/registry decision is no longer
# indistinguishable from a machine one), publication_metadata.citation_count_updated and
# .citation_source (so a populated citation_count says WHEN and FROM WHERE), and started decoding
# HTML entities in title/abstract. All additive -- no existing field changed or removed, and
# `llm_classification.classification` deliberately stays the single queryable classification field
# so the `positives_text` partial index and observatory-ws's canUseTextIndex guard are untouched.
SCHEMA_VERSION = "1.2.0"

# The three values source.decision_provenance can take. "llm" is every document Step 23a produced;
# the other two come from canonical_dataset.csv's `label_confidence`. Deliberately a flat
# discriminator rather than a second classification field -- see FINALISATION_ROADMAP.md F1.
PROVENANCE_LLM = "llm"
PROVENANCE_HUMAN = "human_curated"
PROVENANCE_REGISTRY = "registry_confirmed"
VALID_PROVENANCE = frozenset({PROVENANCE_LLM, PROVENANCE_HUMAN, PROVENANCE_REGISTRY})

CITATION_SOURCE_EPMC = "europepmc"

# `classification`'s permitted values. There is deliberately no "skipped": a curator declining to
# judge is not a verdict, and the 7 canonical rows carrying it are excluded from the merge rather
# than given a value the schema does not have (FINALISATION_ROADMAP.md F1, decided 2026-09-03).
VALID_CLASSIFICATIONS = frozenset({"positive", "negative", "undeterminable"})

_TRUE_STRINGS = {"true", "y", "yes"}
_FALSE_STRINGS = {"false", "n", "no"}

# A named or numeric HTML entity. Used only to decide whether an unescape pass is worth attempting.
_ENTITY_RE = re.compile(r"&(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[xX][0-9A-Fa-f]{1,6});")
# Measured on the real 827,061-row corpus (2026-09-03): every affected `title` is stable after
# exactly ONE unescape pass; 49 of 300,000 `abstract` values need TWO (they were encoded twice
# upstream). 3 is a real safety ceiling, not a guess -- entity decoding strictly shortens the
# string, so the loop terminates on its own; the cap only bounds a pathological input.
_MAX_UNESCAPE_PASSES = 3


def _none_if_blank(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _decode_entities(value: Optional[str]) -> Optional[str]:
    """Repairs HTML-entity-encoded text, repeatedly until stable.

    Real measured shape of the problem (full corpus, 2026-09-03) -- this is why the repair belongs
    here and not in the UI:

    - `title`: 1.85% carry entities (~15.3k records), e.g.
      `Detection of stipe rot disease in &lt;i&gt;Morchella sextelata&lt;/i&gt;`, which renders as
      literal markup in the browser. All stable after one pass.
    - `abstract`: 0.36% carry entities (~2.9k records), and ~135 are encoded TWICE, e.g.
      `dataset size (ranging from &amp;lt;50 to &amp;gt;25,000 subjects)`.
    - `authors`, `journal`, `rationale`: 0.000% -- measured, not assumed. Not decoded, so a future
      reader knows the absence is deliberate.

    The twice-encoded cases are the interesting ones and they settle the design question: they are
    `<` and `>` used as *comparison operators* ("ranging from <50 to >25,000", "Sharpe > 1.5"), so
    decoding all the way to a stable string reproduces exactly what the author wrote. A single
    fixed pass would leave those 135 abstracts displaying `&lt;50`.

    A UI-side decode was explicitly rejected in ../dome-ml-observatory/ROADMAP.md: 1.48% of titles
    already contain *raw* `<i>`/`<sub>` markup, so an entity decode in the view layer would have to
    guess which angle brackets were markup and which were arithmetic. Decoding at ingestion instead
    normalises the encoded group into the same form the raw group already has.
    """
    if value is None:
        return None
    for _ in range(_MAX_UNESCAPE_PASSES):
        if not _ENTITY_RE.search(value):
            break
        decoded = html.unescape(value)
        if decoded == value:
            break
        value = decoded
    return value


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    return None


def _parse_year(value: Optional[str]) -> Optional[int]:
    """Source CSV stores year as a float-string, e.g. "2025.0"."""
    value = _none_if_blank(value)
    if value is None:
        return None
    return int(float(value))


def _parse_int(value: Optional[str]) -> Optional[int]:
    value = _none_if_blank(value)
    if value is None:
        return None
    return int(float(value))


def _parse_json_list(value: Optional[str]) -> list:
    value = _none_if_blank(value)
    if value is None:
        return []
    return json.loads(value)


def _resolve_open_access(is_open_access: str, license_checked: str, epmc_is_open_access: str) -> Optional[bool]:
    """EPMC's freshly-fetched flag wins when a real lookup happened (confirmed with Gavin,
    2026-08-28 -- 395/738,198 rows disagreed with the original pipeline value, and the EPMC value
    is treated as canonical). Falls back to the original `is_open_access` column when there was no
    license lookup at all (no pmid, or a genuine EPMC miss) or EPMC returned no flag."""
    if license_checked.strip() == "True":
        resolved = _parse_bool(epmc_is_open_access)
        if resolved is not None:
            return resolved
    return _parse_bool(is_open_access)


def _resolve_license(license_value: str, license_checked: str) -> Optional[str]:
    """"" (empty string) means "looked up, EPMC disclosed none" -- distinct from None, which means
    "never looked up" (no pmid, or a genuine EPMC miss -- see join_license.py)."""
    if license_checked.strip() == "True":
        return license_value  # may legitimately be "" -- looked up, none disclosed
    return None


# ---------------------------------------------------------------------------
# Group builders -- shared by both public builders so their shapes cannot drift.
# ---------------------------------------------------------------------------


def _identifiers(row: dict[str, str]) -> dict[str, Any]:
    return {
        "pmid": _none_if_blank(row["pmid"]),
        "pmcid": _none_if_blank(row["pmcid"]),
        "doi": _none_if_blank(row["doi"]),
        # External registry/repo cross-references -- not in the source CSV at all yet, always
        # null for now. Reserved so a future linking pass (e.g. DOME Registry submission
        # status, or scraping paper text for repo links) has a home with no schema migration.
        "dome_registry": None,
        "bioai_repo": None,
        "huggingface": None,
        "kaggle": None,
        "zenodo": None,
    }


def _publication_metadata(row: dict[str, str]) -> dict[str, Any]:
    """`citation_count_updated` / `citation_source` are read with `.get()` on purpose: the citation
    fetch is a separate staging stage (`join_citations.py`), so a document built before that stage
    has legitimately never had a count looked up. All three null together carries exactly that
    meaning -- the same "null means never looked up" convention `_resolve_license` uses."""
    return {
        "title": _decode_entities(_none_if_blank(row["title"])),
        "abstract": _decode_entities(_none_if_blank(row["abstract"])),
        "authors": _none_if_blank(row["authors"]),
        "year": _parse_year(row["year"]),
        "journal": _none_if_blank(row["journal"]),
        "citation_count": _parse_int(row["citation_count"]),
        "citation_count_updated": _none_if_blank(row.get("citation_count_updated")),
        "citation_source": _none_if_blank(row.get("citation_source")),
    }


def _source(row: dict[str, str], decision_provenance: str) -> dict[str, Any]:
    if decision_provenance not in VALID_PROVENANCE:
        raise ValueError(
            f"decision_provenance must be one of {sorted(VALID_PROVENANCE)}, got {decision_provenance!r}"
        )
    return {
        "abstract_source": _none_if_blank(row["abstract_source"]),
        "metadata_repair_sources": _none_if_blank(row["metadata_repair_sources"]),
        # Who decided this record's classification. Never null -- a document with no answer here
        # would be exactly the ambiguity this field was added to remove.
        "decision_provenance": decision_provenance,
        "access": {
            "open_access": _resolve_open_access(
                row["is_open_access"], row["license_checked"], row["epmc_is_open_access"]
            ),
            "license": _resolve_license(row["license"], row["license_checked"]),
            "fulltext_available": _parse_bool(row["fulltext_available"]),
        },
    }


def _content_filters(row: dict[str, str]) -> dict[str, Any]:
    return {
        "mesh_headings": _parse_json_list(row["mesh_headings"]),
        "pub_types": _parse_json_list(row["pub_types"]),
        "keywords_author": _parse_json_list(row["keywords_author"]),
        # Reserved for the enrichment pass (Step 23b) -- see module docstring.
        "domain_tier1": None,
        "domain_tier2": [],
        "domain_tier3": [],
        "learning_paradigm": [],
        "model_family": [],
        "model_type": [],
    }


def _empty_enrichment() -> dict[str, Any]:
    """Parallel to llm_classification, fully fleshed out (not a single null) -- see module
    docstring. All null until an enrichment run actually populates it."""
    return {
        "provider": None,
        "model_tier": None,
        "model_id": None,
        "mode": None,
        "rationale": None,
        "prompt_version": None,
        "ruleset_sha256": None,
        "batch_id": None,
        "timestamp": None,
        "vocab_violations": None,
        "parse_status": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_hit_tokens": None,
        "parse_fallback_used": None,
    }


def _validate_classification(value: Optional[str]) -> Optional[str]:
    if value is not None and value not in VALID_CLASSIFICATIONS:
        raise ValueError(
            f"classification must be one of {sorted(VALID_CLASSIFICATIONS)} or blank, got "
            f"{value!r} -- 'skipped' in particular has no schema value and must be filtered "
            f"upstream, not coerced here"
        )
    return value


def _llm_classification(row: dict[str, str]) -> dict[str, Any]:
    model_tier = _none_if_blank(row["model_tier"])
    return {
        "provider": "deepseek",
        "model_tier": model_tier,
        "model_id": TIER_MODEL_IDS.get(model_tier),
        "mode": _none_if_blank(row["mode"]),
        "classification": _validate_classification(_none_if_blank(row["classification"])),
        "rationale": _none_if_blank(row["rationale"]),
        "prompt_version": _none_if_blank(row["prompt_version"]),
        "ruleset_sha256": _none_if_blank(row["criteria_sha256"]),
        "batch_id": _none_if_blank(row["batch_id"]),
        "timestamp": _none_if_blank(row["timestamp"]),
    }


def _curated_classification(row: dict[str, str]) -> dict[str, Any]:
    """A human or registry decision, written into the same group and the same `classification`
    field every query, facet and index already reads.

    The six model-describing fields are null because no model was involved -- and
    `source.decision_provenance` is what actually *says* "human", since a null provider only fails
    to say "machine". Nothing here invents a rationale: `curation_rationale` is assembled by
    `build_curated_documents.py` from the record's real `sources` list and `cross_curate_notes`.
    """
    return {
        "provider": None,
        "model_tier": None,
        "model_id": None,
        "mode": None,
        "classification": _validate_classification(_none_if_blank(row["label"])),
        "rationale": _none_if_blank(row["curation_rationale"]),
        "prompt_version": None,
        "ruleset_sha256": None,
        "batch_id": _none_if_blank(row["curation_batch_id"]),
        "timestamp": _none_if_blank(row["curation_timestamp"]),
    }


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_document(row: dict[str, str]) -> dict[str, Any]:
    """`row` is one dict from `csv.DictReader` over `ai_ml_landscape_classified_usable.csv` (all
    string values, `pid` already present as a column -- see add_pid_column.py). Pure, no I/O."""
    return {
        "_id": row["pid"],
        "schema_version": SCHEMA_VERSION,
        "identifiers": _identifiers(row),
        "publication_metadata": _publication_metadata(row),
        "source": _source(row, PROVENANCE_LLM),
        "content_filters": _content_filters(row),
        "llm_classification": _llm_classification(row),
        "llm_enrichment": _empty_enrichment(),
    }


def build_curated_document(row: dict[str, str]) -> dict[str, Any]:
    """`row` is one dict from the normalised curated CSV built by `build_curated_documents.py`:
    the landscape file's column names, plus `label`, `label_confidence`, `curation_rationale`,
    `curation_timestamp` and `curation_batch_id`. Pure, no I/O.

    `label_confidence` maps 1:1 onto `source.decision_provenance`; anything else raises, because a
    `heuristic_candidate` row has no business in the published corpus (it was fetched with the
    structural inverse of the AI/ML query -- FINALISATION_ROADMAP.md §2)."""
    confidence = _none_if_blank(row["label_confidence"])
    if confidence not in (PROVENANCE_HUMAN, PROVENANCE_REGISTRY):
        raise ValueError(
            f"label_confidence must be {PROVENANCE_HUMAN!r} or {PROVENANCE_REGISTRY!r} to be "
            f"published, got {confidence!r}"
        )
    return {
        "_id": row["pid"],
        "schema_version": SCHEMA_VERSION,
        "identifiers": _identifiers(row),
        "publication_metadata": _publication_metadata(row),
        "source": _source(row, confidence),
        "content_filters": _content_filters(row),
        "llm_classification": _curated_classification(row),
        "llm_enrichment": _empty_enrichment(),
    }
