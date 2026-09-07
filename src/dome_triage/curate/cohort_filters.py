"""Builds the pre-Streamlit-app manual curation cohort for Step 19's re-review pass, and draws a
seeded stratified sample from it. Mirrors `bulk_pool.py`'s role: a dedicated loader producing a
DataFrame shaped exactly like `CurationSession` expects, with no join back to
`canonical_dataset.csv` needed beyond the filter applied here.

Deliberately does NOT import from `pipeline/steps.py` (pulls in `torch`), matching every other
module in this package -- see `bulk_scores.py`'s module docstring for the same reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dome_triage.sampling.stratified import build_strata, stratified_sample

_REGISTRY_CONFIRMED = "registry_confirmed"


def _has_registry_source(sources_cell: object) -> bool:
    """True if any contributing source of this record carries `registry_confirmed` confidence --
    excludes the DOME registry gold PDF pulls (`dome_registry_231_gold`/`dome_registry_222_gold`)
    and the DOME API dump (`ebi_search_dome_api`) from the "manual, pre-app" cohort, even on the
    rare record whose overall `label_confidence` resolved to `human_curated` because it *also*
    carries a `human_curated` source (`merge_label` picks the strongest tier across all of a
    record's sources -- see `dedupe/conflicts.py::merge_label`). Confirmed via direct query against
    the live dataset: exactly 1 such record exists (combines a registry gold source with a
    dome_top_curate source); without this check it would incorrectly land in this cohort."""
    if not isinstance(sources_cell, str) or not sources_cell.strip():
        return False
    try:
        entries = json.loads(sources_cell)
    except json.JSONDecodeError:
        return False
    return any(
        isinstance(e, dict) and e.get("source_label_confidence") == _REGISTRY_CONFIRMED for e in entries
    )


def build_original_cohort(dataset: pd.DataFrame, events_path: Path) -> pd.DataFrame:
    """The cohort curated before this Curate app existed (`DOME_Top_Curate`, `copilot_1012`,
    etc. -- confirmed 3,356 records live: 1,449 positive / 1,907 negative): `label_confidence ==
    "human_curated"`, not already reviewed via this app (`record_id` not in `curation_events.csv`),
    and carrying no `registry_confirmed` source. Scoped to positive/negative only -- Step 19's
    stated goal is re-checking pos/neg calls, and this cohort has effectively no
    skipped/undeterminable rows to re-check anyway."""
    events_path = Path(events_path)
    reviewed_ids: set[str] = set()
    if events_path.exists():
        reviewed_ids = set(pd.read_csv(events_path, usecols=["record_id"], dtype=str)["record_id"])

    cohort = dataset[
        (dataset["label_confidence"] == "human_curated")
        & (~dataset["record_id"].isin(reviewed_ids))
        & (dataset["label"].isin(["positive", "negative"]))
        & (~dataset["sources"].apply(_has_registry_source))
    ]
    return cohort.reset_index(drop=True)


def sample_cohort(
    cohort: pd.DataFrame, sample_size: int = 250, random_state: int = 42, exclude_ids=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draws a fresh, seeded, stratified sample of approximately `sample_size` records from
    `cohort` -- the same `build_strata`/`stratified_sample` pair Step 13 uses for the original
    curation queue (score band x journal bucket x year bucket), so this re-review sample is
    diverse across the same dimensions instead of an arbitrary head/tail slice. Score-band
    stratification is skipped gracefully (falls back to journal/year only) when `cohort` has no
    `bulk_match_score` column, or none of its rows are actually scored -- true for roughly 11% of
    this cohort, which was never matched by the bulk AI/ML search that produced BM25 scores.

    `stratified_sample` caps a fixed count *per stratum combination*, not a global total, so the
    per-stratum cap here is derived from `sample_size / (number of stratum combinations present)`
    to land close to the requested total -- exact only when every stratum has at least that many
    records available, same approximation the rest of this codebase's stratified sampling accepts
    (see `stratum_report`'s `available` vs `sampled` columns for the real achieved counts).

    `exclude_ids`, if given, removes already-sampled-this-pass record_ids before drawing -- so
    clicking "draw a fresh sample" again during the same review round doesn't hand back records
    already decided under a previous draw in that round.

    Returns `(sampled_df, stratum_report_df)` -- the report is empty when `cohort` is small enough
    that no sampling was needed (the whole cohort is returned as-is in that case)."""
    if exclude_ids:
        cohort = cohort[~cohort["record_id"].isin(set(exclude_ids))]
    if cohort.empty or len(cohort) <= sample_size:
        return cohort.reset_index(drop=True), pd.DataFrame()

    has_scores = "bulk_match_score" in cohort.columns and cohort["bulk_match_score"].notna().any()
    strata_df = build_strata(cohort, score_col="bulk_match_score" if has_scores else None)
    strata_cols = (["match_score_band__bulk_match_score"] if has_scores else []) + [
        "journal_bucket",
        "year_bucket",
    ]

    n_strata = strata_df.groupby(strata_cols, dropna=False).ngroups
    cap_per_stratum = max(1, round(sample_size / n_strata)) if n_strata else sample_size

    sampled, report = stratified_sample(strata_df, strata_cols, cap_per_stratum, random_state=random_state)
    return sampled.reset_index(drop=True), report


_NON_METHODS_PUBTYPE_NOTE = "non_methods_pubtype"


def load_review_term_list(exclusionary_lexicon_path: Path) -> list[str]:
    """Reads the review/non-methods keyword list already curated during Step 9-11's lexicon work
    (`keyword_lexicon_exclusionary.csv`'s `notes == "non_methods_pubtype"` rows -- "systematic
    review", "meta-analysis", "narrative review", "editorial", "commentary", "case report",
    "letter to the editor", "correspondence", "news", "opinion", "perspective", "viewpoint",
    "survey"; 13 terms confirmed live), reused here rather than duplicated -- this is the "big
    list" of review/non-methods signal terms the BM25 exclusionary scorer already uses, repurposed
    for a direct keyword match instead of a weighted score. Returns `[]` gracefully if the file
    doesn't exist yet (e.g. Step 9 hasn't run) or has no `notes` column."""
    exclusionary_lexicon_path = Path(exclusionary_lexicon_path)
    if not exclusionary_lexicon_path.exists():
        return []
    df = pd.read_csv(exclusionary_lexicon_path)
    if "notes" not in df.columns or "term" not in df.columns:
        return []
    return df.loc[df["notes"] == _NON_METHODS_PUBTYPE_NOTE, "term"].dropna().tolist()


_DISAGREEMENT_TRUSTED_LABEL_CONFIDENCE = ("human_curated", "registry_confirmed")


def build_disagreement_queue(
    dataset: pd.DataFrame, llm_events: pd.DataFrame, tier: str | None = None, exclude_ids: set | None = None
) -> pd.DataFrame:
    """Step 20: joins DeepSeek's latest PRIMARY-mode classification per (record_id, tier) onto the
    current human label (trusted-confidence positive/negative rows only -- same tier
    `curate/state.py::CurationSession` already treats as settled ground truth), keeping rows where
    the two differ. An LLM "undeterminable" against a human positive/negative call counts as a
    disagreement too, surfaced for review rather than silently dropped -- only records with a
    PARSE_ERROR classification are excluded (there's no real verdict to compare).

    `tier=None` (default) checks every tier present in `llm_events` -- a record disagreeing under
    more than one tier appears once PER disagreeing tier, distinguished by the returned
    `llm_tier` column, so the review page can show each tier's independent verdict side by side
    rather than collapsing them into one ambiguous row. Pass a specific tier to restrict to just
    that one.

    `exclude_ids` (typically every record_id already in `cross_curate_resolution_events.csv`) drops
    already-resolved records from the returned queue. Real, confirmed gap this fixes: a record
    whose Cross Curate Resolve final decision UPHELD the original label (the human explicitly
    agreed with the human, not DeepSeek) still has `label != llm_classification` by construction --
    without this exclusion, every such record would resurface here as if it were a brand-new,
    never-reviewed disagreement, on every future call over a dataset/events file that still
    contains its now-stale classification event."""
    empty_columns = list(dataset.columns) + ["llm_classification", "llm_rationale", "llm_tier"]
    if llm_events.empty:
        return pd.DataFrame(columns=empty_columns)

    primary = llm_events[(llm_events["mode"] == "primary") & (llm_events["classification"] != "parse_error")]
    if tier is not None:
        primary = primary[primary["model_tier"] == tier]
    if primary.empty:
        return pd.DataFrame(columns=empty_columns)

    latest = (
        primary.sort_values("timestamp")
        .groupby(["record_id", "model_tier"])
        .last()
        .reset_index()[["record_id", "model_tier", "classification", "rationale"]]
    )

    trusted = dataset[
        dataset["label_confidence"].isin(_DISAGREEMENT_TRUSTED_LABEL_CONFIDENCE)
        & dataset["label"].isin(["positive", "negative"])
    ]

    merged = trusted.merge(latest, on="record_id", how="inner")
    disagreements = merged[merged["classification"] != merged["label"]].copy()
    if exclude_ids:
        disagreements = disagreements[~disagreements["record_id"].isin(set(exclude_ids))]
    disagreements = disagreements.rename(
        columns={"classification": "llm_classification", "rationale": "llm_rationale", "model_tier": "llm_tier"}
    )
    return disagreements.reset_index(drop=True)


def annotate_review_term_match(dataset: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Adds a `matches_review_term` boolean column: does this record's title+abstract contain any
    of `terms` (case-insensitive substring match)? Direct keyword matching, not a BM25 score --
    the point is a simple, explainable "this record's own text literally says 'systematic
    review'" signal a curator can use to down-weight/skip re-verifying an already-likely-correct
    negative, not a weighted relevance score. Returns a copy; `dataset` itself is never mutated.
    All-False when `terms` is empty (e.g. the exclusionary lexicon hasn't been built yet)."""
    dataset = dataset.copy()
    if not terms:
        dataset["matches_review_term"] = False
        return dataset
    combined_text = (dataset["title"].fillna("") + " " + dataset["abstract"].fillna("")).str.lower()
    dataset["matches_review_term"] = combined_text.apply(
        lambda text: any(term.lower() in text for term in terms)
    )
    return dataset
