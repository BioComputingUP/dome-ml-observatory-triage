"""Shared step functions. Every CLI subcommand in cli.py and `dome-triage pipeline run` call the
SAME functions defined here -- there is no separate workflow-engine orchestration, just the
STEP_FUNCS dict below called in sequence (see AGENTS.md). Every step calls `finish_step(...)`
before returning -- no generated file without a provenance entry (AGENTS.md rule)."""

from __future__ import annotations

import contextlib
import csv
import fcntl
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dome_triage.config import PipelineConfig, resolve_path
from dome_triage.curate.bulk_scores import annotate_bulk_scores, load_bulk_score_lookup, load_youden_threshold
from dome_triage.curate.review_detector import annotate_non_methods
from dome_triage.curate.state import backup_file, flag_likely_reviews, materialize_cross_curate_resolutions
from dome_triage.curate.term_review_state import materialize_term_events
from dome_triage.dedupe.consolidate import conflicts_dataframe, consolidate, to_dataframe
from dome_triage.dedupe.keys import record_id_from_ids
from dome_triage.fulltext.manifest import build_manifest
from dome_triage.ingest.bulk_match import (
    core_result_to_raw_record,
    count_ai_ml_breakdown,
    fetch_ai_ml_range,
    load_bulk_match_year,
)
from dome_triage.ingest.clear_negative_sampler import (
    fetch_clear_negatives,
    fetch_filtered_clear_negatives,
    select_diversified_pool,
    select_strong_negatives,
)
from dome_triage.ingest.enrich import enrich_missing_canonical_metadata, enrich_missing_metadata
from dome_triage.ingest.epmc_client import EpmcClient
from dome_triage.ingest import metadata_repair
from dome_triage.ingest.id_mapping import clean_doi, clean_pmcid
from dome_triage.ingest.source_loaders import (
    dataframe_to_raw_records,
    load_all_sources,
    raw_records_to_dataframe,
)
from dome_triage.keywords.curated_terms import (
    ADDED_NEGATIVE_TERMS,
    ADDED_POSITIVE_TERMS,
    PROTECTED_UNIGRAMS,
)
from dome_triage.keywords.keybert_extract import extract_keybert_terms
from dome_triage.keywords.lexicon import build_candidate_lexicon, lexicon_stats, load_seed_terms
from dome_triage.keywords.lexicon_cleanup import clean_lexicon
from dome_triage.keywords.scoring import SCORERS, WeightedSumScorer, load_lexicon_terms_and_weights
from dome_triage.keywords.scoring_bakeoff import run_bakeoff
from dome_triage.keywords.tfidf_extract import extract_tfidf_terms
from dome_triage.llm_classify import budget as llm_budget
from dome_triage.llm_classify import cost_estimator as llm_cost
from dome_triage.llm_classify import enrichment as llm_enrichment
from dome_triage.llm_classify import landscape_consolidate
from dome_triage.llm_classify import runner as llm_runner
from dome_triage.llm_classify.deepseek_client import DeepSeekClient
from dome_triage.llm_classify.prompts import build_prompt, criteria_sha256, load_criteria_text
from dome_triage.llm_classify.sampling import (
    draw_blind_sample,
    select_bulk_pool_excluding_curated,
    select_staged_file,
    select_full_population,
)
from dome_triage.provenance import _git_commit, finish_step
from dome_triage.reporting import agreement as agreement_reporting
from dome_triage.reporting import landscape_profile as landscape_reporting
from dome_triage.reporting import enrichment_profile as enrichment_reporting
from dome_triage.reporting import run_comparison
from dome_triage.reporting.dataset_profile import (
    plot_bm25_score_distribution,
    plot_bm25_youden_performance,
    plot_journal_diversity,
    plot_label_overview,
    plot_label_vs_review_flag_breakdown,
    plot_provenance_class_breakdown,
    plot_review_flag_term_frequency,
    plot_second_review_bm25_confusion,
    plot_second_review_bm25_score_distribution,
    plot_second_review_decision_breakdown,
    plot_year_coverage_vs_bulk_pool,
    plot_year_coverage_vs_bulk_pool_linear,
    plot_year_distribution,
)
from dome_triage.sampling.stratified import build_strata, stratified_sample
from dome_triage.schema import RawRecord


def step_ingest_load_sources(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    records, unresolved = load_all_sources(cfg.sources)
    raw_records_to_dataframe(records).to_csv(cfg.path("raw_records"), index=False)
    if unresolved:
        pd.DataFrame(unresolved).to_csv(cfg.path("unresolved_needs_id_lookup"), index=False)

    outputs = [cfg.path("raw_records")]
    if unresolved:
        outputs.append(cfg.path("unresolved_needs_id_lookup"))
    finish_step(
        "ingest.load-sources",
        inputs=[],
        outputs=outputs,
        params={"n_sources": len(cfg.sources["label_sources"])},
        notes=f"{len(records)} raw records, {len(unresolved)} unresolved",
        started_at=started_at,
    )


def step_ingest_enrich_metadata(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    df = pd.read_csv(cfg.path("raw_records"), dtype=str)
    records = dataframe_to_raw_records(df)

    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        enriched = enrich_missing_metadata(records, client)
    finally:
        client.close()

    raw_records_to_dataframe(enriched).to_csv(cfg.path("raw_records_enriched"), index=False)
    finish_step(
        "ingest.enrich-metadata",
        inputs=[cfg.path("raw_records")],
        outputs=[cfg.path("raw_records_enriched")],
        started_at=started_at,
    )


def step_dedupe_consolidate(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    enriched_path = cfg.path("raw_records_enriched")
    source_path = enriched_path if enriched_path.exists() else cfg.path("raw_records")
    records = dataframe_to_raw_records(pd.read_csv(source_path, dtype=str))

    id_priority = tuple(cfg.sources.get("dedup", {}).get("id_priority", ["pmcid", "doi", "pmid"]))
    canonical_records = consolidate(records, id_priority)

    to_dataframe(canonical_records).to_csv(cfg.path("canonical_dataset"), index=False)
    conflicts_dataframe(canonical_records).to_csv(cfg.path("conflicts_for_review"), index=False)

    n_conflict = sum(1 for r in canonical_records if r.has_conflict)
    finish_step(
        "dedupe.consolidate",
        inputs=[source_path],
        outputs=[cfg.path("canonical_dataset"), cfg.path("conflicts_for_review")],
        params={"id_priority": list(id_priority)},
        notes=f"{len(records)} raw -> {len(canonical_records)} canonical ({n_conflict} conflicts)",
        started_at=started_at,
    )


def step_fulltext_build_manifest(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    manifest = build_manifest(cfg.sources)
    manifest.to_csv(cfg.path("fulltext_manifest"), index=False)

    canonical_path = cfg.path("canonical_dataset")
    if canonical_path.exists() and not manifest.empty:
        dataset = pd.read_csv(canonical_path, dtype=str)
        available_pmcids = set(manifest["pmcid"].dropna())
        is_available = dataset["pmcid"].isin(available_pmcids)
        dataset["fulltext_available"] = is_available
        dataset["fulltext_manifest_ref"] = dataset["pmcid"].where(is_available)
        dataset.to_csv(canonical_path, index=False)

    finish_step(
        "fulltext.build-manifest",
        inputs=[canonical_path] if canonical_path.exists() else [],
        outputs=[cfg.path("fulltext_manifest")] + ([canonical_path] if canonical_path.exists() else []),
        started_at=started_at,
    )


def _load_labeled_texts(cfg: PipelineConfig, label: str) -> list[str]:
    dataset = pd.read_csv(cfg.path("canonical_dataset"), dtype=str)
    subset = dataset[dataset["label"] == label]
    texts = (subset["title"].fillna("") + ". " + subset["abstract"].fillna("")).tolist()
    return [t for t in texts if t.strip(". ")]


def step_keywords_tfidf(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    corpora_cfg = cfg.tfidf.get("corpora", {})
    positive_texts = _load_labeled_texts(cfg, corpora_cfg.get("positive_label", "positive"))
    baseline_texts = _load_labeled_texts(cfg, corpora_cfg.get("baseline_label", "negative"))

    terms_df = extract_tfidf_terms(positive_texts, baseline_texts, cfg.tfidf)
    output_path = cfg.path("interim_dir") / "tfidf_terms.csv"
    terms_df.to_csv(output_path, index=False)
    finish_step(
        "keywords.tfidf",
        inputs=[cfg.path("canonical_dataset")],
        outputs=[output_path],
        notes=f"{len(positive_texts)} positive / {len(baseline_texts)} baseline documents",
        started_at=started_at,
    )


def step_keywords_keybert(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    positive_label = cfg.tfidf.get("corpora", {}).get("positive_label", "positive")
    positive_texts = _load_labeled_texts(cfg, positive_label)

    terms_df = extract_keybert_terms(positive_texts, cfg.keybert)
    output_path = cfg.path("interim_dir") / "keybert_terms.csv"
    terms_df.to_csv(output_path, index=False)
    finish_step(
        "keywords.keybert",
        inputs=[cfg.path("canonical_dataset")],
        outputs=[output_path],
        notes=f"{len(positive_texts)} positive documents",
        started_at=started_at,
    )


def step_keywords_build_lexicon(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    cfg.ensure_dirs()
    tfidf_path = cfg.path("interim_dir") / "tfidf_terms.csv"
    keybert_path = cfg.path("interim_dir") / "keybert_terms.csv"
    tfidf_df = pd.read_csv(tfidf_path)
    keybert_df = pd.read_csv(keybert_path)

    seed_path = resolve_path(cfg.pipeline["keywords"]["seed_terms"])
    seed_df = load_seed_terms(seed_path) if seed_path.exists() else None

    candidates = build_candidate_lexicon(tfidf_df, keybert_df, seed_df)
    output_path = resolve_path(cfg.pipeline["keywords"]["candidates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    finish_step(
        "keywords.build-lexicon",
        inputs=[tfidf_path, keybert_path] + ([seed_path] if seed_path.exists() else []),
        outputs=[output_path],
        notes=f"{len(candidates)} candidate terms -- review via `keywords lexicon-stats` "
        "then the Streamlit Keyword Review page",
        started_at=started_at,
    )


def step_keywords_materialize_lexicon(cfg: PipelineConfig) -> None:
    """Folds keyword_review_events.csv (from the Streamlit Keyword Review page) into
    keyword_lexicon.csv (positive), keyword_lexicon_exclusionary.csv (negative), and
    keyword_lexicon_irrelevant.csv (irrelevant) -- last decision per term wins, regardless of
    which pile or manual entry produced it. Unlike `curate materialize`, this calls finish_step:
    keyword_lexicon.csv is a first-class pipeline artifact consumed by
    scoring-bakeoff/score-bulk-match, not a dataset that already has provenance from earlier
    ingest/dedupe steps."""
    started_at = time.monotonic()
    candidates_path = resolve_path(cfg.pipeline["keywords"]["candidates"])
    events_path = resolve_path(cfg.pipeline["keyword_review"]["events_log"])
    lexicon_path = resolve_path(cfg.pipeline["keywords"]["lexicon"])
    exclusionary_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    irrelevant_path = resolve_path(cfg.pipeline["keyword_review"]["irrelevant_terms"])

    counts = materialize_term_events(
        candidates_path, events_path, lexicon_path, exclusionary_path, irrelevant_path
    )

    finish_step(
        "keywords.materialize-lexicon",
        inputs=[candidates_path, events_path],
        outputs=[lexicon_path, exclusionary_path, irrelevant_path, candidates_path],
        notes=f"{counts['positive']} positive / {counts['negative']} negative / "
        f"{counts['irrelevant']} irrelevant",
        started_at=started_at,
    )


def _lookup_candidate_stats(candidates_df: pd.DataFrame, term: str):
    match = candidates_df[candidates_df["term"].str.lower() == term.lower()]
    if match.empty:
        return None, None
    row = match.iloc[0]
    return row.get("discriminative_score"), row.get("document_frequency")


def _already_present_terms(*paths: Path) -> set[str]:
    terms: set[str] = set()
    for path in paths:
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            if "term" in df.columns:
                terms |= set(df["term"].dropna().str.lower())
    return terms


_ADDED_TERM_COLUMNS = ["term", "discriminative_score", "document_frequency", "source", "notes"]


def step_keywords_seed_additional_terms(cfg: PipelineConfig) -> None:
    """Writes keywords/curated_terms.py's positive/negative additions to their own tier-2 files
    (added_positive_terms / added_negative_terms) -- skipping anything already decided in
    keyword_review_events.csv or already present in the materialized tier-1 lexicon/exclusionary
    files, and pulling real discriminative_score/document_frequency from
    keyword_lexicon_candidates.csv where the term was actually extracted (blank otherwise). Never
    touches the tier-1 files -- see keywords.suggest-final-lexicon for how tier 2 gets combined
    with tier 1."""
    started_at = time.monotonic()
    candidates_path = resolve_path(cfg.pipeline["keywords"]["candidates"])
    candidates_df = pd.read_csv(candidates_path)

    events_path = resolve_path(cfg.pipeline["keyword_review"]["events_log"])
    lexicon_path = resolve_path(cfg.pipeline["keywords"]["lexicon"])
    exclusionary_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    already_decided = _already_present_terms(events_path, lexicon_path, exclusionary_path)

    def _build_rows(term_specs: list[dict]) -> list[dict]:
        rows = []
        for spec in term_specs:
            term = spec["term"]
            if term.lower() in already_decided:
                continue
            discriminative_score, document_frequency = _lookup_candidate_stats(candidates_df, term)
            rows.append(
                {
                    "term": term,
                    "discriminative_score": discriminative_score,
                    "document_frequency": document_frequency,
                    "source": f"claude_seed_{spec['category']}",
                    "notes": spec["category"],
                }
            )
        return rows

    positive_rows = _build_rows(ADDED_POSITIVE_TERMS)
    negative_rows = _build_rows(ADDED_NEGATIVE_TERMS)

    added_positive_path = resolve_path(cfg.pipeline["keywords"]["added_positive_terms"])
    added_negative_path = resolve_path(cfg.pipeline["keywords"]["added_negative_terms"])
    added_positive_path.parent.mkdir(parents=True, exist_ok=True)
    backup_file(added_positive_path)
    backup_file(added_negative_path)

    pd.DataFrame(positive_rows, columns=_ADDED_TERM_COLUMNS).to_csv(added_positive_path, index=False)
    pd.DataFrame(negative_rows, columns=_ADDED_TERM_COLUMNS).to_csv(added_negative_path, index=False)

    finish_step(
        "keywords.seed-additional-terms",
        inputs=[candidates_path]
        + ([events_path] if events_path.exists() else [])
        + ([lexicon_path] if lexicon_path.exists() else [])
        + ([exclusionary_path] if exclusionary_path.exists() else []),
        outputs=[added_positive_path, added_negative_path],
        notes=f"{len(positive_rows)}/{len(ADDED_POSITIVE_TERMS)} positive, "
        f"{len(negative_rows)}/{len(ADDED_NEGATIVE_TERMS)} negative added "
        "(rest skipped -- already decided)",
        started_at=started_at,
    )


def step_keywords_suggest_final_lexicon(cfg: PipelineConfig) -> None:
    """Combines tier 1 (materialized keyword_lexicon.csv / keyword_lexicon_exclusionary.csv) with
    tier 2 (keyword_lexicon_added_positive.csv / _added_negative.csv), runs the cleanup heuristic
    (keywords/lexicon_cleanup.py::clean_lexicon), and writes tier 3: suggested_lexicon,
    suggested_exclusionary_lexicon, suggested_cleanup_log. Never modifies tier 1's live files --
    promoting tier 3 to production is a separate, manual decision."""
    started_at = time.monotonic()
    lexicon_path = resolve_path(cfg.pipeline["keywords"]["lexicon"])
    exclusionary_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    added_positive_path = resolve_path(cfg.pipeline["keywords"]["added_positive_terms"])
    added_negative_path = resolve_path(cfg.pipeline["keywords"]["added_negative_terms"])

    if not lexicon_path.exists():
        raise FileNotFoundError(f"{lexicon_path} not found -- run `keywords materialize-lexicon` first.")
    if not added_positive_path.exists():
        raise FileNotFoundError(f"{added_positive_path} not found -- run `keywords seed-additional-terms` first.")

    positive_df = pd.concat(
        [pd.read_csv(lexicon_path), pd.read_csv(added_positive_path)], ignore_index=True
    )
    negative_frames = [pd.read_csv(exclusionary_path)] if exclusionary_path.exists() else []
    if added_negative_path.exists():
        negative_frames.append(pd.read_csv(added_negative_path))
    negative_df = (
        pd.concat(negative_frames, ignore_index=True) if negative_frames else pd.DataFrame(columns=_ADDED_TERM_COLUMNS)
    )

    cleaned_positive, cleaned_negative, log_df = clean_lexicon(
        positive_df, negative_df, protected_unigrams=PROTECTED_UNIGRAMS
    )

    suggested_lexicon_path = resolve_path(cfg.pipeline["keywords"]["suggested_lexicon"])
    suggested_exclusionary_path = resolve_path(cfg.pipeline["keywords"]["suggested_exclusionary_lexicon"])
    suggested_log_path = resolve_path(cfg.pipeline["keywords"]["suggested_cleanup_log"])
    suggested_lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    backup_file(suggested_lexicon_path)
    backup_file(suggested_exclusionary_path)
    backup_file(suggested_log_path)

    cleaned_positive.to_csv(suggested_lexicon_path, index=False)
    cleaned_negative.to_csv(suggested_exclusionary_path, index=False)
    log_df.to_csv(suggested_log_path, index=False)

    n_removed = int((log_df["action"] == "removed").sum()) if not log_df.empty else 0
    n_flagged = int((log_df["action"] == "kept_flagged").sum()) if not log_df.empty else 0

    finish_step(
        "keywords.suggest-final-lexicon",
        inputs=[lexicon_path, added_positive_path]
        + ([exclusionary_path] if exclusionary_path.exists() else [])
        + ([added_negative_path] if added_negative_path.exists() else []),
        outputs=[suggested_lexicon_path, suggested_exclusionary_path, suggested_log_path],
        notes=f"{len(cleaned_positive)} positive / {len(cleaned_negative)} negative terms "
        f"suggested; {n_removed} removed, {n_flagged} flagged (tension) -- see cleanup log",
        started_at=started_at,
    )


def step_keywords_lexicon_stats(cfg: PipelineConfig) -> None:
    """Prints (and saves) term-counts remaining at a range of thresholds -- the data-driven
    cutoff decision support tool, since reviewing all ~40k raw candidates by hand isn't
    practical."""
    started_at = time.monotonic()
    candidates_path = resolve_path(cfg.pipeline["keywords"]["candidates"])
    candidates = pd.read_csv(candidates_path)
    stats = lexicon_stats(candidates)

    output_path = cfg.path("processed_dir") / "lexicon_stats_report.csv"
    stats.to_csv(output_path, index=False)
    print(stats.to_string(index=False))

    finish_step(
        "keywords.lexicon-stats",
        inputs=[candidates_path],
        outputs=[output_path],
        started_at=started_at,
    )


def step_keywords_scoring_bakeoff(cfg: PipelineConfig, exclusionary_weight: float = 1.0) -> None:
    """Validates every MatchScorer against the already-known-labeled records in
    canonical_dataset.csv, TWICE per scorer: once using only the approved positive lexicon
    (condition "positive_lexicon_only"), and -- if keyword_lexicon_exclusionary.csv exists --
    again with the exclusionary lexicon's penalty applied too (condition
    "positive_plus_exclusionary_lexicon"). Two conditions side by side is a genuine before/after
    comparison of whether the exclusionary lexicon actually improves ranking quality, not just an
    assumption that it does."""
    started_at = time.monotonic()
    lexicon_path = cfg.path("processed_dir") / "keyword_lexicon.csv"
    if not lexicon_path.exists():
        raise FileNotFoundError(
            f"{lexicon_path} not found -- approve terms via the Streamlit Keyword Review page first."
        )
    lexicon_df = pd.read_csv(lexicon_path)
    lexicon_terms, term_weights = load_lexicon_terms_and_weights(lexicon_df)

    exclusionary_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    exclusionary_terms: list[str] = []
    exclusionary_term_weights: dict[str, float] = {}
    if exclusionary_path.exists():
        exclusionary_df = pd.read_csv(exclusionary_path)
        exclusionary_terms, exclusionary_term_weights = load_lexicon_terms_and_weights(exclusionary_df)

    dataset = pd.read_csv(cfg.path("canonical_dataset"), dtype=str)
    labeled = dataset[dataset["label"].isin(["positive", "negative"])].copy()
    texts = (labeled["title"].fillna("") + ". " + labeled["abstract"].fillna("")).tolist()
    true_labels = (labeled["label"] == "positive").astype(int).tolist()
    n_positive = sum(true_labels)
    n_negative = len(true_labels) - n_positive

    def _build_scorers() -> dict:
        return {
            "weighted-sum": WeightedSumScorer(term_weights, exclusionary_term_weights),
            **{name: cls() for name, cls in SCORERS.items() if name != "weighted-sum"},
        }

    reports = [
        run_bakeoff(
            _build_scorers(), texts, true_labels, lexicon_terms, condition_label="positive_lexicon_only"
        )
    ]
    if exclusionary_terms:
        reports.append(
            run_bakeoff(
                _build_scorers(),
                texts,
                true_labels,
                lexicon_terms,
                exclusionary_terms=exclusionary_terms,
                exclusionary_weight=exclusionary_weight,
                condition_label="positive_plus_exclusionary_lexicon",
            )
        )
    report = pd.concat(reports, ignore_index=True).sort_values(["scorer", "condition"]).reset_index(drop=True)

    output_path = cfg.path("processed_dir") / "scoring_bakeoff_report.csv"
    report.to_csv(output_path, index=False)
    print(report.to_string(index=False))

    finish_step(
        "keywords.scoring-bakeoff",
        inputs=[lexicon_path, cfg.path("canonical_dataset")] + ([exclusionary_path] if exclusionary_terms else []),
        outputs=[output_path],
        params={
            "exclusionary_weight": exclusionary_weight,
            "n_exclusionary_terms": len(exclusionary_terms),
            "conditions_run": sorted(report["condition"].unique().tolist()),
        },
        notes=f"validated against {len(labeled)} already-labeled records "
        f"({n_positive} positive / {n_negative} negative) -- "
        + (
            "2 conditions (with/without exclusionary lexicon)"
            if exclusionary_terms
            else "1 condition (no keyword_lexicon_exclusionary.csv found)"
        ),
        started_at=started_at,
    )


def step_bulk_match_fetch(cfg: PipelineConfig, year_from: int, year_to: int) -> None:
    """Fetches every year in [year_from, year_to] of AI/ML-matching Europe PMC records in one
    invocation -- one EPMC query per year internally (checkpointed/resumable via existing .done
    markers), live per-year progress printed for a human watching a long multi-year run. Also
    computes and appends the AI-only/ML-only/combined-deduplicated hit-count breakdown for the
    requested range to bulk_match_summary.csv (three cheap count-only queries, no extra fetch)."""
    started_at = time.monotonic()
    checkpoint_dir = cfg.path("interim_dir") / "bulk_match_cache"
    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        output_paths = fetch_ai_ml_range(client, year_from, year_to, checkpoint_dir)
        breakdown = count_ai_ml_breakdown(client, year_from, year_to)
    finally:
        client.close()

    print(
        f"AI-mentioning: {breakdown['ai_count']} | ML-mentioning: {breakdown['ml_count']} | "
        f"Combined (deduplicated): {breakdown['combined_count']}"
    )

    summary_path = cfg.path("processed_dir") / "bulk_match_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_row = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "year_from": year_from,
        "year_to": year_to,
        **breakdown,
    }
    header_needed = not summary_path.exists()
    pd.DataFrame([summary_row]).to_csv(summary_path, mode="a", header=header_needed, index=False)

    finish_step(
        "bulk-match.fetch",
        inputs=[],
        outputs=output_paths + [summary_path],
        params={"year_from": year_from, "year_to": year_to, **breakdown},
        notes=f"{len(output_paths)} year(s) fetched ({year_from}-{year_to})",
        started_at=started_at,
    )


def dedupe_bulk_match_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Pure, testable core of `step_bulk_match_build_candidates`'s dedup: keyed on the SAME
    priority-based identifier (pmcid -> doi -> pmid) `dedupe/keys.py::record_id_from_ids` already
    uses everywhere else in this codebase, not bare pmid alone.

    **Real, confirmed bug this replaces (2026-08-27)**: the original dedup was
    `df.sort_values("pmid").drop_duplicates(subset=["pmid"], keep="first")`. pandas'
    `drop_duplicates()` treats NaN as equal to NaN, so that collapsed EVERY record lacking a pmid
    (preprints, PMC-only, older non-MEDLINE literature -- 80,926 of the real 842,378-record
    all-time fetch, confirmed live) down to just 1-2 survivors, as if they were all the same
    paper. Never triggered by the original 2000-2026-only fetch's real pmid coverage; surfaced by
    the wider all-time re-fetch.

    Of those 80,926 pmid-less records, 57,477 have a real DOI and 15,362 a real PMCID (confirmed
    live) -- deduping on the pmcid/doi/pmid priority id genuinely catches real cross-year
    duplicates among them, instead of either wrongly merging them all (the bug above) or leaving
    them all forever undeduped (a real but much smaller gap). Records with no id at all (8,231,
    confirmed live) are always kept, since there's nothing to dedupe them against. The NaN ->
    None conversion is required because NaN is truthy in Python, so `record_id_from_ids`'
    `if value:` check would otherwise treat a missing id as if it were real."""
    if df.empty:
        return df
    id_cols = df[["pmcid", "pmid", "doi"]].where(pd.notna(df[["pmcid", "pmid", "doi"]]), None)
    df = df.copy()
    df["_dedup_id"] = id_cols.apply(lambda row: record_id_from_ids(row["pmcid"], row["pmid"], row["doi"]), axis=1)
    has_id = df["_dedup_id"].notna()
    deduped_with_id = df[has_id].sort_values("_dedup_id").drop_duplicates(subset=["_dedup_id"], keep="first")
    return pd.concat([deduped_with_id, df[~has_id]], ignore_index=True).drop(columns=["_dedup_id"])


def step_bulk_match_build_candidates(cfg: PipelineConfig) -> None:
    """Consolidates every *completed* per-year JSONL cache (still one file per year on disk even
    when `bulk-match fetch --year-from --year-to` fetched a whole range in a single invocation --
    that's an internal checkpointing detail, not something the human triggers) into one
    deduplicated candidate pool, keyed on the pmcid -> doi -> pmid priority id (see
    `dedupe_bulk_match_batch`'s docstring for the real bug this design replaced). Rerun anytime
    after fetching more years to pick up newly completed ones.

    Processes one year's JSONL at a time -- converting each year straight to a DataFrame and
    discarding its RawRecord/Pydantic objects before loading the next year, deduplicating the
    running frame incrementally -- rather than materializing every year's Pydantic objects
    simultaneously. Confirmed necessary, not just theoretical: an earlier all-at-once version got
    OOM-killed (exit 137) on the real 2000-2026 fetch (~828k records, 5.3GB of JSONL) on a 15GB
    host; Pydantic model instances carry substantially more memory overhead per record than a
    DataFrame row."""
    started_at = time.monotonic()
    checkpoint_dir = cfg.path("interim_dir") / "bulk_match_cache"
    completed_years = []
    done_markers = sorted(checkpoint_dir.glob("bulk_match_*.done"))
    combined_df: pd.DataFrame | None = None
    for done_marker in tqdm(done_markers, desc="Loading + deduplicating per-year caches", unit="year"):
        year = int(done_marker.stem.split("_")[-1])
        jsonl_path = checkpoint_dir / f"bulk_match_{year}.jsonl"
        year_df = raw_records_to_dataframe(load_bulk_match_year(jsonl_path, year))
        running_total = len(year_df) if combined_df is None else len(combined_df) + len(year_df)
        print(f"  {year}: {len(year_df)} records (running total before dedup: {running_total})")
        completed_years.append(year)
        if year_df.empty:
            # Real, confirmed bug fixed here (2026-08-27): raw_records_to_dataframe([]) on a
            # genuinely zero-hit year (common for pre-2000 years, never triggered by the original
            # 2000-2026-only fetch) returns a DataFrame with ZERO COLUMNS, not just zero rows --
            # dedupe_bulk_match_batch's sort_values below then KeyErrors on a combined_df that has
            # nothing to merge in anyway. Nothing to concat or dedupe against, so just skip the year.
            continue
        combined_df = year_df if combined_df is None else pd.concat([combined_df, year_df], ignore_index=True)
        combined_df = dedupe_bulk_match_batch(combined_df)

    df = combined_df if combined_df is not None else raw_records_to_dataframe([])
    print(f"Deduplicated total: {len(df)} unique records (by pmcid -> doi -> pmid priority)")

    output_path = cfg.sampling_path("bulk_candidates")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    finish_step(
        "bulk-match.build-candidates",
        inputs=[checkpoint_dir / f"bulk_match_{y}.jsonl" for y in completed_years],
        outputs=[output_path],
        params={"years": completed_years},
        notes=f"{len(df)} deduplicated candidates from {len(completed_years)} completed year(s)",
        started_at=started_at,
    )


def _build_scorer(scorer_name: str, term_weights: dict[str, float], exclusionary_term_weights: dict[str, float]):
    if scorer_name == "weighted-sum":
        return WeightedSumScorer(term_weights, exclusionary_term_weights)
    return SCORERS[scorer_name]()


def _lookup_youden_threshold(bakeoff_report_path: Path, scorer_name: str, condition: str) -> float | None:
    """Reuses the threshold Step 11's bake-off already validated for this exact scorer+condition,
    rather than recomputing one -- keeps Step 12's positive/negative classification consistent
    with what was empirically checked, not a fresh guess. Returns None (no classification column
    added for this scorer) if the report doesn't exist yet or has no matching row -- run
    `keywords scoring-bakeoff` first to get one."""
    if not bakeoff_report_path.exists():
        return None
    report = pd.read_csv(bakeoff_report_path)
    match = report[(report["scorer"] == scorer_name) & (report["condition"] == condition)]
    if match.empty or pd.isna(match.iloc[0]["threshold_youden"]):
        return None
    return float(match.iloc[0]["threshold_youden"])


def _build_existing_label_lookup(canonical_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Maps every pmcid/pmid/doi in the canonical dataset to that record's (label,
    label_confidence). Small (~4k rows), so a plain loop is fine here -- contrast
    `_annotate_already_curated` below, which is vectorized because it runs against the
    700k+-row bulk candidate pool."""
    lookup: dict[str, tuple[str, str]] = {}
    for id_field in ("pmcid", "pmid", "doi"):
        subset = canonical_df[canonical_df[id_field].notna() & (canonical_df[id_field] != "")]
        for value, label, confidence in zip(subset[id_field], subset["label"], subset["label_confidence"]):
            lookup.setdefault(value, (label, confidence))
    return lookup


def _annotate_already_curated(candidates: pd.DataFrame, lookup: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Adds `already_curated`/`existing_label`/`existing_label_confidence` -- whether this bulk
    candidate already exists in canonical_dataset.csv from a prior curation round (by pmcid, else
    pmid, else doi), and if so what it was already decided as. Lets a human reviewing the
    stratified sample later (or `curate/state.py::CurationSession`'s `include_already_labeled`
    toggle) see and choose to skip or deliberately redo already-curated overlaps, rather than
    silently either re-reviewing or hiding them. Vectorized (three dict `.map()` calls, not a
    per-row Python loop) -- this runs against the full 700k+-row bulk pool."""
    label_lookup = {k: v[0] for k, v in lookup.items()}
    confidence_lookup = {k: v[1] for k, v in lookup.items()}

    existing_label = (
        candidates["pmcid"].map(label_lookup).fillna(candidates["pmid"].map(label_lookup)).fillna(
            candidates["doi"].map(label_lookup)
        )
    )
    existing_confidence = (
        candidates["pmcid"].map(confidence_lookup).fillna(candidates["pmid"].map(confidence_lookup)).fillna(
            candidates["doi"].map(confidence_lookup)
        )
    )

    candidates["already_curated"] = existing_label.notna()
    candidates["existing_label"] = existing_label.fillna("")
    candidates["existing_label_confidence"] = existing_confidence.fillna("")
    return candidates


def step_keywords_score_bulk_match(
    cfg: PipelineConfig, scorer_name: str, exclusionary_weight: float = 1.0
) -> None:
    """`scorer_name` is one of SCORERS' keys, or "all" to keep every scorer as a separate
    match_score__<name> column for later comparison. If keyword_lexicon_exclusionary.csv exists
    (from `keywords materialize-lexicon`), its terms are subtracted as a penalty, weighted by
    `exclusionary_weight`.

    Also adds, once regardless of how many scorers run: `has_pmcid` (full text available in PMC
    or not) and `already_curated`/`existing_label`/`existing_label_confidence` (does this
    candidate already exist in canonical_dataset.csv from a prior curation round -- see
    `_annotate_already_curated`). And per scorer, `match_classification__<name>`
    (positive/negative) if `scoring_bakeoff_report.csv` has a validated Youden threshold for that
    exact scorer + condition (positive-only vs positive-plus-exclusionary, auto-detected from
    whether an exclusionary lexicon was found) -- see `_lookup_youden_threshold`."""
    started_at = time.monotonic()
    print("score-bulk-match: [1/6] loading positive lexicon...")
    lexicon_path = cfg.path("processed_dir") / "keyword_lexicon.csv"
    lexicon_df = pd.read_csv(lexicon_path)
    lexicon_terms, term_weights = load_lexicon_terms_and_weights(lexicon_df)
    print(f"score-bulk-match:   {len(lexicon_terms)} positive terms loaded.")

    print("score-bulk-match: [2/6] loading exclusionary lexicon (if present)...")
    exclusionary_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    exclusionary_terms: list[str] = []
    exclusionary_term_weights: dict[str, float] = {}
    if exclusionary_path.exists():
        exclusionary_df = pd.read_csv(exclusionary_path)
        exclusionary_terms, exclusionary_term_weights = load_lexicon_terms_and_weights(exclusionary_df)
        print(f"score-bulk-match:   {len(exclusionary_terms)} exclusionary terms loaded.")
    else:
        print("score-bulk-match:   no exclusionary lexicon found -- positive-only condition.")
    condition = "positive_plus_exclusionary_lexicon" if exclusionary_terms else "positive_lexicon_only"

    print("score-bulk-match: [3/6] loading bulk candidates (this can take a minute at 700k+ rows)...")
    candidates_path = cfg.sampling_path("bulk_candidates")
    candidates = pd.read_csv(candidates_path, dtype=str)
    texts = (candidates["title"].fillna("") + ". " + candidates["abstract"].fillna("")).tolist()
    print(f"score-bulk-match:   {len(candidates)} candidates loaded.")

    candidates["has_pmcid"] = candidates["pmcid"].notna() & (candidates["pmcid"] != "")

    print("score-bulk-match: [4/6] cross-referencing against canonical_dataset.csv for already-curated records...")
    canonical_path = cfg.path("canonical_dataset")
    if canonical_path.exists():
        existing_lookup = _build_existing_label_lookup(pd.read_csv(canonical_path, dtype=str))
        candidates = _annotate_already_curated(candidates, existing_lookup)
    else:
        candidates["already_curated"] = False
        candidates["existing_label"] = ""
        candidates["existing_label_confidence"] = ""
    print(f"score-bulk-match:   {int(candidates['already_curated'].sum())} already curated; "
          f"{int(candidates['has_pmcid'].sum())} have a PMCID.")

    bakeoff_report_path = cfg.path("processed_dir") / "scoring_bakeoff_report.csv"
    thresholds_used: dict[str, float] = {}

    names_to_run = list(SCORERS) if scorer_name == "all" else [scorer_name]
    print(f"score-bulk-match: [5/6] scoring with: {', '.join(names_to_run)} (condition={condition})")
    for i, name in enumerate(names_to_run, start=1):
        print(f"score-bulk-match:   scorer {i}/{len(names_to_run)}: {name}")
        scorer = _build_scorer(name, term_weights, exclusionary_term_weights)
        checkpoint_path = cfg.path("interim_dir") / f"score_bulk_match_checkpoint__{name}.json"
        scored = scorer.score_corpus(
            texts,
            lexicon_terms,
            exclusionary_terms=exclusionary_terms or None,
            exclusionary_weight=exclusionary_weight,
            checkpoint_path=checkpoint_path,
        )
        print(f"score-bulk-match:     progress checkpoint written to {checkpoint_path}")
        candidates[f"match_score__{name}"] = [s for s, _ in scored]
        candidates[f"matched_terms__{name}"] = [";".join(terms) for _, terms in scored]

        threshold = _lookup_youden_threshold(bakeoff_report_path, name, condition)
        if threshold is not None:
            thresholds_used[name] = threshold
            candidates[f"match_classification__{name}"] = [
                "positive" if s >= threshold else "negative" for s in candidates[f"match_score__{name}"]
            ]
            n_pos = int((candidates[f"match_classification__{name}"] == "positive").sum())
            n_neg = int((candidates[f"match_classification__{name}"] == "negative").sum())
            print(
                f"score-bulk-match:     classified at Youden threshold {threshold:.5g} "
                f"(from Step 11's bake-off): {n_pos} positive, {n_neg} negative -- "
                "both retained in the output, negative is NOT dropped."
            )
        else:
            print(
                f"score-bulk-match:     no validated threshold found for {name}/{condition} in "
                "scoring_bakeoff_report.csv -- match_score written, no classification column added."
            )

    print(f"score-bulk-match: [6/6] writing {len(candidates)} scored candidates to disk...")
    output_path = cfg.sampling_path("bulk_candidates_scored")
    candidates.to_csv(output_path, index=False)
    print(f"score-bulk-match: done -- wrote {output_path}")

    n_already_curated = int(candidates["already_curated"].sum())
    n_has_pmcid = int(candidates["has_pmcid"].sum())

    finish_step(
        "keywords.score-bulk-match",
        inputs=[lexicon_path, candidates_path]
        + ([exclusionary_path] if exclusionary_terms else [])
        + ([bakeoff_report_path] if thresholds_used else []),
        outputs=[output_path],
        params={
            "scorer": scorer_name,
            "exclusionary_weight": exclusionary_weight,
            "n_exclusionary_terms": len(exclusionary_terms),
            "condition": condition,
            "thresholds_used": thresholds_used,
        },
        notes=f"{len(candidates)} candidates scored; {n_already_curated} already curated in "
        f"canonical_dataset.csv ({n_already_curated / len(candidates) * 100:.1f}%); "
        f"{n_has_pmcid} have a PMCID ({n_has_pmcid / len(candidates) * 100:.1f}%)"
        + (
            f"; classified at threshold(s) {thresholds_used} (from Step 11's bake-off)"
            if thresholds_used
            else "; no classification column added -- run `keywords scoring-bakeoff` first for one"
        ),
        started_at=started_at,
    )


def _load_existing_ids(canonical_path: Path) -> set[str]:
    if not canonical_path.exists():
        return set()
    dataset = pd.read_csv(canonical_path, dtype=str)
    ids = set(dataset["pmcid"].dropna()) | set(dataset["pmid"].dropna()) | set(dataset["doi"].dropna())
    return ids


def _merge_new_candidates_into_canonical(cfg: PipelineConfig, new_records: list[RawRecord]) -> int:
    """Filters out anything already present (by pmcid/pmid/doi), consolidates the remaining new
    batch (handles duplicates within the batch itself), and appends to canonical_dataset.csv --
    this is what makes a freshly sampled/fetched candidate pool show up in the curation queue."""
    canonical_path = cfg.path("canonical_dataset")
    existing_ids = _load_existing_ids(canonical_path)

    fresh = [r for r in new_records if not ({r.pmcid, r.pmid, r.doi} & existing_ids)]
    if not fresh:
        return 0

    new_canonical = consolidate(fresh)
    new_df = to_dataframe(new_canonical)

    if canonical_path.exists():
        existing_df = pd.read_csv(canonical_path, dtype=str)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(canonical_path, index=False)
    return len(new_canonical)


def step_sampling_stratify(cfg: PipelineConfig) -> None:
    started_at = time.monotonic()
    scored_path = cfg.sampling_path("bulk_candidates_scored")
    df = pd.read_csv(scored_path, dtype=str)

    score_cols = [c for c in df.columns if c.startswith("match_score__")]
    if not score_cols:
        raise ValueError(f"No match_score__* column found in {scored_path} -- run score-bulk-match first.")
    score_col = score_cols[0]
    df[score_col] = df[score_col].astype(float)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    strata_cfg = cfg.sampling.get("strata", {})
    strata_df = build_strata(
        df,
        score_col=score_col,
        n_score_bands=strata_cfg.get("n_score_bands", 4),
        top_n_journals=strata_cfg.get("top_n_journals", 15),
        year_bucket_width=strata_cfg.get("year_bucket_width", 5),
    )
    strata_cols = [f"match_score_band__{score_col}", "journal_bucket", "year_bucket"]

    sampling_cfg = cfg.sampling.get("sampling", {})
    sampled, report = stratified_sample(
        strata_df,
        strata_cols,
        cap_per_stratum=sampling_cfg.get("cap_per_stratum", 10),
        random_state=sampling_cfg.get("random_state", 42),
    )

    pool_path = cfg.sampling_path("stratified_candidate_pool")
    report_path = cfg.sampling_path("stratum_report")
    sampled.to_csv(pool_path, index=False)
    report.to_csv(report_path, index=False)
    print(report.to_string(index=False))

    new_records = dataframe_to_raw_records(sampled)
    n_added = _merge_new_candidates_into_canonical(cfg, new_records)

    finish_step(
        "sampling.stratify",
        inputs=[scored_path],
        outputs=[pool_path, report_path],
        params={"strata_cols": strata_cols, "cap_per_stratum": sampling_cfg.get("cap_per_stratum", 10)},
        notes=f"{len(sampled)} sampled, {n_added} new records merged into canonical_dataset.csv "
        "for curation (rest were already present)",
        started_at=started_at,
    )


def step_ingest_fetch_clear_negatives(
    cfg: PipelineConfig,
    year_from: int,
    year_to: int,
    sample_size: int,
    merge_limit: int | None = None,
    n_windows: int = 40,
    max_per_window: int = 1500,
) -> None:
    """`sample_size` controls the full diverse pool built and written to disk (cheap -- an interim
    file, no canonical impact by itself). `merge_limit` (defaults to `sample_size` if omitted)
    caps how much of that pool actually merges into `canonical_dataset.csv` this run -- these
    candidates are stamped `label="negative"` *at fetch time*, so an uncapped merge would push the
    dataset's raw label balance sharply negative before any human review. Because the fetch is
    deterministically seeded, re-running with a higher `--merge-limit` refetches the *same* pool
    and the existing dedup-by-ID merge logic pulls in only the incremental delta -- phased merging
    for free. See STEPS_Progress.md Step 14 for the worked ratio arithmetic and for why this is a
    two-phase fetch (`max_per_window` caps Phase 1's cheap lookup; see
    `clear_negative_sampler.fetch_clear_negatives`'s docstring for the full performance reasoning)."""
    started_at = time.monotonic()
    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        df = fetch_clear_negatives(
            client, year_from, year_to, sample_size, n_windows=n_windows, max_per_window=max_per_window
        )
    finally:
        client.close()

    output_path = cfg.sampling_path("clear_negative_candidates")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    merge_limit = sample_size if merge_limit is None else merge_limit
    to_merge = df if len(df) <= merge_limit else df.sample(n=merge_limit, random_state=42)
    print(f"ingest.fetch-clear-negatives: merging up to {merge_limit} of the {len(df)}-row pool "
          f"({len(to_merge)} selected) into canonical_dataset.csv...")

    new_records = dataframe_to_raw_records(to_merge)
    n_added = _merge_new_candidates_into_canonical(cfg, new_records)

    canonical_path = cfg.path("canonical_dataset")
    if canonical_path.exists():
        label_counts = pd.read_csv(canonical_path, dtype=str)["label"].value_counts()
        n_pos, n_neg = int(label_counts.get("positive", 0)), int(label_counts.get("negative", 0))
        ratio = f"{n_neg / n_pos:.2f}:1" if n_pos else "n/a"
        print(f"ingest.fetch-clear-negatives: canonical_dataset.csv is now {n_pos} positive / "
              f"{n_neg} negative (ratio {ratio}) -- watch this before increasing --merge-limit "
              "in a later round.")

    finish_step(
        "ingest.fetch-clear-negatives",
        inputs=[],
        outputs=[output_path],
        params={
            "year_from": year_from,
            "year_to": year_to,
            "sample_size": sample_size,
            "merge_limit": merge_limit,
            "n_windows": n_windows,
            "max_per_window": max_per_window,
        },
        notes=f"{len(df)} diverse candidates fetched (journal/year-stratified), {n_added} new "
        f"records merged into canonical_dataset.csv for curation (merge capped at {merge_limit})",
        started_at=started_at,
    )


def _plot_clear_negative_score_distribution(
    scores: pd.Series, threshold: float | None, output_path: Path
) -> None:
    """Histogram of clear-negative candidates' `lexicon_score__bm25` -- a diagnostic for your own
    inspection, not a gate. Step 14c merges the whole screened pool regardless of where a
    candidate falls on this plot (see that step's docstring for why the earlier hard exclusion
    was removed on your explicit instruction). The Youden threshold, if available, is drawn as a
    reference line so it's visually obvious how many candidates sit above the point that would
    flag an AI/ML-matched paper as positive -- context for a human spot-check, nothing more."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(scores, bins=50, color="#4C72B0", edgecolor="white")
    if threshold is not None:
        ax.axvline(
            threshold, color="#C44E52", linestyle="--", linewidth=1.5,
            label=f"Youden threshold ({threshold:.1f})",
        )
        ax.legend()
    ax.set_xlabel("lexicon_score__bm25")
    ax.set_ylabel("candidate count")
    ax.set_title(
        "Clear-negative candidates: BM25 lexicon score distribution\n"
        "(diagnostic only -- Step 14c merges the full pool regardless of score)"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def step_ingest_screen_clear_negatives(cfg: PipelineConfig) -> None:
    """Step 14b: re-scores clear_negative_candidates.csv (fetched specifically for NOT mentioning
    AI/ML terms) against the *same* promoted lexicon Step 12 already scores the AI/ML-matched pool
    with, and plots the resulting score distribution (`_plot_clear_negative_score_distribution`)
    against the already-validated Youden threshold. This is diagnostic visibility, not a filter --
    on your explicit instruction, `needs_screening` is recorded for every candidate but Step 14c
    does not exclude flagged rows from the merge; a candidate that scores at/above the threshold
    despite lacking the literal AI/ML phrase is still merged, just visibly flagged in the data
    (and in the plot) for you to spot-check later if you want to, e.g. via the Curate app's
    needs_screening filter. Small-scale reuse of exactly what step_keywords_score_bulk_match
    already does (same scorer, same threshold lookup) -- at up to ~10k rows this finishes in well
    under a minute, no chunking/checkpointing needed at that scale."""
    started_at = time.monotonic()
    candidates_path = cfg.sampling_path("clear_negative_candidates")
    if not candidates_path.exists():
        raise ValueError(f"{candidates_path} does not exist -- run `ingest fetch-clear-negatives` first.")
    candidates = pd.read_csv(candidates_path, dtype=str)
    texts = (candidates["title"].fillna("") + ". " + candidates["abstract"].fillna("")).tolist()

    lexicon_path = cfg.path("processed_dir") / "keyword_lexicon.csv"
    lexicon_terms, _ = load_lexicon_terms_and_weights(pd.read_csv(lexicon_path))

    exclusionary_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    exclusionary_terms: list[str] = []
    if exclusionary_path.exists():
        exclusionary_terms, _ = load_lexicon_terms_and_weights(pd.read_csv(exclusionary_path))
    condition = "positive_plus_exclusionary_lexicon" if exclusionary_terms else "positive_lexicon_only"

    print(f"screen-clear-negatives: scoring {len(candidates)} candidates against the lexicon...")
    scored = SCORERS["bm25"]().score_corpus(
        texts, lexicon_terms, exclusionary_terms=exclusionary_terms or None
    )
    candidates["lexicon_score__bm25"] = [s for s, _ in scored]

    bakeoff_report_path = cfg.path("processed_dir") / "scoring_bakeoff_report.csv"
    threshold = _lookup_youden_threshold(bakeoff_report_path, "bm25", condition)
    if threshold is None:
        candidates["needs_screening"] = False
        print("screen-clear-negatives: no validated Youden threshold found -- run "
              "`keywords scoring-bakeoff` first; all candidates left unflagged for now.")
    else:
        candidates["needs_screening"] = candidates["lexicon_score__bm25"] >= threshold
        n_flagged = int(candidates["needs_screening"].sum())
        print(f"screen-clear-negatives: {n_flagged}/{len(candidates)} score >= {threshold:.1f} "
              f"(the validated bm25/{condition} threshold) despite the AI/ML exclusion query "
              "used to fetch them -- flagged in the data for visibility only, NOT excluded; "
              "Step 14c merges the full pool regardless of this flag.")

    plot_path = cfg.sampling_path("clear_negative_score_plot")
    candidates["lexicon_score__bm25"] = candidates["lexicon_score__bm25"].astype(float)
    _plot_clear_negative_score_distribution(candidates["lexicon_score__bm25"], threshold, plot_path)
    print(f"screen-clear-negatives: score distribution plotted to {plot_path}")

    output_path = cfg.sampling_path("clear_negative_candidates_screened")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)

    finish_step(
        "ingest.screen-clear-negatives",
        inputs=[candidates_path, lexicon_path]
        + ([exclusionary_path] if exclusionary_terms else [])
        + ([bakeoff_report_path] if threshold is not None else []),
        outputs=[output_path, plot_path],
        params={"condition": condition, "threshold": threshold},
        notes=f"{len(candidates)} candidates screened"
        + (
            f"; {int(candidates['needs_screening'].sum())} flagged for visibility (not excluded)"
            if threshold is not None
            else "; none flagged (no threshold available)"
        ),
        started_at=started_at,
    )


def step_ingest_merge_strong_negatives(cfg: PipelineConfig, limit: int) -> None:
    """Step 14c: merges up to `limit` candidates from `clear_negative_candidates_screened.csv`
    (Step 14b's output) into canonical_dataset.csv. "Strong" comes from construction -- every
    candidate was independently confirmed by a live EPMC query to not mention any AI/ML term
    (Step 14's expanded `EXCLUDE_QUERY`) -- not from a second BM25 gate. **On explicit
    instruction, Step 14b's `needs_screening` flag does NOT exclude anyone here** -- see
    `select_strong_negatives` in clear_negative_sampler.py for the actual re-diversify/cap logic;
    the flag (and its plotted score distribution from Step 14b) stays in the data purely for your
    own optional spot-checking later, e.g. via the Curate app's needs_screening filter. Tags every
    merged row's provenance with `source_name="clear_negative_sampler_strong"` (distinct from the
    plain `"clear_negative_sampler"` interim-pool tag Step 14 uses) plus each row's own
    `lexicon_score__bm25` in `match_metadata`, so the score is traceable per row even though it
    didn't gate inclusion."""
    started_at = time.monotonic()
    screened_path = cfg.sampling_path("clear_negative_candidates_screened")
    if not screened_path.exists():
        raise ValueError(f"{screened_path} does not exist -- run `ingest screen-clear-negatives` first.")
    screened = pd.read_csv(screened_path, dtype=str)
    if "needs_screening" not in screened.columns:
        raise ValueError(
            f"{screened_path} has no needs_screening column -- re-run `ingest screen-clear-negatives`."
        )

    selected, n_flagged = select_strong_negatives(screened, limit)
    print(f"ingest.merge-strong-negatives: merging {len(selected)}/{len(screened)} candidates "
          f"(the full screened pool is eligible -- {n_flagged} of the merged batch score above "
          "the lexicon threshold and are flagged needs_screening=True in the data, but none were "
          "excluded from this merge).")

    selected["source_name"] = "clear_negative_sampler_strong"
    selected["match_metadata"] = selected["lexicon_score__bm25"].apply(
        lambda score: json.dumps({"strong_negative_screen": {"lexicon_score__bm25": float(score)}})
    )

    new_records = dataframe_to_raw_records(selected)
    n_added = _merge_new_candidates_into_canonical(cfg, new_records)

    canonical_path = cfg.path("canonical_dataset")
    if canonical_path.exists():
        label_counts = pd.read_csv(canonical_path, dtype=str)["label"].value_counts()
        n_pos, n_neg = int(label_counts.get("positive", 0)), int(label_counts.get("negative", 0))
        ratio = f"{n_neg / n_pos:.2f}:1" if n_pos else "n/a"
        print(f"ingest.merge-strong-negatives: canonical_dataset.csv is now {n_pos} positive / "
              f"{n_neg} negative (ratio {ratio}).")

    finish_step(
        "ingest.merge-strong-negatives",
        inputs=[screened_path],
        outputs=[canonical_path],
        params={"limit": limit},
        notes=f"{len(selected)} strong negatives merged ({n_flagged} of them also flagged "
        "needs_screening=True but not excluded), "
        f"{n_added} genuinely new records merged into canonical_dataset.csv (rest already present)",
        started_at=started_at,
    )


def step_ingest_fetch_clear_negatives_filtered(
    cfg: PipelineConfig,
    year_from: int,
    year_to: int,
    raw_pool_size: int = 3000,
    target_size: int = 500,
    n_windows: int = 40,
    max_per_window: int = 1500,
) -> None:
    """Step 19d: fetches a new batch of clear negatives using the exact same method as Step 14
    (`clear_negative_sampler.fetch_clear_negatives`, unchanged), but this time gated by the
    robust, NLTK-based non-methods detector (`curate/review_detector.py`) as a HARD exclusion --
    unlike Step 14b's `needs_screening`, which is deliberately diagnostic-only, review/commentary/
    meta-analysis/case-report/etc. content is dropped outright before diversification. Writes an
    interim file only -- `ingest merge-clear-negatives-filtered` is the separate step that actually
    merges into `canonical_dataset.csv`, so the fetched batch stays inspectable first.

    `raw_pool_size` (default 3000) is deliberately much larger than `target_size` (default 500) --
    on explicit instruction, a real share of ordinary biomedical literature is review/commentary/
    etc. content, so a generous raw pool is needed to net enough clean survivors."""
    started_at = time.monotonic()
    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    exclusionary_lexicon_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    try:
        selected, stats = fetch_filtered_clear_negatives(
            client,
            year_from,
            year_to,
            raw_pool_size=raw_pool_size,
            target_size=target_size,
            exclusionary_lexicon_path=exclusionary_lexicon_path,
            n_windows=n_windows,
            max_per_window=max_per_window,
        )
    finally:
        client.close()

    output_path = cfg.sampling_path("clear_negative_candidates_filtered")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    print(f"ingest.fetch-clear-negatives-filtered: {stats['fetched']} fetched, "
          f"{stats['dropped_by_gate']} dropped as likely review/non-methods, "
          f"{stats['selected']} written to {output_path} -- inspect it before running "
          "`ingest merge-clear-negatives-filtered`.")
    if stats["shortfall"]:
        print(f"ingest.fetch-clear-negatives-filtered: WARNING -- {stats['shortfall']} short of "
              f"the {target_size} target; re-run with a larger --raw-pool-size if you need the "
              "full count.")

    finish_step(
        "ingest.fetch-clear-negatives-filtered",
        inputs=[],
        outputs=[output_path],
        params={
            "year_from": year_from,
            "year_to": year_to,
            "raw_pool_size": raw_pool_size,
            "target_size": target_size,
            "n_windows": n_windows,
            "max_per_window": max_per_window,
        },
        notes=f"fetched={stats['fetched']}, dropped_by_gate={stats['dropped_by_gate']}, "
        f"survivors={stats['survivors']}, selected={stats['selected']}, shortfall={stats['shortfall']}",
        started_at=started_at,
    )


def step_ingest_merge_filtered_clear_negatives(
    cfg: PipelineConfig,
    limit: int = 500,
    source_tag: str = "clear_negative_sampler_strong_filtered_v2",
) -> None:
    """Step 19d: merges `clear_negative_candidates_filtered.csv` (`ingest
    fetch-clear-negatives-filtered`'s output) into `canonical_dataset.csv`, tagged with a distinct
    `source_name` (`"clear_negative_sampler_strong_filtered_v2"` by default) so this batch stays
    separately auditable from Step 14c's `"clear_negative_sampler_strong"` in the provenance
    breakdown chart. **Hard-stops if the interim file still contains any row flagged
    `likely_review_or_non_methods=True`** -- defense-in-depth against a stale or hand-edited
    interim file; the fetch step should never have written one, but this merge step never trusts
    that blindly (same "refuse on a wrong precondition rather than silently proceed" pattern
    `step_ingest_merge_strong_negatives` above already uses for a missing `needs_screening`
    column)."""
    started_at = time.monotonic()
    filtered_path = cfg.sampling_path("clear_negative_candidates_filtered")
    if not filtered_path.exists():
        raise ValueError(
            f"{filtered_path} does not exist -- run `ingest fetch-clear-negatives-filtered` first."
        )
    filtered = pd.read_csv(filtered_path, dtype=str)
    if "likely_review_or_non_methods" not in filtered.columns:
        raise ValueError(
            f"{filtered_path} has no likely_review_or_non_methods column -- re-run "
            "`ingest fetch-clear-negatives-filtered`."
        )
    still_flagged = int((filtered["likely_review_or_non_methods"] == "True").sum())
    if still_flagged:
        raise ValueError(
            f"{filtered_path} has {still_flagged} row(s) still flagged "
            "likely_review_or_non_methods=True -- refusing to merge; re-run "
            "`ingest fetch-clear-negatives-filtered` for a clean interim file."
        )

    selected = select_diversified_pool(filtered, limit)
    selected["source_name"] = source_tag
    selected["match_metadata"] = selected["likely_review_or_non_methods_detail"].apply(
        lambda detail: json.dumps({
            "step19d_review_gate": {
                "gate_version": "v1",
                **(json.loads(detail) if isinstance(detail, str) and detail.strip() else {"text_hits": [], "pub_type_hits": []}),
            }
        })
    )

    new_records = dataframe_to_raw_records(
        selected.drop(columns=["likely_review_or_non_methods", "likely_review_or_non_methods_detail"])
    )
    n_added = _merge_new_candidates_into_canonical(cfg, new_records)

    canonical_path = cfg.path("canonical_dataset")
    if canonical_path.exists():
        label_counts = pd.read_csv(canonical_path, dtype=str)["label"].value_counts()
        n_pos, n_neg = int(label_counts.get("positive", 0)), int(label_counts.get("negative", 0))
        ratio = f"{n_neg / n_pos:.2f}:1" if n_pos else "n/a"
        print(f"ingest.merge-clear-negatives-filtered: canonical_dataset.csv is now {n_pos} "
              f"positive / {n_neg} negative (ratio {ratio}).")

    finish_step(
        "ingest.merge-clear-negatives-filtered",
        inputs=[filtered_path],
        outputs=[canonical_path],
        params={"limit": limit, "source_tag": source_tag},
        notes=f"{len(selected)} filtered clear negatives merged, {n_added} genuinely new records "
        "merged into canonical_dataset.csv (rest already present)",
        started_at=started_at,
    )


def step_curate_flag_likely_reviews(cfg: PipelineConfig) -> None:
    """Step 19d: runs the same shared non-methods detector across the WHOLE
    `canonical_dataset.csv` (not just the new negative batch), adding
    `likely_review_or_non_methods`/`likely_review_or_non_methods_detail` as new, additive columns
    -- see `curate/state.py::flag_likely_reviews` for the exact mechanism (never touches
    `label`/any other existing column). Run this AFTER `ingest merge-clear-negatives-filtered`, so
    the flag pass covers the newly merged rows too and the profiling charts in `reporting
    profile-review-filter` are internally consistent against the final dataset."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    exclusionary_lexicon_path = resolve_path(cfg.pipeline["keywords"]["exclusionary_lexicon"])
    flag_likely_reviews(canonical_path, exclusionary_lexicon_path, canonical_path)

    finish_step(
        "curate.flag-likely-reviews",
        inputs=[canonical_path],
        outputs=[canonical_path],
        params={},
        notes="likely_review_or_non_methods flag added dataset-wide (additive columns only)",
        started_at=started_at,
    )


def step_curate_materialize_cross_curate_resolutions(cfg: PipelineConfig) -> None:
    """Step 20b: folds `cross_curate_resolution_events.csv` (the "Cross Curate Resolve" page's
    human tie-break decisions on Step 20 DeepSeek/human disagreements) into `canonical_dataset.csv`
    via `curate/state.py::materialize_cross_curate_resolutions` -- NOT `materialize_events()`,
    which would have silently turned any genuinely-changed final decision into the literal string
    `"conflict"` (see that function's docstring for the full incident). Previously this command
    bypassed `pipeline/steps.py` entirely and never got a `finish_step()` provenance entry -- fixed
    here while the underlying merge function itself was already being fixed."""
    started_at = time.monotonic()
    dataset_path = cfg.path("canonical_dataset")
    events_path = resolve_path(cfg.pipeline["curation"]["cross_curate_resolution_events"])
    llm_events_path = cfg.path("llm_classification_events")
    materialize_cross_curate_resolutions(dataset_path, events_path, llm_events_path, dataset_path)

    finish_step(
        "curate.materialize-cross-curate-resolutions",
        inputs=[dataset_path, events_path, llm_events_path],
        outputs=[dataset_path],
        params={},
        notes="cross-curate resolutions merged: label/label_confidence updated to the final "
        "decision, prior state preserved additively in new cross_curate_* columns",
        started_at=started_at,
    )


def step_ingest_backfill_canonical_metadata(cfg: PipelineConfig) -> None:
    """Fills in missing title/abstract/journal/authors/year/MeSH/etc. directly on
    `canonical_dataset.csv` for any row that has an ID (pmcid/pmid/doi) but is missing its title
    or abstract -- see METHODS_REVIEW.md Sec 8.2: 428 `registry_confirmed` positives (the DOME
    registry API dump) have no abstract at all, both a genuine information gap and a
    training-data leak (an empty abstract is currently a 100%-precision positive detector). See
    `enrich_missing_canonical_metadata` in ingest/enrich.py for the actual fill logic -- never
    overwrites an already-populated field, only fills genuine blanks."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    df = pd.read_csv(canonical_path, dtype=str)

    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        enriched_df, stats = enrich_missing_canonical_metadata(df, client)
    finally:
        client.close()

    print(f"ingest.backfill-canonical-metadata: {stats['targeted']} rows targeted (missing "
          f"title/abstract, had at least one ID to look up), {stats['found']} enriched via EPMC, "
          f"{stats['still_missing']} still missing (no EPMC match under any of their IDs).")

    backup_file(canonical_path)
    enriched_df.to_csv(canonical_path, index=False)

    finish_step(
        "ingest.backfill-canonical-metadata",
        inputs=[canonical_path],
        outputs=[canonical_path],
        params={},
        notes=f"{stats['targeted']} targeted, {stats['found']} enriched, "
        f"{stats['still_missing']} still missing after EPMC lookup",
        started_at=started_at,
    )


def _parse_llm_seed_pmids(path: Path, require_use_marked: bool = True) -> list[str]:
    """Parses the PMID seed file for Step 19c. Uses `csv.DictReader`, not manual string-splitting
    -- paper titles routinely contain commas, so a naive split-on-comma would misalign columns the
    moment a real title has one in it. Two shapes are supported, auto-detected from the header:

    1. **Curated-suggestions shape** -- header `pmid,name,url,use` (what this file ships with,
       pre-filled with 192 real EPMC search hits so Gavin doesn't start from a blank page).
       `name`/`url` exist purely for his own review in this file; the record actually used always
       gets its metadata freshly re-fetched from EPMC by pmid, never this file's `name` column.
       When `require_use_marked=True` (the original design: `curate
       add-llm-language-model-positives` merges marked rows straight in as positives, no curation
       step), only rows with a non-blank `use` value are returned. When `require_use_marked=False`
       (`ingest fetch-llm-seed-pool`'s use, for the current standard-curation-route design: fetch
       every candidate's metadata so it can be curated normally, decision not pre-made by a CSV
       column), the `use` column is ignored entirely and every row is returned.
    2. **Bare shape** -- just a `pmid` header, one PMID per line below it, nothing else. Every PMID
       listed is included regardless of `require_use_marked` (there's no `use` column to gate on).

    Duplicate PMIDs collapse to one entry, first occurrence wins."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = {(c or "").strip().lower() for c in (reader.fieldnames or [])}
        has_use_col = "use" in fieldnames
        pmids: list[str] = []
        seen: set[str] = set()
        for row in reader:
            normalized = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items() if k}
            pmid = normalized.get("pmid", "")
            if not pmid or pmid in seen:
                continue
            if require_use_marked and has_use_col and not normalized.get("use", ""):
                continue
            seen.add(pmid)
            pmids.append(pmid)
    return pmids


def step_ingest_add_llm_language_model_positives(cfg: PipelineConfig, pmids_file: Path) -> None:
    """Step 19c: adds a hand-picked batch of BERT/LLM/agentic/biological-language-model papers as
    positives, by PMID -- Gavin already knows these are genuine AI/ML-method papers; this is a
    targeted supplement to the organically-discovered positive side (Step 9-13's lexicon-driven
    bulk match), not a replacement for it. Reuses `EpmcClient.get_by_ids` and
    `bulk_match.core_result_to_raw_record` -- the exact same tiered metadata-fetch pattern Step
    13b's canonical-metadata backfill and Step 14's Phase-2 fetch already use, no new fetch logic.
    Every merged row is tagged `source_name="manual_llm_language_model_seed"` (kept distinguishable
    from every other positive-sourcing pathway in the dataset-profile provenance breakdown, Step
    18), with `match_metadata.llm_language_model_seed.pmid_as_supplied` recording the exact pmid
    string from the seed file for a direct trace back to that row.

    Goes through the exact same by-pmcid/pmid/doi dedup merge (`_merge_new_candidates_into_canonical`)
    every other ingest step uses -- **which means a PMID already present anywhere in
    canonical_dataset.csv, under any prior label, is simply skipped, not relabeled** (this merge
    helper is presence-based, not label-aware -- true for every step that calls it, not a new
    limitation introduced here). If one of these hand-picked positives happens to already be in
    the dataset labeled negative (e.g. from the bulk AI/ML match or a clear-negative batch), this
    step will NOT flip it to positive -- it prints an explicit warning naming the PMID and its
    current label instead of silently doing nothing, so that case is never invisible; resolving it
    is a separate, deliberate human decision (e.g. via the Curate app), not something this step
    attempts on its own."""
    started_at = time.monotonic()
    if not pmids_file.exists():
        raise ValueError(f"{pmids_file} does not exist -- create it and paste in PMIDs first.")

    pmids = _parse_llm_seed_pmids(pmids_file)
    if not pmids:
        print(f"ingest.add-llm-language-model-positives: {pmids_file} has no PMIDs marked for use "
              f"yet -- nothing to do. If it's the curated-suggestions shape (pmid,name,url,use), "
              f"mark the `use` column for the rows you want first.")
        return

    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        found = client.get_by_ids(pmids, id_type="pmid")
    finally:
        client.close()

    still_missing = [pmid for pmid in pmids if pmid not in found]
    for pmid in still_missing:
        print(f"ingest.add-llm-language-model-positives: WARNING -- PMID {pmid} not found in EPMC; skipped.")

    canonical_path = cfg.path("canonical_dataset")
    if canonical_path.exists() and found:
        existing_df = pd.read_csv(canonical_path, dtype=str)
        for pmid, result in found.items():
            ids = {pmid, clean_pmcid(result.get("pmcid")), clean_doi(result.get("doi"))} - {None}
            match = existing_df[
                existing_df["pmid"].isin(ids) | existing_df["pmcid"].isin(ids) | existing_df["doi"].isin(ids)
            ]
            if not match.empty:
                existing_label = match.iloc[0]["label"]
                print(
                    f"ingest.add-llm-language-model-positives: WARNING -- PMID {pmid} already "
                    f"exists in canonical_dataset.csv with label={existing_label!r} under a "
                    f"different source; this positive determination will be SKIPPED (not "
                    f"relabeled) by the merge below -- resolve manually if that's not intended."
                )

    new_records = []
    for pmid, result in found.items():
        record = core_result_to_raw_record(
            result,
            source_name="manual_llm_language_model_seed",
            source_file=str(pmids_file),
            label="positive",
            label_confidence="human_curated",
        )
        record.match_metadata = {"llm_language_model_seed": {"pmid_as_supplied": pmid}}
        new_records.append(record)

    n_added = _merge_new_candidates_into_canonical(cfg, new_records)

    print(f"ingest.add-llm-language-model-positives: {len(pmids)} PMIDs marked for use, {len(found)} "
          f"found in EPMC, {len(still_missing)} not found. {n_added} genuinely new records merged "
          f"into canonical_dataset.csv as positive/human_curated ({len(found) - n_added} already "
          f"present under another source -- skipped, not duplicated; see any WARNING lines above "
          f"for which ones and their current label).")

    finish_step(
        "ingest.add-llm-language-model-positives",
        inputs=[pmids_file],
        outputs=[canonical_path],
        params={"pmids_supplied": len(pmids)},
        notes=f"{len(pmids)} PMIDs marked for use, {len(found)} found in EPMC ({len(still_missing)} "
        f"not found), {n_added} genuinely new positive records merged",
        started_at=started_at,
    )


def step_ingest_fetch_llm_seed_pool(cfg: PipelineConfig) -> None:
    """Step 19c (standard-curation-route build): fetches full EPMC metadata (title/abstract/
    journal/authors/year/MeSH/pub types/etc.) for every PMID in the Step 19c seed file
    (`llm_language_model_seed_pmids.csv`, 192 real, individually-verified candidates -- see
    STEPS_Progress.md's "Exact search provenance" section for how they were found), regardless of
    the file's `use` column -- that column drove the earlier, superseded design where marked rows
    were merged straight in as positives; the current design is to curate every candidate through
    the normal Curate-app P/N/U/S workflow instead (same as every other batch in this dataset), so
    every candidate needs its full metadata fetched for display, not just the ones pre-marked.

    Writes a `RawRecord`-shaped pool file (`raw_records_to_dataframe`, the same serialization
    `bulk_candidates.csv`/`clear_negative_candidates.csv` already use) so it's a drop-in
    `bulk_pool_path` for `materialize_events()` later -- `label="unlabeled"`/
    `label_confidence="unscored"` on every row (nothing is pre-judged positive or negative; that's
    exactly what the curation pass decides), `source_name="manual_llm_language_model_seed"` (kept
    distinguishable from every other positive/negative-sourcing pathway, same as the earlier
    design's provenance tag)."""
    started_at = time.monotonic()
    seed_path = cfg.path("llm_seed_pmids")
    if not seed_path.exists():
        raise ValueError(f"{seed_path} does not exist.")

    pmids = _parse_llm_seed_pmids(seed_path, require_use_marked=False)
    if not pmids:
        print(f"ingest.fetch-llm-seed-pool: {seed_path} has no PMIDs -- nothing to fetch.")
        return

    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        found = client.get_by_ids(pmids, id_type="pmid")
    finally:
        client.close()

    still_missing = [pmid for pmid in pmids if pmid not in found]
    for pmid in still_missing:
        print(f"ingest.fetch-llm-seed-pool: WARNING -- PMID {pmid} not found in EPMC; skipped.")

    records = [
        core_result_to_raw_record(
            result,
            source_name="manual_llm_language_model_seed",
            source_file=str(seed_path),
            label="unlabeled",
            label_confidence="unscored",
        )
        for result in found.values()
    ]

    pool_path = cfg.path("llm_seed_candidate_pool")
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    raw_records_to_dataframe(records).to_csv(pool_path, index=False)

    print(f"ingest.fetch-llm-seed-pool: {len(pmids)} PMIDs in seed file, {len(found)} found in "
          f"EPMC, {len(still_missing)} not found. Pool written to {pool_path} -- open the "
          f"'LLM Seed Review' page in the Curate app to review them.")

    finish_step(
        "ingest.fetch-llm-seed-pool",
        inputs=[seed_path],
        outputs=[pool_path],
        params={"pmids_in_seed_file": len(pmids)},
        notes=f"{len(pmids)} PMIDs in seed file, {len(found)} found in EPMC "
        f"({len(still_missing)} not found)",
        started_at=started_at,
    )


_DATASET_PROFILE_README = """# Dataset profile snapshot

**This is a mid-pipeline snapshot, not the final dataset profile.**

It was generated by `dome-triage reporting profile-dataset` (Step 18) after Steps 1-17 (sourcing,
dedup, keyword lexicon, bulk AI/ML match, stratified sampling, human curation of the main queue,
strong-negative augmentation, and curation-criteria consolidation) but **before**:

- Step 19: re-review of the ~3,356-record pre-app curation cohort
- Step 20: second-curator (LLM) blind classification and human/Claude agreement resolution

Labels, journal mix, and score distributions shown here can still change as a result of those
later steps. Re-run `dome-triage reporting profile-dataset` after each one to get an updated
snapshot -- this directory is overwritten in place on every run; `data/provenance.jsonl` has the
full history of every regeneration, and `profile_metadata.json` in this directory records this
run's exact git commit, timestamp, and row counts.

## Charts

- `label_overview.png` -- positive/negative/skipped/undeterminable counts.
- `journal_diversity.png` -- how many distinct journals contribute how many records (bucketed:
  1, 2-5, 6-10, ...), not a top-20 list -- shows the long-tail concentration directly.
- `provenance_class_breakdown.png` -- every record assigned to exactly one of four mutually
  exclusive categories (Streamlit Curated, Manual Curated Pre-App, DOME Registry, EPMC Negatives);
  see the chart's own caption for the exact classification rule and a direct cross-check on
  whether the two DOME registry snapshots (`dome_registry_231_gold`/`dome_registry_222_gold`)
  double-count any records (they don't).
- `bm25_score_distribution.png` -- BM25 lexicon score histogram by label, with the validated
  Youden threshold marked, for records that came through the bulk-match pool. Records added via
  other pathways (e.g. Step 14c's live-EPMC clear negatives) were never scored by the bulk match
  and are excluded from this specific chart only, not from the dataset.
- `year_distribution.png` -- curated positive/negative/skipped/undeterminable counts per
  publication year, 2000-2026 (the dataset has a small number of pre-2000 records; excluded from
  this chart's range, noted in its caption).
- `year_coverage_vs_bulk_pool.png` -- the curated sample's per-year counts against the full
  ~745k-record AI/ML-matched bulk pool it was drawn from (log scale, 2000-2026), showing what
  fraction of each year's true population is actually represented in the curated dataset.
- `year_coverage_vs_bulk_pool_linear.png` -- the same comparison on a linear y-axis. True to
  scale, so the curated bars are visually dwarfed by the pool in every year; use the log-scale
  version above to read the curated sample's own shape across years.
- `bm25_youden_performance.png` (only if at least one record has been decided via the Curate app
  and a Youden threshold is available) -- how well the Youden threshold's call actually agreed
  with the human curator, measured over every record actually decided through the Curate app
  (`curation_events.csv`) with a definitive label and a BM25 score, quartile-banded fresh over
  that exact population (matching how the live app itself computes Q1-Q4, not a stale one-time
  snapshot): a confusion matrix (human label vs. BM25 call) plus accuracy by quartile, each
  quartile's real BM25 score range/average, and precision/recall/specificity in the caption.
"""


def step_reporting_profile_dataset(cfg: PipelineConfig) -> None:
    """Step 18: profiles canonical_dataset.csv as a set of charts plus a metadata/README pair, so
    the dataset's current shape (label balance, journal diversity, provenance mix, BM25 score
    distribution, year coverage vs. the full bulk pool) is visible as artifacts instead of only ad
    hoc queries. Writes to a fixed `data/processed/dataset_profile/` directory, overwritten on
    every run -- `data/provenance.jsonl` already keeps a full history of every regeneration, so
    there's no need for timestamped-folder proliferation here. Deliberately a mid-pipeline
    snapshot: generated before Steps 19-20's re-review and second-curator passes, not the final
    profile of the dataset that will actually train the model -- stated in the generated
    README.md too, not just here, since that's the artifact someone will actually be reading
    later. The BM25 chart reuses `curate/bulk_scores.py`'s existing lookup/annotate/threshold
    helpers rather than reimplementing the join (see that module's docstring for why
    `Series.map()` is deliberately avoided at the 2.07M-entry lookup scale)."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    dataset = pd.read_csv(canonical_path, dtype=str)

    output_dir = cfg.path("dataset_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = resolve_path(cfg.pipeline["curation"]["events_log"])
    if events_path.exists():
        streamlit_curated_ids = set(pd.read_csv(events_path, usecols=["record_id"], dtype=str)["record_id"])
    else:
        streamlit_curated_ids = set()

    label_counts = plot_label_overview(dataset, output_dir / "label_overview.png")
    plot_journal_diversity(dataset, output_dir / "journal_diversity.png")
    plot_provenance_class_breakdown(
        dataset, output_dir / "provenance_class_breakdown.png", streamlit_curated_ids
    )

    scored_path = cfg.sampling_path("bulk_candidates_scored")
    lookup = load_bulk_score_lookup(scored_path)
    dataset_scored = annotate_bulk_scores(dataset, lookup)
    bakeoff_report_path = cfg.path("processed_dir") / "scoring_bakeoff_report.csv"
    threshold = load_youden_threshold(bakeoff_report_path)
    plot_bm25_score_distribution(dataset_scored, output_dir / "bm25_score_distribution.png", threshold)

    plot_year_distribution(dataset, output_dir / "year_distribution.png")

    if scored_path.exists():
        bulk_pool_years = pd.read_csv(scored_path, usecols=["year"], dtype=str)["year"]
    else:
        bulk_pool_years = pd.Series([], dtype=str)
    plot_year_coverage_vs_bulk_pool(dataset, bulk_pool_years, output_dir / "year_coverage_vs_bulk_pool.png")
    plot_year_coverage_vs_bulk_pool_linear(
        dataset, bulk_pool_years, output_dir / "year_coverage_vs_bulk_pool_linear.png"
    )

    # BM25 Youden-threshold performance: how well the Youden threshold's call actually agreed with
    # the human curator, measured over every record actually decided through the Streamlit Curate
    # app (record_id in curation_events.csv) that currently has a definitive label and a BM25
    # score.
    #
    # CORRECTED 2026-08-17 -- the first version of this chart was WRONG, and Gavin was right to
    # push back on it. It used only `stratified_candidate_pool.csv` (Step 13's one-time, static
    # 2,328-row snapshot) as "the Q1-Q4 queue", landing at ~580-597/quartile -- but the live
    # Curate app's own quartile bands (`state.py::CurationSession._scored_pool`/
    # `score_band_summary`, via `build_strata`) are NOT read from that static file at all: they're
    # recomputed FRESH via `pd.qcut` over whatever canonical_dataset.csv's reviewable population
    # is at the time, every time the app loads -- and Gavin also curated real records beyond the
    # static stratified sample, via the app's "Full AI/ML bulk pool" browsing mode. Verified
    # directly: `curation_events.csv` has 2,862 distinct decided record_ids; only 2,327 of those
    # are in `stratified_candidate_pool.csv` -- 535 decisions (about a fifth of all of them) came
    # from that other browsing mode and were silently excluded by the first version of this chart.
    # Re-deriving quartiles the same way the app actually does -- fresh `build_strata()` over ALL
    # 2,814 streamlit-curated records with a definitive label and a BM25 score -- lands at
    # 703-704 per quartile, matching Gavin's own recollection (~700/quartile) exactly. The
    # `record_id_from_ids`/`stratified_candidate_pool.csv` join this used to require is gone
    # entirely: `dataset_scored` (already computed above for the BM25 score-distribution chart)
    # and `streamlit_curated_ids` (already computed above for the provenance chart) are reused
    # directly, so this chart's population can never again silently drift from either of those.
    youden_chart_produced = False
    if streamlit_curated_ids and threshold is not None:
        youden_pool = dataset_scored[
            dataset_scored["record_id"].isin(streamlit_curated_ids)
            & dataset_scored["label"].isin(["positive", "negative"])
            & dataset_scored["bulk_match_score"].notna()
        ].copy()
        if not youden_pool.empty:
            youden_pool["bulk_match_score"] = youden_pool["bulk_match_score"].astype(float)
            youden_strata = build_strata(youden_pool, score_col="bulk_match_score", n_score_bands=4)
            youden_pool["quartile"] = youden_strata["match_score_band__bulk_match_score"] + 1
            youden_pool["match_score__bm25"] = youden_pool["bulk_match_score"]
            youden_pool["match_classification__bm25"] = youden_pool["bulk_match_score"].apply(
                lambda score: "positive" if score >= threshold else "negative"
            )
            plot_bm25_youden_performance(youden_pool, output_dir / "bm25_youden_performance.png")
            youden_chart_produced = True

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "total_rows": len(dataset),
        "label_counts": label_counts.to_dict(),
    }
    (output_dir / "profile_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "README.md").write_text(_DATASET_PROFILE_README)

    print(f"reporting.profile-dataset: {len(dataset)} rows profiled "
          f"({dict(label_counts)}), {'8' if youden_chart_produced else '7'} charts + "
          f"profile_metadata.json + README.md written to {output_dir}")

    finish_step(
        "reporting.profile-dataset",
        inputs=[canonical_path] + ([scored_path] if scored_path.exists() else [])
        + ([events_path] if events_path.exists() else []),
        outputs=[
            output_dir / "label_overview.png",
            output_dir / "journal_diversity.png",
            output_dir / "provenance_class_breakdown.png",
            output_dir / "bm25_score_distribution.png",
            output_dir / "year_distribution.png",
            output_dir / "year_coverage_vs_bulk_pool.png",
            output_dir / "year_coverage_vs_bulk_pool_linear.png",
        ]
        + ([output_dir / "bm25_youden_performance.png"] if youden_chart_produced else [])
        + [
            output_dir / "profile_metadata.json",
            output_dir / "README.md",
        ],
        params={},
        notes=f"{len(dataset)} rows profiled ({dict(label_counts)}); mid-pipeline snapshot, "
        "before Steps 19-20's re-review and second-curator passes",
        started_at=started_at,
    )


_ORIGINAL_COHORT_SECOND_REVIEW_README = """# Original-cohort second-review profile

Step 19b (additive variant) re-reviewed every record in the ~3,356-record pre-app curation cohort
that was a trusted `negative` -- 1,907 records, all independently re-decided via the Curate app's
"Original Cohort Review" page. Those decisions were merged into `canonical_dataset.csv` via
`curate merge-original-cohort-second-review` -- **additively, as new `original_cohort_review_*`
columns, never overwriting the original `label`/`label_confidence`** (see that command's docstring
in `curate/state.py::merge_original_cohort_second_review` for the full column list and reasoning).
This folder is a diagnostic profile of that second review, not a relabeling -- nothing in
`canonical_dataset.csv`'s original `label` column was changed to produce any chart here.

## Charts

- `second_review_decision_breakdown.png` -- how many of the 1,907 re-reviewed negatives were
  reconfirmed negative vs. flipped to positive vs. undeterminable on independent second look.
- `second_review_bm25_confusion.png` -- confusion matrix of the second-review (corrected) label
  against the BM25 Youden-threshold call, restricted to records with a definitive second-review
  label and a BM25 score. Caption compares accuracy against this corrected ground truth vs. the
  stale pre-review assumption that every one of these records was still "negative".
- `second_review_bm25_score_distribution.png` -- BM25 score histograms for the reconfirmed-negative
  group vs. the flipped-to-positive group, with the Youden threshold marked -- shows whether BM25
  was already picking up signal in the records the original manual curation missed.

Generated by `dome-triage reporting profile-original-cohort-second-review` -- overwritten in place
on every run; `profile_metadata.json` in this directory records the exact git commit, timestamp,
and counts for this run.
"""


def step_reporting_profile_original_cohort_second_review(cfg: PipelineConfig) -> None:
    """Step 19b's diagnostic companion: profiles the original-cohort second review (the 1,907
    pre-app negatives re-reviewed via the Curate app's "Original Cohort Review" page, merged
    additively into canonical_dataset.csv's original_cohort_review_* columns by `curate
    merge-original-cohort-second-review`) as its own small chart set in a dedicated folder --
    distinct from Step 18's whole-dataset dataset_profile/, since this reports on one specific
    sub-cohort/sub-question (did the second review agree with the original curation, and did BM25
    already "see" the disagreements), not the dataset as a whole. Reuses `curate/bulk_scores.py`'s
    existing lookup/annotate/threshold helpers for the BM25 join, same as Step 18."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    dataset = pd.read_csv(canonical_path, dtype=str)

    output_dir = cfg.path("original_cohort_second_review_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if (
        "original_cohort_review_label" not in dataset.columns
        or dataset["original_cohort_review_label"].dropna().empty
    ):
        print(
            "reporting.profile-original-cohort-second-review: no original_cohort_review_label "
            "data found in canonical_dataset.csv -- run `curate merge-original-cohort-second-review` "
            "first. Nothing written.",
            flush=True,
        )
        return

    decision_counts = plot_second_review_decision_breakdown(
        dataset, output_dir / "second_review_decision_breakdown.png"
    )

    scored_path = cfg.sampling_path("bulk_candidates_scored")
    lookup = load_bulk_score_lookup(scored_path)
    dataset_scored = annotate_bulk_scores(dataset, lookup)
    bakeoff_report_path = cfg.path("processed_dir") / "scoring_bakeoff_report.csv"
    threshold = load_youden_threshold(bakeoff_report_path)

    confusion_produced = False
    reviewed = dataset_scored[dataset_scored["original_cohort_review_label"].notna()].copy()
    if threshold is not None and not reviewed.empty:
        ground_truth_pool = reviewed[
            reviewed["original_cohort_review_label"].isin(["positive", "negative"])
            & reviewed["bulk_match_score"].notna()
        ].copy()
        if not ground_truth_pool.empty:
            ground_truth_pool["bulk_match_score"] = ground_truth_pool["bulk_match_score"].astype(float)
            ground_truth_pool["match_score__bm25"] = ground_truth_pool["bulk_match_score"]
            ground_truth_pool["match_classification__bm25"] = ground_truth_pool["bulk_match_score"].apply(
                lambda score: "positive" if score >= threshold else "negative"
            )
            plot_second_review_bm25_confusion(
                ground_truth_pool, output_dir / "second_review_bm25_confusion.png"
            )
            plot_second_review_bm25_score_distribution(
                ground_truth_pool, output_dir / "second_review_bm25_score_distribution.png", threshold
            )
            confusion_produced = True

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "total_reviewed": int(dataset["original_cohort_review_label"].notna().sum()),
        "decision_counts": decision_counts.to_dict(),
    }
    (output_dir / "profile_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "README.md").write_text(_ORIGINAL_COHORT_SECOND_REVIEW_README)

    print(
        f"reporting.profile-original-cohort-second-review: {metadata['total_reviewed']} reviewed "
        f"records profiled ({dict(decision_counts)}), "
        f"{'3' if confusion_produced else '1'} charts + profile_metadata.json + README.md written "
        f"to {output_dir}",
        flush=True,
    )

    finish_step(
        "reporting.profile-original-cohort-second-review",
        inputs=[canonical_path]
        + ([scored_path] if scored_path.exists() else [])
        + ([bakeoff_report_path] if bakeoff_report_path.exists() else []),
        outputs=[output_dir / "second_review_decision_breakdown.png"]
        + (
            [
                output_dir / "second_review_bm25_confusion.png",
                output_dir / "second_review_bm25_score_distribution.png",
            ]
            if confusion_produced
            else []
        )
        + [output_dir / "profile_metadata.json", output_dir / "README.md"],
        params={},
        notes=f"{metadata['total_reviewed']} reviewed records profiled ({dict(decision_counts)})",
        started_at=started_at,
    )


_REVIEW_FILTER_PROFILE_README = """# Review/non-methods flag profile (Step 19d)

Profiles `canonical_dataset.csv`'s `likely_review_or_non_methods` column -- added dataset-wide by
`curate flag-likely-reviews`, using the same NLTK-based non-methods detector
(`curate/review_detector.py`) that gates Step 19d's new ~500-negative batch
(`ingest fetch-clear-negatives-filtered` / `ingest merge-clear-negatives-filtered`). This is a
diagnostic profile, not a relabeling -- nothing in `canonical_dataset.csv`'s original `label`
column is touched by either the flag column or this profile.

## Charts

- `label_vs_review_flag_breakdown.png` -- grouped bar: how many of each `label`
  (positive/negative/skipped/undeterminable) are flagged `likely_review_or_non_methods=True` vs.
  not, with exact counts/percentages.
- `review_flag_term_frequency.png` -- the top text terms / EPMC `pub_types` tags actually
  triggering the flag, parsed from `likely_review_or_non_methods_detail` -- a direct check that
  the gate is catching genuine review/non-methods content, not a black box.

Generated by `dome-triage reporting profile-review-filter` -- overwritten in place on every run;
`profile_metadata.json` records the exact git commit, timestamp, and counts for this run.
"""


def step_reporting_profile_review_filter(cfg: PipelineConfig) -> None:
    """Step 19d's diagnostic companion: profiles the `likely_review_or_non_methods` flag
    (`curate flag-likely-reviews`'s output) dataset-wide, in its own dedicated folder -- distinct
    from Step 18's whole-dataset `dataset_profile/`, since this reports on one specific
    sub-question (how much of each label is flagged, and by what), not the dataset's overall
    shape."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    dataset = pd.read_csv(canonical_path, dtype=str)

    output_dir = cfg.path("review_filter_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    if "likely_review_or_non_methods" not in dataset.columns:
        print(
            "reporting.profile-review-filter: no likely_review_or_non_methods column found in "
            "canonical_dataset.csv -- run `curate flag-likely-reviews` first. Nothing written.",
            flush=True,
        )
        return

    breakdown = plot_label_vs_review_flag_breakdown(dataset, output_dir / "label_vs_review_flag_breakdown.png")
    plot_review_flag_term_frequency(dataset, output_dir / "review_flag_term_frequency.png")

    flagged_series = dataset["likely_review_or_non_methods"].dropna().astype(str)
    n_flagged = int((flagged_series == "True").sum())
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "total_rows": len(dataset),
        "n_scored": int(len(flagged_series)),
        "n_flagged": n_flagged,
        "label_vs_flag_breakdown": {
            str(label): {"not_flagged": int(row["not_flagged"]), "flagged": int(row["flagged"])}
            for label, row in breakdown.iterrows()
        },
    }
    (output_dir / "profile_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "README.md").write_text(_REVIEW_FILTER_PROFILE_README)

    print(
        f"reporting.profile-review-filter: {metadata['n_scored']} rows scored, {n_flagged} "
        f"flagged likely_review_or_non_methods=True, 2 charts + profile_metadata.json + "
        f"README.md written to {output_dir}",
        flush=True,
    )

    finish_step(
        "reporting.profile-review-filter",
        inputs=[canonical_path],
        outputs=[
            output_dir / "label_vs_review_flag_breakdown.png",
            output_dir / "review_flag_term_frequency.png",
            output_dir / "profile_metadata.json",
            output_dir / "README.md",
        ],
        params={},
        notes=f"{metadata['n_scored']} rows scored, {n_flagged} flagged",
        started_at=started_at,
    )


def _deepseek_client(concurrency: int = 20) -> DeepSeekClient:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set -- see STEPS_Progress.md Step 20's security setup "
            "(add it to a repo-root .env, already gitignored)."
        )
    # The connection pool must cover the caller's thread-pool width or the extra threads just
    # queue on a connection -- see create_session's comment. Defaults to the historical 50 for
    # every caller that doesn't run a wide pool.
    return DeepSeekClient(api_key=api_key, pool_maxsize=max(50, concurrency))


def _criteria_text_and_hash() -> tuple[Path, str, str]:
    criteria_path = resolve_path("curation_criteria/CRITERIA.md")
    if not criteria_path.exists():
        raise ValueError(f"{criteria_path} does not exist.")
    text = load_criteria_text(criteria_path)
    return criteria_path, text, criteria_sha256(text)


def _deepseek_budget_cfg(cfg: PipelineConfig) -> dict:
    return cfg.pipeline["budget"]["deepseek_second_curator"]


@contextlib.contextmanager
def _events_file_lock(events_path: Path, ignore_lock: bool = False):
    """Exclusive advisory lock on an event log, held for the run.

    Why this exists, precisely: a detached `docker compose run` against this repo's `pipeline`
    service exits NON-ZERO while its container keeps running (the service sets `tty: true`). A
    retry loop that reads that exit code as failure therefore starts a second container on top of
    a live one. That happened on 2026-09-03 -- **ten containers ran concurrently**, all enriching
    overlapping records into one events file, producing 5,871 rows for 3,359 records and **$9.63 of
    duplicated paid work**. Nothing was lost (events stream to disk per record), but everything
    past the first container was paid for twice.

    Documentation did not prevent it and would not have. `flock` does: the second process cannot
    acquire the lock and exits before spending anything. The kernel releases it when the holder
    dies -- including a `kill -9` or a container the daemon loses -- so a crash never strands a
    lock that blocks the legitimate retry.

    Advisory, not mandatory: it constrains callers that take it, which is every paid run here.
    """
    lock_path = events_path.with_suffix(events_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "(holder wrote no details)"
            if not ignore_lock:
                raise ValueError(
                    f"{events_path.name} is already being written by another run: {holder}. "
                    f"Refusing to start -- two processes on one event log means paying twice for "
                    f"the same records. Wait for it, or if that process is genuinely gone (a "
                    f"container the daemon lost), re-run with --ignore-lock."
                )
            print(f"llm-classify: --ignore-lock -- proceeding despite the lock held by {holder}. "
                  f"If that run is still alive, both are now paying for the same records.")
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} host={socket.gethostname()} "
                     f"started={datetime.now(timezone.utc).isoformat()}\n")
        handle.flush()
        yield
    finally:
        handle.close()  # releases the lock


def _stream_classify_events_to_disk(events_path: Path, event_iterator, columns: list[str] | None = None) -> int:
    """Writes each yielded classification event to `events_path` immediately, one row at a time --
    NOT buffered into a list and written only after the whole run finishes. Fixes a real, confirmed
    incident: a 39-minute, 840/1000-call real paid run died on a transient network error
    (`ChunkedEncodingError`) with the old batch-at-the-end code, and because nothing had been
    written to disk yet, all 840 already-paid-for results were lost outright -- `llm_classification
    _events.csv` didn't even exist afterward. With this streaming write, any exception from
    `event_iterator` (network error, etc.) now only ever loses the one in-flight call; every event
    already yielded is already safely on disk, and `classify_records`'s own resumability check
    (keyed on tier/mode/prompt_version/criteria_sha256) will skip those records -- not re-pay for
    them -- the next time the exact same command is re-run. Returns the count actually written,
    which may be less than the full sample if the iterator raised partway through."""
    events_path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns if columns is not None else llm_runner.EVENT_COLUMNS
    n_written = 0
    for event in event_iterator:
        pd.DataFrame([event], columns=columns).to_csv(
            events_path, mode="a", header=not events_path.exists(), index=False
        )
        n_written += 1
    return n_written


def step_llm_classify_validate_criteria(cfg: PipelineConfig, tier: str = "flash", confirmed: bool = False) -> None:
    """Step 20's explicit "first" requirement: is the prompt actually working, checked BEFORE any
    calibration or real-sample spend. Runs BOTH the primary (3-way) and forced-choice (2-way)
    prompt variants over the hand-picked `curation_criteria/validation_fixtures.csv` fixtures --
    flash tier only in practice, since the same prompt drives both DeepSeek tiers by construction.
    Prints a pass/fail table per variant, the primary variant's undetermined rate (the number that
    decides which variant drives the real 500+500 sample -- see STEPS_Progress.md Step 20's
    decision rule), and the full constructed prompt for one record so you can visually confirm the
    criteria text really made it in."""
    started_at = time.monotonic()
    fixtures_path = resolve_path("curation_criteria/validation_fixtures.csv")
    if not fixtures_path.exists():
        raise ValueError(
            f"{fixtures_path} does not exist -- hand-pick ~12-15 real records with an "
            "expected_classification first (see STEPS_Progress.md Step 20)."
        )
    fixtures = pd.read_csv(fixtures_path, dtype=str)
    criteria_path, criteria_text, _ = _criteria_text_and_hash()

    budget_cfg = _deepseek_budget_cfg(cfg)
    spend_log_path = resolve_path(budget_cfg["spend_log"])
    # A small, fixed placeholder ceiling for this tiny first live check -- NOT a per-token
    # estimate (no calibration data exists yet at this point in the sequence).
    llm_budget.check_cap(spend_log_path, budget_cfg["total_cap_usd"], 0.50, confirmed)

    client = _deepseek_client()
    try:
        result = llm_runner.run_criteria_validation(fixtures, tier, client, criteria_text)
    finally:
        client.close()

    output_path = cfg.path("llm_classify_validation_results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))

    for variant in ("primary", "forced_choice"):
        subset = result[result["variant"] == variant]
        match_rate = subset["match"].mean() * 100 if len(subset) else 0.0
        print(f"llm-classify.validate-criteria: variant={variant} match_rate={match_rate:.1f}% "
              f"({int(subset['match'].sum())}/{len(subset)})")
    primary = result[result["variant"] == "primary"]
    undetermined_rate = (primary["actual"] == "undeterminable").mean() * 100 if len(primary) else 0.0
    print(f"llm-classify.validate-criteria: primary-variant undetermined rate = {undetermined_rate:.1f}% "
          "-- see STEPS_Progress.md Step 20's decision rule for which variant should drive the real "
          "500+500 sample.")

    if len(fixtures):
        first = fixtures.iloc[0]
        example_prompt = build_prompt(
            {"title": first.get("title"), "abstract": first.get("abstract"),
             "journal": first.get("journal"), "year": first.get("year")},
            criteria_text,
        )
        print("\n--- Example constructed prompt (first fixture, primary variant) ---")
        print(example_prompt[0]["content"])
        print("\n--- User message ---")
        print(example_prompt[1]["content"])

    llm_budget.log_spend(
        spend_log_path, "llm-classify.validate-criteria", tier, "primary+forced_choice",
        len(fixtures) * 2, estimated_usd=0.50, actual_usd=None,
        confirmed_by=cfg.pipeline["curation"]["default_curator"],
    )

    finish_step(
        "llm-classify.validate-criteria",
        inputs=[fixtures_path, criteria_path],
        outputs=[output_path],
        params={"tier": tier},
        notes=f"{len(fixtures)} fixtures x 2 variants -- see printed match rates and undetermined rate",
        started_at=started_at,
    )


def step_llm_classify_calibrate(
    cfg: PipelineConfig, tier: str, n: int = 8, mode: str = "primary", confirmed: bool = False
) -> None:
    """Step 20: fires `n` REAL, live calls against `tier` (default `mode="primary"`) through the
    exact real prompt-building path, logging real token usage to
    `second_curator_calibration_log.csv`. No dollar figure is computed or checked here -- this IS
    the ground truth `project-cost` reads real numbers from; check your real DeepSeek dashboard
    balance deduction for this exact batch yourself, then pass it to `llm-classify project-cost`."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    dataset = pd.read_csv(canonical_path, dtype=str)
    pool = dataset[dataset["label"].isin(["positive", "negative"])]
    sample = pool.sample(n=min(n, len(pool)), random_state=1)

    criteria_path, criteria_text, _ = _criteria_text_and_hash()
    budget_cfg = _deepseek_budget_cfg(cfg)
    spend_log_path = resolve_path(budget_cfg["spend_log"])
    llm_budget.check_cap(spend_log_path, budget_cfg["total_cap_usd"], 0.20, confirmed)

    batch_id = f"calibrate_{tier}_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    client = _deepseek_client()
    try:
        result = llm_cost.run_calibration_batch(sample, tier, client, criteria_text, batch_id, n=n, mode=mode)
    finally:
        client.close()

    output_path = cfg.path("second_curator_calibration_log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not output_path.exists()
    result.to_csv(output_path, mode="a", header=header_needed, index=False)

    print(f"llm-classify.calibrate: {len(result)} real calls made (tier={tier}, mode={mode}), "
          f"avg prompt_tokens={result['prompt_tokens'].mean():.0f}, "
          f"avg completion_tokens={result['completion_tokens'].mean():.0f}. Check your real "
          "DeepSeek dashboard balance deduction for this batch, then run `llm-classify "
          "project-cost --observed-usd-spent-... <real number>`.")

    llm_budget.log_spend(
        spend_log_path, "llm-classify.calibrate", tier, mode, len(result),
        estimated_usd=0.20, actual_usd=None, confirmed_by=cfg.pipeline["curation"]["default_curator"],
    )

    finish_step(
        "llm-classify.calibrate",
        inputs=[canonical_path, criteria_path],
        outputs=[output_path],
        params={"tier": tier, "mode": mode, "n": n},
        notes=f"{len(result)} real calibration calls, batch_id={batch_id}",
        started_at=started_at,
    )


def step_llm_classify_project_cost(
    cfg: PipelineConfig,
    target_n: int,
    observed_usd_spent_flash: float | None = None,
    observed_usd_spent_pro: float | None = None,
    mode: str = "primary",
) -> dict:
    """Pure read/print -- no API call, no spend. Prints the real, calibration-derived $/record and
    projected cost for `target_n` records, for each tier an observed dollar figure was given for.
    This is the number `llm-classify classify`'s `--estimated-usd` should be copied from."""
    calibration_path = cfg.path("second_curator_calibration_log")
    if not calibration_path.exists():
        raise ValueError(f"{calibration_path} does not exist -- run `llm-classify calibrate` first.")
    calibration_log = pd.read_csv(calibration_path)

    projections: dict = {}
    for tier, observed in (("flash", observed_usd_spent_flash), ("pro", observed_usd_spent_pro)):
        if observed is None:
            continue
        projection = llm_cost.project_cost(calibration_log, tier, observed, target_n, mode=mode)
        projections[tier] = projection
        print(
            f"llm-classify.project-cost: tier={tier} mode={mode} batch_id={projection['batch_id']} -- "
            f"${projection['usd_per_record']:.6f}/record from {projection['n_calibration_calls']} "
            f"real calibration calls in THIS batch only (${projection['observed_usd_spent']:.4f} "
            f"spent). Projected cost for {target_n} records: ${projection['projected_usd_for_target_n']:.2f}."
        )
    if not projections:
        print(
            "llm-classify.project-cost: pass --observed-usd-spent-flash and/or "
            "--observed-usd-spent-pro (the real dollar amount deducted from your DeepSeek "
            "dashboard for that tier's calibration batch)."
        )
    return projections


def step_llm_classify_sample(
    cfg: PipelineConfig, n_positive: int = 500, n_negative: int = 500, random_state: int = 42
) -> None:
    """Step 20: draws the shared blind 500+500 sample once (both tiers judge the same records) --
    plain random across the trusted human-curated population, not oversampled toward Step 19d's
    flagged-likely-review negatives (Gavin's confirmed choice). Excludes any record_id already
    used as a validate-criteria fixture, so the "easy" hand-picked fixture set can never double-dip
    into this real evaluation sample."""
    started_at = time.monotonic()
    canonical_path = cfg.path("canonical_dataset")
    dataset = pd.read_csv(canonical_path, dtype=str)

    fixtures_path = resolve_path("curation_criteria/validation_fixtures.csv")
    exclude_ids: set = set()
    if fixtures_path.exists():
        exclude_ids = set(pd.read_csv(fixtures_path, dtype=str)["record_id"])

    sample = draw_blind_sample(
        dataset, n_positive=n_positive, n_negative=n_negative, random_state=random_state,
        exclude_ids=exclude_ids,
    )

    output_path = cfg.path("second_curator_sample")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

    print(f"llm-classify.sample: drew {len(sample)} records ({n_positive} positive / "
          f"{n_negative} negative, plain random, seed={random_state}) to {output_path}.")

    finish_step(
        "llm-classify.sample",
        inputs=[canonical_path] + ([fixtures_path] if fixtures_path.exists() else []),
        outputs=[output_path],
        params={"n_positive": n_positive, "n_negative": n_negative, "random_state": random_state},
        notes=f"{len(sample)} records drawn, plain random (not oversampled toward flagged negatives)",
        started_at=started_at,
    )


def step_llm_classify_compare_runs(cfg: PipelineConfig, path_a: str, path_b: str) -> dict:
    """A/B comparison of two classification event logs over their shared records -- used to prove
    an infrastructure change (concurrency, pooling, output path) did not change what the model
    decides. Read-only, no spend."""
    a, b = resolve_path(path_a), resolve_path(path_b)
    result = run_comparison.compare_runs(a, b)
    run_comparison.print_comparison(result, a.name, b.name)
    return result


def step_llm_classify_classify(
    cfg: PipelineConfig,
    tier: str,
    scope: str = "sample",
    estimated_usd: float | None = None,
    confirmed: bool = False,
    concurrency: int = 20,
    mode: str = "primary",
    limit: int | None = None,
    events_out: str | None = None,
    input_path: str | None = None,
    ignore_lock: bool = False,
) -> None:
    """Step 20's primary paid classify run. Never touches canonical_dataset.csv -- this is a
    comparison signal only (AGENTS.md's "human curation is never bypassed" rule). Requires
    `--estimated-usd` (copied from `llm-classify project-cost`'s printed projection for this exact
    scope size) AND `--confirm` -- classify never invents its own cost figure, and never spends
    without an explicit, informed confirmation regardless of how small the projection is.
    `concurrency` (default 20) runs that many API calls in parallel -- see
    `llm_classify/runner.py::classify_records`'s docstring for why this is safe.

    `mode` (default "primary", the 3-way positive/negative/undeterminable prompt) can also be set
    to "forced_choice" (2-way, no undeterminable option) to run the FULL sample under that variant
    instead -- distinct from `run-fallback --mode forced_guess`, which only re-asks the primary
    run's already-undetermined subset. Writing to the same `llm_classification_events.csv` under a
    different `mode` tag never overwrites or removes the other mode's rows -- both stay inspectable
    side by side, keyed by (record_id, mode).

    `scope="all"` and `scope="all_plus_candidates"` both exclude every record_id already in
    `second_curator_sample.csv` (Step 20's original 1,000-record evaluation sample, since resolved
    via Cross Curate Resolve) -- those already have a real, human-final decision under the refined
    Step 20c criteria, so re-classifying them would spend money without adding new signal.
    `all_plus_candidates` additionally includes `label_confidence == "heuristic_candidate"` rows --
    the EPMC "clear negative" background sample (Step 14's `clear_negative_sampler_strong` plus
    Step 19d's `clear_negative_sampler_strong_filtered_v2`, fetched via a query designed to exclude
    AI/ML terms, never individually human-reviewed record-by-record). Provenance for every
    classified record (which pool it came from) is never lost -- it's always re-derivable by
    joining `record_id` back to `canonical_dataset.csv`'s `label_confidence`/`sources` columns, so
    no new column is needed on the classification event log itself."""
    started_at = time.monotonic()
    if scope == "sample":
        sample_path = cfg.path("second_curator_sample")
        if not sample_path.exists():
            raise ValueError(f"{sample_path} does not exist -- run `llm-classify sample` first.")
        sample = pd.read_csv(sample_path, dtype=str)
        input_paths = [sample_path]
    elif scope in ("all", "all_plus_candidates"):
        canonical_path = cfg.path("canonical_dataset")
        dataset = pd.read_csv(canonical_path, dtype=str)
        held_out_path = cfg.path("second_curator_sample")
        held_out_ids: set = set()
        input_paths = [canonical_path]
        if held_out_path.exists():
            held_out_ids = set(pd.read_csv(held_out_path, dtype=str)["record_id"])
            input_paths.append(held_out_path)
        sample = select_full_population(
            dataset, include_candidates=(scope == "all_plus_candidates"), held_out_ids=held_out_ids
        )
    elif scope == "bulk_pool_excluding_curated":
        # Step 23a: full EPMC AI/ML landscape triage. Deliberately reuses classify_records/
        # build_prompt/_stream_classify_events_to_disk completely unchanged below -- the ONLY
        # things that differ from the trusted-pool scopes above are the input population and the
        # output file, so this run is provably the exact same mechanism, not a reimplementation.
        bulk_path = cfg.sampling_path("bulk_candidates")
        canonical_path = cfg.path("canonical_dataset")
        bulk_df = pd.read_csv(bulk_path, dtype=str)
        existing_ids = _load_existing_ids(canonical_path)
        # Real, confirmed bug fixed here (2026-08-27, caught by the pre-500-test smoke check):
        # dtype=str + pd.read_csv leaves a missing pmcid/pmid/doi as float NaN, not None -- and
        # NaN is truthy in Python, so record_id_from_ids' `if value:` check treated a missing
        # pmcid as the literal string "nan" for every row lacking one, collapsing 157,412
        # genuinely distinct records onto a single colliding record_id. `.where(pd.notna(...),
        # None)` converts NaN to real None first so the priority fallback (pmcid -> doi -> pmid)
        # actually engages instead of getting short-circuited by a truthy NaN.
        id_cols = bulk_df[["pmcid", "pmid", "doi"]].where(pd.notna(bulk_df[["pmcid", "pmid", "doi"]]), None)
        record_ids = id_cols.apply(
            lambda row: record_id_from_ids(row["pmcid"], row["pmid"], row["doi"]), axis=1
        )
        sample = select_bulk_pool_excluding_curated(bulk_df, existing_ids, record_ids)
        sample = sample[sample["record_id"].notna()]
        input_paths = [bulk_path, canonical_path]
    elif scope == "staged_file":
        # Step 24 phase 2: the incremental loop. Population comes from an explicit --input rather
        # than from configs, because each batch is a different file (one per fetched window), and
        # the filtering that the other scopes do at this point has already happened upstream in
        # build_incoming_documents.py. Everything downstream -- prompt, parser, blinding boundary,
        # budget gate, resumability -- is identical to the other scopes.
        if input_path is None:
            raise ValueError("--scope staged_file requires --input <csv> (the output of "
                             "moros_pipeline/scripts/build_incoming_documents.py)")
        staged_path = Path(input_path)
        if not staged_path.exists():
            raise ValueError(f"{staged_path} does not exist")
        staged_df = pd.read_csv(staged_path, dtype=str)
        sample = select_staged_file(staged_df)
        n_dropped = len(staged_df) - len(sample)
        if n_dropped:
            print(f"llm-classify.classify: {n_dropped:,} staged record(s) have no abstract and "
                  f"are not classifiable from title alone -- left in {staged_path.name}, not "
                  f"classified.")
        input_paths = [staged_path]
    else:
        raise ValueError(
            f"Unknown scope {scope!r} -- expected 'sample', 'all', 'all_plus_candidates', "
            "'bulk_pool_excluding_curated', or 'staged_file'."
        )

    criteria_path, criteria_text, criteria_hash = _criteria_text_and_hash()
    if scope == "staged_file" and not events_out:
        # Each staged batch gets its own log beside its input, so batches are never conflated and
        # a re-run resumes the right one. --events-out still overrides.
        events_out = str(Path(input_path).with_name(Path(input_path).stem + "_classification_events.csv"))
    events_path = (
        cfg.path("landscape_classification_events") if scope == "bulk_pool_excluding_curated"
        else cfg.path("llm_classification_events")
    )
    # `events_out` overrides BOTH the read and the write of the event log. Resumability keys on
    # (tier, mode, prompt_version, criteria_sha256), so pointing at a fresh file means nothing is
    # seen as already-done and the exact same records get classified again -- the whole point of an
    # A/B re-validation run (e.g. proving a concurrency change yields the same answers). Nothing
    # about the prompt, criteria, or parsing differs.
    if events_out:
        events_path = resolve_path(events_out)
    existing_events = (
        pd.read_csv(events_path, dtype=str) if events_path.exists()
        else pd.DataFrame(columns=llm_runner.EVENT_COLUMNS)
    )

    if limit is not None:
        # Safety net for a real, explicit "N-record test first" run (e.g. Step 23a's 500-record
        # check before the full ~800k-record pool) -- same done-aware limiting pattern as
        # llm-classify enrich's --limit. Already-classified records stay in `sample` (so cost/
        # progress reporting isn't confused), only the NOT-yet-done remainder is capped, so a
        # second, larger --limit run continues from where the first stopped rather than re-testing
        # the same handful of records.
        done_ids = llm_runner._already_classified_ids(existing_events, tier, mode, criteria_hash)
        done_mask = sample["record_id"].isin(done_ids)
        sample = pd.concat([sample[done_mask], sample[~done_mask].head(limit)], ignore_index=True)

    if estimated_usd is None:
        raise ValueError(
            "Pass --estimated-usd, taken from `llm-classify project-cost`'s printed "
            "projected_usd_for_target_n for this exact number of records -- classify never "
            "invents its own cost figure."
        )
    budget_cfg = _deepseek_budget_cfg(cfg)
    spend_log_path = resolve_path(budget_cfg["spend_log"])
    llm_budget.check_cap(spend_log_path, budget_cfg["total_cap_usd"], estimated_usd, confirmed)
    if not confirmed:
        raise ValueError("Pass --confirm to actually spend real money on this classify run.")

    batch_id = f"classify_{tier}_{scope}_{mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    client = _deepseek_client(concurrency)
    n_written = 0
    try:
        # Same one-writer-per-event-log guard the enrich path uses -- see _events_file_lock. This
        # path will run unattended against a Europe PMC batch every two months, so it needs it at least as
        # much: two classify processes on one log pay twice for the same records.
        with _events_file_lock(events_path, ignore_lock=ignore_lock):
            n_written = _stream_classify_events_to_disk(
                events_path,
                llm_runner.classify_records(
                    sample, tier, client, criteria_text, criteria_hash, batch_id, mode=mode,
                    existing_events=existing_events, max_workers=concurrency,
                ),
            )
    except Exception:
        # Every record up to the crash is already safely on disk (streamed one row at a time,
        # above) -- this is best-effort bookkeeping for the interrupted attempt itself, not the
        # safety net. Re-running the exact same command resumes automatically.
        llm_budget.log_spend(
            spend_log_path, "llm-classify.classify", tier, mode, n_written,
            estimated_usd=estimated_usd, actual_usd=None,
            confirmed_by=cfg.pipeline["curation"]["default_curator"],
        )
        print(
            f"llm-classify.classify: CRASHED after {n_written} record(s) successfully classified "
            f"and saved to {events_path}. Nothing beyond the one in-flight call was lost -- "
            "re-run the exact same command to resume; already-classified records are skipped "
            "automatically, not re-paid for."
        )
        raise
    finally:
        client.close()

    llm_budget.log_spend(
        spend_log_path, "llm-classify.classify", tier, mode, n_written,
        estimated_usd=estimated_usd, actual_usd=None,
        confirmed_by=cfg.pipeline["curation"]["default_curator"],
    )

    finish_step(
        "llm-classify.classify",
        inputs=input_paths + [criteria_path],
        outputs=[events_path],
        params={"tier": tier, "scope": scope, "mode": mode, "estimated_usd": estimated_usd},
        notes=f"{n_written} newly classified (of {len(sample)} in scope; rest already classified "
        "this tier/criteria this run and were skipped -- resumable)",
        started_at=started_at,
    )


def step_llm_classify_calibrate_fallback(
    cfg: PipelineConfig, mode: str, tier: str, n: int = 8, confirmed: bool = False
) -> None:
    """Step 20's undetermined-subset fallback calibration (`mode` in "forced_guess"/"rag") -- a
    RAG-enabled call may cost meaningfully more than a closed-book one (extra retrieved content in
    context), so this is calibrated separately from the primary run rather than assumed
    proportional."""
    started_at = time.monotonic()
    criteria_path, criteria_text, criteria_hash = _criteria_text_and_hash()

    sample_path = cfg.path("second_curator_sample")
    if not sample_path.exists():
        raise ValueError(f"{sample_path} does not exist -- run `llm-classify sample` and `llm-classify classify` first.")
    sample = pd.read_csv(sample_path, dtype=str)

    events_path = cfg.path("llm_classification_events")
    primary_events = (
        pd.read_csv(events_path, dtype=str) if events_path.exists()
        else pd.DataFrame(columns=llm_runner.EVENT_COLUMNS)
    )
    undetermined = llm_runner.select_undetermined_subset(sample, primary_events, tier, criteria_hash)
    if undetermined.empty:
        print("llm-classify.calibrate-fallback: no undetermined records for this tier yet -- run "
              "`llm-classify classify` first (or there may genuinely be none).")
        return

    budget_cfg = _deepseek_budget_cfg(cfg)
    spend_log_path = resolve_path(budget_cfg["spend_log"])
    llm_budget.check_cap(spend_log_path, budget_cfg["total_cap_usd"], 0.20, confirmed)

    batch_id = f"calibrate_fallback_{mode}_{tier}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    enable_search = mode == "rag"
    client = _deepseek_client()
    try:
        result = llm_cost.run_calibration_batch(
            undetermined, tier, client, criteria_text, batch_id, n=n, mode=mode, enable_search=enable_search
        )
    finally:
        client.close()

    output_path = cfg.path("second_curator_calibration_log")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not output_path.exists()
    result.to_csv(output_path, mode="a", header=header_needed, index=False)

    print(f"llm-classify.calibrate-fallback: {len(result)} real calls (mode={mode}, tier={tier}) "
          f"over the {len(undetermined)}-record undetermined subset. Check your real DeepSeek "
          f"dashboard balance deduction, then run `llm-classify project-cost --mode {mode}`.")

    llm_budget.log_spend(
        spend_log_path, "llm-classify.calibrate-fallback", tier, mode, len(result),
        estimated_usd=0.20, actual_usd=None, confirmed_by=cfg.pipeline["curation"]["default_curator"],
    )

    finish_step(
        "llm-classify.calibrate-fallback",
        inputs=[sample_path, events_path, criteria_path],
        outputs=[output_path],
        params={"mode": mode, "tier": tier, "n": n},
        notes=f"{len(result)} real calibration calls over {len(undetermined)} undetermined records",
        started_at=started_at,
    )


def step_llm_classify_run_fallback(
    cfg: PipelineConfig,
    mode: str,
    tier: str,
    estimated_usd: float | None = None,
    confirmed: bool = False,
    concurrency: int = 20,
) -> None:
    """Step 20's undetermined-subset fallback re-ask (`mode` in "forced_guess"/"rag") -- only over
    records the PRIMARY run answered undeterminable for this tier. Same `--estimated-usd` +
    `--confirm` requirement as the primary `classify` command, for the same reason."""
    started_at = time.monotonic()
    if estimated_usd is None:
        raise ValueError(f"Pass --estimated-usd from `llm-classify project-cost --mode {mode}`'s printed projection.")

    criteria_path, criteria_text, criteria_hash = _criteria_text_and_hash()
    sample_path = cfg.path("second_curator_sample")
    if not sample_path.exists():
        raise ValueError(f"{sample_path} does not exist -- run `llm-classify sample` and `llm-classify classify` first.")
    sample = pd.read_csv(sample_path, dtype=str)

    events_path = cfg.path("llm_classification_events")
    existing_events = (
        pd.read_csv(events_path, dtype=str) if events_path.exists()
        else pd.DataFrame(columns=llm_runner.EVENT_COLUMNS)
    )
    undetermined = llm_runner.select_undetermined_subset(sample, existing_events, tier, criteria_hash)
    if undetermined.empty:
        print("llm-classify.run-fallback: no undetermined records for this tier -- nothing to do.")
        return

    budget_cfg = _deepseek_budget_cfg(cfg)
    spend_log_path = resolve_path(budget_cfg["spend_log"])
    llm_budget.check_cap(spend_log_path, budget_cfg["total_cap_usd"], estimated_usd, confirmed)
    if not confirmed:
        raise ValueError("Pass --confirm to actually spend real money on this fallback run.")

    enable_search = mode == "rag"
    batch_id = f"fallback_{mode}_{tier}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    client = _deepseek_client()
    n_written = 0
    try:
        try:
            n_written = _stream_classify_events_to_disk(
                events_path,
                llm_runner.classify_records(
                    undetermined, tier, client, criteria_text, criteria_hash, batch_id, mode=mode,
                    existing_events=existing_events, enable_search=enable_search, max_workers=concurrency,
                ),
            )
        finally:
            client.close()
    except Exception:
        llm_budget.log_spend(
            spend_log_path, "llm-classify.run-fallback", tier, mode, n_written,
            estimated_usd=estimated_usd, actual_usd=None,
            confirmed_by=cfg.pipeline["curation"]["default_curator"],
        )
        print(
            f"llm-classify.run-fallback: CRASHED after {n_written} record(s) successfully "
            f"classified and saved to {events_path}. Re-run the exact same command to resume."
        )
        raise

    llm_budget.log_spend(
        spend_log_path, "llm-classify.run-fallback", tier, mode, n_written,
        estimated_usd=estimated_usd, actual_usd=None,
        confirmed_by=cfg.pipeline["curation"]["default_curator"],
    )

    finish_step(
        "llm-classify.run-fallback",
        inputs=[sample_path, events_path, criteria_path],
        outputs=[events_path],
        params={"mode": mode, "tier": tier, "estimated_usd": estimated_usd},
        notes=f"{n_written} newly classified over the {len(undetermined)}-record undetermined subset",
        started_at=started_at,
    )


def step_reporting_agreement(cfg: PipelineConfig) -> None:
    """Step 20: Cohen's kappa + accuracy for every tier that has a PRIMARY-mode classification
    against the shared blind sample, written to `human_llm_agreement_report.csv`."""
    started_at = time.monotonic()
    sample_path = cfg.path("second_curator_sample")
    events_path = cfg.path("llm_classification_events")
    if not sample_path.exists() or not events_path.exists():
        print("reporting.agreement: second_curator_sample.csv or llm_classification_events.csv "
              "missing -- run `llm-classify sample` and `llm-classify classify` first.")
        return

    sample = pd.read_csv(sample_path, dtype=str)[["record_id", "label"]]
    events = pd.read_csv(events_path, dtype=str)
    primary = events[events["mode"] == "primary"]
    if primary.empty:
        print("reporting.agreement: no primary-mode classification events yet.")
        return

    rows = []
    for tier in sorted(primary["model_tier"].unique()):
        tier_events = primary[primary["model_tier"] == tier].sort_values("timestamp").groupby("record_id").last()
        joined = sample.merge(tier_events[["classification"]], left_on="record_id", right_index=True, how="inner")
        agreement = agreement_reporting.compute_agreement(joined, tier)
        rows.append(agreement)
        print(f"reporting.agreement: tier={tier} kappa={agreement['kappa']:.3f} "
              f"accuracy_among_decided={agreement['accuracy_among_decided'] * 100:.1f}% "
              f"undetermined_rate={agreement['undetermined_rate'] * 100:.1f}%")

    report = pd.DataFrame(rows)
    output_path = cfg.path("human_llm_agreement_report")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    finish_step(
        "reporting.agreement",
        inputs=[sample_path, events_path],
        outputs=[output_path],
        notes=f"{len(rows)} tier(s) scored -- see printed kappa/accuracy/undetermined-rate above",
        started_at=started_at,
    )


_SECOND_CURATOR_PROFILE_README = """# Second Curator (DeepSeek) Profile

Step 20: human-vs-DeepSeek agreement metrics for the blind, independent second-curator
classification pass over the shared 500+500 (or custom-sized) sample. DeepSeek was never shown
the human's existing label, notes, curation_tag, or MeSH headings -- only title/abstract/journal/
year -- and its primary pass had no web/search access (see STEPS_Progress.md Step 20).

See `profile_metadata.json` for exact per-tier kappa / accuracy-among-decided / undetermined-rate
figures, and the `undeterminable_fallback_comparison_*.png` charts (if present) for whether a
forced-guess or RAG-assisted re-ask recovers a usable answer on the records DeepSeek's primary pass
couldn't decide.

**Caveat**: DeepSeek may have latent pretraining knowledge of some of these public Europe PMC
papers beyond what pure closed-book title/abstract reasoning would give -- a very high kappa should
not be over-read as evidence of pure abstract-reasoning skill alone.
"""


def step_reporting_profile_second_curator(cfg: PipelineConfig) -> None:
    """Step 20's dedicated visualization folder -- confusion matrix + disagreement breakdown per
    tier, a cross-tier kappa/accuracy summary, a flash-vs-pro comparison (if both tiers were run),
    and the undeterminable-subset fallback comparison (if any fallback batch was run)."""
    started_at = time.monotonic()
    sample_path = cfg.path("second_curator_sample")
    events_path = cfg.path("llm_classification_events")
    if not sample_path.exists() or not events_path.exists():
        print("reporting.profile-second-curator: missing sample/events -- run `llm-classify "
              "sample` and `llm-classify classify` first.")
        return

    sample = pd.read_csv(sample_path, dtype=str)
    events = pd.read_csv(events_path, dtype=str)
    primary = events[events["mode"] == "primary"]
    if primary.empty:
        print("reporting.profile-second-curator: no primary-mode classification events yet.")
        return

    output_dir = cfg.path("second_curator_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = sorted(primary["model_tier"].unique())
    joined_by_tier: dict = {}
    agreements: dict = {}
    output_paths = []
    for tier in tiers:
        tier_events = primary[primary["model_tier"] == tier].sort_values("timestamp").groupby("record_id").last()
        joined = sample[["record_id", "label"]].merge(
            tier_events[["classification"]], left_on="record_id", right_index=True, how="inner"
        )
        joined_by_tier[tier] = joined
        agreements[tier] = agreement_reporting.compute_agreement(joined, tier)
        cm_path = output_dir / f"confusion_matrix_{tier}.png"
        agreement_reporting.plot_confusion_matrix_per_tier(joined, tier, cm_path)
        db_path = output_dir / f"disagreement_breakdown_{tier}.png"
        agreement_reporting.plot_disagreement_breakdown(joined, tier, db_path)
        output_paths += [cm_path, db_path]

    summary_path = output_dir / "kappa_accuracy_summary.png"
    agreement_reporting.plot_kappa_accuracy_summary(agreements, summary_path)
    output_paths.append(summary_path)

    if "flash" in joined_by_tier and "pro" in joined_by_tier:
        comparison_path = output_dir / "flash_vs_pro_agreement_comparison.png"
        agreement_reporting.plot_flash_vs_pro_comparison(
            joined_by_tier["flash"], joined_by_tier["pro"], comparison_path
        )
        output_paths.append(comparison_path)

    for tier in tiers:
        _, _, criteria_hash = _criteria_text_and_hash()
        forced_guess_events = events[(events["mode"] == "forced_guess") & (events["model_tier"] == tier)]
        rag_events = events[(events["mode"] == "rag") & (events["model_tier"] == tier)]
        forced_guess = (
            agreement_reporting.compute_fallback_accuracy(forced_guess_events, sample, "forced_guess")
            if not forced_guess_events.empty else None
        )
        rag = (
            agreement_reporting.compute_fallback_accuracy(rag_events, sample, "rag")
            if not rag_events.empty else None
        )
        if forced_guess is not None or rag is not None:
            fallback_path = output_dir / f"undeterminable_fallback_comparison_{tier}.png"
            agreement_reporting.plot_undeterminable_fallback_comparison(
                agreements[tier], forced_guess, rag, fallback_path
            )
            output_paths.append(fallback_path)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "tiers": tiers,
        "agreements": agreements,
    }
    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_SECOND_CURATOR_PROFILE_README)
    output_paths += [metadata_path, readme_path]

    print(f"reporting.profile-second-curator: {len(tiers)} tier(s) profiled ({tiers}) -- "
          f"charts + profile_metadata.json + README.md written to {output_dir}")

    finish_step(
        "reporting.profile-second-curator",
        inputs=[sample_path, events_path],
        outputs=output_paths,
        notes=f"{len(tiers)} tier(s) profiled: {tiers}",
        started_at=started_at,
    )


_POST_REVIEW_CONSENSUS_PROFILE_README = """# Post-Review Consensus Profile (Step 20a)

Charts here answer one question: after Gavin's Cross Curate Resolve human review of every
Step 20 DeepSeek/human disagreement, did the human's FINAL decision end up agreeing with DeepSeek
more often than the ORIGINAL label did?

This is a SEPARATE folder from `second_curator_profile/`, which stays as the honest PRE-review
snapshot -- neither folder is overwritten by the other.

## Charts

- `post_review_reversal_breakdown_<tier>.png` -- for each tier's original disagreements, how many
  were "Reversed to DeepSeek" (the final decision matches what that tier said) vs. "Upheld
  Original" (the final decision confirms the original label) vs. "Neither".
- `original_vs_post_review_kappa.png` -- Cohen's kappa scored against the original label vs. the
  post-review final label, per tier, over the same full sample population.
"""


def step_reporting_profile_post_review_consensus(cfg: PipelineConfig) -> None:
    """Step 20a: visualizes what the Cross Curate Resolve review actually changed. Built directly
    off `cross_curate_resolution_events.csv` -- deliberately no dependency on Step 20b's
    `materialize_cross_curate_resolutions()` having run first, same as `build_disagreement_queue`
    already reads the raw event log rather than requiring a prior materialize step."""
    started_at = time.monotonic()
    sample_path = cfg.path("second_curator_sample")
    resolution_events_path = resolve_path(cfg.pipeline["curation"]["cross_curate_resolution_events"])
    llm_events_path = cfg.path("llm_classification_events")
    if not sample_path.exists() or not resolution_events_path.exists() or not llm_events_path.exists():
        print("reporting.profile-post-review-consensus: missing sample/resolution-events/"
              "llm-events -- run `llm-classify sample`, `llm-classify classify`, and complete "
              "Cross Curate Resolve first.")
        return

    sample = pd.read_csv(sample_path, dtype=str)
    resolution_events = pd.read_csv(resolution_events_path, dtype=str)
    llm_events = pd.read_csv(llm_events_path, dtype=str)
    primary = llm_events[llm_events["mode"] == "primary"]
    if primary.empty:
        print("reporting.profile-post-review-consensus: no primary-mode classification events yet.")
        return

    output_dir = cfg.path("post_review_consensus_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    tiers = sorted(primary["model_tier"].unique())
    output_paths = []
    reversal_breakdowns = {}
    for tier in tiers:
        path = output_dir / f"post_review_reversal_breakdown_{tier}.png"
        data = agreement_reporting.plot_post_review_reversal_breakdown(sample, resolution_events, llm_events, tier, path)
        reversal_breakdowns[tier] = data["count"].to_dict()
        output_paths.append(path)

    kappa_path = output_dir / "original_vs_post_review_kappa.png"
    kappa_data = agreement_reporting.plot_original_vs_post_review_kappa(sample, resolution_events, llm_events, kappa_path)
    output_paths.append(kappa_path)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "tiers": tiers,
        "reversal_breakdowns": reversal_breakdowns,
        "kappa": kappa_data.to_dict(orient="records"),
    }
    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_POST_REVIEW_CONSENSUS_PROFILE_README)
    output_paths += [metadata_path, readme_path]

    print(f"reporting.profile-post-review-consensus: {len(tiers)} tier(s) profiled ({tiers}) -- "
          f"charts + profile_metadata.json + README.md written to {output_dir}")

    finish_step(
        "reporting.profile-post-review-consensus",
        inputs=[sample_path, resolution_events_path, llm_events_path],
        outputs=output_paths,
        notes=f"{len(tiers)} tier(s) profiled: {tiers}",
        started_at=started_at,
    )


_FULL_POPULATION_PROFILE_README = """# Full Population Profile (Step 20e / Step 20f)

Three separate, closely-related answers about Step 20e's full re-run (7,600 records: the trusted
pool minus the held-out 1,000, plus the EPMC "clear negative" candidate pool), never blended
together since each is a genuinely different population/question:

## Charts

- `trusted_pool_confusion_matrix_<tier>.png`, `trusted_pool_disagreement_breakdown_<tier>.png`,
  `trusted_pool_kappa_accuracy_<tier>.png` -- DeepSeek vs. the human's current label, over the
  5,825-record trusted pool (excluding the held-out 1,000), under the REFINED (current)
  `CRITERIA.md`. Same chart conventions as `second_curator_profile/`, just at full-population scale
  instead of a 1,000-record sample.
- `candidate_pool_confirmation_<tier>.png` -- the EPMC `heuristic_candidate` "clear negative" pool
  (1,775 records, NEVER individually human-reviewed -- `label='negative'` is assumed by the
  fetch-query design, not verified per record). This is NOT a kappa chart -- there's no real human
  ground truth to score against here. It's a confirmation-RATE chart: how much of this
  never-reviewed pool does DeepSeek independently agree is negative, split by the two fetch
  batches (Step 14's original vs. Step 19d's `filtered_v2`)? A high confirmation rate says the
  fetch-query design is clean; a meaningful positive/undeterminable rate flags records worth a
  manual look before trusting a wider EPMC search built the same way.
- `whole_evaluated_population_comparison_<tier>.png` -- kappa/accuracy, side by side, for the two
  distinct populations DeepSeek has ever been evaluated against: the original 1,000-sample (scored
  against its TRUEST label -- the post-Cross-Curate-Resolve final decision -- under the criteria it
  was actually classified under) and the new 5,825 (scored against the current label, under the
  refined criteria). Deliberately kept as two separate bars, never pooled into one blended number,
  since the two groups were classified under different `CRITERIA.md` versions.

## Not included here

The candidate pool is diagnostic only -- it never enters Cross Curate Resolve or gets materialized
into `canonical_dataset.csv`, since it was never individually human-reviewed to begin with.
"""


def step_reporting_profile_full_population(cfg: PipelineConfig, tier: str = "flash") -> None:
    """Step 20f: profiles Step 20e's full re-run -- trusted-pool agreement at full scale, the EPMC
    candidate-pool confirmation rate, and whole-evaluated-population alignment. See
    `_FULL_POPULATION_PROFILE_README` above for what each chart answers and why they're kept
    separate."""
    started_at = time.monotonic()
    dataset_path = cfg.path("canonical_dataset")
    llm_events_path = cfg.path("llm_classification_events")
    sample_path = cfg.path("second_curator_sample")
    resolution_events_path = resolve_path(cfg.pipeline["curation"]["cross_curate_resolution_events"])
    if not dataset_path.exists() or not llm_events_path.exists():
        print("reporting.profile-full-population: missing canonical_dataset.csv/llm_classification_events.csv"
              " -- run Step 20e's `llm-classify classify --scope all_plus_candidates` first.")
        return

    dataset = pd.read_csv(dataset_path, dtype=str)
    llm_events = pd.read_csv(llm_events_path, dtype=str)
    _, _, current_hash = _criteria_text_and_hash()

    output_dir = cfg.path("full_population_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    # Batch A: trusted-pool agreement at full scale, under the current (refined) criteria.
    trusted_joined = agreement_reporting.build_population_run_joined(dataset, llm_events, tier, current_hash)
    if trusted_joined.empty:
        print(f"reporting.profile-full-population: no trusted-pool classification events for tier={tier!r} "
              f"under the current criteria ({current_hash[:12]}...) -- nothing to profile yet.")
        return

    cm_path = output_dir / f"trusted_pool_confusion_matrix_{tier}.png"
    agreement_reporting.plot_confusion_matrix_per_tier(trusted_joined, tier, cm_path)
    output_paths.append(cm_path)

    db_path = output_dir / f"trusted_pool_disagreement_breakdown_{tier}.png"
    agreement_reporting.plot_disagreement_breakdown(trusted_joined, tier, db_path)
    output_paths.append(db_path)

    trusted_agreement = agreement_reporting.compute_agreement(trusted_joined, tier)
    kappa_path = output_dir / f"trusted_pool_kappa_accuracy_{tier}.png"
    agreement_reporting.plot_kappa_accuracy_summary({tier: trusted_agreement}, kappa_path)
    output_paths.append(kappa_path)

    # Batch B: EPMC candidate-pool confirmation rate (diagnostic only, not kappa-scored).
    candidate_confirmation_dict = None
    candidate_joined = agreement_reporting.build_candidate_pool_joined(dataset, llm_events, tier, current_hash)
    if not candidate_joined.empty:
        candidate_path = output_dir / f"candidate_pool_confirmation_{tier}.png"
        candidate_data = agreement_reporting.plot_candidate_pool_confirmation(candidate_joined, tier, candidate_path)
        candidate_confirmation_dict = candidate_data.to_dict(orient="index")
        output_paths.append(candidate_path)

    # Batch C: whole-evaluated-population comparison (original 1,000 vs. new 5,825).
    whole_set_dict = None
    if sample_path.exists() and resolution_events_path.exists():
        sample = pd.read_csv(sample_path, dtype=str)
        resolution_events = pd.read_csv(resolution_events_path, dtype=str)
        try:
            old_hash = agreement_reporting.resolve_prior_criteria_hash(
                llm_events, set(sample["record_id"]), tier, current_hash
            )
            original_joined = agreement_reporting.build_original_sample_vs_final_joined(
                sample, resolution_events, llm_events, tier, old_hash
            )
            whole_set_path = output_dir / f"whole_evaluated_population_comparison_{tier}.png"
            whole_set_data = agreement_reporting.plot_whole_evaluated_population_comparison(
                original_joined, trusted_joined, tier, whole_set_path
            )
            whole_set_dict = whole_set_data.to_dict(orient="records")
            output_paths.append(whole_set_path)
        except ValueError as exc:
            print(f"reporting.profile-full-population: skipped whole-population comparison -- {exc}")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "tier": tier,
        "current_criteria_sha256": current_hash,
        "trusted_pool_agreement": trusted_agreement,
        "candidate_pool_confirmation": candidate_confirmation_dict,
        "whole_evaluated_population_comparison": whole_set_dict,
    }
    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_FULL_POPULATION_PROFILE_README)
    output_paths += [metadata_path, readme_path]

    print(f"reporting.profile-full-population: tier={tier} profiled -- trusted pool kappa="
          f"{trusted_agreement['kappa']:.3f}, charts + profile_metadata.json + README.md written to {output_dir}")

    finish_step(
        "reporting.profile-full-population",
        inputs=[dataset_path, llm_events_path] + ([sample_path, resolution_events_path] if sample_path.exists() else []),
        outputs=output_paths,
        params={"tier": tier},
        notes=f"trusted_pool n={len(trusted_joined)}, candidate_pool present={candidate_confirmation_dict is not None}",
        started_at=started_at,
    )


_FULL_POPULATION_CONSENSUS_PROFILE_README = """# Full Population Consensus Profile (Step 20f)

The full-population analog of `post_review_consensus_profile/` (Step 20a) -- after Gavin's Cross
Curate Resolve review of Step 20e's NEW disagreements (the trusted-pool subset of the 7,600-record
full re-run, not the original 1,000), did the human's FINAL decision end up agreeing with DeepSeek
more often than the current label did?

## Charts

- `full_population_reversal_breakdown_<tier>.png` -- for each tier's Step 20e disagreements, how
  many were "Reversed to DeepSeek" vs. "Upheld Original" vs. "Neither".
- `full_population_original_vs_post_review_kappa.png` -- Cohen's kappa scored against the current
  label vs. the post-review final label, per tier, over the full trusted-pool population.
"""


def step_reporting_profile_full_population_consensus(cfg: PipelineConfig, tier: str = "flash") -> None:
    """Step 20f: the full-population analog of `step_reporting_profile_post_review_consensus` --
    run this AFTER Gavin has manually resolved Step 20e's new trusted-pool disagreements via Cross
    Curate Resolve. Reuses `plot_post_review_reversal_breakdown` unchanged (already generic over
    any population DataFrame) and the new `plot_population_original_vs_post_review_kappa` (a
    population-size-aware variant of Step 20a's chart, since that one's caption text is specific to
    the 1,000-sample)."""
    started_at = time.monotonic()
    dataset_path = cfg.path("canonical_dataset")
    llm_events_path = cfg.path("llm_classification_events")
    resolution_events_path = resolve_path(cfg.pipeline["curation"]["cross_curate_resolution_events"])
    if not dataset_path.exists() or not llm_events_path.exists() or not resolution_events_path.exists():
        print("reporting.profile-full-population-consensus: missing canonical_dataset.csv/"
              "llm_classification_events.csv/cross_curate_resolution_events.csv.")
        return

    dataset = pd.read_csv(dataset_path, dtype=str)
    llm_events = pd.read_csv(llm_events_path, dtype=str)
    resolution_events = pd.read_csv(resolution_events_path, dtype=str)
    _, _, current_hash = _criteria_text_and_hash()

    trusted = dataset[
        dataset["label_confidence"].isin(("human_curated", "registry_confirmed"))
        & dataset["label"].isin(["positive", "negative"])
    ][["record_id", "label"]]

    output_dir = cfg.path("full_population_consensus_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    reversal_path = output_dir / f"full_population_reversal_breakdown_{tier}.png"
    reversal_data = agreement_reporting.plot_post_review_reversal_breakdown(
        trusted, resolution_events, llm_events, tier, reversal_path
    )
    output_paths.append(reversal_path)

    kappa_path = output_dir / "full_population_original_vs_post_review_kappa.png"
    kappa_data = agreement_reporting.plot_population_original_vs_post_review_kappa(
        trusted, resolution_events, llm_events, kappa_path
    )
    output_paths.append(kappa_path)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "tier": tier,
        "reversal_breakdown": reversal_data["count"].to_dict(),
        "kappa": kappa_data.to_dict(orient="records"),
    }
    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_FULL_POPULATION_CONSENSUS_PROFILE_README)
    output_paths += [metadata_path, readme_path]

    print(f"reporting.profile-full-population-consensus: tier={tier} -- "
          f"charts + profile_metadata.json + README.md written to {output_dir}")

    finish_step(
        "reporting.profile-full-population-consensus",
        inputs=[dataset_path, llm_events_path, resolution_events_path],
        outputs=output_paths,
        params={"tier": tier},
        notes=f"reversal_breakdown={reversal_data['count'].to_dict()}",
        started_at=started_at,
    )


def step_llm_classify_enrich(
    cfg: PipelineConfig,
    tier: str = "flash",
    concurrency: int = 100,
    limit: int | None = None,
    input_path: Path | None = None,
    events_out: str | None = None,
    reasoning_effort: str | None = None,
    domain_rendering: str = "flat",
    retry_truncated: bool = False,
    ignore_lock: bool = False,
) -> None:
    """Step 20j: the positive-set enrichment run. Additive-only by construction (the prompt never
    asks the positive/negative question -- see `llm_classify/enrichment.py`), reading the isolated
    trial CSV, never `canonical_dataset.csv` directly. Streams every event to disk the moment it
    completes (a machine/network death loses at most the in-flight calls) and resumes for free:
    re-running the exact same command skips everything already enriched under the current
    vocab_sha256 and retries parse_errors.

    Deliberately NO --estimated-usd/--confirm cost gate (Gavin's explicit 2026-08-27 call for this
    step -- "for the cost estimate i dont care over this rigidity"): the run just goes, and real
    token totals (including the prefix-cache hit share, the actual efficiency mechanism at scale)
    are printed at the end and recorded per-event, so the real cost story is still fully
    reconstructable from data rather than gated up front. `concurrency` defaults to 100 -- five
    times the classify path's 20, still far under DeepSeek's documented 2500-concurrency flash
    ceiling, sized for the eventual 100k+-positive full-landscape pass (Step 23b) this trial is
    the dress rehearsal for. `limit` runs only the first N remaining records -- a cheap real-API
    smoke check before committing to the full set."""
    started_at = time.monotonic()
    source_path = Path(input_path) if input_path is not None else cfg.path("enrichment_positive_set")
    if not source_path.exists():
        raise ValueError(
            f"{source_path} does not exist -- run enrichment_trial_dataset/build_positive_test_set.py "
            "on the host first (it is standalone, outside Docker)."
        )
    records = pd.read_csv(source_path, dtype=str)

    criteria_dir = resolve_path("curation_criteria")
    vocabs = llm_enrichment.load_vocabularies(criteria_dir)
    static_system_text = llm_enrichment.build_static_system_text(vocabs, domain_rendering)
    lookup = llm_enrichment._build_lookup(vocabs)
    vocab_hash = llm_enrichment.vocab_sha256(static_system_text)

    # `events_out` is used for BOTH the resume read and the write, exactly like the classify
    # path's own --events-out. It has to be both: resumability keys on record_id, and a
    # journal-scoped run out of Mongo uses the document `_id` (a UUID5) as its record_id while the
    # Step 20j trial log uses the sha1 `record_id` from canonical_dataset.csv. Reading one file
    # while writing another would conflate two id spaces in one log.
    events_path = Path(events_out) if events_out else cfg.path("enrichment_classification_events")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    existing_events = (
        pd.read_csv(events_path, dtype=str) if events_path.exists()
        else pd.DataFrame(columns=llm_enrichment.EVENT_COLUMNS)
    )
    if not existing_events.empty:
        existing_events["parse_status"] = existing_events["parse_status"].fillna("")

    if limit is not None:
        done = llm_enrichment._already_enriched_ids(existing_events, tier, vocab_hash)
        remaining = records[~records["record_id"].isin(done)]
        records = pd.concat([records[records["record_id"].isin(done)], remaining.head(limit)])

    # Pre-flight, before a single call is paid for: how much is actually left after resumability,
    # what it will cost, and how long it will take at this concurrency. The ETA is the number that
    # matters -- throughput is concurrency / ~45s per call, and a too-low setting is invisible
    # until hours have passed (a real 2026-09-03 run crawled at 53 rec/min on --concurrency 40
    # when 800 gives ~400/min).
    _n_todo = len(records) - len(llm_enrichment._already_enriched_ids(existing_events, tier, vocab_hash))
    _n_todo = max(_n_todo, 0)
    print(
        f"llm-classify.enrich: {_n_todo:,} record(s) to enrich at concurrency {concurrency} "
        f"-> ~{_n_todo / max(concurrency / 45.0, 1e-9) / 60:.1f} min, "
        f"~${_n_todo / 1000 * 10.0:.2f} at the billed ~$10/1k (tokens x list price undershoots the bill). "
        f"Events stream to disk per record; a re-run never re-pays for finished work."
    )

    batch_id = f"enrich_{tier}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    # `concurrency` MUST be passed: `_deepseek_client` sizes the HTTP connection pool from it, and
    # threads beyond `pool_maxsize` queue on a connection instead of running. This call omitted it
    # until 2026-09-03, pinning every enrichment run this project has ever done -- the Step 20j
    # trial included -- to the historical 50-connection default no matter what --concurrency said.
    # The classify path (see `_deepseek_client(concurrency)` above) always had it right.
    client = _deepseek_client(concurrency)
    n_written = 0
    try:
        # One writer per event log, enforced by the kernel -- see _events_file_lock.
        with _events_file_lock(events_path, ignore_lock=ignore_lock):
            n_written = _stream_classify_events_to_disk(
                events_path,
                llm_enrichment.enrich_records(
                    records, tier, client, static_system_text, lookup, batch_id,
                    existing_events=existing_events, max_workers=concurrency,
                    reasoning_effort=reasoning_effort,
                    retry_truncated=retry_truncated,
                ),
                columns=llm_enrichment.EVENT_COLUMNS,
            )
    except Exception:
        print(
            f"llm-classify.enrich: CRASHED after {n_written} record(s) safely written to "
            f"{events_path}. Nothing beyond the in-flight calls was lost -- re-run the exact same "
            "command to resume; already-enriched records are skipped automatically."
        )
        raise
    finally:
        client.close()

    this_batch = pd.read_csv(events_path, dtype=str)
    this_batch = this_batch[this_batch["batch_id"] == batch_id]
    tokens_in = pd.to_numeric(this_batch["input_tokens"], errors="coerce").sum()
    tokens_out = pd.to_numeric(this_batch["output_tokens"], errors="coerce").sum()
    tokens_cached = pd.to_numeric(this_batch["cache_hit_tokens"], errors="coerce").sum()
    tokens_reasoning = pd.to_numeric(
        this_batch.get("reasoning_tokens", pd.Series(dtype=str)), errors="coerce"
    ).sum()
    cache_pct = (tokens_cached / tokens_in * 100) if tokens_in else 0.0
    think_pct = (tokens_reasoning / tokens_out * 100) if tokens_out else 0.0
    n_parse_error = int((this_batch["parse_status"] == llm_enrichment.PARSE_ERROR).sum())
    n_truncated = int((this_batch.get("finish_reason", pd.Series(dtype=str)) == "length").sum())
    print(
        f"llm-classify.enrich: {n_written} record(s) enriched this run ({n_parse_error} parse_error"
        f" -- re-run the same command to retry those). Real tokens: {int(tokens_in):,} in / "
        f"{int(tokens_out):,} out, prefix-cache hit rate {cache_pct:.1f}% of input tokens "
        f"(vocab_sha256={vocab_hash[:12]}...). Check the real dashboard for billed cost."
    )
    print(
        f"llm-classify.enrich: reasoning was {int(tokens_reasoning):,} of {int(tokens_out):,} "
        f"output tokens ({think_pct:.1f}%) at effort "
        f"{reasoning_effort or 'provider default (high)'}; {n_truncated} response(s) hit the "
        f"{llm_enrichment.ENRICHMENT_MAX_TOKENS:,}-token cap and are skipped on re-run "
        f"(retrying a truncation at the same cap re-truncates -- raise "
        f"ENRICHMENT_MAX_TOKENS first, then pass --retry-truncated)."
    )

    finish_step(
        "llm-classify.enrich",
        inputs=[source_path],
        outputs=[events_path],
        params={"tier": tier, "concurrency": concurrency, "limit": limit,
                "vocab_sha256": vocab_hash, "reasoning_effort": reasoning_effort,
                "domain_rendering": domain_rendering},
        notes=f"{n_written} enriched (of {len(records)} in scope; rest already done -- resumable), "
        f"cache hit {cache_pct:.1f}%",
        started_at=started_at,
    )


def step_reporting_profile_enrichment(cfg: PipelineConfig, tier: str = "flash") -> None:
    """Step 20j's success/usage visualization: tag-count distributions per field, top model types,
    paradigm/family distributions, seed-normalization + violation rates. See
    `reporting/enrichment_profile.py` for what each chart answers."""
    started_at = time.monotonic()
    events_path = cfg.path("enrichment_classification_events")
    if not events_path.exists():
        print("reporting.profile-enrichment: no enrichment events yet -- run `llm-classify enrich` first.")
        return
    events = pd.read_csv(events_path, dtype=str)

    criteria_dir = resolve_path("curation_criteria")
    vocabs = llm_enrichment.load_vocabularies(criteria_dir)

    output_dir = cfg.path("enrichment_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, output_paths = enrichment_reporting.build_profile(events, vocabs, tier, output_dir)

    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(enrichment_reporting.PROFILE_README)
    output_paths += [metadata_path, readme_path]

    print(f"reporting.profile-enrichment: {metadata['n_events']} event(s) profiled -- charts + "
          f"profile_metadata.json + README.md written to {output_dir}")

    finish_step(
        "reporting.profile-enrichment",
        inputs=[events_path],
        outputs=output_paths,
        params={"tier": tier},
        notes=f"n_events={metadata['n_events']}, parse_error={metadata['n_parse_error']}, "
        f"violation_rate={metadata['violation_rate']:.3f}",
        started_at=started_at,
    )


_LANDSCAPE_CLASSIFICATION_PROFILE_README = """# AI/ML Landscape Classification Profile (Step 23a(b))

A real, reproducible version of the manual duplicate/parse-error audit already done by hand once
the full 827k-record landscape run landed, plus a rate comparison against the curated trusted set.

- `classification_breakdown.png` -- positive/negative/undeterminable/parse_error counts over the
  full landscape, one row per unique record_id (last event wins).
- `duplicate_id_audit.png` -- every repeated record_id in the raw event log, bucketed into: a
  parse_error successfully retried (expected, healthy), a genuine same-paper duplicate identity
  already present in the source pool agreeing with itself (benign), or a genuine duplicate that
  disagreed (real model nondeterminism, not reprocessing).
- `curated_vs_landscape_comparison.png` -- positive rate in the curated trusted set (post
  conflict-resolution, i.e. `label` reflects the FINAL human decision, not the pre-resolution one)
  vs. the DeepSeek-classified landscape. **These are disjoint populations by design** (the
  landscape run excluded every already-curated record before classifying) -- this is a
  rate/distribution comparison, NOT a per-record agreement or kappa score.

`profile_metadata.json` carries the exact real numbers behind every chart.
"""


def step_reporting_profile_ai_ml_landscape_classification(cfg: PipelineConfig) -> None:
    """Step 23a(b): profiles the full landscape classification run. Read-only, no spend -- reports
    on `landscape_classification_events.csv` as already produced. See
    `_LANDSCAPE_CLASSIFICATION_PROFILE_README` above for what each chart answers."""
    started_at = time.monotonic()
    events_path = cfg.path("landscape_classification_events")
    canonical_path = cfg.path("canonical_dataset")
    if not events_path.exists():
        print("reporting.profile-ai-ml-landscape-classification: no landscape_classification_events.csv "
              "yet -- run Step 23a's `llm-classify classify --scope bulk_pool_excluding_curated` first.")
        return

    events = pd.read_csv(events_path, dtype=str)
    resolved = landscape_reporting.resolve_latest_per_record(events)

    output_dir = cfg.path("landscape_classification_profile_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    breakdown_path = output_dir / "classification_breakdown.png"
    breakdown = landscape_reporting.plot_classification_breakdown(resolved, breakdown_path)
    output_paths.append(breakdown_path)

    dup_path = output_dir / "duplicate_id_audit.png"
    dup_audit = landscape_reporting.plot_duplicate_id_audit(events, dup_path)
    output_paths.append(dup_path)

    parse_error_stats = landscape_reporting.parse_error_resolution_data(events)

    comparison_dict = None
    if canonical_path.exists():
        canonical_df = pd.read_csv(canonical_path, dtype=str)
        comp_path = output_dir / "curated_vs_landscape_comparison.png"
        comparison = landscape_reporting.plot_curated_vs_landscape_comparison(
            canonical_df, resolved, comp_path
        )
        comparison_dict = comparison.to_dict(orient="records")
        output_paths.append(comp_path)
    else:
        print("reporting.profile-ai-ml-landscape-classification: canonical_dataset.csv missing -- "
              "skipping curated-vs-landscape comparison chart.")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_event_rows": len(events),
        "n_unique_records": int(events["record_id"].nunique()),
        "classification_breakdown": breakdown.to_dict(),
        "duplicate_id_audit": dup_audit,
        "parse_error_resolution": parse_error_stats,
        "curated_vs_landscape_comparison": comparison_dict,
    }
    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_LANDSCAPE_CLASSIFICATION_PROFILE_README)
    output_paths += [metadata_path, readme_path]

    print(
        f"reporting.profile-ai-ml-landscape-classification: {metadata['n_unique_records']:,} unique "
        f"records -- {parse_error_stats['n_still_unresolved']} still-unresolved parse_error(s) -- "
        f"charts + profile_metadata.json + README.md written to {output_dir}"
    )

    finish_step(
        "reporting.profile-ai-ml-landscape-classification",
        inputs=[events_path] + ([canonical_path] if canonical_path.exists() else []),
        outputs=output_paths,
        notes=f"n_unique_records={metadata['n_unique_records']}, "
        f"still_unresolved_parse_error={parse_error_stats['n_still_unresolved']}",
        started_at=started_at,
    )


def step_llm_classify_consolidate_landscape(cfg: PipelineConfig) -> dict:
    """Step 23a(b): consolidates `landscape_classification_events.csv` (one row per DeepSeek call,
    including retries) into one row per unique paper, then merges in the real metadata that already
    exists in `bulk_candidates.csv` -- no new fetches. Read-only against the pool, no spend.

    Any record whose only-ever event is `parse_error` goes to a clearly labeled side file
    (`ai_ml_landscape_classified_unresolved_parse_errors.csv`) instead of the main output --
    resolve those via `llm-classify classify` (same command, same scope) before re-running this."""
    started_at = time.monotonic()
    events_path = cfg.path("landscape_classification_events")
    pool_path = cfg.sampling_path("bulk_candidates")
    output_path = cfg.path("ai_ml_landscape_classified")
    unresolved_path = cfg.path("ai_ml_landscape_classified_unresolved_parse_errors")

    resolved, unresolved = landscape_consolidate.resolve_landscape_classifications(events_path)
    print(f"llm-classify.consolidate-landscape: {len(resolved):,} resolved, "
          f"{len(unresolved):,} still stuck as parse_error (never a successful retry).")

    unresolved.to_csv(unresolved_path, index=False)

    merge_stats = landscape_consolidate.merge_landscape_metadata(resolved, pool_path, output_path)
    print(
        f"llm-classify.consolidate-landscape: merged -- {merge_stats['n_matched']:,} matched, "
        f"{merge_stats['n_unmatched']:,} unmatched (no pool row found), "
        f"{merge_stats['n_duplicate_pool_rows_skipped']:,} duplicate pool rows skipped."
    )

    finish_step(
        "llm-classify.consolidate-landscape",
        inputs=[events_path, pool_path],
        outputs=[output_path, unresolved_path],
        notes=f"resolved={len(resolved)}, unresolved_parse_error={len(unresolved)}, "
        f"matched={merge_stats['n_matched']}, unmatched={merge_stats['n_unmatched']}",
        started_at=started_at,
    )
    return {"n_resolved": len(resolved), "n_unresolved": len(unresolved), **merge_stats}


_OVERNIGHT_CONFIG_VALIDATION_README = """# Overnight Config Validation

Does DeepSeek's classification under last night's overnight-landscape-run configuration (current
code -- including the connection-pool-sizing and NaN-identity fixes made this phase --
current CRITERIA.md, concurrency=1000) still agree with human ground truth? A genuine
apples-to-apples check, not the older 21-Aug run used as a proxy.

**Three-way comparison, all on the same trusted-pool records, all under the current
criteria_sha256:**

- `confusion_matrix_original_run.png` / `confusion_matrix_overnight_rerun.png` -- human label vs.
  DeepSeek, for the original 21-Aug run and this new re-run, side by side.
- `kappa_accuracy_comparison.png` -- Cohen's kappa + accuracy-among-decided, original run vs.
  overnight re-run, in one chart.
- `disagreement_breakdown_overnight_rerun.png` -- every disagreement in the new re-run, ranked.
- `profile_metadata.json` -- the exact real numbers behind every chart, plus a fourth number not
  shown as a chart: **DeepSeek-vs-itself** agreement between the original run and the overnight
  re-run on the records both cover (self-consistency, not human agreement) -- reuses
  `reporting/run_comparison.py`, the same mechanism already used for the earlier 1,000-record A/B
  concurrency re-validation (99.00% agreement there).
"""


def step_reporting_profile_overnight_config_validation(cfg: PipelineConfig, tier: str = "flash") -> None:
    """Step 23a(b) follow-up: re-classifies the trusted pool under last night's exact overnight-run
    configuration (see `llm-classify classify --events-out .../events.csv --scope all --concurrency
    1000`, run separately -- real spend, deliberately not triggered from here) and compares it
    against both human ground truth and the original 21-Aug run, all under the current criteria.
    Read-only, no spend -- this step only profiles a classify run that already happened."""
    started_at = time.monotonic()
    dataset_path = cfg.path("canonical_dataset")
    original_events_path = cfg.path("llm_classification_events")
    output_dir = cfg.path("overnight_config_validation_dir")
    new_events_path = output_dir / "events.csv"

    if not new_events_path.exists():
        print(
            "reporting.profile-overnight-config-validation: no re-run event log at "
            f"{new_events_path} yet -- run `llm-classify classify --events-out {new_events_path} "
            "--scope all --tier flash --mode primary --concurrency 1000 --confirm` first."
        )
        return

    dataset = pd.read_csv(dataset_path, dtype=str)
    original_events = pd.read_csv(original_events_path, dtype=str)
    new_events = pd.read_csv(new_events_path, dtype=str)
    _, _, current_hash = _criteria_text_and_hash()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    joined_original = agreement_reporting.build_population_run_joined(dataset, original_events, tier, current_hash)
    joined_new = agreement_reporting.build_population_run_joined(dataset, new_events, tier, current_hash)
    if joined_new.empty:
        print(
            "reporting.profile-overnight-config-validation: the re-run event log has no events "
            f"under the current criteria ({current_hash[:12]}...) for tier={tier!r} -- nothing to "
            "profile yet."
        )
        return

    agg_original = agreement_reporting.compute_agreement(joined_original, tier) if not joined_original.empty else None
    agg_new = agreement_reporting.compute_agreement(joined_new, tier)

    if agg_original is not None:
        cm_orig_path = output_dir / f"confusion_matrix_original_run_{tier}.png"
        agreement_reporting.plot_confusion_matrix_per_tier(joined_original, tier, cm_orig_path)
        output_paths.append(cm_orig_path)

    cm_new_path = output_dir / f"confusion_matrix_overnight_rerun_{tier}.png"
    agreement_reporting.plot_confusion_matrix_per_tier(joined_new, tier, cm_new_path)
    output_paths.append(cm_new_path)

    kappa_path = output_dir / f"kappa_accuracy_comparison_{tier}.png"
    agreements_for_chart = {"overnight_rerun": agg_new}
    if agg_original is not None:
        agreements_for_chart = {"original_run_2026-08-21": agg_original, **agreements_for_chart}
    agreement_reporting.plot_kappa_accuracy_summary(agreements_for_chart, kappa_path)
    output_paths.append(kappa_path)

    db_path = output_dir / f"disagreement_breakdown_overnight_rerun_{tier}.png"
    agreement_reporting.plot_disagreement_breakdown(joined_new, tier, db_path)
    output_paths.append(db_path)

    raw_self_consistency = run_comparison.compare_runs(original_events_path, new_events_path)
    self_consistency = {k: v for k, v in raw_self_consistency.items() if k != "disagreements"}
    self_consistency["disagreement_count"] = int(len(raw_self_consistency["disagreements"]))

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "tier": tier,
        "current_criteria_sha256": current_hash,
        "human_vs_original_run": agg_original,
        "human_vs_overnight_rerun": agg_new,
        "original_run_vs_overnight_rerun_self_consistency": self_consistency,
    }
    metadata_path = output_dir / "profile_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    readme_path = output_dir / "README.md"
    readme_path.write_text(_OVERNIGHT_CONFIG_VALIDATION_README)
    output_paths += [metadata_path, readme_path]

    print(
        f"reporting.profile-overnight-config-validation: human-vs-overnight-rerun kappa="
        f"{agg_new['kappa']:.3f}, accuracy={agg_new['accuracy_among_decided']:.3%} over "
        f"{len(joined_new):,} records -- charts + profile_metadata.json + README.md written to "
        f"{output_dir}"
    )

    finish_step(
        "reporting.profile-overnight-config-validation",
        inputs=[dataset_path, original_events_path, new_events_path],
        outputs=output_paths,
        params={"tier": tier},
        notes=f"human_vs_overnight_rerun kappa={agg_new['kappa']:.3f}, n={len(joined_new)}",
        started_at=started_at,
    )


def step_bulk_match_check_metadata(cfg: PipelineConfig, backfill: bool = False) -> dict:
    """Step 23a's metadata-completeness gate, run BEFORE any classify call on the bulk pool --
    don't let a genuinely missing abstract silently become a wasted/degraded DeepSeek call.
    `enrich_missing_canonical_metadata` is reused completely unchanged (it's already generic over
    any dataframe with pmcid/pmid/doi/title/abstract columns, not specific to
    canonical_dataset.csv) -- same real re-fetch mechanism already proven on the curated dataset.

    `backfill=False` (default) only reports the real null/empty counts -- no write, no spend.
    `backfill=True` actually calls EPMC (free, but real network time at 744k+ rows) and overwrites
    `bulk_candidates.csv` with whatever it could fill in -- a backup is written first, same
    convention as every other mutating step in this project."""
    started_at = time.monotonic()
    bulk_path = cfg.sampling_path("bulk_candidates")
    df = pd.read_csv(bulk_path, dtype=str)
    missing_title = int((df["title"].isna() | (df["title"] == "")).sum())
    missing_abstract = int((df["abstract"].isna() | (df["abstract"] == "")).sum())
    print(f"bulk-match.check-metadata: {len(df):,} rows -- {missing_title:,} missing title, "
          f"{missing_abstract:,} missing abstract.")

    if not backfill:
        return {"n_rows": len(df), "missing_title": missing_title, "missing_abstract": missing_abstract,
                "backfilled": False}

    backup_file(bulk_path)
    epmc_cfg = cfg.sources.get("epmc", {})
    client = EpmcClient(
        base_url=epmc_cfg.get("base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest"),
        page_size=epmc_cfg.get("page_size", 100),
        max_retries=epmc_cfg.get("max_retries", 5),
        backoff_factor=epmc_cfg.get("backoff_factor", 1.5),
    )
    try:
        updated, stats = enrich_missing_canonical_metadata(df, client, show_progress=True)
    finally:
        client.close()
    updated.to_csv(bulk_path, index=False)
    print(f"bulk-match.check-metadata: backfill -- targeted {stats['targeted']:,}, "
          f"found {stats['found']:,}, still missing {stats['still_missing']:,}.")

    finish_step(
        "bulk-match.check-metadata",
        inputs=[bulk_path],
        outputs=[bulk_path],
        params={"backfill": True},
        notes=f"targeted={stats['targeted']}, found={stats['found']}, still_missing={stats['still_missing']}",
        started_at=started_at,
    )
    return {"n_rows": len(updated), "missing_title": missing_title, "missing_abstract": missing_abstract,
            "backfilled": True, **stats}


def step_bulk_match_repair_metadata(
    cfg: PipelineConfig,
    merge: bool = True,
    limit: Optional[int] = None,
    fetch_only: bool = False,
    merge_only: bool = False,
) -> dict:
    """Step 23a's metadata repair -- the real replacement for the EPMC `--backfill` pass.

    Goes to **NCBI**, not Europe PMC: the pool was built from EPMC in the first place, so the rows
    missing an abstract are exactly the rows EPMC did not supply one for, and re-asking EPMC is
    largely the same question twice (confirmed live 2026-08-27 -- slow, and mostly redundant).
    PubMed is the upstream NCBI mirrors, with independent coverage.

    Fully resumable: both the pmcid/doi -> pmid conversion pass and the PubMed efetch pass append
    to JSONL checkpoints and flush per batch, so an interrupt costs at most one in-flight batch and
    a re-run picks up exactly where it stopped. The merge writes to a temp file and atomically
    replaces the pool, so the 1.8GB CSV can never be left half-written."""
    started_at = time.monotonic()
    bulk_path = cfg.sampling_path("bulk_candidates")
    records_cp = cfg.sampling_path("metadata_repair_records")
    idmap_cp = cfg.sampling_path("metadata_repair_idmap")

    stats: dict = {}
    if not merge_only:
        stats = metadata_repair.fetch_repairs(
            csv_path=bulk_path,
            records_checkpoint=records_cp,
            idmap_checkpoint=idmap_cp,
            limit=limit,
        )

    if merge and not fetch_only:
        backup_file(bulk_path)
        merge_stats = metadata_repair.merge_repairs(bulk_path, records_cp)
        stats.update(merge_stats)

        finish_step(
            "bulk-match.repair-metadata",
            inputs=[bulk_path, records_cp],
            outputs=[bulk_path],
            params={"source": "ncbi", "limit": limit},
            notes=(
                f"rows_repaired={merge_stats['rows_repaired']}, "
                f"still_missing={merge_stats['still_missing']}"
            ),
            started_at=started_at,
        )
    return stats


def step_bulk_match_complete_ids(cfg: PipelineConfig, limit: Optional[int] = None) -> dict:
    """Fills in the missing members of the pmid/pmcid/doi triple across the whole pool (not just
    the abstract-less subset) via NCBI's ID converter, so downstream linking/dedup/external lookup
    has the best identifier coverage the sources can give. Resumable; blanks only."""
    started_at = time.monotonic()
    bulk_path = cfg.sampling_path("bulk_candidates")
    checkpoint = cfg.sampling_path("id_completion_checkpoint")

    stats = metadata_repair.complete_ids(bulk_path, checkpoint, limit=limit)
    backup_file(bulk_path)
    stats.update(metadata_repair.merge_ids(bulk_path, checkpoint))

    finish_step(
        "bulk-match.complete-ids",
        inputs=[bulk_path, checkpoint],
        outputs=[bulk_path],
        params={"source": "ncbi-idconv", "limit": limit},
        notes=(
            f"filled pmid={stats['filled_pmid']}, pmcid={stats['filled_pmcid']}, "
            f"doi={stats['filled_doi']}"
        ),
        started_at=started_at,
    )
    return stats


def step_bulk_match_repair_abstracts_crossref(
    cfg: PipelineConfig, limit: Optional[int] = None, concurrency: int = 40
) -> dict:
    """Last-resort abstract recovery from Crossref for the DOIs PubMed had no abstract for."""
    started_at = time.monotonic()
    bulk_path = cfg.sampling_path("bulk_candidates")
    checkpoint = cfg.sampling_path("crossref_abstract_checkpoint")

    stats = metadata_repair.fetch_crossref_abstracts(
        bulk_path, checkpoint, limit=limit, concurrency=concurrency
    )
    backup_file(bulk_path)
    stats.update(metadata_repair.merge_crossref(bulk_path, checkpoint))

    finish_step(
        "bulk-match.repair-abstracts-crossref",
        inputs=[bulk_path, checkpoint],
        outputs=[bulk_path],
        params={"source": "crossref", "limit": limit, "concurrency": concurrency},
        notes=f"tried={stats['tried']}, found={stats['found']}, merged={stats['rows_repaired']}",
        started_at=started_at,
    )
    return stats


def step_bulk_match_annotate_provenance(cfg: PipelineConfig) -> dict:
    """Stamps abstract_source / metadata_repair_sources onto every row so the pool records WHERE
    each abstract came from (Europe PMC vs NCBI PubMed vs Crossref), rather than leaving that
    implicit in the repair checkpoints."""
    started_at = time.monotonic()
    bulk_path = cfg.sampling_path("bulk_candidates")
    backup_file(bulk_path)
    stats = metadata_repair.annotate_provenance(
        bulk_path,
        cfg.sampling_path("metadata_repair_records"),
        cfg.sampling_path("crossref_abstract_checkpoint"),
        cfg.sampling_path("metadata_repair_idmap"),
        cfg.sampling_path("id_completion_checkpoint"),
    )
    finish_step(
        "bulk-match.annotate-provenance",
        inputs=[bulk_path],
        outputs=[bulk_path],
        params={},
        notes=", ".join(f"{k or 'none'}={v}" for k, v in sorted(stats.items())),
        started_at=started_at,
    )
    return stats


def step_bulk_match_completeness_report(cfg: PipelineConfig) -> dict:
    """Read-only final state of the pool -- identifier coverage plus title/abstract gaps."""
    return metadata_repair.report_completeness(cfg.sampling_path("bulk_candidates"))


STEP_FUNCS = {
    "ingest": step_ingest_load_sources,
    "enrich": step_ingest_enrich_metadata,
    "dedupe": step_dedupe_consolidate,
    "manifest": step_fulltext_build_manifest,
    "tfidf": step_keywords_tfidf,
    "keybert": step_keywords_keybert,
    "build-lexicon": step_keywords_build_lexicon,
    "lexicon-stats": step_keywords_lexicon_stats,
    "scoring-bakeoff": step_keywords_scoring_bakeoff,
    "bulk-match-build-candidates": step_bulk_match_build_candidates,
    "sampling-stratify": step_sampling_stratify,
    "profile-dataset": step_reporting_profile_dataset,
}


def run_steps(cfg: PipelineConfig, steps: list[str]) -> None:
    for step in steps:
        if step not in STEP_FUNCS:
            raise ValueError(f"Unknown step '{step}'. Valid steps: {list(STEP_FUNCS)}")
        STEP_FUNCS[step](cfg)
