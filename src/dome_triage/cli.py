"""Typer CLI. Every subcommand is a thin wrapper around the shared functions in
pipeline/steps.py (or, for `curate` / `fulltext fetch`, curate/state.py and fulltext/manifest.py
directly) -- there is no separate workflow-engine orchestration, see AGENTS.md.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from dome_triage.config import PipelineConfig, resolve_path
from dome_triage.curate.state import materialize_events, merge_original_cohort_second_review
from dome_triage.fulltext.manifest import save_fulltext_xml
from dome_triage.pipeline import steps as pipeline_steps

app = typer.Typer(help="dome-triage: literature triage pipeline for the DOME registry.")

ingest_app = typer.Typer(help="Load and enrich records from the configured label sources.")
dedupe_app = typer.Typer(help="Cluster and consolidate raw records into the canonical dataset.")
fulltext_app = typer.Typer(help="Build/query the full-text availability manifest.")
keywords_app = typer.Typer(help="TF-IDF + KeyBERT keyword extraction, lexicon, and scoring.")
bulk_match_app = typer.Typer(help="Bulk blunt-match candidate construction against Europe PMC.")
sampling_app = typer.Typer(help="Stratified sampling over the scored bulk candidate pool.")
curate_app = typer.Typer(help="Human curation app and event-log materialization.")
reporting_app = typer.Typer(help="Dataset profiling and visualization.")
pipeline_app = typer.Typer(help="Run multiple steps in sequence.")
llm_classify_app = typer.Typer(help="Step 20: DeepSeek second-curator blind classification.")

app.add_typer(ingest_app, name="ingest")
app.add_typer(dedupe_app, name="dedupe")
app.add_typer(fulltext_app, name="fulltext")
app.add_typer(keywords_app, name="keywords")
app.add_typer(bulk_match_app, name="bulk-match")
app.add_typer(sampling_app, name="sampling")
app.add_typer(curate_app, name="curate")
app.add_typer(reporting_app, name="reporting")
app.add_typer(pipeline_app, name="pipeline")
app.add_typer(llm_classify_app, name="llm-classify")

_CONFIG_DIR_OPTION = typer.Option("configs", "--config-dir", help="Directory containing the config YAMLs.")


def _load_config(config_dir: str) -> PipelineConfig:
    base = resolve_path(config_dir)
    return PipelineConfig(
        sources_path=base / "sources.yaml",
        pipeline_path=base / "pipeline.yaml",
        tfidf_path=base / "tfidf.yaml",
        keybert_path=base / "keybert.yaml",
        sampling_path=base / "sampling.yaml",
    )


@ingest_app.command("load-sources")
def ingest_load_sources(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_ingest_load_sources(_load_config(config_dir))


@ingest_app.command("enrich-metadata")
def ingest_enrich_metadata(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_ingest_enrich_metadata(_load_config(config_dir))


@dedupe_app.command("consolidate")
def dedupe_consolidate(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_dedupe_consolidate(_load_config(config_dir))


@fulltext_app.command("build-manifest")
def fulltext_build_manifest(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_fulltext_build_manifest(_load_config(config_dir))


@fulltext_app.command("fetch")
def fulltext_fetch(
    pmcid: str = typer.Option(..., "--pmcid", help="PMCID to fetch full-text XML for."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    cfg = _load_config(config_dir)
    output_dir = resolve_path(cfg.pipeline["fulltext"]["local_dir"])
    saved_path = save_fulltext_xml(pmcid, output_dir)
    if saved_path is None:
        typer.echo(f"Could not fetch full-text XML for {pmcid} (not open-access or not found).")
        raise typer.Exit(code=1)
    typer.echo(f"Saved {saved_path}")


@keywords_app.command("tfidf")
def keywords_tfidf(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_keywords_tfidf(_load_config(config_dir))


@keywords_app.command("keybert")
def keywords_keybert(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_keywords_keybert(_load_config(config_dir))


@keywords_app.command("build-lexicon")
def keywords_build_lexicon(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    pipeline_steps.step_keywords_build_lexicon(_load_config(config_dir))


@keywords_app.command("lexicon-stats")
def keywords_lexicon_stats(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Term-counts remaining at a range of thresholds -- pick a defensible cutoff from real
    numbers instead of reviewing all ~40k candidates by hand."""
    pipeline_steps.step_keywords_lexicon_stats(_load_config(config_dir))


@keywords_app.command("materialize-lexicon")
def keywords_materialize_lexicon(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Folds keyword_review_events.csv (from the Streamlit Keyword Review page) into
    keyword_lexicon.csv (positive), keyword_lexicon_exclusionary.csv (negative), and
    keyword_lexicon_irrelevant.csv (irrelevant) -- last decision per term wins, regardless of
    which pile or manual entry produced it."""
    pipeline_steps.step_keywords_materialize_lexicon(_load_config(config_dir))


@keywords_app.command("seed-additional-terms")
def keywords_seed_additional_terms(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Writes keywords/curated_terms.py's curated positive/negative additions (ML algorithms,
    generative/agentic/LLM terms, flagship biodata terms, non-methods publication-type terms) to
    their own tier-2 files -- separate from your human-curated tier-1 lexicon/exclusionary_lexicon,
    skipping anything already decided."""
    pipeline_steps.step_keywords_seed_additional_terms(_load_config(config_dir))


@keywords_app.command("suggest-final-lexicon")
def keywords_suggest_final_lexicon(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Combines tier 1 (materialized lexicon/exclusionary_lexicon) with tier 2 (added terms),
    runs the explainable cleanup heuristic (redundant-unigram removal, cross-list tension
    flagging), and writes tier 3 -- suggested_lexicon / suggested_exclusionary_lexicon / a
    cleanup log. Never modifies tier 1's live files; promoting tier 3 to production is manual."""
    pipeline_steps.step_keywords_suggest_final_lexicon(_load_config(config_dir))


_EXCLUSIONARY_WEIGHT_OPTION = typer.Option(
    1.0,
    "--exclusionary-weight",
    help="Penalty weight applied to the exclusionary lexicon's score, if "
    "data/processed/keyword_lexicon_exclusionary.csv exists (from `keywords materialize-lexicon`).",
)


@keywords_app.command("scoring-bakeoff")
def keywords_scoring_bakeoff(
    exclusionary_weight: float = _EXCLUSIONARY_WEIGHT_OPTION,
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Validates every relevance-scoring algorithm against the already-known-labeled records
    before trusting any of them to rank the unlabeled bulk pool."""
    pipeline_steps.step_keywords_scoring_bakeoff(_load_config(config_dir), exclusionary_weight)


@keywords_app.command("score-bulk-match")
def keywords_score_bulk_match(
    scorer: str = typer.Option(
        "weighted-sum", "--scorer", help='One of "weighted-sum", "bm25", "tfidf-cosine", or "all".'
    ),
    exclusionary_weight: float = _EXCLUSIONARY_WEIGHT_OPTION,
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    pipeline_steps.step_keywords_score_bulk_match(_load_config(config_dir), scorer, exclusionary_weight)


@bulk_match_app.command("fetch")
def bulk_match_fetch(
    year_from: int = typer.Option(..., "--year-from", help="First year to fetch (inclusive)."),
    year_to: int = typer.Option(..., "--year-to", help="Last year to fetch (inclusive)."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Fetches every AI/ML-matching Europe PMC record (resultType=core, full metadata incl. MeSH)
    for the whole [year_from, year_to] range in one invocation -- one EPMC query per year
    internally (checkpointed, resumable), with live per-year progress and a printed + logged
    AI-only/ML-only/combined-deduplicated count breakdown."""
    pipeline_steps.step_bulk_match_fetch(_load_config(config_dir), year_from, year_to)


@bulk_match_app.command("build-candidates")
def bulk_match_build_candidates(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Consolidates every completed year fetched so far into one deduplicated candidate pool."""
    pipeline_steps.step_bulk_match_build_candidates(_load_config(config_dir))


@bulk_match_app.command("check-metadata")
def bulk_match_check_metadata(
    backfill: bool = typer.Option(False, "--backfill", help="Actually re-fetch and fill missing title/abstract (free, real network time). Default: report only."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 23a's metadata-completeness gate for the bulk pool -- real null/empty title/abstract
    counts, with an optional real backfill pass (same EpmcClient.get_by_ids mechanism already
    proven on canonical_dataset.csv) before any classify call is made."""
    pipeline_steps.step_bulk_match_check_metadata(_load_config(config_dir), backfill)


@bulk_match_app.command("repair-metadata")
def bulk_match_repair_metadata(
    limit: Optional[int] = typer.Option(None, "--limit", help="Only fetch this many pmids this run (testing). Resumable -- a later larger run continues, never repeats."),
    fetch_only: bool = typer.Option(False, "--fetch-only", help="Fetch to the checkpoint but do not touch the pool CSV."),
    merge_only: bool = typer.Option(False, "--merge-only", help="Skip fetching; just apply an existing checkpoint into the pool."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Repairs missing abstracts in the bulk pool via NCBI PubMed (not Europe PMC -- the pool came
    from EPMC, so EPMC is the source that already didn't have them). Batched 200 ids/request,
    concurrent up to NCBI's rate ceiling, fully resumable via JSONL checkpoints, live progress with
    ETA, and an atomic replace of the pool at the end. Set NCBI_API_KEY to go from 3 to 10 req/s."""
    pipeline_steps.step_bulk_match_repair_metadata(
        _load_config(config_dir), limit=limit, fetch_only=fetch_only, merge_only=merge_only
    )


@bulk_match_app.command("complete-ids")
def bulk_match_complete_ids(
    limit: Optional[int] = typer.Option(None, "--limit", help="Only look up this many ids per type this run (testing). Resumable."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Fills in the missing members of the pmid/pmcid/doi triple across the WHOLE pool via NCBI's
    ID converter -- batched, concurrent, resumable, blanks only. Set NCBI_API_KEY for 10 req/s."""
    pipeline_steps.step_bulk_match_complete_ids(_load_config(config_dir), limit=limit)


@bulk_match_app.command("repair-abstracts-crossref")
def bulk_match_repair_abstracts_crossref(
    limit: Optional[int] = typer.Option(None, "--limit", help="Only try this many DOIs this run (testing). Resumable."),
    concurrency: int = typer.Option(40, "--concurrency", help="Parallel Crossref requests; the client still paces itself from Crossref's own rate headers."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Last-resort abstract recovery from Crossref for DOIs PubMed had no abstract for. Set
    CROSSREF_MAILTO to use Crossref's faster polite pool."""
    pipeline_steps.step_bulk_match_repair_abstracts_crossref(
        _load_config(config_dir), limit=limit, concurrency=concurrency
    )


@bulk_match_app.command("annotate-provenance")
def bulk_match_annotate_provenance(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Stamps abstract_source (europepmc/pubmed/crossref) and metadata_repair_sources onto every
    row, so the pool itself records where each abstract came from."""
    pipeline_steps.step_bulk_match_annotate_provenance(_load_config(config_dir))


@bulk_match_app.command("completeness-report")
def bulk_match_completeness_report(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Read-only: identifier coverage and title/abstract gaps across the bulk pool."""
    pipeline_steps.step_bulk_match_completeness_report(_load_config(config_dir))


@sampling_app.command("stratify")
def sampling_stratify(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Stratified sample (match-score band x journal x year, capped per stratum in
    configs/sampling.yaml) over the scored bulk pool -- merges new candidates into
    canonical_dataset.csv for the curation app's queue."""
    pipeline_steps.step_sampling_stratify(_load_config(config_dir))


@ingest_app.command("fetch-clear-negatives")
def ingest_fetch_clear_negatives(
    year_from: int = typer.Option(..., "--year-from"),
    year_to: int = typer.Option(..., "--year-to"),
    sample_size: int = typer.Option(2000, "--sample-size"),
    merge_limit: Optional[int] = typer.Option(
        None,
        "--merge-limit",
        help="Caps how many of the sample-size pool actually merge into canonical_dataset.csv "
        "this run (defaults to sample_size). Keeps the raw label balance from swinging sharply "
        "negative before human review -- see STEPS_Progress.md Step 14.",
    ),
    n_windows: int = typer.Option(40, "--n-windows"),
    max_per_window: int = typer.Option(
        1500,
        "--max-per-window",
        help="Caps Phase 1's cheap resultType=lite lookup per date window (breaks out of the "
        "results generator early -- fewer HTTP round-trips, not a server-side limit). Full "
        "resultType=core records are only fetched afterward, for the diversified winners. See "
        "STEPS_Progress.md Step 14 for why this two-phase design replaced the original "
        "single-phase one.",
    ),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Samples genuine AI/ML-free negatives (live EPMC query, the inverse of bulk-match's AI/ML
    query -- see clear_negative_sampler.py), journal/year-stratified -- merges up to --merge-limit
    into canonical_dataset.csv for the curation app's queue, same as sampling stratify."""
    pipeline_steps.step_ingest_fetch_clear_negatives(
        _load_config(config_dir), year_from, year_to, sample_size, merge_limit, n_windows, max_per_window
    )


@ingest_app.command("screen-clear-negatives")
def ingest_screen_clear_negatives(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 14b: re-scores clear_negative_candidates.csv against the promoted lexicon (same
    scorer + Youden threshold Step 12 already validated) and flags any that score suspiciously
    high despite the AI/ML exclusion query -- surfaced in the Curate app for extra scrutiny
    before you trust them as genuine negatives."""
    pipeline_steps.step_ingest_screen_clear_negatives(_load_config(config_dir))


@ingest_app.command("merge-strong-negatives")
def ingest_merge_strong_negatives(
    limit: int = typer.Option(2000, "--limit"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 14c: merges up to `--limit` of the clear-negative candidates screened in Step 14b into
    canonical_dataset.csv, tagged source_name=clear_negative_sampler_strong. The whole screened
    pool is eligible -- `needs_screening` (Step 14b's BM25 flag) is diagnostic only and does NOT
    exclude anyone from this merge; flagged rows are merged too, just visibly flagged in the data
    for optional later spot-checking. Run after `ingest screen-clear-negatives`."""
    pipeline_steps.step_ingest_merge_strong_negatives(_load_config(config_dir), limit)


@ingest_app.command("fetch-clear-negatives-filtered")
def ingest_fetch_clear_negatives_filtered(
    year_from: int = typer.Option(..., "--year-from"),
    year_to: int = typer.Option(..., "--year-to"),
    raw_pool_size: int = typer.Option(3000, "--raw-pool-size"),
    target_size: int = typer.Option(500, "--target-size"),
    n_windows: int = typer.Option(40, "--n-windows"),
    max_per_window: int = typer.Option(1500, "--max-per-window"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 19d: fetches a new clear-negatives batch using the exact same method as Step 14
    (unchanged), but this time gated by the robust NLTK-based non-methods detector as a HARD
    exclusion -- review/commentary/meta-analysis/case-report/etc. content is dropped before
    diversification, unlike Step 14b's diagnostic-only needs_screening. --raw-pool-size (default
    3000) is deliberately much larger than --target-size (default 500) -- a real share of ordinary
    biomedical literature is review/non-methods content, so a generous raw pool is needed to net
    enough clean survivors. Writes an interim file only (data/interim/clear_negative_candidates_
    filtered.csv) -- inspect it, then run `ingest merge-clear-negatives-filtered` to actually merge."""
    pipeline_steps.step_ingest_fetch_clear_negatives_filtered(
        _load_config(config_dir), year_from, year_to, raw_pool_size, target_size, n_windows, max_per_window
    )


@ingest_app.command("merge-clear-negatives-filtered")
def ingest_merge_clear_negatives_filtered(
    limit: int = typer.Option(500, "--limit"),
    source_tag: str = typer.Option("clear_negative_sampler_strong_filtered_v2", "--source-tag"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 19d: merges the interim file from `ingest fetch-clear-negatives-filtered` into
    canonical_dataset.csv, tagged with a distinct source_name (default
    clear_negative_sampler_strong_filtered_v2) so this batch stays separately auditable from Step
    14c's clear_negative_sampler_strong in the provenance breakdown chart. Refuses to run (raises)
    if the interim file still has any row flagged likely_review_or_non_methods=True -- defense in
    depth against a stale/hand-edited interim file."""
    pipeline_steps.step_ingest_merge_filtered_clear_negatives(_load_config(config_dir), limit, source_tag)


@ingest_app.command("backfill-canonical-metadata")
def ingest_backfill_canonical_metadata(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Fills in missing title/abstract/journal/authors/year/MeSH for canonical_dataset.csv rows
    that have an ID but are missing their title or abstract -- see METHODS_REVIEW.md Sec 8.2 (the
    428 registry_confirmed positives with no abstract, a real training-data leak). Never
    overwrites an already-populated field."""
    pipeline_steps.step_ingest_backfill_canonical_metadata(_load_config(config_dir))


@ingest_app.command("add-llm-language-model-positives")
def ingest_add_llm_language_model_positives(
    pmids_file: str = typer.Option(
        "data/processed/llm_language_model_seed_pmids.csv", "--pmids-file",
        help="Path to the PMID seed file (pmid,name,url,use -- only rows with `use` marked are included).",
    ),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 19c: fetches full EPMC metadata for whichever PMIDs are marked in the `use` column of
    the seed file (~200 pre-populated real EPMC search hits across BERT/bio, foundation-model,
    agentic-AI, and biomedical-LLM papers -- mark the ones you want, blank ones are skipped) and
    merges them into canonical_dataset.csv as positive/human_curated, tagged
    source_name="manual_llm_language_model_seed". A PMID already present in canonical_dataset.csv
    under any other source/label is skipped, not relabeled -- a warning names it and its current
    label so that's never silent."""
    cfg = _load_config(config_dir)
    pipeline_steps.step_ingest_add_llm_language_model_positives(cfg, resolve_path(pmids_file))


@ingest_app.command("fetch-llm-seed-pool")
def ingest_fetch_llm_seed_pool(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19c (standard-curation-route build): fetches full EPMC metadata for every PMID in the
    seed file (all 192, regardless of the `use` column -- that column drove the earlier, superseded
    "mark and bulk-merge" design) and writes data/interim/llm_seed_candidates_pool.csv, a
    RawRecord-shaped pool ready for the Curate app's "LLM Seed Review" page. Run this once (or
    again after editing the seed file) before opening that page."""
    pipeline_steps.step_ingest_fetch_llm_seed_pool(_load_config(config_dir))


@reporting_app.command("profile-dataset")
def reporting_profile_dataset(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 18: profiles canonical_dataset.csv into seven charts (label overview, journal
    diversity, provenance breakdown, BM25 score distribution, year distribution, year coverage
    vs. the full bulk pool in both log and linear scale) plus profile_metadata.json and a README,
    written to data/processed/dataset_profile/ (overwritten each run). A mid-pipeline snapshot --
    see the generated README.md for what it does and doesn't reflect yet."""
    pipeline_steps.step_reporting_profile_dataset(_load_config(config_dir))


@reporting_app.command("profile-original-cohort-second-review")
def reporting_profile_original_cohort_second_review(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19b's diagnostic companion: profiles the original-cohort second review (the 1,907
    pre-app negatives re-reviewed via the Curate app's "Original Cohort Review" page) into its own
    3-chart set -- decision breakdown, BM25-vs-second-review confusion matrix, and BM25 score
    distribution split by outcome -- written to
    data/processed/original_cohort_second_review_profile/. Requires `curate
    merge-original-cohort-second-review` to have been run first (reads the
    original_cohort_review_* columns it writes)."""
    pipeline_steps.step_reporting_profile_original_cohort_second_review(_load_config(config_dir))


@reporting_app.command("profile-review-filter")
def reporting_profile_review_filter(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19d's diagnostic companion: profiles the likely_review_or_non_methods flag
    (`curate flag-likely-reviews`'s output) dataset-wide into a 2-chart set -- label vs. flag
    breakdown, and the top text-terms/pub_types actually triggering the flag -- written to
    data/processed/review_filter_profile/. Requires `curate flag-likely-reviews` to have been run
    first."""
    pipeline_steps.step_reporting_profile_review_filter(_load_config(config_dir))


@reporting_app.command("agreement")
def reporting_agreement(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 20: Cohen's kappa + accuracy-among-decided for every DeepSeek tier classified against
    the shared blind sample, written to human_llm_agreement_report.csv."""
    pipeline_steps.step_reporting_agreement(_load_config(config_dir))


@reporting_app.command("profile-second-curator")
def reporting_profile_second_curator(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 20's dedicated visualization folder: confusion matrix + disagreement breakdown per
    tier, a cross-tier kappa/accuracy summary, a flash-vs-pro comparison (if both tiers were run),
    and the undeterminable-subset fallback comparison (if any fallback batch was run) -- written to
    data/processed/second_curator_profile/."""
    pipeline_steps.step_reporting_profile_second_curator(_load_config(config_dir))


@reporting_app.command("profile-post-review-consensus")
def reporting_profile_post_review_consensus(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 20a's dedicated visualization folder: did the human's FINAL Cross Curate Resolve
    decision end up agreeing with DeepSeek more often than the ORIGINAL label did? A reversal
    breakdown per tier plus an original-vs-post-review kappa comparison -- written to
    data/processed/post_review_consensus_profile/, a separate folder from second_curator_profile/
    (which stays as the pre-review snapshot)."""
    pipeline_steps.step_reporting_profile_post_review_consensus(_load_config(config_dir))


@reporting_app.command("profile-full-population")
def reporting_profile_full_population(
    tier: str = typer.Option(..., "--tier", help='"flash" (cheap) or "pro" (expensive).'),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 20f: profiles Step 20e's full re-run -- trusted-pool confusion matrix/disagreement
    breakdown/kappa at full scale (5,825 records, not just the 1,000-sample), the EPMC
    "clear negative" candidate-pool confirmation rate (1,775 records, never individually
    human-reviewed), and a whole-evaluated-population comparison (original 1,000 vs. new 5,825,
    kept as two separate bars since they were classified under different CRITERIA.md versions) --
    written to data/processed/full_population_profile/."""
    pipeline_steps.step_reporting_profile_full_population(_load_config(config_dir), tier)


@reporting_app.command("profile-ai-ml-landscape-classification")
def reporting_profile_ai_ml_landscape_classification(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 23a(b): profiles the full ~827k-record landscape classification run -- classification
    breakdown, a real reproducible duplicate/parse-error audit, and a positive-rate comparison
    against the curated trusted set (disjoint populations by design, so a rate comparison, not a
    kappa score). Read-only, no spend. Written to data/processed/landscape_classification_profile/."""
    pipeline_steps.step_reporting_profile_ai_ml_landscape_classification(_load_config(config_dir))


@reporting_app.command("profile-overnight-config-validation")
def reporting_profile_overnight_config_validation(
    tier: str = typer.Option("flash", "--tier"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 23a(b) follow-up: three-way comparison -- human ground truth vs. the original 21-Aug
    DeepSeek run vs. a fresh re-run under last night's exact overnight-landscape-run config
    (current code, current criteria, concurrency=1000). Requires the re-run event log to already
    exist at data/processed/overnight_config_validation/events.csv -- see `llm-classify classify
    --events-out ...` in STEPS_Progress.md for the (real-spend) command that produces it.
    Read-only, no spend itself."""
    pipeline_steps.step_reporting_profile_overnight_config_validation(_load_config(config_dir), tier)


@reporting_app.command("profile-full-population-consensus")
def reporting_profile_full_population_consensus(
    tier: str = typer.Option(..., "--tier", help='"flash" (cheap) or "pro" (expensive).'),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 20f: the full-population analog of profile-post-review-consensus -- run this AFTER
    manually resolving Step 20e's new trusted-pool disagreements via Cross Curate Resolve. A
    reversal breakdown plus an original-vs-post-review kappa comparison, written to
    data/processed/full_population_consensus_profile/."""
    pipeline_steps.step_reporting_profile_full_population_consensus(_load_config(config_dir), tier)


@curate_app.command("launch")
def curate_launch(
    port: int = typer.Option(8501, "--port"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Launches the Streamlit curation app (blocking). Prefer `docker compose up curate` for
    normal use; this command exists for local (non-Docker) development."""
    app_path = Path(__file__).resolve().parent / "curate" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )


@curate_app.command("materialize")
def curate_materialize(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Folds curation_events.csv into canonical_dataset.csv (last decision wins per record,
    conflicts with a trusted prior label are flagged, never silently overwritten). `bulk_pool_path`
    is passed so decisions made by browsing directly from the full bulk pool (Curate page's "Full
    AI/ML bulk pool" queue source) -- reaching records never sampled into canonical_dataset.csv by
    `sampling stratify` -- get inserted as new rows instead of silently dropped."""
    cfg = _load_config(config_dir)
    events_path = resolve_path(cfg.pipeline["curation"]["events_log"])
    dataset_path = cfg.path("canonical_dataset")
    bulk_pool_path = cfg.sampling_path("bulk_candidates_scored")
    materialize_events(dataset_path, events_path, dataset_path, bulk_pool_path=bulk_pool_path)
    typer.echo(f"Materialized {events_path} into {dataset_path}")


@curate_app.command("materialize-original-cohort-review")
def curate_materialize_original_cohort_review(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19: folds original_cohort_review_events.csv (the pre-app manual-cohort re-review
    pass, `4_Original_Cohort_Review.py`) into canonical_dataset.csv -- reuses `materialize_events`
    unchanged, same "last decision wins, conflict with a trusted prior label is flagged, never
    silently overwritten" behavior as `curate materialize`. Every record in this cohort already
    exists in canonical_dataset.csv by construction (it's filtered FROM that file), so no
    `bulk_pool_path` fallback is needed here."""
    cfg = _load_config(config_dir)
    events_path = resolve_path(cfg.pipeline["curation"]["original_cohort_review_events"])
    dataset_path = cfg.path("canonical_dataset")
    materialize_events(dataset_path, events_path, dataset_path)
    typer.echo(f"Materialized {events_path} into {dataset_path}")


@curate_app.command("merge-original-cohort-second-review")
def curate_merge_original_cohort_second_review(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19b (additive variant): folds original_cohort_review_events.csv into
    canonical_dataset.csv as new, clearly-named original_cohort_review_* columns -- deliberately
    NOT via `materialize_events` (used by `curate materialize-original-cohort-review` above).
    `label`/`label_confidence`/`has_conflict`/`sources`/`curation_tag`/`notes`/`updated_at` are
    never touched by this command; the original decision stays exactly as it was. Use this instead
    of `materialize-original-cohort-review` when the second review should be visible and
    provenanced alongside the original label rather than folded into (or flagged as conflicting
    with) it. See `merge_original_cohort_second_review`'s docstring in curate/state.py for the
    full column list and reasoning."""
    cfg = _load_config(config_dir)
    events_path = resolve_path(cfg.pipeline["curation"]["original_cohort_review_events"])
    dataset_path = cfg.path("canonical_dataset")
    merge_original_cohort_second_review(dataset_path, events_path, dataset_path)
    typer.echo(f"Merged {events_path} into {dataset_path} as new original_cohort_review_* columns")


@curate_app.command("materialize-llm-seed-review")
def curate_materialize_llm_seed_review(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19c (standard-curation-route build): folds llm_seed_review_events.csv (the "LLM Seed
    Review" Curate-app page's decisions) into canonical_dataset.csv, reusing `materialize_events`
    unchanged -- same "last decision wins, conflict with a trusted prior label is flagged, never
    silently overwritten" behavior as `curate materialize`. Unlike `materialize-original-cohort-
    review`, these records do NOT already exist in canonical_dataset.csv (they're freshly fetched
    candidates, not an existing cohort), so `bulk_pool_path` is set to the fetched candidate pool
    (`ingest fetch-llm-seed-pool`'s output) -- exactly the same "insert as a new row with full
    provenance/metadata, using the exact construction dedupe consolidate would produce" mechanism
    the main Curate page's "Full AI/ML bulk pool" browsing mode already relies on
    (`_insert_missing_records_from_bulk_pool`), reused here rather than duplicated."""
    cfg = _load_config(config_dir)
    events_path = resolve_path(cfg.pipeline["curation"]["llm_seed_review_events"])
    dataset_path = cfg.path("canonical_dataset")
    bulk_pool_path = cfg.path("llm_seed_candidate_pool")
    materialize_events(dataset_path, events_path, dataset_path, bulk_pool_path=bulk_pool_path)
    typer.echo(f"Materialized {events_path} into {dataset_path}")


@curate_app.command("flag-likely-reviews")
def curate_flag_likely_reviews(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 19d: runs the same shared NLTK-based non-methods detector
    (curate/review_detector.py) used to gate the new negative batch across the WHOLE
    canonical_dataset.csv, adding likely_review_or_non_methods / likely_review_or_non_methods_
    detail as new, additive columns -- label/label_confidence/sources/every other existing column
    is never touched. Run this AFTER `ingest merge-clear-negatives-filtered`, so the flag also
    covers the newly merged rows and `reporting profile-review-filter`'s charts are internally
    consistent against the final dataset."""
    pipeline_steps.step_curate_flag_likely_reviews(_load_config(config_dir))


@curate_app.command("materialize-cross-curate-resolutions")
def curate_materialize_cross_curate_resolutions(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 20b: folds cross_curate_resolution_events.csv (the "Cross Curate Resolve" page's human
    tie-break decisions on DeepSeek/human disagreements) into canonical_dataset.csv via
    materialize_cross_curate_resolutions() -- deliberately NOT materialize_events(), which would
    turn any genuinely-changed final decision into the literal string "conflict" instead of
    applying it (every record here already has a trusted prior label by construction, which is
    exactly the case materialize_events() treats as suspicious). Snapshots the prior label and
    both DeepSeek tiers' original classifications into new cross_curate_* columns, then actually
    applies the human's final decision as the real label."""
    pipeline_steps.step_curate_materialize_cross_curate_resolutions(_load_config(config_dir))


_TIER_OPTION = typer.Option(..., "--tier", help='"flash" (cheap) or "pro" (expensive).')
_CONFIRM_OPTION = typer.Option(
    False, "--confirm",
    help="Required to actually spend real money -- review the printed cost projection first.",
)


@llm_classify_app.command("validate-criteria")
def llm_classify_validate_criteria(
    tier: str = typer.Option("flash", "--tier"),
    confirm: bool = _CONFIRM_OPTION,
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 20's explicit "first" requirement: runs both the primary (3-way) and forced-choice
    (2-way) prompts over curation_criteria/validation_fixtures.csv (hand-pick ~12-15 real records
    with an expected_classification first) and prints a pass/fail table + the primary variant's
    undetermined rate + one full constructed prompt, before any larger spend."""
    pipeline_steps.step_llm_classify_validate_criteria(_load_config(config_dir), tier, confirm)


@llm_classify_app.command("calibrate")
def llm_classify_calibrate(
    tier: str = _TIER_OPTION,
    n: int = typer.Option(8, "--n"),
    mode: str = typer.Option("primary", "--mode", help='"primary" (default), "forced_guess", or "rag".'),
    confirm: bool = _CONFIRM_OPTION,
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Fires `n` real, live calls against `tier` and logs real token usage -- no dollar figure is
    computed here; check your real DeepSeek dashboard balance deduction, then run `project-cost`."""
    pipeline_steps.step_llm_classify_calibrate(_load_config(config_dir), tier, n, mode, confirm)


@llm_classify_app.command("project-cost")
def llm_classify_project_cost(
    target_n: int = typer.Option(..., "--target-n"),
    observed_usd_spent_flash: Optional[float] = typer.Option(None, "--observed-usd-spent-flash"),
    observed_usd_spent_pro: Optional[float] = typer.Option(None, "--observed-usd-spent-pro"),
    mode: str = typer.Option("primary", "--mode"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Pure read/print, no spend: real, calibration-derived $/record and projected cost for
    --target-n records, for whichever tier(s) you pass a real observed dashboard-deduction figure
    for. This is the number `classify --estimated-usd` should be copied from."""
    pipeline_steps.step_llm_classify_project_cost(
        _load_config(config_dir), target_n, observed_usd_spent_flash, observed_usd_spent_pro, mode
    )


@llm_classify_app.command("sample")
def llm_classify_sample(
    n_positive: int = typer.Option(500, "--n-positive"),
    n_negative: int = typer.Option(500, "--n-negative"),
    seed: int = typer.Option(42, "--seed"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Draws the shared blind sample once (both tiers judge the same records) -- plain random
    across the trusted human-curated population, excluding validate-criteria's fixture records."""
    pipeline_steps.step_llm_classify_sample(_load_config(config_dir), n_positive, n_negative, seed)


@llm_classify_app.command("compare-runs")
def llm_classify_compare_runs(
    path_a: str = typer.Option(..., "--a", help="First event log (e.g. the original run)."),
    path_b: str = typer.Option(..., "--b", help="Second event log (e.g. the high-concurrency re-run)."),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Compares two classification event logs over their shared records: agreement rate, crosstab,
    and every disagreement. Read-only, no spend."""
    pipeline_steps.step_llm_classify_compare_runs(_load_config(config_dir), path_a, path_b)


@llm_classify_app.command("consolidate-landscape")
def llm_classify_consolidate_landscape(config_dir: str = _CONFIG_DIR_OPTION) -> None:
    """Step 23a(b): consolidates the landscape event log into one row per unique paper and merges
    in real metadata (title/abstract/pmid/pmcid/doi/year/authors/...) already in bulk_candidates.csv
    -- no new fetches. Records whose only-ever event is parse_error go to a clearly labeled side
    file instead of the main output. Read-only against the pool, no spend."""
    pipeline_steps.step_llm_classify_consolidate_landscape(_load_config(config_dir))


@llm_classify_app.command("classify")
def llm_classify_classify(
    tier: str = _TIER_OPTION,
    scope: str = typer.Option(
        "sample", "--scope",
        help='"sample" (default), "all" (trusted pool, held-out 1,000 excluded), '
        '"staged_file" (an incremental batch from build_incoming_documents.py -- needs --input), '
        '"all_plus_candidates" (also includes the heuristic_candidate EPMC clear-negative pool), '
        'or "bulk_pool_excluding_curated" (Step 23a -- the full bulk EPMC AI/ML pool minus '
        'anything already in canonical_dataset.csv; writes to a separate '
        'landscape_classification_events.csv, never touches the trusted-pool event log).',
    ),
    estimated_usd: Optional[float] = typer.Option(
        None, "--estimated-usd", help="Copied from `project-cost`'s printed projection for this exact scope."
    ),
    confirm: bool = _CONFIRM_OPTION,
    concurrency: int = typer.Option(
        20, "--concurrency",
        help="Concurrent API calls. " + """Throughput is concurrency / per-call latency, and the latency is not tunable -- the model writes its answer at ~133 tokens/sec, so a call takes as long as its output is. """
        "Classification output is tiny (~57 tokens) so calls are fast and 20 is fine for small "
        "runs; Step 23a's full-landscape pass used 1000. DeepSeek's limits: 500 pro / 2500 flash.",
    ),
    mode: str = typer.Option(
        "primary", "--mode",
        help='"primary" (default, 3-way incl. undeterminable) or "forced_choice" (2-way, no '
        "undeterminable -- runs the FULL scope under forced choice, distinct from "
        "run-fallback's undetermined-subset-only re-ask).",
    ),
    events_out: Optional[str] = typer.Option(
        None, "--events-out",
        help="Write (and resume from) this event-log path instead of the scope default. A fresh "
             "path means nothing counts as already-classified, so the same records re-run -- for "
             "A/B re-validation. Prompt, criteria and parsing are unchanged.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit",
        help="Classify only the first N not-yet-classified records in scope -- a real, safe "
        "N-record test before a full-scope run (e.g. Step 23a's 500-record check). Already-"
        "classified records don't count against the limit and re-running with a larger --limit "
        "continues from where the previous run stopped.",
    ),
    ignore_lock: bool = typer.Option(
        False, "--ignore-lock",
        help="Start even though another run holds this event log's lock. Only for a genuinely "
             "stale lock (a container the daemon lost). Two live runs on one log pay twice for "
             "the same records -- that cost $9.63 on 2026-09-03.",
    ),
    input_path: Optional[str] = typer.Option(
        None, "--input",
        help="Required by --scope staged_file: the CSV that "
             "moros_pipeline/scripts/build_incoming_documents.py wrote. Its `pid` column is the "
             "document _id, and becomes the record_id in the event log so the classification "
             "merges back to the right document.",
    ),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 20's primary paid classify run -- never touches canonical_dataset.csv (comparison
    signal only). Resumable: already-classified records for this tier/criteria are skipped, not
    re-paid for."""
    pipeline_steps.step_llm_classify_classify(
        _load_config(config_dir), tier, scope, estimated_usd, confirm, concurrency, mode, limit,
        events_out, input_path, ignore_lock,
    )


@llm_classify_app.command("calibrate-fallback")
def llm_classify_calibrate_fallback(
    mode: str = typer.Option(..., "--mode", help='"forced_guess" or "rag".'),
    tier: str = _TIER_OPTION,
    n: int = typer.Option(8, "--n"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Calibrates the small undetermined-subset fallback batch separately from the primary run --
    a RAG-enabled call may cost meaningfully more than closed-book, never assumed proportional."""
    pipeline_steps.step_llm_classify_calibrate_fallback(_load_config(config_dir), mode, tier, n)


@llm_classify_app.command("run-fallback")
def llm_classify_run_fallback(
    mode: str = typer.Option(..., "--mode", help='"forced_guess" or "rag".'),
    tier: str = _TIER_OPTION,
    estimated_usd: Optional[float] = typer.Option(None, "--estimated-usd"),
    confirm: bool = _CONFIRM_OPTION,
    concurrency: int = typer.Option(20, "--concurrency"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Re-asks only the records the PRIMARY run answered undeterminable for this tier -- "rag"
    requires DeepSeek's search-tool support to be verified first (see STEPS_Progress.md Step 20)."""
    pipeline_steps.step_llm_classify_run_fallback(
        _load_config(config_dir), mode, tier, estimated_usd, confirm, concurrency
    )


@llm_classify_app.command("enrich")
def llm_classify_enrich(
    tier: str = typer.Option("flash", "--tier", help='"flash" (default) or "pro".'),
    concurrency: int = typer.Option(
        800, "--concurrency",
        help="Concurrent API calls. " + """Throughput is concurrency / per-call latency, and the latency is not tunable -- the model writes its answer at ~133 tokens/sec, so a call takes as long as its output is. """
        "Enrichment averages ~6,000 output tokens per record, so ~45s per call: 800 gives "
        "~400 records/min, 100 gives ~130, 40 gives ~53 (a real 2026-09-03 run crawled for hours "
        "at 40 before this was understood). DeepSeek's documented flash ceiling is 2500 and Step "
        "23a ran classification at 1000. The HTTP pool is sized from this value.",
    ),
    events_out: Optional[str] = typer.Option(
        None, "--events-out",
        help="Write (and resume from) this event-log path instead of the scope default. A fresh "
             "path means nothing counts as already-classified, so the same records re-run -- for "
             "A/B re-validation. Prompt, criteria and parsing are unchanged.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Enrich only the first N remaining records -- cheap real-API smoke check."
    ),
    input_path: Optional[str] = typer.Option(
        None, "--input", help="Override the input CSV (default: the Step 20j trial positive set from configs)."
    ),
    reasoning_effort: Optional[str] = typer.Option(
        None, "--reasoning-effort",
        help='Thinking effort: "none" | "low" | "high" | "max". Omit to keep the provider '
             'default (high). Output tokens are ~94% of this step\'s cost and ~98% of them are '
             'the reasoning trace, so this is the cost lever -- but "none" was measured to push '
             "vocabulary violations from 6% to 46.5% (thinking_ablation/), so lower it on "
             "evidence, not by default.",
    ),
    domain_rendering: str = typer.Option(
        "flat", "--domain-rendering",
        help='"flat" (three comma-separated tier lists, what every run to date used) or "tree" '
             "(the 259 EDAM terms printed as the one subtree they actually are). Changes "
             "vocab_sha256, so it correctly starts a fresh batch.",
    ),
    ignore_lock: bool = typer.Option(
        False, "--ignore-lock",
        help="Start even though another run holds this event log's lock. Only for a genuinely "
             "stale lock (a container the daemon lost). Two live runs on one log pay twice for "
             "the same records -- that cost $9.63 on 2026-09-03.",
    ),
    retry_truncated: bool = typer.Option(
        False, "--retry-truncated",
        help="Re-attempt records whose response hit max_tokens. Off by default: retrying at the "
             "same cap re-truncates deterministically and burns a full budget each time. Worth "
             "passing only after raising ENRICHMENT_MAX_TOKENS.",
    ),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 20j: enrich the positive trial set with domain (3 EDAM tiers), learning_paradigm,
    model_family, and open-vocabulary model_type -- additive-only (never re-asks the
    positive/negative question, so it cannot conflict with the finalized classification).
    Resumable: re-running skips everything already enriched under the current vocabularies and
    retries parse_errors. No cost gate on this command -- real token totals and the prefix-cache
    hit rate are printed at the end and recorded per event."""
    pipeline_steps.step_llm_classify_enrich(
        _load_config(config_dir), tier, concurrency, limit,
        Path(input_path) if input_path else None,
        events_out,
        reasoning_effort,
        domain_rendering,
        retry_truncated,
        ignore_lock,
    )


@reporting_app.command("profile-enrichment")
def reporting_profile_enrichment(
    tier: str = typer.Option("flash", "--tier"),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    """Step 20j's success/usage visualization: tags-per-record distributions, top model types,
    paradigm/family frequencies, seed-normalization share and vocab-violation rates -- written to
    data/processed/enrichment_profile/."""
    pipeline_steps.step_reporting_profile_enrichment(_load_config(config_dir), tier)


@pipeline_app.command("run")
def pipeline_run(
    steps: str = typer.Option(
        ...,
        "--steps",
        help=(
            "Comma-separated step names from: ingest, enrich, dedupe, manifest, tfidf, keybert, "
            "build-lexicon, lexicon-stats, scoring-bakeoff, bulk-match-build-candidates, "
            "sampling-stratify. A convenience for re-running an already-verified chain quickly -- "
            "the manual one-command-at-a-time flow (see README.md) is the primary workflow."
        ),
    ),
    config_dir: str = _CONFIG_DIR_OPTION,
) -> None:
    cfg = _load_config(config_dir)
    step_list = [s.strip() for s in steps.split(",") if s.strip()]
    pipeline_steps.run_steps(cfg, step_list)


if __name__ == "__main__":
    app()
