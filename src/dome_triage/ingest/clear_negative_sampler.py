"""Separate true-negative sampler. The AI/ML-filtered bulk query (bulk_match.py) structurally
cannot produce a paper with zero AI/ML mention -- true negatives need a different source.

Design: sample several random week-long date windows across the target year range, then a
**two-phase** fetch against each window (see `fetch_clear_negatives`'s docstring for the real
performance reasoning this was rewritten for), then stratify-downsample by journal x year via
`sampling/stratified.py` (the same tested bucketing Step 13's bulk-pool sampling uses) rather than
a plain random sample -- so the resulting negative pool is diverse across journals and years, not
just whatever the random date windows happened to catch. This is a pragmatic way to get genuine
randomness without needing true random access into a 40M+ record cursor-only API -- a documented
design choice, not a hidden assumption.

**Data source, worth being explicit about**: `EXCLUDE_QUERY` below is a live query against the
*full* Europe PMC corpus via `EpmcClient.search()`, the structural inverse of `bulk_match.py`'s
`AI_ML_QUERY` (which requires "artificial intelligence"/"machine learning"; this requires their
absence). It never reads `bulk_candidates.csv` or any other local file -- every record here is
independently confirmed by EPMC's own search to not mention AI/ML terms at all, genuinely disjoint
from the 750k AI/ML-matched pool by construction, not a downsample or filter of it.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from dome_triage.curate.review_detector import annotate_non_methods
from dome_triage.ingest.bulk_match import core_result_to_raw_record
from dome_triage.ingest.epmc_client import EpmcClient
from dome_triage.ingest.id_mapping import clean_doi, clean_pmcid, clean_pmid
from dome_triage.ingest.source_loaders import raw_records_to_dataframe
from dome_triage.sampling.stratified import build_strata, stratified_sample

# Core AI/ML domain phrases -- the original 4 terms plus the other headline subfields a paper
# could be "about AI" through without ever saying "artificial intelligence"/"machine learning".
CORE_AI_ML_TERMS = (
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "large language model",
    "generative ai",
)

# Specific, unambiguous ML method/algorithm names -- drawn directly from the human-approved
# positive lexicon (keyword_lexicon.csv, Step 8d of STEPS_Progress.md), not a separate guess at
# what "common" means. A paper can easily use "random forest" or "XGBoost" without ever saying
# "machine learning", so the original 4-phrase query alone let plenty of real ML-methods papers
# through into the "negative" pool. Deliberately excludes generic single words that are also
# ordinary English/stats vocabulary outside ML (e.g. bare "regression", "forest", "transformer") --
# same reasoning as keyword_lexicon_exclusionary.csv's cross-token dampening problem (see
# STEPS_Progress.md Step 12); those bare words would over-exclude huge swaths of ordinary
# biomedical literature (anatomic "regions", ecological "forest" cover, electrical
# "transformers") for comparatively little precision gain here, since Step 14b's BM25 lexicon
# screen (using the *full* lexicon, not just this list) is the real, more precise second check
# that actually earns this batch the "strong negative" label -- this query is just a coarse,
# cheap first pass to keep obvious ML-methods papers from ever entering the fetched pool at all.
COMMON_ML_METHOD_TERMS = (
    "random forest",
    "support vector machine",
    "convolutional neural network",
    "recurrent neural network",
    "long short-term memory",
    "gradient boosting",
    "xgboost",
    "naive bayes",
    "k-means clustering",
    "k-nearest neighbors",
    "logistic regression model",
    "decision tree",
    "gaussian mixture model",
    "generative adversarial network",
    "variational autoencoder",
    "transformer model",
    "ensemble learning",
    "hierarchical clustering",
    "self-organizing map",
    "principal component analysis",
)

_EXCLUDED_TERMS = CORE_AI_ML_TERMS + COMMON_ML_METHOD_TERMS
_EXCLUDE_CLAUSE = " OR ".join(f'"{term}"' for term in _EXCLUDED_TERMS)
EXCLUDE_QUERY = f"SRC:MED NOT ({_EXCLUDE_CLAUSE})"


def _random_week_windows(
    year_from: int, year_to: int, n_windows: int, seed: int = 42
) -> list[tuple[date, date]]:
    rng = random.Random(seed)
    start = date(year_from, 1, 1)
    end = date(year_to, 12, 25)
    span_days = (end - start).days

    windows = []
    for _ in range(n_windows):
        offset = rng.randint(0, span_days)
        window_start = start + timedelta(days=offset)
        window_end = window_start + timedelta(days=6)
        windows.append((window_start, window_end))
    return windows


def _lite_result_to_row(result: dict, window_start: date, window_end: date) -> dict:
    """Extracts only what's needed to stratify (IDs + journal + year), from a `resultType=lite`
    EPMC result -- a much smaller payload than `resultType=core` (no abstract, no MeSH, no
    keywords). `journalTitle` is lite's flat field name; the `journalInfo.journal.title` fallback
    covers both older test fixtures and the (unlikely but not impossible) case of a differently
    shaped response, at zero cost if the flat field is already there."""
    journal = result.get("journalTitle") or (result.get("journalInfo") or {}).get("journal", {}).get("title")
    return {
        "pmid": clean_pmid(result.get("pmid")),
        "pmcid": clean_pmcid(result.get("pmcid")),
        "doi": clean_doi(result.get("doi")),
        "journal": journal,
        "year": result.get("pubYear"),
        "window_start": str(window_start),
        "window_end": str(window_end),
    }


def _fetch_full_records_for_winners(client: EpmcClient, winners: pd.DataFrame) -> list:
    """Phase 2: batch-fetches full `resultType=core` records (title/abstract/MeSH/etc.) for only
    the diversified winners, via `EpmcClient.get_by_ids` -- the same tiered pmid -> pmcid -> doi
    lookup pattern `ingest/enrich.py::enrich_missing_canonical_metadata` already uses. This is
    where the expensive, full-metadata fetch actually happens -- deliberately only for the ~2,000
    records that made it through stratification, not the full weekly volume Phase 1 saw."""
    remaining = set(winners.index)
    core_results: dict[int, dict] = {}
    for id_col, id_type in (("pmid", "pmid"), ("pmcid", "pmcid"), ("doi", "doi")):
        if not remaining:
            break
        lookup: dict[str, int] = {}
        for i in remaining:
            value = winners.at[i, id_col]
            if isinstance(value, str) and value.strip():
                lookup[value] = i
        if not lookup:
            continue
        found = client.get_by_ids(list(lookup.keys()), id_type=id_type)
        for id_value, result in found.items():
            i = lookup[id_value]
            core_results[i] = result
            remaining.discard(i)

    records = []
    for i, result in core_results.items():
        records.append(
            core_result_to_raw_record(
                result,
                source_name="clear_negative_sampler",
                source_file=f"live_query:{winners.at[i, 'window_start']}_{winners.at[i, 'window_end']}",
                label="negative",
                label_confidence="heuristic_candidate",
            )
        )
    if remaining:
        print(f"clear_negative_sampler: {len(remaining)} winners had no full-record match on the "
              "follow-up lookup (no ID at all, or a stale/dropped record) -- dropped, not "
              "substituted, so the final pool may be slightly smaller than requested.")
    return records


def fetch_clear_negatives(
    client: EpmcClient,
    year_from: int,
    year_to: int,
    sample_size: int,
    n_windows: int = 40,
    top_n_journals: int = 15,
    year_bucket_width: int = 5,
    max_per_window: int = 1500,
) -> pd.DataFrame:
    """Two-phase fetch -- rewritten after the original single-phase version proved far too slow
    against real EPMC volumes. A single week's exclude-query hit count can be in the tens of
    thousands (confirmed live: 67,695-70,912 hits for one real test week), so the original
    approach -- fetching full `resultType=core` records, abstracts and all, for *every* matching
    paper in *every* one of `n_windows` weeks before ever downsampling to `sample_size` -- could
    mean paginating through well over a million full-metadata records just to keep ~2,000 of them.
    That volume, not the number of excluded terms in the query, was the actual bottleneck (query
    *evaluation* is a fraction of a second either way -- it's the sheer number of matching records
    and the size of each one that costs the time).

    Phase 1 (this function, first half): queries each of the `n_windows` random week windows with
    `resultType=lite` (a small payload -- just enough to stratify: journal, year, IDs), capped at
    `max_per_window` per window. The cap is enforced by simply breaking out of `client.search()`'s
    result generator early once reached -- since that generator fetches one page per HTTP request
    and yields records from it, breaking early means no further pages (and no further requests)
    are ever issued for that window, not a server-side limit. Phase 2
    (`_fetch_full_records_for_winners`): the lite pool is stratified/diversified down to
    `sample_size` exactly as before (same `build_strata`/`stratified_sample` mechanism), and only
    *then* are full `resultType=core` records fetched -- for just the winners, via
    `EpmcClient.get_by_ids` (batches of 40 IDs per request internally). The expensive full-metadata
    fetch now only ever happens for the records that actually make it into the pool.
    """
    windows = _random_week_windows(year_from, year_to, n_windows)
    lite_rows = []
    for window_start, window_end in windows:
        query = f"({EXCLUDE_QUERY}) AND (FIRST_PDATE:[{window_start} TO {window_end}])"
        n_from_window = 0
        for result in client.search(query, result_type="lite", show_progress=True):
            lite_rows.append(_lite_result_to_row(result, window_start, window_end))
            n_from_window += 1
            if n_from_window >= max_per_window:
                break

    lite_df = pd.DataFrame(lite_rows)
    print(f"clear_negative_sampler: {len(lite_df)} raw candidates (lightweight lookup, capped at "
          f"{max_per_window}/window) across {n_windows} live-EPMC date windows before journal/year "
          "stratification")

    # Stratify by journal x year (build_strata/stratified_sample -- the same tested bucketing
    # Step 13's bulk-pool sampling uses; score_col=None since these candidates have no meaningful
    # lexicon score to band by -- they were explicitly selected for NOT matching it) so the
    # downsample is diverse, not just whatever the random date windows happened to catch.
    lite_df["year"] = pd.to_numeric(lite_df["year"], errors="coerce")
    if len(lite_df) > sample_size:
        strata_df = build_strata(
            lite_df, score_col=None, top_n_journals=top_n_journals, year_bucket_width=year_bucket_width
        )
        strata_cols = ["journal_bucket", "year_bucket"]
        n_strata = strata_df[strata_cols].drop_duplicates().shape[0]
        cap_per_stratum = max(1, math.ceil(sample_size / n_strata)) if n_strata else sample_size
        sampled, report = stratified_sample(strata_df, strata_cols, cap_per_stratum, random_state=42)
        print(f"clear_negative_sampler: {n_strata} journal x year strata, cap_per_stratum="
              f"{cap_per_stratum} -> {len(sampled)} stratified candidates")
        print(report.to_string(index=False))

        if len(sampled) > sample_size:
            sampled = sampled.sample(n=sample_size, random_state=42)
        winners = sampled.drop(columns=strata_cols).reset_index(drop=True)
    else:
        winners = lite_df.reset_index(drop=True)

    print(f"clear_negative_sampler: fetching full records for {len(winners)} winners...")
    records = _fetch_full_records_for_winners(client, winners)
    return raw_records_to_dataframe(records)


def select_diversified_pool(
    pool: pd.DataFrame,
    limit: int,
    top_n_journals: int = 15,
    year_bucket_width: int = 5,
) -> pd.DataFrame:
    """Re-diversifies `pool` by journal x year (`build_strata`/`stratified_sample` -- the same
    tested bucketing `fetch_clear_negatives` above uses) and caps to `limit`. Extracted from what
    used to be `select_strong_negatives`'s own body (that function is now a thin wrapper around
    this, its own public behavior/tests unchanged) so Step 19d's review-gated batch
    (`fetch_filtered_clear_negatives` below) can reuse the exact same re-diversify/cap logic
    without going through Step 14b's separate BM25 screening at all -- an independent signal from
    Step 19d's own text/pub-type gate. If `pool` already has <= `limit` rows, returned unchanged
    (no sampling needed)."""
    if len(pool) <= limit:
        return pool.reset_index(drop=True)

    working = pool.copy()
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    strata_df = build_strata(
        working, score_col=None, top_n_journals=top_n_journals, year_bucket_width=year_bucket_width
    )
    strata_cols = ["journal_bucket", "year_bucket"]
    n_strata = strata_df[strata_cols].drop_duplicates().shape[0]
    cap_per_stratum = max(1, math.ceil(limit / n_strata)) if n_strata else limit
    sampled, report = stratified_sample(strata_df, strata_cols, cap_per_stratum, random_state=42)
    print(f"clear_negative_sampler: {n_strata} journal x year strata over the pool, "
          f"cap_per_stratum={cap_per_stratum} -> {len(sampled)} re-diversified candidates")
    print(report.to_string(index=False))

    if len(sampled) > limit:
        sampled = sampled.sample(n=limit, random_state=42)
    return sampled.drop(columns=strata_cols).reset_index(drop=True)


def select_strong_negatives(
    screened_df: pd.DataFrame,
    limit: int,
    top_n_journals: int = 15,
    year_bucket_width: int = 5,
) -> tuple[pd.DataFrame, int]:
    """Step 14c's core logic: from `clear_negative_candidates_screened.csv` (Step 14b's output --
    same rows as the fetch, plus `lexicon_score__bm25`/`needs_screening`), select up to `limit`
    candidates to merge as confirmed negatives. "Strong" here comes from construction -- every row
    was independently confirmed by a live EPMC query to not mention any AI/ML term (Step 14, the
    expanded `EXCLUDE_QUERY` above) -- not from a second BM25 gate. **On explicit instruction,
    `needs_screening` is never used to exclude a candidate here** -- it's Step 14b's diagnostic
    flag (visible in the data and in that step's plotted score distribution) for optional manual
    follow-up later, not a filter. Every candidate in `screened_df` is eligible for selection
    regardless of its flag.

    Delegates the actual re-diversify/cap-to-`limit` work to `select_diversified_pool` above.
    Returns (selected_df, n_flagged) where `n_flagged` counts `needs_screening == True` rows
    *within the selected batch* -- purely informational, so the caller can report how many of what
    got merged also happen to score above the lexicon threshold, without it having affected who
    got merged.
    """
    selected = select_diversified_pool(screened_df, limit, top_n_journals, year_bucket_width)
    n_flagged = int((selected["needs_screening"].astype(str) == "True").sum())
    return selected, n_flagged


def fetch_filtered_clear_negatives(
    client: EpmcClient,
    year_from: int,
    year_to: int,
    raw_pool_size: int,
    target_size: int,
    exclusionary_lexicon_path: Path,
    n_windows: int = 40,
    top_n_journals: int = 15,
    year_bucket_width: int = 5,
    max_per_window: int = 1500,
) -> tuple[pd.DataFrame, dict]:
    """Step 19d: fetches a large raw pool via `fetch_clear_negatives` above -- UNCHANGED, "the way
    we had working before" -- then applies the robust NLTK-based non-methods detector
    (`curate/review_detector.py::annotate_non_methods`) as a HARD exclusion gate before
    diversifying. Unlike Step 14b's `needs_screening`, which is diagnostic-only by deliberate
    earlier design (never gates the merge), this drops flagged rows outright: a negative-class
    training set needs genuine primary research, not review/commentary/case-report/etc. content
    that simply fails to mention AI/ML terms.

    `raw_pool_size` is deliberately decoupled from `target_size` and expected to be much larger
    (the CLI default is 3000 for a 500 target) -- per explicit instruction, a real share of
    ordinary biomedical literature is review/commentary/meta-analysis/etc. content, so a generous
    raw pool is needed to net enough clean survivors after the gate runs.

    Returns `(selected, stats)` -- `stats` = `{fetched, dropped_by_gate, survivors, selected,
    shortfall}`, always printed, never raised. A final `selected` smaller than `target_size` is a
    real, loudly-reported outcome (not enough clean candidates turned up in this fetch), not a
    silent failure or a hard error -- matching `_fetch_full_records_for_winners`'s existing
    "dropped, not substituted" reporting convention above."""
    raw_pool = fetch_clear_negatives(
        client,
        year_from,
        year_to,
        raw_pool_size,
        n_windows=n_windows,
        top_n_journals=top_n_journals,
        year_bucket_width=year_bucket_width,
        max_per_window=max_per_window,
    )
    flagged = annotate_non_methods(raw_pool, exclusionary_lexicon_path)
    survivors = flagged[~flagged["likely_review_or_non_methods"]].reset_index(drop=True)
    selected = select_diversified_pool(survivors, target_size, top_n_journals, year_bucket_width)

    stats = {
        "fetched": len(raw_pool),
        "dropped_by_gate": len(raw_pool) - len(survivors),
        "survivors": len(survivors),
        "selected": len(selected),
        "shortfall": max(0, target_size - len(selected)),
    }
    print(
        f"clear_negative_sampler: fetched {stats['fetched']}, dropped {stats['dropped_by_gate']} "
        f"as likely review/non-methods content, {stats['survivors']} survivors, diversified down "
        f"to {stats['selected']} (target {target_size}"
        + (f", shortfall {stats['shortfall']}" if stats["shortfall"] else "")
        + ")"
    )
    return selected, stats
