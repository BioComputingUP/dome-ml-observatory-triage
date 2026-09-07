"""Merges TF-IDF + KeyBERT extraction output with the categorized_terms.csv seed vocabulary
(MLit-Triage-Nextflow, 500 already human-vetted terms, schema Term/TF-IDF/Category -- confirmed)
into one candidate lexicon for human review at curate/pages/3_Keyword_Review.py. The reviewed
output is data/processed/keyword_lexicon.csv -- this is the "keyword trawl" checkpoint.

`lexicon_stats` exists because reviewing all ~40k raw candidates by hand isn't practical -- it
shows real term-counts at a range of thresholds so the cutoff is a data-driven decision, not a
guess (see the plan's "Lexicon threshold tool" section).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_seed_terms(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.rename(
        columns={"Term": "term", "TF-IDF": "seed_tfidf_score", "Category": "seed_category"}
    )


def _source_label(row: pd.Series) -> str:
    sources = []
    if pd.notna(row.get("tfidf_positive_mean")):
        sources.append("tfidf")
    if pd.notna(row.get("keybert_score_mean")):
        sources.append("keybert")
    if pd.notna(row.get("seed_category")):
        sources.append("seed")
    return "+".join(sources) if sources else "unknown"


def build_candidate_lexicon(
    tfidf_df: pd.DataFrame,
    keybert_df: pd.DataFrame,
    seed_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = pd.merge(tfidf_df, keybert_df, on="term", how="outer")
    if seed_df is not None and not seed_df.empty:
        merged = pd.merge(merged, seed_df, on="term", how="left")

    merged["source"] = merged.apply(_source_label, axis=1)
    merged["review_status"] = "pending"
    merged["notes"] = ""

    sort_cols = [c for c in ("discriminative_score", "document_frequency") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(by=sort_cols, ascending=False)
    return merged.reset_index(drop=True)


def lexicon_stats(
    candidates_df: pd.DataFrame,
    discriminative_thresholds: list[float] | None = None,
    document_frequency_thresholds: list[int] | None = None,
) -> pd.DataFrame:
    """Term-counts remaining at a range of thresholds, so a cutoff can be picked from real
    numbers. Both threshold dimensions are reported independently (each ANDed with "score/freq
    is not null") since a term might qualify via TF-IDF discriminative_score, KeyBERT
    document_frequency, or both."""
    discriminative_thresholds = discriminative_thresholds or [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    document_frequency_thresholds = document_frequency_thresholds or [1, 2, 5, 10, 20, 50]

    rows = []
    if "discriminative_score" in candidates_df.columns:
        for threshold in discriminative_thresholds:
            count = (candidates_df["discriminative_score"] >= threshold).sum()
            rows.append({"dimension": "discriminative_score", "min_threshold": threshold, "terms_remaining": int(count)})
    if "document_frequency" in candidates_df.columns:
        for threshold in document_frequency_thresholds:
            count = (candidates_df["document_frequency"] >= threshold).sum()
            rows.append({"dimension": "document_frequency", "min_threshold": threshold, "terms_remaining": int(count)})
    return pd.DataFrame(rows)
