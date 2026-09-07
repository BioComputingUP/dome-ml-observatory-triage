"""TF-IDF extraction with a positive-vs-baseline discriminative score. Config: configs/tfidf.yaml.
negative_entries.csv doubles as the baseline general-biomedical-ML corpus (see AGENTS.md /
ROADMAP.md Phase 1) so discriminative_score = tfidf_mean(positive) - tfidf_mean(baseline) surfaces
terms that distinguish DOME-relevant papers from ML/bio literature in general, not just from
plain English.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from dome_triage.keywords.preprocess import clean_text


def extract_tfidf_terms(
    positive_texts: list[str], baseline_texts: list[str], config: dict
) -> pd.DataFrame:
    extra_stopwords = set(config.get("extra_stopwords", []))
    cleaned_positive = [clean_text(t, extra_stopwords) for t in positive_texts]
    cleaned_baseline = [clean_text(t, extra_stopwords) for t in baseline_texts]

    vectorizer = TfidfVectorizer(
        sublinear_tf=config.get("sublinear_tf", True),
        ngram_range=tuple(config.get("ngram_range", [1, 3])),
        min_df=config.get("min_df", 3),
        max_df=config.get("max_df", 0.85),
        max_features=config.get("max_features", 20000),
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(cleaned_positive + cleaned_baseline)
    terms = vectorizer.get_feature_names_out()

    n_pos = len(cleaned_positive)
    pos_mean = np.asarray(matrix[:n_pos].mean(axis=0)).ravel()
    baseline_mean = np.asarray(matrix[n_pos:].mean(axis=0)).ravel()

    return (
        pd.DataFrame(
            {
                "term": terms,
                "tfidf_positive_mean": pos_mean,
                "tfidf_baseline_mean": baseline_mean,
                "discriminative_score": pos_mean - baseline_mean,
            }
        )
        .sort_values("discriminative_score", ascending=False)
        .reset_index(drop=True)
    )
