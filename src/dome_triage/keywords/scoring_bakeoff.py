"""Empirically compares MatchScorer implementations against the already-known-labeled ~4,320
consolidated records (real positive/negative ground truth) -- answers "which scoring method
actually separates known positives from known negatives" before any scorer is trusted to rank
the much larger unlabeled bulk pool from ingest/bulk_match.py.

Every scorer (weighted-sum, BM25, TF-IDF-cosine) produces a continuous ranking score, not an
inherent yes/no classification -- AUROC is rank-based and needs no cutoff, but accuracy/
precision/recall/the confusion matrix (true/false positive/negative) do. The cutoff used here is
each scorer's own Youden's-J-optimal point (the point on its ROC curve maximizing true-positive-
rate minus false-positive-rate) -- a standard, principled way to pick one operating point from a
continuous score, not an arbitrary one, and reported explicitly (`threshold_youden`) so it's
transparent and reproducible. See `run_bakeoff`'s `condition_label` for running the same scorers
with vs without the exclusionary lexicon applied, to compare directly.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dome_triage.keywords.scoring import MatchScorer

_NAN_CONFUSION_METRICS = {
    "accuracy": float("nan"),
    "precision": float("nan"),
    "recall": float("nan"),
    "f1": float("nan"),
    "true_positive": 0,
    "true_negative": 0,
    "false_positive": 0,
    "false_negative": 0,
}


def _precision_recall_at_quantile(scores: pd.Series, true_labels: pd.Series, quantile: float) -> tuple[float, float]:
    threshold = scores.quantile(quantile)
    predicted_positive = scores >= threshold
    true_positive = int((predicted_positive & (true_labels == 1)).sum())
    false_positive = int((predicted_positive & (true_labels == 0)).sum())
    false_negative = int((~predicted_positive & (true_labels == 1)).sum())

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    return precision, recall


def _youden_optimal_threshold(labels: pd.Series, scores: pd.Series) -> float:
    """The score cutoff that maximizes (true positive rate - false positive rate) across the ROC
    curve -- the standard way to convert a continuous ranking score into a single yes/no operating
    point, per this module's docstring."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = (tpr - fpr).argmax()
    return float(thresholds[best_idx])


def _confusion_metrics_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict:
    predicted_positive = (scores >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(labels, predicted_positive),
        "precision": precision_score(labels, predicted_positive, zero_division=0),
        "recall": recall_score(labels, predicted_positive, zero_division=0),
        "f1": f1_score(labels, predicted_positive, zero_division=0),
        "true_positive": int(((predicted_positive == 1) & (labels == 1)).sum()),
        "true_negative": int(((predicted_positive == 0) & (labels == 0)).sum()),
        "false_positive": int(((predicted_positive == 1) & (labels == 0)).sum()),
        "false_negative": int(((predicted_positive == 0) & (labels == 1)).sum()),
    }


def run_bakeoff(
    scorers: dict[str, MatchScorer],
    texts: list[str],
    true_labels: list[int],
    lexicon_terms: list[str],
    exclusionary_terms: list[str] | None = None,
    exclusionary_weight: float = 1.0,
    condition_label: str = "positive_lexicon_only",
) -> pd.DataFrame:
    """`true_labels` must be 1 for positive / 0 for negative -- already-labeled records only.
    `exclusionary_terms`, if given, lets this run reflect the actual production scoring approach
    (positive lexicon minus exclusionary penalty) rather than positive-lexicon-only scoring --
    tag the run with `condition_label` (e.g. "positive_lexicon_only" vs
    "positive_plus_exclusionary_lexicon") so two separate calls can be concatenated into one
    before/after comparison report -- see pipeline/steps.py::step_keywords_scoring_bakeoff, which
    calls this twice."""
    labels = pd.Series(true_labels)
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    rows = []

    for name, scorer in scorers.items():
        scored = scorer.score_corpus(
            texts, lexicon_terms, exclusionary_terms=exclusionary_terms, exclusionary_weight=exclusionary_weight
        )
        scores = pd.Series([s for s, _ in scored])

        try:
            auroc = roc_auc_score(labels, scores)
            threshold = _youden_optimal_threshold(labels, scores)
            confusion_metrics = _confusion_metrics_at_threshold(labels, scores, threshold)
        except ValueError:
            # e.g. only one class present in this sample -- ROC/AUROC undefined
            auroc = float("nan")
            threshold = float("nan")
            confusion_metrics = dict(_NAN_CONFUSION_METRICS)

        quantile_report = []
        for quantile in (0.5, 0.75, 0.9):
            precision, recall = _precision_recall_at_quantile(scores, labels, quantile)
            quantile_report.append(f"q{int(quantile * 100)}: P={precision:.2f} R={recall:.2f}")

        rows.append(
            {
                "scorer": name,
                "condition": condition_label,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_total": n_positive + n_negative,
                "auroc": auroc,
                "threshold_youden": threshold,
                **confusion_metrics,
                "correlation_with_label": scores.corr(labels),
                "precision_recall_at_quantiles": "; ".join(quantile_report),
            }
        )

    return pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
