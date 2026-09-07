import pandas as pd

from dome_triage.keywords.scoring import WeightedSumScorer
from dome_triage.keywords.scoring_bakeoff import (
    _confusion_metrics_at_threshold,
    _youden_optimal_threshold,
    run_bakeoff,
)

TEXTS = [
    "We trained a random forest classifier on genomic data to predict cancer subtype.",
    "This paper reviews the history of impressionist painting in France.",
    "Machine learning models including deep neural networks were applied to protein prediction.",
    "A cross-sectional survey of patient satisfaction in three hospitals.",
]
TRUE_LABELS = [1, 0, 1, 0]
LEXICON_TERMS = ["random forest", "machine learning", "neural network", "classifier"]


def test_run_bakeoff_reports_one_row_per_scorer():
    scorers = {"weighted-sum": WeightedSumScorer({t: 1.0 for t in LEXICON_TERMS})}
    report = run_bakeoff(scorers, TEXTS, TRUE_LABELS, LEXICON_TERMS)

    assert len(report) == 1
    assert report.iloc[0]["scorer"] == "weighted-sum"
    expected_columns = {
        "auroc",
        "threshold_youden",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "n_positive",
        "n_negative",
        "n_total",
        "condition",
        "correlation_with_label",
        "precision_recall_at_quantiles",
    }
    assert expected_columns.issubset(report.columns)


def test_run_bakeoff_gives_perfect_auroc_for_a_clearly_separable_scorer():
    scorers = {"weighted-sum": WeightedSumScorer({t: 1.0 for t in LEXICON_TERMS})}
    report = run_bakeoff(scorers, TEXTS, TRUE_LABELS, LEXICON_TERMS)
    assert report.iloc[0]["auroc"] == 1.0


def test_run_bakeoff_confusion_matrix_is_perfect_for_a_clearly_separable_scorer():
    # LEXICON_TERMS cleanly separate TEXTS 0&2 (positive) from 1&3 (negative) -- at the
    # Youden-optimal threshold this scorer should classify every record correctly.
    scorers = {"weighted-sum": WeightedSumScorer({t: 1.0 for t in LEXICON_TERMS})}
    report = run_bakeoff(scorers, TEXTS, TRUE_LABELS, LEXICON_TERMS)
    row = report.iloc[0]

    assert row["true_positive"] == 2
    assert row["true_negative"] == 2
    assert row["false_positive"] == 0
    assert row["false_negative"] == 0
    assert row["accuracy"] == 1.0
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0


def test_run_bakeoff_reports_correct_positive_and_negative_volume():
    # TRUE_LABELS = [1, 0, 1, 0] -- 2 positive, 2 negative, regardless of how many scorers run.
    scorers = {
        "weighted-sum": WeightedSumScorer({t: 1.0 for t in LEXICON_TERMS}),
        "also-weighted-sum": WeightedSumScorer({}),
    }
    report = run_bakeoff(scorers, TEXTS, TRUE_LABELS, LEXICON_TERMS)

    assert (report["n_positive"] == 2).all()
    assert (report["n_negative"] == 2).all()
    assert (report["n_total"] == 4).all()


def test_run_bakeoff_tags_rows_with_the_given_condition_label():
    scorers = {"weighted-sum": WeightedSumScorer({t: 1.0 for t in LEXICON_TERMS})}
    report = run_bakeoff(scorers, TEXTS, TRUE_LABELS, LEXICON_TERMS, condition_label="my_condition")
    assert (report["condition"] == "my_condition").all()


def test_run_bakeoff_sorts_by_auroc_descending():
    good_scorer = WeightedSumScorer({t: 1.0 for t in LEXICON_TERMS})
    bad_scorer = WeightedSumScorer({})  # never matches anything -- score is always 0
    report = run_bakeoff(
        {"good": good_scorer, "bad": bad_scorer}, TEXTS, TRUE_LABELS, LEXICON_TERMS
    )
    assert report.iloc[0]["scorer"] == "good"


def test_youden_optimal_threshold_picks_the_perfect_separation_point():
    labels = pd.Series([1, 1, 0, 0])
    scores = pd.Series([0.9, 0.8, 0.3, 0.1])  # a clean gap between 0.8 and 0.3
    threshold = _youden_optimal_threshold(labels, scores)
    assert 0.3 < threshold <= 0.8


def test_confusion_metrics_at_threshold_counts_correctly():
    labels = pd.Series([1, 1, 0, 0])
    scores = pd.Series([0.9, 0.2, 0.8, 0.1])  # one of each: TP, FN, FP, TN
    metrics = _confusion_metrics_at_threshold(labels, scores, threshold=0.5)

    assert metrics["true_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
