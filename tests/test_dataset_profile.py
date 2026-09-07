import json

import pandas as pd

from dome_triage.reporting.dataset_profile import (
    _bm25_score_distribution_data,
    _bm25_youden_confusion_matrix_data,
    _bm25_youden_performance_data,
    _journal_diversity_data,
    _label_overview_data,
    _parse_source_names,
    _provenance_category_data,
    _year_coverage_vs_bulk_pool_data,
    _year_distribution_data,
)


def _sources(*names: str) -> str:
    return json.dumps([{"source_name": name} for name in names])


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "r1", "journal": "Nature", "year": "2020", "label": "positive",
                "label_confidence": "human_curated", "sources": _sources("dome_top_curate_positive"),
            },
            {
                "record_id": "r2", "journal": "Nature", "year": "2020", "label": "negative",
                "label_confidence": "human_curated", "sources": _sources("dome_top_curate_negative"),
            },
            {
                "record_id": "r3", "journal": "Nature", "year": "2021", "label": "negative",
                "label_confidence": "heuristic_candidate", "sources": _sources("clear_negative_sampler_strong"),
            },
            {
                "record_id": "r4", "journal": "PLOS ONE", "year": "2019", "label": "skipped",
                "label_confidence": "registry_confirmed", "sources": _sources("dome_registry_231_gold"),
            },
            {
                "record_id": "r5", "journal": "PLOS ONE", "year": "2019.0", "label": "undeterminable",
                "label_confidence": "human_curated", "sources": "not valid json",
            },
        ]
    )


def test_label_overview_data_counts_every_label():
    counts = _label_overview_data(_dataset())
    assert counts.to_dict() == {"positive": 1, "negative": 2, "skipped": 1, "undeterminable": 1}


def test_journal_diversity_data_buckets_by_records_per_journal():
    # Nature: 3 records -> bucket "2-5". PLOS ONE: 2 records -> bucket "2-5" too.
    counts = _journal_diversity_data(_dataset())
    assert counts["2-5"] == 2  # two journals (Nature, PLOS ONE) each land in the 2-5 bucket
    assert counts["1"] == 0
    assert counts.sum() == 2  # total distinct journals, not total records


def test_journal_diversity_data_separates_singletons_from_larger_journals():
    df = pd.DataFrame(
        {"journal": ["A", "B", "B", "B", "C", "C", "C", "C", "C", "C"], "label": ["positive"] * 10}
    )
    counts = _journal_diversity_data(df)
    assert counts["1"] == 1  # journal A
    assert counts["2-5"] == 1  # journal B (3 records)
    assert counts["6-10"] == 1  # journal C (6 records)


def test_parse_source_names_handles_malformed_json():
    assert _parse_source_names(_sources("a", "b")) == ["a", "b"]
    assert _parse_source_names("not valid json") == []
    assert _parse_source_names(None) == []
    assert _parse_source_names("") == []


def test_provenance_category_data_is_a_mutually_exclusive_partition():
    dataset = _dataset()
    counts = _provenance_category_data(dataset, streamlit_curated_ids={"r1"})
    # r1: in streamlit_curated_ids -> Streamlit Curated, even though its source looks pre-app.
    # r2: human_curated, not streamlit-curated -> Manual Curated (Pre-App).
    # r3: heuristic_candidate but EPMC-negative source -> EPMC Negatives, not DOME Registry.
    # r4: registry_confirmed -> DOME Registry.
    # r5: human_curated (malformed sources cell doesn't crash the classifier) -> Manual Curated (Pre-App).
    assert counts.to_dict() == {
        "Streamlit Curated": 1,
        "Manual Curated (Pre-App)": 2,
        "DOME Registry": 1,
        "EPMC Negatives": 1,
    }
    assert counts.sum() == len(dataset)  # every record counted exactly once


def test_bm25_score_distribution_data_excludes_unscored_and_non_pos_neg_rows():
    df = _dataset()
    df["bulk_match_score"] = [12.5, 8.0, None, 3.0, 5.0]
    scored = _bm25_score_distribution_data(df)
    # r3 has no score (excluded), r4/r5 are skipped/undeterminable (excluded).
    assert sorted(scored["label"]) == ["negative", "positive"]
    assert set(scored["bulk_match_score"]) == {12.5, 8.0}


def test_year_distribution_data_parses_float_artifact_years_and_keeps_canonical_label_names():
    pivot = _year_distribution_data(_dataset())
    # "2019.0" (r5) must parse to the same year bucket as a plain "2019" would.
    assert 2019 in pivot.index
    assert list(pivot.columns) == ["positive", "negative", "skipped", "undeterminable"]
    assert pivot.loc[2019, "skipped"] == 1  # r4
    assert pivot.loc[2019, "undeterminable"] == 1  # r5
    assert pivot.loc[2020, "positive"] == 1
    assert pivot.loc[2020, "negative"] == 1
    assert pivot.loc[2021, "negative"] == 1


def test_year_distribution_data_covers_full_default_range_and_drops_out_of_range_years():
    pivot = _year_distribution_data(_dataset())
    assert pivot.index.min() == 2000
    assert pivot.index.max() == 2026
    assert len(pivot) == 27


def test_year_distribution_data_excludes_pre_2000_years():
    df = pd.DataFrame({"record_id": ["a", "b"], "year": ["1998", "2020"], "label": ["negative", "positive"]})
    pivot = _year_distribution_data(df)
    assert 1998 not in pivot.index
    assert pivot.loc[2020, "positive"] == 1


def test_year_coverage_vs_bulk_pool_data_computes_percentage():
    dataset = pd.DataFrame({"year": ["2020", "2020", "2021"]})
    pool_years = pd.Series(["2020"] * 10 + ["2021"] * 4 + ["2022"] * 1)
    coverage = _year_coverage_vs_bulk_pool_data(dataset, pool_years)
    assert coverage.loc[2020, "curated"] == 2
    assert coverage.loc[2020, "pool"] == 10
    assert coverage.loc[2020, "coverage_pct"] == 20.0
    assert coverage.loc[2021, "coverage_pct"] == 25.0
    # 2022 appears only in the pool -- curated count fills in as 0, not dropped.
    assert coverage.loc[2022, "curated"] == 0
    assert coverage.loc[2022, "pool"] == 1
    # Default range is 2000-2026 regardless of what years actually appear in the inputs.
    assert coverage.index.min() == 2000
    assert coverage.index.max() == 2026


def _youden_joined() -> pd.DataFrame:
    # Q1 (quartile 1): both correct, scores 10/20 (avg 15). Q2: one correct, one wrong (false
    # positive: BM25 said positive, human said negative), scores 30/40 (avg 35). "All" must
    # aggregate across both quartiles (score range 10-40, avg 25).
    return pd.DataFrame(
        {
            "label": ["negative", "positive", "positive", "negative"],
            "match_classification__bm25": ["negative", "positive", "negative", "positive"],
            "quartile": [1, 1, 2, 2],
            "match_score__bm25": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_bm25_youden_performance_data_computes_per_quartile_and_all_accuracy():
    performance = _bm25_youden_performance_data(_youden_joined())
    by_quartile = performance.set_index("quartile")

    assert by_quartile.loc["Q1", "correct"] == 2
    assert by_quartile.loc["Q1", "incorrect"] == 0
    assert by_quartile.loc["Q1", "accuracy_pct"] == 100.0

    assert by_quartile.loc["Q2", "correct"] == 0
    assert by_quartile.loc["Q2", "incorrect"] == 2
    assert by_quartile.loc["Q2", "accuracy_pct"] == 0.0

    assert by_quartile.loc["All", "total"] == 4
    assert by_quartile.loc["All", "correct"] == 2
    assert by_quartile.loc["All", "accuracy_pct"] == 50.0


def test_bm25_youden_performance_data_computes_score_range_and_average_per_quartile():
    performance = _bm25_youden_performance_data(_youden_joined())
    by_quartile = performance.set_index("quartile")

    assert by_quartile.loc["Q1", "score_min"] == 10.0
    assert by_quartile.loc["Q1", "score_max"] == 20.0
    assert by_quartile.loc["Q1", "score_avg"] == 15.0

    assert by_quartile.loc["Q2", "score_min"] == 30.0
    assert by_quartile.loc["Q2", "score_max"] == 40.0
    assert by_quartile.loc["Q2", "score_avg"] == 35.0

    assert by_quartile.loc["All", "score_min"] == 10.0
    assert by_quartile.loc["All", "score_max"] == 40.0
    assert by_quartile.loc["All", "score_avg"] == 25.0


def test_bm25_youden_confusion_matrix_data_orders_positive_before_negative():
    confusion = _bm25_youden_confusion_matrix_data(_youden_joined())
    assert list(confusion.index) == ["positive", "negative"]
    assert list(confusion.columns) == ["positive", "negative"]
    # True positive: human=positive, bm25=positive -> 1 (from Q1's second row).
    assert confusion.loc["positive", "positive"] == 1
    # False negative: human=positive, bm25=negative -> 1 (from Q2's first row).
    assert confusion.loc["positive", "negative"] == 1
    # True negative: human=negative, bm25=negative -> 1 (from Q1's first row).
    assert confusion.loc["negative", "negative"] == 1
    # False positive: human=negative, bm25=positive -> 1 (from Q2's second row).
    assert confusion.loc["negative", "positive"] == 1


def test_bm25_youden_confusion_matrix_data_handles_single_class_gracefully():
    df = pd.DataFrame(
        {"label": ["negative", "negative"], "match_classification__bm25": ["negative", "negative"]}
    )
    confusion = _bm25_youden_confusion_matrix_data(df)
    assert list(confusion.index) == ["negative"]
    assert list(confusion.columns) == ["negative"]
    assert confusion.loc["negative", "negative"] == 2
