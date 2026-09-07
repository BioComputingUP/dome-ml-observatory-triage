import pandas as pd

from dome_triage.pipeline.steps import (
    _annotate_already_curated,
    _build_existing_label_lookup,
    _lookup_youden_threshold,
)


def test_lookup_youden_threshold_returns_none_when_report_missing(tmp_path):
    result = _lookup_youden_threshold(tmp_path / "no_such_report.csv", "bm25", "positive_lexicon_only")
    assert result is None


def test_lookup_youden_threshold_returns_none_when_no_matching_row(tmp_path):
    report_path = tmp_path / "scoring_bakeoff_report.csv"
    pd.DataFrame(
        [{"scorer": "bm25", "condition": "positive_lexicon_only", "threshold_youden": 1.5}]
    ).to_csv(report_path, index=False)

    # scorer exists but condition doesn't
    result = _lookup_youden_threshold(report_path, "bm25", "positive_plus_exclusionary_lexicon")
    assert result is None


def test_lookup_youden_threshold_returns_the_matching_value(tmp_path):
    report_path = tmp_path / "scoring_bakeoff_report.csv"
    pd.DataFrame(
        [
            {"scorer": "bm25", "condition": "positive_lexicon_only", "threshold_youden": 1.5},
            {"scorer": "tfidf-cosine", "condition": "positive_plus_exclusionary_lexicon", "threshold_youden": 0.0494},
        ]
    ).to_csv(report_path, index=False)

    result = _lookup_youden_threshold(report_path, "tfidf-cosine", "positive_plus_exclusionary_lexicon")
    assert result == 0.0494


def test_build_existing_label_lookup_indexes_by_all_three_id_fields():
    canonical_df = pd.DataFrame(
        {
            "pmcid": ["PMC1", None, ""],
            "pmid": ["111", "222", None],
            "doi": [None, None, "10.1/x"],
            "label": ["positive", "negative", "positive"],
            "label_confidence": ["human_curated", "registry_confirmed", "heuristic_candidate"],
        }
    )
    lookup = _build_existing_label_lookup(canonical_df)

    assert lookup["PMC1"] == ("positive", "human_curated")
    assert lookup["111"] == ("positive", "human_curated")
    assert lookup["222"] == ("negative", "registry_confirmed")
    assert lookup["10.1/x"] == ("positive", "heuristic_candidate")
    assert "" not in lookup  # blank ids never indexed


def test_annotate_already_curated_marks_matches_and_carries_the_prior_label():
    candidates = pd.DataFrame(
        {
            "pmcid": ["PMC1", "PMC999", None],
            "pmid": ["111", "999", "333"],
            "doi": [None, None, None],
        }
    )
    lookup = {"PMC1": ("positive", "human_curated"), "333": ("negative", "heuristic_candidate")}

    annotated = _annotate_already_curated(candidates, lookup)

    assert annotated["already_curated"].tolist() == [True, False, True]
    assert annotated["existing_label"].tolist() == ["positive", "", "negative"]
    assert annotated["existing_label_confidence"].tolist() == ["human_curated", "", "heuristic_candidate"]


def test_annotate_already_curated_falls_back_through_pmcid_pmid_doi_in_order():
    # A record with no pmcid match but a pmid match should still be found.
    candidates = pd.DataFrame({"pmcid": [None], "pmid": ["555"], "doi": [None]})
    lookup = {"555": ("positive", "registry_confirmed")}

    annotated = _annotate_already_curated(candidates, lookup)

    assert annotated["already_curated"].iloc[0]
    assert annotated["existing_label"].iloc[0] == "positive"
