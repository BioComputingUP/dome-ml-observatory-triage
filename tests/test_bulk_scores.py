import pandas as pd

from dome_triage.curate.bulk_scores import (
    annotate_bulk_scores,
    annotate_screening,
    load_bulk_score_lookup,
    load_screening_lookup,
)


def _write_scored_csv(path):
    pd.DataFrame(
        {
            "pmcid": ["PMC1", "", "PMC3"],
            "pmid": ["111", "222", ""],
            "doi": ["", "", "10.1/x"],
            "title": ["t1", "t2", "t3"],  # not read by the lookup -- present to mimic the real file
            "match_score__bm25": [12.5, 3.1, 200.0],
            "match_classification__bm25": ["negative", "negative", "positive"],
        }
    ).to_csv(path, index=False)
    return path


def test_load_bulk_score_lookup_indexes_by_all_three_id_fields(tmp_path):
    scored_path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    lookup = load_bulk_score_lookup(scored_path)

    assert lookup["PMC1"] == (12.5, "negative")
    assert lookup["222"] == (3.1, "negative")
    assert lookup["10.1/x"] == (200.0, "positive")
    assert "" not in lookup


def test_load_bulk_score_lookup_returns_empty_when_file_missing(tmp_path):
    assert load_bulk_score_lookup(tmp_path / "does_not_exist.csv") == {}


def test_load_bulk_score_lookup_returns_empty_when_score_col_missing(tmp_path):
    path = tmp_path / "scored.csv"
    pd.DataFrame({"pmcid": ["PMC1"], "pmid": ["1"], "doi": [""]}).to_csv(path, index=False)
    assert load_bulk_score_lookup(path) == {}


def test_annotate_bulk_scores_falls_back_through_pmcid_pmid_doi(tmp_path):
    scored_path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    lookup = load_bulk_score_lookup(scored_path)

    dataset = pd.DataFrame(
        {
            "pmcid": ["PMC1", "", None],
            "pmid": ["999", "222", "999"],
            "doi": [None, None, "10.1/x"],
        }
    )
    annotated = annotate_bulk_scores(dataset, lookup)

    assert annotated["bulk_match_score"].tolist() == [12.5, 3.1, 200.0]
    assert annotated["bulk_match_classification"].tolist() == ["negative", "negative", "positive"]


def test_annotate_bulk_scores_never_mutates_input(tmp_path):
    scored_path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    lookup = load_bulk_score_lookup(scored_path)
    dataset = pd.DataFrame({"pmcid": ["PMC1"], "pmid": ["1"], "doi": [""]})

    annotate_bulk_scores(dataset, lookup)

    assert "bulk_match_score" not in dataset.columns


def test_load_screening_lookup_and_annotate(tmp_path):
    screened_path = tmp_path / "clear_negative_candidates_screened.csv"
    pd.DataFrame(
        {
            "pmcid": ["PMC1", "PMC2"],
            "pmid": ["", ""],
            "doi": ["", ""],
            "needs_screening": [True, False],
        }
    ).to_csv(screened_path, index=False)

    lookup = load_screening_lookup(screened_path)
    assert lookup == {"PMC1": True, "PMC2": False}

    dataset = pd.DataFrame({"pmcid": ["PMC1", "PMC2", "PMC3"], "pmid": ["", "", ""], "doi": ["", "", ""]})
    annotated = annotate_screening(dataset, lookup)
    assert annotated["needs_screening"].tolist() == [True, False, False]


def test_load_screening_lookup_returns_empty_when_file_missing(tmp_path):
    assert load_screening_lookup(tmp_path / "nope.csv") == {}


class _NoScanDict(dict):
    """A lookup that blows up if anything walks the *whole* mapping instead of doing O(1) gets.

    Guards the real performance contract of `annotate_bulk_scores`/`annotate_screening`, which a
    plain timing assertion couldn't pin without being flaky. In production this lookup holds ~2.07
    million entries while `dataset` holds a few thousand rows, so any full scan dominates. Notably
    `Series.map(some_dict)` *looks* like it can't do this, but pandas' `map_array` converts a plain
    dict via `Series(mapper)` -- hashing every key -- which trips this guard. That regression has
    already happened twice here; this test is what makes the third time fail loudly.
    """

    def _boom(self, *_args, **_kwargs):
        raise AssertionError(
            "the full lookup was scanned -- annotate_* must only do O(1) .get() calls, never "
            "anything that walks every entry (see bulk_scores.annotate_bulk_scores' docstring)"
        )

    __iter__ = _boom
    keys = _boom
    values = _boom
    items = _boom


def test_annotate_bulk_scores_never_scans_the_whole_lookup(tmp_path):
    scored_path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    lookup = _NoScanDict(load_bulk_score_lookup(scored_path))
    dataset = pd.DataFrame({"pmcid": ["PMC1", ""], "pmid": ["999", "222"], "doi": [None, None]})

    annotated = annotate_bulk_scores(dataset, lookup)

    assert annotated["bulk_match_score"].tolist() == [12.5, 3.1]


def test_annotate_screening_never_scans_the_whole_lookup():
    lookup = _NoScanDict({"PMC1": True})
    dataset = pd.DataFrame({"pmcid": ["PMC1", "PMC2"], "pmid": ["", ""], "doi": ["", ""]})

    annotated = annotate_screening(dataset, lookup)

    assert annotated["needs_screening"].tolist() == [True, False]
