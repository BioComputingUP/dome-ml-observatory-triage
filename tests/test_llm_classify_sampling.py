import pandas as pd
import pytest

from dome_triage.llm_classify.sampling import (
    draw_blind_sample,
    select_bulk_pool_excluding_curated,
    select_full_population,
    strip_for_api,
)


def _make_dataset(n_positive: int, n_negative: int, label_confidence: str = "human_curated") -> pd.DataFrame:
    rows = []
    for i in range(n_positive):
        rows.append(
            {
                "record_id": f"pos_{i}",
                "label": "positive",
                "label_confidence": label_confidence,
                "title": f"pos title {i}",
                "abstract": f"pos abstract {i}",
                "journal": "J",
                "year": "2020",
            }
        )
    for i in range(n_negative):
        rows.append(
            {
                "record_id": f"neg_{i}",
                "label": "negative",
                "label_confidence": label_confidence,
                "title": f"neg title {i}",
                "abstract": f"neg abstract {i}",
                "journal": "J",
                "year": "2020",
            }
        )
    return pd.DataFrame(rows)


def test_draw_blind_sample_returns_exact_requested_split():
    dataset = _make_dataset(50, 50)
    sample = draw_blind_sample(dataset, n_positive=10, n_negative=15, random_state=1)
    assert (sample["label"] == "positive").sum() == 10
    assert (sample["label"] == "negative").sum() == 15
    assert len(sample) == 25


def test_draw_blind_sample_is_deterministic_for_a_fixed_seed():
    dataset = _make_dataset(50, 50)
    sample_a = draw_blind_sample(dataset, n_positive=10, n_negative=10, random_state=7)
    sample_b = draw_blind_sample(dataset, n_positive=10, n_negative=10, random_state=7)
    assert sorted(sample_a["record_id"]) == sorted(sample_b["record_id"])


def test_draw_blind_sample_excludes_untrusted_label_confidence():
    dataset = _make_dataset(10, 10, label_confidence="heuristic_candidate")
    with pytest.raises(ValueError):
        draw_blind_sample(dataset, n_positive=5, n_negative=5, random_state=1)


def test_draw_blind_sample_excludes_given_ids():
    dataset = _make_dataset(20, 20)
    exclude_ids = {f"pos_{i}" for i in range(15)}  # leaves exactly 5 eligible positives
    sample = draw_blind_sample(dataset, n_positive=5, n_negative=5, random_state=1, exclude_ids=exclude_ids)
    assert not (set(sample["record_id"]) & exclude_ids)


def test_draw_blind_sample_raises_when_pool_too_small():
    dataset = _make_dataset(3, 50)
    with pytest.raises(ValueError):
        draw_blind_sample(dataset, n_positive=10, n_negative=10, random_state=1)


def test_draw_blind_sample_retains_label_column_for_later_scoring():
    dataset = _make_dataset(20, 20)
    sample = draw_blind_sample(dataset, n_positive=5, n_negative=5, random_state=1)
    assert "label" in sample.columns


def test_select_full_population_defaults_to_trusted_only():
    trusted = _make_dataset(5, 5, label_confidence="human_curated")
    candidates = _make_dataset(3, 3, label_confidence="heuristic_candidate")
    candidates["record_id"] = candidates["record_id"] + "_cand"
    dataset = pd.concat([trusted, candidates], ignore_index=True)
    pool = select_full_population(dataset)
    assert len(pool) == 10
    assert set(pool["label_confidence"]) == {"human_curated"}


def test_select_full_population_include_candidates_adds_heuristic_candidate_rows():
    trusted = _make_dataset(5, 5, label_confidence="human_curated")
    candidates = _make_dataset(3, 3, label_confidence="heuristic_candidate")
    candidates["record_id"] = candidates["record_id"] + "_cand"
    dataset = pd.concat([trusted, candidates], ignore_index=True)
    pool = select_full_population(dataset, include_candidates=True)
    assert len(pool) == 16
    assert set(pool["label_confidence"]) == {"human_curated", "heuristic_candidate"}


def test_select_full_population_excludes_held_out_ids():
    dataset = _make_dataset(10, 10)
    held_out_ids = {f"pos_{i}" for i in range(4)}
    pool = select_full_population(dataset, held_out_ids=held_out_ids)
    assert not (set(pool["record_id"]) & held_out_ids)
    assert len(pool) == 16


def test_select_full_population_excludes_held_out_ids_from_candidates_too():
    trusted = _make_dataset(5, 5, label_confidence="human_curated")
    candidates = _make_dataset(3, 3, label_confidence="heuristic_candidate")
    candidates["record_id"] = candidates["record_id"] + "_cand"
    dataset = pd.concat([trusted, candidates], ignore_index=True)
    held_out_ids = {"neg_0_cand"}
    pool = select_full_population(dataset, include_candidates=True, held_out_ids=held_out_ids)
    assert "neg_0_cand" not in set(pool["record_id"])
    assert len(pool) == 15


def test_select_full_population_excludes_non_positive_negative_labels():
    dataset = _make_dataset(5, 5)
    dataset.loc[0, "label"] = "undeterminable"
    pool = select_full_population(dataset)
    assert len(pool) == 9


def _bulk_pool(rows: list[dict]) -> pd.DataFrame:
    defaults = {"pmcid": None, "pmid": None, "doi": None, "title": "T", "abstract": "A", "journal": "J", "year": "2020"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_select_bulk_pool_excluding_curated_drops_already_curated_by_pmid():
    pool = _bulk_pool([{"pmid": "111"}, {"pmid": "222"}])
    record_ids = pd.Series(["rid_a", "rid_b"])
    result = select_bulk_pool_excluding_curated(pool, existing_ids={"111"}, record_ids=record_ids)
    assert list(result["pmid"]) == ["222"]


def test_select_bulk_pool_excluding_curated_checks_pmcid_pmid_and_doi():
    pool = _bulk_pool(
        [
            {"pmcid": "PMC1"}, {"pmid": "222"}, {"doi": "10.1/x"}, {"pmid": "999"},
        ]
    )
    record_ids = pd.Series([f"rid_{i}" for i in range(4)])
    result = select_bulk_pool_excluding_curated(
        pool, existing_ids={"PMC1", "222", "10.1/x"}, record_ids=record_ids
    )
    assert list(result["pmid"]) == ["999"]


def test_select_bulk_pool_excluding_curated_stamps_the_precomputed_record_id():
    pool = _bulk_pool([{"pmid": "111"}])
    record_ids = pd.Series(["exact_record_id_value"])
    result = select_bulk_pool_excluding_curated(pool, existing_ids=set(), record_ids=record_ids)
    assert result.iloc[0]["record_id"] == "exact_record_id_value"


def test_select_bulk_pool_excluding_curated_keeps_everything_when_nothing_curated_yet():
    pool = _bulk_pool([{"pmid": "111"}, {"pmid": "222"}])
    record_ids = pd.Series(["rid_a", "rid_b"])
    result = select_bulk_pool_excluding_curated(pool, existing_ids=set(), record_ids=record_ids)
    assert len(result) == 2


def test_strip_for_api_returns_only_the_four_allowed_fields():
    record = pd.Series(
        {
            "record_id": "r1",
            "title": "T",
            "abstract": "A",
            "journal": "J",
            "year": "2020",
            "label": "positive",
            "notes": "secret",
            "mesh_headings": '["Machine Learning"]',
        }
    )
    stripped = strip_for_api(record)
    assert set(stripped.keys()) == {"title", "abstract", "journal", "year"}
    assert stripped["title"] == "T"


def test_strip_for_api_works_on_plain_dict_too():
    record = {"title": "T", "abstract": "A", "journal": "J", "year": "2020", "label": "positive"}
    stripped = strip_for_api(record)
    assert set(stripped.keys()) == {"title", "abstract", "journal", "year"}
