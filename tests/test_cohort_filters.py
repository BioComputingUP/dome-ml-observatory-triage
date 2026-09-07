import json

import pandas as pd

from dome_triage.curate.cohort_filters import (
    annotate_review_term_match,
    build_disagreement_queue,
    build_original_cohort,
    load_review_term_list,
    sample_cohort,
)


def _sources(*entries: tuple[str, str]) -> str:
    """`entries` is (source_name, source_label_confidence) pairs."""
    return json.dumps([{"source_name": name, "source_label_confidence": conf} for name, conf in entries])


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "r1", "journal": "J1", "year": "2018", "label": "positive",
                "label_confidence": "human_curated",
                "sources": _sources(("dome_top_curate_positive", "human_curated")),
            },
            {
                "record_id": "r2", "journal": "J2", "year": "2019", "label": "negative",
                "label_confidence": "human_curated",
                "sources": _sources(("copilot_1012_negative", "human_curated")),
            },
            {
                # Combines a human_curated source with a registry_confirmed source -- must be
                # excluded even though its overall label_confidence resolved to human_curated
                # (merge_label picks the strongest tier across sources).
                "record_id": "r3", "journal": "J3", "year": "2020", "label": "positive",
                "label_confidence": "human_curated",
                "sources": _sources(
                    ("dome_top_curate_positive", "human_curated"),
                    ("dome_registry_231_gold", "registry_confirmed"),
                ),
            },
            {
                "record_id": "r4", "journal": "J4", "year": "2021", "label": "positive",
                "label_confidence": "registry_confirmed",
                "sources": _sources(("dome_registry_222_gold", "registry_confirmed")),
            },
            {
                "record_id": "r5", "journal": "J5", "year": "2022", "label": "skipped",
                "label_confidence": "human_curated",
                "sources": _sources(("dome_top_curate_skipped", "human_curated")),
            },
            {
                # Otherwise eligible, but already reviewed via the main Curate app.
                "record_id": "r6", "journal": "J6", "year": "2023", "label": "positive",
                "label_confidence": "human_curated",
                "sources": _sources(("dome_top_curate_positive", "human_curated")),
            },
        ]
    )


def test_build_original_cohort_applies_all_four_filters(tmp_path):
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame({"record_id": ["r6"]}).to_csv(events_path, index=False)

    cohort = build_original_cohort(_dataset(), events_path)

    # r1, r2 pass every filter. r3 excluded (registry source despite human_curated confidence).
    # r4 excluded (not human_curated). r5 excluded (skipped, not pos/neg). r6 excluded (already
    # reviewed via the main app's curation_events.csv).
    assert sorted(cohort["record_id"]) == ["r1", "r2"]


def test_build_original_cohort_handles_missing_events_file(tmp_path):
    # No events file -- nothing is excluded as "already reviewed", so r6 (otherwise eligible)
    # is included this time, unlike the test above where it's in curation_events.csv.
    cohort = build_original_cohort(_dataset(), tmp_path / "does_not_exist.csv")
    assert sorted(cohort["record_id"]) == ["r1", "r2", "r6"]


def test_sample_cohort_returns_whole_cohort_when_smaller_than_sample_size(tmp_path):
    cohort = build_original_cohort(_dataset(), tmp_path / "no_events.csv")
    sampled, report = sample_cohort(cohort, sample_size=250, random_state=42)
    assert sorted(sampled["record_id"]) == sorted(cohort["record_id"])
    assert report.empty


def test_sample_cohort_respects_exclude_ids(tmp_path):
    cohort = build_original_cohort(_dataset(), tmp_path / "no_events.csv")
    sampled, _report = sample_cohort(cohort, sample_size=250, random_state=42, exclude_ids={"r1"})
    assert "r1" not in set(sampled["record_id"])
    assert "r2" in set(sampled["record_id"])


def test_sample_cohort_stratifies_a_larger_pool_to_approximately_the_target_size():
    # 200 synthetic records across 10 journals x 4 years -- large enough that stratified_sample's
    # per-stratum cap actually engages (see cohort_filters.py's docstring for the approximation).
    rows = []
    for i in range(200):
        rows.append(
            {
                "record_id": f"rec{i}",
                "journal": f"J{i % 10}",
                "year": str(2010 + (i % 4)),
                "label": "positive" if i % 2 == 0 else "negative",
                "label_confidence": "human_curated",
            }
        )
    cohort = pd.DataFrame(rows)

    sampled, report = sample_cohort(cohort, sample_size=50, random_state=42)

    assert len(sampled) <= len(cohort)
    assert 30 <= len(sampled) <= 70  # approximate -- exact size isn't guaranteed, see docstring
    assert not report.empty
    assert sampled["record_id"].is_unique


def test_sample_cohort_is_deterministic_for_a_fixed_seed():
    rows = [
        {
            "record_id": f"rec{i}", "journal": f"J{i % 10}", "year": str(2010 + (i % 4)),
            "label": "positive", "label_confidence": "human_curated",
        }
        for i in range(200)
    ]
    cohort = pd.DataFrame(rows)
    sampled_a, _ = sample_cohort(cohort, sample_size=50, random_state=42)
    sampled_b, _ = sample_cohort(cohort, sample_size=50, random_state=42)
    assert sorted(sampled_a["record_id"]) == sorted(sampled_b["record_id"])


def _write_exclusionary_lexicon(path):
    pd.DataFrame(
        [
            {"term": "forest", "notes": ""},  # general exclusionary term, NOT a review term
            {"term": "systematic review", "notes": "non_methods_pubtype"},
            {"term": "meta-analysis", "notes": "non_methods_pubtype"},
            {"term": "editorial", "notes": "non_methods_pubtype"},
        ]
    ).to_csv(path, index=False)
    return path


def test_load_review_term_list_filters_to_non_methods_pubtype_only(tmp_path):
    path = _write_exclusionary_lexicon(tmp_path / "exclusionary.csv")
    terms = load_review_term_list(path)
    assert sorted(terms) == ["editorial", "meta-analysis", "systematic review"]
    assert "forest" not in terms  # general exclusionary term, not a review/non-methods signal


def test_load_review_term_list_returns_empty_list_when_file_missing(tmp_path):
    assert load_review_term_list(tmp_path / "does_not_exist.csv") == []


def test_annotate_review_term_match_is_case_insensitive_substring_match():
    df = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "title": ["A Systematic Review of X", "A Novel Method for Y", "Editorial: on Z"],
            "abstract": ["We review the literature.", "We propose a new model.", "Commentary."],
        }
    )
    result = annotate_review_term_match(df, ["systematic review", "editorial"])
    assert bool(result.loc[result["record_id"] == "a", "matches_review_term"].iloc[0]) is True
    assert bool(result.loc[result["record_id"] == "b", "matches_review_term"].iloc[0]) is False
    assert bool(result.loc[result["record_id"] == "c", "matches_review_term"].iloc[0]) is True


def test_annotate_review_term_match_all_false_when_terms_empty():
    df = pd.DataFrame({"title": ["A Systematic Review"], "abstract": ["review text"]})
    result = annotate_review_term_match(df, [])
    assert result["matches_review_term"].tolist() == [False]


def test_annotate_review_term_match_handles_missing_title_or_abstract():
    df = pd.DataFrame({"title": [None], "abstract": ["a systematic review of things"]})
    result = annotate_review_term_match(df, ["systematic review"])
    assert bool(result["matches_review_term"].iloc[0]) is True


def _disagreement_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"record_id": "r1", "label": "positive", "label_confidence": "human_curated"},
            {"record_id": "r2", "label": "negative", "label_confidence": "human_curated"},
            {"record_id": "r3", "label": "positive", "label_confidence": "human_curated"},
            {"record_id": "r4", "label": "positive", "label_confidence": "heuristic_candidate"},  # untrusted
        ]
    )


def _llm_events(rows: list[dict]) -> pd.DataFrame:
    defaults = {"mode": "primary", "timestamp": "2026-01-01T00:00:00+00:00"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_build_disagreement_queue_keeps_only_records_where_llm_differs_from_human():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [
            {"record_id": "r1", "model_tier": "flash", "classification": "positive", "rationale": "agrees"},
            {"record_id": "r2", "model_tier": "flash", "classification": "positive", "rationale": "disagrees"},
        ]
    )
    result = build_disagreement_queue(dataset, events)
    assert list(result["record_id"]) == ["r2"]
    assert result.iloc[0]["llm_classification"] == "positive"
    assert result.iloc[0]["llm_tier"] == "flash"


def test_build_disagreement_queue_undeterminable_against_human_decision_counts_as_disagreement():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [{"record_id": "r3", "model_tier": "pro", "classification": "undeterminable", "rationale": "unclear"}]
    )
    result = build_disagreement_queue(dataset, events)
    assert list(result["record_id"]) == ["r3"]


def test_build_disagreement_queue_excludes_untrusted_label_confidence():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [{"record_id": "r4", "model_tier": "flash", "classification": "negative", "rationale": "disagrees"}]
    )
    result = build_disagreement_queue(dataset, events)
    assert result.empty


def test_build_disagreement_queue_excludes_parse_error_rows():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [{"record_id": "r1", "model_tier": "flash", "classification": "parse_error", "rationale": "x"}]
    )
    result = build_disagreement_queue(dataset, events)
    assert result.empty


def test_build_disagreement_queue_appears_once_per_disagreeing_tier():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [
            {"record_id": "r1", "model_tier": "flash", "classification": "negative", "rationale": "x"},
            {"record_id": "r1", "model_tier": "pro", "classification": "negative", "rationale": "y"},
        ]
    )
    result = build_disagreement_queue(dataset, events)
    assert sorted(result["llm_tier"]) == ["flash", "pro"]


def test_build_disagreement_queue_tier_filter_restricts_to_one_tier():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [
            {"record_id": "r1", "model_tier": "flash", "classification": "negative", "rationale": "x"},
            {"record_id": "r1", "model_tier": "pro", "classification": "negative", "rationale": "y"},
        ]
    )
    result = build_disagreement_queue(dataset, events, tier="flash")
    assert list(result["llm_tier"]) == ["flash"]


def test_build_disagreement_queue_only_uses_the_latest_event_per_record_and_tier():
    dataset = _disagreement_dataset()
    events = pd.DataFrame(
        [
            {
                "record_id": "r1", "model_tier": "flash", "classification": "negative", "rationale": "old",
                "mode": "primary", "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "record_id": "r1", "model_tier": "flash", "classification": "positive", "rationale": "new",
                "mode": "primary", "timestamp": "2026-01-02T00:00:00+00:00",
            },
        ]
    )
    result = build_disagreement_queue(dataset, events)
    assert result.empty  # the latest event (positive) agrees with the human label


def test_build_disagreement_queue_empty_events_returns_empty_frame_with_expected_columns():
    dataset = _disagreement_dataset()
    result = build_disagreement_queue(dataset, pd.DataFrame())
    assert result.empty
    assert "llm_classification" in result.columns


def test_build_disagreement_queue_exclude_ids_drops_already_resolved_records():
    # Real, confirmed gap this covers: a record whose Cross Curate Resolve final decision UPHELD
    # the original label still has label != llm_classification by construction, and would
    # otherwise resurface here forever as if it were a brand-new disagreement.
    dataset = _disagreement_dataset()
    events = _llm_events(
        [
            {"record_id": "r1", "model_tier": "flash", "classification": "positive", "rationale": "agrees"},
            {"record_id": "r2", "model_tier": "flash", "classification": "positive", "rationale": "disagrees"},
        ]
    )
    result = build_disagreement_queue(dataset, events, exclude_ids={"r2"})
    assert result.empty


def test_build_disagreement_queue_exclude_ids_leaves_other_disagreements_untouched():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [{"record_id": "r2", "model_tier": "flash", "classification": "positive", "rationale": "disagrees"}]
    )
    result = build_disagreement_queue(dataset, events, exclude_ids={"some_other_record"})
    assert list(result["record_id"]) == ["r2"]


def test_build_disagreement_queue_none_exclude_ids_is_a_no_op():
    dataset = _disagreement_dataset()
    events = _llm_events(
        [{"record_id": "r2", "model_tier": "flash", "classification": "positive", "rationale": "disagrees"}]
    )
    result = build_disagreement_queue(dataset, events, exclude_ids=None)
    assert list(result["record_id"]) == ["r2"]
