import pandas as pd
import pytest

from dome_triage.reporting.agreement import (
    _confusion_matrix_data,
    _disagreement_breakdown_data,
    _flash_vs_pro_comparison_data,
    build_candidate_pool_joined,
    build_original_sample_vs_final_joined,
    build_population_run_joined,
    build_post_review_final_labels,
    compute_agreement,
    compute_candidate_pool_confirmation,
    compute_fallback_accuracy,
    compute_post_review_reversal_breakdown,
    resolve_prior_criteria_hash,
)


def _joined(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["label", "classification"]).assign(
        record_id=lambda df: [f"r{i}" for i in range(len(df))]
    )


def test_compute_agreement_perfect_agreement_gives_kappa_one():
    joined = _joined([("positive", "positive"), ("negative", "negative")] * 5)
    result = compute_agreement(joined, tier="flash")
    assert result["kappa"] == 1.0
    assert result["accuracy_among_decided"] == 1.0
    assert result["n_parse_error"] == 0
    assert result["n_undetermined"] == 0


def test_compute_agreement_excludes_parse_error_from_kappa_and_reports_count():
    joined = _joined(
        [("positive", "positive"), ("negative", "negative"), ("positive", "parse_error"), ("negative", "parse_error")]
    )
    result = compute_agreement(joined, tier="flash")
    assert result["n_parse_error"] == 2
    assert result["n_scored"] == 2
    assert result["kappa"] == 1.0  # the two parse_error rows must not drag this down


def test_compute_agreement_undetermined_rate_excludes_parse_error_from_denominator():
    joined = _joined(
        [("positive", "undeterminable"), ("negative", "negative"), ("positive", "parse_error")]
    )
    result = compute_agreement(joined, tier="flash")
    assert result["n_scored"] == 2  # parse_error excluded
    assert result["n_undetermined"] == 1
    assert result["undetermined_rate"] == 0.5


def test_confusion_matrix_data_is_2x3_never_assumed_square():
    joined = _joined([("positive", "positive"), ("positive", "undeterminable"), ("negative", "negative")])
    matrix = _confusion_matrix_data(joined)
    assert list(matrix.index) == ["positive", "negative"]
    assert "undeterminable" in matrix.columns
    assert matrix.loc["positive", "undeterminable"] == 1


def test_confusion_matrix_data_excludes_parse_error_rows():
    joined = _joined([("positive", "parse_error"), ("negative", "negative")])
    matrix = _confusion_matrix_data(joined)
    assert matrix.values.sum() == 1


def test_disagreement_breakdown_excludes_the_diagonal():
    joined = _joined([("positive", "positive"), ("positive", "negative"), ("negative", "negative")])
    data = _disagreement_breakdown_data(joined)
    assert len(data) == 1
    assert data.iloc[0]["human_label"] == "positive"
    assert data.iloc[0]["llm_classification"] == "negative"


def test_flash_vs_pro_comparison_buckets_agreement_correctly():
    flash = pd.DataFrame(
        [
            {"record_id": "r1", "label": "positive", "classification": "positive"},
            {"record_id": "r2", "label": "positive", "classification": "negative"},
            {"record_id": "r4", "label": "negative", "classification": "positive"},
        ]
    )
    pro = pd.DataFrame(
        [
            {"record_id": "r1", "classification": "positive"},  # both agree, correct
            {"record_id": "r2", "classification": "positive"},  # disagree, pro correct
            {"record_id": "r4", "classification": "undeterminable"},  # disagree, neither correct
        ]
    )
    data = _flash_vs_pro_comparison_data(flash, pro)
    assert data.loc["Both Agree, Match Human", "count"] == 1
    assert data.loc["Disagree, Pro Matches Human", "count"] == 1
    assert data.loc["Disagree, Neither Matches Human", "count"] == 1


def test_flash_vs_pro_comparison_excludes_parse_error_rows():
    flash = pd.DataFrame([{"record_id": "r1", "label": "positive", "classification": "parse_error"}])
    pro = pd.DataFrame([{"record_id": "r1", "classification": "positive"}])
    data = _flash_vs_pro_comparison_data(flash, pro)
    assert data["count"].sum() == 0


def test_compute_fallback_accuracy_scores_against_true_label():
    fallback_events = pd.DataFrame(
        [
            {"record_id": "r1", "classification": "positive"},
            {"record_id": "r2", "classification": "negative"},
            {"record_id": "r3", "classification": "parse_error"},
        ]
    )
    sample_df = pd.DataFrame(
        [
            {"record_id": "r1", "label": "positive"},
            {"record_id": "r2", "label": "positive"},
            {"record_id": "r3", "label": "negative"},
        ]
    )
    result = compute_fallback_accuracy(fallback_events, sample_df, mode="forced_guess")
    assert result["n"] == 2  # r3's parse_error excluded
    assert result["accuracy"] == 0.5


def _sample(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["record_id", "label"])


def test_build_post_review_final_labels_uses_latest_decision_for_reviewed_records():
    sample = _sample([("r1", "positive"), ("r2", "negative"), ("r3", "positive")])
    resolution_events = pd.DataFrame(
        [{"record_id": "r1", "decision": "negative", "timestamp": "2026-01-01T00:00:00+00:00"}]
    )
    result = build_post_review_final_labels(sample, resolution_events)
    r1 = result[result["record_id"] == "r1"].iloc[0]
    r2 = result[result["record_id"] == "r2"].iloc[0]
    assert r1["original_label"] == "positive"
    assert r1["final_label"] == "negative"
    assert r2["original_label"] == r2["final_label"] == "negative"  # never reviewed -- unchanged


def test_build_post_review_final_labels_uses_the_latest_event_when_a_record_was_redecided():
    sample = _sample([("r1", "positive")])
    resolution_events = pd.DataFrame(
        [
            {"record_id": "r1", "decision": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "r1", "decision": "positive", "timestamp": "2026-01-02T00:00:00+00:00"},
        ]
    )
    result = build_post_review_final_labels(sample, resolution_events)
    assert result.iloc[0]["final_label"] == "positive"  # the later event wins


def test_build_post_review_final_labels_empty_resolution_events_leaves_everything_unchanged():
    sample = _sample([("r1", "positive"), ("r2", "negative")])
    result = build_post_review_final_labels(sample, pd.DataFrame())
    assert (result["original_label"] == result["final_label"]).all()


def test_compute_post_review_reversal_breakdown_buckets_correctly():
    # r1: flash said negative (disagreed with original "positive"); final decision agrees with
    #     flash -> Reversed to DeepSeek.
    # r2: flash said negative (disagreed with original "positive"); final decision upholds the
    #     original "positive" -> Upheld Original.
    # r3: flash agrees with the original label -> not in the disagreement queue at all, excluded.
    sample = _sample([("r1", "positive"), ("r2", "positive"), ("r3", "positive")])
    resolution_events = pd.DataFrame(
        [
            {"record_id": "r1", "decision": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "r2", "decision": "positive", "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
    )
    llm_events = pd.DataFrame(
        [
            {"record_id": "r1", "model_tier": "flash", "mode": "primary", "classification": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "r2", "model_tier": "flash", "mode": "primary", "classification": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "r3", "model_tier": "flash", "mode": "primary", "classification": "positive", "timestamp": "2026-01-01T00:00:00+00:00"},
        ]
    )
    data = compute_post_review_reversal_breakdown(sample, resolution_events, llm_events, tier="flash")
    assert data.loc["Reversed to DeepSeek", "count"] == 1
    assert data.loc["Upheld Original", "count"] == 1
    assert data["count"].sum() == 2  # r3 correctly excluded -- it was never a disagreement


def _population_dataset(rows: list[dict]) -> pd.DataFrame:
    defaults = {"label_confidence": "human_curated", "label": "positive", "sources": ""}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _population_llm_events(rows: list[dict]) -> pd.DataFrame:
    defaults = {"mode": "primary", "model_tier": "flash", "classification": "positive",
                "criteria_sha256": "new_hash", "timestamp": "2026-01-01T00:00:00+00:00"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_build_population_run_joined_filters_by_criteria_hash():
    dataset = _population_dataset([{"record_id": "r1", "label": "positive"}])
    events = _population_llm_events(
        [
            {"record_id": "r1", "classification": "negative", "criteria_sha256": "old_hash"},
            {"record_id": "r1", "classification": "positive", "criteria_sha256": "new_hash"},
        ]
    )
    joined = build_population_run_joined(dataset, events, tier="flash", criteria_hash="new_hash")
    assert list(joined["classification"]) == ["positive"]


def test_build_population_run_joined_excludes_untrusted_label_confidence_by_default():
    dataset = _population_dataset(
        [{"record_id": "r1", "label_confidence": "heuristic_candidate", "label": "negative"}]
    )
    events = _population_llm_events([{"record_id": "r1", "classification": "negative"}])
    joined = build_population_run_joined(dataset, events, tier="flash", criteria_hash="new_hash")
    assert joined.empty


def test_build_population_run_joined_includes_heuristic_candidate_when_requested():
    dataset = _population_dataset(
        [{"record_id": "r1", "label_confidence": "heuristic_candidate", "label": "negative"}]
    )
    events = _population_llm_events([{"record_id": "r1", "classification": "negative"}])
    joined = build_population_run_joined(
        dataset, events, tier="flash", criteria_hash="new_hash", label_confidences=("heuristic_candidate",)
    )
    assert list(joined["record_id"]) == ["r1"]


def test_build_population_run_joined_uses_latest_event_per_record():
    dataset = _population_dataset([{"record_id": "r1", "label": "positive"}])
    events = _population_llm_events(
        [
            {"record_id": "r1", "classification": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "r1", "classification": "positive", "timestamp": "2026-01-02T00:00:00+00:00"},
        ]
    )
    joined = build_population_run_joined(dataset, events, tier="flash", criteria_hash="new_hash")
    assert list(joined["classification"]) == ["positive"]


def test_build_candidate_pool_joined_tags_source_batch_from_sources_json():
    dataset = _population_dataset(
        [
            {"record_id": "r1", "label_confidence": "heuristic_candidate", "label": "negative",
             "sources": '[{"source_name": "clear_negative_sampler_strong"}]'},
            {"record_id": "r2", "label_confidence": "heuristic_candidate", "label": "negative",
             "sources": '[{"source_name": "clear_negative_sampler_strong_filtered_v2"}]'},
        ]
    )
    events = _population_llm_events(
        [
            {"record_id": "r1", "classification": "negative"},
            {"record_id": "r2", "classification": "negative"},
        ]
    )
    joined = build_candidate_pool_joined(dataset, events, tier="flash", criteria_hash="new_hash")
    batches = dict(zip(joined["record_id"], joined["source_batch"]))
    assert batches["r1"] == "Step 14 (original)"
    assert batches["r2"] == "Step 19d (filtered_v2)"


def test_compute_candidate_pool_confirmation_rates_per_batch():
    candidate_joined = pd.DataFrame(
        [
            {"record_id": "r1", "label": "negative", "classification": "negative", "source_batch": "A"},
            {"record_id": "r2", "label": "negative", "classification": "negative", "source_batch": "A"},
            {"record_id": "r3", "label": "negative", "classification": "positive", "source_batch": "A"},
            {"record_id": "r4", "label": "negative", "classification": "negative", "source_batch": "B"},
        ]
    )
    result = compute_candidate_pool_confirmation(candidate_joined)
    assert result.loc["A", "negative"] == 2
    assert result.loc["A", "positive"] == 1
    assert result.loc["A", "n_total"] == 3
    assert result.loc["A", "negative_rate"] == pytest.approx(2 / 3)
    assert result.loc["B", "negative_rate"] == 1.0


def test_resolve_prior_criteria_hash_finds_the_non_current_hash():
    events = _population_llm_events(
        [
            {"record_id": "r1", "classification": "positive", "criteria_sha256": "old_hash"},
            {"record_id": "r99", "classification": "negative", "criteria_sha256": "new_hash"},
        ]
    )
    result = resolve_prior_criteria_hash(events, {"r1"}, tier="flash", current_hash="new_hash")
    assert result == "old_hash"


def test_resolve_prior_criteria_hash_raises_when_none_found():
    events = _population_llm_events([{"record_id": "r1", "classification": "positive", "criteria_sha256": "new_hash"}])
    with pytest.raises(ValueError):
        resolve_prior_criteria_hash(events, {"r1"}, tier="flash", current_hash="new_hash")


def test_resolve_prior_criteria_hash_raises_when_ambiguous():
    events = _population_llm_events(
        [
            {"record_id": "r1", "classification": "positive", "criteria_sha256": "hash_a"},
            {"record_id": "r1", "classification": "negative", "criteria_sha256": "hash_b"},
        ]
    )
    with pytest.raises(ValueError):
        resolve_prior_criteria_hash(events, {"r1"}, tier="flash", current_hash="new_hash")


def test_build_original_sample_vs_final_joined_uses_final_label_not_original():
    sample = _sample([("r1", "positive")])
    resolution_events = pd.DataFrame(
        [{"record_id": "r1", "decision": "negative", "timestamp": "2026-01-01T00:00:00+00:00"}]
    )
    events = _population_llm_events([{"record_id": "r1", "classification": "negative", "criteria_sha256": "old_hash"}])
    joined = build_original_sample_vs_final_joined(sample, resolution_events, events, tier="flash", criteria_hash="old_hash")
    assert joined.iloc[0]["label"] == "negative"  # final_label, not the stale original "positive"
    assert joined.iloc[0]["classification"] == "negative"
