import pandas as pd

from dome_triage.reporting.landscape_profile import (
    classification_breakdown_data,
    curated_vs_landscape_comparison_data,
    duplicate_id_audit_data,
    parse_error_resolution_data,
    resolve_latest_per_record,
)


def _events(rows):
    defaults = {"criteria_sha256": "abc", "timestamp": "2026-08-27T00:00:00"}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_resolve_latest_per_record_keeps_last_by_timestamp():
    events = _events([
        {"record_id": "r1", "classification": "parse_error", "timestamp": "2026-08-27T00:00:00"},
        {"record_id": "r1", "classification": "positive", "timestamp": "2026-08-27T01:00:00"},
        {"record_id": "r2", "classification": "negative", "timestamp": "2026-08-27T00:00:00"},
    ])
    resolved = resolve_latest_per_record(events)
    assert resolved.set_index("record_id").loc["r1", "classification"] == "positive"
    assert len(resolved) == 2


def test_classification_breakdown_counts_and_orders():
    events = _events([
        {"record_id": "r1", "classification": "positive"},
        {"record_id": "r2", "classification": "positive"},
        {"record_id": "r3", "classification": "negative"},
        {"record_id": "r4", "classification": "parse_error"},
    ])
    counts = classification_breakdown_data(events)
    assert counts.to_dict() == {"positive": 2, "negative": 1, "parse_error": 1}
    # order follows the fixed display order, not insertion order
    assert list(counts.index) == ["positive", "negative", "parse_error"]


def test_duplicate_audit_no_duplicates_is_all_zero():
    events = _events([
        {"record_id": "r1", "classification": "positive"},
        {"record_id": "r2", "classification": "negative"},
    ])
    data = duplicate_id_audit_data(events)
    assert data["n_rows"] == 2
    assert data["n_unique_ids"] == 2
    assert data["n_excess_rows"] == 0
    assert data["n_ids_repeated"] == 0


def test_duplicate_audit_classifies_retried_parse_error_correctly():
    # The exact real-world shape: one record failed once then succeeded on retry.
    events = _events([
        {"record_id": "r1", "classification": "parse_error"},
        {"record_id": "r1", "classification": "positive"},
    ])
    data = duplicate_id_audit_data(events)
    assert data["n_ids_repeated"] == 1
    assert data["n_repeats_involving_a_parse_error_retry"] == 1
    assert data["n_repeats_same_answer_both_times"] == 0
    assert data["n_repeats_genuinely_disagreeing"] == 0


def test_duplicate_audit_classifies_genuine_agreeing_duplicate():
    # Same paper appears twice in the pool (a known, pre-existing phenomenon), same answer both times.
    events = _events([
        {"record_id": "r1", "classification": "negative"},
        {"record_id": "r1", "classification": "negative"},
    ])
    data = duplicate_id_audit_data(events)
    assert data["n_repeats_involving_a_parse_error_retry"] == 0
    assert data["n_repeats_same_answer_both_times"] == 1
    assert data["n_repeats_genuinely_disagreeing"] == 0


def test_duplicate_audit_classifies_genuine_disagreeing_duplicate():
    # Same identity, no parse_error involved, but the model gave different answers -- nondeterminism.
    events = _events([
        {"record_id": "r1", "classification": "positive"},
        {"record_id": "r1", "classification": "negative"},
    ])
    data = duplicate_id_audit_data(events)
    assert data["n_repeats_involving_a_parse_error_retry"] == 0
    assert data["n_repeats_same_answer_both_times"] == 0
    assert data["n_repeats_genuinely_disagreeing"] == 1


def test_parse_error_resolution_splits_retried_from_stuck():
    events = _events([
        {"record_id": "r1", "classification": "parse_error", "timestamp": "2026-08-27T00:00:00"},
        {"record_id": "r1", "classification": "positive", "timestamp": "2026-08-27T01:00:00"},
        {"record_id": "r2", "classification": "parse_error", "timestamp": "2026-08-27T00:00:00"},
    ])
    data = parse_error_resolution_data(events)
    assert data["n_unique_records_that_ever_parse_errored"] == 2
    assert data["n_resolved_via_retry"] == 1
    assert data["n_still_unresolved"] == 1
    assert data["still_unresolved_record_ids"] == ["r2"]


def test_parse_error_resolution_all_clean_is_zero():
    events = _events([{"record_id": "r1", "classification": "positive"}])
    data = parse_error_resolution_data(events)
    assert data["n_unique_records_that_ever_parse_errored"] == 0
    assert data["n_still_unresolved"] == 0


def test_curated_vs_landscape_only_uses_trusted_label_confidence():
    canonical = pd.DataFrame([
        {"label": "positive", "label_confidence": "human_curated"},
        {"label": "negative", "label_confidence": "human_curated"},
        {"label": "negative", "label_confidence": "registry_confirmed"},
        # heuristic_candidate must NOT count -- it's the EPMC clear-negative sanity pool, never trusted.
        {"label": "negative", "label_confidence": "heuristic_candidate"},
    ])
    landscape = _events([
        {"record_id": "r1", "classification": "positive"},
        {"record_id": "r2", "classification": "negative"},
        {"record_id": "r3", "classification": "negative"},
        {"record_id": "r4", "classification": "undeterminable"},
    ])
    data = curated_vs_landscape_comparison_data(canonical, landscape)
    curated_row = data[data.source.str.contains("Curated")].iloc[0]
    assert curated_row.n_positive == 1
    assert curated_row.n_negative == 2
    assert curated_row.n_total == 3  # heuristic_candidate excluded

    landscape_row = data[data.source.str.contains("landscape")].iloc[0]
    assert landscape_row.n_positive == 1
    assert landscape_row.n_negative == 2
    assert landscape_row.n_total == 3  # undeterminable excluded from "decided"


def test_curated_vs_landscape_handles_empty_landscape_without_dividing_by_zero():
    canonical = pd.DataFrame([{"label": "positive", "label_confidence": "human_curated"}])
    landscape = _events([{"record_id": "r1", "classification": "undeterminable"}])
    data = curated_vs_landscape_comparison_data(canonical, landscape)
    landscape_row = data[data.source.str.contains("landscape")].iloc[0]
    assert landscape_row.n_total == 0
    assert pd.isna(landscape_row.positive_rate)
