import json

import pandas as pd

from dome_triage.reporting.enrichment_profile import (
    latest_ok_events,
    normalization_breakdown,
    tag_count_distribution,
    value_frequencies,
    violation_stats,
)


def _events(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "record_id": "r1", "model_tier": "flash", "parse_status": "ok",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "domain_tier1": "[]", "domain_tier2": "[]", "domain_tier3": "[]",
        "learning_paradigm": "[]", "model_family": "[]", "model_type": "[]",
        "vocab_violations": "[]",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_latest_ok_events_takes_last_per_record_and_drops_parse_errors():
    events = _events(
        [
            {"record_id": "r1", "model_type": '["old"]', "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "r1", "model_type": '["new"]', "timestamp": "2026-01-02T00:00:00+00:00"},
            {"record_id": "r2", "parse_status": "parse_error"},
        ]
    )
    ok = latest_ok_events(events, "flash")
    assert len(ok) == 1
    assert json.loads(ok.iloc[0]["model_type"]) == ["new"]


def test_tag_count_distribution_buckets_0_1_2_3plus():
    events = _events(
        [
            {"record_id": "r1", "model_type": "[]"},
            {"record_id": "r2", "model_type": '["a"]'},
            {"record_id": "r3", "model_type": '["a", "b", "c", "d"]'},
        ]
    )
    dist = tag_count_distribution(latest_ok_events(events, "flash"))
    assert dist.loc["model_type", "0"] == 1
    assert dist.loc["model_type", "1"] == 1
    assert dist.loc["model_type", "3+"] == 1


def test_value_frequencies_explodes_multi_label_lists():
    events = _events(
        [
            {"record_id": "r1", "model_family": '["deep learning", "ensemble learning"]'},
            {"record_id": "r2", "model_family": '["deep learning"]'},
        ]
    )
    freq = value_frequencies(latest_ok_events(events, "flash"), "model_family")
    assert freq["deep learning"] == 2
    assert freq["ensemble learning"] == 1


def test_normalization_breakdown_splits_canonical_vs_novel():
    seed = {"terms": [{"canonical": "XGBoost", "aliases": ["xgb"]}]}
    events = _events([{"record_id": "r1", "model_type": '["XGBoost", "NovelNet-3000"]'}])
    result = normalization_breakdown(latest_ok_events(events, "flash"), seed)
    assert result == {"n_model_type_values": 2, "n_seed_canonical": 1, "n_novel_free_text": 1}


def test_violation_stats_counts_per_field_and_affected_records():
    events = _events(
        [
            {"record_id": "r1", "vocab_violations": '["domain_tier1:unknown:Astrology", "model_family:cap_exceeded:4>3"]'},
            {"record_id": "r2", "vocab_violations": "[]"},
        ]
    )
    result = violation_stats(latest_ok_events(events, "flash"))
    assert result["n_events_with_violation"] == 1
    assert result["violations_per_field"] == {"domain_tier1": 1, "model_family": 1}


def test_broken_json_cells_are_tolerated_as_empty():
    events = _events([{"record_id": "r1", "model_type": "not json at all"}])
    dist = tag_count_distribution(latest_ok_events(events, "flash"))
    assert dist.loc["model_type", "0"] == 1
