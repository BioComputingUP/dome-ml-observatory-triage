import pandas as pd
import pytest

from dome_triage.llm_classify.cost_estimator import project_cost


def _calibration_log(rows: list[dict]) -> pd.DataFrame:
    columns = ["batch_id", "tier", "mode", "record_id", "prompt_tokens", "completion_tokens", "total_tokens", "timestamp"]
    return pd.DataFrame(rows, columns=columns)


def test_project_cost_scopes_to_the_most_recent_batch_only():
    # Real incident (2026-08-21): an older, shorter-prompt batch (8 calls) and a newer,
    # longer-prompt batch (10 calls, after a CRITERIA.md edit) both exist for flash+primary.
    # observed_usd_spent was read off the dashboard for the NEW batch only -- it must not be
    # diluted across the old batch's rows too.
    log = _calibration_log(
        [
            {"batch_id": "old_batch", "tier": "flash", "mode": "primary", "record_id": f"r{i}",
             "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
             "timestamp": "2026-08-20T12:00:00+00:00"}
            for i in range(8)
        ]
        + [
            {"batch_id": "new_batch", "tier": "flash", "mode": "primary", "record_id": f"n{i}",
             "prompt_tokens": 8000, "completion_tokens": 200, "total_tokens": 8200,
             "timestamp": "2026-08-21T14:00:00+00:00"}
            for i in range(10)
        ]
    )
    result = project_cost(log, tier="flash", observed_usd_spent=0.01, target_n=100)
    assert result["batch_id"] == "new_batch"
    assert result["n_calibration_calls"] == 10  # NOT 18 (old + new combined)
    assert result["usd_per_record"] == pytest.approx(0.01 / 10)


def test_project_cost_explicit_batch_id_overrides_auto_resolution():
    log = _calibration_log(
        [
            {"batch_id": "old_batch", "tier": "flash", "mode": "primary", "record_id": "r0",
             "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
             "timestamp": "2026-08-20T12:00:00+00:00"},
            {"batch_id": "new_batch", "tier": "flash", "mode": "primary", "record_id": "n0",
             "prompt_tokens": 8000, "completion_tokens": 200, "total_tokens": 8200,
             "timestamp": "2026-08-21T14:00:00+00:00"},
        ]
    )
    result = project_cost(log, tier="flash", observed_usd_spent=0.02, target_n=50, batch_id="old_batch")
    assert result["batch_id"] == "old_batch"
    assert result["n_calibration_calls"] == 1


def test_project_cost_raises_when_no_rows_for_tier_mode():
    log = _calibration_log(
        [{"batch_id": "b", "tier": "pro", "mode": "primary", "record_id": "r0",
          "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
          "timestamp": "2026-08-20T12:00:00+00:00"}]
    )
    with pytest.raises(ValueError):
        project_cost(log, tier="flash", observed_usd_spent=0.01, target_n=100)


def test_project_cost_linearly_extrapolates_to_target_n():
    log = _calibration_log(
        [{"batch_id": "b", "tier": "flash", "mode": "primary", "record_id": f"r{i}",
          "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
          "timestamp": "2026-08-20T12:00:00+00:00"} for i in range(4)]
    )
    result = project_cost(log, tier="flash", observed_usd_spent=0.04, target_n=1000)
    assert result["usd_per_record"] == pytest.approx(0.01)
    assert result["projected_usd_for_target_n"] == pytest.approx(10.0)
