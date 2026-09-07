import pandas as pd
import pytest

from dome_triage.llm_classify.budget import BudgetExceededError, check_cap, log_spend


def test_check_cap_raises_when_projection_exceeds_cap_and_not_confirmed(tmp_path):
    spend_log = tmp_path / "spend.csv"
    with pytest.raises(BudgetExceededError):
        check_cap(spend_log, total_cap_usd=10.0, projected_additional_usd=15.0, confirmed=False)


def test_check_cap_allows_over_cap_when_confirmed(tmp_path, capsys):
    spend_log = tmp_path / "spend.csv"
    check_cap(spend_log, total_cap_usd=10.0, projected_additional_usd=15.0, confirmed=True)
    assert "WARNING" in capsys.readouterr().out


def test_check_cap_allows_under_cap_without_confirm(tmp_path):
    spend_log = tmp_path / "spend.csv"
    check_cap(spend_log, total_cap_usd=10.0, projected_additional_usd=2.0, confirmed=False)


def test_check_cap_accounts_for_prior_logged_spend(tmp_path):
    spend_log = tmp_path / "spend.csv"
    log_spend(spend_log, "llm-classify.validate-criteria", "flash", "primary", 12, 0.05, 0.04, "gavinfarrell")
    log_spend(spend_log, "llm-classify.calibrate", "flash", "primary", 8, 0.03, 0.03, "gavinfarrell")

    # Already spent ~$0.07; a further $9.95 projection would cross the $10 cap.
    with pytest.raises(BudgetExceededError):
        check_cap(spend_log, total_cap_usd=10.0, projected_additional_usd=9.95, confirmed=False)


def test_log_spend_appends_rows_with_header_only_once(tmp_path):
    spend_log = tmp_path / "spend.csv"
    log_spend(spend_log, "step_a", "flash", "primary", 10, 0.1, None, "gavinfarrell")
    log_spend(spend_log, "step_b", "pro", "primary", 5, 0.2, 0.25, "gavinfarrell")

    df = pd.read_csv(spend_log)
    assert len(df) == 2
    assert list(df["step_name"]) == ["step_a", "step_b"]


def test_cumulative_spend_prefers_actual_over_estimated(tmp_path):
    spend_log = tmp_path / "spend.csv"
    log_spend(spend_log, "step_a", "flash", "primary", 10, 5.0, 1.0, "gavinfarrell")  # actual much lower

    # If estimated were used instead of actual, this would exceed the cap.
    check_cap(spend_log, total_cap_usd=2.0, projected_additional_usd=0.5, confirmed=False)
