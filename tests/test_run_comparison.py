import pandas as pd

from dome_triage.reporting.run_comparison import compare_runs


def _log(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(rid, cls, ts="2026-08-27T00:00:00", sha="abc", pv="v1"):
    return {"record_id": rid, "classification": cls, "timestamp": ts,
            "criteria_sha256": sha, "prompt_version": pv}


def test_identical_runs_report_full_agreement(tmp_path):
    rows = [_row("r1", "positive"), _row("r2", "negative")]
    _log(tmp_path / "a.csv", rows)
    _log(tmp_path / "b.csv", rows)
    result = compare_runs(tmp_path / "a.csv", tmp_path / "b.csv")
    assert result["n_shared"] == 2
    assert result["agreement_rate"] == 1.0
    assert result["disagreements"].empty


def test_disagreements_are_reported_with_both_sides(tmp_path):
    _log(tmp_path / "a.csv", [_row("r1", "positive"), _row("r2", "negative")])
    _log(tmp_path / "b.csv", [_row("r1", "negative"), _row("r2", "negative")])
    result = compare_runs(tmp_path / "a.csv", tmp_path / "b.csv")
    assert result["agreement_rate"] == 0.5
    d = result["disagreements"]
    assert list(d.record_id) == ["r1"]
    assert d.iloc[0].classification_a == "positive"
    assert d.iloc[0].classification_b == "negative"


def test_only_shared_records_are_compared(tmp_path):
    _log(tmp_path / "a.csv", [_row("r1", "positive"), _row("only_in_a", "negative")])
    _log(tmp_path / "b.csv", [_row("r1", "positive"), _row("only_in_b", "positive")])
    result = compare_runs(tmp_path / "a.csv", tmp_path / "b.csv")
    assert result["n_shared"] == 1
    assert result["agreement_rate"] == 1.0


def test_no_overlap_is_reported_rather_than_crashing(tmp_path):
    _log(tmp_path / "a.csv", [_row("x", "positive")])
    _log(tmp_path / "b.csv", [_row("y", "positive")])
    result = compare_runs(tmp_path / "a.csv", tmp_path / "b.csv")
    assert result["n_shared"] == 0
    assert result["agreement_rate"] is None


def test_latest_event_per_record_wins_matching_project_convention(tmp_path):
    _log(tmp_path / "a.csv", [
        _row("r1", "negative", ts="2026-08-27T00:00:00"),
        _row("r1", "positive", ts="2026-08-27T01:00:00"),
    ])
    _log(tmp_path / "b.csv", [_row("r1", "positive")])
    result = compare_runs(tmp_path / "a.csv", tmp_path / "b.csv")
    assert result["n_shared"] == 1
    assert result["agreement_rate"] == 1.0


def test_criteria_and_prompt_mismatch_is_surfaced(tmp_path):
    # If the two runs used different criteria, the comparison is meaningless -- say so loudly.
    _log(tmp_path / "a.csv", [_row("r1", "positive", sha="AAA")])
    _log(tmp_path / "b.csv", [_row("r1", "positive", sha="BBB")])
    result = compare_runs(tmp_path / "a.csv", tmp_path / "b.csv")
    assert result["same_criteria_sha256"] is False
