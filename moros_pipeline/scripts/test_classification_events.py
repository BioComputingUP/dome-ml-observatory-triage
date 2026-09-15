"""Tests for reading verdicts off a classification event log."""

from __future__ import annotations

import classification_events as ce


def test_the_latest_usable_verdict_wins_and_parse_errors_and_blank_ids_are_ignored(tmp_path):
    path = tmp_path / "events.csv"
    path.write_text(
        "record_id,classification,rationale\n"
        "p1,negative,first pass\n"
        "p1,positive,re-run\n"
        "p2,parse_error,no verdict\n"
        "p3,undeterminable,unclear\n"
        "p3,parse_error,a retry that failed\n"
        ",positive,no id\n", encoding="utf-8")
    assert ce.latest_verdicts(path) == {"p1": "positive", "p3": "undeterminable"}
