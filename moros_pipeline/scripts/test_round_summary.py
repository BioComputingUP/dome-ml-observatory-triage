"""Tests for round_summary.py: the batch filter and the shaping of grouped rows into one round.

Why these exist: the processing history page shows these figures as fact, so a round that ran as
several batches must add up, verdicts must not leak between tiles, and a curated batch with no model
must say so rather than borrow one.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import round_summary as rs


def _row(batch, verdict, n, first, last, model="deepseek-v4-flash", prompt="v1"):
    return {"_id": {"batch": batch, "verdict": verdict, "model_id": model, "prompt_version": prompt},
            "n": n, "first": first, "last": last}


def test_filter_exact_ids_only():
    assert rs.batch_filter("llm_classification.batch_id", ["a", "b"], []) == {
        "llm_classification.batch_id": {"$in": ["a", "b"]}}


def test_filter_prefix_is_anchored_and_escaped():
    f = rs.batch_filter("llm_enrichment.batch_id", [], ["enrich_flash_2026.09"])
    assert f == {"llm_enrichment.batch_id": {"$regex": r"^enrich_flash_2026\.09"}}


def test_filter_ids_and_prefixes_combine_with_or():
    f = rs.batch_filter("x", ["a"], ["p"])
    assert f == {"$or": [{"x": {"$in": ["a"]}}, {"x": {"$regex": "^p"}}]}


def test_filter_needs_something():
    with pytest.raises(ValueError):
        rs.batch_filter("x", [], [])


def test_a_round_of_several_batches_adds_up():
    rows = [
        _row("b1", "positive", 299, "2026-08-27T22:44:56", "2026-08-27T22:45:26"),
        _row("b1", "negative", 444, "2026-08-27T22:44:56", "2026-08-27T22:45:14"),
        _row("b2", "positive", 21573, "2026-08-27T23:21:42", "2026-08-27T23:25:15"),
        _row("b2", "undeterminable", 390, "2026-08-27T23:21:46", "2026-08-27T23:25:11"),
    ]
    s = rs.summarise(rows)
    assert s["total"] == 22706
    assert (s["positive"], s["negative"], s["undeterminable"]) == (21872, 444, 390)
    assert s["batch_ids"] == ["b1", "b2"]
    assert (s["first"], s["last"]) == ("2026-08-27T22:44:56", "2026-08-27T23:25:15")
    assert s["model_ids"] == ["deepseek-v4-flash"] and s["prompt_versions"] == ["v1"]


def test_a_curated_batch_reports_no_model():
    s = rs.summarise([_row("curated_merge_2026-09-03", "positive", 3307, "2026-07-31T13:08:14",
                           "2026-08-26T21:22:59", model=None, prompt=None)])
    assert s["model_ids"] == [None] and s["prompt_versions"] == [None]


def test_datetime_timestamps_become_utc_iso():
    t = datetime(2026, 9, 3, 20, 12, 20, tzinfo=timezone.utc)
    s = rs.summarise([_row("b", "positive", 1, t, t)])
    assert s["first"] == "2026-09-03T20:12:20+00:00"


def test_empty_round():
    s = rs.summarise([])
    assert s["total"] == 0 and s["first"] is None and s["batch_ids"] == []
