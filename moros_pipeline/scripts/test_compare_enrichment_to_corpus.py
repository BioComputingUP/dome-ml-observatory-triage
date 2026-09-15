"""Tests for compare_enrichment_to_corpus.py.

Why these exist: the comparison decides whether a model change is loaded into the corpus, so it
must read the events file the way `load_enrichment.py` would write it (tier 1 as its first term,
the other fields as lists), take the last parsed event per record, and count agreement per field
without letting an empty field inflate a mismatch.
"""

from __future__ import annotations

import json

import pytest

import compare_enrichment_to_corpus as ce


def _event(rid, ts="2026-09-15T21:00:00", status="ok", **fields):
    row = {"record_id": rid, "timestamp": ts, "parse_status": status}
    for f in ce.FIELDS:
        row[f] = json.dumps(fields.get(f, []))
    return row


def _doc(rid, **fields):
    cf = {f: fields.get(f, []) for f in ce.FIELDS}
    cf["domain_tier1"] = fields.get("domain_tier1")
    return {"_id": rid, "content_filters": cf, "llm_enrichment": {"provider": "deepseek"}}


def test_json_list_tolerates_blank_and_bad_cells():
    assert ce.json_list("") == [] and ce.json_list("not json") == [] and ce.json_list('{"a": 1}') == []
    assert ce.json_list('["a", "b"]') == ["a", "b"]


def test_latest_parsed_event_wins():
    rows = [_event("r", ts="2026-09-15T21:00:02", status="parse_error"),
            _event("r", ts="2026-09-15T21:00:01", model_family=["tree ensembles"]),
            _event("r", ts="2026-09-15T21:00:00", model_family=["neural networks"])]
    assert json.loads(ce.latest_ok(rows)["r"]["model_family"]) == ["tree ensembles"]


def test_tier1_is_compared_as_stored_first_term_only():
    row = _event("r", domain_tier1=["Biology", "Medicine"])
    assert ce.event_terms(row, "domain_tier1") == {"biology"}
    assert ce.stored_terms(_doc("r", domain_tier1="Biology"), "domain_tier1") == {"biology"}
    assert ce.stored_terms(_doc("r"), "domain_tier1") == set()


def test_jaccard():
    assert ce.jaccard(set(), set()) == 1.0
    assert ce.jaccard({"a"}, {"a", "b"}) == pytest.approx(0.5)
    assert ce.jaccard({"a"}, {"b"}) == 0.0


def test_compare_counts_fields_and_all_six():
    same = {"domain_tier1": ["Biology"], "learning_paradigm": ["supervised learning"], "model_type": ["Random Forest"]}
    pairs = [
        (_event("a", **same), _doc("a", domain_tier1="Biology", learning_paradigm=["supervised learning"], model_type=["random forest"])),
        (_event("b", **same), _doc("b", domain_tier1="Biology", learning_paradigm=["unsupervised learning"], model_type=["Random Forest"])),
    ]
    r = ce.compare(pairs)
    assert r["n"] == 2
    assert r["fields"]["domain_tier1"]["exact"] == 1.0
    assert r["fields"]["learning_paradigm"]["exact"] == 0.5
    assert r["fields"]["model_type"]["exact"] == 1.0          # case-folded
    assert r["fields"]["domain_tier2"]["both_empty"] == 1.0
    assert r["all_six"] == 0.5


def test_compare_nothing():
    assert ce.compare([]) == {"n": 0, "fields": {}, "all_six": 0}


def test_disagreement_examples():
    pairs = [(_event("a", model_family=["x"]), _doc("a", model_family=["y"]))]
    assert ce.disagreements(pairs, "model_family", 5) == [{"record_id": "a", "events": ["x"], "stored": ["y"]}]
