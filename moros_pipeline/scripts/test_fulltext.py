"""Tests for the full-text refresh: the Europe PMC lookup, the row mapping, and the write mode.

Why these exist: `source.access.fulltext_available` read false on 16,379 documents with a PMCID on
2026-09-25 -- the curated merge never derived it, and embargoes lift after a fetch. The refresh has
to set it by exactly the rule the bulk match did (`inEPMC` or `inPMC`), leave an unanswered record
alone, and be unable to reach anything but that one leaf.
"""

from __future__ import annotations

import pytest

import fetch_fulltext as ff
import load_fields as lf
import moros_write as mw


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.params: dict = {}

    def get(self, url, params=None, timeout=None):
        self.params = params or {}
        return _FakeResponse({"resultList": {"result": self.results}})


def test_lookup_uses_the_lite_response_which_carries_both_flags():
    session = _FakeSession([{"pmid": "34265844", "source": "MED", "inEPMC": "Y", "inPMC": "Y"}])
    found = ff.fetch_chunk(session, "pmid", ["34265844"])
    assert session.params["resultType"] == "lite"
    assert ff.fulltext_flags(found["34265844"]) == ("Y", "Y")


def test_a_key_europe_pmc_does_not_answer_gets_no_row():
    session = _FakeSession([{"pmid": "1", "source": "MED", "inEPMC": "N", "inPMC": "N"}])
    assert set(ff.fetch_chunk(session, "pmid", ["1", "2"])) == {"1"}


def test_the_published_article_wins_over_its_preprint_as_in_the_citation_fetch():
    session = _FakeSession([
        {"doi": "10.1/x", "source": "PPR", "inEPMC": "N", "inPMC": "N", "id": "PPR1"},
        {"doi": "10.1/X", "source": "MED", "inEPMC": "Y", "inPMC": "Y", "id": "123"},
    ])
    assert ff.fetch_chunk(session, "doi", ["10.1/x"])["10.1/x"]["source"] == "MED"


@pytest.mark.parametrize("in_epmc, in_pmc, expected", [
    ("Y", "Y", True),
    ("Y", "N", True),   # Europe PMC's own full text, not in PMC
    ("N", "Y", True),
    ("N", "N", False),  # still embargoed, or never deposited
])
def test_the_flag_follows_the_bulk_match_rule(in_epmc, in_pmc, expected):
    row = {"pid": "p1", "in_epmc": in_epmc, "in_pmc": in_pmc}
    assert lf.fulltext_row_to_update(row) == ("p1", {"source.access.fulltext_available": expected})


def test_a_row_with_neither_flag_leaves_the_value_alone():
    assert lf.fulltext_row_to_update({"pid": "p1", "in_epmc": "", "in_pmc": ""}) is None


def test_the_mode_writes_the_one_leaf_and_nothing_that_decides_a_record():
    allowed = mw.WRITE_MODES["fulltext"]
    assert "source.access.fulltext_available" in allowed
    for path in ("source.access.open_access", "source.access.license",
                 "llm_classification.classification", "source.decision_provenance"):
        assert path not in allowed


def test_a_fulltext_change_does_not_force_a_re_harvest():
    # The flag is in neither the Dublin Core nor the JSON-LD the observatory serves.
    assert "fulltext" not in mw.STAMPS_RECORD_MODIFIED
