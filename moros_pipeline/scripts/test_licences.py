"""Tests for the licence backfill: the fetch, the join, and the field load.

Why these exist: `source.access.license` distinguishes three states, and the whole gap arose from
conflating two of them. `null` means "never looked up", `""` means "looked up, EPMC disclosed
none", and a real string is a disclosed licence. A fetch that drops an unanswerable key leaves the
document `null` forever and re-fetches it on every future run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import fetch_citations as fc
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
    """Captures request params so the resultType can be asserted directly."""

    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.params: dict = {}

    def get(self, url, params=None, timeout=None):
        self.params = params or {}
        return _FakeResponse({"resultList": {"result": self.results}})


# ---------------------------------------------------------------------------
# resultType
# ---------------------------------------------------------------------------


def test_licence_fetch_uses_core_because_lite_has_no_license_field():
    """Verified live: `lite` carries citedByCount but no `license` key at all."""
    session = _FakeSession([])
    fc.fetch_chunk(session, "pmid", ["1"], "now", with_licence=True)
    assert session.params["resultType"] == "core"


def test_a_plain_citation_run_still_uses_lite():
    """A routine citation refresh must not pull heavier core responses it does not need."""
    session = _FakeSession([])
    fc.fetch_chunk(session, "pmid", ["1"], "now")
    assert session.params["resultType"] == "lite"


# ---------------------------------------------------------------------------
# Every key gets an answer
# ---------------------------------------------------------------------------


def test_licence_and_count_come_back_together():
    session = _FakeSession([
        {"source": "MED", "pmid": "1", "doi": "10.1/a", "citedByCount": 7,
         "license": "cc by", "isOpenAccess": "Y"},
    ])
    row = fc.fetch_chunk(session, "pmid", ["1"], "now", with_licence=True)[0]
    assert row["license"] == "cc by"
    assert row["epmc_is_open_access"] == "Y"
    assert row["citation_count"] == 7


def test_a_key_epmc_cannot_answer_still_records_none_disclosed():
    """The defect this fixes: 12,980 DOI-keyed lookups returned a PMC-source record carrying
    neither the doi nor a licence, so they could not be attributed and were dropped -- leaving
    the document null and guaranteeing it would be re-fetched forever."""
    session = _FakeSession([{"source": "PMC", "pmcid": "PMC1", "doi": None, "citedByCount": 0}])
    rows = fc.fetch_chunk(session, "doi", ["10.1/unanswerable"], "now", with_licence=True)
    answered = {r["key"]: r for r in rows}
    assert "10.1/unanswerable" in answered
    assert answered["10.1/unanswerable"]["license"] == ""
    assert answered["10.1/unanswerable"]["citation_count"] == ""


def test_an_unanswerable_key_is_not_invented_on_a_citations_only_run():
    """Without --with-licence a miss means "no count", which is correctly expressed by writing
    no row at all -- inventing one would claim a count of zero."""
    session = _FakeSession([])
    assert fc.fetch_chunk(session, "doi", ["10.1/missing"], "now") == []


def test_doi_keys_are_normalised_consistently_for_misses_too():
    session = _FakeSession([])
    rows = fc.fetch_chunk(session, "doi", ["10.1/MixedCase"], "now", with_licence=True)
    assert rows[0]["key"] == "10.1/mixedcase"


# ---------------------------------------------------------------------------
# The field load
# ---------------------------------------------------------------------------


def test_a_disclosed_licence_writes_both_fields():
    """open_access is written too, because schema.py's _resolve_open_access fixes the rule that
    EPMC's fresh flag wins wherever a real lookup happened."""
    pid, update = lf.licence_row_to_update(
        {"pid": "p1", "license": "cc by", "epmc_is_open_access": "Y"})
    assert pid == "p1"
    assert update == {"source.access.license": "cc by", "source.access.open_access": True}


def test_none_disclosed_is_written_as_empty_string_not_skipped():
    """Skipping it would leave the document null and re-fetch it on every future backfill."""
    _, update = lf.licence_row_to_update({"pid": "p1", "license": "", "epmc_is_open_access": ""})
    assert update["source.access.license"] == ""


def test_no_epmc_flag_leaves_open_access_alone():
    """The existing value is a better guess than overwriting it with nothing."""
    _, update = lf.licence_row_to_update({"pid": "p1", "license": "cc by",
                                          "epmc_is_open_access": ""})
    assert "source.access.open_access" not in update


def test_a_citations_only_row_yields_no_licence_update():
    """It must not blank a licence that is already correct."""
    assert lf.licence_row_to_update({"pid": "p1", "citation_count": "5"}) is None


def test_the_licences_allowlist_is_exactly_two_fields_plus_the_version_and_the_stamp():
    assert mw.WRITE_MODES["licences"] == frozenset({
        "schema_version", "record_modified", "source.access.license", "source.access.open_access",
    })


def test_the_licences_mode_cannot_reach_classification_or_enrichment():
    allowed = mw.WRITE_MODES["licences"]
    assert not any(p.startswith("llm_classification") for p in allowed)
    assert not any(p.startswith("llm_enrichment") for p in allowed)
    assert "source.access.fulltext_available" not in allowed  # ours, not EPMC's


def test_every_mapper_output_path_is_inside_its_allowlist():
    """The mapper and the allowlist are edited in different files; a drift between them would only
    surface as a refused write at load time, against production."""
    _, update = lf.licence_row_to_update({"pid": "p", "license": "cc by",
                                          "epmc_is_open_access": "N"})
    assert set(update) <= mw.WRITE_MODES["licences"]
