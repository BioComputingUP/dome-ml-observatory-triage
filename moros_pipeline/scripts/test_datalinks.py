"""Tests for the /datalinks fetch: the Scholix flattening and the target rule."""

from __future__ import annotations

import json
from pathlib import Path

import fetch_datalinks as fd

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "epmc_datalinks_33024307.json")
                     .read_text(encoding="utf-8"))


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload, status=200):
        self.payload, self.status, self.url = payload, status, None

    def get(self, url, params=None, timeout=None):
        self.url = url
        return _FakeResponse(self.payload, self.status)


def test_the_scholix_payload_flattens_to_one_row_per_link():
    links = fd.reduce_datalinks(FIXTURE)
    assert [(l["category"], l["id_scheme"], l["id"]) for l in links] == [
        ("Nucleotide Sequences", "ENA", "AY278488"),
        ("Data Citations", "DOI", "10.5281/zenodo.4457982"),
        ("Data Citations", "DOI", "10.5061/dryad.dm57j"),
    ]
    zenodo = links[1]
    assert zenodo["publisher"] == "Zenodo"
    assert zenodo["relationship"] == "IsSupplementedBy"
    assert zenodo["obtained_by"] == "ext_links"
    assert zenodo["url"] == "https://doi.org/10.5281/zenodo.4457982"
    assert zenodo["title"].startswith("Structural basis")
    assert links[0]["obtained_by"] == "tm_accession" and links[0]["link_provider"] == "Europe PMC"


def test_missing_levels_yield_no_links_not_errors():
    assert fd.reduce_datalinks({}) == []
    assert fd.reduce_datalinks({"dataLinkList": {}}) == []
    assert fd.reduce_datalinks({"dataLinkList": {"Category": [{"Name": "x"}]}}) == []


def test_the_url_is_source_id_datalinks():
    session = _FakeSession(FIXTURE)
    rec = fd.fetch_one(session, "PPR", "PPR18364", "now")
    assert session.url.endswith("/PPR/PPR18364/datalinks")
    assert rec["http_status"] == 200 and rec["hit_count"] == 3 and len(rec["links"]) == 3


def test_a_404_is_an_answer_with_no_links():
    rec = fd.fetch_one(_FakeSession(None, 404), "MED", "1", "now")
    assert rec == {"source": "MED", "id": "1", "fetched_at": "now", "http_status": 404,
                   "hit_count": 0, "links": []}


def test_the_residual_scope_is_db_xrefs_or_related_data():
    assert fd.wanted({"has_db_xrefs": "Y", "data_links_tags": "[]"}, "residual")
    assert fd.wanted({"has_db_xrefs": "N", "data_links_tags": '["related_data"]'}, "residual")
    assert not fd.wanted({"has_db_xrefs": "N", "data_links_tags": '["supporting_data"]'}, "residual")
    assert fd.wanted({"has_db_xrefs": "N", "data_links_tags": '["supporting_data"]'}, "supporting")
    assert fd.wanted({"has_data": "Y", "data_links_tags": '["altmetrics"]'}, "all")
    assert not fd.wanted({"has_data": "N"}, "all")
