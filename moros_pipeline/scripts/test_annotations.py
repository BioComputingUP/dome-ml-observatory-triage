"""Tests for the annotations-API fetch, against a real captured response (fixtures/)."""

from __future__ import annotations

import json
from pathlib import Path

import fetch_annotations as fa

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "epmc_annotations_8ids.json")
                     .read_text(encoding="utf-8"))


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.params = {}

    def get(self, url, params=None, timeout=None):
        self.params = params or {}
        return _FakeResponse(self.payload)


def test_the_call_asks_for_accession_numbers_in_json_by_source_and_id():
    session = _FakeSession([])
    fa.fetch_batch(session, [("MED", "1"), ("PPR", "PPR2")], "now")
    assert session.params["articleIds"] == "MED:1,PPR:PPR2"
    assert session.params["type"] == "Accession Numbers"
    assert session.params["format"] == "JSON"


def test_eight_ids_per_call_is_the_hard_limit():
    assert fa.IDS_PER_CALL == 8


def test_a_real_response_reduces_to_one_record_per_requested_id():
    requested = [(a["source"], a["extId"]) for a in FIXTURE] + [("MED", "0")]
    records = fa.reduce_response(FIXTURE, requested, "now")
    assert [r["id"] for r in records] == [a["extId"] for a in FIXTURE] + ["0"]
    assert records[-1]["status"] == "absent" and records[-1]["annotations"] == []
    assert all(r["status"] == "ok" for r in records[:-1])


def test_an_article_with_no_accessions_is_ok_with_an_empty_list():
    empty = next(a for a in FIXTURE if not a["annotations"])
    rec = fa.reduce_response([empty], [(empty["source"], empty["extId"])], "now")[0]
    assert rec["status"] == "ok" and rec["annotations"] == []


def test_text_mined_accessions_keep_type_uri_and_section():
    alphafold = next(a for a in FIXTURE if a["extId"] == "34265844")
    reduced = [fa.reduce_annotation(x) for x in alphafold["annotations"]]
    pdb = next(r for r in reduced if r["sub_type"] == "PDBe")
    assert pdb["uri"].startswith("http://identifiers.org/pdbe/pdb:")
    assert pdb["section"] == "Article"
    assert pdb["provider"] == "Europe PMC"


def test_a_supplementary_file_accession_recovers_its_type_from_the_uri():
    """BioStudies-provider annotations carry no subType; the identifiers.org path does."""
    alphafold = next(a for a in FIXTURE if a["extId"] == "34265844")
    biostudies = next(x for x in alphafold["annotations"] if x["provider"] == "Biostudies")
    reduced = fa.reduce_annotation(biostudies)
    assert reduced["sub_type"] == "pdbe"
    assert reduced["file_name"] and reduced["frequency"]
    assert reduced["section"] == "Supplementary material"


def test_a_reference_list_doi_is_still_carried_with_its_section():
    """Filtering literature DOIs out is build_data_links.py's job; the fetch records what it saw."""
    alphafold = next(a for a in FIXTURE if a["extId"] == "34265844")
    doi = next(fa.reduce_annotation(x) for x in alphafold["annotations"]
               if x.get("subType") == "DOI")
    assert doi["section"] == "References"
    assert doi["exact"].startswith("10.")
