"""Tests for the Europe PMC metadata pass: the key rule, the PPR-first selection, and the row."""

from __future__ import annotations

import json

import fetch_citations as fc
import fetch_epmc_metadata as fm


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, results):
        self.results = results
        self.params = {}

    def get(self, url, params=None, timeout=None):
        self.params = params or {}
        return _FakeResponse({"resultList": {"result": self.results}})


PPR = {"id": "PPR18364", "source": "PPR", "doi": "10.1101/270413", "citedByCount": 2,
       "bookOrReportDetails": {"publisher": "bioRxiv"}, "hasData": "Y",
       "dataLinksTagsList": {"dataLinkstag": ["supporting_data"]},
       "tmAccessionTypeList": {"accessionType": ["pdb"]}, "hasTMAccessionNumbers": "Y",
       "hasDbCrossReferences": "N", "hasSuppl": "Y"}
MED = {"id": "30001", "source": "MED", "pmid": "30001", "doi": "10.1101/270413",
       "citedByCount": 40, "journalTitle": "Nature"}


# -- the key rule -----------------------------------------------------------------------------


def test_a_preprint_is_keyed_by_doi_in_the_ppr_pass_even_with_a_pmid():
    row = {"pmid": "3157", "pmcid": "", "doi": "10.1101/270413", "is_preprint": "True"}
    assert fm.metadata_key(row) == (fm.PPR_KEY_TYPE, "10.1101/270413")


def test_a_staging_row_declares_preprint_through_pub_types():
    row = {"pmid": "", "pmcid": "", "doi": "10.21203/rs.3.rs-1", "pub_types": json.dumps(["Preprint"])}
    assert fm.metadata_key(row) == (fm.PPR_KEY_TYPE, "10.21203/rs.3.rs-1")
    lower = dict(row, pub_types=json.dumps(["preprint"]))
    assert fm.metadata_key(lower)[0] == fm.PPR_KEY_TYPE


def test_everything_else_follows_assign_key():
    row = {"pmid": "1", "pmcid": "PMC1", "doi": "10.1/A", "is_preprint": "False"}
    assert fm.metadata_key(row) == fc.assign_key("1", "PMC1", "10.1/A") == ("pmid", "1")
    assert fm.metadata_key({"pmid": "", "pmcid": "PMC1", "doi": ""}) == ("pmcid", "PMC1")


# -- the PPR pass -----------------------------------------------------------------------------


def test_the_ppr_pass_restricts_the_query_to_src_ppr():
    session = _FakeSession([])
    fm.fetch_chunk(session, fm.PPR_KEY_TYPE, ["10.1101/270413"], "now")
    assert session.params["query"] == '(DOI:"10.1101/270413") AND SRC:PPR'
    assert session.params["resultType"] == "core"


def test_the_ppr_record_wins_over_its_published_twin():
    """pick_best() would take MED (40 citations, preferred source); this pass must not."""
    assert fm.select_record(fm.PPR_KEY_TYPE, [MED, PPR]) is PPR
    assert fm.select_record("doi", [MED, PPR]) is MED


def test_the_default_query_rule_is_untouched():
    assert fc.build_query("pmid", ["1", "2"]) == "(EXT_ID:1 OR EXT_ID:2) AND SRC:MED"
    assert fc.build_query("doi", ["10.1/a"]) == '(DOI:"10.1/a")'
    assert fc.build_query("doi", ["10.1/a"], src="PPR") == '(DOI:"10.1/a") AND SRC:PPR'


# -- the row ----------------------------------------------------------------------------------


def test_a_preprint_row_carries_identity_server_and_summary():
    session = _FakeSession([PPR, MED])
    rows = fm.fetch_chunk(session, fm.PPR_KEY_TYPE, ["10.1101/270413"], "now")
    assert len(rows) == 1
    row = rows[0]
    assert (row["epmc_source"], row["epmc_id"], row["preprint_server"]) == ("PPR", "PPR18364", "bioRxiv")
    assert row["has_data"] == "Y"
    assert json.loads(row["data_links_tags"]) == ["supporting_data"]
    assert json.loads(row["accession_types"]) == ["pdb"]
    assert json.loads(row["db_cross_references"]) == []
    assert (row["has_tm_accessions"], row["has_db_xrefs"], row["has_suppl"]) == ("Y", "N", "Y")


def test_a_medline_row_has_no_server_and_n_for_absent_flags():
    session = _FakeSession([MED])
    row = fm.fetch_chunk(session, "pmid", ["30001"], "now")[0]
    assert (row["epmc_source"], row["epmc_id"], row["preprint_server"]) == ("MED", "30001", "")
    assert row["has_data"] == "N"
    assert row["data_links_tags"] == "[]"


def test_a_key_epmc_did_not_return_is_recorded_as_a_miss_not_dropped():
    """Dropping it would re-fetch it on every run, forever (the licence lesson)."""
    session = _FakeSession([])
    rows = fm.fetch_chunk(session, "doi", ["10.1/Missing"], "now")
    assert rows == [fm.miss_row("doi", "10.1/missing", "now")]
    assert rows[0]["epmc_id"] == "" and rows[0]["has_data"] == "N"


def test_output_columns_are_stable():
    assert set(fm.record_to_row("pmid", "1", MED, "now")) == set(fm.OUTPUT_COLUMNS)
    assert set(fm.miss_row("pmid", "1", "now")) == set(fm.OUTPUT_COLUMNS)


# -- keys a chunk cannot attribute ------------------------------------------------------------


def test_a_single_key_query_attributes_a_record_that_does_not_echo_the_doi():
    """The measured case: DOI-keyed lookups answered by a PMC-source record with no doi field."""
    pmc = {"id": "PMC12739028", "source": "PMC", "pmcid": "PMC12739028", "doi": None, "hasData": "Y",
           "hasTMAccessionNumbers": "Y"}
    session = _FakeSession([pmc])
    row = fm.fetch_single(session, "doi", "10.1002/ALZ70856_101044", "now")
    assert session.params["query"] == '(DOI:"10.1002/ALZ70856_101044")'
    assert (row["key"], row["epmc_source"], row["epmc_id"]) == ("10.1002/alz70856_101044", "PMC", "PMC12739028")
    assert row["has_tm_accessions"] == "Y"


def test_a_single_key_query_ignores_a_record_echoing_a_different_key():
    other = {"id": "1", "source": "MED", "doi": "10.1/other"}
    row = fm.fetch_single(_FakeSession([other]), "doi", "10.1/wanted", "now")
    assert row == fm.miss_row("doi", "10.1/wanted", "now")


def test_misses_are_read_from_the_latest_row_per_key(tmp_path):
    import csv as _csv
    out = tmp_path / "meta.csv"
    with out.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fm.OUTPUT_COLUMNS)
        w.writeheader()
        w.writerow(fm.miss_row("doi", "10.1/a", "t1"))
        w.writerow(fm.record_to_row("doi", "10.1/a", {"id": "PMC1", "source": "PMC"}, "t2"))
        w.writerow(fm.miss_row("doi", "10.1/b", "t1"))
        w.writerow(fm.miss_row("pmid", "9", "t1"))
    targets = {"doi": ["10.1/a", "10.1/b"], "pmid": [], "pmcid": [], fm.PPR_KEY_TYPE: []}
    assert fm.load_misses(out, targets) == [("doi", "10.1/b")]   # pmid 9 is not this input's key


def test_a_doi_key_is_answered_through_its_pmcid_in_a_batch_and_stays_keyed_by_the_doi():
    pmc = {"id": "PMC7", "source": "PMC", "pmcid": "PMC7", "doi": None, "hasData": "Y"}
    session = _FakeSession([pmc])
    rows, left = fm.fetch_via_pmcid(session, [(("doi", "10.1/a"), "PMC7"), (("doi", "10.1/b"), "PMC8")], "now")
    assert session.params["query"] == "(PMCID:PMC7 OR PMCID:PMC8)"
    assert [(r["key_type"], r["key"], r["epmc_id"]) for r in rows] == [("doi", "10.1/a", "PMC7")]
    assert left == [("doi", "10.1/b")]


def test_only_plain_doi_keys_get_a_pmcid_alternate(tmp_path):
    import csv as _csv
    keys = tmp_path / "keys.csv"
    with keys.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["pid", "pmid", "pmcid", "doi", "is_preprint"])
        w.writeheader()
        w.writerow({"pid": "a", "pmid": "", "pmcid": "PMC1", "doi": "10.1/A", "is_preprint": "False"})
        w.writerow({"pid": "b", "pmid": "", "pmcid": "PMC2", "doi": "10.1101/2", "is_preprint": "True"})
        w.writerow({"pid": "c", "pmid": "3", "pmcid": "PMC3", "doi": "10.1/c", "is_preprint": "False"})
    assert fm.load_pmcid_alternates(keys) == {("doi", "10.1/a"): "PMC1"}


def test_a_non_preprint_publisher_is_not_a_preprint_server():
    thesis = {"id": "ETH:1", "source": "ETH", "bookOrReportDetails": {"publisher": "University of Leeds"}}
    assert fm.identity_fields(thesis)["preprint_server"] == ""
    assert fm.identity_fields(PPR)["preprint_server"] == "bioRxiv"
