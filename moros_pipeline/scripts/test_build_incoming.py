"""Tests for the staging row a new Europe PMC record becomes: the v1.3.0 identity and the v1.4.0
data-links summary must ride in on the core search record the pipeline already fetches."""

from __future__ import annotations

import json

import build_incoming_documents as bi
import fetch_epmc_metadata as fm

RECORD = {
    "id": "PPR18364", "source": "PPR", "doi": "10.1101/270413", "title": "A preprint",
    "abstractText": "An abstract", "authorString": "A B", "pubYear": "2018",
    "bookOrReportDetails": {"publisher": "bioRxiv", "yearOfPublication": 2018},
    "pubTypeList": {"pubType": ["Preprint"]}, "isOpenAccess": "Y", "inEPMC": "Y",
    "hasData": "Y", "dataLinksTagsList": {"dataLinkstag": ["supporting_data", "altmetrics"]},
    "tmAccessionTypeList": {"accessionType": ["pdb"]}, "hasTMAccessionNumbers": "Y",
    "hasDbCrossReferences": "N", "hasSuppl": "Y", "firstPublicationDate": "2018-02-23",
}


def test_a_new_record_carries_identity_server_and_summary():
    row = bi.epmc_record_to_row(RECORD)
    assert row["epmc_source"] == "PPR" and row["epmc_id"] == "PPR18364"
    assert row["preprint_server"] == "bioRxiv"
    assert row["journal"] == ""              # PPR records have no journalTitle, by design
    assert row["has_data"] == "Y"
    assert json.loads(row["data_links_tags"]) == ["supporting_data", "altmetrics"]
    assert json.loads(row["accession_types"]) == ["pdb"]
    assert json.loads(row["db_cross_references"]) == []
    assert (row["has_tm_accessions"], row["has_db_xrefs"], row["has_suppl"]) == ("Y", "N", "Y")


def test_the_row_has_every_output_column_and_nothing_else():
    assert set(bi.epmc_record_to_row(RECORD)) == set(bi.OUTPUT_COLUMNS)


def test_the_forward_and_retrospective_readers_are_the_same_code():
    """A backfilled document and a newly loaded one must read the same Europe PMC record the same
    way; both go through fetch_epmc_metadata's two readers."""
    row = bi.epmc_record_to_row(RECORD)
    meta = fm.record_to_row("ppr_doi", "10.1101/270413", RECORD, "now")
    for column in ("epmc_source", "epmc_id", "preprint_server", *fm.SUMMARY_COLUMNS):
        assert row[column] == meta[column]


def test_a_medline_record_has_no_server_and_n_flags():
    row = bi.epmc_record_to_row({"id": "1", "source": "MED", "pmid": "1", "title": "t",
                                 "journalTitle": "Nature"})
    assert row["preprint_server"] == "" and row["epmc_id"] == "1"
    assert row["has_data"] == "N" and row["data_links_tags"] == "[]"
