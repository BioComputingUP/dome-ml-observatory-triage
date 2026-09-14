"""Tests for the one definition of a clean data-link identifier.

Every raw string below was seen in the 2026-09-14 corpus build, stored verbatim, and produced a link
that 404'd or pointed at nothing.
"""

from __future__ import annotations

import re

import pytest

import link_identifiers as li


@pytest.mark.parametrize("raw, expected", [
    ("10.5281/zenodo.18675888.", "10.5281/zenodo.18675888"),
    ("10.5281/zenodo.20657011,", "10.5281/zenodo.20657011"),
    ("10.17632/bcmn9cxyzs.4]", "10.17632/bcmn9cxyzs.4"),
    ("10.7910/DVN/BKP1RH],", "10.7910/DVN/BKP1RH"),
    ("10.5281/zenodo.17121940].", "10.5281/zenodo.17121940"),
    ("10.6084/m9.figshare.24123303”.", "10.6084/m9.figshare.24123303"),
    ("10.5061/dryad.040h9t7.\xa0Behavioural", "10.5061/dryad.040h9t7"),
    ("10.5061/dryad.1k84r），该数据集已经公开，可用于科学研究。NinaPro", "10.5061/dryad.1k84r"),
    ("10.21227/mps8-kb56”\xa0[", "10.21227/mps8-kb56"),
    ("10.5061/dryad.k6djh9w3c\n", "10.5061/dryad.k6djh9w3c"),
    ("10.15468/dl.qjuw4q;", "10.15468/dl.qjuw4q"),
    ("10.17605/OSF.IO/T584 M.", "10.17605/OSF.IO/T584"),
    ("10.5281/zenodo.1</a>", "10.5281/zenodo.1"),
    # balanced brackets inside an identifier are part of it
    ("10.1002/(SICI)1097-4636(199709)36:4<471::AID-JBM4>3.0.CO;2-G",
     "10.1002/(SICI)1097-4636(199709)36:4<471::AID-JBM4>3.0.CO;2-G"),
    ("GO:0002376", "GO:0002376"),
])
def test_clean_cuts_the_identifier_out_of_the_text_around_it(raw, expected):
    assert li.clean(raw) == expected


def test_doi_candidates_try_the_repaired_forms_in_order():
    assert li.doi_candidates("10.5281/zenodo.18675888.") == ["10.5281/zenodo.18675888"]
    assert li.doi_candidates("see 10.5281/zenodo.18675888.") == ["10.5281/zenodo.18675888"]
    assert li.doi_candidates("10.7910/DVN/UFC6B5,HarvardDataverse,V2") == [
        "10.7910/DVN/UFC6B5,HarvardDataverse,V2", "10.7910/DVN/UFC6B5"]
    assert li.doi_candidates("10.34740/kaggle/ds/6,979,862") == [
        "10.34740/kaggle/ds/6,979,862", "10.34740/kaggle/ds/6979862", "10.34740/kaggle/ds/6"]


def test_pdf_glyphs_are_repaired_and_the_truncation_is_never_offered():
    assert li.doi_candidates("10.17632/trvb5k4×5m.1") == ["10.17632/trvb5k4x5m.1"]
    assert li.doi_candidates("10.4121/96303227-5886-41c9–8607-70fdd2cfe7c1.v1") == [
        "10.4121/96303227-5886-41c9-8607-70fdd2cfe7c1.v1"]
    candidates = li.doi_candidates("10.17632/k82\xa0×\xa07czd87.1Direct")
    assert "10.17632/k82" not in candidates


def test_placeholders_and_non_dois_yield_no_candidates():
    assert li.doi_candidates("10.5281/zenodo.XXXXX") == []
    assert li.doi_candidates("GSE109308") == []
    assert li.doi_candidates("") == []


@pytest.mark.parametrize("clean_id, resource", [
    ("10.5281/zenodo.18675888", "zenodo"),
    ("10.1002/(SICI)1097-4636(199709)36:4<471::AID-JBM4>3.0.CO;2-G", None),
    ("GO:0002376", "go"),
    ("RRID:AB_1658454", "rrid"),
    ("RRID:MGI:2159769", "rrid"),
    ("EPI_ISL_6841980", "gisaid"),
    ("S-EPMC8371605", "biostudies"),
    ("NM_000546.6", "refseq"),
    ("6VW1", "pdb"),
])
def test_clean_identifiers_have_no_problems(clean_id, resource):
    assert li.problems(clean_id, resource) == []
    assert not re.search(li.MONGO_MALFORMED_REGEX, clean_id)


@pytest.mark.parametrize("bad_id, resource, reason", [
    ("", None, "empty"),
    ("10.5281/zenodo.18675888.", "zenodo", "trailing punctuation"),
    ("10.17632/bcmn9cxyzs.4]", "mendeley_data", "trailing punctuation"),
    ("(odc-tbi.org)", "rrid", "leading punctuation"),
    ("10.17632/trvb5k4×5m.1", "mendeley_data", "whitespace or non-ASCII"),
    ("RRID: AB_1658454", "rrid", "whitespace or non-ASCII"),
    ("ORPHA 401777", "orphanet", "whitespace or non-ASCII"),
    ("10.1234/abc)", None, "unbalanced brackets"),
    ("10.12/abc", None, "not a DOI"),
    ("10.5281/zenodo.XXXXX", "zenodo", "placeholder"),
    ("GraphPad:Prism", "rrid", "not an RRID"),
    ("106173", "rrid", "not an RRID"),
])
def test_malformed_identifiers_are_named(bad_id, resource, reason):
    assert reason in li.problems(bad_id, resource)


@pytest.mark.parametrize("bad_id", [
    "10.5281/zenodo.18675888.", "10.5281/zenodo.20657011,", "10.17632/bcmn9cxyzs.4]",
    "(odc-tbi.org)", "10.17632/trvb5k4×5m.1", "RRID: AB_1658454", "10.21227/mps8-kb56”\xa0[",
    "'quoted'",
])
def test_the_mongo_regex_catches_every_punctuation_whitespace_and_non_ascii_case(bad_id):
    """verify_corpus.py counts documents with this regex; it must agree with problems() on the
    classes it can express."""
    assert re.search(li.MONGO_MALFORMED_REGEX, bad_id)
    assert li.problems(bad_id)


def test_normalise_prefixed():
    assert li.normalise_prefixed("RRID: AB_1658454") == "RRID:AB_1658454"
    assert li.normalise_prefixed("ORPHA 401777") == "ORPHA:401777"
    assert li.normalise_prefixed("ZFIN:\xa0ZDB-GENO-060207-1") == "ZFIN:ZDB-GENO-060207-1"
    assert li.normalise_prefixed("GO:0002376") == "GO:0002376"


def test_resolver_local_id():
    assert li.resolver_local_id("https://scicrunch.org/resolver/RRID:AB_2535812") == "RRID:AB_2535812"
    assert li.resolver_local_id("http://identifiers.org/orphanet:401777") == "401777"
    assert li.resolver_local_id("http://identifiers.org/pdbe/pdb:6w6w") == "6w6w"
    assert li.resolver_local_id("http://identifiers.org/ebi/ena.embl:AY278488") == "AY278488"
    assert li.resolver_local_id("http://identifiers.org/doi:10.5281/zenodo.1") == "10.5281/zenodo.1"
    assert li.resolver_local_id("http://gisaid.org/EPI_ISL/1") is None
    assert li.resolver_local_id(None) is None


def test_canonical_url():
    # a dirty Europe PMC URL for a DOI is replaced by doi.org
    assert li.canonical_url("zenodo", "10.5281/zenodo.18675888",
                            "http://identifiers.org/doi:10.5281/zenodo.18675888.") == \
        "https://doi.org/10.5281/zenodo.18675888"
    # a clean one that ends with the DOI is kept
    assert li.canonical_url("zenodo", "10.5281/zenodo.1", "http://identifiers.org/doi:10.5281/zenodo.1") == \
        "http://identifiers.org/doi:10.5281/zenodo.1"
    # a clean one that names a different string (Europe PMC's comma split) is not
    assert li.canonical_url("dataverse", "10.7910/DVN/UFC6B5",
                            "http://identifiers.org/doi:10.7910/DVN/UFC6B5|acc:HarvardDataverse") == \
        "https://doi.org/10.7910/DVN/UFC6B5"
    assert li.canonical_url("rrid", "RRID:AB_1658454", None) == \
        "https://scicrunch.org/resolver/RRID:AB_1658454"
    assert li.canonical_url("rrid", "RRID:AB_2535812", "https://scicrunch.org/resolver/RRID:AB_2535812") == \
        "https://scicrunch.org/resolver/RRID:AB_2535812"
    # an identifiers.org URL gets its local part cleaned
    assert li.canonical_url("pdb", "6VW1", "http://identifiers.org/pdbe/pdb:6VW1.") == \
        "http://identifiers.org/pdbe/pdb:6VW1"
    # a clean site URL is kept even when it spells the id differently; a dirty unknown one is dropped
    assert li.canonical_url("gisaid", "EPI_ISL_1", "http://gisaid.org/EPI_ISL/1") == "http://gisaid.org/EPI_ISL/1"
    assert li.canonical_url("gisaid", "EPI_ISL_1", "http://gisaid.org/EPI_ISL/1,") is None
    assert li.canonical_url("dryad", "10.5061/dryad.k6djh9w3c", "http://identifiers.org/doi:10.5061/dryad.k6djh9w3c\n") == \
        "http://identifiers.org/doi:10.5061/dryad.k6djh9w3c"


def test_malformed_links_checks_ids_and_urls():
    detail = {"links": [
        {"resource": "pdb", "id": "6VW1", "url": "http://identifiers.org/pdbe/pdb:6VW1"},
        {"resource": "zenodo", "id": "10.5281/zenodo.18675888.", "url": "https://doi.org/10.5281/zenodo.18675888"},
        {"resource": "pdb", "id": "6W6W", "url": "http://identifiers.org/pdbe/pdb:6W6W."},
        {"resource": "biostudies", "id": "S-EPMC1", "url": None},
    ]}
    found = li.malformed_links(detail)
    assert [(res, link_id) for res, link_id, _, _ in found] == [
        ("zenodo", "10.5281/zenodo.18675888."), ("pdb", "6W6W")]
    assert found[1][3] == ["url trailing punctuation"]
    assert li.malformed_links(None) == [] and li.malformed_links({"links": []}) == []
