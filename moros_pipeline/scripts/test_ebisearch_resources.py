"""Tests for the EBI Search accept list: what is in scope, where each accession is filed, and the
URLs a link may carry."""

from __future__ import annotations

import pytest

import datalinks_resources as dr
import ebisearch_resources as er
from link_identifiers import problems, url_problems


def test_every_accepted_domain_files_under_a_catalogue_resource():
    assert {d.slug for d in er.DOMAINS.values()} <= set(dr.RESOURCES)
    assert {r.slug for r in er.ID_RULES} <= set(dr.RESOURCES)
    assert {d.kind for d in er.DOMAINS.values()} == {er.DEPOSIT, er.REGISTRY, er.SUPPLEMENTARY}


def test_the_registries_and_the_supplementary_entries_are_accepted():
    assert er.DOMAINS["biotools"].kind == er.REGISTRY
    assert er.DOMAINS["dome-registry"].kind == er.REGISTRY
    assert er.DOMAINS["biostudies-literature"].kind == er.SUPPLEMENTARY


@pytest.mark.parametrize("domain", [
    "uniprot", "uniprot-covid19", "proteomes", "pdbekb", "interpro7_family", "interpro7_domain",
    "go", "efo", "intact-interactions", "complex-portal", "reactome", "rhea", "intenz", "chebi",
    "chembl-document", "gwas_catalog", "hgnc", "g2p", "omim", "cellosaurus", "ensemblGenomes_gene",
    "sc-genes", "rnacentral", "rfam", "biomodels_autogen", "peptide_atlas", "gpmdb", "paxdb",
    "lincs", "imgt-hla", "mesh", "ebiweb_training_online", "atlas-experiments", "geo_datasets",
])
def test_curation_domains_and_derived_experiments_are_not_accepted(domain):
    assert domain not in er.DOMAINS


def test_xref_and_dump_domains_partition_the_accept_list():
    assert er.XREF_DOMAINS <= set(er.DOMAINS)
    assert not (er.XREF_DOMAINS & er.DUMP_DOMAINS)
    assert er.XREF_DOMAINS | er.DUMP_DOMAINS == set(er.DOMAINS)


@pytest.mark.parametrize("raw, expected", [
    ("E-GEOD-38402", ("geo", "GSE38402")),
    ("phs000310.v1.p1", ("dbgap", "phs000310")),
    ("phs000209.v10.p2", ("dbgap", "phs000209")),
    ("phs000310", ("dbgap", "phs000310")),
    ("MODEL2406030003", ("biomodels", "MODEL2406030003")),
    ("BIOMD0000001076", ("biomodels", "BIOMD0000001076")),
    ("EMPIAR-10484", ("empiar", "EMPIAR-10484")),
    ("S-BIAD32", ("bioimage_archive", "S-BIAD32")),
    ("S-EPMC3585919", ("biostudies", "S-EPMC3585919")),
    ("MSV000080757", ("massive", "MSV000080757")),
    ("MTBLS688", ("metabolights", "MTBLS688")),
])
def test_an_accession_is_filed_under_its_home_resource_whatever_listed_it(raw, expected):
    assert er.canonicalise("biostudies", raw) == expected


def test_an_accession_no_rule_knows_keeps_its_domains_resource():
    assert er.canonicalise("bioproject", "PRJNA167815") == ("bioproject", "PRJNA167815")
    assert er.canonicalise("eva", "PRJEB100784") == ("eva", "PRJEB100784")
    assert er.canonicalise("arrayexpress", "E-MTAB-1898") == ("arrayexpress", "E-MTAB-1898")
    assert er.canonicalise("biostudies", "S-SCDT-EMM-2019-10431") == (
        "biostudies", "S-SCDT-EMM-2019-10431")


# Real ids from the 2026-09-14 dumps and detail records, one per resource.
SAMPLE_IDS = {
    "biotools": "suba3", "dome_registry": "3mm086r5pw", "bioproject": "PRJNA167815",
    "ena": "SRP013477", "eva": "PRJEB100784", "geo": "GSE40830", "arrayexpress": "E-MTAB-1898",
    "expression_atlas": "E-CURD-100", "pride": "PXD013455", "iprox": "PXD008846",
    "jpost": "PXD017956", "panorama": "PXD014970", "massive": "MSV000080757",
    "metabolights": "MTBLS688", "pdb": "8tua", "emdb": "EMD-8592", "empiar": "EMPIAR-10484",
    "bioimage_archive": "S-BIAD32", "biostudies": "S-SCDT-EMM-2019-10431",
    "biomodels": "MODEL2406030003", "ega": "EGAD00001003698", "dbgap": "phs000310",
    "node": "OEX00010562",
}


@pytest.mark.parametrize("slug, sample", sorted(SAMPLE_IDS.items()))
def test_every_url_template_yields_a_clean_url_for_a_real_id(slug, sample):
    assert not problems(sample, slug)
    url = er.entry_url(slug, sample)
    assert url and not url_problems(url) and sample in url


def test_every_resource_the_route_can_produce_has_its_url_rule_decided():
    decided = set(er.URL_TEMPLATES) | {"ega"} | er.NO_URL_TEMPLATE
    produced = {d.slug for d in er.DOMAINS.values()} | {r.slug for r in er.ID_RULES}
    assert produced <= decided
    assert produced <= set(SAMPLE_IDS) | er.NO_URL_TEMPLATE


def test_ega_studies_and_datasets_have_different_pages_and_some_resources_have_none():
    assert er.entry_url("ega", "EGAS00001006372") == "https://ega-archive.org/studies/EGAS00001006372"
    assert er.entry_url("ega", "EGAD00001003698") == "https://ega-archive.org/datasets/EGAD00001003698"
    for slug in er.NO_URL_TEMPLATE:
        assert er.entry_url(slug, "526") is None


def test_the_relationship_says_what_the_entry_is_to_the_paper():
    assert er.relationship_for(er.DOMAINS["dome-registry"]) == "IsReviewedBy"
    assert er.relationship_for(er.DOMAINS["biotools"]) == "IsDescribedBy"
    assert er.relationship_for(er.DOMAINS["sra-study"]) == "IsSupplementedBy"
    assert er.relationship_for(er.DOMAINS["biostudies-literature"]) == "IsSupplementedBy"


def test_a_link_a_rule_moves_takes_its_new_resources_url_and_an_untouched_one_is_kept():
    mined = {"resource": "arrayexpress", "id": "E-GEOD-1", "obtained_by": "tm_accession",
             "url": "http://identifiers.org/arrayexpress:E-GEOD-1"}
    moved = er.canonicalise_link(mined)
    assert (moved["resource"], moved["id"]) == ("geo", "GSE1")
    assert moved["url"] == "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE1"
    assert mined["resource"] == "arrayexpress"                 # the input is not mutated
    untouched = {"resource": "pdb", "id": "6VW1", "url": "x"}
    assert er.canonicalise_link(untouched) is untouched


def test_a_pxd_is_filed_under_the_partner_ebi_search_says_hosts_it():
    mined = {"resource": "pride", "id": "PXD008846", "url": "u0", "source_domain": None}
    hosted = {"resource": "iprox", "id": "PXD008846", "url": "u1", "source_domain": "iprox"}
    unclaimed = {"resource": "pride", "id": "PXD013455", "url": "u2", "source_domain": None}
    out = er.resolve_pxd_hosts([mined, None, hosted, unclaimed])
    assert [link["resource"] if link else None for link in out] == ["iprox", None, "iprox", "pride"]
    assert out[0]["url"] == "https://proteomecentral.proteomexchange.org/cgi/GetDataset?ID=PXD008846"
    assert out[3] is unclaimed


def test_pride_wins_when_two_partners_claim_one_pxd():
    links = [{"resource": "jpost", "id": "PXD1", "url": "a", "source_domain": "jpost"},
             {"resource": "pride", "id": "PXD000001", "url": "b", "source_domain": "pride"},
             {"resource": "jpost", "id": "PXD000001", "url": "c", "source_domain": "jpost"}]
    assert [link["resource"] for link in er.resolve_pxd_hosts(links)] == ["jpost", "pride", "pride"]


def test_only_geo_has_a_browse_page_until_the_ebi_search_ui_is_checked():
    assert er.browse_url("geo", "23284283") == (
        "https://www.ncbi.nlm.nih.gov/gds?LinkName=pubmed_gds&from_uid=23284283")
    assert er.browse_url("geo", "") is None and er.browse_url("geo", None) is None
    assert er.browse_url("ena", "23184988") is None
