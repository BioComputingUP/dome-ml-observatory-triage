"""Tests for the resource catalogue: every spelling Europe PMC uses lands on one slug."""

from __future__ import annotations

import datalinks_resources as r


def test_each_route_spelling_of_a_resource_lands_on_one_slug():
    assert r.slug("PDBe") == r.slug("pdb") == r.slug("PDB", "Europe PMC") == "pdb"
    assert r.slug("gen") == r.slug("ENA") == r.slug("GenBank") == "ena"
    assert r.slug("nct") == r.slug("ClinicalTrials.gov") == "clinicaltrials"
    assert r.slug("pxd") == r.slug("PRIDE") == "pride"


def test_a_parenthesised_name_matches_on_either_part():
    assert r.slug("Gene Ontology (GO)") == "go"
    assert r.slug("European Nucleotide Archive (ENA)") == "ena"


def test_an_unknown_scheme_keeps_a_clean_slug_rather_than_being_dropped():
    assert r.slug("NewDB (NDB)") == "newdb"
    assert r.describe("newdb", "NewDB").category == r.OTHER


def test_a_doi_is_a_resource_only_by_a_data_repository_prefix():
    assert r.slug("DOI", None, "10.5061/dryad.dm57j") == "dryad"
    assert r.slug("DOI", None, "10.5281/zenodo.123") == "zenodo"
    assert r.slug("DOI", None, "10.1038/s41586-021-03819-2") is None
    assert r.slug("DOI", "figshare", "10.9999/x") == "figshare"   # a publisher Europe PMC vouches for


def test_identifiers_org_uris_yield_the_scheme():
    assert r.scheme_from_uri("http://identifiers.org/pdbe/pdb:6w6w") == "pdbe"
    assert r.scheme_from_uri("http://identifiers.org/geo:GSE109308") == "geo"
    assert r.scheme_from_uri("http://identifiers.org/ebi/ena.embl:AY278488") == "ena.embl"
    assert r.slug(r.scheme_from_uri("http://identifiers.org/ebi/ena.embl:AY278488")) == "ena"


def test_reference_sections_are_recognised():
    assert r.is_reference_section("References (http://purl.org/orb/References)")
    assert not r.is_reference_section("Article (http://semanticscience.org/resource/SIO_001029)")


def test_every_catalogue_alias_points_at_a_catalogue_entry():
    assert set(r.ALIASES.values()) <= set(r.RESOURCES)
    assert set(r.DOI_PREFIXES.values()) <= set(r.RESOURCES)


def test_supplementary_file_links_to_a_resource_site_are_typed_by_host():
    assert r.slug(r.scheme_from_uri("http://gisaid.org/EPI_ISL/402124")) == "gisaid"
    assert r.slug(r.scheme_from_uri("https://omim.org/entry/620113")) == "omim"
    assert r.slug(r.scheme_from_uri("https://www.proteinatlas.org/search/HPA027524")) == "hpa"
    assert r.slug(r.scheme_from_uri("https://www.ebi.ac.uk/pdbe/entry/pdb/6vw1")) == "pdb"
    assert r.scheme_from_uri("https://example.org/whatever/1") is None


def test_every_host_scheme_is_a_known_alias():
    assert {scheme for _, scheme in r.HOST_SCHEMES} <= set(r.ALIASES)


def test_every_spelling_seen_in_the_corpus_build_maps_to_the_catalogue():
    seen = ["IGSR/1000 Genomes", "BioSamples", "coriell", "px", "chembl.compound", "insdc.gca",
            "ega.study", "ega.dataset", "euclinicaltrials", "EBI Metagenomics", "biomodels.db"]
    for spelling in seen:
        assert r.slug(spelling) in r.RESOURCES, spelling


def test_the_shared_embl_ebi_doi_prefix_is_split_by_suffix():
    assert r.slug("DOI", None, "10.6019/PXD024968") == "pride"
    assert r.slug("DOI", None, "10.6019/EMPIAR-10164") == "empiar"
    assert r.slug("DOI", None, "10.6019/something-else") == "doi"
    assert {res for rules in r.DOI_SUFFIX_RULES.values() for _, res in rules} <= set(r.RESOURCES)


def test_empiar_and_emdb_pages_are_not_typed_as_pdb():
    assert r.slug(r.scheme_from_uri("https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10164")) == "empiar"
    assert r.slug(r.scheme_from_uri("https://www.ebi.ac.uk/emdb/EMD-1234")) == "emdb"
    assert r.slug(r.scheme_from_uri("https://www.ebi.ac.uk/pdbe/entry/pdb/6vw1")) == "pdb"
