"""Tests for build_release_metadata.py: the document it builds, what it refuses, and how it writes."""

from __future__ import annotations

import json
import re

import pytest

import build_release_metadata as brm

REPORT = {
    "run_id": "verify_corpus_20260915T190000",
    "finished_at": "2026-09-15T19:00:00.000000+00:00",
    "facts": {
        "total": 846716,
        "schema_version": {"1.6.0": 846716},
        "classification": {"positive": 366234, "negative": 473503, "undeterminable": 6979},
        "enriched": 3332,
        "data_links": {"with_resources": 311025},
    },
    "checks": [],
    "failures": [],
}
HASHES = {
    "classification": {"prompt_version": "v1", "criteria_sha256": "bd9d66dd"},
    "enrichment": {"prompt_version": "e1", "vocab_sha256": "41db952f"},
}
PEOPLE = [{"@id": "https://orcid.org/0000-0001-5166-8551", "@type": ["foaf:Person", "Person"],
           "name": "Gavin Farrell"}]


def build(**overrides):
    kwargs = dict(release="2026-09", schema_version="1.6.0", hashes=HASHES, query_sha256="abc123",
                  commit="deadbeef", people=PEOPLE, previous=None)
    kwargs.update(overrides)
    return brm.build_document(REPORT, **kwargs)


def nodes(document):
    return {node["@id"]: node for node in document["@graph"] if "@id" in node}


def test_the_graph_holds_catalogue_series_release_distribution_service_and_agents():
    graph = nodes(build())
    catalog = graph[brm.CATALOG_ID]
    series = graph[brm.SERIES_ID]
    release = graph[brm.release_id("2026-09")]
    assert catalog["@type"] == ["dcat:Catalog", "DataCatalog"]
    assert series["@type"] == ["dcat:DatasetSeries", "Dataset"]
    assert release["@type"] == ["dcat:Dataset", "Dataset"]
    assert graph[brm.EXPORT_DISTRIBUTION_ID]["@type"] == ["dcat:Distribution", "DataDownload"]
    assert graph[brm.API_ID]["@type"] == ["dcat:DataService", "WebAPI"]
    assert graph[brm.PUBLISHER_ID]["name"] == "BioComputingUP, University of Padua"
    assert "https://orcid.org/0000-0001-5166-8551" in graph
    # Pages on the site (the landing page, the export URL) are plain IRIs; every fragment identifier
    # on the site is a node, and each one referenced is described in this graph.
    referenced = set(re.findall(r'"@id": "([^"]+#[^"]+)"', json.dumps(build())))
    assert {i for i in referenced if i.startswith(brm.ORIGIN)} <= set(graph)
    assert {i for i in graph if i.startswith(brm.ORIGIN)} == {
        brm.CATALOG_ID, brm.SERIES_ID, brm.release_id("2026-09"), brm.EXPORT_DISTRIBUTION_ID,
        brm.API_ID, brm.CONTACT_ID}


def test_the_release_carries_the_counts_schema_provenance_and_licence_split():
    release = nodes(build())[brm.release_id("2026-09")]
    assert release["version"] == release["dcat:version"] == "2026-09"
    assert release["dct:issued"] == {"@value": "2026-09-15", "@type": "xsd:date"}
    assert release["size"]["value"] == 846716
    assert "366,234 of them classified as AI/ML methods papers" in release["description"]
    assert {"@id": brm.schema_release_url("1.6.0")} in release["dct:conformsTo"]
    assert release["license"] == brm.CC_BY_4 and "Europe PMC" in release["dct:rights"]
    used = release["prov:wasGeneratedBy"]["prov:used"]
    assert [u["identifier"] for u in used] == ["sha256:bd9d66dd", "sha256:41db952f", "sha256:abc123"]
    assert release["prov:wasGeneratedBy"]["prov:wasAssociatedWith"]["version"] == "deadbeef"


def test_no_zenodo_distribution_or_doi_until_the_archive_job_adds_one():
    text = json.dumps(build()).lower()
    assert "zenodo" not in text and "doi.org" not in text


def test_a_later_month_links_back_to_the_previous_one():
    assert "dcat:prev" not in nodes(build())[brm.release_id("2026-09")]
    later = nodes(build(release="2026-10", previous="2026-09"))[brm.release_id("2026-10")]
    assert later["dcat:prev"] == {"@id": brm.release_id("2026-09")}


def test_a_malformed_release_is_refused():
    with pytest.raises(ValueError):
        build(release="2026-9")


def test_a_report_with_failures_or_a_corpus_mid_migration_is_refused(tmp_path):
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(REPORT))
    assert brm.load_report(ok, "1.6.0")["facts"]["total"] == 846716

    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({**REPORT, "failures": ["licence gap is closed"]}))
    with pytest.raises(SystemExit, match="failed invariants"):
        brm.load_report(failed, "1.6.0")

    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps({**REPORT, "facts": {**REPORT["facts"],
                                                     "schema_version": {"1.5.1": 10, "1.6.0": 846706}}}))
    with pytest.raises(SystemExit, match="migrate"):
        brm.load_report(mixed, "1.6.0")


def test_writing_publishes_the_month_moves_current_and_refuses_to_rewrite_it(tmp_path):
    target = brm.write_release(build(), tmp_path, "2026-09", overwrite=False)
    assert target == tmp_path / "metadata" / "releases" / "2026-09" / "dataset.jsonld"
    assert json.loads(target.read_text())["@graph"][0]["@id"] == brm.CATALOG_ID
    assert (tmp_path / "metadata" / "CURRENT").read_text() == "2026-09\n"

    with pytest.raises(SystemExit, match="immutable"):
        brm.write_release(build(), tmp_path, "2026-09", overwrite=False)
    brm.write_release(build(), tmp_path, "2026-09", overwrite=True)

    brm.write_release(build(release="2026-10"), tmp_path, "2026-10", overwrite=False)
    assert (tmp_path / "metadata" / "CURRENT").read_text() == "2026-10\n"
    assert brm.previous_release(tmp_path, "2026-10") == "2026-09"
    # A backfilled earlier month does not wind CURRENT back.
    brm.write_release(build(release="2026-08"), tmp_path, "2026-08", overwrite=False)
    assert (tmp_path / "metadata" / "CURRENT").read_text() == "2026-10\n"


def test_creators_come_from_citation_cff():
    people = brm.creators()
    assert people and all(p["name"] for p in people)
    assert people[0]["@id"].startswith("https://orcid.org/")
