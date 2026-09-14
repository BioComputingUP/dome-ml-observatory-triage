"""Tests for the data-links merge: resource mapping, dedupe, caps, derivation, and the rule that
a half-fetched record is never written as complete."""

from __future__ import annotations

import csv
import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import build_data_links as bd


def _ann(exact, sub_type=None, uri=None, section="Article", provider="Europe PMC", **extra):
    return {"exact": exact, "uri": uri, "sub_type": sub_type, "provider": provider,
            "section": section, "frequency": None, "file_name": None, **extra}


# doi.org verdicts for the DOIs the older tests use (lowercased, as the build keys them).
HANDLES = {"10.5061/dryad.dm57j": True, "10.5281/zenodo.1": True, "10.9999/x": True}


# -- one link from each route ---------------------------------------------------------------


def test_a_text_mined_pdb_accession_becomes_a_pdb_link():
    stats = bd.Stats()
    link = bd.link_from_annotation(_ann("6VW1", "PDBe", "http://identifiers.org/pdbe/pdb:6VW1"), stats)
    assert link["resource"] == "pdb" and link["id"] == "6VW1"
    assert link["obtained_by"] == "tm_accession" and link["section"] == "Article"


def test_a_supplementary_file_accession_is_typed_from_its_uri():
    link = bd.link_from_annotation(
        _ann("6w6w", None, "http://identifiers.org/pdbe/pdb:6w6w", "Supplementary material",
             provider="Biostudies"), bd.Stats())
    assert link["resource"] == "pdb" and link["obtained_by"] == "tm_supplementary"


def test_a_reference_list_doi_is_a_citation_not_data():
    stats = bd.Stats()
    assert bd.link_from_annotation(_ann("10.1038/s41586-020-2951-z", "DOI", section="References"),
                                   stats) is None
    assert stats.reference_dois == 1


def test_a_data_repository_doi_maps_by_prefix_and_an_unknown_one_is_tabulated():
    stats = bd.Stats()
    dryad = bd.link_from_annotation(_ann("10.5061/dryad.dm57j", "DOI"), stats, HANDLES)
    assert dryad["resource"] == "dryad" and dryad["url"] == "https://doi.org/10.5061/dryad.dm57j"
    assert bd.link_from_annotation(_ann("10.1234/journal.1", "DOI"), stats) is None
    assert stats.unmapped_doi_prefix == {"10.1234": 1}


def test_a_scholix_link_uses_scheme_then_publisher():
    stats = bd.Stats()
    ena = bd.link_from_datalink({"id": "AY278488", "id_scheme": "ENA", "publisher": "Europe PMC",
                                 "url": "http://identifiers.org/ebi/ena.embl:AY278488",
                                 "obtained_by": "tm_accession", "relationship": "References",
                                 "category": "Nucleotide Sequences"}, stats)
    assert ena["resource"] == "ena"
    zen = bd.link_from_datalink({"id": "10.5281/zenodo.1", "id_scheme": "DOI", "publisher": "Zenodo",
                                 "url": None, "obtained_by": "ext_links",
                                 "relationship": "IsSupplementedBy", "title": "Source data",
                                 "category": "Data Citations"}, stats, HANDLES)
    assert zen["resource"] == "zenodo" and zen["url"] == "https://doi.org/10.5281/zenodo.1"
    # a DOI Europe PMC vouches for but whose prefix and publisher we do not know stays generic
    other = bd.link_from_datalink({"id": "10.9999/x", "id_scheme": "DOI", "publisher": "NewRepo",
                                   "category": "Data Citations"}, stats, HANDLES)
    assert other["resource"] == "newrepo"


def test_biostudies_is_derived_from_the_pmcid():
    link = bd.derived_biostudies("PMC8371605")
    assert link["id"] == "S-EPMC8371605"
    assert link["url"] == "https://www.ebi.ac.uk/biostudies/studies/S-EPMC8371605"
    assert link["obtained_by"] == "derived"
    assert bd.derived_biostudies("") is None and bd.derived_biostudies("PMCx") is None


# -- assembling -------------------------------------------------------------------------------


def test_overlapping_routes_are_deduplicated_and_summarised():
    stats = bd.Stats()
    links = [
        bd.link_from_annotation(_ann("6VW1", "PDBe", "http://identifiers.org/pdbe/pdb:6VW1"), stats),
        bd.link_from_datalink({"id": "6vw1", "id_scheme": "PDB", "publisher": "Europe PMC",
                               "url": None, "title": "Spike RBD", "category": "Protein Structures"}, stats),
        bd.link_from_annotation(_ann("GSE1", "GEO", "http://identifiers.org/geo:GSE1"), stats),
    ]
    detail = bd.assemble(links, ["epmc_annotations", "epmc_datalinks"], "2026-09-14T00:00:00+00:00")
    assert detail["link_count"] == 2 and detail["truncated"] is False
    assert [r["resource"] for r in detail["resources"]] == ["geo", "pdb"]
    pdb = next(r for r in detail["resources"] if r["resource"] == "pdb")
    assert pdb == {"resource": "pdb", "label": "Protein Data Bank in Europe",
                   "category": "Protein Structures", "id_scheme": "PDBe", "publisher": "Europe PMC",
                   "obtained_by": "tm_accession", "count": 1}
    pdb_link = next(l for l in detail["links"] if l["resource"] == "pdb")
    assert pdb_link["title"] == "Spike RBD"          # the later route filled the missing title
    assert set(pdb_link) == set(bd.LINK_KEYS)          # no private keys leak into the document


def test_caps_keep_true_counts():
    stats = bd.Stats()
    links = [bd.link_from_annotation(_ann(f"{i}ABC", "PDBe"), stats) for i in range(80)]
    detail = bd.assemble(links, ["epmc_annotations"], "now")
    assert detail["link_count"] == 80
    assert detail["resources"][0]["count"] == 80
    assert len(detail["links"]) == bd.MAX_LINKS_PER_RESOURCE and detail["truncated"] is True


# -- the build, end to end on tiny files ------------------------------------------------------


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def _meta(key_type, key, source, ext_id, **over):
    row = {"key_type": key_type, "key": key, "epmc_source": source, "epmc_id": ext_id,
           "pmid": "", "pmcid": "", "doi": "", "preprint_server": "", "has_data": "Y",
           "data_links_tags": '["supporting_data"]', "accession_types": '["pdb"]',
           "db_cross_references": "[]", "has_tm_accessions": "Y", "has_db_xrefs": "N",
           "has_suppl": "N", "fetched_at": "2026-09-14T00:00:00+00:00"}
    row.update(over)
    return row


def test_the_build_writes_both_files_and_holds_back_half_fetched_records(tmp_path):
    keys = tmp_path / "keys.csv"
    _write_csv(keys, [
        {"pid": "p-pre", "pmid": "3157", "pmcid": "", "doi": "10.1101/270413", "is_preprint": "True"},
        {"pid": "p-med", "pmid": "30001", "pmcid": "PMC1", "doi": "", "is_preprint": "False"},
        {"pid": "p-wait", "pmid": "30002", "pmcid": "", "doi": "", "is_preprint": "False"},
        {"pid": "p-none", "pmid": "30003", "pmcid": "", "doi": "", "is_preprint": "False"},
    ])
    metadata = tmp_path / "meta.csv"
    _write_csv(metadata, [
        _meta("ppr_doi", "10.1101/270413", "PPR", "PPR18364", preprint_server="bioRxiv"),
        _meta("pmid", "30001", "MED", "30001", pmcid="PMC1", has_suppl="Y",
              has_db_xrefs="Y", db_cross_references='["PDB"]'),
        _meta("pmid", "30002", "MED", "30002"),
        _meta("pmid", "30003", "MED", "30003", has_data="N", data_links_tags="[]",
              accession_types="[]", has_tm_accessions="N"),
    ])
    ann = tmp_path / "ann.jsonl"
    _write_jsonl(ann, [
        {"source": "PPR", "id": "PPR18364", "fetched_at": "2026-09-14T01:00:00+00:00", "status": "ok",
         "annotations": [_ann("6VW1", "PDBe", "http://identifiers.org/pdbe/pdb:6VW1")]},
        {"source": "MED", "id": "30001", "fetched_at": "2026-09-14T01:00:00+00:00", "status": "absent",
         "annotations": []},
        # 30002 deliberately missing: its annotations have not been fetched yet
    ])
    dl = tmp_path / "dl.jsonl"
    _write_jsonl(dl, [
        {"source": "MED", "id": "30001", "fetched_at": "2026-09-14T02:00:00+00:00", "http_status": 200,
         "hit_count": 1, "links": [{"id": "1ABC", "id_scheme": "PDB", "publisher": "Europe PMC",
                                    "url": "http://identifiers.org/pdbe/pdb:1ABC",
                                    "obtained_by": "ext_links", "relationship": "References",
                                    "category": "Protein Structures", "title": None}]},
    ])
    out_p, out_d = tmp_path / "pid_preprints.csv", tmp_path / "pid_data_links.csv"
    stats = bd.build(keys, metadata, ann, dl, tmp_path / "none.jsonl", "residual", False, out_p, out_d)

    preprints = {r["pid"]: r for r in csv.DictReader(out_p.open())}
    assert preprints["p-pre"] == {"pid": "p-pre", "epmc_source": "PPR", "epmc_id": "PPR18364",
                                  "preprint_server": "bioRxiv"}
    assert preprints["p-med"]["preprint_server"] == "" and preprints["p-med"]["epmc_id"] == "30001"

    rows = {r["pid"]: r for r in csv.DictReader(out_d.open())}
    assert set(rows) == {"p-pre", "p-med", "p-wait", "p-none"}
    pre = json.loads(rows["p-pre"]["data_links_json"])
    assert pre["sources"] == ["epmc_annotations"] and pre["resources"][0]["resource"] == "pdb"
    assert pre["fetched_at"] == "2026-09-14T01:00:00+00:00"

    med = json.loads(rows["p-med"]["data_links_json"])
    assert med["sources"] == ["epmc_annotations", "epmc_datalinks", "derived"]
    assert {r["resource"] for r in med["resources"]} == {"pdb", "biostudies"}
    assert med["fetched_at"] == "2026-09-14T02:00:00+00:00"

    # targeted for annotations, none fetched yet: summary written, detail withheld
    assert rows["p-wait"]["has_data"] == "Y" and rows["p-wait"]["data_links_json"] == ""
    # nothing to fetch at all: resolved from the search record alone
    none = json.loads(rows["p-none"]["data_links_json"])
    assert none["sources"] == ["epmc_search"] and none["link_count"] == 0 and none["resources"] == []
    assert rows["p-none"]["has_data"] == "N"

    assert stats.counts["resolved"] == 3 and stats.counts["unresolved"] == 1
    assert stats.resources == {"pdb": 2, "biostudies": 1}


def test_scope_none_resolves_without_the_datalinks_route(tmp_path):
    keys = tmp_path / "keys.csv"
    _write_csv(keys, [{"pid": "p1", "pmid": "1", "pmcid": "", "doi": "", "is_preprint": "False"}])
    metadata = tmp_path / "meta.csv"
    _write_csv(metadata, [_meta("pmid", "1", "MED", "1", has_tm_accessions="N", has_db_xrefs="Y",
                                db_cross_references='["PDB"]')])
    out_p, out_d = tmp_path / "p.csv", tmp_path / "d.csv"
    missing = tmp_path / "missing.jsonl"
    held = bd.build(keys, metadata, missing, missing, missing, "residual", False, out_p, out_d)
    assert held.counts["unresolved"] == 1            # targeted for /datalinks, not answered
    done = bd.build(keys, metadata, missing, missing, missing, "none", False, out_p, out_d)
    assert done.counts["resolved"] == 1
    row = next(csv.DictReader(out_d.open()))
    assert json.loads(row["data_links_json"])["sources"] == ["epmc_search"]


def test_a_staged_batch_csv_serves_as_its_own_metadata(tmp_path):
    staged = tmp_path / "incoming_new.csv"
    _write_csv(staged, [{"pid": "p-new", "pmid": "2", "pmcid": "PMC2", "doi": "",
                         "pub_types": '["Journal Article"]', "epmc_source": "MED", "epmc_id": "2",
                         "preprint_server": "", "has_data": "Y", "data_links_tags": '["supporting_data"]',
                         "accession_types": "[]", "db_cross_references": "[]",
                         "has_tm_accessions": "N", "has_db_xrefs": "N", "has_suppl": "Y"}])
    out_p, out_d = tmp_path / "p.csv", tmp_path / "d.csv"
    missing = tmp_path / "missing.jsonl"
    stats = bd.build(staged, staged, missing, missing, missing, "residual", False, out_p, out_d)
    assert stats.counts["identified"] == 1 and stats.counts["resolved"] == 1
    detail = json.loads(next(csv.DictReader(out_d.open()))["data_links_json"])
    assert detail["resources"][0]["resource"] == "biostudies"
    assert detail["links"][0]["id"] == "S-EPMC2"


def test_sharding_changes_memory_not_output(tmp_path):
    """Same rows, same content, whatever the shard count; only the row order may differ."""
    keys = tmp_path / "keys.csv"
    rows = [{"pid": f"p{i}", "pmid": str(1000 + i), "pmcid": f"PMC{i}", "doi": "", "is_preprint": "False"}
            for i in range(40)]
    _write_csv(keys, rows)
    metadata = tmp_path / "meta.csv"
    _write_csv(metadata, [_meta("pmid", str(1000 + i), "MED", str(1000 + i), pmcid=f"PMC{i}",
                                has_suppl="Y" if i % 2 else "N") for i in range(40)])
    ann = tmp_path / "ann.jsonl"
    _write_jsonl(ann, [{"source": "MED", "id": str(1000 + i), "fetched_at": "2026-09-14T01:00:00+00:00",
                        "status": "ok", "annotations": [_ann(f"{i}ABC", "PDBe")]} for i in range(40)])
    missing = tmp_path / "missing.jsonl"
    outputs = {}
    for shards in (1, 7):
        out_p, out_d = tmp_path / f"p{shards}.csv", tmp_path / f"d{shards}.csv"
        stats = bd.build(keys, metadata, ann, missing, missing, "residual", False, out_p, out_d, shards)
        assert stats.counts["documents"] == 40 and stats.counts["resolved"] == 40
        outputs[shards] = ({r["pid"]: r for r in csv.DictReader(out_p.open())},
                           {r["pid"]: r for r in csv.DictReader(out_d.open())})
    assert outputs[1] == outputs[7]


def test_a_jsonl_line_identity_is_read_from_its_prefix():
    line = json.dumps({"source": "PPR", "id": "PPR1", "fetched_at": "now", "annotations": []})
    assert bd.line_identity(line) == ("PPR", "PPR1")
    assert bd.line_identity('{"id": "2", "source": "MED"}') == ("MED", "2")   # fallback parse
    assert bd.line_identity("not json") is None


def test_a_stored_publisher_on_a_non_preprint_never_reaches_pid_preprints(tmp_path):
    keys = tmp_path / "keys.csv"
    _write_csv(keys, [{"pid": "p-eth", "pmid": "", "pmcid": "", "doi": "10.1/thesis", "is_preprint": "False"}])
    metadata = tmp_path / "meta.csv"
    _write_csv(metadata, [_meta("doi", "10.1/thesis", "ETH", "ETH:1", preprint_server="University of Leeds",
                                has_tm_accessions="N")])
    out_p, out_d = tmp_path / "p.csv", tmp_path / "d.csv"
    missing = tmp_path / "missing.jsonl"
    stats = bd.build(keys, metadata, missing, missing, missing, "none", False, out_p, out_d)
    row = next(csv.DictReader(out_p.open()))
    assert (row["epmc_source"], row["preprint_server"]) == ("ETH", "")
    assert stats.counts["with_preprint_server"] == 0


def test_an_untyped_supplementary_accession_becomes_a_link_through_its_host():
    stats = bd.Stats()
    link = bd.link_from_annotation(_ann("620113", None, "https://omim.org/entry/620113",
                                        "Supplementary material", provider="Biostudies"), stats)
    assert (link["resource"], link["obtained_by"]) == ("omim", "tm_supplementary")
    assert stats.unmapped_scheme == {}


# -- identifier repair and the doi.org check (2026-09-14: 1,513 corpus documents had links that 404'd)


class _HandleResponse:
    def __init__(self, registered: bool):
        self.status_code = 200 if registered else 404
        self._payload = {"responseCode": 1 if registered else 100}

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _HandleSession:
    """Stands in for doi.org's handle API; records every DOI it was asked about."""

    def __init__(self, registered=(), fail=False):
        self.registered = {d.lower() for d in registered}
        self.fail = fail
        self.calls: list[str] = []

    def get(self, url, timeout=None):
        doi = urllib.parse.unquote(url.split("/api/handles/", 1)[1]).lower()
        self.calls.append(doi)
        if self.fail:
            raise ConnectionError("doi.org unreachable")
        return _HandleResponse(doi in self.registered)

    def close(self):
        pass


def _one_doi_record(tmp_path, exact):
    keys = tmp_path / "keys.csv"
    _write_csv(keys, [{"pid": "p1", "pmid": "1", "pmcid": "", "doi": "", "is_preprint": "False"}])
    metadata = tmp_path / "meta.csv"
    _write_csv(metadata, [_meta("pmid", "1", "MED", "1")])
    ann = tmp_path / "ann.jsonl"
    _write_jsonl(ann, [{"source": "MED", "id": "1", "fetched_at": "2026-09-14T01:00:00+00:00",
                        "status": "ok",
                        "annotations": [_ann(exact, "DOI", f"http://identifiers.org/doi:{exact}")]}])
    return keys, metadata, ann


def test_a_dotted_doi_is_withheld_until_confirmed_then_written_repaired_and_cached(tmp_path):
    keys, metadata, ann = _one_doi_record(tmp_path, "10.5281/zenodo.18675888.")
    missing, handles = tmp_path / "missing.jsonl", tmp_path / "doi_handles.csv"
    out_p, out_d = tmp_path / "p.csv", tmp_path / "d.csv"
    args = (keys, metadata, ann, missing, missing, "residual", False, out_p, out_d)

    down = _HandleSession(fail=True)
    stats = bd.build(*args, handles_path=handles, session=down)
    assert down.calls == ["10.5281/zenodo.18675888"]          # the cleaned DOI, not the dotted one
    assert stats.counts["unresolved"] == 1 and stats.counts["unresolved_pending_doi"] == 1
    assert next(csv.DictReader(out_d.open()))["data_links_json"] == ""

    up = _HandleSession(registered=["10.5281/zenodo.18675888"])
    stats = bd.build(*args, handles_path=handles, session=up)
    link = json.loads(next(csv.DictReader(out_d.open()))["data_links_json"])["links"][0]
    assert (link["resource"], link["id"], link["url"]) == (
        "zenodo", "10.5281/zenodo.18675888", "https://doi.org/10.5281/zenodo.18675888")
    assert stats.repaired == {"zenodo": 1}

    cached = _HandleSession(fail=True)
    bd.build(*args, handles_path=handles, session=cached)
    assert cached.calls == []                                   # the registration is remembered


def test_a_comma_list_doi_takes_the_first_candidate_doi_org_confirms():
    stats = bd.Stats()
    handles = {"10.7910/dvn/ufc6b5,harvarddataverse,v2": False, "10.7910/dvn/ufc6b5": True}
    link = bd.link_from_annotation(_ann("10.7910/DVN/UFC6B5,HarvardDataverse,V2", "DOI"), stats, handles)
    assert (link["resource"], link["id"]) == ("dataverse", "10.7910/DVN/UFC6B5")


def test_an_unregistered_doi_and_a_placeholder_are_dropped_and_counted():
    stats = bd.Stats()
    assert bd.link_from_annotation(_ann("10.5281/zenodo.99999999", "DOI"), stats,
                                   {"10.5281/zenodo.99999999": False}) is None
    assert bd.link_from_annotation(_ann("10.5281/zenodo.XXXXX", "DOI"), stats) is None
    assert stats.dropped == {("DOI not registered", "zenodo"): 1, ("malformed DOI", "zenodo"): 1}


def test_a_doi_with_no_verdict_is_pending_never_guessed():
    assert bd.link_from_annotation(_ann("10.5281/zenodo.5", "DOI"), bd.Stats(), {}) is bd.PENDING


def test_negative_verdicts_are_re_asked_once_stale_registrations_never(tmp_path):
    cache = tmp_path / "doi_handles.csv"
    old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    _write_csv(cache, [{"doi": "10.1234/old-no", "exists": "N", "checked_at": old},
                       {"doi": "10.1234/recent-no", "exists": "N", "checked_at": recent},
                       {"doi": "10.1234/old-yes", "exists": "Y", "checked_at": old}])
    session = _HandleSession(registered=["10.1234/old-no"])
    verdicts = bd.verify_doi_handles({"10.1234/old-no", "10.1234/recent-no", "10.1234/old-yes"},
                                     cache, 4, 30, session)
    assert session.calls == ["10.1234/old-no"]
    assert verdicts == {"10.1234/old-no": True, "10.1234/recent-no": False, "10.1234/old-yes": True}


def test_literature_dois_are_never_sent_to_doi_org():
    assert bd.annotation_doi_candidates(_ann("10.1038/s41586-021-03819-2", "DOI")) == []
    assert bd.annotation_doi_candidates(_ann("10.5281/zenodo.1", "DOI", section="References")) == []
    assert bd.annotation_doi_candidates(_ann("10.5281/zenodo.1.", "DOI")) == ["10.5281/zenodo.1"]


def test_free_text_rrids_take_the_id_from_europe_pmcs_resolver_url():
    stats = bd.Stats()
    link = bd.link_from_annotation(_ann("Alexa Fluor 647-conjugated goat anti-rabbit", "RRID",
                                        "https://scicrunch.org/resolver/RRID:AB_2535812"), stats)
    assert (link["resource"], link["id"], link["url"]) == (
        "rrid", "RRID:AB_2535812", "https://scicrunch.org/resolver/RRID:AB_2535812")
    assert stats.from_resolver == 1 and stats.repaired == {"rrid": 1}


def test_a_spaced_prefix_is_normalised_and_a_clean_curie_is_kept():
    stats = bd.Stats()
    orpha = bd.link_from_annotation(_ann("ORPHA 401777", "Orphanet", "http://identifiers.org/orphanet:401777"), stats)
    assert orpha["id"] == "ORPHA:401777"
    go = bd.link_from_annotation(_ann("GO:0002376", "Gene Ontology (GO)", "http://identifiers.org/go:0002376"), stats)
    assert go["id"] == "GO:0002376"


def test_an_identifier_nothing_clean_can_be_recovered_from_is_dropped():
    stats = bd.Stats()
    assert bd.link_from_annotation(_ann("(odc-tbi.org)", "RRID"), stats) is None
    assert stats.dropped == {("malformed", "rrid"): 1}


def test_an_untyped_empiar_supplementary_link_is_empiar_not_pdb():
    link = bd.link_from_annotation(_ann("EMPIAR-10164", None, "https://www.ebi.ac.uk/pdbe/emdb/empiar/entry/10164",
                                        "Supplementary material", provider="Biostudies"), bd.Stats())
    assert link["resource"] == "empiar"


def test_the_build_refuses_to_emit_a_malformed_link():
    with pytest.raises(RuntimeError, match="malformed"):
        bd.assert_clean("p1", {"links": [{"resource": "zenodo", "id": "10.5281/zenodo.1.", "url": None}]})


def test_every_link_built_from_the_real_annotations_fixture_is_clean():
    import fetch_annotations as fa
    from link_identifiers import malformed_links

    articles = json.loads((Path(__file__).parent / "fixtures" / "epmc_annotations_8ids.json")
                          .read_text(encoding="utf-8"))
    annotations = [fa.reduce_annotation(a) for article in articles for a in article["annotations"]]
    handles = {c.lower(): True for a in annotations for c in bd.annotation_doi_candidates(a)}
    stats = bd.Stats()
    links = [bd.link_from_annotation(a, stats, handles) for a in annotations]
    built = [link for link in links if link is not None and link is not bd.PENDING]
    assert built and not any(link is bd.PENDING for link in links)
    assert malformed_links({"links": built}) == []
