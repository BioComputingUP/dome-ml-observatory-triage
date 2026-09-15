"""Tests for the EBI Search route of the merge: publication values classified by shape, the dump
index, the answered / withheld rule, and links in the merge's shape."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

import ebisearch_links as el


class _Stats:
    def __init__(self):
        self.counts = Counter()
        self.dropped = Counter()


@pytest.mark.parametrize("value, expected", [
    ("38427602", ("pmid", "38427602")),
    ("PubMed:41603203", ("pmid", "41603203")),
    ("15362224/", ("pmid", "15362224")),
    ("pmc3232365", ("pmcid", "PMC3232365")),
    ("doi:10.1073/pnas.2320493121", ("doi", "10.1073/pnas.2320493121")),
    ("https://doi.org/10.17617/3.2VLJ6X", ("doi", "10.17617/3.2vlj6x")),
    ("http://dx.doi.org/10.1210/me.2012-1248", ("doi", "10.1210/me.2012-1248")),
    ("doi.org/10.1111/tpj.15499", ("doi", "10.1111/tpj.15499")),
    ("10.21430%2FM3KXJHSP4T", ("doi", "10.21430/m3kxjhsp4t")),
    ("10.1038/ncomms11840", ("doi", "10.1038/ncomms11840")),
    # wrappers measured in the dumps: a PubMed URL, stacked prefixes, `+`-wrapped form values, a
    # space after the DOI's slash
    ("https://www.ncbi.nlm.nih.gov/pubmed/28681415", ("pmid", "28681415")),
    ("https://pubmed.ncbi.nlm.nih.gov/25838424/", ("pmid", "25838424")),
    ("doi:https://doi.org/10.1016/j.molcel.2023.04.025", ("doi", "10.1016/j.molcel.2023.04.025")),
    ("doi:doi.org/10.1016/j.celrep.2021.109350", ("doi", "10.1016/j.celrep.2021.109350")),
    ("11495900+", ("pmid", "11495900")), ("+6360378", ("pmid", "6360378")),
    ("+10.1128%2FMCB.14.12.7909+", ("doi", "10.1128/mcb.14.12.7909")),
    ("10.1128/ mSystems.00442-19", ("doi", "10.1128/msystems.00442-19")),
    # never guessed: truncated, concatenated or mistyped ids, and placeholders
    ("doi:0.1038/s41467-020-19887-3", None), ("1654110710.1038/sj.emboj.7601039", None),
    ("39011652c", None), ("UNKNOWN_1657550_E-MTAB-9301", None), ("none", None),
    ("n/a", None), ("NA", None), ("in preparation", None), ("2000-02-29", None), ("---", None),
    ("", None), (None, None),
])
def test_publication_values_are_classified_by_shape_not_by_field_name(value, expected):
    assert el.classify_value(value) == expected


def test_our_keys_are_normalised_the_way_entries_are():
    assert el.paper_keys("23184988", "pmc3531127", "10.1093/NAR/gks1151") == [
        ("pmid", "23184988"), ("pmcid", "PMC3531127"), ("doi", "10.1093/nar/gks1151")]
    assert el.paper_keys("", "3531127", "") == [("pmcid", "PMC3531127")]
    assert el.paper_keys(None, "", "not a doi") == []


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


AT = "2026-09-14T02:00:00+00:00"


@pytest.fixture
def dumps(tmp_path) -> Path:
    directory = tmp_path / "dumps"
    directory.mkdir()
    entries = {
        "dome-registry": [{"id": "3mm086r5pw", "fields": {
            "id": ["3mm086r5pw"], "name": [], "title": ["Residue-level prediction"],
            "EUROPE_PMC": ["17316627"], "PMC": []}}],
        "emdb": [{"id": "EMD-8592", "fields": {
            "id": ["EMD-8592"], "name": ["Subtomogram average"], "DOI": ["doi:10.1234/emdb.paper"],
            "PUBMED": ["n/a"]}}],
        "node": [{"id": "OEX00010562", "fields": {
            "id": ["OEX00010562"], "name": ["Benign tissue"],
            "PUBMED": ["10.1038/ncomms11840", "27291620"],
            "full_dataset_link": ["https://www.biosino.org/node/experiment/detail/OEX00010562"]}}],
        "biotools": [{"id": "suba3", "fields": {
            "id": ["suba3"], "name": ["SUBA3"], "PMID": ["23180787"], "PMCID": ["PMC3531127"],
            "DOI": ["10.1093/nar/gks1151"]}}],
        "ega": [{"id": "phs000310.v1.p1", "fields": {
            "id": ["phs000310.v1.p1"], "name": ["Gene fusions"], "PUBMED": ["20233430"],
            "full_dataset_link": ["https://ega-archive.org/studies/phs000310.v1.p1"]}}],
    }
    manifest = {}
    for domain, items in entries.items():
        _write_jsonl(directory / f"{domain}.jsonl",
                     [{"domain": domain, **item, "fetched_at": AT} for item in items])
        manifest[domain] = {"fetched_at": AT}
    (directory / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_the_dump_index_holds_every_shape_of_publication_value(dumps):
    index = el.DumpIndex.load(dumps)
    assert index.entries[("pmid", "17316627")] == [
        ("dome-registry", "3mm086r5pw", "Residue-level prediction", None)]
    assert index.entries[("doi", "10.1234/emdb.paper")][0][:2] == ("emdb", "EMD-8592")
    assert [e[1] for e in index.entries[("doi", "10.1038/ncomms11840")]] == ["OEX00010562"]
    assert [e[1] for e in index.entries[("pmid", "27291620")]] == ["OEX00010562"]
    assert index.unclassified == {("emdb", "PUBMED"): 1}
    assert index.fetched_at["biotools"] == AT


def test_the_dump_index_keeps_only_the_keys_asked_for(dumps):
    index = el.DumpIndex.load(dumps, {("pmid", "17316627")})
    assert list(index.entries) == [("pmid", "17316627")]


def _route(dumps, discovery=None, detail=None):
    return el.EbiRoute(None, discovery or {}, detail or {}, el.DumpIndex.load(dumps), Counter())


def _found(pmid="", failed=False, **counts):
    return {pmid: {"fetched_at": "t1", "failed": failed, "counts": counts}}


def _detail(pmid, domain, refs, count=None, complete=True):
    return {"source": "MED", "id": pmid, "domain": domain, "fetched_at": "t2",
            "reference_count": len(refs) if count is None else count, "truncated": False,
            "complete": complete, "references": refs}


def test_a_registry_entry_naming_the_paper_three_ways_is_one_link_on_the_strongest_key(dumps):
    route = _route(dumps, _found("23180787"))
    links, answered, times = route.links_for("23180787", "PMC3531127", "10.1093/nar/gks1151",
                                             _Stats())
    assert answered and times == ["t1", AT]
    assert [{k: link[k] for k in ("resource", "id", "url", "title", "obtained_by", "relationship",
                                  "matched_by", "source_domain")} for link in links] == [{
        "resource": "biotools", "id": "suba3", "url": "https://bio.tools/suba3", "title": "SUBA3",
        "obtained_by": "ebisearch_domain", "relationship": "IsDescribedBy", "matched_by": "pmid",
        "source_domain": "biotools"}]
    assert links[0]["_routes"] == {"ebisearch_domain"}


def test_a_doi_only_paper_is_reached_through_the_dumps_without_discovery(dumps):
    links, answered, _ = _route(dumps).links_for("", "", "10.1093/NAR/gks1151", _Stats())
    assert answered and [(link["id"], link["matched_by"]) for link in links] == [("suba3", "doi")]


def test_a_pmid_never_discovered_is_not_answered_but_its_dump_matches_stand(dumps):
    stats = _Stats()
    links, answered, _ = _route(dumps).links_for("17316627", "", "", stats)
    assert not answered and stats.counts["ebisearch_waiting_discovery"] == 1
    assert [link["resource"] for link in links] == ["dome_registry"]


def test_every_discovered_accepted_domain_must_be_detailed_before_the_record_answers(dumps):
    discovery = _found("23184988", **{"sra-study": 1, "project": 1})
    sra = _detail("23184988", "sra-study", [
        {"id": "SRP013477", "acc": "SRP013477", "fields": {"id": ["SRP013477"], "name": []}}])
    stats = _Stats()
    _, answered, _ = _route(dumps, discovery, {("23184988", "sra-study"): sra}).links_for(
        "23184988", "", "", stats)
    assert not answered and stats.counts["ebisearch_waiting_detail"] == 1

    unfinished = _detail("23184988", "project", [], count=1, complete=False)
    _, answered, _ = _route(dumps, discovery, {("23184988", "sra-study"): sra,
                                               ("23184988", "project"): unfinished}).links_for(
        "23184988", "", "", _Stats())
    assert not answered

    project = _detail("23184988", "project", [
        {"id": "PRJNA167815", "fields": {"id": ["PRJNA167815"], "name": ["Drosophila melanogaster"]}}])
    links, answered, times = _route(dumps, discovery, {("23184988", "sra-study"): sra,
                                                       ("23184988", "project"): project}).links_for(
        "23184988", "", "", _Stats())
    assert answered and times == ["t1", "t2", "t2"]
    by_id = {link["id"]: link for link in links}
    assert (by_id["SRP013477"]["resource"], by_id["SRP013477"]["url"]) == (
        "ena", "https://www.ebi.ac.uk/ena/browser/view/SRP013477")
    assert by_id["SRP013477"]["title"] is None
    assert (by_id["PRJNA167815"]["resource"], by_id["PRJNA167815"]["title"]) == (
        "bioproject", "Drosophila melanogaster")
    assert by_id["PRJNA167815"]["obtained_by"] == "ebisearch_xref"


def test_a_failed_discovery_record_answers_with_no_xref_links(dumps):
    stats = _Stats()
    links, answered, _ = _route(dumps, _found("20233430", failed=True)).links_for(
        "20233430", "", "", stats)
    assert answered and stats.counts["ebisearch_failed_discovery"] == 1
    assert [(link["resource"], link["id"]) for link in links] == [("dbgap", "phs000310")]


def test_a_moved_entry_takes_its_new_resources_url_and_an_unmoved_one_the_repositorys_own(dumps):
    route = _route(dumps, {**_found("20233430"), **_found("27291620")})
    dbgap = route.links_for("20233430", "", "", _Stats())[0][0]
    assert dbgap["url"] == ("https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?"
                            "study_id=phs000310")
    node = route.links_for("27291620", "", "", _Stats())[0][0]
    assert node["url"] == "https://www.biosino.org/node/experiment/detail/OEX00010562"


def test_entries_a_source_counted_but_did_not_return_are_carried_as_unfetched(dumps):
    refs = [{"id": f"SRP{i:06d}", "fields": {"id": [f"SRP{i:06d}"], "name": []}} for i in range(100)]
    route = _route(dumps, _found("1", **{"sra-study": 150}),
                   {("1", "sra-study"): _detail("1", "sra-study", refs, count=150)})
    links, answered, _ = route.links_for("1", "", "", _Stats())
    assert answered and len(links) == 100
    assert sum(link.get("_unfetched", 0) for link in links) == 50


def test_an_unclean_entry_id_is_dropped_and_counted():
    stats = _Stats()
    assert el.make_link("biotools", "tool.", None, None, "ebisearch_domain", "pmid", stats) is None
    assert stats.dropped == {("malformed (EBI Search)", "biotools"): 1}


def test_loading_reads_only_the_keys_in_scope_and_tabulates_rejected_domains(dumps, tmp_path):
    keys = tmp_path / "keys.csv"
    keys.write_text("pid,pmid,pmcid,doi\np1,23180787,,\np2,17316627,,\n", encoding="utf-8")
    discovery = tmp_path / "discovery.jsonl"
    _write_jsonl(discovery, [
        {"source": "MED", "id": "23180787", "fetched_at": "t0", "http_status": 200,
         "domains": [{"id": "geo", "referenceEntryCount": 2}]},
        {"source": "MED", "id": "23180787", "fetched_at": "t9", "http_status": 200,
         "domains": [{"id": "uniprot", "referenceEntryCount": 3},
                     {"id": "biotools", "referenceEntryCount": 1},
                     {"id": "geo", "referenceEntryCount": 0}]},
        {"source": "MED", "id": "17316627", "fetched_at": "t0", "http_status": 200,
         "domains": [{"id": "mesh", "referenceEntryCount": 1}]},
    ])
    detail = tmp_path / "detail.jsonl"
    _write_jsonl(detail, [_detail("23180787", "uniprot", []), _detail("23180787", "geo", [])])

    route = el.EbiRoute.load(keys, {"p1"}, discovery, detail, dumps, require_all_dumps=False)
    assert route.discovery == {"23180787": {"fetched_at": "t9", "failed": False, "counts": {}}}
    assert route.rejected == {"uniprot": 1}
    assert set(route.detail) == {("23180787", "geo")}
    assert set(route.dumps.entries) == {("pmid", "23180787")}
    assert route.covers("p1") and not route.covers("p2")

    with pytest.raises(SystemExit, match="not dumped"):
        el.EbiRoute.load(keys, {"p1"}, discovery, detail, dumps)
    with pytest.raises(SystemExit, match="discover first"):
        el.EbiRoute.load(keys, {"p1"}, tmp_path / "none.jsonl", detail, dumps,
                         require_all_dumps=False)
