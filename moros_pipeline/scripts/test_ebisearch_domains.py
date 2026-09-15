"""Tests for the whole-domain EBI Search dumps: reading the domain tree, choosing domains, and a
dump that is either complete or not written at all."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import fetch_ebisearch_domains as fd

FIXTURES = Path(__file__).parent / "fixtures"
TREE = json.loads((FIXTURES / "ebisearch_root_trimmed.json").read_text(encoding="utf-8"))
DOME_PAGE = json.loads((FIXTURES / "ebisearch_domain_dome_registry_page.json")
                       .read_text(encoding="utf-8"))
LEAF = {"id": "d", "entries": 250, "publication_fields": ["PUBMED"], "label_fields": ["name"],
        "index_updated": "u", "index_modified": "m"}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _PagedSession:
    """A domain of `total` entries. The page at `empty_at` always answers with none; the page at
    `short_once_at` is one entry short the first time it is asked. Positions in `idless` carry an
    entry with no id; `copies` maps a position to the earlier position it repeats (with changed
    fields when `differ`); an `id:` query counts `id_hits` copies."""

    def __init__(self, total, empty_at=None, short_once_at=None, idless=(), copies=None,
                 id_hits=2, differ=False):
        self.total, self.empty_at, self.short_once_at = total, empty_at, short_once_at
        self.idless, self.copies = set(idless), copies or {}
        self.id_hits, self.differ = id_hits, differ
        self.starts: list[int] = []
        self.id_queries: list[str] = []

    def _entry(self, position):
        if position in self.idless:
            return {"source": "d", "fields": {"id": [], "PUBMED": []}}
        n = self.copies.get(position, position)
        fields = {"id": [f"E{n}"], "PUBMED": [str(n)]}
        if position in self.copies and self.differ:
            fields["PUBMED"] = ["changed"]
        return {"id": f"E{n}", "source": "d", "fields": fields}

    def get(self, url, params=None, timeout=None):
        if "start" not in params:
            self.id_queries.append(params["query"])
            return _FakeResponse({"hitCount": self.id_hits, "entries": []})
        start = int(params["start"])
        n = 0 if start == self.empty_at else max(0, min(fd.PAGE_SIZE, self.total - start))
        if start == self.short_once_at and start not in self.starts:
            n -= 1
        self.starts.append(start)
        return _FakeResponse({"hitCount": self.total,
                              "entries": [self._entry(start + i) for i in range(n)]})

def test_the_tree_yields_each_leaf_with_its_publication_fields():
    leaves = {leaf["id"]: leaf for leaf in fd.leaf_domains(TREE)}
    assert set(leaves) == {"dome-registry", "biotools", "uniprot", "nrnl1"}
    assert leaves["dome-registry"]["entries"] == 1279
    # PMC is typed as an NCBI reference but is a paper identifier all the same.
    assert leaves["dome-registry"]["publication_fields"] == ["EUROPE_PMC", "PMC"]
    assert leaves["biotools"]["publication_fields"] == ["DOI", "PMCID", "PMID"]
    assert leaves["uniprot"]["publication_fields"] == ["DOI", "PUBMED"]
    assert leaves["nrnl1"]["publication_fields"] == []      # patent numbers never name a paper


def test_selection_is_every_citing_domain_under_the_cap_or_the_ones_named():
    leaves = fd.leaf_domains(TREE)
    assert [leaf["id"] for leaf in fd.select_domains(leaves, [], 100_000)] == ["dome-registry",
                                                                                "biotools"]
    assert [leaf["id"] for leaf in fd.select_domains(leaves, ["uniprot"], 100_000)] == ["uniprot"]
    assert fd.select_domains(leaves, ["nrnl1", "nope"], 100_000) == []


def test_the_fields_asked_for_are_the_id_the_labels_and_the_publication_fields():
    leaves = {leaf["id"]: leaf for leaf in fd.leaf_domains(TREE)}
    assert fd.fields_param(leaves["dome-registry"]) == "id,name,title,EUROPE_PMC,PMC"
    assert fd.fields_param(leaves["biotools"]) == "id,name,DOI,PMCID,PMID"
    assert fd.fields_param(leaves["uniprot"]) == "id,DOI,PUBMED"


def test_a_page_reduces_to_one_line_per_entry_with_its_fields_verbatim():
    lines = fd.reduce_page(DOME_PAGE, "dome-registry")
    assert [line["id"] for line in lines] == ["5i5yt6gjy3", "0mlbkqclbr", "du3gc2b5fz"]
    assert lines[0] == {"domain": "dome-registry", "id": "5i5yt6gjy3",
                        "fields": DOME_PAGE["entries"][0]["fields"]}
    assert lines[0]["fields"]["EUROPE_PMC"] == ["26422234"]


def test_a_domain_is_dumped_whole_through_a_temporary_file(tmp_path):
    session = _PagedSession(250)
    entry = fd.dump_domain(session, LEAF, tmp_path, 4, "now")
    lines = [json.loads(line) for line in (tmp_path / "d.jsonl").read_text(encoding="utf-8")
             .splitlines()]
    assert [line["id"] for line in lines] == [f"E{i}" for i in range(250)]
    assert lines[0] == {"domain": "d", "id": "E0", "fields": {"id": ["E0"], "PUBMED": ["0"]},
                        "fetched_at": "now"}
    assert sorted(session.starts) == [0, 100, 200]
    assert (entry["entries_written"], entry["hit_count"], entry["calls"]) == (250, 250, 3)
    assert entry["fields_requested"] == "id,name,PUBMED"
    assert entry["publication_fields"] == ["PUBMED"]
    assert list(tmp_path.iterdir()) == [tmp_path / "d.jsonl"]


def test_a_page_short_once_is_asked_again_and_the_dump_completes(tmp_path):
    session = _PagedSession(250, short_once_at=100)
    entry = fd.dump_domain(session, LEAF, tmp_path, 4, "now")
    assert sorted(session.starts) == [0, 100, 100, 200]
    assert (entry["entries_written"], entry["calls"]) == (250, 4)


def test_an_empty_page_before_the_end_fails_the_domain_and_writes_nothing(tmp_path):
    session = _PagedSession(250, empty_at=200)
    with pytest.raises(fd.DumpError, match="start=200"):
        fd.dump_domain(session, LEAF, tmp_path, 4, "now")
    assert session.starts.count(200) == 1 + fd.SHORT_PAGE_RETRIES
    assert list(tmp_path.iterdir()) == []


def test_a_dump_is_fresh_while_its_file_exists_and_it_is_younger_than_max_age(tmp_path):
    now = datetime(2026, 9, 14, tzinfo=timezone.utc)
    manifest = {"d": {"fetched_at": (now - timedelta(days=10)).isoformat()}}
    assert not fd.is_fresh(manifest, tmp_path, "d", None, now)        # no file yet
    (tmp_path / "d.jsonl").write_text("", encoding="utf-8")
    assert fd.is_fresh(manifest, tmp_path, "d", None, now)
    assert fd.is_fresh(manifest, tmp_path, "d", 30, now)
    assert not fd.is_fresh(manifest, tmp_path, "d", 7, now)
    assert not fd.is_fresh({}, tmp_path, "d", None, now)


def test_the_manifest_round_trips_and_leaves_no_temporary_file(tmp_path):
    fd.save_manifest(tmp_path, {"d": {"entries_written": 3}})
    assert fd.load_manifest(tmp_path) == {"d": {"entries_written": 3}}
    assert [p.name for p in tmp_path.iterdir()] == [fd.MANIFEST_NAME]


def _written_ids(tmp_path):
    return [json.loads(line)["id"]
            for line in (tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()]


def test_an_entry_without_an_id_is_counted_and_left_out(tmp_path):
    entry = fd.dump_domain(_PagedSession(250, idless={150}), LEAF, tmp_path, 4, "now")
    ids = _written_ids(tmp_path)
    assert "E150" not in ids and len(ids) == 249
    assert (entry["entries_written"], entry["entries_without_id"], entry["duplicate_ids"]) == (
        249, 1, [])


def test_an_entry_indexed_twice_is_written_once(tmp_path):
    session = _PagedSession(250, copies={200: 150})
    entry = fd.dump_domain(session, LEAF, tmp_path, 4, "now")
    ids = _written_ids(tmp_path)
    assert ids.count("E150") == 1 and len(ids) == 249
    assert entry["duplicate_ids"] == ["E150"] and entry["entries_without_id"] == 0
    assert session.id_queries == ['id:"E150"'] and entry["calls"] == 4


def test_a_repeat_indexed_once_means_the_pages_moved(tmp_path):
    with pytest.raises(fd.DumpError, match="indexed once"):
        fd.dump_domain(_PagedSession(250, copies={200: 150}, id_hits=1), LEAF, tmp_path, 4, "now")
    assert list(tmp_path.iterdir()) == []


def test_a_repeat_with_different_fields_fails_the_domain(tmp_path):
    with pytest.raises(fd.DumpError, match="different fields"):
        fd.dump_domain(_PagedSession(250, copies={200: 150}, differ=True), LEAF, tmp_path, 4,
                       "now")
    assert list(tmp_path.iterdir()) == []


def test_by_default_only_the_accepted_domains_are_dumped_but_a_named_one_always_is():
    leaves = fd.leaf_domains(TREE)
    assert [leaf["id"] for leaf in fd.select_domains(leaves, [], 100_000, {"biotools"})] == ["biotools"]
    assert [leaf["id"] for leaf in fd.select_domains(leaves, ["uniprot"], 100_000, {"biotools"})] == [
        "uniprot"]


def test_the_repositorys_own_entry_link_is_asked_for_where_the_domain_has_one():
    assert fd.fields_param(dict(LEAF, link_fields=["full_dataset_link"])) == (
        "id,name,full_dataset_link,PUBMED")
