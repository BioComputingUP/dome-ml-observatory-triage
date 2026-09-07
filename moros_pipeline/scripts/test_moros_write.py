"""Hermetic tests for the safe writer -- no live server, no network.

The fake collection below is deliberately faithful about the two things the writer depends on:
dotted-path `$set` semantics, and projections that can return a document with a path *absent*
rather than null. Those are exactly what the rollback snapshot has to get right.

Per AGENTS.md, fixtures must never depend on the multi-GB sibling repos or on a reachable server.
"""

from __future__ import annotations

import copy
import json

import pytest

import moros_write as mw
from moros_write import SafeWriter, dig, replay_rollback


# ---------------------------------------------------------------------------
# A small in-memory stand-in for the collection + Moros wrapper
# ---------------------------------------------------------------------------


class _BulkResult:
    def __init__(self, matched: int, modified: int) -> None:
        self.matched_count = matched
        self.modified_count = modified


class FakeCollection:
    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = copy.deepcopy(docs)
        self.ops_seen: list[dict] = []

    @staticmethod
    def _project(doc: dict, projection: dict | None) -> dict:
        if not projection:
            return copy.deepcopy(doc)
        out: dict = {"_id": doc["_id"]}
        for path in projection:
            if path == "_id":
                continue
            value = dig(doc, path)
            if value is mw._ABSENT:
                continue  # a projected-but-missing path is simply absent, as in real MongoDB
            node = out
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = copy.deepcopy(value)
        return out

    def find(self, query: dict, projection: dict | None = None, **kwargs):
        wanted = query.get("_id", {})
        ids = wanted.get("$in", []) if isinstance(wanted, dict) else [wanted]
        return [self._project(self.docs[i], projection) for i in ids if i in self.docs]

    def find_one(self, query: dict, projection: dict | None = None, **kwargs):
        results = self.find(query, projection)
        return results[0] if results else None

    def count_documents(self, query: dict, **kwargs) -> int:
        return len(self.docs) if not query else len(self.find(query))

    def bulk_write(self, ops: list, ordered: bool = True) -> _BulkResult:
        matched = modified = 0
        for op in ops:
            self.ops_seen.append({"filter": op._filter, "doc": op._doc})
            doc_id = op._filter["_id"]
            if doc_id not in self.docs:
                continue
            matched += 1
            doc = self.docs[doc_id]
            before = copy.deepcopy(doc)
            for path, value in op._doc.get("$set", {}).items():
                node = doc
                parts = path.split(".")
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = value
            for path in op._doc.get("$unset", {}):
                node = doc
                parts = path.split(".")
                for part in parts[:-1]:
                    node = node.get(part, {})
                node.pop(parts[-1], None)
            if doc != before:
                modified += 1
        return _BulkResult(matched, modified)


class FakeMoros:
    def __init__(self, docs: dict[str, dict]) -> None:
        self.collection = FakeCollection(docs)

    def describe(self) -> str:
        return "fake-host/dome_observatory.Content (MongoDB 4.2.25, 2 documents)"

    def get(self, doc_id, projection=None):
        return self.collection.find_one({"_id": doc_id}, projection)


DOCS = {
    "id-1": {
        "_id": "id-1",
        "schema_version": "1.1.0",
        "publication_metadata": {"title": "A", "citation_count": None},
        "source": {"abstract_source": "europepmc"},
        "llm_classification": {"classification": "positive", "provider": "deepseek"},
    },
    "id-2": {
        "_id": "id-2",
        "schema_version": "1.1.0",
        "publication_metadata": {"title": "B", "citation_count": 7,
                                 "citation_count_updated": "2026-01-01T00:00:00+00:00"},
        "source": {"abstract_source": "pubmed"},
        "llm_classification": {"classification": "negative", "provider": "deepseek"},
    },
}


@pytest.fixture
def writer(tmp_path):
    def _make(mode="citations", dry_run=False):
        return SafeWriter(FakeMoros(DOCS), mode=mode, run_id="test_run",
                          dry_run=dry_run, rollback_dir=tmp_path)
    return _make


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown write mode"):
        SafeWriter(FakeMoros(DOCS), mode="whatever")


def test_path_outside_the_mode_allowlist_is_refused(writer):
    w = writer("citations")
    with pytest.raises(ValueError, match="may not write"):
        w.validate({"llm_classification.classification": "negative"})


def test_citation_mode_cannot_relabel_who_decided_the_record(writer):
    # A number refresh must never be able to rewrite provenance.
    w = writer("citations")
    with pytest.raises(ValueError, match="may not write"):
        w.validate({"source.decision_provenance": "llm"})


def test_enrichment_mode_cannot_touch_the_verdict(writer):
    # Enrichment is additive by construction; the allowlist is what makes that structural.
    w = writer("enrichment")
    with pytest.raises(ValueError, match="may not write"):
        w.validate({"llm_classification.classification": "positive"})
    w.validate({"content_filters.model_type": ["random forest"]})  # allowed


def test_group_paths_are_refused_because_set_would_replace_the_subdocument(writer):
    # `$set: {source: {...}}` silently drops every sibling field in the group.
    w = writer("migrate_v1_2_0")
    with pytest.raises(ValueError, match="may not write"):
        w.validate({"source": {"decision_provenance": "llm"}})


def test_writing_the_merge_key_is_refused(writer):
    with pytest.raises(ValueError, match="refusing to write _id"):
        writer("citations").validate({"_id": "other"})


def test_operator_shaped_field_names_are_refused(writer):
    with pytest.raises(ValueError, match="operator-shaped"):
        writer("citations").validate({"$unset": {"publication_metadata.citation_count": ""}})


def test_empty_update_is_refused(writer):
    with pytest.raises(ValueError, match="empty update"):
        writer("citations").validate({})


def test_every_update_is_validated_not_just_a_sample(writer):
    w = writer("citations")
    updates = [("id-1", {"publication_metadata.citation_count": 1})] * 50
    updates.append(("id-2", {"llm_classification.provider": "hand-edited"}))
    with pytest.raises(ValueError, match="may not write"):
        w.apply(updates)
    assert w.moros.collection.ops_seen == []  # nothing was issued before the raise


# ---------------------------------------------------------------------------
# dig(): absent is not null
# ---------------------------------------------------------------------------


def test_dig_distinguishes_absent_from_null():
    doc = {"a": {"b": None}}
    assert dig(doc, "a.b") is None
    assert dig(doc, "a.c") is mw._ABSENT
    assert dig(doc, "x.y.z") is mw._ABSENT


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing(writer):
    w = writer("citations", dry_run=True)
    result = w.apply([("id-1", {"publication_metadata.citation_count": 42})])
    assert w.moros.collection.ops_seen == []
    assert result.matched == 0
    assert not w.rollback_path.exists()
    assert w.moros.collection.docs["id-1"]["publication_metadata"]["citation_count"] is None


# ---------------------------------------------------------------------------
# Rollback snapshot + replay
# ---------------------------------------------------------------------------


def test_snapshot_records_absent_and_present_paths_separately(writer):
    w = writer("citations")
    updates = [
        ("id-1", {"publication_metadata.citation_count": 10,
                  "publication_metadata.citation_count_updated": "now"}),
        ("id-2", {"publication_metadata.citation_count": 20,
                  "publication_metadata.citation_count_updated": "now"}),
    ]
    w.snapshot(updates)
    lines = w.rollback_path.read_text().splitlines()
    meta = json.loads(lines[0])["_meta"]
    assert meta["mode"] == "citations" and meta["documents"] == 2
    entries = {json.loads(l)["_id"]: json.loads(l) for l in lines[1:]}

    # id-1 has citation_count (null) but no citation_count_updated at all.
    assert entries["id-1"]["set"] == {"publication_metadata.citation_count": None}
    assert entries["id-1"]["unset"] == ["publication_metadata.citation_count_updated"]
    # id-2 has both.
    assert entries["id-2"]["set"] == {
        "publication_metadata.citation_count": 7,
        "publication_metadata.citation_count_updated": "2026-01-01T00:00:00+00:00",
    }
    assert entries["id-2"]["unset"] == []


def test_a_real_write_is_fully_reversible(writer):
    w = writer("citations")
    updates = [
        ("id-1", {"publication_metadata.citation_count": 34984,
                  "publication_metadata.citation_count_updated": "2026-09-03T00:00:00+00:00"}),
        ("id-2", {"publication_metadata.citation_count": 21,
                  "publication_metadata.citation_count_updated": "2026-09-03T00:00:00+00:00"}),
    ]
    before = copy.deepcopy(w.moros.collection.docs)
    result = w.apply(updates)
    assert result.matched == 2 and result.modified == 2
    assert w.moros.collection.docs["id-1"]["publication_metadata"]["citation_count"] == 34984

    replay_rollback(w.moros, w.rollback_path, confirm=True)
    assert w.moros.collection.docs == before  # byte-for-byte back, including the absent field


def test_rollback_replay_is_dry_by_default(writer):
    w = writer("citations")
    w.apply([("id-1", {"publication_metadata.citation_count": 99})])
    mutated = copy.deepcopy(w.moros.collection.docs)
    replay_rollback(w.moros, w.rollback_path, confirm=False)
    assert w.moros.collection.docs == mutated  # unchanged


def test_only_set_is_ever_issued(writer):
    w = writer("migrate_v1_2_0")
    w.apply([("id-1", {"schema_version": "1.2.0", "source.decision_provenance": "llm"})])
    assert w.moros.collection.ops_seen, "expected at least one op"
    for op in w.moros.collection.ops_seen:
        assert set(op["doc"]) == {"$set"}, f"unexpected operator in {op['doc']}"


def test_a_missing_document_does_not_get_created(writer):
    # $set via UpdateOne without upsert must not invent a document -- the merge key is a UUID5,
    # and a typo'd id silently creating a junk record is exactly what we do not want.
    w = writer("citations")
    result = w.apply([("id-does-not-exist", {"publication_metadata.citation_count": 1})])
    assert result.matched == 0
    assert "id-does-not-exist" not in w.moros.collection.docs


# ---------------------------------------------------------------------------
# apply_streaming: same guarantees, batch by batch
# ---------------------------------------------------------------------------


def test_streaming_write_snapshots_before_writing_each_batch(writer, monkeypatch):
    """The ordering property that makes an interrupted streaming load safe: the snapshot for a
    batch is flushed to disk before that batch's write is issued. Simulated by failing the write
    and asserting the snapshot still describes the pre-write state."""
    w = writer("citations")
    monkeypatch.setattr(mw, "BATCH_SIZE", 1)

    calls = {"n": 0}
    real_bulk = w.moros.collection.bulk_write

    def flaky(ops, ordered=True):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated network death mid-load")
        return real_bulk(ops, ordered=ordered)

    w.moros.collection.bulk_write = flaky
    updates = [
        ("id-1", {"publication_metadata.citation_count": 111}),
        ("id-2", {"publication_metadata.citation_count": 222}),
    ]
    result = w.apply_streaming(iter(updates), total=2)

    assert result.matched == 1 and len(result.errors) == 1
    entries = [json.loads(l) for l in w.rollback_path.read_text().splitlines()[1:]]
    # Both documents were snapshotted -- including the one whose write then failed.
    assert {e["_id"] for e in entries} == {"id-1", "id-2"}
    assert entries[0]["set"] == {"publication_metadata.citation_count": None}


def test_streaming_write_is_reversible(writer, monkeypatch):
    w = writer("citations")
    monkeypatch.setattr(mw, "BATCH_SIZE", 1)
    before = copy.deepcopy(w.moros.collection.docs)
    w.apply_streaming(
        iter([("id-1", {"publication_metadata.citation_count": 10}),
              ("id-2", {"publication_metadata.citation_count": 20})]),
        total=2,
    )
    assert w.moros.collection.docs["id-2"]["publication_metadata"]["citation_count"] == 20
    replay_rollback(w.moros, w.rollback_path, confirm=True)
    assert w.moros.collection.docs == before


def test_streaming_dry_run_writes_nothing_and_takes_no_snapshot(writer):
    w = writer("citations", dry_run=True)
    w.apply_streaming(iter([("id-1", {"publication_metadata.citation_count": 5})]), total=1)
    assert w.moros.collection.ops_seen == []
    assert not w.rollback_path.exists()


def test_streaming_still_validates_every_batch(writer):
    w = writer("citations")
    bad = [("id-1", {"publication_metadata.citation_count": 1}),
           ("id-2", {"llm_classification.provider": "nope"})]
    with pytest.raises(ValueError, match="may not write"):
        w.apply_streaming(iter(bad), total=2)
