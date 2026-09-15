"""v1.6.0's record_modified: who stamps it, when, in what format, and that a stamp rolls back.

The rule under test is the one that keeps OAI-PMH incremental harvesting honest and cheap: a write
stamps a document only when it changes a value the observatory's metadata exposes, never for a
rewrite of identical values and never from the citation refresh.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ensure_indexes
import load_documents as ld
import moros_write as mw
import verify_corpus as vc
from moros_write import SafeWriter, replay_rollback
from test_moros_write import DOCS, FakeMoros

STAMP = re.compile(mw.RECORD_MODIFIED_PATTERN)


def _licence_docs() -> dict[str, dict]:
    return {
        "same": {"_id": "same", "source": {"access": {"license": "cc by", "open_access": True}}},
        "changed": {"_id": "changed", "record_modified": "2026-01-01T00:00:00Z",
                    "source": {"access": {"license": "", "open_access": False}}},
    }


LICENCE_UPDATES = [
    ("same", {"source.access.license": "cc by", "source.access.open_access": True}),
    ("changed", {"source.access.license": "cc by", "source.access.open_access": True}),
]


def _write(writer: SafeWriter, updates, streaming: bool):
    if streaming:
        return writer.apply_streaming(iter(updates), total=len(updates))
    return writer.apply(updates)


def test_the_stamp_format_is_utc_to_the_second_with_a_z():
    local = timezone(timedelta(hours=2))
    assert mw.record_modified_stamp(datetime(2026, 9, 15, 20, 5, 7, 999999, tzinfo=local)) \
        == "2026-09-15T18:05:07Z"
    assert STAMP.match(mw.record_modified_stamp())
    assert not STAMP.match("2026-09-15T18:05:07+00:00")


def test_the_stamping_modes_are_exactly_those_that_change_exposed_fields():
    assert mw.STAMPS_RECORD_MODIFIED == {"enrichment", "licences", "preprints", "data_links",
                                         "identifiers"}
    for mode, allowed in mw.WRITE_MODES.items():
        stamps = mode in mw.STAMPS_RECORD_MODIFIED or mode == "migrate_v1_6_0"
        assert (mw.RECORD_MODIFIED_PATH in allowed) == stamps, mode


@pytest.mark.parametrize("streaming", [False, True])
def test_only_documents_whose_values_change_are_stamped(tmp_path, streaming):
    w = SafeWriter(FakeMoros(_licence_docs()), mode="licences", run_id="t", dry_run=False,
                   rollback_dir=tmp_path)
    result = _write(w, LICENCE_UPDATES, streaming)
    docs = w.moros.collection.docs
    assert "record_modified" not in docs["same"]
    assert docs["changed"]["record_modified"] == w.stamp and STAMP.match(w.stamp)
    assert result.stamped == 1 and result.as_dict()["stamped"] == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_a_stamp_rolls_back_with_the_values_it_came_with(tmp_path, streaming):
    w = SafeWriter(FakeMoros(_licence_docs()), mode="licences", run_id="t", dry_run=False,
                   rollback_dir=tmp_path)
    before = copy.deepcopy(w.moros.collection.docs)
    _write(w, LICENCE_UPDATES, streaming)
    assert w.moros.collection.docs != before
    replay_rollback(w.moros, w.rollback_path, confirm=True)
    assert w.moros.collection.docs == before  # the old stamp is back, and "same" still has none


@pytest.mark.parametrize("streaming", [False, True])
def test_the_citation_refresh_never_stamps(tmp_path, streaming):
    w = SafeWriter(FakeMoros(DOCS), mode="citations", run_id="t", dry_run=False, rollback_dir=tmp_path)
    assert w.stamp is None
    result = _write(w, [("id-1", {"publication_metadata.citation_count": 5})], streaming)
    assert result.modified == 1 and result.stamped == 0
    assert "record_modified" not in w.moros.collection.docs["id-1"]


def test_a_dry_run_shows_the_stamp_and_writes_nothing(tmp_path, capsys):
    w = SafeWriter(FakeMoros(_licence_docs()), mode="licences", run_id="t", dry_run=True,
                   rollback_dir=tmp_path)
    w.apply(LICENCE_UPDATES)
    out = capsys.readouterr().out
    assert f"record_modified: '2026-01-01T00:00:00Z' -> '{w.stamp}'" in out
    assert w.moros.collection.ops_seen == [] and not w.rollback_path.exists()


# ---------------------------------------------------------------------------
# load_documents.py: every written document is stamped; resuming ignores the stamp
# ---------------------------------------------------------------------------


class _LoadCollection:
    def __init__(self, docs: dict[str, dict]) -> None:
        self.docs = docs
        self.replaced: list[dict] = []

    def find(self, query, projection=None):
        return [copy.deepcopy(self.docs[i]) for i in query["_id"]["$in"] if i in self.docs]

    def bulk_write(self, ops, ordered=True):
        for op in ops:
            self.replaced.append(op._doc)


class _LoadMoros:
    def __init__(self, docs):
        self.collection = _LoadCollection(docs)


def test_the_loader_stamps_what_it_writes_and_skips_what_only_the_stamp_would_change(tmp_path):
    built = {"_id": "new", "schema_version": "1.6.0", "record_modified": None, "title": "N"}
    resumed = {"_id": "resumed", "schema_version": "1.6.0", "record_modified": None, "title": "R"}
    differing = {"_id": "differing", "schema_version": "1.6.0", "record_modified": None, "title": "D2"}
    path = tmp_path / "batch_documents.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in (built, resumed, differing)), encoding="utf-8")

    moros = _LoadMoros({
        "resumed": {**resumed, "record_modified": "2026-09-15T10:00:00Z"},
        "differing": {**differing, "title": "D1", "record_modified": "2026-09-15T10:00:00Z"},
    })
    stats = ld.upsert_via_pymongo(moros, path, None, {"resumed", "differing"},
                                  allow_replace_existing=False, stamp="2026-09-16T00:00:00Z")

    assert (stats["inserted"], stats["identical_skipped"], stats["differing"]) == (1, 1, ["differing"])
    assert moros.collection.replaced == [{**built, "record_modified": "2026-09-16T00:00:00Z"}]


# ---------------------------------------------------------------------------
# The checks agree with the writers
# ---------------------------------------------------------------------------


def test_verify_corpus_expects_the_authored_version_and_every_required_index():
    schema_py = Path(vc.__file__).resolve().parents[2] / "mongo_landscape_export" / "scripts" / "schema.py"
    authored = re.search(r'^SCHEMA_VERSION\s*=\s*"([^"]+)"', schema_py.read_text(), re.M).group(1)
    assert vc.EXPECTED_SCHEMA_VERSION == authored
    assert set(vc.REQUIRED_INDEXES) == {"_id_", *ensure_indexes.REQUIRED_INDEXES}
    spec = ensure_indexes.REQUIRED_INDEXES["record_modified_positive"]
    assert spec["keys"] == [("record_modified", 1), ("_id", 1)]
    assert spec["options"]["partialFilterExpression"] == {"llm_classification.classification": "positive"}
