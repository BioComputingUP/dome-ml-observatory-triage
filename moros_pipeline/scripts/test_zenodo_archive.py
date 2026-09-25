"""Hermetic tests for the Zenodo archive (scripts/zenodo_archive.py) -- no network, no server.

Why these exist: the archive publishes a citable, permanent version. What it must never get wrong is
access (a placeholder `restricted` record silently hiding a release forever, or an embargo that
never lapses), the change check that stops a pointless version, and the export's integrity, which
the sidecar's hashes promise.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import zenodo_archive as za  # noqa: E402

TODAY = date(2026, 9, 25)


# -- access -----------------------------------------------------------------------------------------


def test_the_placeholder_restricted_state_is_never_inherited_silently():
    with pytest.raises(SystemExit, match="--embargo-until"):
        za.access_fields({"access_right": "restricted"}, None, False, TODAY)


def test_an_explicit_embargo_sets_the_date():
    assert za.access_fields({"access_right": "restricted"}, date(2026, 10, 25), False, TODAY) == {
        "access_right": "embargoed", "embargo_date": "2026-10-25"}


def test_an_embargo_in_the_past_is_refused():
    with pytest.raises(SystemExit, match="not in the future"):
        za.access_fields({}, date(2026, 9, 1), False, TODAY)


def test_a_running_embargo_is_inherited_and_a_lapsed_one_becomes_open():
    running = {"access_right": "embargoed", "embargo_date": "2026-10-25"}
    lapsed = {"access_right": "embargoed", "embargo_date": "2026-09-01"}
    assert za.access_fields(running, None, False, TODAY)["embargo_date"] == "2026-10-25"
    assert za.access_fields(lapsed, None, False, TODAY) == {"access_right": "open"}


def test_open_stays_open():
    assert za.access_fields({"access_right": "open"}, None, False, TODAY) == {"access_right": "open"}


# -- the change check ------------------------------------------------------------------------------


LIVE = {"record_count": 876_324, "schema_version": "1.6.0",
        "last_classification": "2026-09-15T23:21:27+00:00", "last_enrichment": "2026-09-15T22:35:21+00:00"}


def test_nothing_archived_yet_counts_as_changed():
    assert not za.unchanged(LIVE, None)


def test_identical_figures_are_unchanged_and_any_difference_is_not():
    assert za.unchanged(LIVE, dict(LIVE, exported_at="anything"))
    for key in za.CHANGE_KEYS:
        assert not za.unchanged(LIVE, dict(LIVE, **{key: "different"}))


# -- the version's metadata ------------------------------------------------------------------------


FIG = {**LIVE, "version": "2026-09-25",
       "classification_counts": {"positive": 367_348, "negative": 502_002, "undeterminable": 6_974}}
CURRENT = {"title": "DOME Observatory", "doi": "10.5281/zenodo.22259906", "access_right": "restricted",
           "creators": [{"name": "Farrell, Gavin", "affiliation": None}], "license": "cc-by-4.0",
           "upload_type": "dataset", "publication_date": "2026-09-02"}


def test_a_version_changes_only_what_a_release_changes():
    meta = za.version_metadata(CURRENT, FIG, {"access_right": "open"}, "schema.json")
    assert meta["version"] == meta["publication_date"] == "2026-09-25"
    assert meta["title"] == "DOME Observatory" and meta["license"] == "cc-by-4.0"
    assert "doi" not in meta  # the new version reserves its own
    assert meta["creators"] == [{"name": "Farrell, Gavin"}]  # a null the API would refuse, dropped
    assert "876,324 publications" in meta["description"]
    assert "not individually curator-reviewed" in meta["description"]


def test_the_links_back_are_not_duplicated_by_a_second_release():
    first = za.version_metadata(CURRENT, FIG, {"access_right": "open"}, "schema.json")
    second = za.version_metadata(first, FIG, {"access_right": "open"}, "schema.json")
    identifiers = [r["identifier"] for r in second["related_identifiers"]]
    assert len(identifiers) == len(set(identifiers)) == 2


# -- the export ------------------------------------------------------------------------------------


class _Collection:
    def __init__(self, docs):
        self.docs = docs
        self.kwargs = {}

    def find(self, query, **kwargs):
        self.kwargs = kwargs
        return iter(self.docs)


class _Moros:
    def __init__(self, docs):
        self.collection = _Collection(docs)


def test_the_export_is_one_record_per_line_in_id_order_and_its_hashes_are_true(tmp_path):
    docs = [{"_id": "a", "title": "Ångström"}, {"_id": "b", "n": 1}]
    moros = _Moros(docs)
    out = tmp_path / "corpus.jsonl.gz"
    result = za.export(moros, out, None)

    assert moros.collection.kwargs["sort"] == [("_id", 1)]
    raw = gzip.decompress(out.read_bytes())
    assert [json.loads(line) for line in raw.decode().splitlines()] == docs
    assert result["records"] == 2
    assert result["uncompressed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert result["md5"] == za.file_digest(out)["md5"]


def test_the_same_corpus_always_produces_the_same_bytes(tmp_path):
    docs = [{"_id": "a"}]
    one, two = tmp_path / "1.gz", tmp_path / "2.gz"
    za.export(_Moros(docs), one, None)
    za.export(_Moros(docs), two, None)
    assert one.read_bytes() == two.read_bytes()  # mtime 0: no timestamp in the gzip header


# -- reading the latest version's sidecar ----------------------------------------------------------


class _Resp:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status_code, self._payload, self.ok, self.text = status, payload, status < 400, ""

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, resp: _Resp) -> None:
        self.resp, self.urls, self.headers = resp, [], {}

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.resp


def _zenodo(resp: _Resp) -> za.Zenodo:
    zen = za.Zenodo("https://zenodo.org", "t")
    zen.session = _Session(resp)
    return zen


def test_the_sidecar_is_read_from_the_published_record_not_a_file_list():
    # The version listing returns no files, and a published deposition's download links point at
    # its gone draft: reading either made every run look like the first (2026-09-25).
    zen = _zenodo(_Resp(200, {"record_count": 876_324}))
    assert zen.read_sidecar({"id": 22967974, "files": []}) == {"record_count": 876_324}
    assert zen.session.urls == ["https://zenodo.org/api/records/22967974/files/archive-metadata.json/content"]


def test_a_version_without_a_sidecar_reads_as_none():
    assert _zenodo(_Resp(404)).read_sidecar({"id": 22259906}) is None
