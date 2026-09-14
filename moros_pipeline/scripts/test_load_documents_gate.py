"""The whole-document loader refuses a batch carrying malformed data-link identifiers, before it
connects to moros. This is the forward triage path's gate: new batches never go through
load_fields.py, so without it a stale data-links file would reach the corpus inside new documents."""

from __future__ import annotations

import json

import pytest

import load_documents as ld

CLEAN = {"resource": "zenodo", "id": "10.5281/zenodo.18675888",
         "url": "https://doi.org/10.5281/zenodo.18675888", "title": None,
         "obtained_by": "tm_accession", "relationship": "References", "section": "Article",
         "frequency": None}


def _doc(pid: str, link: dict | None) -> dict:
    return {"_id": pid, "data_links": {"links": [link] if link else []}}


def _write(path, docs) -> None:
    path.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")


def test_the_scan_finds_only_the_malformed_documents(tmp_path):
    path = tmp_path / "incoming_new_documents.jsonl"
    _write(path, [_doc("ok", CLEAN), _doc("empty", None),
                  _doc("bad", dict(CLEAN, id="10.5281/zenodo.18675888.")),
                  _doc("bad-url", dict(CLEAN, url="https://doi.org/10.5281/zenodo.18675888\xa0"))])
    assert [pid for pid, _ in ld.find_malformed_data_links(path, None)] == ["bad", "bad-url"]


def test_a_document_without_a_data_links_group_is_not_an_error(tmp_path):
    path = tmp_path / "old_documents.jsonl"
    _write(path, [{"_id": "pre-1.4.0"}, {"_id": "null-group", "data_links": None}])
    assert ld.find_malformed_data_links(path, None) == []


def test_the_load_is_refused_before_moros_is_contacted(tmp_path, monkeypatch):
    path = tmp_path / "incoming_new_documents.jsonl"
    _write(path, [_doc("ok", CLEAN), _doc("bad", dict(CLEAN, id="10.17632/bcmn9cxyzs.4]"))])

    def no_connect(*args, **kwargs):
        raise AssertionError("moros must not be contacted")

    monkeypatch.setattr(ld, "load_env", lambda: {})
    monkeypatch.setattr(ld.Moros, "from_env", no_connect)
    with pytest.raises(SystemExit, match="malformed"):
        ld.run(path, tmp_path / "no_gate_report.json", True, None)
