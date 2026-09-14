"""Tests for the EBI Search cross-reference fetch: the targets, the discovery record, and the detail
pass's batching, short-answer re-asks, paging and resume."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import fetch_ebisearch_xrefs as fx

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


DISCOVERY = _fixture("ebisearch_xref_discovery_33024307.json")
PDBEKB = _fixture("ebisearch_xref_detail_pdbekb.json")                 # 33024307: 5, 32015508: 12
PDBEKB_SHORT = _fixture("ebisearch_xref_detail_pdbekb_short.json")     # 33024307: 1 of 5, no size
BIOSTUDIES = _fixture("ebisearch_xref_detail_biostudies_literature.json")


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Session:
    """Answers each GET from `answer(ids, domain, params)` and records (ids, domain, params)."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        ids, _, domain = url.split("/europepmc/entry/", 1)[1].partition("/xref")
        call = (ids.split(","), domain.lstrip("/"), params)
        self.calls.append(call)
        payload, status = self.answer(*call)
        return _FakeResponse(payload, status)


def _queue(*payloads):
    remaining = list(payloads)
    return lambda ids, domain, params: (remaining.pop(0), 200)


def _only(payload: dict, pmid: str) -> dict:
    return {"entries": [e for e in payload["entries"] if e["id"] == pmid]}


def _entry(pmid: str, count: int, refs: range) -> dict:
    return {"entries": [{"id": pmid, "source": "europepmc", "referenceCount": count,
                         "references": [{"id": f"R{i}", "source": "uniprot",
                                         "fields": {"id": [f"R{i}"]}} for i in refs]}]}


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _discovery(pmid: str, counts: dict[str, int]) -> dict:
    return {"source": "MED", "id": pmid,
            "domains": [{"id": d, "referenceEntryCount": n} for d, n in counts.items()]}


def test_targets_are_pmids_of_the_classification_once_each(tmp_path):
    csv_path = tmp_path / "keys.csv"
    csv_path.write_text(
        "pid,pmid,pmcid,doi,classification\n"
        "a,1,,,positive\n"
        "b,1,,,positive\n"
        "c,2,,,negative\n"
        "d,,PMC9,10.1/x,positive\n"
        "e,PMC3,,,positive\n", encoding="utf-8")
    assert fx.load_targets(csv_path, "positive") == ["1"]
    assert fx.load_targets(csv_path, None) == ["1", "2"]
    no_column = tmp_path / "sample.csv"
    no_column.write_text("pmid\n5\n6\n", encoding="utf-8")
    assert fx.load_targets(no_column, "positive") == ["5", "6"]


def test_the_discovery_record_keeps_the_domains_verbatim():
    session = _Session(lambda *call: (DISCOVERY, 200))
    rec = fx.fetch_discovery(session, "33024307", "now")
    assert session.calls == [(["33024307"], "", {"format": "json"})]
    assert rec == {"source": "MED", "id": "33024307", "fetched_at": "now", "http_status": 200,
                   "domains": [{"id": "biostudies-literature", "referenceEntryCount": 1},
                               {"id": "pdbekb", "referenceEntryCount": 5}]}


def test_an_unknown_pmid_is_an_answer_with_no_domains_and_a_server_error_raises():
    empty = _Session(lambda *call: ({"domains": []}, 200))
    assert fx.fetch_discovery(empty, "1", "now")["domains"] == []
    with pytest.raises(RuntimeError):
        fx.fetch_discovery(_Session(lambda *c: ({}, 500)), "1", "now")


def test_detail_pairs_follow_the_latest_discovery_and_skip_dumped_domains(tmp_path):
    path = _write_jsonl(tmp_path / "discovery.jsonl", [
        _discovery("1", {"pdbekb": 5, "geo": 0, "dome-registry": 1}),
        _discovery("2", {"pdbekb": 2}),
        _discovery("2", {}),                 # a refresh found none: the latest line wins
    ])
    discovery = fx.latest_discovery(path)
    assert fx.detail_pairs(discovery, {"dome-registry"}) == [("pdbekb", "1")]
    assert fx.detail_pairs(discovery, set()) == [("dome-registry", "1"), ("pdbekb", "1")]
    assert fx.detail_pairs(discovery, set(), {"geo"}) == []


def test_batches_are_per_domain_and_at_most_100_pmids():
    pairs = [("pdbekb", str(i)) for i in range(250)] + [("geo", "7")]
    assert [(d, len(ids)) for d, ids in fx.batches(pairs)] == [
        ("geo", 1), ("pdbekb", 100), ("pdbekb", 100), ("pdbekb", 50)]


def test_a_complete_answer_takes_one_call_with_size_100():
    session = _Session(_queue(PDBEKB))
    records, calls = fx.fetch_detail_batch(session, "pdbekb", ["33024307", "32015508"], "now")
    assert calls == 1
    assert session.calls[0] == (["33024307", "32015508"], "pdbekb",
                                {"format": "json", "fields": "id,name", "size": 100})
    by_id = {r["id"]: r for r in records}
    assert (by_id["33024307"]["reference_count"], len(by_id["33024307"]["references"])) == (5, 5)
    assert (by_id["32015508"]["reference_count"], len(by_id["32015508"]["references"])) == (12, 12)
    assert all(r["complete"] and not r["truncated"] and r["domain"] == "pdbekb" for r in records)
    assert by_id["33024307"]["references"][0] == {"id": "P0DTC2", "fields": {"id": ["P0DTC2"],
                                                                             "name": []}}


def test_a_short_answer_is_asked_again_on_its_own():
    session = _Session(_queue(PDBEKB_SHORT, _only(PDBEKB, "33024307")))
    records, calls = fx.fetch_detail_batch(session, "pdbekb", ["33024307"], "now")
    assert calls == 2 and session.calls[1][0] == ["33024307"]
    assert len(records[0]["references"]) == 5 and records[0]["complete"]


def test_an_answer_that_stays_short_is_written_incomplete():
    session = _Session(lambda *call: (PDBEKB_SHORT, 200))
    records, calls = fx.fetch_detail_batch(session, "pdbekb", ["33024307"], "now")
    assert calls == 1 + fx.SHORT_RETRIES
    assert records[0]["reference_count"] == 5 and len(records[0]["references"]) == 1
    assert not records[0]["complete"] and not records[0]["truncated"]


def test_a_pmid_left_out_of_the_answer_is_written_incomplete():
    session = _Session(_queue(_only(PDBEKB, "33024307")))
    records, calls = fx.fetch_detail_batch(session, "pdbekb", ["33024307", "99999999999"], "now")
    missing = [r for r in records if r["id"] == "99999999999"][0]
    assert calls == 1
    assert missing["reference_count"] is None and missing["references"] == []
    assert not missing["complete"]


def test_references_past_one_page_are_paged_up_to_max_refs():
    def answer(ids, domain, params):
        start = params.get("start", 0)
        return _entry("1", 250, range(start, min(start + 100, 250))), 200

    session = _Session(answer)
    records, calls = fx.fetch_detail_batch(session, "uniprot", ["1"], "now", max_refs=200)
    assert calls == 2 and [c[2].get("start") for c in session.calls] == [None, 100]
    assert [r["id"] for r in records[0]["references"]] == [f"R{i}" for i in range(200)]
    assert records[0]["truncated"] and records[0]["complete"]

    session = _Session(answer)
    records, calls = fx.fetch_detail_batch(session, "uniprot", ["1"], "now", max_refs=300)
    assert calls == 3 and len(records[0]["references"]) == 250
    assert not records[0]["truncated"] and records[0]["complete"]

    records, calls = fx.fetch_detail_batch(_Session(answer), "uniprot", ["1"], "now")
    assert calls == 1 and len(records[0]["references"]) == 100 and records[0]["truncated"]


def test_a_biostudies_reference_keeps_its_acc():
    records, _ = fx.fetch_detail_batch(_Session(_queue(BIOSTUDIES)), "biostudies-literature",
                                       ["33024307"], "now")
    assert records[0]["references"] == [{
        "id": "S-EPMC7537588", "acc": "S-EPMC7537588",
        "fields": {"id": ["S-EPMC7537588"],
                   "name": ["Characteristics of SARS-CoV-2 and COVID-19."]},
    }]


def test_resume_counts_a_pair_done_only_when_its_latest_record_is_complete_and_fresh(tmp_path):
    now = datetime.now(timezone.utc)
    recent, old = now.isoformat(), (now - timedelta(days=400)).isoformat()
    path = _write_jsonl(tmp_path / "detail.jsonl", [
        {"domain": "pdbekb", "id": "1", "complete": True, "fetched_at": recent},
        {"domain": "geo", "id": "1", "complete": False, "fetched_at": recent},
        {"domain": "pdbekb", "id": "2", "complete": True, "fetched_at": old},
        {"domain": "pdbekb", "id": "3", "complete": True, "fetched_at": recent},
        {"domain": "pdbekb", "id": "3", "complete": False, "fetched_at": recent},
    ])
    assert fx.load_done_pairs(path, None) == {("pdbekb", "1"), ("pdbekb", "2")}
    assert fx.load_done_pairs(path, 30) == {("pdbekb", "1")}
