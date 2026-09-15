"""Tests for the v1.4.0 data_links load: the row mapper and the `data_links` allowlist.

The fetchers (`fetch_annotations.py`, `fetch_datalinks.py`) and the merge (`build_data_links.py`)
have their own test modules. Here: `load_fields.py --mode data_links` and
`WRITE_MODES["data_links"]`, which together are the only way a link reaches moros.
"""

from __future__ import annotations

import csv
import json

import pytest

import load_fields as lf
import moros_write as mw

DETAIL = {
    "fetched_at": "2026-09-14T10:00:00+00:00",
    "sources": ["epmc_annotations", "derived"],
    "link_count": 1,
    "truncated": False,
    "resources": [{"resource": "pdb", "label": "Protein Data Bank in Europe",
                   "category": "Protein Structures", "id_scheme": "PDBe",
                   "publisher": "Europe PMC", "obtained_by": "tm_accession", "count": 1}],
    "links": [{"resource": "pdb", "id": "6VW1", "url": "http://identifiers.org/pdbe/pdb:6VW1",
               "title": None, "obtained_by": "tm_accession", "relationship": "References",
               "section": "Article", "frequency": 3}],
}


def test_summary_columns_write_the_four_summary_leaves():
    pid, update = lf.data_links_row_to_update({
        "pid": "p1", "has_data": "Y", "data_links_tags": '["supporting_data"]',
        "accession_types": '["pdb"]', "db_cross_references": "[]",
    })
    assert pid == "p1"
    assert update == {
        "data_links.has_data": True,
        "data_links.tags": ["supporting_data"],
        "data_links.accession_types": ["pdb"],
        "data_links.db_cross_references": [],
    }


def test_has_data_n_is_written_as_false():
    _, update = lf.data_links_row_to_update({"pid": "p1", "has_data": "N"})
    assert update == {"data_links.has_data": False}


def test_a_blank_flag_leaves_has_data_alone():
    _, update = lf.data_links_row_to_update({"pid": "p1", "has_data": "",
                                             "accession_types": '["geo"]'})
    assert "data_links.has_data" not in update


def test_the_json_cell_writes_exactly_the_six_link_leaves():
    _, update = lf.data_links_row_to_update({"pid": "p1", "data_links_json": json.dumps(DETAIL)})
    assert update == {f"data_links.{k}": v for k, v in DETAIL.items()}


def test_a_stray_key_in_the_cell_cannot_reach_the_writer():
    cell = json.dumps({**DETAIL, "has_data": True, "identifiers.zenodo": "x"})
    _, update = lf.data_links_row_to_update({"pid": "p1", "data_links_json": cell})
    assert "data_links.has_data" not in update
    assert not any("zenodo" in k for k in update)


def test_a_row_carrying_nothing_yields_no_update():
    assert lf.data_links_row_to_update({"pid": "p1"}) is None
    assert lf.data_links_row_to_update({"pid": "", "has_data": "Y"}) is None


def test_both_passes_can_share_one_row():
    _, update = lf.data_links_row_to_update({"pid": "p1", "has_data": "Y",
                                             "data_links_json": json.dumps(DETAIL)})
    assert update["data_links.has_data"] is True
    assert update["data_links.link_count"] == 1


def test_the_data_links_allowlist_is_the_ten_leaves_plus_the_version_and_the_stamp():
    assert mw.WRITE_MODES["data_links"] == frozenset({
        "schema_version", "record_modified",
        "data_links.has_data", "data_links.tags", "data_links.accession_types",
        "data_links.db_cross_references", "data_links.fetched_at", "data_links.sources",
        "data_links.link_count", "data_links.truncated", "data_links.resources",
        "data_links.links",
    })


def test_the_data_links_mode_cannot_reach_identifiers_or_verdicts():
    allowed = mw.WRITE_MODES["data_links"]
    assert not any(p.startswith("identifiers") for p in allowed)
    assert not any(p.startswith("llm_classification") for p in allowed)
    assert not any(p.startswith("llm_enrichment") for p in allowed)
    assert "source.decision_provenance" not in allowed


def test_the_migration_mode_covers_exactly_the_new_leaves():
    # v1.6.0's record_modified joined the two field modes later; the 1.4.0 migration predates it.
    assert mw.WRITE_MODES["migrate_v1_4_0"] == (
        mw.WRITE_MODES["preprints"] | mw.WRITE_MODES["data_links"]
    ) - {"record_modified"}


def test_every_mapper_output_path_is_inside_its_allowlist():
    _, update = lf.data_links_row_to_update({
        "pid": "p", "has_data": "Y", "data_links_tags": "[]", "accession_types": "[]",
        "db_cross_references": "[]", "data_links_json": json.dumps(DETAIL),
    })
    assert set(update) <= mw.WRITE_MODES["data_links"]



def test_a_malformed_link_cannot_reach_the_writer():
    bad = dict(DETAIL, links=[dict(DETAIL["links"][0], id="6VW1.")])
    with pytest.raises(ValueError, match="malformed"):
        lf.data_links_row_to_update({"pid": "p1", "data_links_json": json.dumps(bad)})


def test_a_dirty_staging_file_is_refused_before_moros_is_contacted(tmp_path, monkeypatch):
    path = tmp_path / "pid_data_links.csv"
    bad = dict(DETAIL, links=[dict(DETAIL["links"][0], url="http://identifiers.org/pdbe/pdb:6VW1.")])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pid", "data_links_json"])
        writer.writeheader()
        writer.writerow({"pid": "p0", "data_links_json": json.dumps(DETAIL)})
        writer.writerow({"pid": "p1", "data_links_json": json.dumps(bad)})

    def no_connect(*args, **kwargs):
        raise AssertionError("moros must not be contacted")

    monkeypatch.setattr(lf.Moros, "from_env", no_connect)
    with pytest.raises(SystemExit, match="malformed"):
        lf.run("data_links", path, confirm=True, limit=None)
