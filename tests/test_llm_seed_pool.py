"""Step 19c (standard-curation-route build): tests for load_llm_seed_pool -- the record_id
computation that makes a fetched RawRecord-shaped pool usable as a CurationSession.dataset row,
same pattern as bulk_pool.py::load_bulk_pool (no dedicated test file for that one either; this
pool is small enough that a pure round-trip test is cheap and worth having)."""

import csv
from pathlib import Path

from dome_triage.curate.llm_seed_pool import load_llm_seed_pool
from dome_triage.dedupe.keys import record_id_from_ids


def _write_pool(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "pool.csv"
    fieldnames = [
        "source_name", "source_file", "label", "label_confidence", "pmcid", "pmid", "doi",
        "title", "abstract", "journal", "authors", "year", "citation_count", "match_metadata",
        "mesh_headings", "pub_types", "is_open_access", "keywords_author", "fulltext_available",
        "fulltext_source_root",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _base_row(**overrides) -> dict:
    row = {
        "source_name": "manual_llm_language_model_seed",
        "source_file": "seed.csv",
        "label": "unlabeled",
        "label_confidence": "unscored",
        "pmcid": "",
        "pmid": "12345",
        "doi": "",
        "title": "A test paper",
        "abstract": "A test abstract",
        "journal": "Test Journal",
        "year": "2026",
    }
    row.update(overrides)
    return row


def test_record_id_matches_dedupe_keys_computation(tmp_path):
    path = _write_pool(tmp_path, [_base_row(pmid="31501885")])
    df = load_llm_seed_pool(path)

    expected = record_id_from_ids("", "31501885", "")
    assert df["record_id"].iloc[0] == expected


def test_record_id_present_for_every_row(tmp_path):
    path = _write_pool(tmp_path, [_base_row(pmid="1"), _base_row(pmid="2"), _base_row(pmid="3")])
    df = load_llm_seed_pool(path)

    assert df["record_id"].notna().all()
    assert df["record_id"].nunique() == 3


def test_missing_id_columns_are_filled_not_nan(tmp_path):
    """A row with a blank pmcid/doi must not have that blank read back as float NaN -- `nan` is
    truthy in Python, so `record_id_from_ids`'s `if value:` id-presence check would otherwise treat
    a genuinely-empty pmcid as a *present* one and build a wrong key from it (same class of bug
    `bulk_pool.py::load_bulk_pool`'s own fillna guards against)."""
    path = _write_pool(tmp_path, [_base_row(pmcid="", pmid="99999", doi="")])
    df = load_llm_seed_pool(path)

    assert df["record_id"].iloc[0] == record_id_from_ids("", "99999", "")


def test_original_columns_preserved_for_display(tmp_path):
    """CurationSession.current_record() reads title/abstract/journal/year/mesh_headings straight
    off the dataset row -- load_llm_seed_pool must not drop or rename any of them."""
    path = _write_pool(
        tmp_path,
        [_base_row(pmid="1", title="Real Title", abstract="Real abstract", journal="Nature")],
    )
    df = load_llm_seed_pool(path)
    row = df.iloc[0]

    assert row["title"] == "Real Title"
    assert row["abstract"] == "Real abstract"
    assert row["journal"] == "Nature"
    assert row["label"] == "unlabeled"
    assert row["label_confidence"] == "unscored"
