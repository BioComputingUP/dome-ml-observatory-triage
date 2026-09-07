import pandas as pd

from dome_triage.llm_classify.landscape_consolidate import (
    merge_landscape_metadata,
    resolve_landscape_classifications,
)


def _write_events(path, rows):
    defaults = {
        "model_tier": "flash", "mode": "primary", "prompt_version": "v1",
        "criteria_sha256": "abc", "batch_id": "b1", "timestamp": "2026-08-27T00:00:00",
        "rationale": "r",
    }
    pd.DataFrame([{**defaults, **r} for r in rows]).to_csv(path, index=False)


def _pool_row(pmcid=None, pmid=None, doi=None, title="T", abstract="A"):
    return {
        "pmcid": pmcid, "pmid": pmid, "doi": doi, "title": title, "abstract": abstract,
        "journal": "J", "authors": "Au", "year": "2024", "citation_count": "",
        "match_metadata": "", "mesh_headings": "[]", "pub_types": "[]",
        "is_open_access": "True", "keywords_author": "[]", "fulltext_available": "True",
        "fulltext_source_root": "", "abstract_source": "europepmc", "metadata_repair_sources": "",
    }


def test_resolve_splits_parse_error_only_records_into_unresolved(tmp_path):
    path = tmp_path / "events.csv"
    _write_events(path, [
        {"record_id": "r1", "classification": "positive", "timestamp": "2026-08-27T00:00:00"},
        {"record_id": "r2", "classification": "parse_error", "timestamp": "2026-08-27T00:00:00"},
    ])
    resolved, unresolved = resolve_landscape_classifications(path)
    assert list(resolved.record_id) == ["r1"]
    assert list(unresolved.record_id) == ["r2"]


def test_resolve_resolves_a_retried_parse_error_into_resolved(tmp_path):
    path = tmp_path / "events.csv"
    _write_events(path, [
        {"record_id": "r1", "classification": "parse_error", "timestamp": "2026-08-27T00:00:00"},
        {"record_id": "r1", "classification": "negative", "timestamp": "2026-08-27T01:00:00"},
    ])
    resolved, unresolved = resolve_landscape_classifications(path)
    assert list(resolved.record_id) == ["r1"]
    assert resolved.iloc[0].classification == "negative"
    assert unresolved.empty


def test_merge_joins_real_metadata_by_record_id(tmp_path):
    events_path = tmp_path / "events.csv"
    _write_events(events_path, [{"record_id": None, "classification": "positive"}])
    # record_id_from_ids("PMC1", None, None) -- compute what the real function would produce
    from dome_triage.dedupe.keys import record_id_from_ids
    rid = record_id_from_ids("PMC1", None, None)
    resolved = pd.DataFrame([{
        "record_id": rid, "classification": "positive", "rationale": "r", "model_tier": "flash",
        "mode": "primary", "prompt_version": "v1", "criteria_sha256": "abc", "batch_id": "b1",
        "timestamp": "2026-08-27T00:00:00",
    }])

    pool_path = tmp_path / "pool.csv"
    pd.DataFrame([_pool_row(pmcid="PMC1", title="Real Title", abstract="Real Abstract")]).to_csv(
        pool_path, index=False
    )

    out_path = tmp_path / "out.csv"
    stats = merge_landscape_metadata(resolved, pool_path, out_path)
    result = pd.read_csv(out_path, dtype=str)
    assert stats["n_matched"] == 1
    assert len(result) == 1
    assert result.iloc[0].title == "Real Title"
    assert result.iloc[0].classification == "positive"
    assert list(result.columns[:3]) == ["pmid", "pmcid", "doi"]


def test_merge_excludes_the_pool_placeholder_label_columns(tmp_path):
    from dome_triage.dedupe.keys import record_id_from_ids
    rid = record_id_from_ids("PMC1", None, None)
    resolved = pd.DataFrame([{
        "record_id": rid, "classification": "positive", "rationale": "r", "model_tier": "flash",
        "mode": "primary", "prompt_version": "v1", "criteria_sha256": "abc", "batch_id": "b1",
        "timestamp": "2026-08-27T00:00:00",
    }])
    pool_path = tmp_path / "pool.csv"
    row = _pool_row(pmcid="PMC1")
    row.update({"label": "unlabeled", "label_confidence": "unscored", "source_name": "x", "source_file": "y"})
    pd.DataFrame([row]).to_csv(pool_path, index=False)

    out_path = tmp_path / "out.csv"
    merge_landscape_metadata(resolved, pool_path, out_path)
    result = pd.read_csv(out_path, dtype=str)
    for col in ("label", "label_confidence", "source_name", "source_file"):
        assert col not in result.columns


def test_merge_never_produces_more_than_one_row_per_record_id_even_when_the_pool_has_duplicates(tmp_path):
    # Real, confirmed condition in the actual pool: bulk_candidates.csv has ~9,050 rows sharing an
    # identity with another row (833,281 unique ids from 842,331 rows). Without a guard, matching
    # against `wanted` naively would write MORE than one row for the same record_id.
    from dome_triage.dedupe.keys import record_id_from_ids
    rid = record_id_from_ids("PMC1", None, None)
    resolved = pd.DataFrame([{
        "record_id": rid, "classification": "positive", "rationale": "r", "model_tier": "flash",
        "mode": "primary", "prompt_version": "v1", "criteria_sha256": "abc", "batch_id": "b1",
        "timestamp": "2026-08-27T00:00:00",
    }])
    pool_path = tmp_path / "pool.csv"
    pd.DataFrame([
        _pool_row(pmcid="PMC1", title="First copy"),
        _pool_row(pmcid="PMC1", title="Second copy, same identity"),
    ]).to_csv(pool_path, index=False)

    out_path = tmp_path / "out.csv"
    stats = merge_landscape_metadata(resolved, pool_path, out_path)
    result = pd.read_csv(out_path, dtype=str)
    assert len(result) == 1
    assert stats["n_matched"] == 1
    assert stats["n_duplicate_pool_rows_skipped"] == 1
    assert result.iloc[0].title == "First copy"  # first occurrence wins


def test_merge_never_produces_duplicates_when_they_span_a_chunk_boundary(tmp_path, monkeypatch):
    # Same guarantee as above but forcing the two duplicate rows into DIFFERENT chunks, which
    # exercises the cross-chunk `seen` set rather than the within-chunk dedup alone.
    import dome_triage.llm_classify.landscape_consolidate as mod
    monkeypatch.setattr(mod, "_READ_CHUNK_ROWS", 1)

    from dome_triage.dedupe.keys import record_id_from_ids
    rid = record_id_from_ids("PMC1", None, None)
    resolved = pd.DataFrame([{
        "record_id": rid, "classification": "positive", "rationale": "r", "model_tier": "flash",
        "mode": "primary", "prompt_version": "v1", "criteria_sha256": "abc", "batch_id": "b1",
        "timestamp": "2026-08-27T00:00:00",
    }])
    pool_path = tmp_path / "pool.csv"
    pd.DataFrame([
        _pool_row(pmcid="PMC1", title="Chunk 1 copy"),
        _pool_row(pmcid="PMC1", title="Chunk 2 copy"),
    ]).to_csv(pool_path, index=False)

    out_path = tmp_path / "out.csv"
    stats = merge_landscape_metadata(resolved, pool_path, out_path)
    result = pd.read_csv(out_path, dtype=str)
    assert len(result) == 1
    assert stats["n_matched"] == 1


def test_merge_with_no_matches_writes_an_empty_file_with_the_right_header(tmp_path):
    resolved = pd.DataFrame(columns=[
        "record_id", "classification", "rationale", "model_tier", "mode", "prompt_version",
        "criteria_sha256", "batch_id", "timestamp",
    ])
    pool_path = tmp_path / "pool.csv"
    pd.DataFrame([_pool_row(pmcid="PMC1")]).to_csv(pool_path, index=False)

    out_path = tmp_path / "out.csv"
    stats = merge_landscape_metadata(resolved, pool_path, out_path)
    result = pd.read_csv(out_path, dtype=str)
    assert result.empty
    assert "title" in result.columns
    assert "classification" in result.columns
    assert stats["n_matched"] == 0
