import pandas as pd

from dome_triage.curate.bulk_pool import load_bulk_pool
from dome_triage.dedupe.keys import record_id_from_ids


def _write_scored_csv(path):
    pd.DataFrame(
        {
            "source_name": ["bulk_match_2020", "bulk_match_2021", "bulk_match_2022"],
            "source_file": ["a.jsonl", "b.jsonl", "c.jsonl"],
            "label": ["unlabeled", "unlabeled", "unlabeled"],
            "label_confidence": ["unscored", "unscored", "unscored"],
            "pmcid": ["PMC1000001", "", ""],
            "pmid": ["11111111", "22222222", "33333333"],
            "doi": ["", "", "10.1000/z"],
            "title": ["Fresh candidate", "Already curated positive", "Already curated negative"],
            "abstract": ["Abstract one.", "Abstract two.", "Abstract three."],
            "journal": ["J1", "J2", "J3"],
            "authors": ["A", "B", "C"],
            "year": ["2020", "2021", "2022"],
            "citation_count": ["1", "2", "3"],
            "mesh_headings": ['["Term A"]', "[]", '["Term B", "Term C"]'],
            "pub_types": ['["Journal Article"]', '["Journal Article"]', '["Journal Article"]'],
            "is_open_access": ["True", "False", "True"],
            "keywords_author": ["[]", "[]", "[]"],
            "fulltext_available": ["False", "True", "False"],
            "fulltext_source_root": ["", "", ""],
            "has_pmcid": ["True", "False", "False"],
            # Row 0: genuinely new, never curated. Row 1: already curated positive (trusted).
            # Row 2: already curated negative (trusted) -- both matching the real
            # already_curated/existing_label/existing_label_confidence columns Step 12 annotates.
            "already_curated": ["False", "True", "True"],
            "existing_label": ["", "positive", "negative"],
            "existing_label_confidence": ["", "human_curated", "human_curated"],
            "match_score__bm25": ["150.5", "80.2", "10.0"],
            "matched_terms__bm25": ["[]", "[]", "[]"],
            "match_classification__bm25": ["positive", "negative", "negative"],
        }
    ).to_csv(path, index=False)
    return path


def test_load_bulk_pool_computes_record_id_matching_dedupe_keys(tmp_path):
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    pool = load_bulk_pool(path)

    expected = record_id_from_ids("PMC1000001", "11111111", "")
    assert pool.loc[0, "record_id"] == expected


def test_load_bulk_pool_uses_existing_label_when_already_curated(tmp_path):
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    pool = load_bulk_pool(path)

    # Row 0 was never curated -- keeps the raw (fresh-candidate) label/confidence.
    assert pool.loc[0, "label"] == "unlabeled"
    assert pool.loc[0, "label_confidence"] == "unscored"
    # Rows 1/2 are already_curated=True -- must reflect the *existing* canonical label/confidence,
    # not the raw bulk-match defaults, so CurationSession's trusted-label exclusion logic works.
    assert pool.loc[1, "label"] == "positive"
    assert pool.loc[1, "label_confidence"] == "human_curated"
    assert pool.loc[2, "label"] == "negative"
    assert pool.loc[2, "label_confidence"] == "human_curated"


def test_load_bulk_pool_renames_score_columns_for_curation_session(tmp_path):
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    pool = load_bulk_pool(path)

    assert "bulk_match_score" in pool.columns
    assert "bulk_match_classification" in pool.columns
    assert pool.loc[0, "bulk_match_score"] == 150.5
    assert pool.loc[0, "bulk_match_classification"] == "positive"


def test_load_bulk_pool_falls_back_correctly_when_higher_priority_id_is_empty(tmp_path):
    # Row 1 has pmcid="" (an empty CSV cell -- read back as NaN, not the string "") and a real
    # pmid. record_id must be computed from the pmid, not from a wrongly-"present" NaN pmcid --
    # this is the exact bug caught by this test: `if value:` on an un-filled NaN is True in
    # Python, which would have produced a bogus "PMCID:nan"-based id instead of falling through.
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    pool = load_bulk_pool(path)

    expected = record_id_from_ids("", "22222222", "")
    assert pool.loc[1, "record_id"] == expected
    # Updated 2026-08-27: this used to assert the "nan" form produced a DIFFERENT id, which was
    # only true because canonical_key_from_ids still accepted the literal string "nan" as a real
    # identifier. That acceptance was the defect (it fused 250 distinct papers onto one id in the
    # Step 23a 1,000-record test), and it is now rejected at source -- so a "nan" pmcid correctly
    # falls through to the pmid and yields the SAME id, which is exactly what this test's own
    # comment above asks for.
    assert pool.loc[1, "record_id"] == record_id_from_ids("nan", "22222222", "")
    assert record_id_from_ids("nan", None, None) is None


def test_load_bulk_pool_carries_display_columns_needed_by_the_curate_page(tmp_path):
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    pool = load_bulk_pool(path)

    for col in ("title", "abstract", "journal", "year", "mesh_headings", "pmcid", "pmid", "doi"):
        assert col in pool.columns


def test_load_bulk_pool_excludes_unused_columns_that_caused_a_real_oom(tmp_path):
    # Real, measured incident: including authors/pub_types in usecols pushed pd.read_csv's C
    # engine to ~5.6GB peak RSS parsing the real ~745k-row file (vs ~1.9GB without them) --
    # disproportionate to their own ~90MB of final column data -- and, combined with the
    # separately-cached bulk_scores.py score lookup both resident in the same Streamlit process,
    # OOM-killed the host (confirmed via journalctl -k). Neither column is used for display on the
    # Curate page. This test can't reproduce the RSS spike on a tiny fixture, but it pins the one
    # thing that actually prevents it: these columns must never be re-added to _USECOLS without
    # re-measuring RSS against the real file first.
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    pool = load_bulk_pool(path)

    assert "authors" not in pool.columns
    assert "pub_types" not in pool.columns


def test_load_bulk_pool_assigns_placeholder_id_without_dropping_the_row(tmp_path):
    # A record with no pmcid/pmid/doi at all can't get a real record_id -- a real, measured
    # incident (see AGENTS.md's "Curate app performance" section) found that *dropping* such a
    # row via `df[df["record_id"].notna()]` copies the entire ~745k-row frame just to remove one
    # row, spiking RSS by several GB. The fix keeps the row and gives it a synthetic id instead.
    path = _write_scored_csv(tmp_path / "bulk_candidates_scored.csv")
    df = pd.read_csv(path, dtype=str)
    df.loc[len(df)] = df.iloc[0]  # duplicate a row, then blank its ids
    df.loc[len(df) - 1, ["pmcid", "pmid", "doi"]] = ["", "", ""]
    df.to_csv(path, index=False)

    pool = load_bulk_pool(path)

    assert len(pool) == 4  # the row was kept, not dropped
    assert pool["record_id"].notna().all()
    assert pool.loc[3, "record_id"] == "bulkpool-no-id-row-3"
