"""Step 19c: tests for the PMID seed-file parser (`_parse_llm_seed_pmids`). Pure-function tests
only -- no EPMC calls here, that's covered by test_epmc_client.py's get_by_ids tests and the
live smoke test documented in STEPS_Progress.md."""

from pathlib import Path

from dome_triage.pipeline.steps import _parse_llm_seed_pmids


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "seed.csv"
    path.write_text(content)
    return path


def test_curated_shape_only_includes_rows_marked_in_use_column(tmp_path):
    content = (
        "pmid,name,url,use\n"
        '111,"Paper, with a comma in the title",https://europepmc.org/article/MED/111,x\n'
        "222,Some other paper,https://europepmc.org/article/MED/222,\n"
        "333,A third paper,https://europepmc.org/article/MED/333,y\n"
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == ["111", "333"]


def test_curated_shape_any_nonblank_use_value_counts_not_just_x_or_y(tmp_path):
    content = (
        "pmid,name,url,use\n"
        "111,Paper A,https://example.com/111,1\n"
        "222,Paper B,https://example.com/222,yes\n"
        "333,Paper C,https://example.com/333, \n"  # whitespace-only -> not marked
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == ["111", "222"]


def test_curated_shape_returns_empty_when_nothing_marked(tmp_path):
    content = (
        "pmid,name,url,use\n"
        "111,Paper A,https://example.com/111,\n"
        "222,Paper B,https://example.com/222,\n"
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == []


def test_bare_shape_includes_every_pmid_with_no_use_gate(tmp_path):
    content = "pmid\n111\n222\n333\n"
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == ["111", "222", "333"]


def test_bare_shape_skips_blank_lines(tmp_path):
    content = "pmid\n111\n\n222\n"
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == ["111", "222"]


def test_duplicate_pmids_collapse_to_one_first_occurrence_wins(tmp_path):
    content = (
        "pmid,name,url,use\n"
        "111,First occurrence,https://example.com/111,x\n"
        "111,Second occurrence,https://example.com/111,x\n"
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == ["111"]


def test_comma_in_title_does_not_split_into_extra_columns(tmp_path):
    """Regression check for exactly the bug this csv.DictReader-based rewrite fixes -- a naive
    line.split(",") would have misaligned every column after a comma inside `name`."""
    content = (
        "pmid,name,url,use\n"
        '111,"Multi-Label Text Classifier at Publication Level Based on ""PubMedBERT + TextRNN""",'
        "https://example.com/111,x\n"
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path) == ["111"]


def test_empty_file_with_only_header_returns_empty(tmp_path):
    path = _write(tmp_path, "pmid,name,url,use\n")
    assert _parse_llm_seed_pmids(path) == []


def test_require_use_marked_false_ignores_use_column_entirely(tmp_path):
    """`ingest fetch-llm-seed-pool`'s use (the standard-curation-route build): fetch metadata for
    every candidate so it can be curated normally, not pre-filtered by a CSV column."""
    content = (
        "pmid,name,url,use\n"
        "111,Paper A,https://example.com/111,x\n"
        "222,Paper B,https://example.com/222,\n"
        "333,Paper C,https://example.com/333,y\n"
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path, require_use_marked=False) == ["111", "222", "333"]


def test_require_use_marked_false_still_dedupes_and_skips_blanks(tmp_path):
    content = (
        "pmid,name,url,use\n"
        "111,Paper A,https://example.com/111,\n"
        "111,Duplicate,https://example.com/111,\n"
        ",Blank pmid row,https://example.com/000,\n"
        "222,Paper B,https://example.com/222,\n"
    )
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path, require_use_marked=False) == ["111", "222"]


def test_require_use_marked_false_on_bare_shape_is_unaffected(tmp_path):
    content = "pmid\n111\n222\n"
    path = _write(tmp_path, content)

    assert _parse_llm_seed_pmids(path, require_use_marked=False) == ["111", "222"]
