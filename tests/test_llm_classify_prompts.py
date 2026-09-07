from dome_triage.llm_classify.prompts import (
    build_forced_choice_prompt,
    build_prompt,
    criteria_sha256,
    load_criteria_text,
)

_CRITERIA_TEXT = "## Positive\nApplies AI/ML methods.\n## Negative\nDoes not.\n## Skipped\nn/a"

_LEAKY_RECORD = {
    "title": "A Study Of X",
    "abstract": "Results were positive for improved accuracy on the held-out test set.",
    "journal": "J Test",
    "year": "2020",
    "label": "positive",
    "notes": "external search confirmed ML positive",
    "curation_tag": "clear_case",
    "sources": '[{"source_name": "dome_top_curate_positive"}]',
    "mesh_headings": '["Machine Learning"]',
    "curation_features": '{"reviewed_carefully": true}',
}


def _flatten(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def test_prompt_never_contains_human_label_or_notes_field_values():
    text = _flatten(build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert "external search confirmed ML positive" not in text
    assert "clear_case" not in text
    assert "dome_top_curate_positive" not in text


def test_prompt_still_includes_a_legitimate_abstract_use_of_the_word_positive():
    """The blinding check must be field-provenance-based, not a crude word-ban -- an abstract that
    legitimately contains the word "positive" (as ordinary scientific prose, not the label) must
    still pass through untouched."""
    text = _flatten(build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert "Results were positive for improved accuracy" in text


def test_prompt_excludes_mesh_headings():
    text = _flatten(build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert "Machine Learning" not in text
    assert "mesh_headings" not in text


def test_prompt_excludes_curation_features():
    text = _flatten(build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert "reviewed_carefully" not in text


def test_prompt_contains_full_criteria_verbatim():
    text = _flatten(build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert _CRITERIA_TEXT in text


def test_prompt_static_content_precedes_record_content():
    messages = build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert _CRITERIA_TEXT in messages[0]["content"]
    assert "A Study Of X" in messages[1]["content"]
    assert "A Study Of X" not in messages[0]["content"]


def test_primary_prompt_output_space_allows_undeterminable_and_bans_skipped():
    text = _flatten(build_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert "undeterminable" in text.lower()
    assert "never output it" in text.lower()


def test_forced_choice_prompt_bans_undeterminable():
    text = _flatten(build_forced_choice_prompt(_LEAKY_RECORD, _CRITERIA_TEXT))
    assert "never allowed to answer" in text.lower()


def test_prompt_handles_missing_fields_gracefully():
    sparse = {"title": None, "abstract": None, "journal": None, "year": None}
    messages = build_prompt(sparse, _CRITERIA_TEXT)
    assert "no title" in messages[1]["content"]
    assert "no abstract available" in messages[1]["content"]


def test_load_criteria_text_reads_file_verbatim(tmp_path):
    path = tmp_path / "CRITERIA.md"
    path.write_text("hello criteria", encoding="utf-8")
    assert load_criteria_text(path) == "hello criteria"


def test_criteria_sha256_is_deterministic_and_changes_with_content():
    a = criteria_sha256("hello")
    b = criteria_sha256("hello")
    c = criteria_sha256("hello!")
    assert a == b
    assert a != c
