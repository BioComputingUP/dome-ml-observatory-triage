import csv
import json

import pandas as pd

from dome_triage.curate.review_detector import (
    NON_METHODS_PUB_TYPES,
    annotate_non_methods,
    build_non_methods_term_sequences,
    contains_term,
    lemma_tokens,
    matches_non_methods_pub_types,
    matches_non_methods_text,
    strip_formatting,
    term_lemma_sequence,
)


def _write_lexicon(tmp_path, rows: list[tuple[str, str]]):
    """`rows` is (term, notes) pairs -- mirrors keyword_lexicon_exclusionary.csv's real columns."""
    path = tmp_path / "exclusionary.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "discriminative_score", "document_frequency", "source", "notes"])
        for term, notes in rows:
            writer.writerow([term, "", "", "test", notes])
    return path


def _standard_lexicon(tmp_path):
    return _write_lexicon(
        tmp_path,
        [
            ("review", "non_methods_pubtype"),
            ("systematic review", "non_methods_pubtype"),
            ("meta-analysis", "non_methods_pubtype"),
            ("case report", "non_methods_pubtype"),
            ("case study", "non_methods_pubtype"),
            ("case studies", "non_methods_pubtype"),
            ("random forest", "some_other_note"),  # not a review term -- must be excluded
        ],
    )


# ---------------------------------------------------------------------------
# strip_formatting
# ---------------------------------------------------------------------------


def test_strip_formatting_decodes_html_entities():
    assert strip_formatting("systematic &amp; review") == "systematic & review"


def test_strip_formatting_strips_markdown_artifacts():
    assert strip_formatting("**systematic** review") == "systematic review"
    assert strip_formatting("*narrative* review") == "narrative review"
    assert strip_formatting("_narrative_ review") == "narrative review"
    assert strip_formatting("`code` review") == "code review"
    assert strip_formatting("# Heading\nreview") == "Heading\nreview"  # strips the marker, not the heading text
    assert strip_formatting("[systematic review](http://example.com)") == "systematic review"
    assert strip_formatting("> a review") == "a review"


def test_strip_formatting_normalizes_hyphens_to_spaces():
    assert strip_formatting("meta-analysis") == "meta analysis"


def test_strip_formatting_handles_none_and_non_string():
    assert strip_formatting(None) == ""
    assert strip_formatting(float("nan")) == ""


# ---------------------------------------------------------------------------
# lemma_tokens -- word-boundary correctness and lemmatization
# ---------------------------------------------------------------------------


def test_preview_does_not_contain_a_review_token():
    """The real, verified false-positive risk plain substring matching can't avoid:
    `"review" in "preview"` is `True` as a plain substring check. Tokenization gives "preview"
    its own distinct token, never containing a separate "review" token."""
    assert "review" not in lemma_tokens("a preview of the results")
    assert "review" not in lemma_tokens("these outcomes were previewed at a conference")


def test_review_word_forms_all_lemmatize_to_review():
    for sentence in ["we review the literature", "as reviewed here", "reviewing the data", "several reviews exist"]:
        assert "review" in lemma_tokens(sentence), sentence


def test_analyses_lemmatizes_to_analysis():
    """Regression test for the real reason lemmatization was chosen over stemming (see
    STEPS_Progress.md Step 19d's cross-check): Porter/Snowball both stem "analysis" -> "analysi"
    but "analyses" -> "analys" -- two DIFFERENT stems for the singular/plural of the exact word
    "meta-analysis" is built from, which would silently miss "meta-analyses" in real text.
    Lemmatization correctly reduces both to "analysis"."""
    assert "analysis" in lemma_tokens("we performed several analyses")


def test_errata_lemmatizes_to_erratum():
    """Second real case from the same stemming-vs-lemmatization cross-check: Porter/Snowball
    don't touch the irregular Latin plural "errata" at all (stemmers only strip regular English
    suffixes), so a term list built from "erratum" would miss "errata were published..." in real
    text. Lemmatization's WordNet-backed approach correctly reduces "errata" -> "erratum"."""
    assert "erratum" in lemma_tokens("errata were published in a subsequent issue")


def test_lemma_tokens_drops_punctuation_and_numeric_tokens():
    tokens = lemma_tokens("a study, from 2021: results!")
    assert "," not in tokens and "2021" not in tokens and ":" not in tokens


def test_lemma_tokens_empty_input():
    assert lemma_tokens("") == []


# ---------------------------------------------------------------------------
# contains_term
# ---------------------------------------------------------------------------


def test_contains_term_exact_sequence_match():
    doc = ["we", "review", "the", "literature"]
    assert contains_term(doc, ("review",))
    assert not contains_term(doc, ("case", "report"))


def test_contains_term_multi_word_sequence():
    doc = ["a", "systematic", "review", "of", "trials"]
    assert contains_term(doc, ("systematic", "review"))


def test_contains_term_empty_sequence_never_matches():
    assert not contains_term(["review"], ())


def test_contains_term_sequence_longer_than_doc_never_matches():
    assert not contains_term(["review"], ("systematic", "review"))


# ---------------------------------------------------------------------------
# term_lemma_sequence / build_non_methods_term_sequences
# ---------------------------------------------------------------------------


def test_term_lemma_sequence_hyphenated_term():
    assert term_lemma_sequence("meta-analysis") == ("meta", "analysis")


def test_build_non_methods_term_sequences_excludes_non_review_rows(tmp_path):
    path = _standard_lexicon(tmp_path)
    sequences = build_non_methods_term_sequences(path)
    # "random forest" has notes="some_other_note" -- must not appear.
    assert ("random", "forest") not in sequences
    assert ("systematic", "review") in sequences


def test_build_non_methods_term_sequences_returns_empty_for_missing_file(tmp_path):
    assert build_non_methods_term_sequences(tmp_path / "does_not_exist.csv") == []


# ---------------------------------------------------------------------------
# matches_non_methods_text
# ---------------------------------------------------------------------------


def test_matches_non_methods_text_hyphen_and_space_variants_both_match(tmp_path):
    sequences = build_non_methods_term_sequences(_standard_lexicon(tmp_path))

    hyphen_match, hyphen_hits = matches_non_methods_text(
        "A meta-analysis of trials", "We performed a meta-analysis of randomized trials.", sequences
    )
    space_match, space_hits = matches_non_methods_text(
        "A meta analysis of trials", "We performed a meta analysis of randomized trials.", sequences
    )
    assert hyphen_match and "meta analysis" in hyphen_hits
    assert space_match and "meta analysis" in space_hits


def test_matches_non_methods_text_pure_preview_is_not_flagged(tmp_path):
    sequences = build_non_methods_term_sequences(_standard_lexicon(tmp_path))
    matched, hits = matches_non_methods_text(
        "A preview of new findings",
        "We present a preview of results from an ongoing clinical trial.",
        sequences,
    )
    assert not matched
    assert hits == []


def test_matches_non_methods_text_capitalized_review_matches(tmp_path):
    """Full lowercasing pipeline, per explicit instruction -- capitalized "Review" mid-sentence
    is matched the same as lowercase, since matching only ever runs against title/abstract text,
    never the journal name field."""
    sequences = build_non_methods_term_sequences(_standard_lexicon(tmp_path))
    matched, hits = matches_non_methods_text("Review of the field", "This Review covers recent advances.", sequences)
    assert matched
    assert "review" in hits


def test_matches_non_methods_text_dedupes_overlapping_hits(tmp_path):
    """"case study"/"case studies" both lemmatize to the same normalized form -- both terms
    firing on the same text should report one deduped hit, not two identical strings."""
    sequences = build_non_methods_term_sequences(_standard_lexicon(tmp_path))
    matched, hits = matches_non_methods_text(
        "Case studies in oncology", "We present several case studies from our clinic.", sequences
    )
    assert matched
    assert hits.count("case study") == 1


def test_matches_non_methods_text_no_terms_never_matches():
    matched, hits = matches_non_methods_text("Some title", "Some abstract", [])
    assert not matched
    assert hits == []


def test_matches_non_methods_text_handles_missing_title_or_abstract(tmp_path):
    sequences = build_non_methods_term_sequences(_standard_lexicon(tmp_path))
    matched, hits = matches_non_methods_text(None, None, sequences)
    assert not matched
    assert hits == []


# ---------------------------------------------------------------------------
# matches_non_methods_pub_types
# ---------------------------------------------------------------------------


def test_matches_non_methods_pub_types_medline_title_case():
    matched, hits = matches_non_methods_pub_types(["Review"])
    assert matched and hits == ["Review"]


def test_matches_non_methods_pub_types_jats_lowercase_hyphenated():
    matched, hits = matches_non_methods_pub_types(["review-article"])
    assert matched and hits == ["review-article"]


def test_matches_non_methods_pub_types_genuine_research_types_not_flagged():
    for pt in ["Clinical Trial", "Journal Article", "Randomized Controlled Trial", "Observational Study"]:
        matched, hits = matches_non_methods_pub_types([pt])
        assert not matched, pt
        assert hits == []


def test_matches_non_methods_pub_types_handles_json_string_cell():
    matched, hits = matches_non_methods_pub_types(json.dumps(["Editorial", "Journal Article"]))
    assert matched and hits == ["Editorial"]


def test_matches_non_methods_pub_types_handles_malformed_and_empty_input():
    assert matches_non_methods_pub_types("") == (False, [])
    assert matches_non_methods_pub_types("not valid json") == (False, [])
    assert matches_non_methods_pub_types(None) == (False, [])
    assert matches_non_methods_pub_types([]) == (False, [])


def test_non_methods_pub_types_constant_is_lowercase():
    """matches_non_methods_pub_types lowercases the input before membership-checking -- the
    constant itself must already be lowercase for that comparison to be meaningful."""
    assert all(v == v.lower() for v in NON_METHODS_PUB_TYPES)


# ---------------------------------------------------------------------------
# annotate_non_methods -- combined OR logic, dataset-level
# ---------------------------------------------------------------------------


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "clean1",
                "title": "A novel deep learning approach",
                "abstract": "We trained a random forest classifier on clinical data.",
                "pub_types": json.dumps(["Journal Article"]),
            },
            {
                "record_id": "text_only",
                "title": "Systematic review of X",
                "abstract": "We conducted a systematic review of the literature.",
                "pub_types": json.dumps(["Journal Article"]),
            },
            {
                "record_id": "pubtype_only",
                "title": "Findings on Y",
                "abstract": "A primary-research abstract with an experimental cohort and results.",
                "pub_types": json.dumps(["Review"]),
            },
            {
                "record_id": "both_signals",
                "title": "Meta-analysis of Z",
                "abstract": "This meta-analysis summarizes prior trials.",
                "pub_types": json.dumps(["Meta-Analysis"]),
            },
            {
                "record_id": "missing_fields",
                "title": None,
                "abstract": None,
                "pub_types": None,
            },
        ]
    )


def test_annotate_non_methods_or_combination(tmp_path):
    lexicon = _standard_lexicon(tmp_path)
    result = annotate_non_methods(_dataset(), lexicon)
    flags = result.set_index("record_id")["likely_review_or_non_methods"]
    # bool(flags[...]), not `flags[...] is True/False` -- indexing a pandas bool-dtype column
    # always yields a numpy.bool_ scalar (even when the column was built from a list of real
    # Python bools), and numpy.bool_(False) is not the same object as the Python `False`
    # singleton, so an `is` check fails despite the value being correct.

    assert bool(flags["clean1"]) is False
    assert bool(flags["text_only"]) is True
    assert bool(flags["pubtype_only"]) is True
    assert bool(flags["both_signals"]) is True
    assert bool(flags["missing_fields"]) is False


def test_annotate_non_methods_detail_records_which_signal_fired(tmp_path):
    lexicon = _standard_lexicon(tmp_path)
    result = annotate_non_methods(_dataset(), lexicon)
    detail = result.set_index("record_id")["likely_review_or_non_methods_detail"]

    assert detail["clean1"] == {"text_hits": [], "pub_type_hits": []}
    assert detail["text_only"]["text_hits"] and not detail["text_only"]["pub_type_hits"]
    assert detail["pubtype_only"]["pub_type_hits"] and not detail["pubtype_only"]["text_hits"]
    assert detail["both_signals"]["text_hits"] and detail["both_signals"]["pub_type_hits"]


def test_annotate_non_methods_does_not_mutate_input(tmp_path):
    lexicon = _standard_lexicon(tmp_path)
    original = _dataset()
    original_columns = list(original.columns)
    annotate_non_methods(original, lexicon)
    assert list(original.columns) == original_columns


def test_annotate_non_methods_missing_columns_handled_gracefully(tmp_path):
    lexicon = _standard_lexicon(tmp_path)
    bare = pd.DataFrame([{"record_id": "r1"}])
    result = annotate_non_methods(bare, lexicon)
    assert bool(result["likely_review_or_non_methods"].iloc[0]) is False
