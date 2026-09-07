from __future__ import annotations

import pytest

from schema import (
    PROVENANCE_HUMAN,
    PROVENANCE_LLM,
    PROVENANCE_REGISTRY,
    SCHEMA_VERSION,
    build_curated_document,
    build_document,
)

BASE_ROW = {
    "pid": "abc-123",
    "pmid": "41466298",
    "pmcid": "PMC12829071",
    "doi": "10.1186/s13014-025-02784-8",
    "title": "A paper",
    "abstract": "An abstract",
    "year": "2025.0",
    "authors": "Shi Z, Zhang C",
    "journal": "Radiation oncology",
    "citation_count": "",
    "citation_count_updated": "",
    "citation_source": "",
    "mesh_headings": '["Humans", "Deep Learning"]',
    "pub_types": '["review-article", "Journal Article"]',
    "is_open_access": "True",
    "keywords_author": '["Artificial intelligence"]',
    "fulltext_available": "True",
    "abstract_source": "europepmc",
    "metadata_repair_sources": "",
    "classification": "negative",
    "rationale": "The paper is a review.",
    "model_tier": "flash",
    "mode": "primary",
    "prompt_version": "v1",
    "criteria_sha256": "bd9d66dd8929",
    "batch_id": "classify_flash_bulk_pool_excluding_curated_primary_20260827T224454",
    "timestamp": "2026-08-27T22:44:56.984086+00:00",
    "license": "cc by-nc-nd",
    "license_checked": "True",
    "epmc_is_open_access": "Y",
}


def test_top_level_shape_and_id():
    doc = build_document(BASE_ROW)
    assert doc["_id"] == "abc-123"
    assert set(doc.keys()) == {
        "_id", "schema_version", "identifiers", "publication_metadata", "source",
        "content_filters", "llm_classification", "llm_enrichment",
    }


def test_schema_version_is_stamped_on_every_document():
    doc = build_document(BASE_ROW)
    assert doc["schema_version"] == SCHEMA_VERSION
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION


def test_identifiers_and_types():
    doc = build_document(BASE_ROW)
    assert doc["identifiers"] == {
        "pmid": "41466298", "pmcid": "PMC12829071", "doi": "10.1186/s13014-025-02784-8",
        "dome_registry": None, "bioai_repo": None, "huggingface": None,
        "kaggle": None, "zenodo": None,
    }
    pub = doc["publication_metadata"]
    assert pub["year"] == 2025 and isinstance(pub["year"], int)
    assert pub["citation_count"] is None
    assert pub["citation_count_updated"] is None
    assert pub["citation_source"] is None


def test_json_array_columns_are_real_lists():
    doc = build_document(BASE_ROW)
    cf = doc["content_filters"]
    assert cf["mesh_headings"] == ["Humans", "Deep Learning"]
    assert cf["pub_types"] == ["review-article", "Journal Article"]
    assert cf["keywords_author"] == ["Artificial intelligence"]


def test_enrichment_fields_reserved_and_empty():
    doc = build_document(BASE_ROW)
    cf = doc["content_filters"]
    assert cf["domain_tier1"] is None
    assert cf["domain_tier2"] == []
    assert cf["learning_paradigm"] == []
    assert cf["model_type"] == []


def test_match_metadata_and_fulltext_source_root_absent_entirely():
    doc = build_document(BASE_ROW)
    flat_keys = set()
    for group in doc.values():
        if isinstance(group, dict):
            flat_keys.update(group.keys())
            for sub in group.values():
                if isinstance(sub, dict):
                    flat_keys.update(sub.keys())
    assert "match_metadata" not in flat_keys
    assert "fulltext_source_root" not in flat_keys


def test_llm_classification_and_model_id_derivation():
    doc = build_document(BASE_ROW)
    llm = doc["llm_classification"]
    assert llm["provider"] == "deepseek"
    assert llm["model_tier"] == "flash"
    assert llm["model_id"] == "deepseek-v4-flash"
    assert llm["classification"] == "negative"
    assert llm["ruleset_sha256"] == "bd9d66dd8929"


def test_llm_enrichment_fully_present_and_null_not_absent():
    doc = build_document(BASE_ROW)
    enrichment = doc["llm_enrichment"]
    expected_keys = {
        "provider", "model_tier", "model_id", "mode", "rationale", "prompt_version",
        "ruleset_sha256", "batch_id", "timestamp", "vocab_violations", "parse_status",
        "input_tokens", "output_tokens", "cache_hit_tokens", "parse_fallback_used",
    }
    assert set(enrichment.keys()) == expected_keys
    assert all(v is None for v in enrichment.values())


def test_open_access_uses_fresh_epmc_value_when_checked():
    row = dict(BASE_ROW, is_open_access="False", license_checked="True", epmc_is_open_access="Y")
    doc = build_document(row)
    assert doc["source"]["access"]["open_access"] is True  # EPMC's fresh value wins


def test_open_access_falls_back_when_not_checked():
    row = dict(BASE_ROW, is_open_access="True", license_checked="False", epmc_is_open_access="")
    doc = build_document(row)
    assert doc["source"]["access"]["open_access"] is True  # falls back to original column


def test_license_empty_string_vs_null_distinction():
    checked_none_disclosed = dict(BASE_ROW, license="", license_checked="True")
    never_checked = dict(BASE_ROW, license="", license_checked="False")

    assert build_document(checked_none_disclosed)["source"]["access"]["license"] == ""
    assert build_document(never_checked)["source"]["access"]["license"] is None


# ---------------------------------------------------------------------------
# v1.2.0: source.decision_provenance
# ---------------------------------------------------------------------------

CURATED_ROW = dict(
    BASE_ROW,
    label="positive",
    label_confidence="human_curated",
    curation_rationale="Curated from dome_registry_231_gold (matched_on: pmcid).",
    curation_timestamp="2026-03-14T09:12:03+00:00",
    curation_batch_id="curated_merge_2026-09-03",
)


def test_landscape_documents_are_provenance_llm():
    doc = build_document(BASE_ROW)
    assert doc["source"]["decision_provenance"] == PROVENANCE_LLM


def test_curated_provenance_comes_from_label_confidence():
    human = build_curated_document(dict(CURATED_ROW, label_confidence="human_curated"))
    registry = build_curated_document(dict(CURATED_ROW, label_confidence="registry_confirmed"))
    assert human["source"]["decision_provenance"] == PROVENANCE_HUMAN
    assert registry["source"]["decision_provenance"] == PROVENANCE_REGISTRY


def test_heuristic_candidate_rows_are_refused_not_published():
    # The 2,258 clear-negative-sampler rows were fetched with the structural inverse of the AI/ML
    # query and must never reach the corpus -- FINALISATION_ROADMAP.md section 2.
    with pytest.raises(ValueError, match="label_confidence"):
        build_curated_document(dict(CURATED_ROW, label_confidence="heuristic_candidate"))


def test_curated_classification_says_human_not_machine():
    doc = build_curated_document(CURATED_ROW)
    llm = doc["llm_classification"]
    # The one queryable field carries the verdict -- positives_text and canUseTextIndex untouched.
    assert llm["classification"] == "positive"
    # ...and every model-describing field is null, because no model was involved.
    assert llm["provider"] is None
    assert llm["model_tier"] is None
    assert llm["model_id"] is None
    assert llm["mode"] is None
    assert llm["prompt_version"] is None
    assert llm["ruleset_sha256"] is None
    # Real curation provenance, never an invented sentence.
    assert llm["rationale"] == CURATED_ROW["curation_rationale"]
    assert llm["batch_id"] == "curated_merge_2026-09-03"
    assert llm["timestamp"] == "2026-03-14T09:12:03+00:00"


def test_skipped_has_no_schema_value_and_must_not_be_coerced():
    # 7 canonical rows carry label="skipped"; the enum has no such value, so the builder refuses
    # rather than silently inventing one. They are filtered upstream instead.
    with pytest.raises(ValueError, match="classification must be one of"):
        build_curated_document(dict(CURATED_ROW, label="skipped"))


def test_undeterminable_is_a_valid_curated_verdict():
    doc = build_curated_document(dict(CURATED_ROW, label="undeterminable"))
    assert doc["llm_classification"]["classification"] == "undeterminable"


def test_both_builders_produce_the_same_shape():
    """The drift guard: the two paths share every group helper, so their key sets must be
    identical at every level. Add a field to one and this fails until it is in both."""

    def shape(node):
        if isinstance(node, dict):
            return {k: shape(v) for k, v in node.items()}
        return type(node).__name__ if node is not None else None

    landscape = build_document(BASE_ROW)
    curated = build_curated_document(CURATED_ROW)

    def keys(node, prefix=""):
        out = set()
        for k, v in node.items():
            out.add(prefix + k)
            if isinstance(v, dict):
                out |= keys(v, prefix + k + ".")
        return out

    assert keys(landscape) == keys(curated)
    assert landscape["schema_version"] == curated["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# v1.2.0: citation metadata
# ---------------------------------------------------------------------------


def test_citation_fields_populate_from_the_join():
    row = dict(
        BASE_ROW,
        citation_count="34984",
        citation_count_updated="2026-09-03T11:04:22+00:00",
        citation_source="europepmc",
    )
    pub = build_document(row)["publication_metadata"]
    assert pub["citation_count"] == 34984 and isinstance(pub["citation_count"], int)
    assert pub["citation_count_updated"] == "2026-09-03T11:04:22+00:00"
    assert pub["citation_source"] == "europepmc"


def test_citation_columns_absent_entirely_means_never_looked_up():
    # The citation fetch is a separate staging stage; a document built before it runs has
    # legitimately never had a count looked up. Same convention as license=None.
    row = {k: v for k, v in BASE_ROW.items()
           if k not in ("citation_count_updated", "citation_source")}
    pub = build_document(row)["publication_metadata"]
    assert pub["citation_count_updated"] is None
    assert pub["citation_source"] is None


# ---------------------------------------------------------------------------
# v1.2.0: HTML entity repair in title/abstract (real corpus cases)
# ---------------------------------------------------------------------------


def test_singly_encoded_title_is_decoded():
    row = dict(BASE_ROW, title="Detection of stipe rot disease in &lt;i&gt;Morchella sextelata&lt;/i&gt;.")
    assert build_document(row)["publication_metadata"]["title"] == (
        "Detection of stipe rot disease in <i>Morchella sextelata</i>."
    )


def test_doubly_encoded_abstract_is_decoded_all_the_way():
    # The real case that rules out a single fixed pass: `<` and `>` as comparison operators.
    row = dict(BASE_ROW, abstract="dataset size (ranging from &amp;lt;50 to &amp;gt;25,000 subjects)")
    assert build_document(row)["publication_metadata"]["abstract"] == (
        "dataset size (ranging from <50 to >25,000 subjects)"
    )


def test_ampersand_entity_is_decoded():
    row = dict(BASE_ROW, title="H&amp;E-based MSI/MMR testing with AI in colorectal cancer.")
    assert build_document(row)["publication_metadata"]["title"] == (
        "H&E-based MSI/MMR testing with AI in colorectal cancer."
    )


def test_raw_markup_and_plain_text_titles_are_left_exactly_alone():
    # 1.48% of real titles already carry raw <i>/<sub>; decoding must be a no-op for them.
    for title in (
        "Active learning with non-<i>ab initio</i> features toward efficient CO<sub>2</sub> reduction.",
        "A perfectly ordinary title with no entities at all",
        "Reporting p < 0.05 and n > 100 in plain text",
    ):
        row = dict(BASE_ROW, title=title)
        assert build_document(row)["publication_metadata"]["title"] == title


def test_decoding_is_idempotent():
    once = build_document(dict(BASE_ROW, title="&lt;i&gt;in vitro&lt;/i&gt; study"))
    twice = build_document(dict(BASE_ROW, title=once["publication_metadata"]["title"]))
    assert once["publication_metadata"]["title"] == twice["publication_metadata"]["title"]


def test_authors_and_journal_are_deliberately_not_decoded():
    # Measured 0.000% affected on the real corpus -- leaving them untouched keeps the repair
    # surface as small as the evidence justifies.
    row = dict(BASE_ROW, authors="Smith J &amp; Jones A", journal="Journal of &lt;i&gt;Things&lt;/i&gt;")
    pub = build_document(row)["publication_metadata"]
    assert pub["authors"] == "Smith J &amp; Jones A"
    assert pub["journal"] == "Journal of &lt;i&gt;Things&lt;/i&gt;"
