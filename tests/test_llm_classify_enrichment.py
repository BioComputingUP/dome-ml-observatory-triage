import json

import pandas as pd

from dome_triage.llm_classify.enrichment import (
    ENRICHMENT_PROMPT_VERSION,
    EVENT_COLUMNS,
    LIST_FIELDS,
    PARSE_ERROR,
    _already_enriched_ids,
    _build_lookup,
    build_enrichment_prompt,
    build_static_system_text,
    parse_enrichment,
    vocab_sha256,
)


def _vocabs() -> dict:
    return {
        "domain": {
            "fields": {
                "domain_tier1": {"max_tags": 1, "terms": [{"edam_id": "topic_3070", "label": "Biology"}]},
                "domain_tier2": {"max_tags": 2, "terms": [{"edam_id": "topic_3512", "label": "Genomics"},
                                                          {"edam_id": "topic_3452", "label": "Imaging"}]},
                "domain_tier3": {"max_tags": 3, "terms": [{"edam_id": "topic_3308", "label": "Transcriptomics"}]},
            }
        },
        "modelling_branch": {
            "fields": {
                "learning_paradigm": {"max_tags": 2, "terms": [{"label": "supervised"}, {"label": "unsupervised"}]},
                "model_family": {"max_tags": 3, "terms": [{"label": "classical machine learning"},
                                                          {"label": "deep learning", "definition": "deep nets"}]},
            }
        },
        "model_type_seed": {
            "terms": [
                {"canonical": "XGBoost", "aliases": ["extreme gradient boosting", "xgb"]},
                {"canonical": "random forest", "aliases": ["RF"]},
            ]
        },
    }


def _lookup() -> dict:
    return _build_lookup(_vocabs())


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_static_system_text_contains_all_vocab_labels_and_caps():
    text = build_static_system_text(_vocabs())
    for label in ("Biology", "Genomics", "Transcriptomics", "supervised", "deep learning", "XGBoost"):
        assert label in text
    assert "at most 1" in text and "at most 2" in text and "at most 3" in text


def test_static_system_text_is_deterministic_for_stable_vocab_hash():
    vocabs = _vocabs()
    assert vocab_sha256(build_static_system_text(vocabs)) == vocab_sha256(build_static_system_text(vocabs))


def test_prompt_blinding_provenance_columns_never_reach_the_model():
    # The trial CSV carries label/cross_curate_notes for audit -- even when the FULL row is passed,
    # only title/abstract/journal/year may appear in the messages. Same blinding contract as
    # prompts.py, defended independently here.
    record = pd.Series(
        {
            "record_id": "r1", "title": "T-title", "abstract": "A-abstract", "journal": "J-journal",
            "year": "2020", "label": "SECRET_LABEL", "label_confidence": "human_curated",
            "cross_curate_final_label": "SECRET_FINAL", "cross_curate_notes": "SECRET_NOTES",
        }
    )
    messages = build_enrichment_prompt(record, build_static_system_text(_vocabs()))
    combined = json.dumps(messages)
    assert "T-title" in combined and "A-abstract" in combined
    for secret in ("SECRET_LABEL", "SECRET_FINAL", "SECRET_NOTES", "human_curated"):
        assert secret not in combined


def test_prompt_puts_static_text_in_system_and_record_in_user():
    messages = build_enrichment_prompt({"title": "T", "abstract": "A", "journal": "J", "year": "2020"},
                                       build_static_system_text(_vocabs()))
    assert messages[0]["role"] == "system" and "XGBoost" in messages[0]["content"]
    assert messages[1]["role"] == "user" and "T" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Parsing + validation (tolerate-and-log)
# ---------------------------------------------------------------------------


def test_parse_clean_json_all_fields():
    raw = json.dumps({
        "domain_tier1": ["Biology"], "domain_tier2": ["Genomics"], "domain_tier3": [],
        "learning_paradigm": ["supervised"], "model_family": ["deep learning"],
        "model_type": ["XGBoost"], "rationale": "Clear.",
    })
    result = parse_enrichment(raw, _lookup())
    assert result["parse_status"] == "ok"
    assert result["domain_tier1"] == ["Biology"]
    assert result["vocab_violations"] == []
    assert result["parse_fallback_used"] is False


def test_parse_normalizes_case_and_seed_aliases():
    raw = json.dumps({
        "domain_tier1": ["biology"], "domain_tier2": [], "domain_tier3": [],
        "learning_paradigm": ["SUPERVISED"], "model_family": [],
        "model_type": ["extreme gradient boosting", "rf", "NovelNet-3000"], "rationale": "",
    })
    result = parse_enrichment(raw, _lookup())
    assert result["domain_tier1"] == ["Biology"]  # case-normalized to the vocab label
    assert result["learning_paradigm"] == ["supervised"]
    assert result["model_type"] == ["XGBoost", "random forest", "NovelNet-3000"]  # aliases -> canonical, novel kept verbatim
    assert result["vocab_violations"] == []


def test_parse_flags_unknown_term_but_keeps_it():
    raw = json.dumps({
        "domain_tier1": ["Astrology"], "domain_tier2": [], "domain_tier3": [],
        "learning_paradigm": [], "model_family": [], "model_type": [], "rationale": "",
    })
    result = parse_enrichment(raw, _lookup())
    assert result["domain_tier1"] == ["Astrology"]  # kept, never silently dropped
    assert result["vocab_violations"] == ["domain_tier1:unknown:Astrology"]
    assert result["parse_status"] == "ok"


def test_parse_flags_cap_exceeded():
    raw = json.dumps({
        "domain_tier1": [], "domain_tier2": ["Genomics", "Imaging"], "domain_tier3": [],
        "learning_paradigm": ["supervised", "unsupervised"], "model_family": [],
        "model_type": [], "rationale": "",
    })
    # tier2 cap is 2 (fine), paradigm cap is 2 (fine) -- now break tier1's cap of 1:
    raw2 = json.dumps({
        "domain_tier1": ["Biology", "Biology"], "domain_tier2": [], "domain_tier3": [],
        "learning_paradigm": [], "model_family": [], "model_type": [], "rationale": "",
    })
    assert parse_enrichment(raw, _lookup())["vocab_violations"] == []
    assert "domain_tier1:cap_exceeded:2>1" in parse_enrichment(raw2, _lookup())["vocab_violations"]


def test_parse_think_block_and_prose_wrapped_json_fallbacks():
    inner = json.dumps({"domain_tier1": ["Biology"], "domain_tier2": [], "domain_tier3": [],
                        "learning_paradigm": [], "model_family": [], "model_type": [], "rationale": "r"})
    raw = f"<think>pondering...</think>Here you go: {inner}"
    result = parse_enrichment(raw, _lookup())
    assert result["parse_status"] == "ok"
    assert result["domain_tier1"] == ["Biology"]
    assert result["parse_fallback_used"] is True


def test_parse_unparseable_is_parse_error_never_coerced():
    result = parse_enrichment("I could not decide anything useful here.", _lookup())
    assert result["parse_status"] == PARSE_ERROR
    assert all(result[field] == [] for field in LIST_FIELDS)


def test_parse_non_list_field_values_become_empty_lists():
    raw = json.dumps({"domain_tier1": "Biology", "domain_tier2": None, "domain_tier3": [],
                      "learning_paradigm": [], "model_family": [], "model_type": 42, "rationale": ""})
    result = parse_enrichment(raw, _lookup())
    assert result["domain_tier1"] == [] and result["model_type"] == []


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def _events(rows: list[dict]) -> pd.DataFrame:
    defaults = {"model_tier": "flash", "prompt_version": ENRICHMENT_PROMPT_VERSION,
                "vocab_sha256": "hash_a", "parse_status": "ok"}
    return pd.DataFrame([{**defaults, **row} for row in rows], columns=EVENT_COLUMNS[:1] + list(defaults.keys())).fillna("")


def test_already_enriched_excludes_parse_errors_so_rerun_retries_them():
    events = _events([{"record_id": "r1"}, {"record_id": "r2", "parse_status": PARSE_ERROR}])
    done = _already_enriched_ids(events, "flash", "hash_a")
    assert done == {"r1"}


def test_already_enriched_keys_on_vocab_hash_so_a_vocab_edit_starts_fresh():
    events = _events([{"record_id": "r1"}])
    assert _already_enriched_ids(events, "flash", "hash_B") == set()


def test_already_enriched_empty_events():
    assert _already_enriched_ids(pd.DataFrame(columns=EVENT_COLUMNS), "flash", "hash_a") == set()
