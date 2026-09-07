from dome_triage.llm_classify.response_parser import PARSE_ERROR, parse_classification


def test_parses_clean_json():
    result = parse_classification('{"classification": "positive", "rationale": "clear ML method"}')
    assert result.classification == "positive"
    assert result.rationale == "clear ML method"
    assert result.parse_fallback_used is False


def test_parses_json_wrapped_in_prose():
    result = parse_classification(
        'Here is my answer: {"classification": "negative", "rationale": "no methods applied"}'
    )
    assert result.classification == "negative"
    assert result.parse_fallback_used is True


def test_parses_json_preceded_by_think_block():
    raw = '<think>Let me consider this carefully...</think>{"classification": "undeterminable", "rationale": "too vague"}'
    result = parse_classification(raw)
    assert result.classification == "undeterminable"
    assert result.parse_fallback_used is True


def test_falls_back_to_word_match_when_no_json_present():
    result = parse_classification("Based on the abstract, I believe this is Negative overall.")
    assert result.classification == "negative"
    assert result.parse_fallback_used is True


def test_falls_back_to_word_match_for_wrong_key_names():
    result = parse_classification('{"decision": "negative", "reason": "no ML"}')
    assert result.classification == "negative"
    assert result.parse_fallback_used is True


def test_empty_string_returns_parse_error():
    result = parse_classification("")
    assert result.classification == PARSE_ERROR


def test_unparseable_text_returns_parse_error():
    result = parse_classification("I cannot determine this without more context, sorry.")
    assert result.classification == PARSE_ERROR


def test_case_insensitive_classification_value():
    result = parse_classification('{"classification": "Positive", "rationale": "x"}')
    assert result.classification == "positive"


def test_invalid_classification_value_falls_through_to_word_match():
    result = parse_classification('{"classification": "maybe", "rationale": "leaning negative overall"}')
    assert result.classification == "negative"
