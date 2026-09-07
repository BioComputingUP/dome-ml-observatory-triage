"""Tests for mapping enrichment events onto document fields."""

from __future__ import annotations

from load_enrichment import event_to_update, iter_updates
from moros_write import WRITE_MODES

BASE = {
    "record_id": "a2321c32-4f26-5098-8038-b26a44a1c3f4",
    "domain_tier1": '["Computational biology"]',
    "domain_tier2": '["Bioinformatics", "Machine learning"]',
    "domain_tier3": '["Protein interactions"]',
    "learning_paradigm": '["supervised"]',
    "model_family": '["deep learning"]',
    "model_type": '["convolutional neural network"]',
    "rationale": "A deep learning framework for PPI prediction.",
    "vocab_violations": "[]",
    "parse_status": "ok",
    "model_tier": "flash",
    "prompt_version": "e1",
    "vocab_sha256": "41db952f1511",
    "batch_id": "enrich_flash_20260903T000000",
    "input_tokens": "3174",
    "output_tokens": "1988",
    "cache_hit_tokens": "2900",
    "parse_fallback_used": "False",
    "timestamp": "2026-09-03T00:00:00+00:00",
}


def test_domain_tier1_is_unwrapped_from_a_list_to_a_scalar():
    """The event stores every vocab field as a list because the parser treats them uniformly, but
    domain_tier1 is max_tags=1 and the schema declares it a scalar."""
    _, fields = event_to_update(BASE)
    assert fields["content_filters.domain_tier1"] == "Computational biology"


def test_an_empty_domain_tier1_becomes_null_not_an_empty_string():
    _, fields = event_to_update({**BASE, "domain_tier1": "[]"})
    assert fields["content_filters.domain_tier1"] is None


def test_list_fields_stay_lists():
    _, fields = event_to_update(BASE)
    assert fields["content_filters.model_type"] == ["convolutional neural network"]
    assert fields["content_filters.domain_tier3"] == ["Protein interactions"]


def test_token_counts_are_written_as_integers():
    # They are aggregated and compared downstream; "9" > "10" lexically.
    _, fields = event_to_update(BASE)
    assert fields["llm_enrichment.input_tokens"] == 3174
    assert fields["llm_enrichment.cache_hit_tokens"] == 2900
    assert isinstance(fields["llm_enrichment.output_tokens"], int)


def test_parse_fallback_used_is_a_real_boolean():
    _, fields = event_to_update(BASE)
    assert fields["llm_enrichment.parse_fallback_used"] is False
    _, truthy = event_to_update({**BASE, "parse_fallback_used": "True"})
    assert truthy["llm_enrichment.parse_fallback_used"] is True


def test_provider_and_model_id_are_derived_not_read():
    _, fields = event_to_update(BASE)
    assert fields["llm_enrichment.provider"] == "deepseek"
    assert fields["llm_enrichment.model_id"] == "deepseek-v4-flash"


def test_mode_stays_null_because_enrichment_has_no_modes():
    _, fields = event_to_update(BASE)
    assert fields["llm_enrichment.mode"] is None


def test_vocab_sha256_lands_in_ruleset_sha256():
    _, fields = event_to_update(BASE)
    assert fields["llm_enrichment.ruleset_sha256"] == "41db952f1511"


def test_parse_errors_are_skipped_entirely():
    """A half-parsed enrichment is worse than none, and re-running enrich retries them."""
    assert event_to_update({**BASE, "parse_status": "parse_error"}) is None


def test_a_row_with_no_record_id_is_skipped():
    assert event_to_update({**BASE, "record_id": ""}) is None


def test_every_written_path_is_inside_the_enrichment_allowlist():
    """The structural guarantee: enrichment cannot revise a classification, because the writer
    would refuse the path before issuing anything."""
    _, fields = event_to_update(BASE)
    allowed = WRITE_MODES["enrichment"]
    assert set(fields) <= allowed, f"outside the allowlist: {set(fields) - allowed}"
    assert not any(p.startswith("llm_classification") for p in fields)


def test_the_last_event_for_a_record_wins(tmp_path):
    """The event log is append-only and a retry appends rather than replaces, so the newest
    successful event is the verdict. This test previously asserted the *first* -- it was encoding
    a bug the docstring already contradicted."""
    import csv
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BASE))
        writer.writeheader()
        writer.writerow({**BASE, "parse_status": "parse_error"})
        writer.writerow({**BASE, "model_type": '["an earlier attempt"]'})
        writer.writerow({**BASE, "model_type": '["the newest verdict"]'})
    updates = list(iter_updates(path, limit=None))
    assert len(updates) == 1
    assert updates[0][1]["content_filters.model_type"] == ["the newest verdict"]


def test_record_order_is_first_appearance_so_limit_is_stable(tmp_path):
    """--limit must take a deterministic prefix across re-runs, not whatever order a dict
    happened to end in."""
    import csv
    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BASE))
        writer.writeheader()
        for rid in ("r1", "r2", "r3"):
            writer.writerow({**BASE, "record_id": rid})
        writer.writerow({**BASE, "record_id": "r1", "model_type": '["updated"]'})
    assert [u[0] for u in iter_updates(path, limit=None)] == ["r1", "r2", "r3"]
    assert [u[0] for u in iter_updates(path, limit=2)] == ["r1", "r2"]


def test_malformed_json_in_a_field_yields_an_empty_list_not_a_crash():
    _, fields = event_to_update({**BASE, "model_type": "not json at all"})
    assert fields["content_filters.model_type"] == []
