"""Tests for the enrichment export: the cohort query and the budget gate.

Why these exist: the export is the only spend gate enrichment has (`enrich` itself has none by
design). The query must select exactly the cohort asked for -- a journal, or the batch a refresh
just loaded -- and the gate must cost what will actually be written, at the billed rate. At the old
token-model rate ($4.07/1k) it admitted about 2.5x the spend its --max-usd said, and it costed the
whole cohort even when --limit wrote a slice of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import export_journal_for_enrichment as ex


def test_journal_cohort_query():
    q = ex.build_query(["Bioinformatics (Oxford, England)"], "positive", False)
    assert q == {
        "publication_metadata.journal": {"$in": ["Bioinformatics (Oxford, England)"]},
        "llm_classification.classification": "positive",
        "llm_enrichment.provider": None,
    }


def test_batch_cohort_query_selects_by_classification_batch():
    q = ex.build_query(None, "positive", False, ["classify_flash_staged_file_primary_20260903T201216"])
    assert q["llm_classification.batch_id"] == {"$in": ["classify_flash_staged_file_primary_20260903T201216"]}
    assert "publication_metadata.journal" not in q
    assert q["llm_enrichment.provider"] is None


def test_include_enriched_drops_the_not_enriched_filter():
    q = ex.build_query(["Nature"], "positive", True)
    assert "llm_enrichment.provider" not in q


def test_any_classification_adds_no_filter():
    q = ex.build_query(None, None, False, ["b1"])
    assert "llm_classification.classification" not in q


def test_a_cohort_needs_a_journal_or_a_batch():
    with pytest.raises(ValueError):
        ex.build_query(None, "positive", False, None)
    with pytest.raises(ValueError):
        ex.build_query([], "positive", False, [])


def test_the_gate_costs_at_the_dashboard_s_planning_rate():
    """Both come from the billed log. If they drift, a cohort the dashboard prices at one figure is
    admitted or refused at another."""
    profiles = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "pricing" / "token_profiles.yaml").read_text(encoding="utf-8"))
    assert ex.USD_PER_1000_RECORDS == pytest.approx(profiles["enrichment"]["planning_usd_per_1k"])
    assert ex.USD_PER_1000_RECORDS != pytest.approx(profiles["enrichment"]["token_model_usd_per_1k"])


def test_projection_costs_the_limit_not_the_whole_cohort():
    rate = ex.USD_PER_1000_RECORDS
    assert ex.projected_usd(2_000, 200) == pytest.approx(0.2 * rate)
    assert ex.projected_usd(150, 200) == pytest.approx(0.15 * rate)
    assert ex.projected_usd(2_000, None) == pytest.approx(2 * rate)


def test_the_gate_refuses_a_cohort_over_its_limit():
    cap = ex.projected_usd(300, None)
    assert ex.projected_usd(400, None) > cap          # a bigger cohort is refused
    assert ex.projected_usd(5_000, 300) <= cap        # the same cohort capped by --limit is not


def test_parser_requires_exactly_one_kind_of_cohort():
    parser = ex.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--journal", "Nature", "--batch-id", "b1"])
    args = parser.parse_args(["--batch-id", "b1", "--batch-id", "b2", "--limit", "200"])
    assert args.batch_ids == ["b1", "b2"] and args.journals is None and args.limit == 200


def test_default_output_names():
    assert ex.default_name(["Bioinformatics (Oxford, England)"], None) == "bioinformatics_oxford_england"
    assert ex.default_name(None, ["classify_flash_staged_file_primary_20260903T201216"]).startswith("batch_classify_flash")
    assert ex.default_name(None, ["a", "b"]) == "2batches"
