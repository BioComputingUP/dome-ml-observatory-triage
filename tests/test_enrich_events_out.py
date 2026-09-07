"""Regression test for `llm-classify enrich --events-out`.

The option was declared on the CLI command but never forwarded to the step, so the step always
read and wrote `cfg.path("enrichment_classification_events")` -- the Step 20j trial log -- no
matter what was passed. `classify` forwards its own `--events-out` correctly; only `enrich` did
not.

Why it matters beyond tidiness: a journal-scoped enrichment run exported from Mongo uses the
document `_id` (a UUID5) as its `record_id`, while the trial log's `record_id` is the sha1 from
`canonical_dataset.csv`. Appending one to the other conflates two identifier spaces in a single
event log, and `_already_enriched_ids` then resumes across both.

The step is exercised with the DeepSeek call stubbed out -- these tests must never make a paid API
call, and per AGENTS.md must not depend on the real data files either.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from dome_triage.llm_classify import enrichment as llm_enrichment
from dome_triage.pipeline import steps as pipeline_steps


def test_step_accepts_an_events_out_argument():
    params = inspect.signature(pipeline_steps.step_llm_classify_enrich).parameters
    assert "events_out" in params, (
        "step_llm_classify_enrich must accept events_out, or the CLI option is silently dropped"
    )


def test_cli_forwards_events_out_to_the_step():
    from dome_triage import cli

    source = inspect.getsource(cli.llm_classify_enrich)
    assert "events_out" in source.split("pipeline_steps.step_llm_classify_enrich")[1], (
        "the CLI declares --events-out but does not pass it on"
    )


def test_events_out_is_used_for_both_the_write_and_the_resume_read(tmp_path, monkeypatch):
    """The property that matters: a custom path is where events land AND where already-done
    records are read from. Reading the default while writing a custom path would silently
    re-enrich, and re-pay for, everything."""
    events_out = tmp_path / "journal_run_events.csv"
    default_events = tmp_path / "default_events.csv"

    # A pre-existing event in the CUSTOM log, for a record we are about to ask to enrich.
    done = pd.DataFrame([{c: "" for c in llm_enrichment.EVENT_COLUMNS}])
    done.loc[0, "record_id"] = "already-done"
    done.loc[0, "parse_status"] = "ok"
    done.to_csv(events_out, index=False)

    seen: dict = {}

    def fake_enrich_records(records, tier, client, static_system_text, lookup, batch_id,
                            existing_events=None, max_workers=100, **kwargs):
        seen["record_ids"] = list(records["record_id"])
        seen["existing"] = list(existing_events["record_id"]) if existing_events is not None else []
        return iter(())

    monkeypatch.setattr(llm_enrichment, "enrich_records", fake_enrich_records)
    monkeypatch.setattr(pipeline_steps, "_deepseek_client", lambda *a, **k: _NullClient())
    monkeypatch.setattr(pipeline_steps, "finish_step", lambda *a, **k: None)

    source = tmp_path / "input.csv"
    pd.DataFrame([
        {"record_id": "already-done", "title": "T", "abstract": "A", "journal": "J", "year": "2024"},
        {"record_id": "new-one", "title": "T2", "abstract": "A2", "journal": "J", "year": "2024"},
    ]).to_csv(source, index=False)

    cfg = _StubConfig(default_events)
    pipeline_steps.step_llm_classify_enrich(
        cfg, tier="flash", concurrency=1, limit=None, input_path=source,
        events_out=str(events_out),
    )

    # The resume read came from the custom log, not the configured default.
    assert seen["existing"] == ["already-done"]
    # ...and the configured default was never created.
    assert not default_events.exists()


def test_without_events_out_the_configured_default_is_still_used(tmp_path, monkeypatch):
    default_events = tmp_path / "default_events.csv"
    seen: dict = {}

    def fake_enrich_records(records, tier, client, static_system_text, lookup, batch_id,
                            existing_events=None, max_workers=100, **kwargs):
        seen["existing"] = list(existing_events["record_id"]) if existing_events is not None else []
        return iter(())

    monkeypatch.setattr(llm_enrichment, "enrich_records", fake_enrich_records)
    monkeypatch.setattr(pipeline_steps, "_deepseek_client", lambda *a, **k: _NullClient())
    monkeypatch.setattr(pipeline_steps, "finish_step", lambda *a, **k: None)

    done = pd.DataFrame([{c: "" for c in llm_enrichment.EVENT_COLUMNS}])
    done.loc[0, "record_id"] = "from-the-default-log"
    done.loc[0, "parse_status"] = "ok"
    done.to_csv(default_events, index=False)

    source = tmp_path / "input.csv"
    pd.DataFrame([{"record_id": "r1", "title": "T", "abstract": "A", "journal": "J",
                   "year": "2024"}]).to_csv(source, index=False)

    pipeline_steps.step_llm_classify_enrich(
        _StubConfig(default_events), tier="flash", concurrency=1, limit=None, input_path=source,
    )
    assert seen["existing"] == ["from-the-default-log"]


class _NullClient:
    def close(self) -> None:
        pass


class _StubConfig:
    """Only what the step actually reads. Deliberately not the real PipelineConfig -- a test that
    resolves real config paths is one edit away from writing to real curation data, which this
    repo has been burned by before (AGENTS.md, the smoketest incident)."""

    def __init__(self, events_path: Path) -> None:
        self._events_path = events_path

    def path(self, key: str) -> Path:
        if key == "enrichment_classification_events":
            return self._events_path
        raise KeyError(f"_StubConfig has no path for {key!r}")
