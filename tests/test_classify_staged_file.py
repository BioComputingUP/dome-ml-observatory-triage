"""Tests for `classify --scope staged_file` -- the incremental loop's classification step.

The property that matters most: the staged file's `pid` becomes the event log's `record_id`. That
is what lets `build_staged_documents.py` merge a verdict back onto the right document, and it is
the same identifier-space discipline `--events-out` exists to protect.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from dome_triage.llm_classify.sampling import select_staged_file


def _staged(rows: list[dict]) -> pd.DataFrame:
    base = {"pid": "p1", "pmid": "1", "pmcid": "", "doi": "10.1/a", "title": "T",
            "abstract": "A", "journal": "J", "year": "2026"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_pid_becomes_record_id():
    """The staged pid IS the Mongo _id. Deriving a different record_id here would break the merge
    back to the document, silently."""
    result = select_staged_file(_staged([{"pid": "a2321c32-4f26-5098-8038-b26a44a1c3f4"}]))
    assert result.loc[0, "record_id"] == "a2321c32-4f26-5098-8038-b26a44a1c3f4"


def test_records_pass_through_without_further_filtering():
    """Filtering already happened upstream: build_incoming_documents.py fetched only uncovered
    windows and dropped every _id already in the corpus. Re-filtering here would be wrong."""
    result = select_staged_file(_staged([{"pid": "p1"}, {"pid": "p2"}, {"pid": "p3"}]))
    assert list(result["record_id"]) == ["p1", "p2", "p3"]


def test_records_with_no_abstract_are_dropped():
    """strip_for_api would send a title-only prompt and the verdict would be a guess."""
    result = select_staged_file(_staged([{"pid": "p1"}, {"pid": "p2", "abstract": ""},
                                         {"pid": "p3", "abstract": None}]))
    assert list(result["record_id"]) == ["p1"]


def test_a_missing_required_column_is_named():
    with pytest.raises(ValueError, match="missing"):
        select_staged_file(pd.DataFrame([{"pid": "p1", "title": "T"}]))


def test_a_blank_pid_is_refused():
    """A record that cannot be merged back to a document must not be paid for."""
    with pytest.raises(ValueError, match="no pid"):
        select_staged_file(_staged([{"pid": ""}]))


def test_duplicate_pids_are_refused():
    """build_incoming_documents.py deduplicates, so duplicates mean the file came from elsewhere."""
    with pytest.raises(ValueError, match="duplicate pids"):
        select_staged_file(_staged([{"pid": "p1"}, {"pid": "p1"}]))


def test_the_blinding_boundary_still_applies():
    from dome_triage.llm_classify.sampling import strip_for_api

    row = select_staged_file(_staged([{"pid": "p1", "doi": "10.1/secret"}])).iloc[0]
    assert set(strip_for_api(row)) == {"title", "abstract", "journal", "year"}


def test_the_step_and_cli_accept_and_forward_input():
    from dome_triage import cli
    from dome_triage.pipeline import steps

    assert "input_path" in inspect.signature(steps.step_llm_classify_classify).parameters
    source = inspect.getsource(cli.llm_classify_classify)
    assert "staged_file" in source
    assert "input_path" in source.split("pipeline_steps.step_llm_classify_classify")[1]
