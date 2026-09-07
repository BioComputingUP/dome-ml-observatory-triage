import json
import time

from dome_triage.provenance import finish_step


def test_finish_step_writes_ledger_entry_and_computes_file_refs(tmp_path, monkeypatch):
    import dome_triage.provenance as provenance_module

    monkeypatch.setattr(provenance_module, "resolve_path", lambda p: tmp_path / p)

    input_path = tmp_path / "input.csv"
    input_path.write_text("a,b\n1,2\n3,4\n")
    output_path = tmp_path / "output.csv"
    output_path.write_text("a,b\n1,2\n")

    started_at = time.monotonic()
    record = finish_step(
        "test.step",
        inputs=[input_path],
        outputs=[output_path],
        params={"threshold": 0.5},
        notes="a test run",
        started_at=started_at,
        invoked_command="dome-triage test step",
    )

    assert record.step_name == "test.step"
    assert record.params == {"threshold": 0.5}
    assert record.notes == "a test run"
    assert record.duration_seconds >= 0

    assert len(record.inputs) == 1
    assert record.inputs[0].rows == 2  # header excluded
    assert len(record.outputs) == 1
    assert record.outputs[0].rows == 1

    ledger_path = tmp_path / "data" / "provenance.jsonl"
    assert ledger_path.exists()
    lines = ledger_path.read_text().strip().splitlines()
    assert len(lines) == 1
    logged = json.loads(lines[0])
    assert logged["step_name"] == "test.step"


def test_finish_step_appends_multiple_entries(tmp_path, monkeypatch):
    import dome_triage.provenance as provenance_module

    monkeypatch.setattr(provenance_module, "resolve_path", lambda p: tmp_path / p)

    finish_step("step.one", inputs=[], outputs=[])
    finish_step("step.two", inputs=[], outputs=[])

    ledger_path = tmp_path / "data" / "provenance.jsonl"
    lines = ledger_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step_name"] == "step.one"
    assert json.loads(lines[1])["step_name"] == "step.two"


def test_finish_step_handles_missing_input_files_gracefully(tmp_path, monkeypatch):
    import dome_triage.provenance as provenance_module

    monkeypatch.setattr(provenance_module, "resolve_path", lambda p: tmp_path / p)

    record = finish_step("test.step", inputs=[tmp_path / "does_not_exist.csv"], outputs=[])
    assert record.inputs == []  # nonexistent files are simply skipped, not an error
