"""Provenance ledger. Every pipeline step calls `finish_step(...)` before returning -- see
AGENTS.md's "no generated file without a provenance entry" rule. This module is deliberately a
plain function, not a decorator/context-manager: every step function already knows its own input
and output paths, so wrapping it would add indirection without saving real code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from dome_triage.config import REPO_ROOT, resolve_path


class FileRef(BaseModel):
    path: str
    rows: Optional[int] = None
    sha256: str


class ProvenanceRecord(BaseModel):
    step_name: str
    invoked_command: str
    git_commit: str
    timestamp_utc: str
    duration_seconds: float
    inputs: list[FileRef] = []
    outputs: list[FileRef] = []
    params: dict = {}
    notes: Optional[str] = None


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not commit:
            return "no-git-history"
        return f"{commit}-dirty" if dirty else commit
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _sha256_and_rows(path: Path) -> FileRef:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)

    rows = None
    if path.suffix.lower() in (".csv", ".tsv", ".jsonl"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                rows = sum(1 for _ in f) - (1 if path.suffix.lower() != ".jsonl" else 0)
        except OSError:
            rows = None

    return FileRef(path=str(path), rows=rows, sha256=hasher.hexdigest())


def _file_refs(paths: list[Path]) -> list[FileRef]:
    return [_sha256_and_rows(p) for p in paths if p.exists()]


def finish_step(
    step_name: str,
    inputs: list[Path],
    outputs: list[Path],
    params: Optional[dict] = None,
    notes: Optional[str] = None,
    started_at: Optional[float] = None,
    invoked_command: Optional[str] = None,
) -> ProvenanceRecord:
    """Call at the end of a pipeline step. Appends one line to data/provenance.jsonl and prints
    a human-readable summary. `started_at` should be `time.monotonic()` captured at the top of
    the calling step, so duration is measured accurately."""
    duration = time.monotonic() - started_at if started_at is not None else 0.0

    record = ProvenanceRecord(
        step_name=step_name,
        invoked_command=invoked_command or f"dome-triage {step_name.replace('.', ' ')}",
        git_commit=_git_commit(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        duration_seconds=round(duration, 3),
        inputs=_file_refs(inputs),
        outputs=_file_refs(outputs),
        params=params or {},
        notes=notes,
    )

    ledger_path = resolve_path("data/provenance.jsonl")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(record.model_dump_json() + "\n")

    _print_summary(record, ledger_path)
    return record


def _print_summary(record: ProvenanceRecord, ledger_path: Path) -> None:
    print(f"\n=== dome-triage: {record.step_name} ===")
    if record.inputs:
        print("Inputs:")
        for ref in record.inputs:
            rows_str = f" ({ref.rows} rows)" if ref.rows is not None else ""
            print(f"  {ref.path}{rows_str}")
    if record.params:
        print(f"Params: {json.dumps(record.params, default=str)}")
    if record.outputs:
        print("Outputs:")
        for ref in record.outputs:
            rows_str = f" ({ref.rows} rows)" if ref.rows is not None else ""
            print(f"  {ref.path}{rows_str}")
    print(f"Duration: {record.duration_seconds:.1f}s | git {record.git_commit}")
    print(f"Provenance recorded -> {ledger_path}")
