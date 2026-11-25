"""Hermetic tests for scripts/checkpoint_events.py's loop -- no git, no network.

What must never go wrong: a failed `git pull`/`push` (the network dropped) ending the 15-minute loop, so a
multi-day run goes on with no backup until someone notices (2026-11-25: three hours).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import checkpoint_events as ce  # noqa: E402


class _Stop(Exception):
    pass


def _run_loop(monkeypatch, failures, every="900"):
    calls = []

    def fake_checkpoint(cohort, runs_repo):
        calls.append(cohort)
        if failures:
            raise failures.pop(0)
        return 1

    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 2:
            raise _Stop

    monkeypatch.setattr(ce, "checkpoint", fake_checkpoint)
    monkeypatch.setattr(ce.time, "sleep", fake_sleep)
    with pytest.raises(_Stop):
        ce.main(["--cohort", "x", "--runs-repo", "/nonexistent", "--every", every])
    return calls


def test_a_failed_push_does_not_end_the_loop(monkeypatch, capsys):
    err = subprocess.CalledProcessError(1, ["git", "push"], stderr="fatal: unable to access: Could not resolve host\n")
    assert len(_run_loop(monkeypatch, [err])) == 2
    assert "CHECKPOINT FAILED" in capsys.readouterr().out


def test_a_refused_push_does_not_end_the_loop(monkeypatch, capsys):
    assert len(_run_loop(monkeypatch, [SystemExit("refusing: not private")])) == 2
    assert "CHECKPOINT REFUSED" in capsys.readouterr().out


def test_a_single_checkpoint_still_raises(monkeypatch):
    monkeypatch.setattr(ce, "checkpoint", lambda c, r: (_ for _ in ()).throw(
        subprocess.CalledProcessError(1, ["git", "push"], stderr="x")))
    with pytest.raises(subprocess.CalledProcessError):
        ce.main(["--cohort", "x", "--runs-repo", "/nonexistent"])
