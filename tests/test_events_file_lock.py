"""Tests for the one-writer-per-event-log lock.

This exists because documentation did not prevent the failure it guards against. On 2026-09-03 a
retry loop treated a detached `docker compose run`'s non-zero exit as failure -- while its
container was still running -- and started another. Ten ended up running concurrently against one
events file: 5,871 rows for 3,359 records, **$9.63 of duplicated paid work**. Nothing was lost
(events stream per record), but everything past the first container was bought twice.

`flock` makes that impossible rather than inadvisable.
"""

from __future__ import annotations

import fcntl
import inspect
import os
import time

import pytest

from dome_triage.pipeline.steps import _events_file_lock


def test_the_lock_is_held_for_the_duration(tmp_path):
    events = tmp_path / "events.csv"
    with _events_file_lock(events):
        lock_path = events.with_suffix(events.suffix + ".lock")
        assert lock_path.exists()
        with lock_path.open("a+") as probe:
            with pytest.raises(OSError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_a_second_run_is_refused_and_told_who_holds_it(tmp_path):
    events = tmp_path / "events.csv"
    with _events_file_lock(events):
        with pytest.raises(ValueError, match="already being written by another run"):
            with _events_file_lock(events):
                pytest.fail("the second run must not get in")


def test_the_refusal_names_the_holding_pid(tmp_path):
    """A person hitting this needs to know what to look for, not just that they were blocked."""
    events = tmp_path / "events.csv"
    with _events_file_lock(events):
        with pytest.raises(ValueError) as exc:
            with _events_file_lock(events):
                pass
    assert f"pid={os.getpid()}" in str(exc.value)


def test_ignore_lock_forces_entry_and_says_what_it_is_overriding(tmp_path, capsys):
    events = tmp_path / "events.csv"
    with _events_file_lock(events):
        with _events_file_lock(events, ignore_lock=True):
            pass
    out = capsys.readouterr().out
    assert "--ignore-lock" in out and "paying for the same records" in out


def test_the_lock_is_released_on_exit(tmp_path):
    events = tmp_path / "events.csv"
    with _events_file_lock(events):
        pass
    with _events_file_lock(events):  # must not raise
        pass


def test_the_lock_is_released_when_the_body_raises(tmp_path):
    """A crashed run must not strand a lock that blocks the legitimate retry -- resumability is
    the whole recovery story."""
    events = tmp_path / "events.csv"
    with pytest.raises(RuntimeError):
        with _events_file_lock(events):
            raise RuntimeError("network died mid-run")
    with _events_file_lock(events):
        pass


def test_a_dead_holder_does_not_strand_the_lock(tmp_path):
    """The kernel releases flock when the holding *process* dies -- including SIGKILL, which is
    the case of a container the docker daemon loses. A stranded lock would block the legitimate
    retry, and resumability is the whole recovery story.

    Exercised with a real forked child rather than multiprocessing primitives: an earlier version
    of this test used multiprocessing.Event and hung indefinitely inside the container, which is
    exactly the kind of test that costs more than it protects.
    """
    events = tmp_path / "events.csv"
    lock_path = events.with_suffix(events.suffix + ".lock")

    pid = os.fork()
    if pid == 0:  # child: take the lock, then sit still until killed
        try:
            handle = lock_path.open("a+")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(f"pid={os.getpid()} child\n")
            handle.flush()
            time.sleep(60)
        finally:
            os._exit(0)

    try:
        deadline = time.time() + 10
        held = False
        while time.time() < deadline:
            with lock_path.open("a+") as probe:
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
                except OSError:
                    held = True
                    break
            time.sleep(0.05)
        assert held, "child never acquired the lock"

        os.kill(pid, 9)
        os.waitpid(pid, 0)

        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with _events_file_lock(events):
                    return  # acquired after the kill -- the point of the test
            except ValueError:
                time.sleep(0.05)
        pytest.fail("lock was still held after the holder was killed")
    finally:
        try:
            os.kill(pid, 9)
            os.waitpid(pid, os.WNOHANG)
        except (ProcessLookupError, ChildProcessError):
            pass


def test_both_paid_steps_take_the_lock_and_expose_the_override():
    from dome_triage import cli
    from dome_triage.pipeline import steps

    for step in (steps.step_llm_classify_classify, steps.step_llm_classify_enrich):
        assert "ignore_lock" in inspect.signature(step).parameters
        assert "_events_file_lock" in inspect.getsource(step), f"{step.__name__} is unprotected"
    for command in (cli.llm_classify_classify, cli.llm_classify_enrich):
        assert "--ignore-lock" in inspect.getsource(command)


def test_enrichment_sizes_its_connection_pool_from_concurrency():
    """The bug that throttled every enrichment run this project ever did: the pool stayed at the
    default 50 no matter what --concurrency said, so threads beyond 50 queued on a connection."""
    from dome_triage.pipeline import steps

    source = inspect.getsource(steps.step_llm_classify_enrich)
    assert "_deepseek_client(concurrency)" in source


def test_the_enrich_default_concurrency_is_high_enough_to_matter():
    """Throughput is concurrency / ~45s. A low default is invisible until hours have passed."""
    import typer

    from dome_triage import cli

    default = inspect.signature(cli.llm_classify_enrich).parameters["concurrency"].default
    assert isinstance(default, typer.models.OptionInfo)
    assert default.default >= 500
