"""Hermetic tests for scripts/run_watchdog.py's decisions -- no network.

What must never go wrong: no alert when the VPN drops (the run is silently paused for hours), an alert
every minute for one outage (alert fatigue, then ignored), a stall alert while the gateway is the cause,
or old lines of a resumed run's log raising a fresh "stopped" alert.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import run_watchdog as rw  # noqa: E402


def kinds(actions):
    return [(event, kind) for event, kind, _ in actions]


def test_gateway_alert_after_three_minutes_once_then_recovery():
    s = rw.State()
    assert rw.decide(s, False, 10, set()) == []
    assert rw.decide(s, False, 70, set()) == []
    assert kinds(rw.decide(s, False, 130, set())) == [("enrichment-alert", "gateway")]
    assert rw.decide(s, False, 190, set()) == []                       # once per outage
    assert kinds(rw.decide(s, True, 5, set())) == [("enrichment-recovered", "gateway")]
    assert rw.decide(s, True, 5, set()) == []


def test_a_one_minute_blip_is_not_an_alert():
    s = rw.State()
    rw.decide(s, False, 10, set())
    assert rw.decide(s, True, 5, set()) == []


def test_stall_alerts_only_while_the_gateway_is_up():
    s = rw.State()
    assert rw.decide(s, False, 3600, set()) == []                      # gateway is the cause, not a stall
    s = rw.State()
    assert kinds(rw.decide(s, True, 16 * 60, set())) == [("enrichment-alert", "stall")]
    assert rw.decide(s, True, 17 * 60, set()) == []
    assert kinds(rw.decide(s, True, 3, set())) == [("enrichment-recovered", "stall")]


def test_finish_is_reported_once_and_silences_the_stall():
    s = rw.State()
    assert kinds(rw.decide(s, True, 3, {"finished"})) == [("enrichment-finished", "finished")]
    assert rw.decide(s, True, 3600, set()) == []


def test_stopped_for_review_alerts_once():
    s = rw.State()
    assert kinds(rw.decide(s, True, 3, {"stopped"})) == [("enrichment-alert", "stopped")]
    assert rw.decide(s, True, 3, {"stopped"}) == []


def test_log_tail_ignores_what_was_there_before_start(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("=== old run: a finished chunk ...; stopping for review\n")
    tail = rw.LogTail(log)
    assert tail.new_markers() == set()
    with log.open("a") as f:
        f.write("=== 2026-11-20 cohort x: all selected chunks done\n")
    assert tail.new_markers() == {"finished"}
    assert tail.last_line.startswith("=== 2026-11-20")


def test_read_env_parses_and_ignores_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# c\nEBI_LLM_BASE_URL='https://example.invalid/v1'\nEBI_LLM_API_KEY=k\n\n")
    assert rw.read_env(env) == {"EBI_LLM_BASE_URL": "https://example.invalid/v1", "EBI_LLM_API_KEY": "k"}
