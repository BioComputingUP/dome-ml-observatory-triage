"""Watches a running enrichment cohort and raises an alert, a GitHub issue that emails you, when it goes wrong.

Two layers, so either one still alerts if the other is dead:

1. This watchdog, on the laptop, checks once a minute (runs in the `tools` container, read-only on the run):
   - the EBI gateway: alert after 3 consecutive minutes unreachable (VPN or Wi-Fi down), recover when back;
   - the events files: alert when no new event for 15 min while the gateway is up (the run stalled or died);
   - the run log, only what is appended after the watchdog starts: alert on "stopping for review", and
     report "all selected chunks done".
   An alert is a `repository_dispatch` to the PRIVATE runs repository. Its workflow
   (`.github/workflows/enrichment-alerts.yml`, source in scripts/runs_repo_workflows/) opens the issue as
   github-actions[bot], assigned to and mentioning the owner, because GitHub does not email you about an
   issue you opened yourself. A dispatch that cannot be sent (no internet) is retried every minute.

2. The same workflow runs on a schedule on GitHub (every 20 min): if the runs repository has had no commit
   for 45 min (checkpoints land every 15 min while records are being enriched), it opens a dead-man issue.
   That catches what the laptop cannot report: shutdown, sleep, no internet, this watchdog dead.

The gateway URL and key are read from the repository's `.env`; neither goes into an alert.

    GH_TOKEN=$(gh auth token) docker compose run --rm -e GH_TOKEN tools \\
        python scripts/run_watchdog.py --cohort batch2_high_impact
    ... --test     send one test alert and exit (proves the email arrives)
Started for you by scripts/start_watchdog.sh.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_repo import default_runs_repo, assert_private, origin_slug  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO / "moros_pipeline" / "output" / "enrichment_e3"
WORKFLOW = "enrichment-alerts.yml"
GATEWAY_DOWN_AFTER = 3          # consecutive one-minute checks
STALL_AFTER_S = 15 * 60
LOG_MARKERS = {"stopped": "stopping for review", "finished": "all selected chunks done"}


def read_env(path: Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


@dataclass
class State:
    down_streak: int = 0
    gateway_alerted: bool = False
    stall_alerted: bool = False
    markers_sent: set = field(default_factory=set)


def decide(state: State, gateway_up: bool, event_age_s: float | None, new_markers: set[str],
           down_after: int = GATEWAY_DOWN_AFTER, stall_after_s: float = STALL_AFTER_S) -> list[tuple[str, str, str]]:
    """Pure: given one check's observations, returns the (event_type, kind, title) to dispatch, and updates state.
    An alert is sent once per episode; a recovery closes it."""
    out = []
    state.down_streak = 0 if gateway_up else state.down_streak + 1
    if state.down_streak >= down_after and not state.gateway_alerted:
        state.gateway_alerted = True
        out.append(("enrichment-alert", "gateway", "EBI gateway unreachable (VPN or network down?); the run is paused"))
    elif gateway_up and state.gateway_alerted:
        state.gateway_alerted = False
        out.append(("enrichment-recovered", "gateway", "EBI gateway reachable again"))
    finished = "finished" in state.markers_sent or "finished" in new_markers
    stalled = gateway_up and not finished and event_age_s is not None and event_age_s > stall_after_s
    if stalled and not state.stall_alerted:
        state.stall_alerted = True
        out.append(("enrichment-alert", "stall", f"no new enrichment event for {event_age_s / 60:.0f} min"))
    elif state.stall_alerted and event_age_s is not None and event_age_s <= stall_after_s:
        state.stall_alerted = False
        out.append(("enrichment-recovered", "stall", "events are arriving again"))
    for marker in ("stopped", "finished"):
        if marker in new_markers and marker not in state.markers_sent:
            state.markers_sent.add(marker)
            out.append(("enrichment-alert", "stopped", "the run stopped for review (failure rate or retries)")
                       if marker == "stopped" else ("enrichment-finished", "finished", "the run finished"))
    return out


def gateway_up(base: str, key: str) -> bool:
    try:
        r = requests.get(f"{base.rstrip('/')}/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        return r.status_code == 200
    except requests.RequestException:
        return False


def newest_event_age(work: Path, now: float) -> float | None:
    files = list(work.glob("*_events.csv"))
    return None if not files else now - max(f.stat().st_mtime for f in files)


class LogTail:
    """Reads only what is appended to the run log after the watchdog starts."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0
        self.last_line = ""

    def new_markers(self) -> set[str]:
        if not self.path.exists():
            return set()
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
        with self.path.open("rb") as f:
            f.seek(self.offset)
            text = f.read().decode("utf-8", "replace")
        self.offset = size
        lines = [ln for ln in text.splitlines() if ln.startswith("===")]
        if lines:
            self.last_line = lines[-1][:300]
        return {m for m, needle in LOG_MARKERS.items() if needle in text}


def dispatch(slug: str, token: str, event: str, kind: str, title: str, body: str) -> bool:
    try:
        r = requests.post(f"https://api.github.com/repos/{slug}/dispatches", timeout=30,
                          headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
                          json={"event_type": event, "client_payload": {"kind": kind, "title": title, "body": body}})
        return r.status_code == 204
    except requests.RequestException:
        return False


def enable_workflow(slug: str, token: str) -> None:
    r = requests.put(f"https://api.github.com/repos/{slug}/actions/workflows/{WORKFLOW}/enable", timeout=30,
                     headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    if r.status_code not in (204, 200):
        raise SystemExit(f"cannot enable {WORKFLOW} in {slug} (HTTP {r.status_code}): is it installed? "
                         "See scripts/runs_repo_workflows/README.md")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--runs-repo", type=Path, default=default_runs_repo(REPO))
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--test", action="store_true", help="send one test alert and exit")
    args = ap.parse_args(argv)

    token = os.environ.get("GH_TOKEN", "")
    assert_private(args.runs_repo, token)
    slug = origin_slug(args.runs_repo)
    enable_workflow(slug, token)
    if args.test:
        ok = dispatch(slug, token, "enrichment-alert", "test", f"test alert for cohort {args.cohort}",
                      "If this reached your email, alerts work. Close this issue.")
        raise SystemExit(0 if ok else f"dispatch to {slug} failed")

    env = {**read_env(REPO / ".env"), **os.environ}
    base, key = env.get("EBI_LLM_BASE_URL", ""), env.get("EBI_LLM_API_KEY", "")
    if not base or not key:
        raise SystemExit("EBI_LLM_BASE_URL / EBI_LLM_API_KEY not found in .env")
    work = WORK_ROOT / args.cohort
    tail = LogTail(work / "run.log")
    state, pending = State(), []
    print(f"watching cohort {args.cohort}: alerts go to {slug} (issues by github-actions[bot])", flush=True)
    while True:
        now = time.time()
        up = gateway_up(base, key)
        age = newest_event_age(work, now)
        markers = tail.new_markers()
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for event, kind, title in decide(state, up, age, markers):
            body = (f"Cohort `{args.cohort}`, {stamp}.\n\n- Gateway: {'up' if up else 'unreachable'}\n"
                    f"- Last enrichment event: {'none' if age is None else f'{age / 60:.0f} min ago'}\n"
                    f"- Last run-log line: `{tail.last_line or 'n/a'}`\n\n"
                    "Watch: `tmux attach -t enrich-e3` (Ctrl-b 2 for status).")
            pending.append((event, kind, f"{title} ({args.cohort})", body))
        pending = [p for p in pending if not dispatch(slug, token, *p)]
        print(f"{stamp} gateway={'up' if up else 'DOWN'} last_event="
              f"{'n/a' if age is None else f'{age:.0f}s'} unsent={len(pending)}", flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    main()
