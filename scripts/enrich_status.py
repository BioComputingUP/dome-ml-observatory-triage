"""Live status of an enrichment cohort run: progress, speed, ETA, quality flags, stalls, checkpoint age.

Read-only: it reads the chunk inputs and events files under the cohort's work folder
(moros_pipeline/output/enrichment_e3/<cohort>/, where scripts/run_enrichment.sh puts them), the run log
beside them, and the runs repository's last commit time. It never writes, so it can run beside a live
run at any time, as often as wanted.

    docker compose run --rm tools python scripts/enrich_status.py --cohort core_bioinformatics
    docker compose run --rm tools python scripts/enrich_status.py --cohort core_bioinformatics --once
    ... --once --check     exit 1 if any finished chunk has parse errors or truncations above 2%
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

csv.field_size_limit(sys.maxsize)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_repo import default_runs_repo  # noqa: E402

WORK_ROOT = REPO / "moros_pipeline" / "output" / "enrichment_e3"
PARSE_ERROR = "parse_error"
THRESHOLD = 0.02
RECENT = timedelta(minutes=5)
STALL = timedelta(minutes=3)


@dataclass
class Chunk:
    name: str
    total: int
    done: int = 0
    parse_errors: int = 0
    truncated: int = 0
    flagged: int = 0
    out_tokens: int = 0
    stamps: list[datetime] = field(default_factory=list)


def _count_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def read_chunk(input_path: Path) -> Chunk:
    chunk = Chunk(input_path.stem, _count_rows(input_path))
    events = input_path.with_name(f"{input_path.stem}_events.csv")
    if not events.exists():
        return chunk
    last: dict[str, dict] = {}  # the log is append-only; the last event per record is its state
    with events.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last[row["record_id"]] = row
            ts = _parse_ts(row.get("timestamp", ""))
            if ts:
                chunk.stamps.append(ts)
    for row in last.values():
        if row.get("parse_status") == PARSE_ERROR:
            chunk.parse_errors += 1
            continue
        chunk.done += 1
        chunk.truncated += row.get("finish_reason") == "length"
        chunk.flagged += (row.get("vocab_violations") or "[]") != "[]"
        try:
            chunk.out_tokens += int(float(row.get("output_tokens") or 0))
        except ValueError:
            pass
    return chunk


def checkpoint_age(runs_repo: Path, now: datetime) -> str:
    try:
        out = subprocess.run(["git", "-C", str(runs_repo), "log", "-1", "--format=%ct"], capture_output=True,
                             text=True, check=True).stdout.strip()
        return _fmt_td(now - datetime.fromtimestamp(int(out), tz=timezone.utc)) + " ago"
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return "unknown"


def _fmt_td(td: timedelta) -> str:
    s = max(int(td.total_seconds()), 0)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _pct(n: int, d: int) -> str:
    return f"{n:,} ({n / d:.1%})" if d else f"{n:,}"


def _bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def log_problems(work: Path) -> tuple[int, int]:
    log = work / "run.log"
    if not log.exists():
        return 0, 0
    text = log.read_text(encoding="utf-8", errors="replace")
    return text.count("CRASHED"), text.count("API error")


def render(cohort: str, work: Path, runs_repo: Path, now: datetime) -> tuple[str, bool]:
    inputs = sorted(p for p in work.glob("*.csv") if not p.name.endswith("_events.csv"))
    chunks = [read_chunk(p) for p in inputs]
    total = sum(c.total for c in chunks)
    done = sum(c.done for c in chunks)
    stamps = sorted(t for c in chunks for t in c.stamps)
    recent = [t for t in stamps if t >= now - RECENT]
    rate_recent = len(recent) / (RECENT.total_seconds() / 60)
    rate_all = (len(stamps) / max((stamps[-1] - stamps[0]).total_seconds() / 60, 1e-9)) if len(stamps) > 1 else 0.0
    rate = rate_recent or rate_all
    failed = sum(c.parse_errors for c in chunks)
    # A parse error is a finished call whose answer could not be read (almost always a response cut off
    # at the token cap). It is not in flight: a re-run retries it, so it is shown, not waited for.
    remaining = total - done - failed
    eta = (now + timedelta(minutes=remaining / rate)) if rate and remaining else None
    crashes, api_errors = log_problems(work)
    last_age = (now - stamps[-1]) if stamps else None
    stalled = bool(last_age and last_age > STALL and remaining)

    lines = [
        f"DOME enrichment e3 -- cohort {cohort}            {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        f"  progress   [{_bar(done / total if total else 0)}] {done:,} / {total:,} ({(done / total if total else 0):.1%})"
        + (f"   {failed:,} failed (a re-run retries them)" if failed else ""),
        f"  speed      {rate_recent:,.0f} rec/min (last 5 min)   {rate_all:,.0f} rec/min (whole run)",
        f"  ETA        {('~' + _fmt_td(eta - now) + '  -> ' + eta.strftime('%H:%M UTC')) if eta else ('done' if not remaining else 'n/a')}",
        f"  last event {(_fmt_td(last_age) + ' ago') if last_age else 'none yet'}"
        + ("   ** STALLED: no event for over 3 minutes **" if stalled else ""),
        f"  checkpoint last push {checkpoint_age(runs_repo, now)}",
        f"  run log    {crashes} crash(es), {api_errors} API error line(s)",
        "",
        f"  {'chunk':<52} {'done':>13} {'parse err':>12} {'truncated':>12} {'vocab flag':>13} {'out tok/rec':>11}",
    ]
    over = False
    for c in chunks:
        finished = c.done + c.parse_errors >= c.total
        bad = finished and c.total and ((c.parse_errors / c.total) > THRESHOLD or (c.truncated / c.total) > THRESHOLD)
        over |= bool(bad)
        lines.append(
            f"  {c.name[-52:]:<52} {f'{c.done:,}/{c.total:,}':>13} {_pct(c.parse_errors, c.total):>12} "
            f"{_pct(c.truncated, c.total):>12} {_pct(c.flagged, c.total):>13} "
            f"{(c.out_tokens / c.done if c.done else 0):>11,.0f}" + ("  <-- over 2%: stop and review" if bad else ""))
    return "\n".join(lines), over


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--work-root", type=Path, default=WORK_ROOT)
    ap.add_argument("--runs-repo", type=Path, default=default_runs_repo(REPO))
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--check", action="store_true", help="with --once: exit 1 if a finished chunk is over 2%%")
    args = ap.parse_args(argv)
    work = args.work_root / args.cohort
    if not work.exists():
        print(f"no work folder yet: {work}")
        return 1
    while True:
        text, over = render(args.cohort, work, args.runs_repo, datetime.now(timezone.utc))
        if args.once:
            print(text)
            return 1 if (args.check and over) else 0
        print("\033[2J\033[H" + text + f"\n\n  refreshing every {args.interval}s -- Ctrl-C to stop watching "
              "(the run is not affected)", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
