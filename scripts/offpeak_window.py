#!/usr/bin/env python3
"""Is now off-peak for DeepSeek, and will a run of N minutes started now stay off-peak?

Peak (double price): 01:00-04:00 and 06:00-10:00 UTC, Monday-Friday (pricing/pricing.yaml).

    python3 scripts/offpeak_window.py                 # status now
    python3 scripts/offpeak_window.py --minutes 860   # a 14h20m run: fits now? else next start
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

PEAK = [(1, 4), (6, 10)]  # UTC hours, Mon-Fri


def is_peak(t: datetime) -> bool:
    if t.weekday() >= 5:
        return False
    return any(a <= t.hour < b for a, b in PEAK)


def next_peak_start(t: datetime) -> datetime:
    step = t.replace(minute=0, second=0, microsecond=0)
    for _ in range(24 * 8):
        step += timedelta(hours=1)
        if is_peak(step) and not is_peak(step - timedelta(hours=1)):
            return step
    raise RuntimeError("no peak found in 8 days")


def next_offpeak_start(t: datetime) -> datetime:
    step = t.replace(minute=0, second=0, microsecond=0)
    while is_peak(step):
        step += timedelta(hours=1)
    return max(step, t)


def fits(start: datetime, minutes: float) -> bool:
    end = start + timedelta(minutes=minutes)
    step = start
    while step < end:
        if is_peak(step):
            return False
        step += timedelta(minutes=15)
    return not is_peak(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=None, help="Planned run length.")
    parser.add_argument("--now", default=None, help="Override 'now' as ISO-8601 UTC (for planning).")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.now).replace(tzinfo=timezone.utc) if args.now else datetime.now(timezone.utc)

    state = "PEAK (2x price)" if is_peak(now) else "off-peak"
    print(f"now: {now:%Y-%m-%d %H:%M} UTC ({now:%A}) -> {state}")
    if is_peak(now):
        print(f"off-peak resumes at {next_offpeak_start(now):%Y-%m-%d %H:%M} UTC")
    else:
        print(f"next peak window starts {next_peak_start(now):%Y-%m-%d %H:%M} UTC")

    if args.minutes is not None:
        if fits(now, args.minutes):
            print(f"a {args.minutes:.0f}-minute run started now stays off-peak")
        else:
            cand = next_offpeak_start(now)
            for _ in range(24 * 8 * 4):
                if not is_peak(cand) and fits(cand, args.minutes):
                    break
                cand += timedelta(minutes=15)
            print(f"a {args.minutes:.0f}-minute run started now would cross a peak window; "
                  f"earliest fully off-peak start: {cand:%Y-%m-%d %H:%M} UTC ({cand:%A})")
        print("weekends are entirely off-peak; a run that needs more than ~15 hours fits Fri 10:00 UTC -> Mon 01:00 UTC")


if __name__ == "__main__":
    main()
