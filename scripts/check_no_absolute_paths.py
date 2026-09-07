#!/usr/bin/env python3
"""Fails if any tracked file names a machine-specific absolute path.

A path naming a user's home directory in a committed file makes the repository unrunnable for everyone
else and breaks silently rather than loudly. This existed here until 2026-09-07: eighteen lines of
`configs/` and sixty-two coverage-ledger entries pointed into one laptop's home directory. Config
paths now use `${DOME_TRIAGE_DATA_ROOT}` (see `src/dome_triage/config.py`, which defaults it to
this repository's parent directory), and this check is what stops one coming back.

Container paths (`/app/...`) are fine and expected: they are the same on every machine, which is
the whole point of the image.

    python3 scripts/check_no_absolute_paths.py
    python3 scripts/check_no_absolute_paths.py --quiet     # exit code only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Each pattern names a user-specific root. `/app` and other container-absolute paths are portable
# and deliberately absent.
PATTERNS = {
    # These literals do not trip the check on this file: a match needs a real path segment
    # after the slash, and a character class is not one.
    "unix home": re.compile(r"/home/[A-Za-z0-9._-]+"),
    "macOS home": re.compile(r"/Users/[A-Za-z0-9._-]+"),
    "windows drive": re.compile(r"\b[A-Za-z]:\\\\?[A-Za-z0-9._-]"),
}

# Extensions never worth scanning as text.
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".owl", ".gz", ".zip", ".ico", ".woff",
                   ".woff2", ".ttf", ".parquet"}


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                         capture_output=True, text=True, check=True).stdout
    return [REPO / name for name in out.split("\0") if name]


def scan(path: Path) -> list[tuple[int, str, str]]:
    if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable: nothing a human put a path into deliberately
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in PATTERNS.items():
            match = pattern.search(line)
            if match:
                hits.append((number, label, line.strip()[:150]))
                break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quiet", action="store_true", help="Exit code only, print nothing.")
    args = parser.parse_args()

    failures = 0
    for path in tracked_files():
        for number, label, line in scan(path):
            failures += 1
            if not args.quiet:
                rel = path.relative_to(REPO)
                print(f"{rel}:{number}: {label}: {line}")

    if failures:
        if not args.quiet:
            print(f"\n{failures} machine-specific absolute path(s) in tracked files.")
            print("Use ${DOME_TRIAGE_DATA_ROOT}/... for a sibling data repository (it defaults to "
                  "this repo's parent), or a path relative to the repository root.")
        return 1
    if not args.quiet:
        print(f"no machine-specific absolute paths in {len(tracked_files()):,} tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
