"""Checkpoints a running enrichment cohort to the PRIVATE runs repository: copies its events files and its
provenance rows, rewrites SHA256SUMS, commits and pushes. Unchanged files are skipped; a push happens only
when something changed, and is refused unless GitHub confirms the repository is private.

Copies are written to a temp file and renamed, so the runs repository never holds a half-written file.
An events file being appended to while it is copied is fine: the copy is a prefix of whole rows at worst,
and the next checkpoint (or the final one after the chunk) supersedes it.

    GH_TOKEN=$(gh auth token) docker compose run --rm tools python scripts/checkpoint_events.py \\
        --cohort core_bioinformatics            # once
    ... --every 900                             # every 15 minutes until stopped
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_repo import default_runs_repo, commit_and_push, sha256_file, write_sha256sums  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO / "moros_pipeline" / "output" / "enrichment_e3"
PROVENANCE = REPO / "data" / "provenance.jsonl"
STEP = "llm-classify.enrich-e3"


def _copy_if_changed(src: Path, dst: Path) -> bool:
    if dst.exists() and sha256_file(dst) == sha256_file(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    # Trim a trailing partial line, in case the writer was mid-row when we read.
    data = tmp.read_bytes()
    if data and not data.endswith(b"\n"):
        tmp.write_bytes(data[: data.rfind(b"\n") + 1])
    tmp.replace(dst)
    return True


def _provenance_rows(cohort: str) -> str:
    if not PROVENANCE.exists():
        return ""
    keep = []
    for line in PROVENANCE.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = json.dumps(row)
        if row.get("step_name") == STEP and f"/{cohort}/" in text:
            keep.append(line)
    return "\n".join(keep) + ("\n" if keep else "")


def checkpoint(cohort: str, runs_repo: Path, work_root: Path = WORK_ROOT, push: bool = True) -> int:
    work = work_root / cohort
    changed = 0
    for events in sorted(work.glob("*_events.csv")):
        changed += _copy_if_changed(events, runs_repo / "events" / cohort / events.name)
    prov = _provenance_rows(cohort)
    if prov:
        target = runs_repo / "manifests" / cohort / "provenance.jsonl"
        if not target.exists() or target.read_text(encoding="utf-8") != prov:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(prov, encoding="utf-8")
            tmp.replace(target)
            changed += 1
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if not changed:
        print(f"{stamp}: nothing changed")
        return 0
    write_sha256sums(runs_repo)
    if push:
        pushed = commit_and_push(runs_repo, f"checkpoint {cohort}: {changed} file(s), {stamp}")
        print(f"{stamp}: {changed} file(s) changed; {'pushed' if pushed else 'nothing to commit'}")
    return changed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--runs-repo", type=Path, default=default_runs_repo(REPO))
    ap.add_argument("--every", type=int, default=0, help="seconds between checkpoints; 0 = once")
    args = ap.parse_args(argv)
    while True:
        try:
            checkpoint(args.cohort, args.runs_repo)
        except SystemExit as exc:  # a refused push must be loud but must not kill the loop silently
            print(f"CHECKPOINT REFUSED: {exc}", flush=True)
            if not args.every:
                raise
        except (subprocess.CalledProcessError, OSError) as exc:
            # A network drop fails `git pull`/`push` (2026-11-25: the pusher died at 15:46 and nothing was
            # pushed for 3 h). The commit is kept locally; the next round pushes it with whatever is new.
            detail = (getattr(exc, "stderr", "") or "").strip().splitlines()[-1:] or [str(exc)]
            print(f"CHECKPOINT FAILED (retrying in {args.every}s): {detail[0]}", flush=True)
            if not args.every:
                raise
        if not args.every:
            return
        time.sleep(args.every)


if __name__ == "__main__":
    main()
