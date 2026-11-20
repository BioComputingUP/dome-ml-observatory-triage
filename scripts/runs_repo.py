"""Shared helpers for the PRIVATE enrichment runs repository (gavinf97/dome-ml-observatory-enrichment-runs):
the SHA256SUMS manifest, the privacy check that must pass before anything is pushed, and the commit and
push. Used by scripts/inputs_from_zenodo.py and scripts/checkpoint_events.py; runs in the `tools`
container (stdlib + requests; git from the image).

The runs repository holds titles and abstracts, so a push is refused unless GitHub says the repository
is private. Authentication is GH_TOKEN (pass `GH_TOKEN=$(gh auth token)` to `docker compose run`); the
token is handed to git through a credential helper that reads the environment, so it is never written
to disk or to the repository's config.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import requests

TRACKED_DIRS = ("inputs", "events", "manifests")
SUMS = "SHA256SUMS"
_CRED_HELPER = "!f() { echo username=x-access-token; echo password=$GH_TOKEN; }; f"
# Every git call carries the identity, not only `commit`: `pull --rebase` re-creates local commits when the
# remote moved (2026-11-17: a workflow file was added through the API), and without an identity in the
# container it stopped mid-rebase, the pusher exited, and no checkpoint landed for 14 hours.
_IDENTITY = ("-c", "user.name=dome-enrichment-checkpoint",
             "-c", "user.email=dome-enrichment-checkpoint@users.noreply.github.com")


RUNS_REPO_NAME = "dome-ml-observatory-enrichment-runs"
# Its name until 2026-11-20. A checkout still under it (e.g. on the EBI laptop) keeps working.
_OLD_RUNS_REPO_NAME = "dome-observatory-enrichment-runs"


def default_runs_repo(repo_root: Path) -> Path:
    """`$ENRICHMENT_RUNS_REPO` (relative to this repository's root), else the sibling checkout
    `../dome-ml-observatory-enrichment-runs`, else one still under the old name."""
    env = os.environ.get("ENRICHMENT_RUNS_REPO", "").strip()
    if env:
        path = Path(env)
        return path if path.is_absolute() else (repo_root / path).resolve()
    new, old = repo_root.parent / RUNS_REPO_NAME, repo_root.parent / _OLD_RUNS_REPO_NAME
    return old if (not new.exists() and old.exists()) else new


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_sha256sums(repo: Path) -> Path:
    """Rewrites SHA256SUMS over every file in the tracked folders, sorted, `sha256sum -c` compatible."""
    lines = []
    for folder in TRACKED_DIRS:
        base = repo / folder
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and p.name != ".gitkeep"):
            lines.append(f"{sha256_file(path)}  {path.relative_to(repo).as_posix()}")
    out = repo / SUMS
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), "-c", f"credential.helper={_CRED_HELPER}", *_IDENTITY, *args],
                          check=check, capture_output=True, text=True)


def origin_slug(repo: Path) -> str:
    url = _git(repo, "remote", "get-url", "origin").stdout.strip()
    m = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", url)
    if not m:
        raise SystemExit(f"refusing: origin {url!r} of {repo} is not a GitHub repository")
    return m.group(1)


def assert_private(repo: Path, token: str | None = None) -> None:
    token = token if token is not None else os.environ.get("GH_TOKEN", "")
    if not token:
        raise SystemExit("refusing to push: GH_TOKEN is not set (run with GH_TOKEN=$(gh auth token))")
    slug = origin_slug(repo)
    resp = requests.get(f"https://api.github.com/repos/{slug}", timeout=30,
                        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    if resp.status_code != 200 or resp.json().get("private") is not True:
        raise SystemExit(f"refusing to push: {slug} is not confirmed private (HTTP {resp.status_code})")


def commit_and_push(repo: Path, message: str) -> bool:
    """Stages the tracked folders and SHA256SUMS, commits if anything changed, pushes. False if no change."""
    assert_private(repo)
    if (repo / ".git" / "rebase-merge").exists() or (repo / ".git" / "rebase-apply").exists():
        raise SystemExit(f"refusing: a rebase is in progress in {repo}; finish or abort it first")
    _git(repo, "add", "--", *[d for d in TRACKED_DIRS if (repo / d).exists()], SUMS)
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(repo, "commit", "-q", "-m", message)
    _git(repo, "pull", "-q", "--rebase")
    _git(repo, "push", "-q")
    return True
