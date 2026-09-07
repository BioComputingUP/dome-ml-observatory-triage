"""Config loading. Every CLI step takes its inputs from the four YAML files in configs/ rather
than hardcoded paths, so steps stay debuggable and independently re-runnable (see AGENTS.md)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the external data repositories live. They sit BESIDE this repo, so the default is derived
# from this file's own location -- nothing here names a home directory, and a fresh clone works
# with no configuration as long as the siblings are checked out alongside it. Set the environment
# variable to point somewhere else.
#
# This replaced absolute `/home/<user>/...` paths in configs/sources.yaml and configs/pipeline.yaml
# (2026-09-07), which made the repo unrunnable for anyone but its author. `scripts/
# check_no_absolute_paths.py` fails the build if one ever comes back.
DATA_ROOT_ENV_VAR = "DOME_TRIAGE_DATA_ROOT"

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def data_root() -> Path:
    """The external-data root: `$DOME_TRIAGE_DATA_ROOT` if set, else this repo's parent."""
    configured = os.environ.get(DATA_ROOT_ENV_VAR)
    return Path(configured).expanduser() if configured else REPO_ROOT.parent


def _expand(text: str) -> str:
    """Substitute `${VAR}` references, raising rather than leaving one unresolved.

    An unset variable must never silently collapse into a literal `${VAR}` segment: that would be
    resolved relative to the repo root below and quietly read (or write) the wrong place. Naming
    the variable in the error is the whole point.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == DATA_ROOT_ENV_VAR:
            return str(data_root())
        value = os.environ.get(name)
        if value is None:
            raise ValueError(
                f"{text!r} references ${{{name}}}, which is not set in the environment. "
                f"Set it, or use ${{{DATA_ROOT_ENV_VAR}}} (which defaults to this repo's parent "
                "directory) for a path to a sibling data repository."
            )
        return value

    return _ENV_REF.sub(replace, text)


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repo root, unless it's already absolute.

    Two forms are expanded first, so nothing in a config file has to name a machine:
    `~` for the current user's home, and `${VAR}` for an environment variable -- in practice
    `${DOME_TRIAGE_DATA_ROOT}`, which points at the external data repositories and defaults to
    this repo's parent directory (see `data_root`). An unset variable raises rather than resolving
    somewhere unintended.

    Source file paths in sources.yaml are then absolute (they point at sibling repos); output
    paths are repo-root-relative (e.g. "data/processed/canonical_dataset.csv"). A path containing
    neither `~` nor `${` is unaffected by the expansion step.
    """
    path = Path(_expand(os.path.expanduser(str(path))))
    return path if path.is_absolute() else REPO_ROOT / path


class PipelineConfig:
    """Loads all four config files and exposes helpers for resolving declared output paths."""

    def __init__(
        self,
        sources_path: str | Path = "configs/sources.yaml",
        pipeline_path: str | Path = "configs/pipeline.yaml",
        tfidf_path: str | Path = "configs/tfidf.yaml",
        keybert_path: str | Path = "configs/keybert.yaml",
        sampling_path: str | Path = "configs/sampling.yaml",
    ) -> None:
        self.sources = load_yaml(sources_path)
        self.pipeline = load_yaml(pipeline_path)
        self.tfidf = load_yaml(tfidf_path)
        self.keybert = load_yaml(keybert_path)
        self.sampling = load_yaml(sampling_path)

    def path(self, key: str) -> Path:
        """Resolve one of sources.yaml's top-level `paths:` entries, e.g. path("canonical_dataset")."""
        return resolve_path(self.sources["paths"][key])

    def sampling_path(self, key: str) -> Path:
        """Resolve one of sampling.yaml's top-level `paths:` entries, e.g. sampling_path("bulk_candidates")."""
        return resolve_path(self.sampling["paths"][key])

    def ensure_dirs(self) -> None:
        self.path("interim_dir").mkdir(parents=True, exist_ok=True)
        self.path("processed_dir").mkdir(parents=True, exist_ok=True)
        self.path("fulltext_manifest").parent.mkdir(parents=True, exist_ok=True)
