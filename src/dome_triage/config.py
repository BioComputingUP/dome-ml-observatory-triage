"""Config loading. Every CLI step takes its inputs from the four YAML files in configs/ rather
than hardcoded paths, so steps stay debuggable and independently re-runnable (see AGENTS.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repo root, unless it's already absolute.

    Source file paths in sources.yaml are absolute (they point at sibling repos); output paths
    are repo-root-relative (e.g. "data/processed/canonical_dataset.csv").
    """
    path = Path(path)
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
