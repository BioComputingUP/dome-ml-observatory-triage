"""Path resolution: nothing in a config file may name a machine.

Until 2026-09-07 `configs/sources.yaml` and `configs/pipeline.yaml` carried absolute
`/home/<user>/PhD_Code/...` paths, so the repository ran on exactly one laptop. They now use
`${DOME_TRIAGE_DATA_ROOT}`, which defaults to this repository's parent directory because the
sibling data repositories sit beside it -- derived from `__file__`, never stated.

The property worth pinning hardest is the failure mode: an unset variable must RAISE. Left to
expand to a literal `${VAR}` segment it would be treated as a relative path and silently resolved
under the repo root, so a misconfigured run would read or write a plausible-looking wrong place
instead of stopping.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dome_triage.config import DATA_ROOT_ENV_VAR, REPO_ROOT, data_root, resolve_path


def test_data_root_defaults_to_the_repo_parent(monkeypatch):
    """No configuration needed when the sibling repos are checked out alongside this one."""
    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    assert data_root() == REPO_ROOT.parent


def test_data_root_follows_the_environment_variable(monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, "/somewhere/else")
    assert data_root() == Path("/somewhere/else")


def test_data_root_expands_a_tilde(monkeypatch):
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, "~/data")
    assert data_root() == Path(os.path.expanduser("~/data"))


def test_config_style_path_resolves_against_the_data_root(monkeypatch):
    """The exact form configs/sources.yaml now uses."""
    monkeypatch.setenv(DATA_ROOT_ENV_VAR, "/data")
    resolved = resolve_path("${DOME_TRIAGE_DATA_ROOT}/DOME_Top_Curate/positive_entries.csv")
    assert resolved == Path("/data/DOME_Top_Curate/positive_entries.csv")


def test_unset_variable_raises_and_names_itself(monkeypatch):
    """Never a silent fallback -- see this module's docstring."""
    monkeypatch.delenv("SOME_UNSET_THING", raising=False)
    with pytest.raises(ValueError, match="SOME_UNSET_THING"):
        resolve_path("${SOME_UNSET_THING}/data.csv")


def test_relative_paths_still_resolve_under_the_repo_root():
    """The behaviour every output path in sources.yaml depends on, unchanged."""
    assert resolve_path("data/processed/canonical_dataset.csv") == (
        REPO_ROOT / "data/processed/canonical_dataset.csv"
    )


def test_absolute_paths_are_returned_unchanged():
    assert resolve_path("/tmp/somewhere/file.csv") == Path("/tmp/somewhere/file.csv")


def test_plain_paths_are_untouched_by_expansion():
    """A path with neither `~` nor `${` must take exactly the old code path -- this is what makes
    the change additive, so classification and enrichment behave identically."""
    for value in ("data/interim/x.csv", "/abs/path.csv", "curation_criteria/CRITERIA.md"):
        expected = Path(value) if Path(value).is_absolute() else REPO_ROOT / value
        assert resolve_path(value) == expected


def test_shipped_configs_contain_no_machine_specific_paths():
    """The regression this whole change exists to prevent."""
    for name in ("sources.yaml", "pipeline.yaml"):
        text = (REPO_ROOT / "configs" / name).read_text(encoding="utf-8")
        assert "/home/" not in text, f"configs/{name} names a home directory"
        assert "/Users/" not in text, f"configs/{name} names a home directory"


def test_every_sources_yaml_path_resolves(monkeypatch):
    """Each declared source path expands to something absolute, with the default data root."""
    import yaml

    monkeypatch.delenv(DATA_ROOT_ENV_VAR, raising=False)
    sources = yaml.safe_load((REPO_ROOT / "configs" / "sources.yaml").read_text(encoding="utf-8"))
    declared = [s["path"] for s in sources.get("label_sources", [])]
    declared += [r["path"] for r in sources.get("fulltext_roots", [])]
    assert declared, "sources.yaml declares no paths -- the test is not exercising anything"
    for path in declared:
        assert resolve_path(path).is_absolute()
        assert "${" not in str(resolve_path(path))
