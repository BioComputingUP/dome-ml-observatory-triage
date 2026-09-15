"""Tests for check_alignment.py's static checks, over a fake pair of repositories in tmp_path.

Each test starts from a fully aligned pair and breaks exactly one thing, so a check that stops
firing, or starts firing on an aligned pair, fails here rather than on release day.

    (cd schema && python3 -m pytest .)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import check_alignment as ca

VERSION = "1.6.0"

VOCABS = {
    "domain.json": ("domain_vocab.json", {"fields": {"domain_tier1": {"terms": ["Biology"]}}}),
    "modelling-branch.json": ("modelling_branch_vocab.json", {"fields": {"model_family": {}}}),
    "model-type-seed.json": ("model_type_seed_vocab.json", {"terms": [{"canonical": "SVM"}]}),
}

PUBLISHED_SCHEMA = {
    "properties": {
        "_id": {},
        "schema_version": {},
        "record_modified": {},
        "identifiers": {"properties": {"doi": {}}},
        "data_links": {"properties": {
            "links": {"items": {"properties": {"resource": {}, "id": {}}, "required": ["resource"]}},
            "resources": {"items": {"properties": {"resource": {}}}},
        }},
    },
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json(path: Path, obj) -> None:
    _write(path, json.dumps(obj, indent=2) + "\n")


@pytest.fixture
def pair(tmp_path):
    repo, obs = tmp_path / "triage", tmp_path / "observatory"

    _write(repo / "mongo_landscape_export/scripts/schema.py", f'SCHEMA_VERSION = "{VERSION}"\n')
    _json(repo / "mongo_landscape_export/schema/ai_ml_landscape.schema.json", {
        "_id": None, "schema_version": VERSION, "record_modified": None,
        "identifiers": {"doi": None}, "data_links": {"links": [], "resources": []},
    })
    _write(repo / "moros_pipeline/scripts/build_data_links.py",
           'LINK_KEYS = ("resource", "id")\nRESOURCE_KEYS = ("resource",)\n')
    _write(repo / "moros_pipeline/scripts/moros_write.py",
           'WRITE_MODES: dict[str, frozenset[str]] = {\n'
           '    "citations": frozenset({"publication_metadata.citation_count"}),\n'
           '    "migrate_v1_6_0": frozenset({"schema_version", "record_modified"}),\n}\n')
    for published_name, (authored_name, content) in VOCABS.items():
        _json(repo / "curation_criteria" / authored_name, content)

    release = obs / "schema/releases" / f"v{VERSION}"
    _write(obs / "schema/CURRENT", f"v{VERSION}\n")
    _json(release / "ai-ml-landscape.schema.json", PUBLISHED_SCHEMA)
    _json(release / "ai-ml-landscape.example.json", {"_id": "x", "schema_version": VERSION})
    for published_name, (_, content) in VOCABS.items():
        _json(release / "vocab" / published_name, content)
    _write(obs / "schema/CHANGELOG.md", f"# Changelog\n\n## v{VERSION} — 2026-09-15\n\nAdded.\n")
    _write(obs / "observatory-ws/src/common/schema-version.ts",
           f"const FALLBACK_SCHEMA_VERSION = '{VERSION}';\n")
    _write(obs / "observatory-ui/src/app/core/schema-links.ts",
           f"export const FALLBACK_SCHEMA_VERSION = '{VERSION}';\n")

    snapshot = repo / "schema/observatory_release"
    shutil.copytree(release, snapshot / f"v{VERSION}")
    _write(snapshot / "CURRENT", f"v{VERSION}\n")
    return repo, obs


def problems(pair) -> list[str]:
    return ca.static_problems(*pair, say=lambda _: None)


def _one(pair, fragment: str) -> None:
    found = problems(pair)
    assert len(found) == 1 and fragment in found[0], found


def test_an_aligned_pair_has_no_problems(pair):
    assert problems(pair) == []


def test_authored_ahead_says_to_cut_the_release_there(pair):
    repo, _ = pair
    _write(repo / "mongo_landscape_export/scripts/schema.py", 'SCHEMA_VERSION = "1.7.0"\n')
    found = problems(pair)
    assert any("cut release v1.7.0" in p for p in found), found
    assert any("template schema_version 1.6.0" in p for p in found), found
    assert any("'migrate_v1_7_0'" in p for p in found), found


def test_published_ahead_is_named_as_forbidden(pair):
    repo, _ = pair
    _write(repo / "mongo_landscape_export/scripts/schema.py", 'SCHEMA_VERSION = "1.5.1"\n')
    assert any("AHEAD" in p for p in problems(pair))


def test_a_field_authored_but_not_published(pair):
    repo, _ = pair
    path = repo / "mongo_landscape_export/schema/ai_ml_landscape.schema.json"
    template = json.loads(path.read_text())
    template["identifiers"]["zenodo"] = None
    _json(path, template)
    _one(pair, "fields authored but not published: identifiers.zenodo")


def test_a_data_link_key_built_but_not_published(pair):
    repo, _ = pair
    _write(repo / "moros_pipeline/scripts/build_data_links.py",
           'LINK_KEYS = ("resource", "id", "url")\nRESOURCE_KEYS = ("resource",)\n')
    _one(pair, "data_links.links[] keys built but not published: url")


def test_a_missing_migration_mode(pair):
    repo, _ = pair
    _write(repo / "moros_pipeline/scripts/moros_write.py",
           'WRITE_MODES = {"citations": frozenset()}\n')
    _one(pair, "'migrate_v1_6_0'")


def test_a_stale_fallback_constant(pair):
    _, obs = pair
    _write(obs / "observatory-ui/src/app/core/schema-links.ts",
           "export const FALLBACK_SCHEMA_VERSION = '1.5.1';\n")
    _one(pair, "schema-links.ts FALLBACK_SCHEMA_VERSION is 1.5.1")


def test_a_release_without_a_changelog_entry(pair):
    _, obs = pair
    _write(obs / "schema/CHANGELOG.md", "# Changelog\n\n## v1.5.1 — 2026-09-15\n")
    _one(pair, "no '## v1.6.0' entry")


def test_an_example_stamped_with_another_version(pair):
    _, obs = pair
    _json(obs / "schema/releases/v1.6.0/ai-ml-landscape.example.json", {"schema_version": "1.5.1"})
    found = problems(pair)
    assert any("example carries schema_version 1.5.1" in p for p in found), found


def test_vocab_drift(pair):
    repo, _ = pair
    _json(repo / "curation_criteria/domain_vocab.json", {"fields": {"domain_tier1": {"terms": []}}})
    _one(pair, "vocab drift domain.json")


def test_a_stale_snapshot(pair):
    repo, _ = pair
    _write(repo / "schema/observatory_release/CURRENT", "v1.5.1\n")
    _one(pair, "is not a snapshot of v1.6.0")


def test_a_snapshot_whose_files_differ_from_the_release(pair):
    repo, _ = pair
    _json(repo / "schema/observatory_release/v1.6.0/ai-ml-landscape.example.json", {"schema_version": "1.6.0", "_id": "y"})
    _one(pair, "differs from the published release")


def test_current_naming_a_release_that_does_not_exist(pair):
    _, obs = pair
    _write(obs / "schema/CURRENT", "v9.9.9\n")
    assert any("does not exist" in p for p in problems(pair))
