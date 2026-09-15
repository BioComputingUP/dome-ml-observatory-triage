#!/usr/bin/env python3
"""Reports whether the authored schema (here), the published schema (dome-ml-observatory) and the
live database agree, and whether every place a release writes its version says the same one.
Read-only everywhere. Exit 1 on drift.

    python3 schema/check_alignment.py
    python3 schema/check_alignment.py --live
    python3 schema/check_alignment.py --observatory-dir ../dome-ml-observatory

The release procedure these checks hold both repositories to is in schema/README.md.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
PROCEDURE = "schema/README.md, 'Release procedure'"

# The observatory apps' hand-kept copies of CURRENT, used when schema/CURRENT cannot be read.
FALLBACK_CONSTANT_FILES = (
    "observatory-ws/src/common/schema-version.ts",
    "observatory-ui/src/app/core/schema-links.ts",
)
FALLBACK_RE = re.compile(r"FALLBACK_SCHEMA_VERSION\s*=\s*['\"]([^'\"]+)['\"]")


class Authored:
    """Every authored-side file the checks read, rooted at one checkout of this repository."""

    def __init__(self, repo: Path) -> None:
        self.schema_py = repo / "mongo_landscape_export" / "scripts" / "schema.py"
        self.template = repo / "mongo_landscape_export" / "schema" / "ai_ml_landscape.schema.json"
        # Owns the element shape of data_links.links[] and .resources[] (LINK_KEYS, RESOURCE_KEYS).
        self.build_data_links = repo / "moros_pipeline" / "scripts" / "build_data_links.py"
        # Owns WRITE_MODES, where each release's migration mode is declared.
        self.moros_write = repo / "moros_pipeline" / "scripts" / "moros_write.py"
        self.snapshot = repo / "schema" / "observatory_release"
        self.vocabs = {
            "domain.json": repo / "curation_criteria" / "domain_vocab.json",
            "modelling-branch.json": repo / "curation_criteria" / "modelling_branch_vocab.json",
            "model-type-seed.json": repo / "curation_criteria" / "model_type_seed_vocab.json",
        }


def authored_version(schema_py: Path) -> str:
    m = re.search(r'^SCHEMA_VERSION\s*=\s*"([^"]+)"', schema_py.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"SCHEMA_VERSION not found in {schema_py}")
    return m.group(1)


def migration_mode(version: str) -> str:
    """The WRITE_MODES key a release's in-place migration runs under: 1.6.0 -> migrate_v1_6_0."""
    return "migrate_v" + version.replace(".", "_")


def template_paths(node, prefix="") -> set[str]:
    """Leaf field paths of the reference template (an empty document)."""
    out: set[str] = set()
    if isinstance(node, dict) and node:
        for k, v in node.items():
            out |= template_paths(v, f"{prefix}{k}.")
    else:
        out.add(prefix.rstrip("."))
    return out


def jsonschema_paths(schema: dict, prefix="") -> set[str]:
    """Leaf field paths declared by a draft-07 JSON Schema's `properties`."""
    out: set[str] = set()
    props = schema.get("properties") or {}
    for k, v in props.items():
        if isinstance(v, dict) and v.get("properties"):
            out |= jsonschema_paths(v, f"{prefix}{k}.")
        else:
            out.add(f"{prefix}{k}")
    return out


def _module_assignment(path: Path, name: str) -> ast.expr:
    """The value assigned to a module-level constant, read without importing the module (its
    imports need the pipeline's dependencies; this check needs none)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name and node.value is not None:
            return node.value
    raise SystemExit(f"{name} not found in {path}")


def authored_tuple(path: Path, name: str) -> tuple[str, ...]:
    return tuple(ast.literal_eval(_module_assignment(path, name)))


def write_mode_names(path: Path) -> set[str]:
    """The keys of moros_write.WRITE_MODES (the values are expressions, so only the keys are read)."""
    value = _module_assignment(path, "WRITE_MODES")
    if not isinstance(value, ast.Dict):
        raise SystemExit(f"WRITE_MODES in {path} is not a dict literal")
    return {k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def element_keys(schema: dict, array: str) -> tuple[set[str], set[str]]:
    """(declared, required) item keys of data_links.<array> in a published JSON Schema. The leaf
    comparison stops at an array, so without this the element shape could drift unseen."""
    group = ((schema.get("properties") or {}).get("data_links") or {}).get("properties") or {}
    items = (group.get(array) or {}).get("items") or {}
    return set(items.get("properties") or {}), set(items.get("required") or [])


def _parts(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return ()


def _files_under(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def static_problems(repo: Path, obs: Path, say: Callable[[str], None] = print) -> list[str]:
    """Every check that needs no database: authored vs published, and the version each release
    writes into the files around it. `say` receives the informational lines."""
    authored = Authored(repo)
    problems: list[str] = []
    here_version = authored_version(authored.schema_py)
    say(f"authored  SCHEMA_VERSION (schema.py):          {here_version}")

    if authored.template.exists():
        template = json.loads(authored.template.read_text(encoding="utf-8"))
        if template.get("schema_version") != here_version:
            problems.append(f"template schema_version {template.get('schema_version')} is not the "
                            f"authored {here_version} -> run write_schema_template.py")
    else:
        template = None
        problems.append(f"authored template missing: {authored.template}")

    if migration_mode(here_version) not in write_mode_names(authored.moros_write):
        problems.append(f"no {migration_mode(here_version)!r} in moros_write.WRITE_MODES -> every "
                        f"release restamps moros; add the mode and its migration ({PROCEDURE})")

    current_file = obs / "schema" / "CURRENT"
    if not current_file.exists():
        problems.append(f"published: {current_file} not found (pass --observatory-dir or set DOME_OBSERVATORY_DIR)")
        return problems

    published_version = current_file.read_text(encoding="utf-8").strip().lstrip("v")
    say(f"published CURRENT (dome-ml-observatory):       {published_version}")
    if published_version != here_version:
        if _parts(here_version) > _parts(published_version):
            fix = f"cut release v{here_version} in dome-ml-observatory (its schema-version skill)"
        else:
            fix = (f"the published schema is AHEAD, which the procedure forbids: author "
                   f"v{published_version} here before anything else ({PROCEDURE})")
        problems.append(f"version drift: authored {here_version} vs published {published_version} -> {fix}")

    release = obs / "schema" / "releases" / f"v{published_version}"
    if not release.is_dir():
        problems.append(f"published CURRENT names v{published_version}, but {release} does not exist")
        return problems

    schema_json = release / "ai-ml-landscape.schema.json"
    if schema_json.exists() and template is not None:
        here_paths = template_paths(template)
        published = json.loads(schema_json.read_text(encoding="utf-8"))
        there_paths = jsonschema_paths(published)
        only_here = sorted(here_paths - there_paths)
        only_there = sorted(there_paths - here_paths)
        say(f"field paths: authored {len(here_paths)}, published {len(there_paths)}")
        if only_here:
            problems.append("fields authored but not published: " + ", ".join(only_here))
        if only_there:
            problems.append("fields published but not authored: " + ", ".join(only_there))
        for array, constant in (("links", "LINK_KEYS"), ("resources", "RESOURCE_KEYS")):
            declared, required = element_keys(published, array)
            if not declared:
                continue
            built = set(authored_tuple(authored.build_data_links, constant))
            if built - declared:
                problems.append(f"data_links.{array}[] keys built but not published: "
                                + ", ".join(sorted(built - declared)))
            if required - built:
                problems.append(f"data_links.{array}[] keys published as required but not built: "
                                + ", ".join(sorted(required - built)))
    elif not schema_json.exists():
        problems.append(f"published schema JSON missing: {schema_json}")

    example = release / "ai-ml-landscape.example.json"
    if example.exists():
        example_version = json.loads(example.read_text(encoding="utf-8")).get("schema_version")
        if example_version != published_version:
            problems.append(f"release v{published_version} example carries schema_version "
                            f"{example_version}")
    else:
        problems.append(f"published example missing: {example}")

    changelog = obs / "schema" / "CHANGELOG.md"
    if not changelog.exists() or not re.search(rf"^## v{re.escape(published_version)}\b",
                                                changelog.read_text(encoding="utf-8"), re.M):
        problems.append(f"schema/CHANGELOG.md has no '## v{published_version}' entry")

    for rel in FALLBACK_CONSTANT_FILES:
        path = obs / rel
        m = FALLBACK_RE.search(path.read_text(encoding="utf-8")) if path.exists() else None
        if m is None:
            problems.append(f"FALLBACK_SCHEMA_VERSION not found in {rel}")
        elif m.group(1).lstrip("v") != published_version:
            problems.append(f"{rel} FALLBACK_SCHEMA_VERSION is {m.group(1)}, CURRENT is "
                            f"{published_version}")

    for published_name, authored_path in authored.vocabs.items():
        p = release / "vocab" / published_name
        if not p.exists():
            problems.append(f"published vocab missing: {p}")
            continue
        a = json.loads(authored_path.read_text(encoding="utf-8"))
        b = json.loads(p.read_text(encoding="utf-8"))
        if a == b:
            say(f"vocab {published_name}: identical")
        else:
            extra = sorted(set(a) - set(b))
            missing = sorted(set(b) - set(a))
            detail = []
            if extra:
                detail.append("top-level keys only authored: " + ", ".join(extra))
            if missing:
                detail.append("top-level keys only published: " + ", ".join(missing))
            if a.get("fields") != b.get("fields") or a.get("terms") != b.get("terms"):
                detail.append("term content differs")
            problems.append(f"vocab drift {published_name} <- {authored_path.name}: " + ("; ".join(detail) or "content differs"))

    snapshot_current = authored.snapshot / "CURRENT"
    snapshot_release = authored.snapshot / f"v{published_version}"
    refresh = "refresh it with the schema-sync skill"
    if not snapshot_current.exists() or \
            snapshot_current.read_text(encoding="utf-8").strip().lstrip("v") != published_version:
        problems.append(f"schema/observatory_release/ is not a snapshot of v{published_version} -> {refresh}")
    elif not snapshot_release.is_dir() or _files_under(snapshot_release) != _files_under(release):
        problems.append(f"schema/observatory_release/v{published_version} differs from the "
                        f"published release -> {refresh}")
    return problems


def live_problems(repo: Path, here_version: str) -> list[str]:
    sys.path.insert(0, str(repo / "moros_pipeline" / "scripts"))
    try:
        from moros_client import Moros  # noqa: E402
        with Moros.from_env() as moros:
            dist = list(moros.collection.aggregate([{"$group": {"_id": "$schema_version", "n": {"$sum": 1}}}]))
        live = {d["_id"]: d["n"] for d in dist}
        print(f"live schema_version distribution (moros):     {live}")
        others = {k: v for k, v in live.items() if k != here_version}
        if others:
            return [f"live documents not at authored version {here_version}: {others} -> run "
                    f"{migration_mode(here_version)}.py ({PROCEDURE})"]
    except Exception as exc:  # connection, env, VPN
        return [f"live check failed: {type(exc).__name__}: {exc}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observatory-dir", type=Path,
                        default=Path(os.environ.get("DOME_OBSERVATORY_DIR", REPO.parent / "dome-ml-observatory")))
    parser.add_argument("--live", action="store_true", help="Also read schema_version off moros (read-only).")
    args = parser.parse_args()

    problems = static_problems(REPO, args.observatory_dir)
    if args.live:
        problems += live_problems(REPO, authored_version(Authored(REPO).schema_py))

    if problems:
        print("\nDRIFT:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\naligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
