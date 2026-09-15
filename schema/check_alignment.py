#!/usr/bin/env python3
"""Reports whether the authored schema (here), the published schema (dome-ml-observatory) and the
live database agree. Read-only everywhere. Exit 1 on drift.

    python3 schema/check_alignment.py
    python3 schema/check_alignment.py --live
    python3 schema/check_alignment.py --observatory-dir ../dome-ml-observatory
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCHEMA_PY = REPO / "mongo_landscape_export" / "scripts" / "schema.py"
TEMPLATE = REPO / "mongo_landscape_export" / "schema" / "ai_ml_landscape.schema.json"
# Owns the element shape of data_links.links[] and .resources[] (LINK_KEYS, RESOURCE_KEYS).
BUILD_DATA_LINKS = REPO / "moros_pipeline" / "scripts" / "build_data_links.py"
VOCABS = {
    "domain.json": REPO / "curation_criteria" / "domain_vocab.json",
    "modelling-branch.json": REPO / "curation_criteria" / "modelling_branch_vocab.json",
    "model-type-seed.json": REPO / "curation_criteria" / "model_type_seed_vocab.json",
}


def authored_version() -> str:
    m = re.search(r'^SCHEMA_VERSION\s*=\s*"([^"]+)"', SCHEMA_PY.read_text(encoding="utf-8"), re.M)
    if not m:
        raise SystemExit(f"SCHEMA_VERSION not found in {SCHEMA_PY}")
    return m.group(1)


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


def authored_tuple(name: str) -> tuple[str, ...]:
    """A tuple-of-strings constant read out of build_data_links.py without importing it (its imports
    need the pipeline's dependencies; this check needs none)."""
    tree = ast.parse(BUILD_DATA_LINKS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"{name} not found in {BUILD_DATA_LINKS}")


def element_keys(schema: dict, array: str) -> tuple[set[str], set[str]]:
    """(declared, required) item keys of data_links.<array> in a published JSON Schema. The leaf
    comparison stops at an array, so without this the element shape could drift unseen."""
    group = ((schema.get("properties") or {}).get("data_links") or {}).get("properties") or {}
    items = (group.get(array) or {}).get("items") or {}
    return set(items.get("properties") or {}), set(items.get("required") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observatory-dir", type=Path,
                        default=Path(os.environ.get("DOME_OBSERVATORY_DIR", REPO.parent / "dome-ml-observatory")))
    parser.add_argument("--live", action="store_true", help="Also read schema_version off moros (read-only).")
    args = parser.parse_args()

    problems: list[str] = []
    here_version = authored_version()
    print(f"authored  SCHEMA_VERSION (schema.py):          {here_version}")

    obs = args.observatory_dir
    current_file = obs / "schema" / "CURRENT"
    if not current_file.exists():
        problems.append(f"published: {current_file} not found (pass --observatory-dir or set DOME_OBSERVATORY_DIR)")
        published_version = None
    else:
        published_version = current_file.read_text(encoding="utf-8").strip().lstrip("v")
        print(f"published CURRENT (dome-ml-observatory):       {published_version}")
        if published_version != here_version:
            def _parts(v: str) -> tuple[int, ...]:
                try:
                    return tuple(int(x) for x in v.split("."))
                except ValueError:
                    return ()
            if _parts(here_version) > _parts(published_version):
                fix = f"cut release v{here_version} in dome-ml-observatory (its schema-version skill)"
            else:
                fix = (f"the published schema is AHEAD: author v{published_version} here "
                       "(schema.py + write_schema_template.py + tests), then migrate the corpus in "
                       "place -- see ROADMAP.md")
            problems.append(f"version drift: authored {here_version} vs published {published_version} -> {fix}")

        release = obs / "schema" / "releases" / f"v{published_version}"
        schema_json = release / "ai-ml-landscape.schema.json"
        if schema_json.exists() and TEMPLATE.exists():
            here_paths = template_paths(json.loads(TEMPLATE.read_text(encoding="utf-8")))
            published = json.loads(schema_json.read_text(encoding="utf-8"))
            there_paths = jsonschema_paths(published)
            only_here = sorted(here_paths - there_paths)
            only_there = sorted(there_paths - here_paths)
            print(f"field paths: authored {len(here_paths)}, published {len(there_paths)}")
            if only_here:
                problems.append("fields authored but not published: " + ", ".join(only_here))
            if only_there:
                problems.append("fields published but not authored: " + ", ".join(only_there))
            for array, constant in (("links", "LINK_KEYS"), ("resources", "RESOURCE_KEYS")):
                declared, required = element_keys(published, array)
                if not declared:
                    continue
                built = set(authored_tuple(constant))
                if built - declared:
                    problems.append(f"data_links.{array}[] keys built but not published: "
                                    + ", ".join(sorted(built - declared)))
                if required - built:
                    problems.append(f"data_links.{array}[] keys published as required but not built: "
                                    + ", ".join(sorted(required - built)))
        else:
            problems.append(f"published schema JSON or authored template missing ({schema_json}, {TEMPLATE})")

        for published_name, authored_path in VOCABS.items():
            p = release / "vocab" / published_name
            if not p.exists():
                problems.append(f"published vocab missing: {p}")
                continue
            a = json.loads(authored_path.read_text(encoding="utf-8"))
            b = json.loads(p.read_text(encoding="utf-8"))
            if a == b:
                print(f"vocab {published_name}: identical")
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

    if args.live:
        sys.path.insert(0, str(REPO / "moros_pipeline" / "scripts"))
        try:
            from moros_client import Moros  # noqa: E402
            with Moros.from_env() as moros:
                dist = list(moros.collection.aggregate([{"$group": {"_id": "$schema_version", "n": {"$sum": 1}}}]))
            live = {d["_id"]: d["n"] for d in dist}
            print(f"live schema_version distribution (moros):     {live}")
            others = {k: v for k, v in live.items() if k != here_version}
            if others:
                problems.append(f"live documents not at authored version {here_version}: {others} "
                                f"-> an in-place migration is due (see migrate_v1_2_0.py for the pattern)")
        except Exception as exc:  # connection, env, VPN
            problems.append(f"live check failed: {type(exc).__name__}: {exc}")

    if problems:
        print("\nDRIFT:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\naligned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
