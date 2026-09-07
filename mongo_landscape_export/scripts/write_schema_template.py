"""Step 24 phase 1, Step 4: writes the checked-in reference copy of the MongoDB document shape --
`../schema/ai_ml_landscape.schema.json` -- an "empty" document (every field present, no real data)
for humans to look at without reading Python, and for future migrations to diff against.

Hand-authored to mirror `schema.py::build_document()`'s real output shape exactly (not generated
by calling `build_document()` on a blank row, since that function's blank/none-handling is tuned
for real CSV strings, not for representing "this field doesn't exist yet" as cleanly as a literal
template can). Run this after any change to that shape and commit the result, so the template
never silently drifts from what the code actually produces -- `test_write_schema_template.py`
guards the two staying in sync (same top-level and group-level key sets).

Usage:
    python3 write_schema_template.py [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from schema import SCHEMA_VERSION

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = THIS_DIR.parent / "schema" / "ai_ml_landscape.schema.json"


def build_template() -> dict:
    return {
        "_id": None,
        "schema_version": SCHEMA_VERSION,
        "identifiers": {
            "pmid": None,
            "pmcid": None,
            "doi": None,
            "dome_registry": None,
            "bioai_repo": None,
            "huggingface": None,
            "kaggle": None,
            "zenodo": None,
        },
        "publication_metadata": {
            "title": None,
            "abstract": None,
            "authors": None,
            "year": None,
            "journal": None,
            "citation_count": None,
            "citation_count_updated": None,
            "citation_source": None,
        },
        "source": {
            "abstract_source": None,
            "metadata_repair_sources": None,
            "decision_provenance": None,
            "access": {
                "open_access": None,
                "license": None,
                "fulltext_available": None,
            },
        },
        "content_filters": {
            "mesh_headings": [],
            "pub_types": [],
            "keywords_author": [],
            "domain_tier1": None,
            "domain_tier2": [],
            "domain_tier3": [],
            "learning_paradigm": [],
            "model_family": [],
            "model_type": [],
        },
        "llm_classification": {
            "provider": None,
            "model_tier": None,
            "model_id": None,
            "mode": None,
            "classification": None,
            "rationale": None,
            "prompt_version": None,
            "ruleset_sha256": None,
            "batch_id": None,
            "timestamp": None,
        },
        "llm_enrichment": {
            "provider": None,
            "model_tier": None,
            "model_id": None,
            "mode": None,
            "rationale": None,
            "prompt_version": None,
            "ruleset_sha256": None,
            "batch_id": None,
            "timestamp": None,
            "vocab_violations": None,
            "parse_status": None,
            "input_tokens": None,
            "output_tokens": None,
            "cache_hit_tokens": None,
            "parse_fallback_used": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_template(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
