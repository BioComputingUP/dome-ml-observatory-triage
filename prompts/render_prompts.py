#!/usr/bin/env python3
"""Render the exact system messages the engine sends, and verify the committed copies.

Imports the engine's own prompt builders from `src/` -- no reimplementation, so what is written
here is by construction what `dome-triage llm-classify classify|enrich` sends. Host Python is
enough: the two modules need only the standard library plus pandas/tqdm/requests, which the
host-side scripts already require.

    python3 prompts/render_prompts.py           # (re)write the rendered files + PROMPT_HASHES.json
    python3 prompts/render_prompts.py --check   # exit 1 if anything on disk differs from a fresh render
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dome_triage.llm_classify import enrichment, prompts  # noqa: E402

HERE = REPO / "prompts"
CRITERIA_DIR = REPO / "curation_criteria"

FILES = {
    "classification_system_message.v1.txt": None,
    "classification_forced_choice_system_message.v1.txt": None,
    "enrichment_system_message.e1.txt": None,
}


def render() -> tuple[dict[str, str], dict]:
    criteria_text = prompts.load_criteria_text(CRITERIA_DIR / "CRITERIA.md")
    primary = prompts.build_prompt({}, criteria_text)[0]["content"]
    forced = prompts.build_forced_choice_prompt({}, criteria_text)[0]["content"]
    vocabs = enrichment.load_vocabularies(CRITERIA_DIR)
    static = enrichment.build_static_system_text(vocabs, "flat")
    rendered = {
        f"classification_system_message.{prompts.PROMPT_VERSION}.txt": primary,
        f"classification_forced_choice_system_message.{prompts.PROMPT_VERSION}.txt": forced,
        f"enrichment_system_message.{enrichment.ENRICHMENT_PROMPT_VERSION}.txt": static,
    }
    hashes = {
        "classification": {
            "prompt_version": prompts.PROMPT_VERSION,
            "criteria_sha256": prompts.criteria_sha256(criteria_text),
            "criteria_file": "curation_criteria/CRITERIA.md",
            "max_tokens": 6000,
        },
        "enrichment": {
            "prompt_version": enrichment.ENRICHMENT_PROMPT_VERSION,
            "vocab_sha256": enrichment.vocab_sha256(static),
            "domain_rendering": "flat",
            "vocab_files": [
                "curation_criteria/domain_vocab.json",
                "curation_criteria/modelling_branch_vocab.json",
                "curation_criteria/model_type_seed_vocab.json",
            ],
            "vocab_file_sha256": {
                name: hashlib.sha256((CRITERIA_DIR / name).read_bytes()).hexdigest()
                for name in ("domain_vocab.json", "modelling_branch_vocab.json", "model_type_seed_vocab.json")
            },
            "max_tokens": enrichment.ENRICHMENT_MAX_TOKENS,
        },
        "model_id": "deepseek-v4-flash",
        "user_message_template": "Title: {title}\nJournal: {journal}\nYear: {year}\n\nAbstract:\n{abstract}",
    }
    return rendered, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Verify only; never write.")
    args = parser.parse_args()

    rendered, hashes = render()
    hashes_json = json.dumps(hashes, indent=2, sort_keys=True) + "\n"

    if args.check:
        stale = []
        for name, text in rendered.items():
            path = HERE / name
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        hp = HERE / "PROMPT_HASHES.json"
        if not hp.exists() or hp.read_text(encoding="utf-8") != hashes_json:
            stale.append("PROMPT_HASHES.json")
        if stale:
            print("STALE: " + ", ".join(stale))
            print("Re-render with `python3 prompts/render_prompts.py` if the assets changed on purpose "
                  "(that is a new prompt version), or restore the files if they were edited by hand.")
            return 1
        print(f"ok: classification criteria_sha256 {hashes['classification']['criteria_sha256'][:12]}…, "
              f"enrichment vocab_sha256 {hashes['enrichment']['vocab_sha256'][:12]}…")
        return 0

    for name, text in rendered.items():
        (HERE / name).write_text(text, encoding="utf-8")
        print(f"wrote {name} ({len(text):,} chars)")
    (HERE / "PROMPT_HASHES.json").write_text(hashes_json, encoding="utf-8")
    print("wrote PROMPT_HASHES.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
