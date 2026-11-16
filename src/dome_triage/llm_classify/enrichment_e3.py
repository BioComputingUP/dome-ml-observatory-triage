"""Enrichment prompt e3: prompt e2 with the life-science model types it lacked, and nothing else.

Everything but the model-type seed vocabulary is e2's: the same builder (`build_e2_system_text`), the
same lookup and parser (`build_e2_lookup`, `parse_e2`), the same resumable loop (`enrich_records_e2`
with `prompt_version="e3"`) and the same domain and modelling-branch vocabularies. The only file that
differs is `curation_criteria/e3/model_type_seed_vocab.json`, which adds seed entries and aliases chosen
by the pre-agreed rule (>= 100 positives in the corpus, not covered by an existing term, a genuine model);
the evidence is `docs/e3_model_type_evidence.md` and the diff is `curation_criteria/e3/PROVENANCE.md`.
The rendered e3 prompt therefore differs from e2's by added seed lines only, which a test checks.

e3 is the production prompt from 2026-11-16, run on GLM-5.3-Flash on EMBL-EBI's gateway
(`zai_client.EbiClient`). Its events add one column to e2's: `model_id`, the id the server reports.

The rendered system message is frozen: `EXPECTED_E3_SHA256` is its hash, `prompts/render_prompts.py
--check` fails if it drifts, and `e3_problems()` refuses a run on a drifted render. Changing it is e4.
"""

from __future__ import annotations

import json
from pathlib import Path

from dome_triage.llm_classify import enrichment as e1
from dome_triage.llm_classify import enrichment_e2 as e2

ENRICHMENT_E3_PROMPT_VERSION = "e3"

EXPECTED_E3_SHA256 = "3b9d119deb05834f404e541493ba01e221553ad59f23031e8a004c88bccd0239"

E3_SUBDIR = "e3"
E3_VOCAB_FILES = e2.E2_VOCAB_FILES

E3_EVENT_COLUMNS = [*e2.E2_EVENT_COLUMNS, "model_id"]


def load_e3_vocabularies(criteria_dir: Path) -> dict:
    """e1's domain vocabulary plus e3's two modelling vocabularies from `<criteria_dir>/e3/`."""
    criteria_dir = Path(criteria_dir)
    vocabs = e1.load_vocabularies(criteria_dir)
    for key, name in zip(("modelling_branch", "model_type_seed"), E3_VOCAB_FILES):
        vocabs[key] = json.loads((criteria_dir / E3_SUBDIR / name).read_text(encoding="utf-8"))
    return vocabs


def render_e3(criteria_dir: Path) -> tuple[str, dict, dict]:
    vocabs = load_e3_vocabularies(criteria_dir)
    return e2.build_e2_system_text(vocabs), vocabs, e2.build_e2_lookup(vocabs)


def e3_problems(static_system_text: str) -> list[str]:
    """Why this render is not the frozen e3 prompt (empty when it is)."""
    digest = e1.vocab_sha256(static_system_text)
    if digest != EXPECTED_E3_SHA256:
        return [f"rendered e3 system message is {digest[:12]}..., expected {EXPECTED_E3_SHA256[:12]}... "
                f"(the e3 vocabularies or the e2 builder changed: that is a new prompt version, not an edit)"]
    return []


parse_e3 = e2.parse_e2
