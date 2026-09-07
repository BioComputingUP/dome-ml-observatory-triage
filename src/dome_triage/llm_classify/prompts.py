"""Prompt construction for Step 20's DeepSeek second-curator classification.

`build_prompt`/`build_forced_choice_prompt` are deliberately defensive about which record fields
they read -- each pulls only `title`/`abstract`/`journal`/`year` off the given mapping, explicitly,
by key -- so even if a caller accidentally passed a full canonical_dataset.csv row (with `label`,
`notes`, `mesh_headings`, `sources`, `curation_tag` still attached) instead of the stripped 4-field
dict `sampling.py::strip_for_api` produces, nothing beyond those four fields could ever reach the
model. `mesh_headings` is deliberately never read here even though it's harmless-looking -- a
record tagged "Machine Learning" would hand the model the answer. See
`tests/test_llm_classify_prompts.py` for the blinding test this defends.

Full `CRITERIA.md` text goes verbatim into the system message -- summarizing it risks silently
dropping exactly the nuanced boundary rules (classical-ML-counts-as-positive, the PRISMA/
systematic-review signal, the informal-LLM-use exclusion) this whole validation exists to check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPT_VERSION = "v1"

_SYSTEM_PREAMBLE = (
    "You are an independent, second-opinion curator for a biomedical-literature dataset. You will "
    "be shown one paper's title, abstract, journal, and publication year. Apply the curation "
    "criteria below to decide whether this paper describes the application, development, or "
    "validation of an AI/ML method as a primary research activity (\"positive\"), or does not "
    "(\"negative\"). Base your decision ONLY on the text given to you below and your own general "
    "knowledge -- you have no access to any external tool, search engine, or database, and you "
    "have no access to any prior human decision about this paper. Do not guess what a human "
    "curator decided; form your own independent judgment from first principles."
)

_OUTPUT_SPACE_OVERRIDE = (
    "You have exactly three possible answers: \"positive\", \"negative\", or \"undeterminable\". "
    "The criteria document below also describes a fourth category, \"Skipped\" -- that option does "
    "not apply to you here; never output it. Use \"undeterminable\" only when the title and "
    "abstract genuinely give you no reasonable basis to decide either way -- reserve it for "
    "genuinely ambiguous cases, not as a default whenever you are merely less than fully certain. "
    "An ordinary judgment call should still get a \"positive\" or \"negative\" answer."
)

_FORCED_CHOICE_OUTPUT_SPACE_OVERRIDE = (
    "You have exactly two possible answers: \"positive\" or \"negative\". You are never allowed to "
    "answer \"undeterminable\" or \"skipped\", even for a genuinely hard or ambiguous case -- "
    "commit to your single best judgment call based on everything given to you below."
)

_JSON_FORMAT_SPEC = (
    "Respond with a single JSON object and nothing else, in exactly this shape: "
    '{"classification": "<your answer>", "rationale": "<your reasoning, at most 2 sentences>"}. '
    "Do not include any text outside the JSON object."
)


def load_criteria_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def criteria_sha256(text: str) -> str:
    """Staleness detection: every classification event records the hash of the exact criteria
    text it was judged against, so a later CRITERIA.md edit is never silently conflated with an
    earlier run's results (`runner.py`'s resumability check keys on this, not just prompt_version)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_block(record) -> str:
    """Reads exactly `title`/`abstract`/`journal`/`year` off `record` (dict or pandas Series),
    explicitly, by key -- see this module's docstring for why."""
    title = record.get("title") or "(no title)"
    journal = record.get("journal") or "(unknown journal)"
    year = record.get("year") or "(unknown year)"
    abstract = record.get("abstract") or "(no abstract available)"
    return f"Title: {title}\nJournal: {journal}\nYear: {year}\n\nAbstract:\n{abstract}"


def _build(record, criteria_text: str, output_space_override: str) -> list[dict]:
    # Static content (preamble + full criteria + output-space override + format spec) in the
    # system message, first, unchanged across every call in a run -- record-specific content in
    # the user message, last. Costs nothing and is the only thing that could benefit from any
    # provider-side prefix caching, whether or not that can be trusted from this session's
    # unreliable DeepSeek research.
    system = "\n\n---\n\n".join([_SYSTEM_PREAMBLE, criteria_text, output_space_override, _JSON_FORMAT_SPEC])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _record_block(record)},
    ]


def build_prompt(record, criteria_text: str) -> list[dict]:
    """Primary, 3-way (positive/negative/undeterminable) prompt."""
    return _build(record, criteria_text, _OUTPUT_SPACE_OVERRIDE)


def build_forced_choice_prompt(record, criteria_text: str) -> list[dict]:
    """2-way (positive/negative only) variant -- used either as the primary prompt (if
    `validate-criteria`'s fixture-stage comparison shows the 3-way variant's undetermined rate is
    too high -- see STEPS_Progress.md Step 20's decision rule) or as the forced-guess fallback
    re-ask on the primary run's undetermined subset."""
    return _build(record, criteria_text, _FORCED_CHOICE_OUTPUT_SPACE_OVERRIDE)
