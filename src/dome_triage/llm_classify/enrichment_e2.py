"""Enrichment prompt e2: the e1 enrichment with GLM-5.3-Flash's measured failure modes addressed.

Validated provenance, every change from e1, and the evidence behind each are in
`curation_criteria/e2/PROVENANCE.md`. e1 (`enrichment.py`, `curation_criteria/*.json`) is untouched
and still renders to its recorded hash; e2 is a separate, versioned prompt beside it, as AGENTS.md
rule 1 requires.

What e2 is, compared with e1:
- vocabulary: `curation_criteria/e2/modelling_branch_vocab.json` and `model_type_seed_vocab.json`,
  a strict superset of e1's (new closed field `model_use`; `self-supervised`; `active learning`,
  `multimodal learning`, `physics-informed machine learning`; three new seed types). The domain
  (EDAM) vocabulary is e1's file, shared;
- prompt: `build_e2_system_text` (family hints shown, a decision order, model_type and evidence
  rules, four invented worked examples, `model_use` in the output). The user message is e1's
  `build_enrichment_prompt`, byte for byte;
- parsing: `parse_e2` runs e1's `parse_enrichment` unchanged, then three LOGGED steps (paradigm
  label repair, family backfill, model_type kind flags), so metrics can be read with and without
  them. Nothing is ever silently dropped.

The rendered system message is frozen: `EXPECTED_E2_SHA256` is the hash it was validated under
(`c3196ff7…`, called `e2-exp` in the test repository). `prompts/render_prompts.py --check` fails if it
drifts, and `e2_problems()` refuses a run on a drifted render.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from tqdm import tqdm

from dome_triage.llm_classify import enrichment as e1
from dome_triage.llm_classify.response_parser import PARSE_ERROR, _BRACE_BLOCK_RE, _THINK_BLOCK_RE, _try_json_dict

ENRICHMENT_E2_PROMPT_VERSION = "e2"

# sha256 of the rendered e2 system message. A render that differs is not the validated e2 prompt and
# nothing is sent; changing the prompt means a new version (e3), not an edit here.
EXPECTED_E2_SHA256 = "c3196ff722d6a817c034afb7278ebf4266707ba64f98d9afbaf4454927f77c45"

E2_SUBDIR = "e2"
E2_VOCAB_FILES = ("modelling_branch_vocab.json", "model_type_seed_vocab.json")

E2_LIST_FIELDS = (*e1.LIST_FIELDS, "model_use")
# e1's columns, plus `model_use` after the six tag fields and `family_backfilled` after the
# violations: the order the validation events were written in.
E2_EVENT_COLUMNS = [*e1.EVENT_COLUMNS[:7], "model_use", *e1.EVENT_COLUMNS[7:9], "family_backfilled",
                    *e1.EVENT_COLUMNS[9:]]


# ---------------------------------------------------------------------------
# Vocabulary and the static system message
# ---------------------------------------------------------------------------

def load_e2_vocabularies(criteria_dir: Path) -> dict:
    """e1's domain vocabulary plus e2's two modelling vocabularies from `<criteria_dir>/e2/`."""
    criteria_dir = Path(criteria_dir)
    vocabs = e1.load_vocabularies(criteria_dir)
    vocabs["modelling_branch"] = json.loads((criteria_dir / E2_SUBDIR / E2_VOCAB_FILES[0]).read_text(encoding="utf-8"))
    vocabs["model_type_seed"] = json.loads((criteria_dir / E2_SUBDIR / E2_VOCAB_FILES[1]).read_text(encoding="utf-8"))
    return vocabs


def _field_block(name: str, field: dict) -> str:
    lines = [f"- {t['label']}" + (f": {t['definition']}" if t.get("definition") else "") for t in field["terms"]]
    return f"{name} (choose at most {field['max_tags']}). {field['guidance']}\n" + "\n".join(lines)


def _seed_block_with_families(seed: dict) -> str:
    lines = []
    for entry in seed["terms"]:
        aliases = ", ".join(entry.get("aliases", []))
        hint = (entry.get("family_hint") or "").strip()
        lines.append(f"- {entry['canonical']}" + (f" (aliases: {aliases})" if aliases else "")
                     + (f" -> family: {hint}" if hint else ""))
    return (
        "Canonical model-type spellings. When the paper's method matches one of these, by any listed alias or "
        "spelling variant, output the canonical spelling shown. The arrow gives the model_family term(s) that "
        "type belongs to (a paradigm name such as 'reinforcement' after the arrow goes in learning_paradigm, not "
        "model_family). Methods NOT listed here are still valid model types -- tag them verbatim as the paper "
        "names them:\n" + "\n".join(lines)
    )


_DECISION_ORDER = """MODELLING FIELDS -- decide them in this order, from the title and abstract only:
1. model_use: for each model the paper reports, is it trained in this study, fine-tuned/adapted, applied as-is, or prompted?
2. learning_paradigm: how the models trained IN THIS PAPER learn. If the paper also has a distinct second learning step -- unsupervised clustering, embedding or dimensionality reduction feeding or alongside a supervised model, or self-supervised pretraining before supervised fine-tuning -- list both paradigms. If nothing is trained in this paper, use [].
3. model_type: every distinct ML model the paper uses, develops, applies or prompts (rules below).
4. model_family: the family of EACH model type you listed, using the arrows in the canonical list. Every listed model type must have its family listed -- e.g. random forest, XGBoost, gradient boosting, CatBoost, bagging or stacking always also need 'ensemble learning'; PCA, UMAP, t-SNE or NMF always also need 'dimensionality reduction'. A plain 'neural network'/ANN/MLP is 'classic neural network' unless the abstract says it is deep or names a deep architecture. If more than 3 families apply, keep the 3 that best describe the paper's main models."""

_MODEL_TYPE_RULES = """MODEL_TYPE RULES -- a model type is a machine-learning model, algorithm or named trained model/tool. Do NOT list:
- explainability methods: SHAP, LIME, permutation importance, Grad-CAM;
- statistical tests, packages and analyses: Limma, DESeq2, LEfSe, redundancy analysis, LISA, differential expression, enrichment analysis, Mendelian randomization, survival curves; regression models used only to test associations (odds ratios, p-values) rather than as the paper's predictive model;
- bioinformatics pipelines and network analyses: WGCNA, SCENIC, CellChat, NicheNet, Monocle, pseudotime/trajectory analysis;
- tasks or outputs: segmentation, object detection, named entity recognition, classification, feature selection;
- simulations and physics/chemistry methods: molecular docking, molecular dynamics, Monte Carlo, DFT, QSAR descriptors;
- model components and training techniques: attention mechanism, activation functions, loss functions, dropout, data augmentation, calibration, conformal prediction, knowledge distillation, retrieval-augmented generation, chain-of-thought;
- family names or generic phrases: 'machine learning', 'deep learning', 'ensemble machine learning', 'hybrid machine learning', 'transfer learning', 'neural network classifier' (use the canonical 'neural network');
- optimisers (particle swarm optimization, genetic algorithm, grey wolf, Bayesian optimisation, Optuna) when they only tune another model's hyperparameters -- list them only when they are the paper's own learning or search method.

EVIDENCE RULES:
- Tag only model types the abstract names. Do not add an architecture from your own knowledge of a named tool or product (e.g. do not add 'convolutional neural network' for a commercial AI tool whose architecture the abstract does not give): tag the tool's name verbatim.
- For model_family you may use well-established knowledge of a named model (AlphaFold2 -> deep learning; GPT-4 -> large language model), but not guesses about an unnamed one.
- If the abstract says only that 'machine learning models' or 'AI' were used and names no algorithm and no family, leave model_type and model_family as [] (the paradigm may still be stated or clearly implied).
- Do not list a tool that is only a benchmark or comparator the paper does not itself use."""

_EXAMPLES = """WORKED EXAMPLES (invented abstracts, for the modelling fields only):

A. "We trained random forest, XGBoost and logistic regression models on 2,000 ICU admissions to predict sepsis. LASSO selected 12 features. XGBoost performed best (AUC 0.86) and SHAP values explained its predictions."
-> model_use ["trained in this study"]; learning_paradigm ["supervised"]; model_family ["classical machine learning", "ensemble learning"]; model_type ["random forest", "XGBoost", "logistic regression", "lasso"]. (Not SHAP.)

B. "We asked GPT-4 and Llama 3 to answer 300 board-style questions with zero-shot and few-shot prompts and compared their accuracy with that of residents."
-> model_use ["prompted (zero/few-shot)"]; learning_paradigm []; model_family ["deep learning", "large language model", "generative AI"]; model_type ["GPT", "Llama 3"].

C. "We pretrained a transformer on 10 million unlabelled protein sequences by masked-token prediction, then fine-tuned it on 5,000 labelled proteins to predict thermostability."
-> model_use ["trained in this study"]; learning_paradigm ["self-supervised", "supervised"]; model_family ["deep learning", "representation learning", "transfer learning"]; model_type ["transformer"].

D. "Single-cell profiles were reduced with PCA and clustered with k-means; Limma identified marker genes; a support vector machine classified patients by response; AlphaFold2 predicted structures of the top proteins and molecular docking ranked candidate ligands."
-> model_use ["trained in this study", "pretrained, applied as-is"]; learning_paradigm ["unsupervised", "supervised"]; model_family ["classical machine learning", "dimensionality reduction", "deep learning"]; model_type ["principal component analysis", "k-means", "support vector machine", "AlphaFold2"]. (Not Limma, not docking.)"""

_DOMAIN_CAP_REMINDER = ("The domain caps are hard limits: at most 1 domain_tier1 term, at most 2 domain_tier2 terms "
                        "and at most 3 domain_tier3 terms. If more seem to apply, keep the most central ones.")

_E2_FORMAT_SPEC = (
    "Respond with a single JSON object and nothing else, in exactly this shape:\n"
    '{"domain_tier1": [...], "domain_tier2": [...], "domain_tier3": [...], '
    '"model_use": [...], "learning_paradigm": [...], "model_family": [...], "model_type": [...], '
    '"rationale": "<at most 2 sentences>"}\n'
    "Every field is a JSON array of strings (use [] when nothing applies). For the domain, model_use, "
    "learning_paradigm, and model_family fields use ONLY terms from the vocabularies above, spelled exactly as "
    "shown (e.g. 'reinforcement', not 'reinforcement learning'), and respect each field's maximum. model_type is "
    "open: prefer the canonical spellings listed above when applicable, otherwise tag the method verbatim; list "
    "every distinct model type the paper genuinely uses, develops, applies or prompts, following the model_type "
    "rules, with no maximum."
)


def build_e2_system_text(vocabs: dict) -> str:
    domain = vocabs["domain"]["fields"]
    branch = vocabs["modelling_branch"]["fields"]
    blocks = [
        e1._ENRICHMENT_PREAMBLE,
        "DOMAIN VOCABULARIES (EDAM topic branch, three depth tiers -- tag the paper's subject domain(s) at each "
        "tier where applicable):",
        e1._domain_field_block("domain_tier1", domain["domain_tier1"]),
        e1._domain_field_block("domain_tier2", domain["domain_tier2"]),
        e1._domain_field_block("domain_tier3", domain["domain_tier3"]),
        _DOMAIN_CAP_REMINDER,
        _DECISION_ORDER,
        _field_block("model_use", branch["model_use"]),
        _field_block("learning_paradigm", branch["learning_paradigm"]),
        _field_block("model_family", branch["model_family"]),
        _seed_block_with_families(vocabs["model_type_seed"]),
        _MODEL_TYPE_RULES,
        _EXAMPLES,
        _E2_FORMAT_SPEC,
    ]
    return "\n\n---\n\n".join(blocks)


def build_e2_lookup(vocabs: dict) -> dict:
    lookup = e1._build_lookup(vocabs)  # paradigm and family read from the e2 branch file
    use = vocabs["modelling_branch"]["fields"]["model_use"]
    lookup["model_use"] = {t["label"].lower(): t["label"] for t in use["terms"]}
    lookup["_caps"]["model_use"] = use["max_tags"]
    families = {t["label"].lower(): t["label"] for t in vocabs["modelling_branch"]["fields"]["model_family"]["terms"]}
    hints = {}
    for entry in vocabs["model_type_seed"]["terms"]:
        hint = entry.get("family_hint") or ""
        if " or " in hint:  # depth-dependent: not decidable without the paper, never backfilled
            continue
        labels = [families[h.strip().lower()] for h in hint.split("+") if h.strip().lower() in families]
        if labels:
            hints[entry["canonical"]] = labels
    lookup["_family_hints"] = hints
    return lookup


def render_e2(criteria_dir: Path) -> tuple[str, dict, dict]:
    vocabs = load_e2_vocabularies(criteria_dir)
    return build_e2_system_text(vocabs), vocabs, build_e2_lookup(vocabs)


def e2_problems(static_system_text: str) -> list[str]:
    """Why this render is not the validated e2 prompt (empty when it is)."""
    digest = e1.vocab_sha256(static_system_text)
    if digest != EXPECTED_E2_SHA256:
        return [f"rendered e2 system message is {digest[:12]}..., expected {EXPECTED_E2_SHA256[:12]}... "
                f"(the e2 vocabularies or this module changed: that is a new prompt version, not an edit)"]
    return []


# ---------------------------------------------------------------------------
# Parsing: e1's parser, then logged post-processing
# ---------------------------------------------------------------------------

# Heuristic kind flags for model_type items the e2 rules exclude. Flagged, never dropped.
EXCLUDED_KINDS = {
    "explainability": r"\bshap\b|shapley|\blime\b|permutation importance|grad-?cam",
    "statistics": r"\blimma\b|deseq|\blefse\b|redundancy analysis|\blisa\b|local indicators of spatial|differential expression|enrichment analysis|mendelian randomi",
    "pipeline": r"\bwgcna\b|weighted gene co-?expression|\bscenic\b|cellchat|nichenet|\bmonocle\b|pseudotime|trajectory analysis|cell2location",
    "task": r"segmentation$|named entity recognition|^classification$|feature selection|^object detection$",
    "simulation": r"docking|molecular dynamics|monte carlo|\bdft\b|density functional|simulation",
    "component": r"attention mechanism|activation|loss function|dropout|augmentation|calibration|conformal|distillation|retrieval-augmented|chain-of-thought",
    "family_name": r"^(machine learning|deep learning|ensemble machine learning|hybrid machine learning|transfer learning|artificial intelligence)( model| models| algorithms?)?$|^neural network classifier$|^deep learning model$|^machine learning model$",
    # Optimisers are legitimate when they are the learning/search method itself (E2-V8): a check, not an error.
    "optimiser_check": r"particle swarm|grey wolf|bayesian optimi[sz]ation|optuna|genetic algorithm",
}
_EXCLUDED_RE = {kind: re.compile(pattern, re.IGNORECASE) for kind, pattern in EXCLUDED_KINDS.items()}

_PARADIGM_SUFFIX = re.compile(r"\s+(machine\s+)?learning$", re.IGNORECASE)


def _raw_json(text: str) -> Optional[dict]:
    """The same three-step extraction e1's parser uses, to read model_use, which it does not know."""
    text = (text or "").strip()
    data = _try_json_dict(text)
    if data is None:
        stripped = _THINK_BLOCK_RE.sub("", text).strip()
        data = _try_json_dict(stripped)
        if data is None:
            brace = _BRACE_BLOCK_RE.search(stripped)
            data = _try_json_dict(brace.group(0)) if brace else None
    return data


def parse_e2(raw_content: str, lookup: dict) -> dict:
    parsed = e1.parse_enrichment(raw_content, lookup)
    parsed["model_use"] = []
    parsed["family_backfilled"] = []
    if parsed["parse_status"] == PARSE_ERROR:
        return parsed
    violations: list[str] = list(parsed["vocab_violations"])

    # model_use: closed field, tolerate-and-log like e1's closed fields.
    data = _raw_json(raw_content) or {}
    for value in e1._clean_str_list(data.get("model_use")):
        match = lookup["model_use"].get(value.lower())
        if match is None:
            violations.append(f"model_use:unknown:{value}")
            parsed["model_use"].append(value)
        else:
            parsed["model_use"].append(match)
    if len(parsed["model_use"]) > lookup["_caps"]["model_use"]:
        violations.append(f"model_use:cap_exceeded:{len(parsed['model_use'])}>{lookup['_caps']['model_use']}")

    # Paradigm label repair: "reinforcement learning" -> "reinforcement". Logged as repaired.
    paradigms = []
    for value in parsed["learning_paradigm"]:
        tag = f"learning_paradigm:unknown:{value}"
        repaired = lookup["learning_paradigm"].get(_PARADIGM_SUFFIX.sub("", value).lower())
        if tag in violations and repaired is not None:
            violations.remove(tag)
            violations.append(f"learning_paradigm:repaired:{value}->{repaired}")
            value = repaired
        if value not in paradigms:
            paradigms.append(value)
    parsed["learning_paradigm"] = paradigms

    # Family backfill from the seed vocabulary's family_hint. Logged in its own column.
    families = list(parsed["model_family"])
    cap = lookup["_caps"]["model_family"]
    for model_type in parsed["model_type"]:
        for family in lookup["_family_hints"].get(model_type, []):
            if family in families:
                continue
            if len(families) >= cap:
                parsed["family_backfilled"].append(f"skipped_cap:{family}<-{model_type}")
                continue
            families.append(family)
            parsed["family_backfilled"].append(f"{family}<-{model_type}")
    parsed["model_family"] = families

    for value in parsed["model_type"]:
        for kind, pattern in _EXCLUDED_RE.items():
            if pattern.search(value):
                violations.append(f"model_type:excluded_kind:{kind}:{value}")
                break

    parsed["vocab_violations"] = violations
    return parsed


# ---------------------------------------------------------------------------
# The resumable loop (e1's hard-codes prompt_version e1)
# ---------------------------------------------------------------------------

def already_enriched_ids(events: pd.DataFrame, tier: str, prompt_hash: str, retry_truncated: bool = False,
                         prompt_version: str = ENRICHMENT_E2_PROMPT_VERSION) -> set:
    """Same rules as e1's `_already_enriched_ids` (parse errors retried for free, truncations skipped
    unless `retry_truncated`), keyed on `prompt_version` (e2 unless a later prompt reuses this loop)."""
    if events.empty:
        return set()
    matches = events[(events["model_tier"] == tier) & (events["prompt_version"] == prompt_version)
                     & (events["vocab_sha256"] == prompt_hash)]
    done = set(matches[matches["parse_status"] != PARSE_ERROR]["record_id"])
    if not retry_truncated and "finish_reason" in matches.columns:
        done |= set(matches[matches["finish_reason"] == "length"]["record_id"])
    return done


def enrich_records_e2(records_df: pd.DataFrame, tier: str, client, static_system_text: str, lookup: dict,
                      batch_id: str, existing_events: Optional[pd.DataFrame] = None, max_workers: int = 50,
                      retry_truncated: bool = False, prompt_version: str = ENRICHMENT_E2_PROMPT_VERSION,
                      event_columns: Optional[list] = None) -> Iterator[dict]:
    """Yields one event per newly enriched record as it completes (never in input order). `client` is
    any object with `DeepSeekClient.chat_completion`'s signature; e2 was validated on `ZaiClient`.
    Same failure contract as e1's `enrich_records`: one failed call loses only itself, and the first
    exception is re-raised after the pool drains.

    Prompt e3 reuses this loop unchanged with `prompt_version="e3"`. Every event also carries
    `model_id`, the id the server says answered; the e2 columns do not include it, so e2's events
    files are unchanged, and e3's do."""
    existing = existing_events if existing_events is not None else pd.DataFrame(columns=event_columns or E2_EVENT_COLUMNS)
    prompt_hash = e1.vocab_sha256(static_system_text)
    done_ids = already_enriched_ids(existing, tier, prompt_hash, retry_truncated, prompt_version)
    records = [r for _, r in records_df[~records_df["record_id"].isin(done_ids)].iterrows()]

    def _one(record) -> dict:
        response = client.chat_completion(e1.build_enrichment_prompt(record, static_system_text), tier=tier,
                                          max_tokens=e1.ENRICHMENT_MAX_TOKENS, extra_body=None)
        parsed = parse_e2(response.content, lookup)
        usage = response.raw.get("usage", {}) if isinstance(response.raw, dict) else {}
        details = usage.get("completion_tokens_details") or {}
        return {
            "record_id": record["record_id"],
            **{field: json.dumps(parsed[field]) for field in E2_LIST_FIELDS},
            "rationale": parsed["rationale"],
            "vocab_violations": json.dumps(parsed["vocab_violations"]),
            "family_backfilled": json.dumps(parsed["family_backfilled"]),
            "parse_status": parsed["parse_status"],
            "model_tier": tier,
            "prompt_version": prompt_version,
            "vocab_sha256": prompt_hash,
            "batch_id": batch_id,
            "input_tokens": response.prompt_tokens,
            "output_tokens": response.completion_tokens,
            "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
            "finish_reason": response.finish_reason or "",
            "cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens", 0)),
            "parse_fallback_used": str(parsed["parse_fallback_used"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": (response.raw.get("model") if isinstance(response.raw, dict) else None) or "",
        }

    n_ok = n_err = n_viol = 0
    tokens_in = tokens_cached = 0
    first_exception: Optional[BaseException] = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_one, r) for r in records]
        with tqdm(total=len(futures), desc=f"enrich-{prompt_version}:{tier}", unit="rec") as bar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    event = future.result()
                except Exception as exc:  # noqa: BLE001 -- drained, then re-raised
                    first_exception = first_exception or exc
                    bar.update(1)
                    continue
                n_err += event["parse_status"] == PARSE_ERROR
                n_ok += event["parse_status"] != PARSE_ERROR
                n_viol += len(json.loads(event["vocab_violations"]))
                tokens_in += event["input_tokens"]
                tokens_cached += event["cache_hit_tokens"]
                bar.set_postfix(ok=n_ok, err=n_err, viol=n_viol,
                                cache=f"{(tokens_cached / tokens_in * 100) if tokens_in else 0:.0f}%")
                bar.update(1)
                yield event
    if first_exception is not None:
        raise first_exception
