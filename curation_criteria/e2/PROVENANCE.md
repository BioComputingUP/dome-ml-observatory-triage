# Prompt e2: provenance

**Status: production prompt version `e2`, adopted 2026-11-16** for GLM-5.3-Flash (Z.ai) enrichment. It was built and frozen in the `glm5.3_test_dome-ml-observatory-triage` repository, where it was called `e2-exp`. The rendered system message here is byte-identical to that frozen one: sha256 `c3196ff7…`, checked by `python3 prompts/render_prompts.py --check`. Prompt `e1` and its vocabularies in `curation_criteria/` are unchanged and still render to `41db952f…`. Every e1-enriched record stays valid, because e2's vocabularies are a strict superset of e1's.

In this repository the e2 code is `src/dome_triage/llm_classify/enrichment_e2.py`, and the vocabularies are this folder's two JSON files. The domain (EDAM) vocabulary is shared with e1 from `curation_criteria/domain_vocab.json`. Paths below that start `experimental/e2/`, `glm_enrich/` or `vendor/` refer to the test repository. Record numbers marked "Evidence" refer to that repository's review appendices.

| | |
|---|---|
| Base | Prompt e1, vocab sha256 `41db952f15118176fa27dac724eff9dff59714b62854859d9606ef440ee473c7`, vendored files as listed in `vendor/MANIFEST.sha256` |
| Why | Correctness review of GLM-5.3-Flash vs DeepSeek V4.1-Flash on enriched100, 2026-11-05: `reports/tag_correctness_review_enriched100.md` |
| Vocabulary changes approved by | Gavin, 2026-11-05 (E2-V1 to E2-V9 below) |
| Built by | `experimental/e2/build_vocab.py` (vocabularies, from `vendor/` copies) and `glm_enrich/e2.py` (prompt and parser) |
| Prompt version recorded in events | `e2-exp` |
| Rendered prompt | `experimental/e2/enrichment_system_message.e2.txt`, sha256 in `glm_enrich/e2.py` (`EXPECTED_E2_SHA256`) and below |
| File hashes | `experimental/e2/MANIFEST.sha256`, checked before every e2 run |

"Evidence" numbers refer to the record numbers in the appendix of the review. Ontology IDs were looked up on 2026-11-05: MeSH at id.nlm.nih.gov/mesh/lookup (`match=contains`) and AIO on EBI OLS4 (`ontology=aio`).

## Vocabulary changes

Each added or changed JSON entry carries `"extension": "e2-experimental"` and `"provenance": "experimental/e2/PROVENANCE.md#e2-vN"`.

### E2-V1
New closed field `model_use` (max 2): `trained in this study`, `fine-tuned or adapted`, `pretrained, applied as-is`, `prompted (zero/few-shot)`. The learning_paradigm guidance changes with it: paradigm describes only learning done in the paper, and is `[]` when nothing is trained.
- *Why:* in about 11% of papers the model is only applied or prompted, and their paradigm tags were inconsistent. AlphaFold and qER got `supervised` (how the tool was originally trained); most LLM papers got `[]`; one prompting paper got `unsupervised` (DeepSeek) and `supervised` (GLM).
- *Evidence:* 8, 23, 25, 36, 47, 56, 70, 75, 90.
- *Source:* project extension. `prompted (zero/few-shot)` is mapped to AIO:ZeroshotLearning. MeSH has no descriptor for any of the four.

### E2-V2
learning_paradigm adds `self-supervised`.
- *Why:* self-supervised and contrastive pretraining was being filed as `unsupervised` or left out.
- *Evidence:* 6, 54, 60, 77.
- *Source:* AIO:SelfsupervisedLearning. No MeSH descriptor.

### E2-V3
The `semi-supervised` definition is extended to cover weak supervision and pseudo-labelling. Definition text only; the label is unchanged.
- *Evidence:* 43, where the two models disagreed.

### E2-V4
model_family adds `active learning`.
- *Why:* closed-loop, query-driven design was being written into model_type.
- *Evidence:* 22, 49.
- *Source:* AIO:ActiveLearning. No MeSH descriptor.

### E2-V5
model_family adds `multimodal learning`.
- *Evidence:* 20, 52 (image + text fusion, multimodal deep networks).
- *Source:* AIO:MultimodalLearning. No MeSH descriptor.

### E2-V6
model_family adds `physics-informed machine learning`.
- *Evidence:* 5, 35, 96.
- *Source:* project extension. No AIO term or MeSH descriptor found.

### E2-V7
A model_type exclusion rule, in the prompt and flagged by the parser. The following are not model types:
- explainability methods (SHAP, LIME);
- statistical tests and packages (Limma, LEfSe, RDA, LISA, differential expression), and regression used only to test associations;
- bioinformatics pipelines (**WGCNA excluded**, SCENIC, CellChat, NicheNet, pseudotime);
- tasks (segmentation, NER);
- simulations and docking;
- model components and training techniques;
- family names and generic phrases.
- *Why:* GLM had 26/100 records with such items, DeepSeek 9/100, and both were inconsistent about WGCNA.
- *Decision:* excluding WGCNA was chosen by Gavin, 2026-11-05.

### E2-V8
Optimisers (PSO, GA, grey wolf, Bayesian optimisation) are listed as model types only when they are the paper's own learning or search method, not when they only tune another model.
- *Evidence:* 5, 89.
- *Decision:* chosen by Gavin, 2026-11-05. The parser flags them as `optimiser_check` for review; this is not counted as an error on its own.

### E2-V9
Additions to the seed model-type list:
- aliases `BP-ANN`, `backpropagation neural network`, `BPNN` → `neural network`;
- aliases `histogram gradient boosting`, `HistGradientBoosting` → `gradient boosting`;
- new canonical `causal forest` (family: classical machine learning + ensemble learning);
- new canonical `machine-learning interatomic potential` (no family hint: the architecture varies);
- new canonical `state space model` (alias Mamba; family: deep learning).
- *Evidence:* 5, 13, 18, 26, 64, 66, 89, 99.

Not changed: the domain (EDAM) vocabulary, every e1 term, label, alias, MeSH ID, ontology mapping and max_tags. The model_family cap stays at 3.

## Prompt changes (glm_enrich/e2.py)

| ID | Change | Targets |
|---|---|---|
| E2-P1 | Each canonical model type is shown with its family (`-> family: …`, from the seed vocabulary's existing `family_hint`, which e1 never rendered) | GLM added `ensemble learning` for tree ensembles 2/20 (enriched100) and 15/51 (wave1) times, DeepSeek 17/22; PCA → `dimensionality reduction` likewise |
| E2-P2 | Decision order: model_use → paradigm (both paradigms when there is a distinct second learning step) → types → a family for every type | GLM gave two paradigms in 5/100 records against DeepSeek's 12/100; the review found a missing family in 24 records |
| E2-P3 | model_type rules (E2-V7, E2-V8) with examples of what not to output | Non-model items in model_type |
| E2-P4 | Evidence rules: do not add architectures the abstract does not name; leave family and type `[]` when no algorithm or family is named; do not list benchmark-only tools | GLM: CNN for qER, YOLO, k-means, transformer. DeepSeek: unsupported `classical machine learning` guesses |
| E2-P5 | Four worked examples on **invented** abstracts (none taken from any cohort, to avoid leakage): tree ensemble + SHAP; prompted LLMs; self-supervised pretraining + fine-tuning; PCA + clustering + classifier + applied AlphaFold2 + excluded tools | All of the above |
| E2-P6 | Domain cap reminder; output adds `model_use`; the format line spells out "reinforcement, not reinforcement learning" | domain_tier2 cap violations, the one paradigm label violation |

Unchanged from e1: the preamble, the domain blocks, the user message (the vendored `build_enrichment_prompt`), temperature 0, JSON mode, max_tokens 16,000 and reasoning_effort.

## Post-processing (glm_enrich/e2.py `parse_e2`)

The vendored `parse_enrichment` runs first, unchanged. Then three steps run, and each one is recorded so that results can be scored with and without it:

| ID | Step | Recorded as |
|---|---|---|
| E2-PP1 | Paradigm labels ending in "learning"/"machine learning" are mapped to the vocabulary term when one matches | `learning_paradigm:repaired:<from>-><to>` in vocab_violations |
| E2-PP2 | A model_family from a canonical type's family hint is added if missing. Depth-dependent hints (neural network) and non-family hints (reinforcement) are never used. Skipped when the cap of 3 is full | `family_backfilled` column: `<family><-<type>` or `skipped_cap:…` |
| E2-PP3 | model_type items matching the E2-V7/V8 patterns are **flagged, never dropped** | `model_type:excluded_kind:<kind>:<value>` in vocab_violations |

## Rendered prompt hash

Frozen e2 prompt sha256: `c3196ff722d6a817c034afb7278ebf4266707ba64f98d9afbaf4454927f77c45`. (`6da96b83…` was an earlier, unreviewed draft.)


## Validation

All runs used GLM-5.3-Flash at reasoning effort `high`, temperature 0 and JSON mode, through Z.ai's pay-as-you-go API. The reports live in the test repository.

| Evidence | Cohort | Result | Report |
|---|---|---|---|
| e1 vs DeepSeek V4.1, one reviewer | enriched100 | GLM e1 family ~71% / type ~71% acceptable, DeepSeek ~87% / ~85% (this motivated e2) | `reports/tag_correctness_review_enriched100.md` |
| e1 vs e2, one reviewer (not blind) | heldout250 (never used to tune e2) | All three modelling fields right: 48% → 87%. Family 96%, type 90%, paradigm 98%, model_use 99% | `reports/e2_improvement/heldout250_correctness_review.md` |
| Pre-registered validation, Opus 5.5 judge on all 250 papers | wave2 (untouched) | Every primary unit (paradigm, family, type, model_use) passes the pre-registered bar: acceptable rate above 80% (exact one-sided binomial, Holm), point estimate at or above the 90% target | `validation/`, commit `0390dc2` |
| Same study, human rater on 150 papers + adjudicated consensus + AC1 gate | wave2 | **pending**: the pre-registered decision is not final until this is done | `validation/README.md` |

Cost per paper with e2 (list price, effort high): about $0.0006–0.001 depending on the prompt-cache share (the wave2 run was $0.24 for 250).

Known e2 weaknesses (the e3 candidates, none built): it over-prunes stated clustering, trained baselines and PLM embeddings in about 4% of papers; the family cap of 3 can squeeze out `natural language processing`; and a few generic phrases remain in model_type.
