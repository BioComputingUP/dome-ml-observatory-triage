# Vocabulary candidates (not in use)

**Status: candidates only.** Nothing in this folder is part of a prompt, a published vocabulary or
the schema. Prompt **e3** stays the production enrichment prompt; on 2026-11-20 Gavin decided not
to cut an e4 now, but to keep these findings for a future prompt version. Nothing reads this
folder: `render_prompts.py`, `check_alignment.py`, the parsers and the loaders open named files in
`curation_criteria/` and `curation_criteria/e3/` only.

| File | What it is |
|---|---|
| [`model_type_candidates.json`](model_type_candidates.json) | Every candidate in the e3 seed's own entry shape (`canonical`, `aliases`, `family_hint`, `ontology_mappings`), with a `candidate` block: action, measured counts, observed spellings, three example papers and a note |

Why they exist, and the stray domain tags that are *not* fixed by any vocabulary change, are in
[`docs/e3_vocab_gaps.md`](../../docs/e3_vocab_gaps.md).

## Where the numbers come from

- **Runs:** the two e3 cohorts, `core_bioinformatics` (13,065 records) and `batch2_high_impact`
  (87,159), events 2026-11-16 to 2026-11-20; 100,073 parsed, prompt hash `3b9d119d…`, GLM-5.3-Flash
  on the EMBL-EBI gateway. Backed up in the private runs repository (`events/<cohort>/`, commit `c50dde2`).
- **`tag_count` / `records`:** `model_type` tags whose lower-cased value matches the candidate
  (every spelling seen is under `observed_spellings`). Of 163,902 `model_type` tags, 56,890 (34.7%)
  fall outside the e3 seed; there are 30,512 distinct values. `model_type` is open by design, so
  these are never flagged: they are only un-normalised.
- **`zenodo_positives_mentioning`:** positives in `../zenodo_corpus/dome-observatory-corpus-2026-09-25.jsonl.gz`
  (367,348) whose title or abstract mentions the candidate, counted once per paper — the basis e3's
  inclusion rule used (`scripts/e3_term_evidence.py`, rule ≥ 100). A different population from the
  tag counts; both are given.
- Measured read-only on 2026-11-20 after the run finished.

## The candidates

**New seed terms** (each has its own entry in the JSON):

| Canonical | Tags | Zenodo | Note |
|---|---:|---:|---|
| spiking neural network | 490 | 727 | no bare `SNN` alias (shared-nearest-neighbour graph in single-cell) |
| extreme learning machine | 383 | 1,202 | aliases ELM, KELM, OS-ELM |
| reservoir computing | 256 | 613 | echo state network, liquid state machine |
| temporal convolutional network | 221 | 480 | |
| AlexNet | 211 | 889 | own canonical, as ResNet and Inception are |
| adaptive neuro-fuzzy inference system | 203 | 777 | ANFIS |
| vision-language model | 200 | 386 | CLIP folded in as an alias |
| radial basis function network | 199 | 497 | four spellings today |
| Xception | 189 | 761 | XceptionTime is a different model |
| capsule network | 159 | 310 | CapsNet |
| RoBERTa | 153 | 421 | XLM-RoBERTa, DistilRoBERTa |
| DeepLab | 151 | 245 | **not** DeepLabCut (pose estimation; 31 tags excluded) |
| ConvNeXt | 122 | 315 | own canonical (was "alias of CNN" in the first analysis) |
| deep belief network | 121 | 336 | no bare `DBN` alias (dynamic Bayesian network) |
| Bayesian neural network | 82 | 191 | below 100 tags |
| TabNet | 77 | 273 | below 100 tags |
| consensus clustering | 403 | 1,128 | a specific method; kept apart from generic clustering |
| clustering | 1,087 | 3,941* | **decide first:** generic name, arguably a `family_name` exclusion instead |
| particle swarm optimization | 465 | 2,177 | optimiser: seeding only merges spellings; the `optimiser_check` flag stays |
| Bayesian optimization | 198 | 1,402 | optimiser, as above |
| grey wolf optimizer | 177 | 577 | optimiser, as above |
| differential evolution | 109 | 347 | optimiser, as above |

\* "cluster analysis" or "unsupervised clustering" only; bare "clustering" is in far more abstracts.

**Aliases for existing e3 entries:** YOLO += YOLOv3/v4/v5/v7/v8/v11, YOLOX (352 tags); BERT +=
BioBERT, PubMedBERT, ClinicalBERT, SciBERT, BioClinicalBERT, BlueBERT (134); generative adversarial
network += CycleGAN (85).

**New excluded kinds** (flag, never drop — the e2 rule): nomogram → `statistics` (557 tags; a
nomogram presents a fitted regression model); CIBERSORT → `pipeline` (356; deconvolution tool);
Boruta → `task` (218) and SVM-RFE / recursive feature elimination → `task` (142), both feature
selection; response surface methodology → `statistics` (102).

**Domain terms** (rare; listed, not recommended on their own): Rehabilitation medicine (4),
Ophthalmology (1), Geology (1, out of scope). EDAM ids to confirm against EDAM 1.25 before any use.

## How to adopt any of this

It is a new prompt version, never an edit to e3 (AGENTS.md rule 1). Follow how e3 was built on e2
(`docs/plans/glm53_ebi_production.md` Phase 1, `curation_criteria/e3/PROVENANCE.md`):

1. Agree the final list (the `clustering` decision first). Re-measure on the current corpus.
2. `curation_criteria/eN/`: copy e3's two modelling files, append the chosen entries (drop the
   `candidate` block, add an `"eN": {...}` provenance block per entry and at the top level, as e3
   did), fill `ontology_mappings` (`scripts/e3_ontology_mappings.py`), write `PROVENANCE.md`.
3. `src/dome_triage/llm_classify/enrichment_eN.py` (clone of `enrichment_e3.py`), the CLI branch and
   default in `cli.py`, `prompts/render_prompts.py` + `PROMPT_HASHES.json`, tests (e3's
   "differs only in seed lines" test is the model).
4. New excluded kinds: `EXCLUDED_KINDS` is a module global used by `parse_e2`, which e2 and e3 share.
   Pass eN's patterns through the lookup (`lookup.get("_excluded_re", _EXCLUDED_RE)`) so e2/e3 output
   stays byte-identical, and append new kinds last (first match wins). They are not in the rendered
   prompt, so the prompt hash does not cover them: record them in `PROVENANCE.md`.
5. The run tooling is wired to e3: `scripts/run_enrichment.sh` (no `--prompt`, `enrichment_e3/`,
   session `enrich-e3`), `scripts/checkpoint_events.py` (`STEP = "llm-classify.enrich-e3"` — eN runs
   would silently not be checkpointed), `enrich_status.py`, `run_watchdog.py`,
   `compare_enrichment_runs.py` (`new_terms` reads the `"e3"` block). An e3 events file refuses
   eN events ("already holds events from another prompt version"): eN needs its own folder.
6. Validate: the 250-record `qc_wave2` set on EBI, compared with e3's `qc_wave2` events
   (`scripts/compare_enrichment_runs.py`), before any production run.
7. Vocabulary release: `schema/README.md` (author here first, then the sibling's `schema-version`
   skill; `check_alignment.py` must point at `eN/`), migrate, then load.
