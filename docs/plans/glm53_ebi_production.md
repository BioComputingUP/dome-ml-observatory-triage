# Plan: production enrichment on EMBL-EBI GLM-5.3, building on branch `enrichment-e2-glm53`

> **Status 2026-11-20: done, kept as a record.**
> - The branch is merged into `main` and deleted, and the `-e2` worktree is removed.
> - The EBI laptop enriched 100,224 records with prompt e3: `core_bioinformatics` 13,065 and
>   `batch2_high_impact` 87,159. 151 were truncated.
> - 100,073 were loaded into moros at schema 1.8.0. 1.8.0 adds `enrichment_history`, and every load
>   now runs the ingestion cleaning `enrichment_clean.py` (`c1`).
> - The repositories were renamed: `dome-ml-observatory-enrichment-runs`, `dome-ml-observatory-research`
>   (formerly `dome-triage`), and `dome-ml-observatory-glm53-validation` (the test bed, nested in research).
>   The names below are the ones in use when this plan was written.
> - The deviations from Phase 4:
>   - the loader did not refuse already-enriched records; it now keeps them in `enrichment_history`;
>   - the vocabulary gaps are generated (`docs/enrichment_vocab_gaps.md`) rather than kept by hand.
> - Still open:
>   - the 151 truncated records (re-run on EBI with a higher `ENRICHMENT_MAX_TOKENS`);
>   - the remaining unenriched positives;
>   - the human-rater half of the validation (Phase 5).

Written 2026-11-16 so that a fresh session on either laptop can carry it out. Repositories:
- **triage**: this repository; the work is in worktree `../dome-ml-observatory-triage-e2`, on branch `enrichment-e2-glm53`.
- **test bed**: `glm5.3_test_dome-ml-observatory-triage`, which is public.
- **sibling**: `dome-ml-observatory`.
- **runs**: a new, private repository, `gavinf97/dome-observatory-enrichment-runs`.

## Context

- **The prompt is decided.** Prompt e2 on GLM-5.3 tags clearly better than e1:
  - on heldout250, all three modelling fields are right in 87% of papers, against 48% for e1;
  - the Opus judge passes every primary unit on all 250 wave2 papers;
  - the human rating continues: 18 of 150 done.
- **The compute changes.** Production moves to **EMBL-EBI's local model service, with the EBI token only**. Gavin's Z.ai and DeepSeek keys are not used.
- **One addition first.** A minor extension adds the major life-science **model types** that are missing. It goes through vocabulary versioning first. Because seed model types are rendered into the prompt, it then becomes prompt **e3** (e2 + those terms, nothing else). **Production uses only e3.**
- **Moros stays on this laptop.** The EBI laptop cannot write to moros. All moros reads and writes happen on this laptop, and data moves through the private runs repository.

## What other sessions already did (as of 2026-11-16 ~22:00; check again before acting)

| Where | Done | State |
|---|---|---|
| triage worktree `../dome-ml-observatory-triage-e2`, branch `enrichment-e2-glm53` (session `dome-ml-observatory-triage-0f`) | The e2 port beside e1, covering: `curation_criteria/e2/`, `enrichment_e2.py`, `zai_client.py`, `step_llm_classify_enrich_e2`, `enrich --prompt e2`; the rendered prompt and `PROMPT_HASHES.json` (`enrichment_e2`, renders `c3196ff7…`, e1 still `41db952f…`); schema 1.7.0 authored (`content_filters.model_use`, the `migrate_v1_7_0` mode and script); the loader writing `model_use`, and refusing e2 events until 1.7.0; `check_alignment` against the e2 vocabularies; tests (engine 560, host 503/88/14 reported passing); `docs/enrichment_e2_glm53.md`; the enrich skill | **All uncommitted. Nothing touched moros or the sibling.** |
| test bed (session `glm5-3-test-…-da`) | Opus judge finished all 250 and committed (`0390dc2`) | It still holds **uncommitted** paid outputs: the e1 and e2 events, manifests, `spend_log.csv`, `ratings/gf.csv`, the e2 code, `experimental/`, `reports/` |

The branch's decisions are kept unless Phases 1–2 below change them:
- `model_use: []` on e1 documents;
- `family_backfilled` is not stored on documents;
- classification stays on DeepSeek v1;
- the validated defaults stay: effort high, temperature 0, JSON, 16,000 tokens.

## Working alongside other sessions (do this before every step that writes)

1. Run `ListAgents`, and `git -C <repo> status` / `git worktree list` on every repository involved.
2. Do not edit or commit in a worktree or repository while its session is **active**. If it is idle, tell it with `SendMessage` what you are about to commit, then do it.
3. Commit only on the existing branch `enrichment-e2-glm53`. Never start a parallel branch for the same work, and never touch `main` until the merge step.
4. Do not change anything that another session made unless this plan names it. Their work is the base.

## Decisions (Gavin, 2026-11-16)

1. **Already-enriched records keep e1.** e3 runs only on positives that are not yet enriched (`llm_enrichment.provider` null). Do **not** use `--include-enriched`.
2. **Checkpoints go to the private runs repository** every 15 minutes during a run. This laptop loads them into moros at the end.
3. **Scope is enrichment only.**
4. **Human rating does not block.** The release notes state the evidence and that the human half is pending.
5. **The vocabulary is frozen for the whole production run.** Gaps found during the run go to a list for later.

## Phase 0: home laptop. DONE 2026-11-16

1. The other session's e2 work is committed and pushed: branch `enrichment-e2-glm53`, commit `97456a7`, which also adds this plan. Checks that passed: `render_prompts --check`, the three host suites (503/88/14) and the absolute-path check. **The Docker engine gate has not been run on this commit.** Run it after the first `docker compose build` on the EBI laptop.
2. The test bed is committed and pushed: branch `tag-validation`, commit `dc02efe`. That includes all the events, manifests, the spend log, the e2 code, reports, the judge logs and the partial human ratings.
3. The private runs repository exists: https://github.com/gavinf97/dome-observatory-enrichment-runs (private, checked with `isPrivate: true`). It holds `inputs/`, `events/`, `manifests/`, `SHA256SUMS` and a README.
4. **Inputs come from Zenodo, not moros.** The home laptop needs no moros export before the switch. The EBI laptop builds every input from the public corpus dump (Phase E, steps 5–7). The home laptop catches up on anything newer at load time (Phase 4, step 0).

## Phase E: first session on the EBI laptop (start here)

Connect to the EBI VPN, then:

1. **Read this plan and check for other sessions.** Run `ListAgents`, and `git status` in every checkout. Follow the protocol above.
2. **Clone three repositories into one parent folder**, side by side; their folder names matter:
   - `git clone git@github.com:BioComputingUP/dome-ml-observatory-triage.git && git -C dome-ml-observatory-triage checkout enrichment-e2-glm53`
   - `git clone git@github.com:gavinf97/glm5.3_test_dome-ml-observatory-triage.git && git -C glm5.3_test_dome-ml-observatory-triage checkout tag-validation`
   - `git clone git@github.com:gavinf97/dome-observatory-enrichment-runs.git` (private, so `gh auth login` as gavinf97 first)
3. **`.env` in triage and the test bed.** Set `EBI_LLM_BASE_URL`, `EBI_LLM_API_KEY`, `EBI_LLM_MODEL`, `EBI_LLM_MAX_CONCURRENCY` and `ENRICHMENT_RUNS_REPO`. This laptop gets **no** moros `.env`, Z.ai key or DeepSeek key.
4. **Docker.** Run `docker compose build` (it needs internet and downloads about 7 GB of model weights and NLTK data). Then run the AGENTS rule-9 gate and the three host suites. **If Docker is not allowed on this laptop, stop and ask Gavin.**
5. **Download the corpus dump.** Get the latest version of Zenodo concept `10.5281/zenodo.22259905` from the API: `https://zenodo.org/api/records?q=conceptrecid:22259905&all_versions=true&sort=mostrecent`. On 2026-11-16 that was record 22967974, dated 2026-09-25, schema 1.6.0, open access, with `dome-observatory-corpus-2026-09-25.jsonl.gz` at 879 MB. Save it outside every repository, for example `../zenodo_corpus/`, and check its checksum against both Zenodo's file checksum and `archive-metadata.json`.
6. **Build the production inputs.** Write `scripts/inputs_from_zenodo.py` in triage, with tests. It reads the dump and keeps documents with `llm_classification.classification == "positive"`, `llm_enrichment.provider` null, and a non-empty title and abstract. It writes the same columns as `export_journal_for_enrichment.py` (`record_id, title, abstract, journal, year`), in `_id` order, split into 5,000-record chunks: `inputs/enrich_input_e3_zenodo-<version>_cNN.csv`. It also writes `inputs/SOURCE.md` with the Zenodo record id, version, file sha256, the counts and the selection. Then commit and push to the runs repository and update `SHA256SUMS`.
7. **Build the test-bed inputs.** Rebuild `data/input_{wave2_250,enriched_100}.csv` from the same dump, using the record ids in the test bed's committed `cohorts/manifest.csv`. Check that every id was found, and that they are byte-identical to the originals if those are ever available. `data/` stays gitignored.
8. **Gather the term evidence** for Phase 1. For each candidate model type, count the positives in the dump whose title or abstract mentions it (case-insensitive, word-bounded, canonical names and aliases). Write `docs/e3_model_type_evidence.md` in triage with the counts, the share of positives and a few record ids, never abstracts. Commit and push.
9. Then continue with Phase 1.

## Phase 1: minor model-type extension, then prompt e3

1. **Vocabulary version.**
   - Create `curation_criteria/e3/` as a copy of `curation_criteria/e2/`. Change **only** `model_type_seed_vocab.json`, by adding seed entries and aliases. `modelling_branch_vocab.json` and the domain vocabulary are unchanged.
   - Every new entry carries `canonical`, `aliases`, `family_hint` (existing family terms only) and `ontology_mappings`. The mappings use the method in [docs/vocabulary_ontology_mappings.md](docs/vocabulary_ontology_mappings.md) (OLS4, Zooma, NLM).
   - `PROVENANCE.md` records the e2 → e3 diff and the evidence.
2. **Inclusion rule.** A term goes in only if all three hold:
   - at least 100 positives mention it (Phase 0, step 6);
   - it is not covered by an existing canonical term or alias;
   - it is a genuine model or architecture, not a tool, score or generic phrase.

   **Show Gavin the final list before freezing.**
3. **Candidate terms.** These are drawn from the 447 off-seed model types GLM produced across 850 e2 records, plus Gavin's list.
   - **Biological language and foundation models:**
     - `protein language model` (ESM, ESM-2, ProtT5, ProtBERT, ProtTrans, Ankh, ProGen, SaProt)
     - `genomic language model` (DNA/RNA language model, DNABERT(-2), Nucleotide Transformer, HyenaDNA, Evo, Caduceus, RNA-FM)
     - `single-cell foundation model` (scGPT, Geneformer, scBERT, scFoundation, UCE)
     - `chemical language model` (ChemBERTa, MolFormer, Chemformer)
     - `pathology foundation model` (UNI, CONCH, Virchow, Prov-GigaPath)
     - `graph embedding` (node2vec, DeepWalk, knowledge graph embedding)

     All of these take the family `deep learning` + `representation learning`, except `graph embedding`, which takes `representation learning` only.
   - **Structure and imaging:**
     - `AlphaFold` (AlphaFold2/3, AF2/AF3, AlphaFold-Multimer, ColabFold): 10 of 850 records, the biggest gap
     - `RoseTTAFold`
     - `Segment Anything Model` (SAM, SAM 2, MedSAM)
     - imaging backbones, following the existing ResNet precedent: `DenseNet`, `VGG`, `EfficientNet`, `MobileNet`, `Inception`/GoogLeNet, `Faster R-CNN`, `Mask R-CNN`
   - **Clinical and other:**
     - `Cox proportional hazards model`: 10 of 850
     - `random survival forest`
     - `multiple instance learning` (ABMIL, CLAM, TransMIL)
     - `physics-informed neural network` (PINN), filling the type gap under e2's own family
     - `Kolmogorov-Arnold network` (KAN)
     - `ARIMA`
   - **Aliases on existing terms:**
     - ResNet-18/50/101…
     - nnU-Net, UNet++ → U-Net
     - GAT, GIN, MPNN, Chemprop → graph neural network
     - Swin Transformer → vision transformer
     - BiLSTM → LSTM
     - multiple linear regression → linear regression
   - **Not added:**
     - new families or paradigms: "foundation model" would overlap large language model, transfer learning and representation learning, and press on the cap of 3;
     - named LLMs, which stay verbatim under the e2 rule;
     - optimisers;
     - CIBERSORT, Boruta and other tools;
     - generic phrases.
4. **Prompt e3.** Build `enrichment_e3.py` as e2's module pointed at `curation_criteria/e3/` with `prompt_version "e3"`, with no other change.
   - Render `prompts/enrichment_system_message.e3.txt` and register it as `enrichment_e3` in `PROMPT_HASHES.json`.
   - **Check:** the diff of the e3 render against the e2 render contains only added seed lines.
   - Add tests: the hash pin, plus the parser normalising the new aliases to their canonical terms.
5. **Schema 1.7.0 publishes the e3 vocabularies.** 1.7.0 has not been released, so this needs no extra release. Point `check_alignment.py` at `curation_criteria/e3/`.

## Phase 2: EBI provider (branch code), then a light endpoint check (EBI laptop)

1. **Provider code.** Generalise the branch's `zai_client.py` into one OpenAI-compatible GLM client with a `provider` setting, keeping the Z.ai behaviour exactly as it is.
   - `ebi` reads `EBI_LLM_BASE_URL`, `EBI_LLM_API_KEY`, `EBI_LLM_MODEL` and `EBI_LLM_MAX_CONCURRENCY`, and builds the thinking body from the probe.
   - `ebi` uses tier label `ebi-glm-5.3`.
   - Add these to `.env.example` and `docker-compose.yml`, by name only.
2. **CLI.** Add `enrich --prompt e3 --provider ebi`.
3. **Events.** e3 events add a `model_id` column holding the served id.
4. **Loader.** Map `ebi-glm-5.3` to provider `embl-ebi` and take `model_id` from the event.
5. **Tests.** Add tests, then commit and push.
6. **EBI laptop setup** was done in Phase E: the clones, `.env`, Docker and the inputs built from Zenodo.
7. **Probe the endpoint** from the host and from the container: `/models`, then one tiny call. Record the following in a gitignored `EBI_ENDPOINT.local.md`:
   - the served id, and whether it is full or Flash and quantised;
   - the context window and maximum output tokens;
   - the thinking and effort switch;
   - the usage fields;
   - the concurrency limit.
8. **Light endpoint check.** Run e3 on EBI over `wave2_250` and `enriched100`, using the inputs rebuilt in Phase E step 7. Then compare with the Z.ai e2 events in the test bed, field by field (`compare_to_deepseek.py`-style agreement), and run the Opus judge (subscription CLI) on wave2.
   - The new seed terms are the only prompt difference, so on records that do not use them the result should match the Z.ai e2 judge rates within about 5 points.
   - If it clearly falls short, stop and show Gavin before production.

## Phase 3: production run (EBI laptop)

1. **Run each chunk in tmux** (rule 5: never detached):

   `docker compose run --rm pipeline dome-triage llm-classify enrich --prompt e3 --provider ebi --input <runs>/inputs/<chunk>.csv --events-out moros_pipeline/output/enrichment_e3/<batch>_events.csv --concurrency $EBI_LLM_MAX_CONCURRENCY`

   If a chunk is interrupted, re-run the same command; it resumes by row. Keep the laptop on mains power with sleep off.
2. **Checkpoints.** `scripts/checkpoint_events.py` runs every 15 minutes, in its own tmux window.
   - It copies events, manifests and provenance rows into the runs repository, writing to a temp file and then renaming.
   - It updates `SHA256SUMS`, commits and pushes.
   - It skips files that have not changed.
   - It refuses to push unless `gh repo view --json isPrivate` is true.
3. **After each chunk:**
   - check parse errors and truncations, and stop and ask if either is above 2%;
   - make a final checkpoint push;
   - note `vocab_violations` in `docs/e3_vocab_gaps.md`, as input for a later version.

## Phase 4: THIS laptop (moros and sibling)

0. **Catch up on what is newer than the dump.** On the VPN, run `python3 scripts/zenodo_archive.py --check` (read-only). Then count the positives in moros with null `provider` whose `_id` is in none of the input chunks; these are positives loaded after the Zenodo dump was taken.
   - If there are any, export them with `export_journal_for_enrichment.py --max-usd 0`, as chunks after the last one.
   - Push them to `inputs/`, run them on the EBI laptop (Phase 3), then load them here with the rest.
   - Otherwise, leave them for the next refresh round.
1. Pull the branch and the runs repository, and verify `SHA256SUMS`.
2. **Sibling.** Publish v1.7.0 with its `schema-version` skill: `model_use` and the e3 vocabularies. The CHANGELOG states the mixed e1/e3 corpus and the validation status. Then refresh `schema/observatory_release/` here and run `check_alignment.py`, which must report "aligned".
3. **Migrate.** Run `migrate_v1_7_0.py` as a dry run, then `--limit`, then `--confirm`. Then run `verify_corpus.py` and `check_alignment.py --live`.
4. **Load each chunk** with `load_enrichment.py`: dry run, then a 50-record trial, then `--confirm` (a rollback snapshot is written each time), then `verify_corpus.py`. Records enriched since the export are refused, not overwritten.
5. **Merge and finish.**
   - Merge `enrichment-e2-glm53` into `main` and push.
   - Run the AGENTS post-load checklist: restart `observatory-ws`, then the facet stats and the `/api/stats` reconciliation.
   - Run `processing-log`, make the Zenodo archive and update memory.

## Phase 5: human validation (in parallel, either laptop)

Gavin finishes `rate_app.py --rater gf` (150 papers), committing and pushing after each session. Then:
1. run `analyse.py`;
2. adjudicate;
3. write `report.md`;
4. link the report from the e2/e3 docs and the sibling CHANGELOG.

## Verification

- **Prompt hashes:** `render_prompts.py --check` passes for e1 (`41db952f…`), e2 (`c3196ff7…`) and e3 (its new hash). The e3-versus-e2 render diff contains only added seed lines.
- **Test suites:** the engine gate and the three host suites pass at every commit, and `check_no_absolute_paths.py` is clean.
- **After the 50-record trial**, a moros document shows:
  - `prompt_version: "e3"`, `provider: "embl-ebi"` and the served `model_id`;
  - `model_use` filled in.

  An e1 document is unchanged apart from `model_use: []`.
- **Counts:** the number of e3 documents in moros equals the ok events minus the refusals listed in the reports. During a run, the runs repository's newest commit is at most 15 minutes old.
- **Secrets:** no public repository contains the EBI host, a token or abstracts.
