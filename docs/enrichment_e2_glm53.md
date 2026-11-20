# Enrichment prompt e2 on GLM-5.3-Flash: what was built, the evidence, and the steps to production

Branch `enrichment-e2-glm53`, 2026-11-16. Nothing on this branch has touched moros or `dome-ml-observatory`.

## What e2 is

Prompt e2 is the enrichment prompt rebuilt for GLM-5.3-Flash on Z.ai's pay-as-you-go API. It tags the
same six fields as e1 (domain tiers 1–3, learning_paradigm, model_family, model_type) plus `model_use`:
how the paper obtains its models (trained in this study / fine-tuned or adapted / pretrained, applied
as-is / prompted (zero/few-shot)). Every change from e1 and the evidence for it is in
[`curation_criteria/e2/PROVENANCE.md`](../curation_criteria/e2/PROVENANCE.md).

It sits **beside** e1, as AGENTS.md rule 1 requires. e1's files are untouched and still render to
`41db952f…`. e2 renders to `c3196ff7…`, byte-identical to the frozen prompt that was validated.
`python3 prompts/render_prompts.py --check` verifies all three prompts.

## Evidence

| | Result |
|---|---|
| e1 vs e2 on heldout250 (one reviewer, not blind) | All three modelling fields right: 48% → 87% |
| Pre-registered validation on wave2 (untouched), Opus 5.5 judge on all 250 | Every primary unit passes: acceptable rate significantly above 80%, point estimate ≥ 90% |
| Same study: human rater on 150 papers, consensus, agreement gate (AC1 ≥ 0.70) | **Pending.** The pre-registered decision is final only after this |
| Cost | $0.96 per 1,000 records at list price (DeepSeek V4.1 billed $1.80) |
| Speed | ~325 records/min at concurrency 50; 0 parse errors and 0 truncations in 250 |

Full figures and method: the `dome-ml-observatory-glm53-validation` repository (gavinf97; checked out as `dome-ml-observatory-research/dome-ml-observatory-glm53-validation`), `validation/`
and `reports/e2_improvement/`.

## What changed on this branch

| Area | Files |
|---|---|
| Vocabularies (strict superset of e1's) | `curation_criteria/e2/` (+ PROVENANCE.md) |
| Engine | `src/dome_triage/llm_classify/enrichment_e2.py`, `zai_client.py`; `pipeline/steps.py::step_llm_classify_enrich_e2`; `cli.py` (`enrich --prompt e2`) |
| Rendered prompt and hashes | `prompts/enrichment_system_message.e2.txt`, `PROMPT_HASHES.json` (`enrichment_e2`), `render_prompts.py` |
| Key | `ZAI_API_KEY` in `.env.example` and `docker-compose.yml` |
| Schema v1.7.0 (authored only) | `schema.py` (`content_filters.model_use`), the template, `moros_write.py` (`migrate_v1_7_0` mode; `model_use` in the enrichment mode), `migrate_v1_7_0.py`, `check_alignment.py` (published modelling vocabularies are e2's) |
| Loader | `load_enrichment.py`: provider `zai` for GLM tiers; writes `model_use` from e2 events; refuses e2 events until the corpus is at 1.7.0 |
| Tests | `tests/test_enrichment_e2.py` (17), `test_migrate_v1_7_0.py` (5), `test_load_enrichment.py` (+4), `test_check_alignment.py` (+1). Engine gate 560 passed; host suites 503 / 88 / 14 passed |
| Agent instructions | `.claude/skills/enrich/SKILL.md` (e2 section) |

## Steps to production

The schema release follows [`schema/README.md`](../schema/README.md). Do these in order.

1. **Decide.** Finish the human half of the validation (in the test repository: `validation/README.md`,
   steps 4 and 6–8), or decide to go ahead on the judge-only result and say so in the release notes.
2. **Review and merge** this branch into `main`. Then rebuild the image, which copies `src/`:
   `docker compose build`, then `docker image prune -f`.
3. **Key.** Add `ZAI_API_KEY=` (the pay-as-you-go key, not a Coding Plan key) to `.env`.
4. **Publish v1.7.0** in `dome-ml-observatory` with its `schema-version` skill. That means:
   - the new field `content_filters.model_use` (array of strings, max 2, values from the e2 `model_use` field);
   - the vocabularies copied from `curation_criteria/domain_vocab.json` and `curation_criteria/e2/*.json`;
   - a CHANGELOG entry and the `FALLBACK_SCHEMA_VERSION` constants.

   Then refresh `schema/observatory_release/` here (the `schema-sync` skill).
5. **Check.** `python3 schema/check_alignment.py` must say `aligned`. Today it reports exactly the three
   expected drifts: the version, `content_filters.model_use`, and the two modelling vocabularies.
6. **Migrate.** `python3 moros_pipeline/scripts/migrate_v1_7_0.py`, a dry run first, then `--confirm`. It sets
   `model_use: []` everywhere. `--reverse` works until e2 content lands.
7. **Smoke run.** Export 25 positives (`export_journal_for_enrichment.py ... --limit 25`), then run
   `enrich --prompt e2 --limit 25`, `load_enrichment.py --limit 25 --confirm` and `verify_corpus.py`.
8. **Production.** Export the cohort: new positives, or the whole corpus with `--include-enriched` to
   replace e1 tags. Run `enrich --prompt e2`, then load with `load_enrichment.py`, which writes a rollback
   snapshot first. Record the billed figure from https://z.ai/manage-apikey/billing.
9. **After the load.** Follow the AGENTS.md checklist: restart `observatory-ws` and regenerate the facet
   stats. The UI and API may need to show `model_use`; that is a change in the sister repository.

## Decisions taken here (change them before merging if you disagree)

- **e2 runs on Z.ai tiers only, and e1 on DeepSeek only.** Each prompt was validated on one model.
- **Defaults are the validated configuration:** glm-5.3-flash, effort high, temperature 0, JSON mode,
  16,000 max tokens, concurrency 50.
- **`family_backfilled` is not stored on documents.** It stays in the event log as the parser's audit
  trail. The backfilled families themselves are in `model_family`.
- **e1-enriched documents keep `model_use: []`.** `llm_enrichment.prompt_version` tells e1 from e2.
- **The export cost gate stays at DeepSeek's $1.80 per 1,000.** That is stricter for e2. Replace it with
  a billed Z.ai figure once one exists.
- **Classification is unchanged.** It stays on DeepSeek, prompt v1.
