# Enrichment on the EMBL-EBI gateway (prompt e3)

Production enrichment from 2026-11-16 runs prompt **e3** ([`curation_criteria/e3/PROVENANCE.md`](../curation_criteria/e3/PROVENANCE.md)) on **GLM-5.3-Flash** served by EMBL-EBI's LLM gateway. The gateway is LiteLLM in front of vLLM, with an OpenAI-compatible API. Requests are priced against a per-key budget (see below), and the gateway is reachable only from the EBI network or VPN. `dome-triage llm-classify enrich` uses it by default.

## Configuration

The configuration lives in `.env`, which is gitignored; see `.env.example`. This repository is public, so the gateway's address never appears in a tracked file.

| Variable | What |
|---|---|
| `EBI_LLM_BASE_URL` | The gateway's OpenAI-compatible base, ending in `/v1`. It is **not** the admin UI link (`…/ui/api-keys/…`) |
| `EBI_LLM_API_KEY` | Your gateway key |
| `EBI_LLM_MODEL` | The served model id, from `GET $EBI_LLM_BASE_URL/models`. On 2026-11-16 that was `zai-org/GLM-5.3-Flash`, and it is stored on every document as `llm_enrichment.model_id` |
| `EBI_LLM_MAX_CONCURRENCY` | The concurrency measured by the ramp below. `enrich` uses it when `--concurrency` is not given |

Documents record `llm_enrichment.provider: "embl-ebi"`, `model_tier: "ebi-glm-5.3-flash"` and the served `model_id`, verbatim.

## What the endpoint does (probe, 2026-11-16)

| | Result |
|---|---|
| `/models` | One model: `zai-org/GLM-5.3-Flash` |
| `/key/info`, `/spend/logs` | Forbidden for this key (it may call the LLM routes only), so its limits and spend are visible only in the gateway UI |
| `temperature: 0`, `response_format: json_object` | Accepted |
| Z.ai's `thinking: {type}` and `reasoning_effort` | Accepted and **ignored**. Low, high and max give the same reasoning, and so does disabling thinking |
| vLLM's `chat_template_kwargs.enable_thinking` | **The switch that works.** `false` turns thinking off. The client sends `true` |
| Usage | `prompt_tokens`, `completion_tokens` and `total_tokens` only. There are no cached-token or reasoning-token details, so those event columns read 0 |

There is no effort dial, so the model thinks at its own default depth. That depth matches Z.ai's `max`, not the `high` that e2 was validated at: about 2,500 output tokens per record against 441 at `high` and 2,981 at `max`. When `max` was measured on Z.ai (100 papers), tag quality was the same. The deeper thinking costs time and budget, not quality.

## Quality check: e3 on EBI against e2 on Z.ai (wave2, 250 papers)

- **Run:** `moros_pipeline/output/enrichment_e3/qc_wave2/`, concurrency 16, 12 min 18 s. 0 parse errors, 0 truncations, and 3 records with a vocabulary flag.
- **Inputs:** rebuilt from the Zenodo dump by `scripts/inputs_from_zenodo.py --ids-from`.
- **Comparison:** `scripts/compare_enrichment_runs.py`. Exact agreement means the same set of values per field.

| Field | EBI e3 vs Z.ai e2, 230 papers using no new e3 term | Z.ai e2 vs itself (effort high vs max), enriched100, 99 papers |
|---|---:|---:|
| domain_tier1 | 78.3% | 79.8% |
| domain_tier2 | 49.6% | 44.4% |
| domain_tier3 | 40.4% | 47.5% |
| model_use | 96.1% | 92.9% |
| learning_paradigm | 92.6% | 96.0% |
| model_family | 86.5% | 85.9% |
| model_type | 71.7% | 76.8% |
| all three modelling fields | 62.2% | 66.7% |

**Reading.** EBI's e3 agrees with Z.ai's e2 about as closely as Z.ai's e2 agrees with itself across effort settings.
- Every field is within about 5 points of that baseline, except domain_tier3, which was the noisiest field in every earlier comparison.
- Part of the model_type gap is e3's new aliases on existing terms. ResNet-50 now becomes ResNet and BiLSTM becomes long short-term memory, which is intended.
- The baseline is a different set of papers, so this is indicative and not paired.
- The decision was to proceed. The plan's stop condition, results that "clearly fall short", was not met.

On the 20 papers (8%) where e3 used one of its new terms, model_type agreement with e2 falls to 10%. That is the purpose of e3: e2 had no seed term for AlphaFold, protein language models and the rest.

## Concurrency and speed

A call takes as long as its output, about 45 s at about 2,500 tokens. Throughput is therefore roughly concurrency / 45 s, as long as the gateway keeps up.

| Concurrency | Records | records/min | Errors |
|---:|---:|---:|---|
| 16 | 250 (wave2) | ~21 | none |
| 64 | 211 (core_bioinformatics c01) | ~19 | 92 requests rejected by the gateway (429/5xx); the client's HTTP retries resent them, so 0 records were lost, but each rejection shows as "failed" in the gateway's usage page |

**The gateway caps throughput at about 20 records/min for this key and model, whatever the concurrency.** Going above 16 only adds rejected requests. `EBI_LLM_MAX_CONCURRENCY=16`.

**Budget.** The gateway prices each request: about $0.0022 per record on 2026-11-16. A key has a budget limit, which is shown only in the gateway UI under Usage, because a key cannot read `/key/info` or `/spend/logs`. When the budget is exhausted, calls are refused; the run then stops and resumes once the limit is raised. Plan a cohort as about **$2.2 and 50 minutes per 1,000 records**.

## Running a cohort

```bash
scripts/run_enrichment.sh core_bioinformatics      # tmux session enrich-e3: run / checkpoint / status
tmux attach -t enrich-e3                           # watch; Ctrl-b 0/1/2 switches, Ctrl-b d detaches
docker compose run --rm tools python scripts/enrich_status.py --cohort core_bioinformatics   # or just the status
```

- **Inputs:** the private runs repository's `inputs/<cohort>/`, built by `scripts/inputs_from_zenodo.py` and checked against its `SHA256SUMS` before a run starts.
- **Events:** written to `moros_pipeline/output/enrichment_e3/<cohort>/`, and pushed to the runs repository every 15 minutes by `scripts/checkpoint_events.py`.
- **Loading:** happens on the laptop with moros access (`moros-write` section B), using `load_enrichment.py`, once moros is at schema 1.8.0. Ingestion cleans closed-vocabulary values deterministically (`moros_pipeline/scripts/enrichment_clean.py`, rule set `c1`) and keeps any enrichment it supersedes in `enrichment_history`. The vocabulary-gap list, [`enrichment_vocab_gaps.md`](enrichment_vocab_gaps.md), is generated by the same script.

## Production runs

| Cohort | Records | Events ok | Truncated (not loaded) | Cleaned at ingestion | Finished |
|---|---:|---:|---:|---:|---|
| `core_bioinformatics` | 13,065 | 13,042 | 23 | see the load report | 2026-11-17 |
| `batch2_high_impact` | 87,159 | 87,031 | 128 | see the load report | 2026-11-20 |

Inputs were built from Zenodo dump 2026-09-25 (record 22967974), positives with no enrichment, so no
record already enriched with e1 was re-run. The 151 truncated records hit the 16,000-token cap; re-run
them on EBI with a higher `ENRICHMENT_MAX_TOKENS` (a versioned change).
