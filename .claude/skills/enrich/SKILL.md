---
name: enrich
description: >
  Enrich a cohort of positive records with the controlled vocabularies (EDAM domain tiers,
  learning paradigm, model family, model type, model use) using the LATEST enrichment prompt (e3)
  on GLM-5.3-Flash served by the EMBL-EBI gateway (per-key budget, ~$2.2/1,000; needs the EBI network or VPN): build the
  inputs, run the cohort in a watched tmux session with live ETA and 15-minute checkpoints to the
  private runs repository, then hand the events to moros-write. Trigger on "enrich journal X",
  "enrich the new positives", "run the enrichment", "enrich the bioinformatics journals". Older
  prompts (e1 on DeepSeek, e2 on Z.ai) are paid and run only to reproduce earlier work.
---

# enrich

Enrichment is additive by construction: the prompt never asks the positive/negative question, and the
writer's allowlist makes that structural. **Always use the latest prompt**, which is e3 since 2026-11-16
(`curation_criteria/e3/PROVENANCE.md`). `dome-triage llm-classify enrich` defaults to it, on the
EMBL-EBI gateway. Read [`docs/enrichment_ebi.md`](../../../docs/enrichment_ebi.md) before the first
run on a laptop. It covers the configuration, what the endpoint does, the quality check against
Z.ai and the measured concurrency.

Everything runs in containers (AGENTS.md rules 8–9). The laptop that runs the model needs no moros
access. Loading happens wherever moros's `.env` is.

## 1. Choose the cohort and build its inputs

Named cohorts live in `configs/enrichment_cohorts.yaml` (journal names matched **exactly**, as stored).
Build the input chunks from the public Zenodo corpus dump, saved beside the repository in `../zenodo_corpus/`.
Take the latest version of concept `10.5281/zenodo.22259905`, and check its sha256 against the dump's
`archive-metadata.json`:

```bash
export HOST_UID=$(id -u) HOST_GID=$(id -g)
docker compose run --rm tools python scripts/inputs_from_zenodo.py \
    --dump ../zenodo_corpus/<dump>.jsonl.gz --cohort <cohort> \
    --out-dir ../dome-observatory-enrichment-runs/inputs --zenodo-record <record id>
```

The selection is positives with no enrichment, a title and an abstract, in 5,000-record chunks. Push
the inputs to the private runs repository, with the privacy check, using `scripts/runs_repo.py`'s
`commit_and_push`.

On the loading laptop, an alternative source is `moros_pipeline/scripts/export_journal_for_enrichment.py
--journal … --max-usd 0`, which writes the same columns.

## 2. Say what it will take, then ask

EBI requests count against the key's budget (about $2.2 per 1,000 records, shown in the gateway UI under Usage; the key cannot read it via the API). State that, and the time:
- a call takes about 45 s, because the gateway thinks at its default depth, about 2,500 output tokens;
- the gateway caps throughput at about 20 records/min, whatever the concurrency; above 16 it only
  rejects requests (`EBI_LLM_MAX_CONCURRENCY=16`);
- so `minutes ≈ N / 20`.

State N, the chunks, the ETA and the budget, and ask: **"Enrich N records on EBI, about H hours and $X of the key's budget, start now?"**

## 3. Run, in a session the user can watch

```bash
docker ps                                          # no pipeline container may already be running
scripts/run_enrichment.sh <cohort>                 # [--concurrency N] [--limit N] [--chunks c01,c02]
```

This opens tmux session `enrich-e3` with three windows: `run` (the chunks in order, each a foreground
`docker compose run`), `checkpoint` (a push to the private runs repository every 15 minutes) and `status`
(live progress, speed, ETA, quality flags and stalls). Tell the user how to watch it:

- `tmux attach -t enrich-e3`, then Ctrl-b 0/1/2 to switch windows and Ctrl-b d to detach (the run continues);
- or, without tmux, `docker compose run --rm tools python scripts/enrich_status.py --cohort <cohort>`.

Watch it yourself with `enrich_status.py --once` in a Monitor or until-loop, and report progress and the ETA.

**Rules:**
- **Never detach `docker compose run`** (AGENTS.md rule 5).
- Re-running the same command resumes, because the events file is the state.
- The script retries a crashed chunk only after `docker ps` shows no pipeline container.
- It stops on its own if a finished chunk has **parse errors or truncations above 2%**. Report that and
  ask the user before going on.

## 4. After the run

From `enrich_status.py --once`, report:
- ok / parse_error and truncated counts;
- the vocab-flag rate (about 1% on the QC run);
- output tokens per record and records/min.

Then add any vocabulary gaps (`vocab_violations`) to `docs/e3_vocab_gaps.md`, for a later prompt version.
Make a final checkpoint push.

Then, on the laptop with moros, ask **"Merge these into moros?"**. On a yes:
- pull the runs repository;
- `sha256sum -c SHA256SUMS`;
- hand each events file to `moros-write` section B (`load_enrichment.py`: dry run, then `--limit 50`, then `--confirm`).

e3 events write `content_filters.model_use`, so the loader refuses them until moros is at schema 1.7.0.

## Older prompts (reproduction only)

| Prompt | How | Notes |
|---|---|---|
| e2 on Z.ai | `enrich --prompt e2` (needs `ZAI_API_KEY`) | [`docs/enrichment_e2_glm53.md`](../../../docs/enrichment_e2_glm53.md). Paid, about $0.96 per 1,000 at list price; concurrency 50, effort high |
| e3 on Z.ai | `enrich --provider zai` | The same prompt at Z.ai's effort high, paid |
| e1 on DeepSeek | `enrich --prompt e1 --tier flash --concurrency 800` | Paid, $1.80 per 1,000 billed. Run the `cost-estimate` skill and `scripts/offpeak_window.py` first, read the balance, and ask with the dollar figure |

Use a separate `--events-out` for each prompt (the step refuses a mixed log), and always pass `--input`.
Do not pass `--reasoning-effort` or `--domain-rendering` unless reproducing a measured variant.
