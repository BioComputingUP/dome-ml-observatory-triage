---
name: classify
description: >
  Classify a staged batch of new records (positive / negative / undeterminable) with the validated
  DeepSeek V4 Flash second-curator prompt, after a live cost estimate and explicit confirmation.
  Trigger on "classify the staged batch", "run the classification", "classify incoming_new.csv".
  Costs money (about $0.0003 per record off-peak). Writes only an events CSV; never touches moros.
---

# classify

Runs the `dome-triage` CLI in Docker. Input is the staged CSV `triage-fetch` produced. The prompt,
criteria and parser are validated and are not to be changed here (see `AGENTS.md` rule 1 and
`prompts/README.md`).

## Before spending

1. `python3 prompts/render_prompts.py --check` — the classification prompt still renders to the
   recorded criteria sha256. If it fails, stop: the criteria changed and this is a new prompt
   version that needs re-validation, not a run.
2. Run the `cost-estimate` skill for `N = rows in the staged CSV`: live pricing, balance, and
   `python3 scripts/offpeak_window.py --minutes 5`. Classification takes ~3 minutes per 13,000
   records at `--concurrency 400`; if now is peak, waiting is nearly always worth it.
3. Confirm `docker ps` shows no `pipeline` container already running.
4. State the projection and ask the user to confirm the amount. `--estimated-usd` is that number.

## Procedure

Smoke first when anything is new (a new criteria version, a new machine):

```bash
docker compose run --rm pipeline dome-triage llm-classify classify \
    --scope staged_file --input /app/moros_pipeline/output/incoming_new.csv \
    --tier flash --concurrency 400 --limit 50 --estimated-usd 0.02 --confirm
```

Then the batch (resumes: the 50 are not re-paid for):

```bash
docker compose run --rm pipeline dome-triage llm-classify classify \
    --scope staged_file --input /app/moros_pipeline/output/incoming_new.csv \
    --tier flash --concurrency 400 --estimated-usd <X> --confirm
```

The events file defaults to `moros_pipeline/output/incoming_new_classification_events.csv`,
beside the input; pass `--events-out` only to name a different batch, never to append to another
batch's log (different `record_id` spaces break resumption for both).

**Monitoring.** Judge progress by the events file, not the exit code:
`wc -l moros_pipeline/output/incoming_new_classification_events.csv`. A detached
`docker compose run` returns non-zero while its container keeps working; never start a second run
on the same file — the lock will refuse it, and `--ignore-lock` is only for a container the daemon
has genuinely lost (`docker ps` shows none).

**Parse errors** (typically ~0.2%) are retried by simply re-running the same command; already
classified records are skipped.

## After

Summarise from the events file: positive / negative / undeterminable / parse_error counts, mean
input and output tokens, the `batch_id`, and the spend logged to
`data/processed/deepseek_second_curator_spend_log.csv`. Read the balance again and report the
real delta against the projection.

Then ask: **"Build the documents and load them into moros?"** Hand to `moros-write` on a yes.
