---
name: enrich
description: >
  Enrich a cohort of positive records in moros with the controlled vocabularies (EDAM domain
  tiers, learning paradigm, model family, model type) using the validated enrichment prompt (e1,
  DeepSeek's flash tier), after a live cost estimate, an off-peak timing check and explicit
  confirmation; then hand the events file to moros-write. Trigger on "enrich journal X", "enrich
  the new positives", "run the enrichment", "how much would enriching Y cost". Costly: billed at about
  $1.80 per 1,000 records on V4.1 Flash, roughly nine times classification.
---

# enrich

Enrichment is a separate, deliberate, costly cycle, run per journal or cohort when wanted. It is
additive by construction (the prompt never asks the positive/negative question) and the writer's
allowlist makes that structural. The prompt (`e1`), vocabularies and parser are validated and are
not changed here.

## 1. Choose the cohort

Journal names are matched **verbatim** as stored (`Science` is `Science (New York, N.Y.)`). To see
what is available, read-only:

```bash
cd moros_pipeline/scripts
python3 - <<'PY'
from moros_client import Moros
with Moros.from_env() as m:
    pipe = [{"$match": {"llm_classification.classification": "positive", "llm_enrichment.batch_id": None}},
            {"$group": {"_id": "$publication_metadata.journal", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}}, {"$limit": 40}]
    for row in m.collection.aggregate(pipe, maxTimeMS=120000):
        print(f"{row['n']:>7}  {row['_id']}")
PY
```

Export (already-enriched records are excluded; `--max-usd` refuses an input whose projection
exceeds it, default 20):

```bash
python3 export_journal_for_enrichment.py \
    --journal "<name>" [--journal "<name 2>" ...] \
    --out ../output/enrich_input_<name>.csv --max-usd <cap>
```

Or the batch a refresh just loaded, capped at N records:

```bash
python3 export_journal_for_enrichment.py \
    --batch-id <llm_classification.batch_id> --limit <N> \
    --out ../output/enrich_input_<name>.csv --max-usd <cap>
```

`--max-usd` costs what the export will write (`--limit` included) at the billed rate.

## 2. Cost and timing, then ask

Run the `cost-estimate` skill for `N = rows exported`: at the planning rate in `COST_DASHBOARD.md`
(billed: $1.80 per 1,000 on V4.1 Flash, from `deepseek_real_cost_log.csv`; never the list-price
model, which undershot V4-Flash's bill by more than half) and ~418 records/min at `--concurrency 800`, so
`minutes = N / 418`. Then
`python3 scripts/offpeak_window.py --minutes <that>`; if the run would cross a peak window,
propose the earliest fully off-peak start (weekends hold anything). Check the balance covers the
projection, and read it immediately before starting. State all of it and ask: **"Enrich N records for about $X, starting now / at T?"**

## 3. Run

```bash
docker ps                                   # no pipeline container may be running
docker compose run --rm pipeline dome-triage llm-classify enrich \
    --tier flash --concurrency 800 \
    --input /app/moros_pipeline/output/enrich_input_<name>.csv \
    --events-out /app/moros_pipeline/output/enrichment_<name>_events.csv
```

**`--events-out` is mandatory**, one file per cohort. **`--input` is mandatory** (the CLI's default
input is a research file that is not here). Do not pass `--reasoning-effort` or
`--domain-rendering`: the defaults are the validated configuration, and every alternative was
measured worse or no cheaper.

Monitor by `wc -l` on the events file and `docker ps`, never by the compose exit code. A second
run on the same file is refused by the lock; that is correct. `--limit 25` gives a real smoke run
first if anything is new.

## 4. After the run

From the events file report: ok / parse_error, how many `finish_reason == length` (truncated at
the 16,000-token cap, ~1.5%; they are **not** retried automatically because retrying at the same
cap re-truncates — raising `ENRICHMENT_MAX_TOKENS` is a versioned change for the user to decide),
vocabulary-violation rate (trial baseline 6.8%, production 4.1%), prefix-cache hit share, mean
output and reasoning tokens, and the real balance delta. The balance lags the run by minutes: poll it
until it moves and settles, start no other paid run meanwhile, then append a row to
`data/processed/cost_estimates/deepseek_real_cost_log.csv` (`mode` `enrich`, `source`
`balance_delta`).

Then ask: **"Merge these into moros?"** Hand to `moros-write` section B on a yes.
