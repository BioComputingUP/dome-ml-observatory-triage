# DeepSeek Real Cost Log

Real, dashboard-confirmed DeepSeek spend data points, logged as they happen -- never a projected
or list-price figure, per this project's established "always calibrate/confirm from real numbers"
rule. `deepseek_real_cost_log.csv` is the raw log (one row per real, confirmed batch); this file
holds the derived per-record economics and scale-up projections computed from it.

## Current real anchor point (2026-08-21)

Step 20e's full re-run: **7,600 records classified for a real, dashboard-confirmed $2.31** (flash
tier, primary 3-way prompt, current `CRITERIA.md`). This is by far the largest real data point
this project has -- far more reliable than any small n=8-30 calibration batch, since it's an actual
7,600-record production run, not an extrapolation from a handful of calls.

- **$0.00030395 / record** (≈0.0304 cents/record)

**Important caveat**: this is the cost of the *simple* 3-way classification prompt
(positive/negative/undeterminable) only. Step 23's planned full-landscape run uses a *multi-label*
prompt (adds `categories` + `model_types`, two extra list-valued outputs) -- expected to cost more
per record than this figure, not less, because of longer completions. Treat every projection below
as a **lower bound** for a full multi-label landscape run, not a final number -- a fresh calibration
against the actual multi-label prompt (Step 22(c) in `STEPS_Progress.md`) is still required before
spending real money at this scale.

## Scale-up projections (linear extrapolation from the real anchor point above)

| Target population | n records | Projected cost (simple prompt, lower bound) |
|---|---|---|
| Current live bulk pool (`data/interim/bulk_candidates.csv`, real line count) | 745,499 | **$226.59** |
| ~760k (rounded reference figure) | 760,000 | **$231.00** |
| Net of already-curated 8,624 (rough -- not an exact exclusion-set match) | 736,875 | $223.97 |
| 1,000,000 (wider-search reference scenario) | 1,000,000 | **$303.95** |
| 1,500,000 (wider-search reference scenario) | 1,500,000 | **$455.92** |

For context, the live bulk-match query itself (`"artificial intelligence" OR "machine learning"`,
2000-2026) currently returns 827,890 combined EPMC hits (`bulk_match_summary.csv`, run
2026-07-31) -- the 745,499-row materialized pool is somewhat smaller after fetch-time filtering.
The 1M/1.5M scenarios represent widening the query with additional AI/ML-adjacent terms (deep
learning, reinforcement learning, federated learning, named model types, etc. -- see Step 22(a) in
`STEPS_Progress.md`), not yet run for real.

## What this means practically

Both of this project's existing spend caps are far below every scale-up scenario above:
`configs/pipeline.yaml`'s DeepSeek sub-cap ($10) and `AGENTS.md`'s whole-project cap (£100) would
both need to be explicitly raised before any of the 745k+/1M/1.5M scenarios could actually run --
`llm_budget.check_cap`'s existing hard-refuse-without-`--confirm` design already exists to force
that decision into the open at the real number, not this document's estimate.

## Updating this log

Append a new row to `deepseek_real_cost_log.csv` every time a real batch's dashboard cost is
confirmed (not projected) -- especially once a real multi-label-prompt calibration exists (Step
22(c)), since that will be the first real data point for the prompt shape Step 23 actually uses.
