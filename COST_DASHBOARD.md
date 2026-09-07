# Cost dashboard

Generated 2026-09-07 21:02 UTC · prices as of **2026-09-07** (`pricing/pricing.yaml`) · corpus counts as of **2026-09-07 21:02 UTC** · token profiles measured **2026-09-03** (`pricing/token_profiles.yaml`).
Regenerate: `python3 scripts/cost_dashboard.py --live --balance`. Timing: `python3 scripts/offpeak_window.py --minutes N`.

## Corpus now

| | Records |
|---|---:|
| Documents in `dome_observatory.Content` | 846,716 |
| Positive / negative / undeterminable | 366,234 / 473,503 / 6,979 |
| Enriched | 3,332 |
| **Positives left to enrich** (with an abstract) | **359,653** (362,902 incl. no-abstract) |
| Citation counts never fetched / older than 30 days | 16,027 / 0 |

## DeepSeek balance

**USD 4.73** at 2026-09-07 21:02 UTC (`GET /user/balance`). Top up before any run whose projection below exceeds it.

## Price per record

| Step | Tokens per record (cache hit / miss / output) | DeepSeek V4 Flash, off-peak | DeepSeek V4 Flash, peak | GLM-5.3-Flash, list | GLM-5.3-Flash, promo (until 2026-09-09 UTC) | Measured, real |
|---|---|---:|---:|---:|---:|---:|
| Classification (prompt v1) | 8,398 / 300 / 299 | $0.0003 | $0.0006 | $0.0004 | $0.0002 | $0.000304 (7,600 records, dashboard-confirmed) |
| Enrichment (prompt e1) | 2,845 / 233 / 5,877 (5,752 reasoning) | $0.0039 | $0.0079 | $0.0031 | $0.0015 | $0.00407 ($4.07 per 1,000; two independent runs) |

Per 1,000 records, for planning:

| Step | DeepSeek V4 Flash, off-peak | DeepSeek V4 Flash, peak | GLM-5.3-Flash, list | GLM-5.3-Flash, promo (until 2026-09-09 UTC) |
|---|---:|---:|---:|---:|
| Classification | $0.3221 | $0.6443 | $0.4464 | $0.2232 |
| Enrichment | $3.95 | $7.90 | $3.06 | $1.53 |

## What the next runs cost

| Run | Records | DeepSeek V4 Flash, off-peak | DeepSeek V4 Flash, peak | GLM-5.3-Flash, list | GLM-5.3-Flash, promo (until 2026-09-09 UTC) | Wall time |
|---|---:|---:|---:|---:|---:|---|
| Classify one incremental batch (last batch: 13,499) | 13,499 | $4.35 | $8.70 | $6.03 | $3.01 | 3 min at concurrency 400 |
| Enrich that batch's positives (last batch: 7,369) | 7,369 | $29.11 | $58.22 | $22.54 | $11.27 | 18 min at concurrency 800 |
| Enrich 10,000 positives (one journal-sized cohort) | 10,000 | $39.50 | $79.00 | $30.59 | $15.29 | 24 min at concurrency 800 |
| **Enrich every remaining positive** (359,653) | 359,653 | $1,421 | $2,841 | $1,100 | $550 | 14.3 h at concurrency 800 |

At the **measured** enrichment rate ($4.07/1,000, DeepSeek off-peak, including the ~1.5% truncated re-tries) the full backlog is **$1,464**. Classification of a monthly batch is a rounding error next to it.

## Best time to run (DeepSeek)

Peak, at double price: 01:00–04:00 UTC and 06:00–10:00 UTC, Mon-Fri. Everything else, including all weekend, is off-peak.

- A monthly classification batch takes minutes: run it any off-peak hour.
- A 10,000-record enrichment takes ~24 min: start after 10:00 UTC on a weekday, or any time at the weekend.
- The full backlog takes ~14.3 h at concurrency 800: only a weekend (Fri 10:00 UTC → Mon 01:00 UTC) holds it entirely off-peak; otherwise run it as journal cohorts, each inside one off-peak stretch. Resumability makes splitting free.
- GLM-5.3-Flash publishes no off-peak rate; its column is the same price at any hour.

## Assumptions and caveats

- Token counts are DeepSeek's own `usage` figures from real runs. The GLM column applies the same counts to Z.ai's prices: a different tokenizer and a different reasoning budget would change them, and **GLM-5.3-Flash has not been validated** against the human benchmark or the enrichment agreement check. It is a price comparison, not an approved substitute (see ROADMAP.md).
- Classification cache hits are modelled (the event log does not record them): everything but the ~300 per-record tokens is treated as a prefix-cache hit. The model reproduces the dashboard-confirmed $0.000304/record within ~6%.
- Enrichment cost is not reducible by settings: lower reasoning effort cost more with six times the vocabulary violations; thinking off was 88% cheaper and agreed with production on all six fields for 0% of records.
- About 5% of new records have no abstract and are never sent to the model; the backlog figure already excludes abstract-less positives.
- List prices drift. The `cost-estimate` skill re-reads both pricing pages before any paid run and updates `pricing/pricing.yaml`; real spend is confirmed from the balance delta afterwards.
