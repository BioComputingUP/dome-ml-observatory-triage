# Cost dashboard

Generated 2026-09-15 22:41 UTC · prices as of **2026-09-15** (`pricing/pricing.yaml`) · corpus counts as of **2026-09-15 22:41 UTC** · token profiles measured **2026-09-15** on DeepSeek-V4.1-Flash (`pricing/token_profiles.yaml`) · billed runs from `data/processed/cost_estimates/deepseek_real_cost_log.csv`.
Regenerate: `python3 scripts/cost_dashboard.py --live --balance`. Timing: `python3 scripts/offpeak_window.py --minutes N`.

## Corpus now

| | Records |
|---|---:|
| Documents in `dome_observatory.Content` | 852,931 |
| Positive / negative / undeterminable | 368,025 / 477,919 / 6,987 |
| Enriched | 3,532 |
| **Positives left to enrich** (with an abstract) | **361,244** (364,493 incl. no-abstract) |
| Citation counts never fetched / older than 30 days | 16,027 / 0 |

## DeepSeek balance

**USD 12.80** at 2026-09-15 22:41 UTC (`GET /user/balance`). Top up before any run whose planning cost below exceeds it.

## What a record costs

Every run below is costed at the **planning rate**, which comes from the bill, not from list prices:

- Classification: **$0.000199 per record** ($0.20 per 1,000). Billed: 6,216 records on 2026-09-15.
- Enrichment: **$1.80 per 1,000 records** ($0.0018 per record). Billed V4.1 runs on 2026-09-15: $1.80 per 1,000 (100 Bioinformatics records) and $1.00 per 1,000 (200 records of a refresh batch); the higher is kept. V4-Flash was billed about $10 per 1,000. Cost varies with how long the population makes the model think. The latest billed run came to $1.00 per 1,000 (200 records, 2026-09-15); the higher figure is kept.

### Billed runs (balance deltas)

| Date | Step | Tier, mode | Records | Billed | Per 1,000 |
|---|---|---|---:|---:|---:|
| 2026-08-21 | classification | flash, primary | 7,600 | $2.31 | $0.30 |
| 2026-09-15 | classification | flash, primary | 1,000 | $0.31 | $0.31 |
| 2026-09-15 | enrichment | flash, enrich | 100 | $0.18 | $1.80 |
| 2026-09-15 | classification | flash, primary | 6,216 | $1.24 | $0.20 |
| 2026-09-15 | enrichment | flash, enrich | 200 | $0.20 | $1.00 |

### List-price model (tokens × list price, not billed)

For comparing providers and peak against off-peak only. For enrichment it gives $0.9845 per 1,000 off-peak against a planning rate of $1.80: it undershoots the bill, so never budget from it.

| Step | Tokens per record (cache hit / miss / output) | DeepSeek V4.1 Flash, off-peak | DeepSeek V4.1 Flash, peak | GLM-5.3-Flash, list |
|---|---|---:|---:|---:|
| Classification (prompt v1) | 8,367 / 300 / 172 | $0.0002 | $0.0003 | $0.0004 |
| Enrichment (prompt e1) | 2,560 / 588 / 1,481 (1,336 reasoning) | $0.0010 | $0.0020 | $0.0009 |

Per 1,000 records:

| Step | DeepSeek V4.1 Flash, off-peak | DeepSeek V4.1 Flash, peak | GLM-5.3-Flash, list |
|---|---:|---:|---:|
| Classification | $0.1733 | $0.3466 | $0.3820 |
| Enrichment | $0.9845 | $1.97 | $0.9055 |

## What the next runs cost

| Run | Records | **Planning (billed rate)** | DeepSeek V4.1 Flash, off-peak (list model) | DeepSeek V4.1 Flash, peak (list model) | GLM-5.3-Flash, list (list model) | Wall time |
|---|---:|---:|---:|---:|---:|---|
| Classify one incremental batch (last batch: 6,216) | 6,216 | **$1.24** | $1.08 | $2.15 | $2.37 | 1 min at concurrency 400 |
| Enrich that batch's positives (last batch: 1,791) | 1,791 | **$3.22** | $1.76 | $3.53 | $1.62 | 4 min at concurrency 800 |
| Enrich 300 positives (a capped cohort) | 300 | **$0.5400** | $0.2953 | $0.5907 | $0.2717 | 1 min at concurrency 800 |
| Enrich 10,000 positives (one journal-sized cohort) | 10,000 | **$18.00** | $9.84 | $19.69 | $9.05 | 24 min at concurrency 800 |
| **Enrich every remaining positive** (361,244) | 361,244 | **$650** | $356 | $711 | $327 | 14.4 h at concurrency 800 |

At the planning rate the full enrichment backlog is **$650**; the list-price model says $356 off-peak. Classification of a monthly batch is a rounding error next to it.

## Best time to run (DeepSeek)

Peak, at double price: 01:00–04:00 UTC and 06:00–10:00 UTC, Mon-Fri. Everything else, including all weekend, is off-peak.

- A monthly classification batch takes minutes: run it any off-peak hour.
- A 10,000-record enrichment takes ~24 min: start after 10:00 UTC on a weekday, or any time at the weekend.
- The full backlog takes ~14.4 h at concurrency 800: only a weekend (Fri 10:00 UTC → Mon 01:00 UTC) holds it entirely off-peak; otherwise run it as journal cohorts, each inside one off-peak stretch. Resumability makes splitting free.
- GLM-5.3-Flash publishes no off-peak rate; its column is the same price at any hour.

## Assumptions and caveats

- **Enrichment's list-price model undershoots its bill.** Tokens × list price gave $4.07 per 1,000 on V4-Flash; enrichment is billed at about $2. Budget from billed runs only: bracket every paid run with `GET /user/balance` reads and append the delta to the real cost log.
- The GLM columns apply DeepSeek's token counts to Z.ai's prices: a different tokenizer and a different reasoning budget would change them, and **GLM-5.3-Flash has not been validated** against the human benchmark or the enrichment agreement check. They are a price comparison, not an approved substitute (see AGENTS.md, ground rule 1).
- Classification cache hits are modelled (the event log does not record them): everything but the ~300 per-record tokens is treated as a prefix-cache hit. On V4-Flash that model came within ~6% of the billed $0.000199/record.
- Enrichment cost is not reducible by settings: lower reasoning effort cost more with six times the vocabulary violations; thinking off was 88% cheaper and agreed with production on all six fields for 0% of records.
- About 5% of new records have no abstract and are never sent to the model; the backlog figure already excludes abstract-less positives.
- List prices drift. The `cost-estimate` skill re-reads both pricing pages before any paid run and updates `pricing/pricing.yaml`; real spend is confirmed from the balance delta afterwards.
