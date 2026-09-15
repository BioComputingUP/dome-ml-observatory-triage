# Cost dashboard

Generated 2026-09-15 21:33 UTC · prices as of **2026-09-15** (`pricing/pricing.yaml`) · corpus counts as of **2026-09-15 21:33 UTC** · token profiles measured **2026-09-03** on DeepSeek-V4-Flash-0731 (`pricing/token_profiles.yaml`) · billed runs from `data/processed/cost_estimates/deepseek_real_cost_log.csv`.
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

**USD 14.73** at 2026-09-15 21:33 UTC (`GET /user/balance`). Top up before any run whose planning cost below exceeds it.

## What a record costs

Every run below is costed at the **planning rate**, which comes from the bill, not from list prices:

- Classification: **$0.000304 per record** ($0.30 per 1,000). Billed: 7,600 records on 2026-08-21.
- Enrichment: **$10.00 per 1,000 records** ($0.0100 per record). Billing experience, 3,000 records for about $30-40 (2026-09-15). Replace with a balance delta from deepseek_real_cost_log.csv, and never set it below a billed figure.

### Billed runs (balance deltas)

| Date | Step | Tier, mode | Records | Billed | Per 1,000 |
|---|---|---|---:|---:|---:|
| 2026-08-21 | classification | flash, primary | 7,600 | $2.31 | $0.30 |

### List-price model (tokens × list price, not billed)

For comparing providers and peak against off-peak only. For enrichment it gives $3.57 per 1,000 off-peak against a planning rate of $10.00: it undershoots the bill, so never budget from it.

| Step | Tokens per record (cache hit / miss / output) | DeepSeek V4.1 Flash, off-peak | DeepSeek V4.1 Flash, peak | GLM-5.3-Flash, list |
|---|---|---:|---:|---:|
| Classification (prompt v1) | 8,398 / 300 / 299 | $0.0002 | $0.0005 | $0.0004 |
| Enrichment (prompt e1) | 2,845 / 233 / 5,877 (5,752 reasoning) | $0.0036 | $0.0071 | $0.0031 |

Per 1,000 records:

| Step | DeepSeek V4.1 Flash, off-peak | DeepSeek V4.1 Flash, peak | GLM-5.3-Flash, list |
|---|---:|---:|---:|
| Classification | $0.2496 | $0.4992 | $0.4464 |
| Enrichment | $3.57 | $7.14 | $3.06 |

## What the next runs cost

| Run | Records | **Planning (billed rate)** | DeepSeek V4.1 Flash, off-peak (list model) | DeepSeek V4.1 Flash, peak (list model) | GLM-5.3-Flash, list (list model) | Wall time |
|---|---:|---:|---:|---:|---:|---|
| Classify one incremental batch (last batch: 13,499) | 13,499 | **$4.10** | $3.37 | $6.74 | $6.03 | 3 min at concurrency 400 |
| Enrich that batch's positives (last batch: 7,369) | 7,369 | **$73.69** | $26.31 | $52.61 | $22.54 | 18 min at concurrency 800 |
| Enrich 300 positives (a capped cohort) | 300 | **$3.00** | $1.07 | $2.14 | $0.9176 | 1 min at concurrency 800 |
| Enrich 10,000 positives (one journal-sized cohort) | 10,000 | **$100** | $35.70 | $71.39 | $30.59 | 24 min at concurrency 800 |
| **Enrich every remaining positive** (359,653) | 359,653 | **$3,597** | $1,284 | $2,568 | $1,100 | 14.3 h at concurrency 800 |

At the planning rate the full enrichment backlog is **$3,597**; the list-price model says $1,284 off-peak. Classification of a monthly batch is a rounding error next to it.

## Best time to run (DeepSeek)

Peak, at double price: 01:00–04:00 UTC and 06:00–10:00 UTC, Mon-Fri. Everything else, including all weekend, is off-peak.

- A monthly classification batch takes minutes: run it any off-peak hour.
- A 10,000-record enrichment takes ~24 min: start after 10:00 UTC on a weekday, or any time at the weekend.
- The full backlog takes ~14.3 h at concurrency 800: only a weekend (Fri 10:00 UTC → Mon 01:00 UTC) holds it entirely off-peak; otherwise run it as journal cohorts, each inside one off-peak stretch. Resumability makes splitting free.
- GLM-5.3-Flash publishes no off-peak rate; its column is the same price at any hour.

## Assumptions and caveats

- **The model changed.** Token profiles were measured on DeepSeek-V4-Flash-0731; DeepSeek now answers `deepseek-v4-flash` with DeepSeek-V4.1-Flash (from 2026-09-10), and the list prices above are DeepSeek-V4.1-Flash's. Its token use is unmeasured until its first events files are re-measured (`--events-classification` / `--events-enrichment`), and whether it agrees with the validated model is a separate check (see ROADMAP.md).
- **Enrichment's list-price model undershoots its bill.** Tokens × list price gave $4.07 per 1,000 on V4-Flash; enrichment is billed at about $10. Budget from billed runs only: bracket every paid run with `GET /user/balance` reads and append the delta to the real cost log.
- The GLM columns apply DeepSeek's token counts to Z.ai's prices: a different tokenizer and a different reasoning budget would change them, and **GLM-5.3-Flash has not been validated** against the human benchmark or the enrichment agreement check. They are a price comparison, not an approved substitute (see ROADMAP.md).
- Classification cache hits are modelled (the event log does not record them): everything but the ~300 per-record tokens is treated as a prefix-cache hit. On V4-Flash that model came within ~6% of the billed $0.000304/record.
- Enrichment cost is not reducible by settings: lower reasoning effort cost more with six times the vocabulary violations; thinking off was 88% cheaper and agreed with production on all six fields for 0% of records.
- About 5% of new records have no abstract and are never sent to the model; the backlog figure already excludes abstract-less positives.
- List prices drift. The `cost-estimate` skill re-reads both pricing pages before any paid run and updates `pricing/pricing.yaml`; real spend is confirmed from the balance delta afterwards.
