---
name: cost-estimate
description: >
  Before any paid step, or on request: pull the live DeepSeek and Z.ai pricing pages and the live
  DeepSeek balance, refresh pricing/pricing.yaml if a price changed, regenerate COST_DASHBOARD.md
  from live corpus counts, and say what the requested run costs and when to run it (off-peak).
  Trigger on "what would it cost", "update the cost dashboard", "best time to run", "is now
  off-peak", "check the balance", and from inside the classify / enrich / refresh-cycle skills.
  Free and read-only.
---

# cost-estimate

## 1. Live prices

Fetch both pages and compare with `pricing/pricing.yaml`:

- DeepSeek: https://api-docs.deepseek.com/quick_start/pricing — the `deepseek-v4-flash` and
  `deepseek-v4-pro` rows (input cache hit / cache miss / output, per 1M tokens, USD), the peak
  windows and the "off-peak rates are half of the peak rates" rule.
- Z.ai: https://docs.z.ai/guides/overview/pricing — the `GLM-5.3-Flash` row (input / cached input
  / output), any promotion and its end date and timezone.

If any number or window differs, edit `pricing/pricing.yaml` (values and `as_of`), tell the user
exactly what changed, and mention it in the estimate. If a page cannot be fetched, say so and use
the file's values with their `as_of` date; never guess a price.

Confirm the model ids still exist:
`curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/models`.

## 2. Live balance

```bash
curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/user/balance
```

(`DEEPSEEK_API_KEY` from the root `.env`.) Report `total_balance`. A run whose projection exceeds
it will fail part-way; resumability makes that recoverable, not free.

## 3. Regenerate the dashboard

```bash
python3 scripts/cost_dashboard.py --live --balance     # on the VPN; else omit --live for the snapshot
```

Read `COST_DASHBOARD.md` back. Its model: per record, `(cache-hit tokens x hit price + cache-miss
tokens x miss price + output tokens x output price) / 1e6`, with the measured token profiles in
`pricing/token_profiles.yaml` (classification ~8,400 cached / 300 missed / 300 out; enrichment
~2,845 cached / 233 missed / 5,877 out). It reproduces the two real anchors: $0.000304 per
classified record (7,600 records, dashboard-confirmed) and $4.07 per 1,000 enriched records.

## 4. The answer for the requested run

State, in one short table: records, DeepSeek off-peak, DeepSeek peak, GLM-5.3-Flash list (a price
comparison only; that model is not validated for this pipeline), wall time at the standard
concurrency (classify 400 → ~4,500 records/min; enrich 800 → ~418 records/min), and the balance.

Then timing: `python3 scripts/offpeak_window.py --minutes <wall time>`. Peak (double price) is
01:00–04:00 and 06:00–10:00 UTC, Monday to Friday. Recommend the earliest fully off-peak start;
a run longer than ~15 hours only fits a weekend (Fri 10:00 UTC → Mon 01:00 UTC), otherwise split
it into cohorts (resumability makes that free).

## 5. Afterwards

When the run has finished, read the balance again and report the real delta beside the
projection. If they differ by more than ~25%, re-measure the token profile from the run's events
file (`python3 scripts/cost_dashboard.py --events-classification <csv>` or
`--events-enrichment <csv>`) and say what moved.
