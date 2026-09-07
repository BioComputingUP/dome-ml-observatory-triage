# pricing/

- `pricing.yaml` — provider list prices as of a date, with the source page and the off-peak rule.
  Refreshed by the `cost-estimate` skill (it fetches the two pricing pages, compares, and edits
  this file with a new `as_of`).
- `token_profiles.yaml` — measured tokens per record for classification and enrichment, from
  real event logs, with the real-spend anchors.
- `corpus_snapshot.json` — the corpus counts the dashboard was last generated from, written by
  `scripts/cost_dashboard.py --live` (read-only query of moros).

`scripts/cost_dashboard.py` multiplies the three into `COST_DASHBOARD.md`.

Why list price is not trusted on its own: DeepSeek prefix-caches the byte-identical system
message, so real classification spend came out at roughly one fifth of a naive cache-miss
projection. The model below applies the cache-hit rate to everything except the per-record
message, which reproduces the dashboard-confirmed $0.000304 per record within 6%.
