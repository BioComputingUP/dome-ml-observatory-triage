---
name: refresh-cycle
description: >
  Run the whole recurring refresh in order with a check-in before every paid or writing step:
  triage-fetch → (ask) classify → (ask) moros-write load → (ask) enrich → moros-write merge →
  optional citations-refresh, then the post-load checklist. Trigger on "do the monthly refresh",
  "run the whole pipeline", "bring the corpus up to date". Never runs a paid or writing step
  without the user's explicit yes at that step.
---

# refresh-cycle

Each step is its own skill and is independently resumable; this skill only sequences them and
holds the check-ins. Read `SKILLS.md` for the map. Preconditions: VPN, `moros_pipeline/.env`,
root `.env` with `DEEPSEEK_API_KEY`, the Docker image built, `docker ps` shows no `pipeline`
container.

1. **Baseline.** `python3 moros_pipeline/scripts/verify_corpus.py` (note the count),
   `python3 schema/check_alignment.py --live`, and
   `python3 scripts/check_no_absolute_paths.py`. Drift is a stop unless the user accepts it.
2. **`triage-fetch`.** Report windows, new records, no-abstract records.
   → **Ask:** classify N records for ~$X (from `cost-estimate`, with off-peak status)?
3. **`classify`.** Smoke `--limit 50` if anything is new, then the batch. Report counts and the
   real balance delta.
   → **Ask:** build documents and load into moros?
4. **`moros-write` A.** Citations + licences, and data links for the batch (Europe PMC for every
   record, EBI Search for the positives), build, dry run, `--limit 100` trial, confirm,
   `ensure_indexes.py`, `verify_corpus.py --expect-count`. Report inserted / left out, how many new
   documents carry data links, and how many carry a DOME Registry entry.
   → **Ask:** enrich the batch's positives now (~$4 per 1,000; N positives ≈ $Y; T minutes; off-peak
   window from `offpeak_window.py`)? Or a different cohort? Or not this month?
5. **`enrich`** (if yes) with its own `--events-out`, then **`moros-write` B** to merge. Report
   ok / truncated / violations and the balance delta.
6. **`citations-refresh`** (monthly, free) if the user wants it in the same session; the
   **`data-links`** refresh (`--max-age-days 180` on the Europe PMC fetches, 30 on the EBI Search
   dumps; free) when it is due.
7. **Post-load checklist** (from `moros-write`): restart `observatory-ws`, reconcile
   `generate_facet_stats.py --from-api`, `schema-sync` if the shape changed.
8. **Close.** Regenerate `COST_DASHBOARD.md` (`cost-estimate` step 3), and give the user a short
   record: date, windows fetched, new / classified / loaded / enriched counts, the new corpus
   count, spend projected vs real, the run ids and rollback paths, and anything left undone.
