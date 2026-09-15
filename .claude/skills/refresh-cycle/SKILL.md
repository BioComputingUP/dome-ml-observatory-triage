---
name: refresh-cycle
description: >
  Run the whole recurring refresh in order with a check-in before every paid or writing step:
  triage-fetch → (ask) classify → (ask) moros-write load → (ask) enrich → moros-write merge →
  optional citations-refresh, then the post-load checklist and processing-log. Trigger on "do the
  monthly refresh", "run the whole pipeline", "bring the corpus up to date", or a pasted
  BULK_UPDATE block. Never runs a paid or writing step without the user's explicit yes at that step.
---

# refresh-cycle

Each step is its own skill and is independently resumable; this skill only sequences them and
holds the check-ins. Read `SKILLS.md` for the map. Preconditions: VPN, `moros_pipeline/.env`,
root `.env` with `DEEPSEEK_API_KEY`, the Docker image built, `docker ps` shows no `pipeline`
container.

## Parameters

A run is usually started from the block in `BULK_UPDATE.md`, pasted with its values filled in.
Restate the values back in one short table before step 1; anything left out takes the default the
template shows. Parameters choose which steps run and how far. **They never remove a check-in**:
every paid or writing step still stops for a yes.

| Parameter | What it controls |
|---|---|
| `coverage_up_to` | `--up-to` for the fetch dry run and the fetch, which normally runs as `--indexed-since last` (everything first indexed since the last fetch). "N weeks" or "N months" means N after the last cutoff: `coverage_ledger.indexed_through`. Say the date you computed. |
| `classify` | `no` stops after step 2; the batch stays staged. |
| `classify_smoke_limit` | the `--limit` of the smoke run in step 3. |
| `load_to_moros` | `no` stops after step 3; the events file is kept. Every verdict loads: an unloaded record comes back as new and is paid for again. |
| `data_links_for_batch` | `no` passes `--datalinks-scope none` and skips the EBI Search fetches in step 4; the next data-links refresh completes them. |
| `enrich` | `no`, `batch_positives` (this run's classification batch, `--batch-id`) or `journal:"<exact name>"`. |
| `enrich_max_records` | the export's `--limit`. |
| `enrich_max_usd` | the export's `--max-usd`. |
| `citations_refresh` | `yes` runs `citations-refresh` in step 6. |
| `data_links_refresh` | `when_due` runs it only past its age thresholds; `yes` or `no` decides. |
| `wait_for_off_peak` | `yes`: a paid step that would cross a peak window waits for the earliest fully off-peak start and says so. `no`: state the peak cost and ask. |
| `update_processing_page` | `yes` runs `processing-log` in step 8. |
| `deploy_page` | `yes` is the user's request to deploy the page after its commit. Default `no`. |
| `notes` | followed where they do not remove a check-in. |

## Steps

1. **Baseline.** `python3 moros_pipeline/scripts/verify_corpus.py` (note the count),
   `python3 schema/check_alignment.py --live`, `python3 scripts/check_no_absolute_paths.py` and
   `python3 prompts/render_prompts.py --check`. Drift is a stop unless the user accepts it.
2. **`triage-fetch`** with `--indexed-since last --up-to <coverage_up_to>` (year windows only for a
   year never fetched). Report the window, records returned, how many are new, how many are already in
   the corpus under an older `_id`, and how many have no abstract.
   → **Ask:** classify N records for ~$X (`cost-estimate`: live prices, the model that answers,
   balance, dashboard regenerated, off-peak status)?
3. **`classify`.** Read the balance. Smoke `--limit <classify_smoke_limit>`, then the batch. Poll the
   balance until the charge settles (it lags by minutes) and log the delta. Report counts and the
   billed cost.
   → **Ask:** build documents and load into moros?
4. **`moros-write` A.** Citations + licences into the batch's own file, and data links for the batch
   (Europe PMC for every record, EBI Search for the positives), build, dry run, `--limit 100` trial, confirm,
   `ensure_indexes.py`, `verify_corpus.py --expect-count`. Report inserted / left out, how many new
   documents carry data links, and how many carry a DOME Registry entry.
   → **Ask:** enrich now (the `enrich` cohort, capped at `enrich_max_records`; N records at the
   planning rate in `COST_DASHBOARD.md`, which comes from the bill, never the list-price model;
   T minutes; off-peak window from `offpeak_window.py`; balance)? A different cohort? Not this time?
5. **`enrich`** (if yes): `export_journal_for_enrichment.py --batch-id <classification batch id>
   --limit <enrich_max_records> --max-usd <enrich_max_usd>`, read the balance, `--limit 25` smoke,
   the rest, poll the balance and log the delta. Then **`moros-write` B** to merge: dry run,
   `--limit 25 --confirm`, `--confirm`. Report ok / truncated / violations and the billed cost.
6. **`citations-refresh`** if `citations_refresh` is yes (free); the **`data-links`** refresh
   (`--max-age-days 180` on the Europe PMC fetches, 30 on the EBI Search dumps; free) when due.
7. **Post-load checklist** (from `moros-write`): ask before restarting `observatory-ws` (production),
   reconcile `generate_facet_stats.py --from-api`, `schema-sync` if the shape changed.
8. **Close.**
   - Re-measure the token profiles from this run's events files
     (`python3 scripts/cost_dashboard.py --events-classification <csv> --events-enrichment <csv>`),
     set `last_incremental_batch_*` in `pricing/token_profiles.yaml`, and regenerate
     `COST_DASHBOARD.md` with `--live --balance`.
   - If `update_processing_page`: the `processing-log` skill (commit only; deploy only if
     `deploy_page` is yes).
   - Update the "Last run" note at the top of `BULK_UPDATE.md`: date, window, counts, billed spend,
     the model that answered.
   - Give the user a short record: date, window fetched, new / classified / loaded / enriched
     counts, the new corpus count, spend projected vs billed, the run ids and rollback paths, and
     anything left undone.
