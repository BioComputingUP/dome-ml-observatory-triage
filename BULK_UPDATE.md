# BULK_UPDATE

A template for running a refresh of the corpus. Copy the block under **The block** into Claude Code
in this repository, change the values, and send it. Claude runs the `refresh-cycle` skill with those
settings. It still stops and asks before every step that costs money or writes to the database,
whatever the block says.

## Last run

| | |
|---|---|
| Date | 15 September 2026 (practice run) |
| Window fetched | First indexed by Europe PMC 3–10 September 2026, plus papers dated after 3 September |
| New / classified / loaded | 7,342 new (141 set aside as already held under an older id) / 6,215 classified / 6,215 loaded |
| Verdicts | 1,791 positive, 4,416 negative, 8 undeterminable |
| Enriched | 200 of the round's positives (enrichment now covers 3,532 records) |
| Corpus after | 852,931 documents |
| Billed | $1.24 classification, $0.20 enrichment |
| Model that answered | DeepSeek V4.1 Flash, called as `deepseek-v4-flash` |
| Left undone | About 36,700 2026 papers an incomplete fetch on 3 September missed are staged in `moros_pipeline/output/incoming_new_year_refetch_20260915.csv`, unclassified (about $7 to classify). Around 10,000 papers are in the corpus twice, from that same run, and need a decision. |

## The block

```text
BULK_UPDATE
coverage_up_to: today           # today | YYYY-MM-DD | "3 months" (3 months after the last cutoff)
classify: yes                   # yes | no
classify_smoke_limit: 50        # records in the smoke run before the full batch
load_to_moros: yes              # yes | no
data_links_for_batch: yes       # yes | no
enrich: no                      # no | batch_positives | journal:"Bioinformatics (Oxford, England)"
enrich_max_records: 0           # the most records to enrich in this run
enrich_max_usd: 0               # the export refuses a cohort projected above this
citations_refresh: no           # yes | no
data_links_refresh: when_due    # when_due | yes | no
wait_for_off_peak: yes          # yes | no
update_processing_page: yes     # yes | no
deploy_page: no                 # no | yes
notes:
```

## What each setting does

| Setting | Default | What it does |
|---|---|---|
| `coverage_up_to` | `today` | How far the search reaches. `today`, a date, or a period after the last cutoff ("2 weeks", "3 months"); Claude says the date it worked out. See **The window** below. |
| `classify` | `yes` | `no` fetches and stages the new records, then stops. Nothing is paid for. |
| `classify_smoke_limit` | `50` | A small paid trial before the full batch, to catch a problem cheaply. |
| `load_to_moros` | `yes` | `no` classifies and keeps the results (the events file) without writing to the database. |
| `data_links_for_batch` | `yes` | Fetches each new record's data links (Europe PMC, and EBI Search for the positives). Free. `no` leaves them for the next data-links refresh. |
| `enrich` | `no` | `batch_positives` tags this run's new positives; `journal:"…"` tags a journal's untagged positives (the name exactly as stored). |
| `enrich_max_records` | `0` | Caps the enrichment. The rest of the cohort stays untagged for a later run. |
| `enrich_max_usd` | `0` | A hard spending limit on the enrichment export: it refuses to write a cohort projected above this. |
| `citations_refresh` | `no` | Re-fetches citation counts older than 30 days. Free; about 17 minutes for the whole corpus. |
| `data_links_refresh` | `when_due` | Refreshes data links older than 180 days (EBI Search dumps: 30). Free. |
| `wait_for_off_peak` | `yes` | DeepSeek charges double 01:00–04:00 and 06:00–10:00 UTC, Monday to Friday. `yes` waits for off-peak; `no` asks you with the peak price. |
| `update_processing_page` | `yes` | Adds the run to the site's Processing history page and commits it in `dome-ml-observatory`. |
| `deploy_page` | `no` | `yes` also publishes the page to the live site. It goes straight to production, with no staging. |
| `notes` | | Anything else. Followed unless it would skip a check-in. |

## Where Claude always stops and asks

1. **Before starting**, if the corpus fails a check, the schema disagrees between the two
   repositories or moros, or the search query has changed.
2. **Before classifying**: the number of new records, the cost from the dashboard, the balance, and
   whether it is off-peak. The dashboard is regenerated with live prices first.
3. **Before loading**: after a dry run and a 100-record trial that can be rolled back.
4. **Before enriching**: the cohort, the cost at the billed rate, the time, the balance.
5. **Before merging the enrichment**: after a dry run and a 25-record trial.
6. **Before restarting `observatory-ws`**, the production API.
7. **Before deploying the page**, unless `deploy_page: yes`.

After every paid step Claude reads the DeepSeek balance until the charge lands (it lags by a few
minutes) and logs what was billed in `data/processed/cost_estimates/deepseek_real_cost_log.csv`.
`COST_DASHBOARD.md` costs every run from those billed figures, not from list prices.

## What it costs

See `COST_DASHBOARD.md` for the current rates. As billed on 15 September 2026, on DeepSeek V4.1 Flash:

- **Classification: about $0.20 per 1,000 records.** A week of new records is well under a pound.
- **Enrichment: about $1.80 per 1,000 records**, and it varies with the papers: 200 records of a
  mixed batch were billed $1.00 per 1,000, 100 Bioinformatics records $1.80. The old V4 Flash model
  was billed about $10 per 1,000, so keep `enrich_max_records` and `enrich_max_usd` set until a few
  more runs confirm the lower rate.
- Fetching, data links, citations and the page are free.

## The window

- The normal refresh asks Europe PMC for everything **first indexed since the last fetch**, whatever
  its publication date: about 5,000 records a week, and it includes papers indexed late for an
  earlier year (78 of them in the practice week).
- Whole-year windows are the fallback, for a year never fetched before or a ledger that has lost
  history. A year window re-reads the entire year (191,536 records for 2026) to find what is new.
- Either way, what counts as new is whatever moros does not already hold, by id and by PMID, DOI or
  PMCID. A paper that gained a PMCID since it was loaded is set aside, not loaded twice.
- A run that reaches into a new year also fetches the rest of the previous year, so nothing at a
  year's end is skipped, and any window that comes back short of Europe PMC's own count fails loudly
  instead of recording itself as done.
- Changing the search terms is not a setting here. It starts coverage from nothing for the new query
  and needs its own decision.

## Why some things are not settings

- **Negatives cannot be left out of the load.** A record that is not loaded looks new next time and
  is classified, and paid for, again.
- **Classification cannot be limited to positives.** The verdict is what classification produces.
- **The check-ins cannot be switched off.** Every paid or writing step waits for your yes.
- **The model cannot be changed here.** A different model needs checking against the human benchmark
  first (`AGENTS.md`, ground rule 1).

## Examples

Classification only, up to today:

```text
BULK_UPDATE
coverage_up_to: today
classify: yes
load_to_moros: yes
enrich: no
citations_refresh: no
update_processing_page: yes
deploy_page: no
```

Three months after the last cutoff, with a capped enrichment of the new positives:

```text
BULK_UPDATE
coverage_up_to: 3 months
classify: yes
load_to_moros: yes
enrich: batch_positives
enrich_max_records: 2000
enrich_max_usd: 10
citations_refresh: yes
update_processing_page: yes
deploy_page: no
```

The practice run of 15 September 2026:

```text
BULK_UPDATE
coverage_up_to: 2026-09-10
classify: yes
classify_smoke_limit: 50
load_to_moros: yes
enrich: batch_positives
enrich_max_records: 200
enrich_max_usd: 3
citations_refresh: no
data_links_refresh: when_due
update_processing_page: yes
deploy_page: no
notes: practice run; the model check came first
```
