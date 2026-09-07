# EPMC AI/ML Search-Space Licensing

Standalone, deliberately **outside** the `dome_triage` package and **not run through Docker** (per
Gavin's explicit request, 2026-08-24) -- two plain `python3` scripts, only
`requests`/`pandas`/`matplotlib`/`tqdm` (all already present in the host environment).

## Why this exists

Before any future "full download"/redistribution feature ships for the planned searchable AI/ML
landscape database (`STEPS_Progress.md`'s Phase 8), the real licensing terms of the underlying EPMC
records need to be known -- Europe PMC's reuse terms differ per article (Open Access vs. not, and
which specific license when it is), and this project's own data does not carry that yet: `data/
interim/bulk_candidates.csv`'s `is_open_access` column is a bare True/False flag, not the real
license string.

## What it does

1. `fetch_licensing.py` -- for every PMID in the real ~745k-record AI/ML EPMC search-space pool
   (`data/interim/bulk_candidates.csv`, the same pool `STEPS_Progress.md` Steps 22/23 plan to
   triage), fetches the real `license` string and `isOpenAccess` flag from the Europe PMC API,
   batched 300 PMIDs per HTTP request (real, binary-searched ceiling: 360 works, 370+ fails) and 35
   requests concurrently (real-measured: ~685 records/sec, 0 errors -- see the script's own
   docstring for the calibration numbers). Streams results to disk one batch at a time -- resumable,
   re-running the exact same command skips every PMID already fetched rather than re-requesting it.
2. `visualize_licensing.py` -- reads that output and produces the real license-type breakdown, the
   coarser open-access-vs-not breakdown, and a coverage stat (how many target PMIDs actually got a
   result back from EPMC).

## Output

`output/epmc_pmid_licensing.csv` -- exactly `pmid,license,is_open_access`, one row per PMID EPMC
returned a result for (a PMID with no license disclosed -- i.e. not open access -- has an empty
`license` field, not a missing row; a PMID EPMC has no record of at all under `SRC:MED` is simply
absent).

`output/license_breakdown.png`, `output/open_access_breakdown.png`, `output/licensing_summary.json`
-- the real counts behind both charts, plus coverage.

## Running it

```bash
cd epmc_licensing
python3 fetch_licensing.py
# -> ~18 minutes for the full ~745k pool at the default concurrency; safe to Ctrl+C and re-run
#    later, it resumes from wherever it left off (nothing already-fetched is re-requested)
python3 visualize_licensing.py
```

Both scripts default to the real project paths (`../data/interim/bulk_candidates.csv` for input,
`output/` for everything written) -- override with `--input`/`--output`/`--licensing`/
`--target`/`--output-dir` for a smaller test run or a different pool.
