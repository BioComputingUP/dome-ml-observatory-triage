# Curation queue construction — how records are chosen, ordered, and why the mix looks skewed

This is about the **mechanics** of the queue — which records you see and in what order — not the
labeling rules themselves (see [`CRITERIA.md`](CRITERIA.md) for those). Written 2026-08-03 after a
real question while curating: *"why am I seeing ~400 more negatives than positives?"* Short answer:
it's queue **order**, not queue **composition** — the queue itself is balanced. Full mechanics
below, with the real numbers from the current run so this isn't hand-wavy. The authoritative,
code-level version of all of this lives in `STEPS_Progress.md`'s Step 13 and Step 15a sections —
this file is the curator-facing summary of the same thing, kept short on purpose.

## How a record enters the queue at all (Step 13 — stratified sampling)

Nothing about this step chooses positives or negatives. It chooses a **diverse sample** to put in
front of a human, across three independent dimensions (`src/dome_triage/sampling/stratified.py`):

1. **BM25 score band (Q1–Q4)** — quantile bins (`pd.qcut`), so each band gets the **same record
   count**, not the same score range. Q1 = lowest-scoring quarter, Q4 = highest.
2. **Journal bucket** — the top 15 most frequent journals each get their own bucket; every other
   journal collapses into `"other"`.
3. **Year bucket** — flat 5-year windows (2015, 2020, 2025, …).

For every `(score_band, journal_bucket, year_bucket)` combination, at most `cap_per_stratum` (10)
records are drawn — a seeded random sample (`random_state=42`, deterministic and reproducible), or
every record in that cell if fewer than 10 exist. That cap-per-cell is the actual diversity
mechanism: it's what stops one journal, one year, or one score range from dominating the queue.

**Every record sampled this way gets shown for genuine human review, regardless of its BM25 tag.**
The BM25 "positive"/"negative" label you see on each paper is a keyword-search heuristic hint (with
a tooltip against the validated Youden threshold), never something used to pre-filter or pre-decide
what reaches you.

## Why the queue *looks* negative-heavy for a long stretch

Two facts combine to produce this, neither of which is a bug:

**1. The four score bands are balanced by count, but not by BM25 classification**, because the
Youden threshold (the score above which BM25 calls something "positive" — currently **107.6**,
from Step 11's bake-off) doesn't land on a quartile boundary. From the real Step 13 output
(`stratum_report.csv` / `stratified_candidate_pool.csv`, this run):

| Band | Real BM25 score range | Records | BM25 classification |
|---|---|---|---|
| Q1 | −10.0 to 31.2 | 582 | **100% negative** |
| Q2 | 31.2 to 72.1 | 597 | **100% negative** |
| Q3 | 72.2 to 150.5 | 587 | mixed — the 107.6 threshold falls inside this band |
| Q4 | 150.7 to 492.4 | 562 | **100% positive** |

**2. The default queue isn't browsed in random order.** `stratified_sample()` groups by
`(score_band, journal_bucket, year_bucket)` and pandas' `groupby` iterates ascending by default —
so the sampled rows are produced, and later appended into `canonical_dataset.csv`, in **Q1 → Q2 →
Q3 → Q4** order. The Curate app's default (unfiltered) queue order is just that dataset's row
order — nothing reshuffles it per session or per curator.

Put together: starting from the top of the queue, you hit **~1,179 records in a row (all of Q1 +
Q2) that are BM25-negative before reaching anything BM25-positive.** Measured directly against the
live queue: of the first 1,500 positions, **1,101 were tagged negative vs. 365 positive**. The
*last* 1,500 positions flip the other way — **807 positive vs. 693 negative**. It balances out over
the full queue; any partial pass through it from the start will look negative-skewed purely because
of where in the queue you currently are.

**If you want a more balanced mix right now** rather than working strictly top-down: open the
Filters expander → "Match-score band" → select Q3 and/or Q4. Same queue, same records, just
starting from the positive-leaning end instead of the negative-leaning one.

## Presentation order, by queue source

- **"Curation queue" (default)** — `canonical_dataset.csv`'s row order. For the current Step 13
  batch, that's Q1→Q2→Q3→Q4 as above. Selecting a score-band filter subsets to that band but keeps
  the same underlying sub-order (journal bucket, then year bucket) — it doesn't reshuffle.
- **"Full AI/ML bulk pool"** — a different mode entirely, bypasses stratification, always sorted
  **strictly by BM25 score descending** (highest first), with a raw score range you pick directly
  rather than a quartile.

## Snapshot vs. living truth

The table above is a snapshot of one specific Step 13 run (2,328 candidates, 2026-08-02). If
`sampling stratify` is re-run with different config or a refreshed bulk pool, the exact counts and
score ranges will change — re-derive them from `stratum_report.csv` and
`scoring_bakeoff_report.csv` rather than trusting these numbers as permanent. The *mechanism*
(quantile bands, ascending groupby order, Youden threshold independent of band boundaries) is
stable; the numbers aren't.

## Changelog

- 2026-08-03: file created, documenting the stratified-sampling mechanics and explaining the
  negative-heavy start of the queue (real numbers from the 2026-08-02 Step 13 run).
