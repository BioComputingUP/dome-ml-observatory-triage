---
name: processing-log
description: >
  Log a finished processing round on the observatory's Processing history page
  (dome-ml-observatory, /about/processing): read the round's own figures off moros with
  round_summary.py, append one entry to processing-rounds.ts in the page's style, run the UI's
  tests, lint and production build, and commit on main there. Trigger at the close of
  refresh-cycle, or on "log the round", "update the processing history page", "add the last batch
  to the site". Free; read-only against moros; never deploys unless the user asks.
---

# processing-log

The page lives in the sister repository: `dome-ml-observatory/observatory-ui/src/app/about/about-processing/`.
Its cards are rendered from `processing-rounds.ts` (`CLASSIFICATION_ROUNDS`, `ENRICHMENT_ROUNDS`).
The stat pills (last classified, last enriched) and the enrichment coverage bar are live from
`/api/stats` and need no edit. **A card states what its own round did, never the corpus total**:
binding a card to the live total counts every later round into it.

## 1. The round's figures, from moros

From `moros_pipeline/scripts/`, read-only:

```bash
python3 round_summary.py --classification-batch <classify batch id> --json
python3 round_summary.py --enrichment-batch <enrich batch id> [--enrichment-batch <id> ...] --json
```

A round that ran as several batches takes `--classification-prefix` / `--enrichment-prefix`. Batch
ids are the `batch_id` column of the run's events file, `verify_corpus.py`'s `batch_ids`, or the
load report. Check before writing anything: the classification rounds already on the page, plus this
one, less the documents any `CORRECTIONS` entry of kind `removed` took out, add up to `corpus_total`.
If they do not, stop and say why.

The window is the one the run fetched (the ledger, `../output/coverage_ledger.json`, and the
`--up-to` used). The model is the one that actually answered: `pricing/pricing.yaml`'s version for
the model, confirmed by a one-call probe's `model` field whenever the provider has rerouted an id
since the last round (on 2026-09-15 `deepseek-v4-flash` answered as `deepseek-flash`, V4.1 Flash).

## 2. Append the entry

A classification round (next `number`; `started` / `finished` are the first and last
classification timestamps as `YYYY-MM-DD`, UTC):

```ts
  {
    number: 3,
    title: 'Incremental update',
    started: '2026-09-15',
    finished: '2026-09-15',
    searchSpace: CORE_SEARCH_SPACE,
    window: 'First published from 1 January to 10 September 2026 and not yet in the corpus.',
    processed: '2,345 publications new to the corpus, fetched from Europe PMC on 15 September 2026.',
    model: 'DeepSeek V4.1 Flash',
    promptVersion: 'v1',
    outcome: { positive: 1_234, negative: 1_100, undeterminable: 11 },
  },
```

An enrichment round:

```ts
  {
    number: 2,
    title: 'Positives from classification round 3',
    started: '2026-09-15',
    finished: '2026-09-15',
    cohort: '200 of the positive records added in classification round 3.',
    model: 'DeepSeek V4.1 Flash',
    promptVersion: 'e1',
    records: 200,
  },
```

Write it the way the rest of the About pages are written:

- British spelling; plain declarative sentences that end with a full stop. Thousands separators in
  prose (`13,476`); `_` separators in the numeric fields (`13_476`).
- Say what was done, not how well: no adjectives and no claims about quality.
- Records are LLM-classified by a method validated against a hand-annotated expert benchmark. Never
  describe them as curator-reviewed; only the benchmark's own records carry curator labels.
- Name the model as it ran ("DeepSeek V4.1 Flash"). When a provider routes an old id to a new model,
  the new model is the one named.
- Keep `searchSpace: CORE_SEARCH_SPACE` unless `moros_pipeline/config/search_space.yaml` changed;
  then write the new query in words and say it is a new search space.
- Figures only from step 1: never an estimate, never "approximately".

A change to documents already in the corpus goes in `CORRECTIONS` instead, in the same file
(`number`, `kind`, `title`, `date`, `documents`, `why`): the round cards keep saying what their round
did, and the page's Corrections section says what changed since. `kind: 'removed'` for documents taken
out (`resolve_duplicates.py`'s report gives the figures); `kind: 'corrected'` for a value fixed in
place, such as the 2026-09-25 full-text refresh (the load report's `modified`). Only removals come
off the round totals in step 1's check.

Leave the rest of the page alone unless something on it has become untrue (the enrichment
section's intro switches by itself once `ENRICHMENT_ROUNDS` has an entry). A copy change is its own,
named commit.

## 3. Check, commit, stop

```bash
cd ../dome-ml-observatory/observatory-ui        # from this repository's root: ../dome-ml-observatory
npx ng test --watch=false                       # the whole suite: --include breaks the builder's
npm run lint                                    # .scss/.html loaders ("No loader is configured")
npm run build-prod
```

`about-processing.spec.ts` fails if a round is misnumbered, out of order or undated, or if a card
shows the live corpus total. Then commit on `main` in that repository, the round entry only, with a
plain imperative message ("Log processing round 3 on the processing history page"). `build-prod`
runs `sync-schema.js` first; if that changes tracked files, they are not part of this commit.

**Never deploy.** `npm run deploy-prod-quick` publishes straight to production with no staging;
run it only when the user asks in this conversation. Report the commit and that it is not live yet.
