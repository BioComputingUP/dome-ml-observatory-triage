---
name: triage-fetch
description: >
  Run the triage: fetch newly published AI/ML records from Europe PMC for the configured search
  space and reduce them to the records moros has never seen, then report the counts and ask
  whether to classify. Trigger on "run the triage", "fetch new papers", "what is new since the
  last refresh", "start the monthly refresh". Free (Europe PMC costs nothing), read-only against
  moros. Never classifies, loads, or edits the ledger by itself.
---

# triage-fetch

Everything runs on the host from `moros_pipeline/scripts/`. Needs `moros_pipeline/.env` and the
lab VPN (moros is an internal host).

## Before starting

1. `python3 verify_corpus.py` — read-only. Note the document count; it is the baseline for
   `--expect-count` after the load. If the connection fails, stop: the run needs the cross-check
   against moros and must not be forced past it.
2. `python3 ../../schema/check_alignment.py` — the authored shape must match what is published and
   live, or the user must explicitly accept the drift it names before any load.
3. `python3 fetch_search_space.py --show-query` — the exact query and its sha256. It must match
   the `query_sha256` in `../output/coverage_ledger.json`; a different hash means the search space
   was edited and coverage starts from nothing for the new query. Say so and confirm before
   fetching anything.

## Procedure

```bash
cd moros_pipeline/scripts
python3 fetch_search_space.py --dry-run
```

Report which windows it will fetch: normally only the current year (always re-fetched, it is
still filling up) plus any year the ledger has never covered. **If it stops because moros holds
documents in years the ledger never recorded, stop and report.** Never pass
`--ignore-cross-check` without the user deciding to.

```bash
python3 fetch_search_space.py --up-to today 2>&1 | tee ../output/fetch_run.log
```

Resumable: each year is checkpointed with a `.done` marker under `../output/incoming/<hash>/`.
If interrupted, re-run the same command.

```bash
python3 build_incoming_documents.py --incoming ../output/incoming/<query-hash>
```

Output: `../output/incoming_new.csv` (`pid, pmid, pmcid, doi, title, abstract, journal, year,
...`). The `pid` is the deterministic UUID5 and becomes the document `_id`. Each row also carries
the record's Europe PMC identity (`epmc_source`, `epmc_id`), its `preprint_server`, and the
data-links summary (`has_data`, tags, accession types, cross-references, `has_tm_accessions`,
`has_db_xrefs`, `has_suppl`), read off the same search response, so the batch load can fetch its
data links without another metadata pass.

## Report, then stop

Give the user:

- windows fetched and records returned;
- how many were already in the corpus vs genuinely new;
- how many new records have no abstract (they are staged but never classified; the builder leaves
  them out and counts them);
- the classification projection for the new records: run the `cost-estimate` skill (live price,
  balance, off-peak status). At the measured rate it is about $0.0003 per record off-peak.

Then ask: **"Classify these N records for about $X now (off-peak: yes/no)?"** Hand over to the
`classify` skill only on a yes. Do not start it yourself.

## Do not

- Edit `config/search_space.yaml` or the coverage ledger by hand mid-run.
- Delete anything under `../output/incoming/`.
- Merge or reconcile the ledger and moros when they disagree.
