---
name: data-links
description: >
  Fetch and load each record's data links and Europe PMC identity / preprint server into moros:
  the metadata pass (epmc_id, epmc_source, preprint_server, the data-links summary), the Europe
  PMC annotations API and /datalinks fetches, the EBI Search fetches for positives (deposits,
  bio.tools, the DOME Registry), the merge, the v1.4.0 and v1.5.0 migrations and the `preprints` /
  `data_links` / `identifiers` field loads. Trigger on "data links", "fetch the data links", "EBI
  Search links", "DOME Registry links", "backfill the preprint servers", "fill epmc_id", "refresh
  data links", "link papers to their datasets", "roadmap item 1". Free; writes only through
  moros-write.
---

# data-links

Host Python from `moros_pipeline/scripts/`, on the VPN, with `moros_pipeline/.env`. Read
`moros_pipeline/README.md` ("Europe PMC data links and identity", "EBI Search data links") for the
routes and what was measured, `docs/data_links_sources.md` for every accepted and refused source,
and `docs/preprint.md` for the preprint rules. New records get all of this inside the
batch load (`moros-write` A); this skill is the retrospective pass and the periodic refresh.

## Throughput is the point

Europe PMC is free and these endpoints take real load. Never pace for politeness:

- defaults are 64 workers, short timeouts, retries only on 429/5xx;
- every first run is `--limit 3000`: read the calls/s and error rate it prints, then run the full
  pass with more workers (`--max-workers 128`, then higher) until the error rate moves;
- quote wall times only from a measured rate.

Measured on the full corpus (2026-09-14): the metadata pass was 3,079 calls in 372 s at 96
workers, and the annotations pass 48,668 calls in 186 s at 128 workers, both with zero failed
calls. Both have headroom; start there or higher.

## Preconditions

1. `python3 ../../schema/check_alignment.py` says authored 1.5.1 = published v1.5.1. If the
   release is not cut, stop: `schema-sync` first.
2. `python3 verify_corpus.py` — note the count and the schema_version histogram.

## Procedure

```bash
cd moros_pipeline/scripts
python3 ../../scripts/export_corpus_keys.py          # pid, ids, is_preprint, classification

# 1. Identity + preprint server + data-links summary (batched core search; preprints by DOI under SRC:PPR)
python3 fetch_epmc_metadata.py --limit 3000
python3 fetch_epmc_metadata.py

# 2. Text-mined accessions (8 ids per call), then the Scholix residual
python3 fetch_annotations.py --limit 3000
python3 fetch_annotations.py --max-workers 128
python3 fetch_datalinks.py --limit 3000              # if every call fails with 500: the endpoint is down; skip, see below
python3 fetch_datalinks.py

# 2b. EBI Search, positives only. The accept list is ebisearch_resources.py.
python3 fetch_ebisearch_domains.py                   # accepted domains under the 100,000-entry cap, dumped whole
python3 fetch_ebisearch_xrefs.py discover --limit 3000
python3 fetch_ebisearch_xrefs.py discover --max-workers 256
python3 fetch_ebisearch_xrefs.py discover --record-failures   # a re-run: record what still fails
python3 fetch_ebisearch_xrefs.py detail              # accepted domains over the cap, 100 PMIDs a call

# 3. Merge. Read the report: unmapped schemes and DOI prefixes go into datalinks_resources.py first.
python3 build_data_links.py --report-only
python3 build_data_links.py                          # -> ../output/pid_preprints.csv, pid_data_links.csv, pid_identifiers.csv
python3 compare_data_links.py --old <the previous pid_data_links.csv> --new ../output/pid_data_links.csv

# 4. Once per corpus: the migrations (dry run, then confirm), oldest first
python3 migrate_v1_4_0.py
python3 migrate_v1_4_0.py --confirm
python3 migrate_v1_5_0.py
python3 migrate_v1_5_0.py --confirm

# 5. The loads, through moros-write section C
python3 load_fields.py --mode preprints
python3 load_fields.py --mode preprints --limit 500 --confirm
python3 load_fields.py --mode preprints --confirm
python3 load_fields.py --mode data_links
python3 load_fields.py --mode data_links --limit 500 --confirm
python3 load_fields.py --mode data_links --confirm
python3 load_fields.py --mode identifiers
python3 load_fields.py --mode identifiers --limit 500 --confirm
python3 load_fields.py --mode identifiers --confirm
python3 verify_corpus.py
```

If `/datalinks` is down, build with `--datalinks-scope none`: records are completed from the
annotations API and the derived BioStudies entry, and a later `fetch_datalinks.py` +
`build_data_links.py` + `load_fields.py --mode data_links` re-dates and completes them.
`import_textmined_bulk.py` is the zero-call fallback if the annotations API throttles.

**Identifiers are repaired in the merge, and the merge needs doi.org.** Europe PMC's text-mined
strings carry the punctuation and words around them (`10.5281/zenodo.18675888.`); stored verbatim
they made 1,513 documents' links 404. `build_data_links.py` picks every identifier through
`link_identifiers.py`, confirms every data DOI at `doi.org/api/handles` (cached in
`../output/doi_handles.csv`; 16,627 lookups took 100 s over the corpus at 64 workers), drops what cannot be recovered, and withholds a
record whose DOIs could not be checked. Both loaders refuse a malformed link before connecting,
and `verify_corpus.py` fails on one. Never hand-edit identifiers in a staging file; fix
`link_identifiers.py` and rebuild.

Refresh: rerun steps 1–3 and 5 with `--max-age-days 180` on the Europe PMC fetches (data citations
accrue) and `--max-age-days 30` on `fetch_ebisearch_domains.py`. A dump refresh re-dates every
in-scope record it covers.

## Check before loading

- Spot-check preprints in `pid_preprints.csv` against `docs/preprint.md` §4: bioRxiv/medRxiv by
  DOI, Research Square for `10.21203`. When the API and the table disagree, the API wins.
- `--limit 500 --confirm`, then read five documents back: a preprint has `epmc_source: PPR` and
  `epmc_id: PPR...`; a MED record has `epmc_id == pmid`; a PDB paper has a `pdb` resource.
- The report's EBI Search line shows 0 withheld, and its rejected domains are only kinds
  `docs/data_links_sources.md` §3 refuses. A new domain name there is a decision, not a fix.
- `compare_data_links.py` shows only the documented changes: EBI Search additions, `E-GEOD` links
  moving to GEO, dbGaP versions dropped, `PXD` links moving to their partner host.
- Read back a DOME Registry paper (`identifiers.dome_registry` set, one `dome_registry` resource
  with route `ebisearch_domain`) and a bio.tools paper.
- Re-running a completed load changes zero documents.

## Report

Calls/s and error rate per route; records identified / missed; preprints with a server; records
resolved / unresolved / with resources; identifiers repaired and links dropped (by reason); records
waiting on doi.org; the top resources; EBI Search records in scope / with links / withheld and
DOME Registry ids written; run ids and rollback paths; the
`verify_corpus.py` notes. Then the post-load checklist from `moros-write` (restart
`observatory-ws`: the `data_resource` facet is boot-loaded).
