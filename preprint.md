# Preprints: capturing the preprint server as real data

**Status:** the Observatory's frontend and backend read three new fields as of schema v1.3.0.
Nothing populates them yet. This document is the specification for the pipeline work that will,
written to be executed cold and then moved into the processing/data-management repo
(`dome-observatory-triage`, or whatever succeeds it).

**Why it exists.** 56,863 corpus records are preprints, and every one of them has
`publication_metadata.journal: null`. Europe PMC returns nothing in `journalTitle` for a `SRC:PPR`
record, and the harvester never read the field that *does* carry the venue. The result was cards
and record pages showing no venue at all, on 6.7% of the corpus.

The Observatory now shows `Preprint: bioRxiv` by inferring the server from the DOI prefix at
display time (`observatory-ui/src/app/core/venue.ts`). That is a stopgap and is labelled as one in
the code. This document describes replacing it with recorded data.

---

## 1. What to pull from Europe PMC

All three fields are present in `resultType=lite`. **No `core` fetch is needed**, which makes this
the cheapest of the three enrichment fetches the pipeline already runs.

| Europe PMC field | Example value | Target document path | Purpose |
|---|---|---|---|
| `bookOrReportDetails.publisher` | `bioRxiv` | `publication_metadata.preprint_server` | The server's own name, verbatim casing |
| `source` | `PPR` | `source.epmc_source` | Authoritative record-source marker: `MED`, `PPR`, `PMC`, `AGR`, `PAT` |
| `id` | `PPR18364` | `identifiers.epmc_id` | Europe PMC's own accession; the only way to build a correct link for a preprint |

A real response, fetched 2026-09-07 for `DOI:"10.1101/270413" AND SRC:PPR`:

```json
{
  "id": "PPR18364",
  "source": "PPR",
  "doi": "10.1101/270413",
  "bookOrReportDetails": { "publisher": "bioRxiv", "yearOfPublication": 2018 },
  "journalTitle": null
}
```

**Measured yield.** Over 40 randomly sampled corpus preprint DOIs, 38 of 38 completed lookups
returned a publisher; the other 2 were network timeouts, not misses. Expect effectively 100%
coverage, and treat any real miss as a record to investigate rather than a null to accept.

### The one trap that will silently corrupt this

`pick_best()` in `fetch_citations.py` ranks `SOURCE_PREFERENCE = ("MED", "PMC", "PPR", "AGR",
"PAT")` — MED first — because for a citation count the published version of record is the right
answer. **For this fetch that preference is exactly backwards.** A preprint later published in a
journal resolves by DOI to both a PPR and a MED record, and taking the MED one yields a null
publisher and an `epmc_source` of `MED` for a document the corpus holds as a preprint.

So this fetch must either restrict the query with `AND SRC:PPR`, or select the PPR record from the
results explicitly. Do not reuse `pick_best()` unaltered.

---

## 2. The backfill of the existing corpus

~56,863 documents. This is a small job: one cheap `lite` fetch keyed on identifiers the corpus
already holds, then a field-level `$set`.

**Selection.** `{"content_filters.pub_types": {"$in": ["Preprint", "preprint"]}}`. Two records use
the lowercase spelling; matching only `Preprint` misses them. `Preprint-withdrawal` (73) and
`Preprint-removal` (24) always appear alongside `Preprint`, so they need no separate clause.

**Keying.** `doi -> pmid -> pmcid`, the same three passes the licence and citation fetches use.
Every corpus preprint has a DOI (verified: zero without), so the DOI pass alone should reach all of
them; the other two are the safety net. Reuse `build_clause()`/`build_query()` from
`fetch_citations.py` — the quoting rules there are load-bearing and were learned from a real
incident (PMIDs must not be quoted, DOIs must be).

**Writing.** Use `load_fields.py`, not `mongoimport`. It already exists for exactly this, writes
allowlisted leaf paths only, and produces a rollback file. Concretely:

1. Add a `preprints` entry to `WRITE_MODES` in `moros_write.py`, allowlisting exactly these three
   paths and nothing else:
   ```
   publication_metadata.preprint_server
   source.epmc_source
   identifiers.epmc_id
   ```
2. Add a `preprint_row_to_update()` mapper in `load_fields.py` alongside
   `citation_row_to_update()`, reading `output/pid_preprints.csv` keyed on `pid`.
3. Run it the same way as the citations load:
   ```bash
   python3 load_fields.py --mode preprints                       # dry run
   python3 load_fields.py --mode preprints --limit 100 --confirm # real trial
   python3 load_fields.py --mode preprints --confirm             # full load
   ```

**Never reload the 3GB JSONL to populate three fields.** The citation load established this
precedent and the reasoning is unchanged.

**Idempotency.** A blank publisher must produce *no update* for that path rather than an explicit
null, exactly as `citation_row_to_update()` handles a blank count. Re-running the whole load after
a completed run must change zero documents.

**After the load.** `FacetsService` in `observatory-ws` is boot-loaded with no TTL, so **restart
the service** or the new field's facet values stay invisible. `StatsService`, `CountService` and
`JournalsService` are 24h TTL and catch up on their own.

---

## 3. Wiring it into the pipeline so new records arrive complete

Once this is done, no future refresh needs a backfill. The plumbing is already half-built:
`build_incoming_documents.py` captures `epmc_source` into the staging CSV, and `schema.py` drops
it on the floor.

**`moros_pipeline/scripts/build_incoming_documents.py`**
- Add `preprint_server` and `epmc_id` to `OUTPUT_COLUMNS`.
- In `epmc_record_to_row()`, add:
  ```python
  "preprint_server": (record.get("bookOrReportDetails") or {}).get("publisher") or "",
  "epmc_id": record.get("id") or "",
  ```
  `epmc_source` is already written there and needs no change.

**`mongo_landscape_export/scripts/schema.py`**
- `_publication_metadata()` — add `"preprint_server": _none_if_blank(row.get("preprint_server"))`.
  Use `.get()`, matching how `citation_count_updated` is read: a document built before this stage
  existed has legitimately never had the value looked up.
- `_source()` — add `"epmc_source": _none_if_blank(row.get("epmc_source"))`.
- `_identifiers()` — add `"epmc_id": _none_if_blank(row.get("epmc_id"))`.
- Bump `SCHEMA_VERSION` to `1.3.0` and add the changelog comment beside it, in the same style as
  the 1.1.0 and 1.2.0 entries.

Also update `EXPECTED_SCHEMA_VERSION` in `moros_pipeline/scripts/verify_corpus.py`.

---

## 4. Reconciliation: the verified server table

The Observatory's display-time fallback uses this table, and it doubles as the cross-check for the
backfill. Every row was verified against Europe PMC's own `bookOrReportDetails.publisher` on
**2026-09-07**, and these 27 prefixes account for **100%** of corpus preprints.

| DOI prefix | Preprints | Server |
|---|---|---|
| 10.21203 | 20,712 | Research Square |
| 10.1101 | 17,039 | bioRxiv 10,857 / medRxiv 6,182 |
| 10.20944 | 8,479 | Preprints.org |
| 10.64898 | 3,191 | bioRxiv 1,834 / medRxiv 1,357 |
| 10.31234 | 2,071 | PsyArXiv |
| 10.2139 | 1,876 | SSRN |
| 10.22541 | 1,640 | Authorea Preprints |
| 10.12688 | 598 | F1000Research and partner gateways |
| 10.26434 | 454 | ChemRxiv |
| 10.32388 | 232 | Qeios |
| 10.14293 | 196 | ScienceOpen Preprints |
| 10.7287 | 90 | PeerJ Preprints |
| 10.32942 | 78 | EcoEvoRxiv |
| 10.1590 | 60 | SciELO Preprints |
| 10.31222 | 41 | MetaArXiv |
| 10.3897 | 27 | ARPHA Preprints |
| 10.37044 | 16 | BioHackrXiv |
| 10.31220 | 14 | agriRxiv |
| 10.31730 | 14 | AfricArXiv |
| 10.1099 | 8 | Access Microbiology |
| 10.3310 | 7 | NIHR Open Research |
| 10.15694 | 7 | MedEdPublish |
| 10.3762 | 6 | Beilstein Archives |
| 10.21467 | 4 | AIJR Preprints |
| 10.35241 | 2 | Emerald Open Research |
| 10.5281 | 2 | Zenodo |
| 10.31233 | 1 | PaleorXiv |

**bioRxiv vs medRxiv share two prefixes** (`10.1101`, and `10.64898`, openRxiv's newer one). The
final numeric component separates them: 8 digits is medRxiv, 6 is bioRxiv.

```
medRxiv: 10.1101/19008045, 10.1101/2025.10.14.25337964, 10.64898/2026.05.11.26352943
bioRxiv: 10.1101/270413,   10.1101/2024.12.02.626299,   10.64898/2026.05.04.722036
regex:   ^(?:\d{4}\.\d{2}\.\d{2}\.)?\d{8}(?:v\d+)?$   ->  medRxiv, else bioRxiv
```

Validated on all 20,230 such DOIs in the corpus with zero unmatched, and spot-checked against
Europe PMC on both prefixes.

**`10.12688` is not one venue.** It splits by platform slug: `f1000research` (406),
`openreseurope` (117), `wellcomeopenres` (41), `mep` (13), `verixiv` (9), `openresafrica` (6),
`gatesopenres` (5), `hrbopenres` (1), `emeraldopenres` (1).

**When the API and the table disagree, the API wins** and the table row is the bug. The whole point
of recording the field is that inference stops being the answer.

**arXiv is absent.** Prefix `10.48550` appears zero times: Europe PMC's `SRC:PPR` does not index
arXiv. Anyone expecting arXiv coverage in this corpus should be told plainly that there is none.

---

## 5. Exports and downloads must carry these fields

`GET /api/export` streams whole documents, so the three fields ride along automatically the moment
they exist in Mongo. What does *not* update itself:

- **The download page's field documentation and data dictionary** must name the three new fields,
  or a downloader gets columns nothing explains.
- **The schema link on the download and About pages** is built from the version `/api/stats`
  reports, which reads `schema/CURRENT`. Cutting the release moves it; verify it resolves.
- **Any Zenodo archive cut after the backfill must be built from the current schema release.**

State the rule where the archive job can see it: **every export is a snapshot of the schema
version in `schema/CURRENT` at the moment it was cut, and the archive must record which version
that was.** An export whose schema version is not recorded cannot be interpreted later, and this
is precisely the kind of drift that a monthly automated archive introduces silently.

---

## 5b. The alignment check will report drift until this lands

`dome-observatory-triage/schema/check_alignment.py` compares the authored shape, this repo's
`schema/CURRENT` and the live `schema_version` on moros. Because the contract was published here
first, it currently reports, by design:

```
authored  SCHEMA_VERSION (schema.py):          1.2.0
published CURRENT (dome-ml-observatory):       1.3.0
DRIFT:
  - version drift: authored 1.2.0 vs published 1.3.0
  - fields published but not authored: identifiers.epmc_id,
    publication_metadata.preprint_server, source.epmc_source
```

That is this work item, restated by a tool. It clears when section 3 is done: the three fields
become authored, `SCHEMA_VERSION` becomes `1.3.0`, and the next load makes the live corpus agree.
Do not "fix" it by reverting the release here.

## 6. Traps, collected

- **`10.1101` is also a journal prefix.** Cold Spring Harbor Laboratory Press publishes Genome
  Research, Learning & Memory and the Cold Spring Harbor Perspectives titles on it. 588 corpus
  records carry that prefix *with* a real journal name. Never infer a preprint server for a record
  that has a journal — the Observatory's helper enforces this by checking journal first.
- **Three records carry both a journal string and a `Preprint` pub type**, and eight have the
  journal string `bioRxiv : the preprint server for biology`. The journal-wins rule handles both.
- **`journalTitle` is null on PPR records.** This is the original cause of the blank venue, and it
  is why adding a `journalTitle` fallback (as `build_incoming_documents.py` already does) does not
  fix preprints on its own.
- **349 journal-less records are not preprints**: 216 reviews, 93 dissertations, 18 study guides,
  22 with no publication type. They must keep showing no venue rather than being swept into the
  preprint path.
- **The Europe PMC outbound link is currently wrong for preprints.**
  `observatory-ui/src/app/core/outbound-links.ts` hardcodes `europepmc.org/article/MED/{pmid}`;
  a preprint's URL is `/article/PPR/{epmc_id}`. 3,157 preprints carry a PMID and get a wrong link
  today. `identifiers.epmc_id` is what fixes it.

---

## 7. What the Observatory already does, so you can check your work

- `observatory-ui/src/app/core/venue.ts` — prefers `publication_metadata.preprint_server` when
  populated and falls back to the DOI table. So a correct backfill changes nothing visible, and a
  wrong one shows up immediately as a changed card.
- `source.epmc_source === 'PPR'` is preferred over the `pub_types` proxy for deciding whether a
  record is a preprint at all, once populated.
- `observatory-ws` registers `preprint_server` as a facet field path, so
  `GET /api/facets/preprint_server` starts answering the moment data lands.
- Schema v1.3.0 defines all three fields as `["string","null"]`, nullable everywhere, so a partial
  backfill is a valid corpus state rather than a broken one.
