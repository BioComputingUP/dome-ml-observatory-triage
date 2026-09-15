# Release metadata: the corpus in DCAT and schema.org

Each monthly corpus release is described once, in standard vocabularies, and published with the
sibling repository's apps. This page covers what that description contains, where each fact comes
from, and how to publish a release.

## Why it is built here

At release time, only the write side knows what a release is:

- the verified counts, from `verify_corpus.py`;
- the schema version;
- the curation-criteria and vocabulary hashes the classification and enrichment ran under, from
  `prompts/PROMPT_HASHES.json`;
- the hash of the Europe PMC search query, from `config/search_space.yaml`;
- the pipeline commit.

The sibling only serves the file; it never computes any of these.

Per-record metadata works the other way round. `observatory-ws` projects each stored document on
request into schema.org JSON-LD, FAIR Signposting, Dublin Core over OAI-PMH and the sitemaps, and
nothing is stored twice. The sibling's `observatory-ws/README.md`, section "FAIR metadata", covers
it.

## What the file says

The file is `metadata/releases/<YYYY-MM>/dataset.jsonld` in dome-ml-observatory. It holds one
`@graph`. Every node is typed in both DCAT and schema.org, so DCAT harvesters and schema.org crawlers
read the same file.

| Node | DCAT / schema.org type | What it holds |
|---|---|---|
| `https://observatory.dome-ml.org/#catalog` | `dcat:Catalog` / `DataCatalog` (Bioschemas DataCatalog 0.3) | The Observatory, its publisher and licence, and a relation to the DOME Registry |
| `…/download/bulk#corpus` | `dcat:DatasetSeries` / `Dataset` | The corpus across all releases: creator, publisher, licence and rights, contact point. Every record's JSON-LD names this node in `isPartOf`. |
| `…/download/bulk#release-<YYYY-MM>` | `dcat:Dataset` / `Dataset` (Bioschemas Dataset 1.0) | This release: issue date, version, counts, record schema release, previous release, provenance |
| `…/api/export#ndjson` | `dcat:Distribution` / `DataDownload` | The whole-corpus NDJSON export through the API |
| `…/api` | `dcat:DataService` / `WebAPI` | The API, its OpenAPI description, and the dataset it serves |
| `https://biocomputingup.it` | `Organization` | The publisher, part of the University of Padua (ROR `00240q980`) |
| The ORCID of each `CITATION.cff` author | `Person` | The creators |

**Licence split.** CC BY 4.0 covers what the Observatory adds: verdicts, enrichment and data-link
annotations. `dct:rights` states that titles, abstracts and bibliographic metadata stay under Europe
PMC's terms and each article's own licence (see `LICENSE.md`). The per-record JSON-LD keeps the
same split by putting the record and the article in separate nodes.

**Provenance.** `prov:wasGeneratedBy` names the pipeline at its commit. It records three inputs:

- the curation criteria (`criteria_sha256`, prompt `v1`);
- the vocabularies (`vocab_sha256`, prompt `e1`);
- the search space (the query's sha256).

**Not in the file yet.** There is no Zenodo distribution and no DOI. The archive job in `ROADMAP.md`
will deposit a month and then add both. The DOI hardcoded on the sibling's `/download/bulk` is not
registered, so it never goes into this file.

## Publishing a release

Start once the month's loads are done, `verify_corpus.py` passes, and moros is at the authored
schema version.

```bash
cd moros_pipeline/scripts
python3 verify_corpus.py                        # the report the release is built from
python3 build_release_metadata.py               # dry run: prints the document
python3 build_release_metadata.py --write       # -> ../dome-ml-observatory/metadata/releases/<YYYY-MM>/ + CURRENT
```

The builder refuses to run in four cases:

- **A report with a failed invariant.**
- **A mixed-version corpus.** Some documents are not at the authored schema version.
- **A month that is already published.** `--overwrite` is only for a month not yet committed there.
- **A working tree with uncommitted changes.** The recorded commit would not be the code that ran.

`metadata/CURRENT` never moves back to an earlier month.

Then, in dome-ml-observatory:

1. Deploy.
2. Check that `/api/catalog` serves the file.
3. Commit `metadata/`.

Validate the published file with the Schema.org validator and a DCAT-AP SHACL validator. After the
next crawl, the corpus should appear in Google Dataset Search.
