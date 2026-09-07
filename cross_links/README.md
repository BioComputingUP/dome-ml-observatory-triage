# cross_links/ — scaffold

Every document carries five reserved external-identifier fields, all null since the corpus was
built (`mongo_landscape_export/scripts/schema.py::_identifiers`):

| Field | Meant to hold | Candidate sources to evaluate |
|---|---|---|
| `identifiers.dome_registry` | the DOME Registry entry id for this paper, if one exists | DOME Registry API (registry.dome-ml.org), matched by pmid / doi |
| `identifiers.bioai_repo` | a code repository for the method (GitHub / GitLab / Bitbucket URL) | Europe PMC full-text XML URL mining (OA records only); Europe PMC **data links** (`/MED/{pmid}/datalinks`) for software categories; bio.tools (`biotoolsID`, publication links); Papers with Code / OpenAlex `has_repo` style signals |
| `identifiers.huggingface` | a Hugging Face model or dataset id | Hugging Face Hub API (`/api/models?search=`, model-card `arxiv:` / DOI tags), matched by DOI or arXiv id |
| `identifiers.kaggle` | a Kaggle dataset or competition slug | no public paper-to-Kaggle linkage API; text mining of full text for `kaggle.com/` URLs |
| `identifiers.zenodo` | a Zenodo record DOI (`10.5281/zenodo.*`) for code or data | Europe PMC data links; Zenodo REST search by related identifier (`relation:isSupplementTo` the paper DOI); full-text URL mining |

A sixth identifier field is coming: `identifiers.epmc_id` (Europe PMC's own accession, e.g.
`PPR18364`), specified in [`../docs/preprint.md`](../docs/preprint.md) for schema v1.3.0. It is
captured directly from the search response rather than looked up, so it is not this folder's
work — but it is what makes a correct `/article/{source}/{id}` link, and any cross-link that
resolves through Europe PMC should prefer it over the pmid form.

Also worth evaluating as generic sources: Europe PMC **data links** (all categories: data
citations, accession numbers, software), **Europe PMC text-mined annotations**, **OpenAlex** (which
returns related works and some repository signals), **Software Heritage** (archived repository
URLs), **DataCite** event data (citations from datasets/software to the paper DOI).

Nothing here runs yet. `fetch_cross_links.py` is an argparse shell with the structure the real
script should keep; fill it in per source, one source per subcommand, each resumable and
streaming like `fetch_citations.py`.

## Rules for building it out

- **Read-only until a write mode exists.** `moros_write.py::WRITE_MODES` has no `identifiers` mode.
  Add one deliberately (`identifiers.dome_registry`, `bioai_repo`, `huggingface`, `kaggle`,
  `zenodo` and `schema_version` only), with tests, before any load. The writer then structurally
  cannot reach anything else.
- **`null` vs `""` keeps its meaning**: `null` is "never looked up"; `""` is "looked up, nothing
  found". Write the empty string for a confirmed miss or every future pass re-fetches it.
- **Key by `pmid -> doi -> pmcid`** like the citation and licence fetches, and record which key and
  which source produced each link, so a later pass can re-verify it.
- **Full-text mining only where `source.access.open_access` is true**, and only from Europe PMC's
  OA full-text XML; never store full text.
- **One value or a list?** The schema fields are scalars. If a paper has several repositories,
  decide (and version in `schema.py`) before writing, not in the fetcher.
- Stage to `cross_links/output/<source>_links.csv` (gitignored), join to `_id` with the same
  pattern as `join_citations.py`, dry-run, `--limit` trial, then confirm.
