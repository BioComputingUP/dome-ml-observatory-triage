# cross_links/ — scaffold

Every document carries five reserved external-identifier fields, all null since the corpus was
built (`mongo_landscape_export/scripts/schema.py::_identifiers`):

| Field | Meant to hold | Candidate sources to evaluate |
|---|---|---|
| `identifiers.dome_registry` | the DOME Registry entry id for this paper, if one exists | **Filled from schema v1.5.0** by `moros_pipeline/scripts/build_data_links.py`, from the registry's EBI Search domain matched by PMID or PMCID, positives only ([`docs/data_links_sources.md`](../docs/data_links_sources.md) §7) |
| `identifiers.bioai_repo` | a code repository for the method (GitHub / GitLab / Bitbucket URL) | Europe PMC full-text XML URL mining (OA records only); Europe PMC **data links** (`/MED/{pmid}/datalinks`) for software categories; bio.tools (`biotoolsID`, publication links); Papers with Code / OpenAlex `has_repo` style signals |
| `identifiers.huggingface` | a Hugging Face model or dataset id | Hugging Face Hub API (`/api/models?search=`, model-card `arxiv:` / DOI tags), matched by DOI or arXiv id |
| `identifiers.kaggle` | a Kaggle dataset or competition slug | no public paper-to-Kaggle linkage API; text mining of full text for `kaggle.com/` URLs |
| `identifiers.zenodo` | a Zenodo record DOI (`10.5281/zenodo.*`) for code or data | Europe PMC data links; Zenodo REST search by related identifier (`relation:isSupplementTo` the paper DOI); full-text URL mining |

Two things landed next to this folder rather than in it. `identifiers.epmc_id` (Europe PMC's own
accession) and the whole `data_links` group (schema v1.4.0) are fetched by
`moros_pipeline/scripts/fetch_epmc_metadata.py`, `fetch_annotations.py` and `fetch_datalinks.py`
and merged by `build_data_links.py` (see the `data-links` skill). So the Europe PMC data-links
rows in the table above are no longer a fetch for this folder: `identifiers.zenodo` and
`identifiers.bioai_repo` should be derived from `pid_data_links.csv` (resources `zenodo`,
`github`, `software_heritage`), and any link that resolves through Europe PMC should use
`/article/{epmc_source}/{epmc_id}`.

Nothing in this folder runs yet; `identifiers.dome_registry` is filled by the data-links build
instead. `fetch_cross_links.py` is an argparse shell with the structure the real
script should keep; fill it in per source, one source per subcommand, each resumable and
streaming like `fetch_citations.py`.

## Rules for building it out

- **One write mode, already in place.** `moros_write.py::WRITE_MODES["identifiers"]` allows
  `identifiers.dome_registry`, `bioai_repo`, `huggingface`, `kaggle`, `zenodo` and
  `schema_version` only, and `load_fields.py --mode identifiers` maps a `pid, <field>` CSV onto
  it. A new field needs its column in that mapper, with tests, and nothing wider.
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
