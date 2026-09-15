# Data link sources

Status 2026-09-14, schema v1.5.0. This page lists every source that can put a link in a
document's `data_links` block, what each is accepted or refused for, and how one asset stays one
link when several sources name it. The accept list itself is code:
`moros_pipeline/scripts/ebisearch_resources.py`. This page and that module must agree.

A data link is an asset **from the paper**: data the authors deposited, the paper's software
record, its ML transparency report, or its supplementary files. A database that cites the paper as
evidence for a curated fact is not an asset from the paper and is refused.

## 1. Routes

| Route | `obtained_by` | What it finds | Matched by | Records |
|---|---|---|---|---|
| Europe PMC annotations API | `tm_accession`, `tm_supplementary` | accessions text-mined from the article and its supplementary files | Europe PMC id | all |
| Europe PMC `/datalinks` (Scholix) | `ext_links` | data citations and external links Europe PMC holds | Europe PMC id | residual; the endpoint answered HTTP 500 on 2026-09-14, so the corpus built without it |
| Derived | `derived` | the BioStudies `S-EPMC` entry holding a PMC article's supplementary files | PMCID | all with a PMCID |
| EBI Search cross-references | `ebisearch_xref` | entries in the large accepted domains that name the paper | PMID | positives |
| EBI Search domain dumps | `ebisearch_domain` | entries in the other accepted domains whose publication fields name the paper | PMID, PMCID or DOI | positives |

Every EBI Search link comes from an entry that names the paper by PMID, PMCID or DOI.
`links[].matched_by` records which one. No title matching, author matching or text similarity is
used.

EBI Search links go to **positives only**, in the corpus backfill and in every batch. A record's
classification is never changed by what a registry or repository says about it.

## 2. Accepted EBI Search sources

"Papers" is the evaluation count of positives that gained at least one link the corpus did not
already hold (2026-09-14, before canonicalisation). A zero means no positive matched yet; the
domain stays on the list so later batches pick it up.

### Registries

| EBI Search domain | Slug | Category | Entry page | Papers |
|---|---|---|---|---|
| `biotools` | `biotools` | Software Registries | `https://bio.tools/{id}` | 4,052 |
| `dome-registry` | `dome_registry` | Transparency Reports | `https://registry.dome-ml.org/review/{id}` | 1,190 |

### Deposits

| EBI Search domain(s) | Slug | Category | Papers |
|---|---|---|---|
| `project`, `earlycause-molecular-sequences` | `bioproject` | Genomes & Assemblies | 1,957 / 2 |
| `sra-study` | `ena` | Nucleotide Sequences | 1,244 |
| `wgs_masters` | `ena` | Nucleotide Sequences | 3 |
| `sra-analysis`, `sra-analysis-covid19`, `sra-analysis-mpox`, `tsa_masters`, `tls_masters`, `emblstandard`, `emblcon`, `embl-covid19`, `embl-pathogen` | `ena` | Nucleotide Sequences | 0 |
| `node` | `node` | Genomes & Assemblies | 8 |
| `geo` | `geo` | Gene Expression | 1,168 |
| `biostudies-arrayexpress` | `arrayexpress` | Gene Expression | 115 |
| `sc-experiments` | `expression_atlas` | Gene Expression | 0 |
| `pride` | `pride` | Proteomics | 151 |
| `iprox` | `iprox` | Proteomics | 103 |
| `jpost` | `jpost` | Proteomics | 12 |
| `panorama` | `panorama` | Proteomics | 11 |
| `massive`, `gnps` | `massive` | Proteomics | 34 / 5 |
| `metabolights`, `metabolights_dataset` | `metabolights` | Metabolomics | 27 |
| `pdbe` | `pdb` | Protein Structures | 137 |
| `emdb` | `emdb` | Protein Structures | 55 |
| `empiar` | `empiar` | Imaging | 19 |
| `bioimages` | `bioimage_archive` | Imaging | 1 |
| `ega`, `ega-bycovid` | `ega` | Genomic Variation | 18 / 17 |
| `dbgap` | `dbgap` | Genomic Variation | 29 |
| `eva_studies` | `eva` | Genomic Variation | 0 |
| `dgva` | `dgva` | Genomic Variation | 0 |
| `biomodels` | `biomodels` | Models | 42 |
| `fairdomhub`, `physiome`, `cellcollective` | `fairdomhub`, `physiome`, `cellcollective` | Models | 0 |
| `biostudies-other` | `biostudies` | Supplementary Material | 117 |

### Supplementary

| EBI Search domain | Slug | Category | Papers |
|---|---|---|---|
| `biostudies-literature` | `biostudies` | Supplementary Material | 4,526 |

`biostudies-literature` lists the same `S-EPMC` entries the derived route builds. Where both find
one, it is one link carrying both routes. The 4,526 are PMC articles the derived route could not
reach.

Entry URLs per slug are in `URL_TEMPLATES`, each checked live on 2026-09-14. DGVa has no per-entry
page, so its links carry no URL. FAIRDOMHub, Physiome and Cell Collective ids are not the path of
their page, so those links use the entry's own `full_dataset_link`.

## 3. Refused

| Source | Why |
|---|---|
| Curation and evidence domains: `pdbekb`, `uniprot`, `proteomes`, `chembl-document`, `chebi`, `rhea`, `gwas_catalog`, `g2p`, `omim`, `hgnc`, `rnacentral`, `rfam`, the `interpro7_*` domains, `go`, `efo`, `reactome`, `intact-interactions`, `complex-portal`, `paxdb`, `gpmdb`, `peptide_atlas`, `sc-genes`, `ensemblGenomes_gene` | The database cites the paper as evidence for a curated fact. The paper deposited nothing there. |
| `mesh` | Subject headings, not an asset. |
| `atlas-experiments` (Expression Atlas bulk experiments) | A curated re-analysis of an ArrayExpress or GEO deposit the paper already links. |
| `geo_datasets` (GEO DataSets) | An NCBI-curated collection over GEO series the paper already links. |
| Any other domain EBI Search returns | Not on the accept list. It is rejected at merge time and counted by name in the `build_data_links.py` report. |
| ScholeXplorer | Not introduced. A 3,000-positive sample gave 287 papers with 1,521 distinct targets, every one `IsRelatedTo`, with a long tail of third-party tools. |
| DataCite | Not introduced. The same sample gave 106 papers, mostly publisher supplement copies on figshare. |

## 4. One link per asset

1. **Canonical ids first.** Before any merge, every link from every route is filed under its home
   resource (`ID_RULES`):
   - ArrayExpress's `E-GEOD-n` becomes GEO `GSEn`;
   - a versioned dbGaP study `phsNNNNNN.vN.pN` becomes the unversioned study;
   - `BIOMD`/`MODEL` ids go to BioModels, `EMPIAR-` to EMPIAR, `S-BIAD` to the BioImage Archive;
   - `S-EPMC` goes to BioStudies, `MSV` to MassIVE, `MTBLS`/`MTBLC` to MetaboLights.
2. **One home per ProteomeXchange dataset.** A `PXD` id that an EBI Search partner domain lists
   is filed under that partner, in the order PRIDE, iProX, jPOST, Panorama. A text-mined `PXD` no
   partner lists stays under PRIDE.
3. **One key.** Links merge on `(resource, id)`, case-insensitive. Routes run in the order
   annotations, `/datalinks`, bulk, derived, EBI Search. The first route to find a link sets its
   `obtained_by`. A later route fills any empty `url`, `title`, `section`, `frequency`,
   `matched_by` or `source_domain`, and adds itself to the resource's `routes`.
4. **One entry, one link.** An EBI Search entry naming the paper by PMID and by DOI is still one
   link. `matched_by` keeps the first of PMID, PMCID, DOI.
5. **Caps.** At most 50 stored links per resource and 300 per document. `resources[].count` and
   `link_count` keep the true totals, and `truncated` is set when anything was left out. An EBI
   Search entry with more than 100 references at the source counts the rest without storing them.
6. **The gates.** Every id and URL passes `link_identifiers.problems` and `url_problems`, or is
   dropped and counted. Both loaders refuse a malformed link before connecting, and
   `verify_corpus.py` fails on one.

## 5. Provenance on the record (added in v1.5.0)

| Path | Meaning |
|---|---|
| `data_links.sources[]` | Includes `"ebisearch"` when the EBI Search route answered for the record, with links or without. |
| `data_links.resources[].routes` | Every `obtained_by` value among the resource's links, sorted. Both a Europe PMC route and an EBI Search route means the article and the repository agree. |
| `data_links.resources[].browse_url` | One page at the source listing every entry of this resource for the paper. Only GEO has one today (NCBI's PubMed-to-GEO link page). |
| `data_links.links[].matched_by` | `pmid`, `pmcid` or `doi`: the identifier the EBI Search entry named. Null for Europe PMC routes. |
| `data_links.links[].source_domain` | The EBI Search domain that asserted the link, such as `sra-study`. Null for Europe PMC routes. |
| `data_links.links[].relationship` | `IsSupplementedBy` for deposits and supplementary files, `IsDescribedBy` for bio.tools, `IsReviewedBy` for the DOME Registry. DataCite relation types, as the Scholix links already use. |
| `identifiers.dome_registry` | The DOME Registry entry id. See section 7. |

## 6. Answered or withheld

A record is written only when every route it needs has answered. Otherwise its data-links cell is
left empty and the record keeps what it had.

- A positive with a PMID is answered by EBI Search when its discovery record exists and every
  accepted cross-reference domain listed there has a complete detail record.
- A PMID that fails every retry gets an explicit failed record. It counts as answered with no
  cross-references, and a later run asks again.
- A positive without a PMID is answered by the domain dumps alone.
- Any dump missing from `output/ebisearch_domains/` stops the build.

## 7. The DOME Registry

- DOME Registry links go to positives only.
- `identifiers.dome_registry` holds the entry id, the first in sort order when several entries
  name the paper. The record page card shows them all.
- `""` means the paper has a PMID or PMCID, the only keys the registry's entries carry, and no
  entry names it.
- `null` means never looked up: negatives, and positives with only a DOI.
- In the evaluation, 1,200 entries matched 1,190 positives. 22 entries match papers classified
  negative and 57 match no paper in the corpus. Both groups are left alone.

## 8. Caveats

- EBI Search keys Europe PMC papers by PMID only. The 41,205 positives without a PMID are reached
  only through the dumped domains. DOI, PMCID and preprint-id queries against the large domains
  matched none of a 1,000-paper sample.
- No accepted domain carries a preprint (PPR) id field.
- A dump stops at 100,000 entries. Larger domains are asked per PMID, so a paper without a PMID
  cannot reach them.
- Repositories file publication ids in inconsistent fields, so every value is classified by its
  shape. EMDB prefixes DOIs with `doi:`, sometimes before a full DOI URL. NODE puts DOIs in its
  PubMed field, ArrayExpress puts PMIDs in its DOI field and PubMed URLs in its PubMed field, EVA
  prefixes `PubMed:`, BioModels wraps ids in `+`, and BioStudies stores DOI URLs. All of these are
  recovered. A truncated, concatenated or mistyped id, and placeholders such as `N/A` or
  `UNKNOWN_…`, are ignored and counted in the build report.
- Refreshing a dump re-dates `data_links.fetched_at` on every in-scope record it covers.
- The v1.5.0 rebuild changes some documents that EBI Search never touched: `E-GEOD` links become
  GEO series, dbGaP versions are dropped, and text-mined `PXD` links move to their partner host.
  `compare_data_links.py` measures these before a load.

## 9. The corpus build (2026-09-14)

Built for all 846,716 documents with EBI Search for the 366,234 positives; not yet loaded.

| Measure | Documents |
|---|---|
| Unresolved or withheld | 0 |
| Positives with an EBI Search link | 92,259 |
| Positives gaining a link no Europe PMC route had | 12,161 |
| `identifiers.dome_registry` filled | 1,190 |
| Positives looked up with no DOME Registry entry (`""`) | 327,481 |

Documents gaining a link per resource: BioStudies 4,600, bio.tools 4,054, BioProject 1,973, ENA
1,247, DOME Registry 1,190, GEO 1,183, PRIDE 155, PDB 137, iProX 106, EMDB 59, BioModels 42,
MassIVE 35, ArrayExpress 33, dbGaP 29, MetaboLights 27, EGA 24, EMPIAR 19, jPOST 12, Panorama 11,
NODE 8, BioImage Archive 1. Europe PMC and EBI Search found the same link in 84,494 documents for
BioStudies and 941 for GEO, which the dedupe keeps as one link with both routes.

Against the v1.4.0 build (`compare_data_links.py`), no document changed except as section 8 says:

| Change | Links |
|---|---|
| ArrayExpress `E-GEOD` moved to GEO | 169 |
| ProteomeXchange `PXD` moved to PRIDE | 3 |
| Merged into the same accession already held under its home resource | 7 |
| Removed otherwise | 1, an `E-GEOD` mirror in a capped document whose GEO series is counted but not stored |

## 10. Changing the list

Adding a domain takes one `DOMAINS` line in `ebisearch_resources.py`, a slug in
`datalinks_resources.py` if the resource is new, a verified URL template or an entry in
`NO_URL_TEMPLATE`, and the tests in `test_ebisearch_resources.py`. Then update this page, rebuild,
compare and reload. The runbook is the `data-links` skill.
