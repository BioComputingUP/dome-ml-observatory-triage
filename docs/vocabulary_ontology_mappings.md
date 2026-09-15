# Vocabulary ontology mappings

Status 2026-09-15, schema v1.5.1. Every term in `curation_criteria/modelling_branch_vocab.json`
and `curation_criteria/model_type_seed_vocab.json` carries `ontology_mappings`: the ontology terms
it matches. The domain vocabulary is EDAM already and is unchanged. The files are published
verbatim in `dome-ml-observatory/schema/releases/v1.5.1/vocab/`. The key is not rendered into the
enrichment prompt, so `vocab_sha256` is unchanged.

## Shape

`ontology_mappings` is the last key of each term: at most one entry per ontology, ordered mesh,
aio, ncit, obi, swo, stato, edam, and `[]` when nothing was accepted. An existing `mesh_id` stays
and is repeated in the list.

| Field | Meaning |
|---|---|
| `ontology` | OLS ontology name |
| `id`, `iri` | CURIE and IRI, e.g. `mesh:D060388`, `AIO:RandomForest`, `NCIT:C78542` |
| `label` | the ontology's own label |
| `predicate` | `skos:exactMatch` (label agreement) or `skos:closeMatch` (alias agreement) |
| `matched_on`, `match_string` | which vocabulary string agreed (`label`, `canonical`, `mesh_label`, `alias`) and the string |
| `source` | services that returned or confirmed the id: `ols4`, `zooma`, `nlm`, joined by `+` |
| `checked` | date the evidence was fetched |

## Sources

| Service | Used for |
|---|---|
| EBI OLS4 `api/search`, exact on label and synonym | candidates |
| EBI OLS4 `terms/{iri}/ancestors` | the homonym guard |
| EBI Zooma `annotate`, HIGH confidence only | corroboration |
| NLM `id.nlm.nih.gov/mesh/lookup/descriptor`, exact | confirming every MeSH id; OLS carries MeSH 2025, NLM 2026 |

Ontology versions in OLS4: MeSH 2025, AIO 2026-07-22, NCIT 26.02d, OBI 2026-07-27, SWO 2023-03-05,
STATO 2026-04-20, EDAM 1.25-20260626T1230Z.

## How a match was accepted

- **Queries:** the label or canonical, MeSH label and aliases, with hyphen, spelling and plural
  variants, `AI`/`ML` expanded, and "X learning" for learning paradigms. No acronyms were invented.
- **Homonym guard:** a MeSH hit had to be a descriptor. MeSH and NCIT hits needed an ancestor about
  machine learning, AI, algorithms, statistics or computing. OBI needed data transformation,
  algorithm or planned process; SWO algorithm or software; EDAM an operation or topic.
- **exactMatch:** the ontology's label agrees with the term's label, canonical or MeSH label,
  allowing plural, comma inversion ("Neural Networks, Computer"), an end qualifier and spacing.
- **closeMatch:** the label agrees with an alias that names the term itself: it adds only generic
  words ("LASSO regression" for lasso), differs only in spacing, or is the acronym or expansion.
- **Never adopted automatically:** acronyms of five characters or fewer, MeSH ids NLM did not
  confirm or that differ from an existing `mesh_id`, hits reached only through a synonym or only
  through Zooma, and ties within one ontology.
- **Precision check:** all 15 `mesh_id`s the vocabularies already carried were reproduced.

## Coverage

| Vocabulary | Terms | With a mapping | With MeSH, before → after |
|---|---|---|---|
| learning_paradigm and model_family | 17 | 13 | 13 → 13 |
| model_type seed | 76 | 31 | 2 → 16 |

87 mappings: 82 `exactMatch` and 5 `closeMatch`. By ontology: MeSH 29, AIO 32, NCIT 10, OBI 9,
SWO 3, STATO 1, EDAM 3. The four project extensions (semi-supervised, classical machine learning,
AI agent, agentic system) have no term in any of these ontologies.

Not adopted: 39 lower-confidence candidates, mostly broader MeSH headings (XGBoost to "Boosting
Machine Learning Algorithms") and aliases naming another method (word2vec under word embedding),
and 77 homonyms stopped by the guard. The one-off matching code was not kept.

## Changing a mapping

1. Edit the entry in `curation_criteria/`, keeping the fields above and updating `checked`.
2. `python3 prompts/render_prompts.py --check` should fail on `PROMPT_HASHES.json` only; re-render.
3. Patch-bump `SCHEMA_VERSION`, restamp moros with a migration like `migrate_v1_5_1.py`, and cut
   the release with the observatory's `schema-version` skill.
