# Prompt e3: provenance

**Status: production enrichment prompt `e3`, adopted 2026-11-16.** It runs on GLM-5.3-Flash on the EMBL-EBI LLM gateway. e3 is prompt e2 ([`../e2/PROVENANCE.md`](../e2/PROVENANCE.md)) plus the life-science model types that e2's seed vocabulary lacked, **and nothing else**.

- The code is `src/dome_triage/llm_classify/enrichment_e3.py`, which reuses e2's builder, parser and loop.
- The rendered prompt is `prompts/enrichment_system_message.e3.txt`, sha256 `3b9d119deb05834f404e541493ba01e221553ad59f23031e8a004c88bccd0239`. It is pinned as `EXPECTED_E3_SHA256` and checked by `render_prompts.py --check`.
- Prompts e1 (`41db952f…`) and e2 (`c3196ff7…`) and their files are unchanged.

| | |
|---|---|
| Base | Prompt e2. `modelling_branch_vocab.json` here is e2's byte for byte, and the domain vocabulary is `../domain_vocab.json`, shared |
| Changed file | `model_type_seed_vocab.json` only: 20 new seed entries, and aliases added to 6 existing entries |
| Rule (Gavin, 2026-11-16) | A new seed term needs **at least 100 positives** in the corpus that mention it. It must not be covered by an existing canonical term or alias. It must be a genuine model or architecture: not a tool, a score or a generic phrase |
| Evidence | [`docs/e3_model_type_evidence.md`](../../docs/e3_model_type_evidence.md), from `scripts/e3_term_evidence.py` over the 367,348 positives in Zenodo dump 2026-09-25 (record 22967974) |
| Candidates from | The 447 off-seed model types that GLM-5.3 produced across 850 e2 records, plus Gavin's list (`docs/plans/glm53_ebi_production.md`, Phase 1) |
| Prompt diff vs e2 | Seed lines only: 20 lines added and 6 lines extended with aliases. Every other block is byte-identical (checked by `tests/test_enrichment_e3.py`) |

## New seed terms

| Term | Positives | Family hint |
|---|---:|---|
| protein language model | 1,231 | deep learning + representation learning |
| genomic language model | 220 | deep learning + representation learning |
| chemical language model | 128 | deep learning + representation learning |
| pathology foundation model | 122 | deep learning + representation learning |
| graph embedding | 738 | representation learning |
| AlphaFold | 1,407 | deep learning |
| Segment Anything Model | 337 | deep learning |
| DenseNet | 2,388 | deep learning |
| VGG | 3,139 | deep learning |
| EfficientNet | 1,919 | deep learning |
| MobileNet | 1,876 | deep learning |
| Inception | 2,194 | deep learning |
| Faster R-CNN | 558 | deep learning |
| Mask R-CNN | 527 | deep learning |
| Cox proportional hazards model | 6,266 | classical machine learning |
| random survival forest | 1,521 | classical machine learning + ensemble learning (as random forest) |
| multiple instance learning | 616 | deep learning |
| physics-informed neural network | 434 | deep learning + physics-informed machine learning |
| Kolmogorov-Arnold network | 280 | deep learning |
| ARIMA | 972 | classical machine learning |

Each term also passes on its unambiguous long form alone, not only on acronyms; the evidence file gives per-pattern counts. Each new entry carries `"e3": {"added": …, "evidence": …}`.

**Excluded, below 100:** single-cell foundation model (69) and RoseTTAFold (76). The model still tags them verbatim, as for any type not in the seed list.

**Not added, as the plan set out:**
- new families or paradigms ("foundation model" would overlap large language model, transfer learning and representation learning);
- named LLMs, which stay verbatim under e2's rule;
- optimisers, tools such as CIBERSORT and Boruta, and generic phrases.

## Aliases added to existing entries

| Entry | Aliases added | Positives |
|---|---|---:|
| ResNet | ResNet-18, -34, -50, -101, -152 | 4,799 |
| U-Net | nnU-Net, UNet++ | 911 |
| graph neural network | graph attention network, GAT, graph isomorphism network, GIN, message passing neural network, MPNN, Chemprop | 1,037 |
| vision transformer | Swin Transformer | 625 |
| long short-term memory | BiLSTM, bidirectional LSTM | 2,067 |
| linear regression | multiple linear regression | 1,365 |

Each changed entry carries `"e3": {"aliases_added": […]}`.

## Ontology mappings

These follow [`docs/vocabulary_ontology_mappings.md`](../../docs/vocabulary_ontology_mappings.md), applied with `scripts/e3_ontology_mappings.py`: EBI OLS4 exact search, the ancestor homonym guard, NLM confirmation of MeSH ids, and Zooma corroboration. Checked 2026-11-16.

| Term | Mapping | Predicate | Source |
|---|---|---|---|
| multiple instance learning | `mesh:D000098451` Multiple-Instance Learning Algorithms | exactMatch (end qualifier) | ols4 + nlm; ancestors Machine Learning, Supervised Machine Learning |
| ARIMA | `AIO:AutoregressiveIntegratedMovingAverage` | closeMatch (the alias is the expansion) | ols4 + zooma |
| the other 18 | `[]`, because no ontology term agrees with the label | | |

One hit was rejected by hand: `AIO:node2vec` for graph embedding. node2vec is one graph-embedding method, not the term itself, so the closeMatch rule does not hold.

Named architectures such as AlphaFold, VGG and DenseNet have no entries in MeSH, AIO, NCIT, OBI, SWO, STATO or EDAM.

## Validation

e3 is not a new tagging method. The decision order, the rules, the examples, the output format and every e2 term are unchanged. So e2's validation carries over ([`../e2/PROVENANCE.md`](../e2/PROVENANCE.md)): on heldout250, all three modelling fields are right in 87% of papers against 48% for e1, and the Opus judge passes every primary unit on wave2. The human rating is pending.

Before production, e3 on the EBI gateway is compared field by field with the Z.ai e2 run on the same 250 wave2 papers. That is the check that the gateway serves the same model, and it is reported in `docs/enrichment_ebi.md`.
