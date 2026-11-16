# e3 model-type evidence

Written by `scripts/e3_term_evidence.py`. It counts the positives in the Zenodo corpus dump whose title or
abstract mentions each candidate. Each candidate is one model type plus its aliases. The rule a new seed term
must pass: **at least 100 positives**, not covered by an existing e2 term or alias, and a genuine
model or architecture (docs/plans/glm53_ebi_production.md, Phase 1).

- Source: `dome-observatory-corpus-2026-09-25.jsonl.gz`, sha256 `277d664b853a8d8996324404b02ba48fca2e3e7deba5bcc08121898936cb9834`
- Positives scanned: 367,348
- Matching: word-bounded. Short or ambiguous acronyms are case-sensitive. SAM, UNI, Evo and UCE are never
  matched alone, because each collides with common biology.

## New seed terms

| Term | Positives | Share | ≥100 | Family hint | Example record ids |
|---|---:|---:|:-:|---|---|
| protein language model | 1,231 | 0.34% | yes | deep learning + representation learning | `0004d0a6-163c-5823-a2b8-79f4c84de2e5`, `0025058c-abe7-596d-813e-02efc96b148a`, `005461c3-de61-5ca8-acc8-c60b147e75f8` |
| genomic language model | 220 | 0.06% | yes | deep learning + representation learning | `01f66426-766f-5555-a0b0-2b6866f0e2b7`, `064ce2eb-fe3c-54a1-95b6-46333974775a`, `08537dbe-a682-5a58-b477-589d5a0cbd8b` |
| single-cell foundation model | 69 | 0.02% | no | deep learning + representation learning | `007ea2c3-5f9e-5e76-bfd5-5a83b81d2f1f`, `0232c37a-ae2b-5619-ad89-91bcc653da2c`, `027deaae-62a3-5bd3-b4e7-875e1602a3c4` |
| chemical language model | 128 | 0.03% | yes | deep learning + representation learning | `01f66426-766f-5555-a0b0-2b6866f0e2b7`, `030b442d-8446-53c9-a582-b7668177a303`, `037387db-da5b-55e8-91ec-f189257d3ac0` |
| pathology foundation model | 122 | 0.03% | yes | deep learning + representation learning | `008c9475-c593-5b8a-9fb6-5ffd76916e70`, `02196d04-5ef6-52ca-8f0c-8a6def6d4fa0`, `04644b24-e5b0-5d96-b2da-18188251c9dd` |
| graph embedding | 738 | 0.20% | yes | representation learning | `0000075d-6438-5a9c-9100-fc2af9e0de15`, `005461c3-de61-5ca8-acc8-c60b147e75f8`, `0057a51f-be83-5e65-b1c0-b2c361284676` |
| AlphaFold | 1,407 | 0.38% | yes | deep learning | `00a38ef7-d7e2-56be-b3e8-27fdab2209d7`, `00aa16b4-f2a0-58cf-80de-ee89100e4906`, `012410d0-657c-5c56-8f4a-be98603aa8d6` |
| RoseTTAFold | 76 | 0.02% | no | deep learning | `02c41af9-398b-525c-a605-c89e079528f8`, `0868bbe5-53ba-5891-8f63-4cfe3226f450`, `0a1ac3e0-eda9-55ef-9678-5e0ba7b2fa8c` |
| Segment Anything Model | 337 | 0.09% | yes | deep learning | `00db8cc6-c856-56ef-88e7-6a9b77685f1b`, `00f4dfb7-6894-57ea-8cc4-a308f73ada5a`, `012120e0-addd-5fdf-875f-8548ad967402` |
| DenseNet | 2,388 | 0.65% | yes | deep learning | `0023dbf8-c1b1-570e-bf01-f314b2d8f8f3`, `00368f36-426a-5ef0-90cd-36a987fba052`, `00458cc1-534c-5bee-841b-a89801c8bc7b` |
| VGG | 3,139 | 0.85% | yes | deep learning | `001d2893-a2ba-5444-af28-9e7c897e0da5`, `00368f36-426a-5ef0-90cd-36a987fba052`, `00458cc1-534c-5bee-841b-a89801c8bc7b` |
| EfficientNet | 1,919 | 0.52% | yes | deep learning | `00226291-5bf7-5afb-99a0-d6ec0cfe4c05`, `002c28bb-0752-5c5c-a1dc-454065f8afc7`, `009e3702-eabe-5edc-9198-a95b3697091f` |
| MobileNet | 1,876 | 0.51% | yes | deep learning | `000b5704-b3b3-586f-8d06-4c0eba19a391`, `00226291-5bf7-5afb-99a0-d6ec0cfe4c05`, `00458cc1-534c-5bee-841b-a89801c8bc7b` |
| Inception | 2,194 | 0.60% | yes | deep learning | `000bdc50-2e87-5991-8bc1-6b5dc31c2ca4`, `0021d474-1200-501b-abec-701e4fab718c`, `00458cc1-534c-5bee-841b-a89801c8bc7b` |
| Faster R-CNN | 558 | 0.15% | yes | deep learning | `0012f0bd-b04e-5789-b533-ea3d8a3117e6`, `00514844-45ae-58d1-ab86-8122ada2b2a5`, `0056dad8-9cd5-54f4-b31d-17a77f94fa04` |
| Mask R-CNN | 527 | 0.14% | yes | deep learning | `0015a722-dc17-5ad7-b185-ea31f0f79438`, `00ec2605-7ac2-5c38-96ec-6560ac3ac8b4`, `0129a2ed-27f6-527a-9256-2603abb1da18` |
| Cox proportional hazards model | 6,266 | 1.71% | yes | classical machine learning | `00057b84-7fe5-5dab-9dd1-cb3383f1281c`, `00102667-d0a7-5c04-961f-1224093e5de9`, `0010d6ce-e634-5a93-87b5-1fbd471903a6` |
| random survival forest | 1,521 | 0.41% | yes | ensemble learning | `0025bef8-dc5f-5231-aa9c-a875393168a5`, `00321d4f-cd0d-5f09-bf0a-ad40b2421699`, `003c824f-8f01-5aff-b5e2-28be437cf0fc` |
| multiple instance learning | 616 | 0.17% | yes | deep learning | `00740ec8-ef2c-55fa-be0f-0621d76c797e`, `008c9475-c593-5b8a-9fb6-5ffd76916e70`, `00d6a255-1354-5694-b7cb-4f580e4182f5` |
| physics-informed neural network | 434 | 0.12% | yes | deep learning + physics-informed machine learning | `004c76c9-8571-5608-b451-adc9d06ba696`, `005a2273-b4d7-521c-b553-2bcbbb3af960`, `01e398d8-fe36-5ea8-8c07-7241d747054e` |
| Kolmogorov-Arnold network | 280 | 0.08% | yes | deep learning | `0101963e-d09e-58bc-b0d5-43c6f0bb96b5`, `01bf44f8-c357-5b98-a5ac-f1a55c16a327`, `03665a59-b985-5cde-8fc1-a5b68e6111d7` |
| ARIMA | 972 | 0.26% | yes | classical machine learning | `001488ba-d5ed-546d-8f22-f9b28827347b`, `002a7d85-4651-57af-b2da-93c72405951b`, `016e0eb1-15c6-562b-b010-648d5e0a81fe` |

## Aliases added to existing e2 terms

| Alias group | Positives |
|---|---:|
| ResNet <- ResNet-18/34/50/101/152 | 4,799 |
| U-Net <- nnU-Net, UNet++ | 911 |
| graph neural network <- GAT, GIN, MPNN, Chemprop | 1,037 |
| vision transformer <- Swin Transformer | 625 |
| long short-term memory <- BiLSTM | 2,067 |
| linear regression <- multiple linear regression | 1,365 |

## Per-pattern counts

One record can match several patterns, so these do not sum to the term totals.

- **protein language model**: `\bprotein language models?\b` 946; `\bESM-?(?:1b|1v|2|3|C)\b` 392; `\bESMFold\b` 55; `\bProtT5\b` 116; `\bProtBERT\b` 53; `\bProtTrans\b` 30; `\bAnkh\b` 11; `\bProGen2?\b` 12; `\bSaProt\b` 7
- **genomic language model**: `\b(?:DNA|RNA|genomic|nucleotide) language models?\b` 118; `\bDNABERT(?:-2)?\b` 81; `\bNucleotide Transformer\b` 23; `\bHyenaDNA\b` 13; `\bEvo ?2\b` 9; `\bCaduceus\b` 5; `\bRNA-FM\b` 16
- **single-cell foundation model**: `\bsingle[- ]cell foundation models?\b` 33; `\bscGPT\b` 21; `\bGeneformer\b` 23; `\bscBERT\b` 5; `\bscFoundation\b` 7; `\bUniversal Cell Embeddings?\b` 5
- **chemical language model**: `\b(?:chemical|molecular) language models?\b` 82; `\bChemBERTa\b` 40; `\bMolFormer\b` 8; `\bChemformer\b` 7
- **pathology foundation model**: `\bpathology foundation models?\b` 65; `\bfoundation models? for (?:computational )?pathology\b` 9; `\bCONCH\b` 23; `\bVirchow ?2?\b` 34; `\bProv-GigaPath\b` 12
- **graph embedding**: `\bgraph embeddings?\b` 461; `\bnetwork embeddings?\b` 216; `\bnode2vec\b` 95; `\bDeepWalk\b` 26; `\bknowledge graph embeddings?\b` 121
- **AlphaFold**: `\bAlphaFold ?[23]?\b` 1,377; `\bAF[23]\b` 129; `\bAlphaFold-Multimer\b` 62; `\bColabFold\b` 45
- **RoseTTAFold**: `\bRoseTTAFold\b` 76
- **Segment Anything Model**: `\bSegment Anything\b` 284; `\bMedSAM\b` 47; `\bSAM ?2\b` 63
- **DenseNet**: `\bDenseNet(?:-?\d+)?\b` 2,388
- **VGG**: `\bVGG(?:-?\d+)?(?:Net)?\b` 3,139
- **EfficientNet**: `\bEfficientNet(?:-?(?:B\d|V2))?\b` 1,919
- **MobileNet**: `\bMobileNet(?:-?V\d)?\b` 1,876
- **Inception**: `\bInception[- ]?v\d\b` 1,464; `\bInception-ResNet\b` 169; `\bGoogLeNet\b` 580; `\bInception (?:network|module|architecture)s?\b` 133
- **Faster R-CNN**: `\bFaster R-?CNN\b` 558
- **Mask R-CNN**: `\bMask R-?CNN\b` 527
- **Cox proportional hazards model**: `\bCox proportional[- ]hazards?\b` 2,084; `\bCox (?:regression|model)s?\b` 4,593; `\bCoxPH\b` 110
- **random survival forest**: `\brandom survival forests?\b` 1,381; `\bRSF\b` 856
- **multiple instance learning**: `\bmultiple[- ]instance learning\b` 586; `\bABMIL\b` 18; `\bCLAM\b` 63; `\bTransMIL\b` 11
- **physics-informed neural network**: `\bphysics[- ]informed neural networks?\b` 404; `\bPINNs?\b` 333
- **Kolmogorov-Arnold network**: `\bKolmogorov[- ]Arnold networks?\b` 234; `\bKANs?\b` 229
- **ARIMA**: `\bS?ARIMAX?\b` 915; `\bautoregressive integrated moving average\b` 424
- **ResNet <- ResNet-18/34/50/101/152**: `\bResNet-?(?:18|34|50|101|152)\b` 4,799
- **U-Net <- nnU-Net, UNet++**: `\bnnU-?Net\b` 680; `\bU-?Net\+\+` 237
- **graph neural network <- GAT, GIN, MPNN, Chemprop**: `\bgraph attention networks?\b` 609; `\bGAT\b` 359; `\bgraph isomorphism networks?\b` 68; `\bGIN\b` 56; `\bMPNNs?\b` 126; `\bmessage[- ]passing neural networks?\b` 188; `\bChemprop\b` 43
- **vision transformer <- Swin Transformer**: `\bSwin[- ]Transformer\b` 625
- **long short-term memory <- BiLSTM**: `\bBi-?LSTM\b` 1,906; `\bbidirectional LSTM\b` 313
- **linear regression <- multiple linear regression**: `\bmultiple linear regression\b` 1,365
