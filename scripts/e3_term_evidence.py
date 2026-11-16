"""Counts how many positives in the Zenodo corpus dump mention each candidate e3 model type, to apply
the e3 inclusion rule (docs/plans/glm53_ebi_production.md, Phase 1): a new seed term needs at least
100 positives that mention it, must not already be covered by an e2 canonical term or alias, and must
be a genuine model or architecture.

Matching is on title + abstract, word-bounded. Long phrases are case-insensitive; short or ambiguous
acronyms are case-sensitive, and acronyms that collide with common biology (SAM = S-adenosylmethionine,
UNI, Evo, UCE) are not matched on their own -- only their unambiguous forms are. A record counts once
per term however many of its patterns match. Output carries counts and record ids only, never text.

Runs in the `tools` container:
    docker compose run --rm tools python scripts/e3_term_evidence.py \\
        --dump ../zenodo_corpus/dome-observatory-corpus-2026-09-25.jsonl.gz \\
        --out docs/e3_model_type_evidence.md
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

I = re.IGNORECASE

# (term, kind, family_hint, [(pattern, flags)]). kind: "new" = a candidate seed entry (the >=100 rule
# applies); "alias" = an alias added to an existing e2 term (reported for the record; the plan adds these).
CANDIDATES: list[tuple[str, str, str, list[tuple[str, int]]]] = [
    ("protein language model", "new", "deep learning + representation learning", [
        (r"\bprotein language models?\b", I), (r"\bESM-?(?:1b|1v|2|3|C)\b", 0), (r"\bESMFold\b", 0),
        (r"\bProtT5\b", 0), (r"\bProtBERT\b", 0), (r"\bProtTrans\b", 0), (r"\bAnkh\b", 0),
        (r"\bProGen2?\b", 0), (r"\bSaProt\b", 0)]),
    ("genomic language model", "new", "deep learning + representation learning", [
        (r"\b(?:DNA|RNA|genomic|nucleotide) language models?\b", I), (r"\bDNABERT(?:-2)?\b", 0),
        (r"\bNucleotide Transformer\b", I), (r"\bHyenaDNA\b", 0), (r"\bEvo ?2\b", 0),
        (r"\bCaduceus\b", 0), (r"\bRNA-FM\b", 0)]),
    ("single-cell foundation model", "new", "deep learning + representation learning", [
        (r"\bsingle[- ]cell foundation models?\b", I), (r"\bscGPT\b", 0), (r"\bGeneformer\b", I),
        (r"\bscBERT\b", 0), (r"\bscFoundation\b", 0), (r"\bUniversal Cell Embeddings?\b", I)]),
    ("chemical language model", "new", "deep learning + representation learning", [
        (r"\b(?:chemical|molecular) language models?\b", I), (r"\bChemBERTa\b", 0), (r"\bMolFormer\b", 0),
        (r"\bChemformer\b", 0)]),
    ("pathology foundation model", "new", "deep learning + representation learning", [
        (r"\bpathology foundation models?\b", I), (r"\bfoundation models? for (?:computational )?pathology\b", I),
        (r"\bCONCH\b", 0), (r"\bVirchow ?2?\b", 0), (r"\bProv-GigaPath\b", 0)]),
    ("graph embedding", "new", "representation learning", [
        (r"\bgraph embeddings?\b", I), (r"\bnetwork embeddings?\b", I), (r"\bnode2vec\b", I),
        (r"\bDeepWalk\b", I), (r"\bknowledge graph embeddings?\b", I)]),
    ("AlphaFold", "new", "deep learning", [
        (r"\bAlphaFold ?[23]?\b", I), (r"\bAF[23]\b", 0), (r"\bAlphaFold-Multimer\b", I), (r"\bColabFold\b", I)]),
    ("RoseTTAFold", "new", "deep learning", [(r"\bRoseTTAFold\b", I)]),
    ("Segment Anything Model", "new", "deep learning", [
        (r"\bSegment Anything\b", I), (r"\bMedSAM\b", 0), (r"\bSAM ?2\b", 0)]),
    ("DenseNet", "new", "deep learning", [(r"\bDenseNet(?:-?\d+)?\b", I)]),
    ("VGG", "new", "deep learning", [(r"\bVGG(?:-?\d+)?(?:Net)?\b", 0)]),
    ("EfficientNet", "new", "deep learning", [(r"\bEfficientNet(?:-?(?:B\d|V2))?\b", I)]),
    ("MobileNet", "new", "deep learning", [(r"\bMobileNet(?:-?V\d)?\b", I)]),
    ("Inception", "new", "deep learning", [
        (r"\bInception[- ]?v\d\b", I), (r"\bInception-ResNet\b", I), (r"\bGoogLeNet\b", I),
        (r"\bInception (?:network|module|architecture)s?\b", I)]),
    ("Faster R-CNN", "new", "deep learning", [(r"\bFaster R-?CNN\b", I)]),
    ("Mask R-CNN", "new", "deep learning", [(r"\bMask R-?CNN\b", I)]),
    ("Cox proportional hazards model", "new", "classical machine learning", [
        (r"\bCox proportional[- ]hazards?\b", I), (r"\bCox (?:regression|model)s?\b", I), (r"\bCoxPH\b", 0)]),
    ("random survival forest", "new", "ensemble learning", [(r"\brandom survival forests?\b", I), (r"\bRSF\b", 0)]),
    ("multiple instance learning", "new", "deep learning", [
        (r"\bmultiple[- ]instance learning\b", I), (r"\bABMIL\b", 0), (r"\bCLAM\b", 0), (r"\bTransMIL\b", 0)]),
    ("physics-informed neural network", "new", "deep learning + physics-informed machine learning", [
        (r"\bphysics[- ]informed neural networks?\b", I), (r"\bPINNs?\b", 0)]),
    ("Kolmogorov-Arnold network", "new", "deep learning", [
        (r"\bKolmogorov[- ]Arnold networks?\b", I), (r"\bKANs?\b", 0)]),
    ("ARIMA", "new", "classical machine learning", [(r"\bS?ARIMAX?\b", 0), (r"\bautoregressive integrated moving average\b", I)]),
    # aliases on existing e2 terms
    ("ResNet <- ResNet-18/34/50/101/152", "alias", "", [(r"\bResNet-?(?:18|34|50|101|152)\b", I)]),
    ("U-Net <- nnU-Net, UNet++", "alias", "", [(r"\bnnU-?Net\b", I), (r"\bU-?Net\+\+", I)]),
    ("graph neural network <- GAT, GIN, MPNN, Chemprop", "alias", "", [
        (r"\bgraph attention networks?\b", I), (r"\bGAT\b", 0), (r"\bgraph isomorphism networks?\b", I),
        (r"\bGIN\b", 0), (r"\bMPNNs?\b", 0), (r"\bmessage[- ]passing neural networks?\b", I), (r"\bChemprop\b", I)]),
    ("vision transformer <- Swin Transformer", "alias", "", [(r"\bSwin[- ]Transformer\b", I)]),
    ("long short-term memory <- BiLSTM", "alias", "", [(r"\bBi-?LSTM\b", I), (r"\bbidirectional LSTM\b", I)]),
    ("linear regression <- multiple linear regression", "alias", "", [(r"\bmultiple linear regression\b", I)]),
]

COMPILED = [(term, kind, hint, [re.compile(p, f) for p, f in pats]) for term, kind, hint, pats in CANDIDATES]
THRESHOLD = 100
N_EXAMPLES = 5


def text_of(doc: dict) -> str:
    pm = doc.get("publication_metadata") or {}
    return f"{pm.get('title') or ''}\n{pm.get('abstract') or ''}"


def count(dump: Path) -> tuple[int, dict[str, int], dict[str, list[str]], dict[str, dict[str, int]]]:
    n_pos = 0
    hits: dict[str, int] = defaultdict(int)
    examples: dict[str, list[str]] = defaultdict(list)
    per_pattern: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with gzip.open(dump, "rt", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if (doc.get("llm_classification") or {}).get("classification") != "positive":
                continue
            n_pos += 1
            text = text_of(doc)
            for term, _, _, regexes in COMPILED:
                matched = False
                for rx in regexes:
                    if rx.search(text):
                        per_pattern[term][rx.pattern] += 1
                        matched = True
                if matched:
                    hits[term] += 1
                    if len(examples[term]) < N_EXAMPLES:
                        examples[term].append(doc["_id"])
    return n_pos, hits, examples, per_pattern


def render(dump: Path, sha256: str, n_pos: int, hits, examples, per_pattern) -> str:
    out = [
        "# e3 model-type evidence",
        "",
        "Written by `scripts/e3_term_evidence.py`. It counts the positives in the Zenodo corpus dump whose title or",
        "abstract mentions each candidate. Each candidate is one model type plus its aliases. The rule a new seed term",
        f"must pass: **at least {THRESHOLD} positives**, not covered by an existing e2 term or alias, and a genuine",
        "model or architecture (docs/plans/glm53_ebi_production.md, Phase 1).",
        "",
        f"- Source: `{dump.name}`, sha256 `{sha256}`",
        f"- Positives scanned: {n_pos:,}",
        "- Matching: word-bounded. Short or ambiguous acronyms are case-sensitive. SAM, UNI, Evo and UCE are never",
        "  matched alone, because each collides with common biology.",
        "",
        "## New seed terms",
        "",
        "| Term | Positives | Share | ≥100 | Family hint | Example record ids |",
        "|---|---:|---:|:-:|---|---|",
    ]
    for term, kind, hint, _ in COMPILED:
        if kind != "new":
            continue
        n = hits.get(term, 0)
        out.append(f"| {term} | {n:,} | {n / n_pos:.2%} | {'yes' if n >= THRESHOLD else 'no'} | {hint} | "
                   + ", ".join(f"`{i}`" for i in examples.get(term, [])[:3]) + " |")
    out += ["", "## Aliases added to existing e2 terms", "", "| Alias group | Positives |", "|---|---:|"]
    for term, kind, _, _ in COMPILED:
        if kind == "alias":
            out.append(f"| {term} | {hits.get(term, 0):,} |")
    out += ["", "## Per-pattern counts", "",
            "One record can match several patterns, so these do not sum to the term totals.", ""]
    for term, _, _, regexes in COMPILED:
        parts = [f"`{rx.pattern}` {per_pattern[term].get(rx.pattern, 0):,}" for rx in regexes]
        out.append(f"- **{term}**: " + "; ".join(parts))
    return "\n".join(out) + "\n"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n_pos, hits, examples, per_pattern = count(args.dump)
    args.out.write_text(render(args.dump, file_sha256(args.dump), n_pos, hits, examples, per_pattern), encoding="utf-8")
    print(f"{n_pos:,} positives scanned -> {args.out}")


if __name__ == "__main__":
    main()
