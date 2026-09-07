"""Relevance-matching algorithms for scoring candidate papers against the curated keyword
lexicon. Multiple implementations, empirically compared (scoring_bakeoff.py) rather than one
hardcoded method -- this is a genuine open methodological question, trialled the same way the
later LLM backend choice is.

All three score a whole corpus against the lexicon at once (`score_corpus`), not one document at
a time -- BM25 and TF-IDF cosine both need the full corpus to compute their term statistics
(IDF), so a per-document API would force rebuilding those statistics on every call.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Protocol

import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from dome_triage.keywords.preprocess import clean_text, parallel_map

ScoredResult = tuple[float, list[str]]

# Chunk size for the corpus-cleaning stage of bm25/tfidf-cosine: ~5% per chunk (20 chunks) on a
# large corpus, so a checkpoint file gets written roughly every 5% -- see `_write_checkpoint`. On
# a small corpus (e.g. tests) this collapses to one chunk per document, never an empty chunk.
_CHECKPOINT_CHUNKS = 20


class MatchScorer(Protocol):
    name: str

    def score_corpus(
        self,
        corpus_texts: list[str],
        lexicon_terms: list[str],
        exclusionary_terms: list[str] | None = None,
        exclusionary_weight: float = 1.0,
        checkpoint_path: Path | None = None,
    ) -> list[ScoredResult]: ...


def _find_matched_terms(text: str, lexicon_terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in lexicon_terms if term.lower() in lowered]


def _clean_and_match(text: str, lexicon_terms: list[str]) -> tuple[str, list[str]]:
    """Combines `clean_text()` + `_find_matched_terms()` into one per-document unit of work, so
    `parallel_map()` covers both in a single pass/single progress bar. Live-measured reason this
    matters: run separately, `_find_matched_terms` alone (pure Python substring search, no NLTK)
    was a second, *silent* (no progress bar) bottleneck -- 42.2s for 150k real documents with zero
    terminal feedback, found only by explicitly benchmarking it after the cleaning stage's own fix
    landed. Module-level (not a closure) so it stays picklable for `multiprocessing`."""
    return clean_text(text), _find_matched_terms(text, lexicon_terms)


def _write_checkpoint(checkpoint_path: Path, **fields: object) -> None:
    """Best-effort progress checkpoint -- a small JSON status file, not the actual scored data
    (BM25/TF-IDF need the *whole* corpus indexed before any single document's score is valid, so
    there's no meaningful partial *score* to persist mid-corpus-cleaning). This exists so a run
    that's killed or crashes leaves real evidence of exactly how far it got, rather than nothing --
    see STEPS_Progress.md Step 12 for the 12-hour incident this is a direct response to."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps({**fields, "updated_at": datetime.now(timezone.utc).isoformat()}, indent=2)
    )


class WeightedSumScorer:
    """Sum of each present lexicon term's weight (typically its TF-IDF discriminative_score).
    Fully transparent and cheap -- the simplest baseline to sanity-check the others against.

    If `exclusionary_terms` are given, subtracts `exclusionary_weight * (sum of each matched
    exclusionary term's own weight)`, using `exclusionary_term_weights` if supplied. **Do not
    fall back to an unweighted match count here** -- a live bake-off run confirmed this in
    practice: with an unweighted count (1.0 per match, the original implementation), AUROC
    collapsed from 0.700 to 0.454 (worse than random) because a single exclusionary match (e.g.
    the word "forest") swamped an entire paper's positive sum, which is typically ~0.0001-0.02 in
    magnitude (real discriminative_score values are tiny) -- a ~50-100x scale mismatch per match.
    Exclusionary weights are `abs()`-ed before use: a term manually reclassified from the positive
    pile (see curate/term_review_state.py) keeps its originally-snapshotted *positive*
    discriminative_score, which would otherwise flip the subtraction's sign and *reward* a match
    instead of penalizing it."""

    name = "weighted-sum"

    def __init__(
        self,
        term_weights: dict[str, float],
        exclusionary_term_weights: dict[str, float] | None = None,
    ):
        self.term_weights = term_weights
        self.exclusionary_term_weights = exclusionary_term_weights or {}

    def score_corpus(
        self,
        corpus_texts: list[str],
        lexicon_terms: list[str],
        exclusionary_terms: list[str] | None = None,
        exclusionary_weight: float = 1.0,
        checkpoint_path: Path | None = None,
    ) -> list[ScoredResult]:
        results = []
        started = time.monotonic()
        n = len(corpus_texts)
        for i, text in enumerate(tqdm(corpus_texts, desc=f"{self.name}: scoring", unit="doc"), start=1):
            matched = _find_matched_terms(text, lexicon_terms)
            total = sum(self.term_weights.get(term, 0.0) for term in matched)
            if exclusionary_terms:
                matched_exclusionary = _find_matched_terms(text, exclusionary_terms)
                exclusionary_total = sum(
                    abs(self.exclusionary_term_weights.get(term, 1.0)) for term in matched_exclusionary
                )
                total -= exclusionary_weight * exclusionary_total
            results.append((total, matched))
            if checkpoint_path and n and i % max(1, n // _CHECKPOINT_CHUNKS) == 0:
                _write_checkpoint(
                    checkpoint_path,
                    scorer=self.name,
                    stage="scoring",
                    processed=i,
                    total=n,
                    pct_complete=round(i / n * 100, 1),
                    elapsed_seconds=round(time.monotonic() - started, 1),
                )
        return results


class Bm25Scorer:
    """Okapi BM25 (rank_bm25), scoring every document in the corpus against a pseudo-query built
    from the approved lexicon terms. BM25's own IDF + length-normalization naturally up-weights
    rare-but-consistent terms and down-weights matches driven purely by long abstracts -- a
    standard, well-understood IR relevance-ranking method.

    If `exclusionary_terms` are given, a second BM25 score is computed against the same corpus
    index using a pseudo-query built from them, and `exclusionary_weight * that score` is
    subtracted -- real negative evidence from the discriminative_score's negative tail (see
    curate/term_review_state.py), not just the absence of positive matches. Note: BM25's query
    tokenization flattens every lexicon term to unigrams (`.split()` on whitespace) -- multi-word
    phrase structure isn't preserved for either the positive or exclusionary pseudo-query."""

    name = "bm25"

    def score_corpus(
        self,
        corpus_texts: list[str],
        lexicon_terms: list[str],
        exclusionary_terms: list[str] | None = None,
        exclusionary_weight: float = 1.0,
        checkpoint_path: Path | None = None,
    ) -> list[ScoredResult]:
        n = len(corpus_texts)
        started = time.monotonic()

        def _on_progress(processed: int, total: int) -> None:
            elapsed = time.monotonic() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (total - processed) / rate if rate > 0 else 0.0
            print(
                f"bm25: {processed}/{total} docs cleaned+matched ({processed / total * 100:.1f}%) -- "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {rate:.1f} docs/sec"
            )
            if checkpoint_path:
                _write_checkpoint(
                    checkpoint_path,
                    scorer=self.name,
                    stage="cleaning",
                    processed=processed,
                    total=total,
                    pct_complete=round(processed / total * 100, 1),
                    elapsed_seconds=round(elapsed, 1),
                )

        # Consume the generator straight into the two target lists -- never materializes a
        # combined (cleaned, matched) intermediate for the whole corpus. At 744,647 real
        # documents, holding that intermediate *plus* these two lists simultaneously was the
        # direct cause of an OOM-killed container (see parallel_map()'s docstring).
        tokenized_corpus: list[list[str]] = []
        matched_terms: list[list[str]] = []
        for cleaned, matched in parallel_map(
            partial(_clean_and_match, lexicon_terms=lexicon_terms),
            corpus_texts,
            desc="bm25: cleaning+tokenizing+matching corpus",
            on_progress=_on_progress,
        ):
            # sys.intern(): the ~745k-document corpus has enormous token repetition (common
            # words like "model"/"patient"/"study" appear in most documents) -- interning makes
            # every repeated occurrence a pointer to one shared string object instead of a fresh
            # ~50+-byte object per occurrence. `str.split()` always creates new string objects
            # (unlike compile-time literals, CPython doesn't auto-intern runtime substrings), so
            # this needs to be explicit. A real, measured contributor to the OOM incidents this
            # module's other comments describe -- not a micro-optimization added speculatively.
            tokenized_corpus.append([sys.intern(tok) for tok in cleaned.split()])
            matched_terms.append(matched)

        print(f"bm25: indexing {len(tokenized_corpus)} documents...")
        t0 = time.monotonic()
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"bm25: indexed in {time.monotonic() - t0:.1f}s. scoring against positive lexicon...")

        t0 = time.monotonic()
        query_tokens = clean_text(" ".join(lexicon_terms)).split()
        scores = bm25.get_scores(query_tokens)
        print(f"bm25: positive-lexicon scoring done in {time.monotonic() - t0:.1f}s.")
        if exclusionary_terms:
            t0 = time.monotonic()
            print("bm25: scoring against exclusionary lexicon...")
            exclusionary_query_tokens = clean_text(" ".join(exclusionary_terms)).split()
            exclusionary_scores = bm25.get_scores(exclusionary_query_tokens)
            scores = scores - exclusionary_weight * exclusionary_scores
            print(f"bm25: exclusionary-lexicon scoring done in {time.monotonic() - t0:.1f}s.")
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, scorer=self.name, stage="done", processed=n, total=n, pct_complete=100.0)
        print("bm25: done.")
        return list(zip((float(s) for s in scores), matched_terms))


class TfidfCosineScorer:
    """Cosine similarity between each document's TF-IDF vector (fit fresh on the corpus being
    scored) and a pseudo-document built from the approved lexicon terms.

    If `exclusionary_terms` are given, a second pseudo-document is fit into the *same* vectorizer
    call (so both similarities live in the same vector space) and `exclusionary_weight * that
    similarity` is subtracted. Note: the vectorizer here uses its default ngram_range=(1,1), so
    multi-word lexicon/exclusionary terms are flattened to unigrams, same caveat as Bm25Scorer."""

    name = "tfidf-cosine"

    def score_corpus(
        self,
        corpus_texts: list[str],
        lexicon_terms: list[str],
        exclusionary_terms: list[str] | None = None,
        exclusionary_weight: float = 1.0,
        checkpoint_path: Path | None = None,
    ) -> list[ScoredResult]:
        n = len(corpus_texts)
        started = time.monotonic()

        def _on_progress(processed: int, total: int) -> None:
            elapsed = time.monotonic() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (total - processed) / rate if rate > 0 else 0.0
            print(
                f"tfidf-cosine: {processed}/{total} docs cleaned+matched ({processed / total * 100:.1f}%) -- "
                f"{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {rate:.1f} docs/sec"
            )
            if checkpoint_path:
                _write_checkpoint(
                    checkpoint_path,
                    scorer=self.name,
                    stage="cleaning",
                    processed=processed,
                    total=total,
                    pct_complete=round(processed / total * 100, 1),
                    elapsed_seconds=round(elapsed, 1),
                )

        # See Bm25Scorer.score_corpus for why this consumes the generator directly rather than
        # materializing a combined (cleaned, matched) intermediate for the whole corpus.
        cleaned_corpus: list[str] = []
        matched_terms: list[list[str]] = []
        for cleaned, matched in parallel_map(
            partial(_clean_and_match, lexicon_terms=lexicon_terms),
            corpus_texts,
            desc="tfidf-cosine: cleaning+matching corpus",
            on_progress=_on_progress,
        ):
            cleaned_corpus.append(cleaned)
            matched_terms.append(matched)

        pseudo_query = clean_text(" ".join(lexicon_terms))
        extra_docs = [pseudo_query]
        if exclusionary_terms:
            extra_docs.append(clean_text(" ".join(exclusionary_terms)))

        print(f"tfidf-cosine: fitting vectorizer over {len(cleaned_corpus)} documents...")
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([*cleaned_corpus, *extra_docs])
        doc_vectors = matrix[: len(cleaned_corpus)]
        query_vector = matrix[len(cleaned_corpus)]
        print("tfidf-cosine: computing cosine similarity...")
        similarities = cosine_similarity(doc_vectors, query_vector).ravel()

        if exclusionary_terms:
            exclusionary_vector = matrix[len(cleaned_corpus) + 1]
            exclusionary_similarities = cosine_similarity(doc_vectors, exclusionary_vector).ravel()
            similarities = similarities - exclusionary_weight * exclusionary_similarities

        print("tfidf-cosine: done.")
        return list(zip((float(s) for s in similarities), matched_terms))


SCORERS: dict[str, type] = {
    "weighted-sum": WeightedSumScorer,
    "bm25": Bm25Scorer,
    "tfidf-cosine": TfidfCosineScorer,
}


def load_lexicon_terms_and_weights(lexicon_df) -> tuple[list[str], dict[str, float]]:
    """Extracts the approved term list + a weight per term (discriminative_score, defaulting to
    1.0 for terms without one, e.g. seed-only terms) from an approved keyword_lexicon.csv."""
    terms = lexicon_df["term"].dropna().tolist()
    has_score = "discriminative_score" in lexicon_df.columns
    weights = {
        row["term"]: (
            row["discriminative_score"]
            if has_score and pd.notna(row["discriminative_score"])
            else 1.0
        )
        for _, row in lexicon_df.iterrows()
    }
    return terms, weights
