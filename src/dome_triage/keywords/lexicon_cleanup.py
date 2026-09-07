"""Explainable, rule-based cleanup for a combined (original curated + added) lexicon, run before
suggesting it as a candidate for production BM25/TF-IDF-cosine scoring. Four plain, inspectable
rules, each producing a logged entry -- no ML/black-box dedup here, matching AGENTS.md's "no
premature abstraction" (a research pipeline with a handful of real inputs). See
pipeline/steps.py::step_keywords_suggest_final_lexicon for how tier 1 + tier 2 get combined before
being passed in here, and tests/test_lexicon_cleanup.py for the real forest/random/neural/area
regression case this was built to reason about correctly.
"""

from __future__ import annotations

import pandas as pd

LOG_COLUMNS = ["term", "list", "action", "reason"]


def _normalize(term: str) -> str:
    return term.strip().lower()


def _tokens(term: str) -> list[str]:
    return _normalize(term).split()


def clean_lexicon(
    positive_df: pd.DataFrame,
    negative_df: pd.DataFrame,
    protected_unigrams: frozenset = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Applies, in order: (1) exact-duplicate removal, (2) stray short-term removal, (3)
    within-list phrase-subsumption removal (skipping anything in `protected_unigrams` -- see
    keywords/curated_terms.py::PROTECTED_UNIGRAMS -- kept and logged instead of removed), (4)
    cross-list tension flagging (never removed). `positive_df`/`negative_df` must already be the
    fully combined pool (tier 1 + tier 2) -- this function only cleans, it doesn't merge. Returns
    (cleaned_positive, cleaned_negative, log)."""
    protected_unigrams = {_normalize(t) for t in protected_unigrams}
    log_rows: list[dict] = []

    positive_df, log = _drop_exact_duplicates(positive_df, "positive")
    log_rows.extend(log)
    negative_df, log = _drop_exact_duplicates(negative_df, "negative")
    log_rows.extend(log)

    positive_df, log = _drop_short_terms(positive_df, "positive")
    log_rows.extend(log)
    negative_df, log = _drop_short_terms(negative_df, "negative")
    log_rows.extend(log)

    positive_df, log = _drop_subsumed_unigrams(positive_df, "positive", protected_unigrams)
    log_rows.extend(log)
    negative_df, log = _drop_subsumed_unigrams(negative_df, "negative", protected_unigrams)
    log_rows.extend(log)

    log_rows.extend(_flag_cross_list_tension(positive_df, negative_df))

    log_df = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    return positive_df.reset_index(drop=True), negative_df.reset_index(drop=True), log_df


def _drop_exact_duplicates(df: pd.DataFrame, list_name: str) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return df, []
    normalized = df["term"].map(_normalize)
    is_dup = normalized.duplicated(keep="first")
    log = [
        {
            "term": row["term"],
            "list": list_name,
            "action": "removed",
            "reason": "exact duplicate term (case-normalized)",
        }
        for _, row in df[is_dup].iterrows()
    ]
    return df[~is_dup], log


def _drop_short_terms(df: pd.DataFrame, list_name: str, min_length: int = 3) -> tuple[pd.DataFrame, list[dict]]:
    if df.empty:
        return df, []
    is_short = df["term"].str.strip().str.len() < min_length
    log = [
        {
            "term": row["term"],
            "list": list_name,
            "action": "removed",
            "reason": f"term shorter than {min_length} characters -- unlikely to carry real signal",
        }
        for _, row in df[is_short].iterrows()
    ]
    return df[~is_short], log


def _drop_subsumed_unigrams(
    df: pd.DataFrame, list_name: str, protected_unigrams: set = frozenset()
) -> tuple[pd.DataFrame, list[dict]]:
    """A unigram term subsumed by a longer phrase already in the SAME list: the "unigram that's
    only really relevant as part of a bigram" case. BM25/TF-IDF-cosine (keywords/scoring.py) flatten
    every lexicon term to unigram tokens before scoring, so a standalone "learning" entry
    contributes nothing beyond what "machine learning" already contributes once decomposed -- it
    only adds broader, noisier matching (any use of the bare word, not just the ML sense).
    Exception: `protected_unigrams` (specific technical abbreviations, not generic words) are kept
    and logged as such rather than removed."""
    if df.empty:
        return df, []
    terms = df["term"].tolist()
    multiword_terms = [t for t in terms if len(_tokens(t)) > 1]

    to_remove = []
    log = []
    for term in terms:
        tokens = _tokens(term)
        if len(tokens) != 1:
            continue
        subsuming = [p for p in multiword_terms if term != p and tokens[0] in _tokens(p)]
        if not subsuming:
            continue
        if tokens[0] in protected_unigrams:
            log.append(
                {
                    "term": term,
                    "list": list_name,
                    "action": "kept_flagged",
                    "reason": (
                        f"kept -- explicitly protected (see keywords/curated_terms.py::"
                        f"PROTECTED_UNIGRAMS) despite being subsumed by phrase(s) {subsuming!r}: "
                        "specific enough as a standalone technical term/abbreviation to carry "
                        "real signal on its own, unlike a generic word"
                    ),
                }
            )
            continue
        to_remove.append(term)
        log.append(
            {
                "term": term,
                "list": list_name,
                "action": "removed",
                "reason": (
                    f"redundant unigram: subsumed by phrase(s) {subsuming!r} already in the "
                    "same list -- BM25/TF-IDF-cosine flatten phrases to unigram tokens anyway "
                    "(see keywords/scoring.py), so this adds no marginal discriminative "
                    "signal, only broader/noisier matching"
                ),
            }
        )
    cleaned = df[~df["term"].isin(to_remove)]
    return cleaned, log


def _flag_cross_list_tension(positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> list[dict]:
    """A negative unigram that's a token inside a positive phrase (e.g. "forest" vs "random
    forest") is NOT removed -- it's an explicit curated decision (a real exclusionary
    counterweight to the word's generic sense) and human curation is never silently overridden
    (AGENTS.md). It's flagged instead, since given the scorers' unigram-flattening behavior it
    will dampen (not cancel) the overlapping positive phrase's score contribution."""
    if positive_df.empty or negative_df.empty:
        return []
    positive_terms = positive_df["term"].tolist()
    log = []
    for term in negative_df["term"].tolist():
        tokens = _tokens(term)
        if len(tokens) != 1:
            continue
        overlaps = [p for p in positive_terms if tokens[0] in _tokens(p)]
        if overlaps:
            log.append(
                {
                    "term": term,
                    "list": "negative",
                    "action": "kept_flagged",
                    "reason": (
                        f"kept (an explicit decision, not overridden) -- but overlaps token(s) "
                        f"of positive phrase(s) {overlaps!r}. Given the scorers' "
                        "unigram-flattening behavior, this dampens (does not cancel) those "
                        f"phrases' score contribution for any document containing {term!r}."
                    ),
                }
            )
    return log
