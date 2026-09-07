"""Robust non-methods-publication detector (Step 19d) -- flags likely review/commentary/
meta-analysis/case-report/etc. content, independent of any AI/ML signal, so it can (a) gate a new
batch of clear negatives (this project's negative training class needs genuine primary research,
not review articles that merely fail to mention AI/ML) and (b) separately flag the whole existing
`canonical_dataset.csv` the exact same way, as an additive column.

Deliberately a NEW sibling module, not an edit to `cohort_filters.py`'s existing
`annotate_review_term_match` -- that function is live, diagnostic-only (used by the Curate app's
Original Cohort Review page to help a human curator, never gates anything), and changing its
matching semantics under existing callers would silently change what a curator sees there. This
module's `annotate_non_methods` is a materially different contract: a hard-exclusion gate. Both
share the SAME term list (`cohort_filters.load_review_term_list`, unchanged, reused not
duplicated) -- editing `keyword_lexicon_exclusionary.csv` improves both matchers at once, not just
this one.

Full NLTK tokenize+lemmatize+lowercase pipeline, not regex/substring matching --
`cohort_filters.annotate_review_term_match`'s existing approach is a plain case-insensitive
substring check, and that has a real, verified false-positive risk: `"review" in "preview"` is
`True` as a plain substring. Matching a lemmatized token *sequence* instead gets word-boundary
correctness for free from tokenization itself (`"preview"`/`"previewed"` tokenize to their own
single token, never containing a separate "review" token), and lemmatization collapses
morphological variants (`reviews`/`reviewed`/`reviewing` -> `review`, `analyses` -> `analysis`)
without hand-enumerating every suffix form.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from dome_triage.curate.cohort_filters import load_review_term_list
from dome_triage.keywords.preprocess import ensure_nltk_data

_lemmatizer = WordNetLemmatizer()

# Not a markdown parser -- a short, explicit artifact strip, same spirit as Step 19c's HTML-entity
# cleanup for seed-file titles. Applied before tokenizing.
_MARKDOWN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),  # **bold**
    (re.compile(r"(?<!\w)\*(.+?)\*(?!\w)"), r"\1"),  # *italic*
    (re.compile(r"(?<!\w)_(.+?)_(?!\w)"), r"\1"),  # _italic_
    (re.compile(r"`([^`]+)`"), r"\1"),  # `code`
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),  # # heading
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # [text](link)
    (re.compile(r"^>\s?", re.MULTILINE), ""),  # > blockquote
]

_ALPHA_TOKEN_RE = re.compile(r"^[a-z]+$")

# MEDLINE (Title-Case) and JATS (lowercase-hyphenated) publication-type vocabularies both appear
# in this dataset's `pub_types` column -- confirmed live against real data. Matched as discrete,
# case-insensitive EXACT tags, never substring, since these are controlled-vocabulary values, not
# free text.
NON_METHODS_PUB_TYPES: frozenset[str] = frozenset(
    v.lower()
    for v in (
        "Review",
        "Systematic Review",
        "Scoping Review",
        "Meta-Analysis",
        "Case Reports",
        "Comment",
        "Editorial",
        "Letter",
        "News",
        "Practice Guideline",
        "Guideline",
        "review-article",
        "case-report",
        "systematic-review",
        "article-commentary",
    )
)


def strip_formatting(text: object) -> str:
    """html.unescape + the markdown-artifact strip above, then hyphens normalized to spaces so
    "meta-analysis" and "meta analysis" tokenize identically downstream -- one normalization
    point, not two separate matching code paths for the same concept."""
    if not text or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    for pattern, repl in _MARKDOWN_PATTERNS:
        text = pattern.sub(repl, text)
    return text.replace("-", " ")


@lru_cache(maxsize=None)
def _lemma(word: str) -> str:
    """Tries WordNet noun-POS lemmatization first -- correctly reduces plural nouns like
    "reviews"/"analyses" to "review"/"analysis" -- falling back to verb-POS if the noun pass
    doesn't change the word (correctly reduces "reviewing"/"reviewed" to "review", which noun-POS
    alone -- this project's existing `clean_text()` convention -- would miss entirely).

    Noun-first, not verb-first: tried verb-first initially and it broke on "analyses", a real,
    caught-by-test failure -- "analyses" is ambiguous between the plural noun "analyses" (->
    "analysis", what's wanted for matching "meta-analysis") and the rare British-spelling verb
    "he analyses" (-> "analyse", wrong for this purpose). Scientific abstracts are overwhelmingly
    noun-dominated, so resolving the ambiguity toward the noun reading is the right default.
    Cached: common English words recur heavily across thousands of titles/abstracts, and
    lemmatization is the expensive part of this pipeline."""
    noun_lemma = _lemmatizer.lemmatize(word, pos="n")
    if noun_lemma != word:
        return noun_lemma
    return _lemmatizer.lemmatize(word, pos="v")


def lemma_tokens(text: str) -> list[str]:
    """word_tokenize(text.lower()), each alphabetic token lemmatized -- punctuation/numeric
    tokens dropped (they can never be part of a real term match here)."""
    if not text:
        return []
    ensure_nltk_data()
    tokens = word_tokenize(text.lower())
    return [_lemma(t) for t in tokens if _ALPHA_TOKEN_RE.match(t)]


def term_lemma_sequence(term: str) -> tuple[str, ...]:
    """strip_formatting + lemma_tokens applied to a lexicon term itself -- called once per term at
    lexicon-load time (see `build_non_methods_term_sequences`), not per-record."""
    return tuple(lemma_tokens(strip_formatting(term)))


def build_non_methods_term_sequences(exclusionary_lexicon_path: Path) -> list[tuple[str, ...]]:
    """Reuses `cohort_filters.load_review_term_list` unchanged (single authoritative term list,
    shared with the existing diagnostic matcher) and pre-computes each term's lemma-token
    sequence once. Empty sequences (e.g. a blank term) are dropped -- an empty sequence would
    otherwise vacuously "match" via `contains_term`'s own guard."""
    terms = load_review_term_list(exclusionary_lexicon_path)
    return [seq for seq in (term_lemma_sequence(t) for t in terms) if seq]


def contains_term(doc_tokens: list[str], term_tokens: tuple[str, ...]) -> bool:
    """Exact contiguous-subsequence match -- word-boundary correctness comes free from
    tokenization itself: a term like `("review",)` can only match a real standalone "review"
    token, never as part of a longer token like "preview"."""
    n = len(term_tokens)
    if n == 0 or n > len(doc_tokens):
        return False
    return any(tuple(doc_tokens[i : i + n]) == term_tokens for i in range(len(doc_tokens) - n + 1))


def matches_non_methods_text(
    title: object, abstract: object, term_sequences: list[tuple[str, ...]]
) -> tuple[bool, list[str]]:
    """Returns (matched, matched_term_strings). `matched_term_strings` is the lemmatized,
    space-joined form of whichever term(s) fired (for `match_metadata`/detail-column
    transparency) -- the normalized form actually compared, not necessarily the lexicon's original
    hyphenated spelling."""
    if not term_sequences:
        return False, []
    text = f"{title if isinstance(title, str) else ''} {abstract if isinstance(abstract, str) else ''}"
    doc_tokens = lemma_tokens(strip_formatting(text))
    raw_hits = [" ".join(seq) for seq in term_sequences if contains_term(doc_tokens, seq)]
    # dict.fromkeys, not set() -- preserves first-seen order and dedupes distinct lexicon entries
    # that lemmatize to the same normalized form (e.g. "case study"/"case studies").
    hits = list(dict.fromkeys(raw_hits))
    return bool(hits), hits


def matches_non_methods_pub_types(pub_types: object) -> tuple[bool, list[str]]:
    """`pub_types` may already be a list (in-memory RawRecord/CanonicalRecord) or a JSON-string
    cell (as read back from a CSV via `dtype=str`) -- handled either way, malformed/empty input
    treated as "no pub types", never raises."""
    if isinstance(pub_types, str):
        try:
            pub_types = json.loads(pub_types) if pub_types.strip() else []
        except json.JSONDecodeError:
            pub_types = []
    if not isinstance(pub_types, list):
        return False, []
    hits = [pt for pt in pub_types if isinstance(pt, str) and pt.lower() in NON_METHODS_PUB_TYPES]
    return bool(hits), hits


def annotate_non_methods(
    dataset: pd.DataFrame, exclusionary_lexicon_path: Path, pub_types_col: str = "pub_types"
) -> pd.DataFrame:
    """Adds `likely_review_or_non_methods` (real Python `bool`) and
    `likely_review_or_non_methods_detail` (dict: `{"text_hits": [...], "pub_type_hits": [...]}`)
    -- the single shared detector used both to gate Step 19d's new negative batch
    (`ingest/clear_negative_sampler.py::fetch_filtered_clear_negatives`) and to flag the whole
    existing `canonical_dataset.csv` the same way (`curate/state.py::flag_likely_reviews`), so
    "same classification" is literally the same code path, not two implementations that could
    drift apart. Either signal alone is sufficient (pure OR) -- the structured `pub_types` check
    and the text-lexicon check are independent evidence, not required to agree.

    Returns a copy; `dataset` itself is never mutated. The boolean column stays a real Python
    `bool` here -- CSV string conversion (`"True"`/`"False"`) happens only at the write call site,
    this project's established convention (see `curate/state.py`'s documented pyarrow/
    `ArrowStringArray` reasoning for why a bare bool can't be written directly)."""
    dataset = dataset.copy()
    term_sequences = build_non_methods_term_sequences(exclusionary_lexicon_path)

    titles = dataset["title"] if "title" in dataset.columns else pd.Series([None] * len(dataset), index=dataset.index)
    abstracts = (
        dataset["abstract"] if "abstract" in dataset.columns else pd.Series([None] * len(dataset), index=dataset.index)
    )
    pub_types_series = (
        dataset[pub_types_col]
        if pub_types_col in dataset.columns
        else pd.Series([None] * len(dataset), index=dataset.index)
    )

    text_results = [matches_non_methods_text(t, a, term_sequences) for t, a in zip(titles, abstracts)]
    pubtype_results = [matches_non_methods_pub_types(pt) for pt in pub_types_series]

    dataset["likely_review_or_non_methods"] = [
        t_hit or p_hit for (t_hit, _), (p_hit, _) in zip(text_results, pubtype_results)
    ]
    dataset["likely_review_or_non_methods_detail"] = [
        {"text_hits": t_hits, "pub_type_hits": p_hits}
        for (_, t_hits), (_, p_hits) in zip(text_results, pubtype_results)
    ]
    return dataset
