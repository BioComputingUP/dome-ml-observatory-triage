from dome_triage.keywords.scoring import Bm25Scorer, TfidfCosineScorer, WeightedSumScorer

RELEVANT_DOC = (
    "We trained a random forest classifier on genomic data to predict cancer subtype "
    "with high accuracy using machine learning."
)
IRRELEVANT_DOC = "This paper reviews the history of impressionist painting in 19th century France."
LEXICON_TERMS = ["random forest", "machine learning", "classifier", "accuracy"]

# BM25/TF-IDF need more than 2 documents to give a meaningful signal: with exactly 2 docs where
# every term's document-frequency is exactly N/2, the classic Robertson-Sparck-Jones IDF term
# rank_bm25 uses degenerates to exactly zero for every term (a real, documented BM25 property at
# tiny corpus sizes, not specific to this codebase) -- so corpus-level scorer tests use a slightly
# larger, more realistic corpus.
FILLER_DOCS = [
    "A cross-sectional survey of patient satisfaction across three hospitals.",
    "We describe a new surgical technique for knee replacement in elderly patients.",
]


def test_weighted_sum_scorer_ranks_relevant_document_higher():
    weights = {t: 1.0 for t in LEXICON_TERMS}
    scorer = WeightedSumScorer(weights)
    results = scorer.score_corpus([RELEVANT_DOC, IRRELEVANT_DOC], LEXICON_TERMS)

    relevant_score, relevant_matches = results[0]
    irrelevant_score, irrelevant_matches = results[1]

    assert relevant_score > irrelevant_score
    assert irrelevant_score == 0
    assert "machine learning" in relevant_matches
    assert irrelevant_matches == []


def test_weighted_sum_scorer_respects_term_weights():
    scorer = WeightedSumScorer({"machine learning": 5.0, "classifier": 1.0})
    score, matched = scorer.score_corpus([RELEVANT_DOC], ["machine learning", "classifier"])[0]
    assert score == 6.0
    assert set(matched) == {"machine learning", "classifier"}


def test_bm25_scorer_ranks_relevant_document_higher():
    scorer = Bm25Scorer()
    corpus = [RELEVANT_DOC, IRRELEVANT_DOC, *FILLER_DOCS]
    results = scorer.score_corpus(corpus, LEXICON_TERMS)
    assert results[0][0] > results[1][0]


def test_tfidf_cosine_scorer_ranks_relevant_document_higher():
    scorer = TfidfCosineScorer()
    corpus = [RELEVANT_DOC, IRRELEVANT_DOC, *FILLER_DOCS]
    results = scorer.score_corpus(corpus, LEXICON_TERMS)
    assert results[0][0] > results[1][0]


# "genomic data" appears in RELEVANT_DOC but isn't one of LEXICON_TERMS -- a plausible
# exclusionary/negative-tail term (see curate/term_review_state.py) that should pull the score
# down when supplied, without affecting which lexicon terms are reported as matched.
EXCLUSIONARY_TERMS = ["genomic data"]


def test_weighted_sum_scorer_subtracts_exclusionary_matches():
    weights = {t: 1.0 for t in LEXICON_TERMS}
    scorer = WeightedSumScorer(weights)
    without_penalty, matched = scorer.score_corpus([RELEVANT_DOC], LEXICON_TERMS)[0]
    with_penalty, matched_with_penalty = scorer.score_corpus(
        [RELEVANT_DOC], LEXICON_TERMS, exclusionary_terms=EXCLUSIONARY_TERMS, exclusionary_weight=1.0
    )[0]

    assert with_penalty == without_penalty - 1.0
    assert matched_with_penalty == matched  # exclusionary terms never appear in the matched list


def test_weighted_sum_scorer_uses_real_exclusionary_weights_when_given():
    weights = {t: 1.0 for t in LEXICON_TERMS}
    exclusionary_weights = {"genomic data": 0.05}  # a small, realistic discriminative_score-scale weight
    scorer = WeightedSumScorer(weights, exclusionary_weights)

    without_penalty, _ = scorer.score_corpus([RELEVANT_DOC], LEXICON_TERMS)[0]
    with_penalty, _ = scorer.score_corpus(
        [RELEVANT_DOC], LEXICON_TERMS, exclusionary_terms=EXCLUSIONARY_TERMS, exclusionary_weight=1.0
    )[0]

    # Uses the real 0.05 weight, not an unweighted count of 1.0 -- this is the fix: a live
    # bake-off run showed the unweighted-count fallback (penalty=1.0 per match) swamping the
    # positive sum, since real discriminative_score values are ~0.0001-0.02 in magnitude,
    # collapsing AUROC from 0.700 to 0.454 (worse than random).
    assert with_penalty == without_penalty - 0.05


def test_weighted_sum_scorer_exclusionary_penalty_is_sign_safe():
    # A term manually reclassified from the positive pile (see curate/term_review_state.py)
    # keeps its originally-snapshotted *positive* discriminative_score even though it's now an
    # exclusionary term -- the penalty must still SUBTRACT (never flip sign and reward a match).
    weights = {t: 1.0 for t in LEXICON_TERMS}
    exclusionary_weights = {"genomic data": 0.05}  # positive-signed, as a reclassified term would be
    scorer = WeightedSumScorer(weights, exclusionary_weights)

    without_penalty, _ = scorer.score_corpus([RELEVANT_DOC], LEXICON_TERMS)[0]
    with_penalty, _ = scorer.score_corpus(
        [RELEVANT_DOC], LEXICON_TERMS, exclusionary_terms=EXCLUSIONARY_TERMS, exclusionary_weight=1.0
    )[0]

    assert with_penalty < without_penalty


def test_weighted_sum_scorer_falls_back_to_unit_weight_for_unknown_exclusionary_terms():
    weights = {t: 1.0 for t in LEXICON_TERMS}
    scorer = WeightedSumScorer(weights, exclusionary_term_weights={})  # no weight known for this term
    score, _ = scorer.score_corpus(
        [RELEVANT_DOC], LEXICON_TERMS, exclusionary_terms=EXCLUSIONARY_TERMS, exclusionary_weight=1.0
    )[0]
    without_penalty, _ = scorer.score_corpus([RELEVANT_DOC], LEXICON_TERMS)[0]
    assert score == without_penalty - 1.0


def test_bm25_scorer_exclusionary_terms_reduce_score():
    scorer = Bm25Scorer()
    corpus = [RELEVANT_DOC, IRRELEVANT_DOC, *FILLER_DOCS]
    without_penalty = scorer.score_corpus(corpus, LEXICON_TERMS)[0][0]
    with_penalty = scorer.score_corpus(
        corpus, LEXICON_TERMS, exclusionary_terms=EXCLUSIONARY_TERMS, exclusionary_weight=1.0
    )[0][0]
    assert with_penalty < without_penalty


def test_tfidf_cosine_scorer_exclusionary_terms_reduce_score():
    scorer = TfidfCosineScorer()
    corpus = [RELEVANT_DOC, IRRELEVANT_DOC, *FILLER_DOCS]
    without_penalty = scorer.score_corpus(corpus, LEXICON_TERMS)[0][0]
    with_penalty = scorer.score_corpus(
        corpus, LEXICON_TERMS, exclusionary_terms=EXCLUSIONARY_TERMS, exclusionary_weight=1.0
    )[0][0]
    assert with_penalty < without_penalty
