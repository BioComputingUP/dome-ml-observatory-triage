import pandas as pd

from dome_triage.keywords.lexicon_cleanup import clean_lexicon


def _terms_df(terms):
    return pd.DataFrame({"term": terms})


def test_exact_duplicates_are_removed_and_logged():
    positive = _terms_df(["machine learning", "machine learning", "Machine Learning"])
    negative = _terms_df([])

    cleaned_positive, cleaned_negative, log = clean_lexicon(positive, negative)

    assert cleaned_positive["term"].tolist() == ["machine learning"]
    removed = log[log["action"] == "removed"]
    assert len(removed) == 2
    assert all("duplicate" in r for r in removed["reason"])


def test_short_terms_are_removed_and_logged():
    positive = _terms_df(["ml", "machine learning"])
    negative = _terms_df([])

    cleaned_positive, _, log = clean_lexicon(positive, negative)

    assert "ml" not in cleaned_positive["term"].tolist()
    assert "machine learning" in cleaned_positive["term"].tolist()
    short_log = log[log["reason"].str.contains("shorter than", na=False)]
    assert short_log["term"].tolist() == ["ml"]


def test_unigram_subsumed_by_phrase_in_same_list_is_removed():
    positive = _terms_df(["learning", "machine learning", "deep learning"])
    negative = _terms_df([])

    cleaned_positive, _, log = clean_lexicon(positive, negative)

    assert "learning" not in cleaned_positive["term"].tolist()
    assert set(cleaned_positive["term"].tolist()) == {"machine learning", "deep learning"}
    subsumed_log = log[log["term"] == "learning"]
    assert len(subsumed_log) == 1
    assert subsumed_log.iloc[0]["action"] == "removed"
    assert "machine learning" in subsumed_log.iloc[0]["reason"]
    assert "deep learning" in subsumed_log.iloc[0]["reason"]


def test_unigram_not_subsumed_when_no_matching_phrase_exists():
    positive = _terms_df(["genomics", "proteomics"])
    negative = _terms_df([])

    cleaned_positive, _, log = clean_lexicon(positive, negative)

    assert set(cleaned_positive["term"].tolist()) == {"genomics", "proteomics"}
    assert log.empty


def test_real_regression_case_forest_random_neural_kept_and_flagged_not_removed():
    # Mirrors the user's actual curation data: "forest"/"random"/"neural" marked negative, while
    # "random forest"/"neural network" (etc.) are separately approved positive -- these must be
    # KEPT in the negative list (an explicit human decision), not silently removed, but flagged.
    positive = _terms_df(
        ["random forest", "random forest classifier", "neural network", "convolutional neural network"]
    )
    negative = _terms_df(["forest", "random", "neural"])

    cleaned_positive, cleaned_negative, log = clean_lexicon(positive, negative)

    # Nothing removed from the positive list (no unigrams there to subsume).
    assert set(cleaned_positive["term"].tolist()) == set(positive["term"].tolist())
    # Nothing removed from the negative list either -- these are explicit decisions.
    assert set(cleaned_negative["term"].tolist()) == {"forest", "random", "neural"}

    tension_log = log[log["action"] == "kept_flagged"]
    assert set(tension_log["term"].tolist()) == {"forest", "random", "neural"}
    forest_reason = tension_log[tension_log["term"] == "forest"].iloc[0]["reason"]
    assert "random forest" in forest_reason


def test_protected_unigram_is_kept_and_flagged_not_removed():
    # "svm" is a real, specific standalone term (unlike "learning"/"model") -- explicitly
    # protected so it survives the subsumption rule.
    positive = _terms_df(["svm", "support vector machine svm"])
    negative = _terms_df([])

    cleaned_positive, _, log = clean_lexicon(positive, negative, protected_unigrams={"svm"})

    assert "svm" in cleaned_positive["term"].tolist()
    svm_log = log[log["term"] == "svm"]
    assert len(svm_log) == 1
    assert svm_log.iloc[0]["action"] == "kept_flagged"
    assert "protected" in svm_log.iloc[0]["reason"]


def test_unigram_not_in_protected_set_is_still_removed_when_subsumed():
    positive = _terms_df(["svm", "support vector machine svm"])
    negative = _terms_df([])

    # No protection this time -- default behavior.
    cleaned_positive, _, log = clean_lexicon(positive, negative)

    assert "svm" not in cleaned_positive["term"].tolist()
    assert log[log["term"] == "svm"].iloc[0]["action"] == "removed"


def test_negative_unigram_with_no_positive_overlap_is_not_flagged():
    positive = _terms_df(["machine learning"])
    negative = _terms_df(["editorial"])

    _, cleaned_negative, log = clean_lexicon(positive, negative)

    assert "editorial" in cleaned_negative["term"].tolist()
    assert log.empty


def test_clean_lexicon_handles_empty_negative_list():
    positive = _terms_df(["machine learning", "deep learning"])
    negative = pd.DataFrame(columns=["term"])

    cleaned_positive, cleaned_negative, log = clean_lexicon(positive, negative)

    assert len(cleaned_positive) == 2
    assert cleaned_negative.empty
    assert log.empty


def test_log_has_expected_columns():
    positive = _terms_df(["learning", "machine learning"])
    negative = _terms_df(["forest", "random forest"])

    _, _, log = clean_lexicon(positive, negative)

    assert list(log.columns) == ["term", "list", "action", "reason"]
