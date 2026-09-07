import pandas as pd
import pytest

from dome_triage.curate.term_review_state import (
    TermReviewSession,
    build_review_queue,
    materialize_term_events,
)


def _write_candidates(path):
    df = pd.DataFrame(
        {
            "term": ["deep learning", "random forest", "clinical trial", "impressionism", "questionnaire"],
            "discriminative_score": [0.05, 0.02, 0.001, -0.01, -0.05],
            "document_frequency": [10, 5, 1, 0, 0],
            "source": ["tfidf+keybert", "tfidf", "keybert", "tfidf", "tfidf"],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_positive_queue_filters_and_sorts_by_threshold(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    df = pd.read_csv(candidates_path)

    queue = build_review_queue(
        df,
        "positive_candidates",
        already_decided=set(),
        min_discriminative_score=0.0,
        min_document_frequency=1,
    )

    assert queue["term"].tolist() == ["deep learning", "random forest", "clinical trial"]


def test_exclusionary_queue_filters_and_sorts_by_threshold(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    df = pd.read_csv(candidates_path)

    queue = build_review_queue(
        df, "exclusionary_candidates", already_decided=set(), max_discriminative_score=-0.001
    )

    # sorted worst-first (most negative discriminative_score first)
    assert queue["term"].tolist() == ["questionnaire", "impressionism"]


def test_max_terms_caps_the_queue(tmp_path):
    df = pd.DataFrame(
        {
            "term": [f"term{i}" for i in range(5)],
            "discriminative_score": [0.05, 0.04, 0.03, 0.02, 0.01],
            "document_frequency": [1, 1, 1, 1, 1],
        }
    )
    queue = build_review_queue(df, "positive_candidates", already_decided=set(), max_terms=2)
    assert queue["term"].tolist() == ["term0", "term1"]


def test_already_decided_excludes_term_regardless_of_which_pile_qualifies_it(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    df = pd.read_csv(candidates_path)

    # A very permissive positive threshold would ALSO pull in "impressionism" (score -0.01) --
    # but if it's already been decided (e.g. via the exclusionary pile), it must never resurface
    # here either. Decisions are global, not scoped to whichever pile produced them.
    queue = build_review_queue(
        df,
        "positive_candidates",
        already_decided={"impressionism"},
        min_discriminative_score=-1.0,
        min_document_frequency=1,
    )
    assert "impressionism" not in queue["term"].tolist()

    # sanity check: without already_decided, it would have qualified
    unfiltered = build_review_queue(
        df, "positive_candidates", already_decided=set(), min_discriminative_score=-1.0, min_document_frequency=1
    )
    assert "impressionism" in unfiltered["term"].tolist()


def test_fresh_session_queues_filtered_terms(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"

    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    assert session.total() == 3
    assert session.remaining() == 3
    assert session.current_term()["term"] == "deep learning"


def test_record_decision_advances_queue_and_writes_event_log(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    session.record_decision("positive", notes="clearly ML-specific")

    assert events_path.exists()
    events = pd.read_csv(events_path)
    assert len(events) == 1
    assert events.iloc[0]["term"] == "deep learning"
    assert events.iloc[0]["source_queue"] == "positive_candidates"
    assert events.iloc[0]["decision"] == "positive"
    assert events.iloc[0]["discriminative_score"] == 0.05
    assert session.current_term()["term"] == "random forest"
    assert session.remaining() == 2


def test_record_decision_rejects_invalid_decision(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    with pytest.raises(ValueError):
        session.record_decision("maybe")


def test_backup_created_before_second_write(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    session.record_decision("positive")
    backup_path = tmp_path / "keyword_review_events_backup.csv"
    assert not backup_path.exists()  # no prior file to back up on the first write

    session.record_decision("negative")
    assert backup_path.exists()  # second write backs up the file as it existed after the first


def test_resume_after_crash_skips_already_decided_terms_regardless_of_threshold(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"

    first_session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )
    first_session.record_decision("positive")  # decides "deep learning", then session "crashes"

    resumed_session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
        min_discriminative_score=-1.0,  # much wider threshold than the original session
    )

    assert "deep learning" not in resumed_session.queue["term"].tolist()
    assert resumed_session.current_term()["term"] == "random forest"


def test_skip_current_does_not_log_and_term_reappears_in_fresh_session(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    first_term = session.current_term()["term"]
    session.skip_current()
    assert not events_path.exists()
    assert session.current_term()["term"] == "random forest"

    fresh_session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )
    assert fresh_session.current_term()["term"] == first_term


def test_all_time_counts_ignores_current_threshold(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
        max_terms=1,
    )
    session.record_decision("positive")

    wider_session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
        min_discriminative_score=-1.0,
        max_terms=10,
    )
    assert wider_session.all_time_counts() == {"positive": 1, "negative": 0, "irrelevant": 0}


def test_all_time_counts_are_global_across_queue_sources(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"

    positive_session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )
    positive_session.record_decision("positive")  # decides "deep learning"

    exclusionary_session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="exclusionary_candidates",
        curator="alice",
    )
    # A decision made while browsing the OTHER pile is visible here too -- counts are global.
    assert exclusionary_session.all_time_counts() == {"positive": 1, "negative": 0, "irrelevant": 0}


def test_record_manual_decision_for_known_candidate_snapshots_its_stats(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    session.record_manual_decision("random forest", "negative", notes="false friend here")

    events = pd.read_csv(events_path)
    assert len(events) == 1
    row = events.iloc[0]
    assert row["term"] == "random forest"
    assert row["decision"] == "negative"
    assert row["source_queue"] == "manual"
    assert row["discriminative_score"] == 0.02
    assert row["document_frequency"] == 5


def test_record_manual_decision_for_unknown_term_uses_manual_source_and_blank_stats(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    session.record_manual_decision("convolutional neural network", "positive", notes="never auto-extracted")

    events = pd.read_csv(events_path)
    row = events.iloc[0]
    assert row["term"] == "convolutional neural network"
    assert row["source"] == "manual"
    assert pd.isna(row["discriminative_score"])


def test_record_manual_decision_does_not_advance_queue_position(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    first_term = session.current_term()["term"]
    session.record_manual_decision("some other term", "irrelevant")
    assert session.current_term()["term"] == first_term


def test_record_manual_decision_rejects_invalid_decision(tmp_path):
    candidates_path = _write_candidates(tmp_path / "candidates.csv")
    events_path = tmp_path / "keyword_review_events.csv"
    session = TermReviewSession(
        candidates_path=candidates_path,
        events_path=events_path,
        queue_source="positive_candidates",
        curator="alice",
    )

    with pytest.raises(ValueError):
        session.record_manual_decision("some term", "maybe")


def _write_candidates_for_materialize(path):
    df = pd.DataFrame(
        {
            "term": ["deep learning", "random forest", "impressionism"],
            "discriminative_score": [0.05, 0.02, -0.02],
            "document_frequency": [10, 5, 0],
            "source": ["tfidf", "tfidf", "tfidf"],
            "review_status": ["pending", "pending", "pending"],
        }
    )
    df.to_csv(path, index=False)
    return path


def _write_events(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_materialize_term_events_routes_by_decision_including_manual_term(tmp_path):
    candidates_path = _write_candidates_for_materialize(tmp_path / "candidates.csv")
    events_path = _write_events(
        tmp_path / "events.csv",
        [
            {
                "term": "deep learning", "decision": "positive", "source_queue": "positive_candidates",
                "notes": "", "discriminative_score": 0.05, "document_frequency": 10, "source": "tfidf",
                "curator": "alice", "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "term": "random forest", "decision": "irrelevant", "source_queue": "positive_candidates",
                "notes": "too generic", "discriminative_score": 0.02, "document_frequency": 5,
                "source": "tfidf", "curator": "alice", "timestamp": "2026-01-01T00:00:01+00:00",
            },
            {
                "term": "impressionism", "decision": "negative", "source_queue": "exclusionary_candidates",
                "notes": "", "discriminative_score": -0.02, "document_frequency": 0, "source": "tfidf",
                "curator": "alice", "timestamp": "2026-01-01T00:00:02+00:00",
            },
            {
                "term": "convolutional neural network", "decision": "positive", "source_queue": "manual",
                "notes": "obviously ML-specific, wasn't extracted", "discriminative_score": "",
                "document_frequency": "", "source": "manual", "curator": "alice",
                "timestamp": "2026-01-01T00:00:03+00:00",
            },
        ],
    )
    lexicon_path = tmp_path / "lexicon.csv"
    exclusionary_path = tmp_path / "exclusionary.csv"
    irrelevant_path = tmp_path / "irrelevant.csv"

    counts = materialize_term_events(candidates_path, events_path, lexicon_path, exclusionary_path, irrelevant_path)

    assert counts == {"positive": 2, "negative": 1, "irrelevant": 1}
    lexicon_terms = set(pd.read_csv(lexicon_path)["term"].tolist())
    assert lexicon_terms == {"deep learning", "convolutional neural network"}
    assert pd.read_csv(exclusionary_path)["term"].tolist() == ["impressionism"]
    assert pd.read_csv(irrelevant_path)["term"].tolist() == ["random forest"]

    updated_candidates = pd.read_csv(candidates_path)
    status_by_term = dict(zip(updated_candidates["term"], updated_candidates["review_status"]))
    # "convolutional neural network" isn't a row in candidates.csv -- nothing to sync for it there
    assert status_by_term == {
        "deep learning": "positive",
        "random forest": "irrelevant",
        "impressionism": "negative",
    }


def test_materialize_term_events_last_decision_wins(tmp_path):
    candidates_path = _write_candidates_for_materialize(tmp_path / "candidates.csv")
    events_path = _write_events(
        tmp_path / "events.csv",
        [
            {
                "term": "deep learning", "decision": "irrelevant", "source_queue": "positive_candidates",
                "notes": "", "discriminative_score": 0.05, "document_frequency": 10, "source": "tfidf",
                "curator": "alice", "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "term": "deep learning", "decision": "positive", "source_queue": "positive_candidates",
                "notes": "changed my mind", "discriminative_score": 0.05, "document_frequency": 10,
                "source": "tfidf", "curator": "alice", "timestamp": "2026-01-01T00:05:00+00:00",
            },
        ],
    )
    lexicon_path = tmp_path / "lexicon.csv"
    exclusionary_path = tmp_path / "exclusionary.csv"
    irrelevant_path = tmp_path / "irrelevant.csv"

    counts = materialize_term_events(candidates_path, events_path, lexicon_path, exclusionary_path, irrelevant_path)

    assert counts["positive"] == 1
    assert counts["irrelevant"] == 0
    assert pd.read_csv(lexicon_path)["term"].tolist() == ["deep learning"]


def test_materialize_term_events_backs_up_existing_outputs(tmp_path):
    candidates_path = _write_candidates_for_materialize(tmp_path / "candidates.csv")
    events_path = _write_events(
        tmp_path / "events.csv",
        [
            {
                "term": "deep learning", "decision": "positive", "source_queue": "positive_candidates",
                "notes": "", "discriminative_score": 0.05, "document_frequency": 10, "source": "tfidf",
                "curator": "alice", "timestamp": "2026-01-01T00:00:00+00:00",
            },
        ],
    )
    lexicon_path = tmp_path / "lexicon.csv"
    lexicon_path.write_text("term\nold_term\n")  # pre-existing file that must be backed up
    exclusionary_path = tmp_path / "exclusionary.csv"
    irrelevant_path = tmp_path / "irrelevant.csv"

    materialize_term_events(candidates_path, events_path, lexicon_path, exclusionary_path, irrelevant_path)

    backup_path = tmp_path / "lexicon_backup.csv"
    assert backup_path.exists()
    assert backup_path.read_text() == "term\nold_term\n"


def test_materialize_term_events_returns_zero_counts_when_no_events_file(tmp_path):
    candidates_path = _write_candidates_for_materialize(tmp_path / "candidates.csv")
    counts = materialize_term_events(
        candidates_path,
        tmp_path / "no_events.csv",
        tmp_path / "lexicon.csv",
        tmp_path / "exclusionary.csv",
        tmp_path / "irrelevant.csv",
    )
    assert counts == {"positive": 0, "negative": 0, "irrelevant": 0}
