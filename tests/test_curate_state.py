import json

import pandas as pd

from dome_triage.curate.state import CurationSession, materialize_cross_curate_resolutions, materialize_events


def _write_dataset(path):
    # label/label_confidence/pmcid default to neutral values (unlabeled/unscored/no pmcid) so
    # none of them trip the include_already_labeled/require_pmcid filters below -- existing tests
    # that don't care about those toggles see the same 3-record queue as before.
    df = pd.DataFrame(
        {
            "record_id": ["rec1", "rec2", "rec3"],
            "title": ["Paper One", "Paper Two", "Paper Three"],
            "abstract": ["Abstract one.", "Abstract two.", "Abstract three."],
            "journal": ["J1", "J2", "J3"],
            "year": ["2020", "2021", "2022"],
            "label": ["unlabeled", "unlabeled", "unlabeled"],
            "label_confidence": ["unscored", "unscored", "unscored"],
            "pmcid": ["", "", ""],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_fresh_session_queues_all_records(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"

    session = CurationSession(dataset_path=dataset_path, events_path=events_path, curator="alice")

    assert session.stats() == {"total": 3, "decided": 0, "remaining": 3}
    assert session.current_record()["record_id"] == "rec1"


def test_record_decision_advances_queue_and_writes_event_log(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"
    session = CurationSession(dataset_path=dataset_path, events_path=events_path, curator="alice")

    session.record_decision("positive", tag=None, notes="looks good")

    assert events_path.exists()
    events = pd.read_csv(events_path)
    assert len(events) == 1
    assert events.iloc[0]["record_id"] == "rec1"
    assert events.iloc[0]["decision"] == "positive"
    assert session.current_record()["record_id"] == "rec2"
    assert session.stats() == {"total": 3, "decided": 1, "remaining": 2}


def test_backup_created_before_second_write(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"
    session = CurationSession(dataset_path=dataset_path, events_path=events_path, curator="alice")

    session.record_decision("positive")
    backup_path = tmp_path / "curation_events_backup.csv"
    assert not backup_path.exists()  # no prior file to back up on the first write

    session.record_decision("negative")
    assert backup_path.exists()  # second write backs up the file as it existed after the first


def test_resume_after_crash_skips_already_decided_records(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"

    first_session = CurationSession(
        dataset_path=dataset_path, events_path=events_path, curator="alice"
    )
    first_session.record_decision("positive")  # decides rec1, then session "crashes"

    resumed_session = CurationSession(
        dataset_path=dataset_path, events_path=events_path, curator="alice"
    )

    assert resumed_session.stats() == {"total": 3, "decided": 1, "remaining": 2}
    assert resumed_session.current_record()["record_id"] == "rec2"


def test_record_decision_stores_structured_features_as_json(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"
    session = CurationSession(dataset_path=dataset_path, events_path=events_path, curator="alice")

    session.record_decision(
        "positive", features={"applies_ml_to_data": True, "domain_area": "genomics"}
    )

    events = pd.read_csv(events_path)
    stored_features = json.loads(events.iloc[0]["features"])
    assert stored_features == {"applies_ml_to_data": True, "domain_area": "genomics"}


def test_record_decision_allows_undeterminable(tmp_path):
    # "undeterminable" must be accepted as a genuine decision value, not just tolerated.
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"
    session = CurationSession(dataset_path=dataset_path, events_path=events_path, curator="alice")

    session.record_decision("undeterminable", notes="genuinely unclear even with abstract")

    events = pd.read_csv(events_path)
    assert events.iloc[0]["decision"] == "undeterminable"


def _write_dataset_with_mixed_trust(path):
    df = pd.DataFrame(
        {
            "record_id": ["rec1", "rec2", "rec3", "rec4"],
            "title": ["Trusted positive", "Trusted negative", "Heuristic positive", "Unlabeled"],
            "abstract": ["a", "b", "c", "d"],
            "journal": ["J1", "J2", "J3", "J4"],
            "year": ["2020", "2021", "2022", "2023"],
            "label": ["positive", "negative", "positive", "unlabeled"],
            "label_confidence": ["human_curated", "registry_confirmed", "heuristic_candidate", "unscored"],
            "pmcid": ["PMC1", "", "PMC3", ""],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_trusted_prior_labeled_records_excluded_from_queue_by_default(tmp_path):
    # rec1 (human_curated) and rec2 (registry_confirmed) are already-settled ground truth from a
    # prior curation round -- they must NOT flood a fresh queue. rec3 (heuristic_candidate) and
    # rec4 (unlabeled) still need review.
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice")

    assert session.total() == 2
    ids_in_queue = set(session.queue)
    assert ids_in_queue == {"rec3", "rec4"}


def test_include_already_labeled_toggle_restores_full_queue(tmp_path):
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        include_already_labeled=True,
    )

    assert session.total() == 4
    assert set(session.queue) == {"rec1", "rec2", "rec3", "rec4"}


def test_include_already_labeled_true_is_required_when_the_whole_pool_is_trusted(tmp_path):
    # Regression test for a real, confirmed bug in Step 19's Original Cohort Review page: that
    # page's dataset_df is, by construction (cohort_filters.build_original_cohort), 100%
    # label_confidence="human_curated" with label in {positive, negative} -- exactly
    # CurationSession's own definition of "already curated, hide by default". Without
    # include_already_labeled=True, _scored_pool() excluded every single record, and the page
    # showed "0 of 0 remaining" regardless of sample size (reported live against the real app).
    df = pd.DataFrame(
        {
            "record_id": ["rec1", "rec2", "rec3"],
            "title": ["P1", "P2", "P3"],
            "abstract": ["a", "b", "c"],
            "journal": ["J1", "J2", "J3"],
            "year": ["2018", "2019", "2020"],
            "label": ["positive", "negative", "positive"],
            "label_confidence": ["human_curated", "human_curated", "human_curated"],
            "pmcid": ["", "", ""],
        }
    )
    df.to_csv(tmp_path / "cohort.csv", index=False)

    broken = CurationSession(
        dataset_path=tmp_path / "cohort.csv", events_path=tmp_path / "events.csv", curator="alice"
    )
    assert broken.total() == 0  # documents the bug's exact symptom if the flag is ever dropped

    fixed = CurationSession(
        dataset_path=tmp_path / "cohort.csv",
        events_path=tmp_path / "events2.csv",
        curator="alice",
        include_already_labeled=True,
    )
    assert fixed.total() == 3
    assert set(fixed.queue) == {"rec1", "rec2", "rec3"}


def test_only_already_labeled_filters_queue_to_just_the_trusted_records(tmp_path):
    # The inverse of include_already_labeled: instead of hiding (default) or merging in
    # already-trusted records, restrict the queue to *just* rec1/rec2 (human_curated/
    # registry_confirmed) -- for deliberately re-checking past decisions, not fresh review.
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        only_already_labeled=True,
    )

    assert session.total() == 2
    assert set(session.queue) == {"rec1", "rec2"}


def test_only_already_labeled_takes_precedence_over_include_already_labeled(tmp_path):
    # Setting both flags is a real combination the UI can produce transiently (e.g. a stale
    # widget default) -- only_already_labeled must still win rather than the two conflicting.
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        include_already_labeled=True,
        only_already_labeled=True,
    )

    assert set(session.queue) == {"rec1", "rec2"}


def test_require_pmcid_filters_queue_to_records_with_a_pmcid(tmp_path):
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        include_already_labeled=True,  # isolate the pmcid filter from the trust filter
        require_pmcid=True,
    )

    assert set(session.queue) == {"rec1", "rec3"}  # only these two have a pmcid


def _write_canonical_dataset_with_label(path, label="positive", label_confidence="human_curated"):
    df = pd.DataFrame(
        {
            "record_id": ["rec1"],
            "title": ["Paper One"],
            "abstract": ["Abstract one."],
            "label": [label],
            "label_confidence": [label_confidence],
            "has_conflict": [False],
            "curation_tag": [None],
            "notes": [None],
            "updated_at": [None],
        }
    )
    df.to_csv(path, index=False)
    return path


def test_materialize_events_applies_decision_when_no_conflict(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="unlabeled", label_confidence="unscored"
    )
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "decision": "positive",
                "tag": "",
                "notes": "looks good",
                "features": json.dumps({"applies_ml_to_data": True}),
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    output_path = tmp_path / "canonical_dataset.csv"
    result = materialize_events(dataset_path, events_path, output_path)

    assert result.loc[0, "label"] == "positive"
    assert result.loc[0, "has_conflict"] == "False"
    assert result.loc[0, "label_confidence"] == "human_curated"
    assert json.loads(result.loc[0, "curation_features"]) == {"applies_ml_to_data": True}


def test_materialize_events_upgrades_heuristic_candidate_confidence_on_review(tmp_path):
    # The exact real-world case this exists for: a bulk-match/clear-negative candidate merged in
    # at heuristic_candidate confidence, then a human actually reviews it via the app -- that
    # review must count as a real human judgment, not stay stuck at the original heuristic tier.
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="negative", label_confidence="heuristic_candidate"
    )
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "decision": "negative",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "canonical_dataset.csv")

    assert result.loc[0, "label"] == "negative"
    assert result.loc[0, "label_confidence"] == "human_curated"


def test_materialize_events_flags_conflict_with_trusted_prior_label(tmp_path):
    # A record already trusted as human_curated positive must NOT be silently overwritten if a
    # new curation session decides negative -- this must surface as a conflict instead.
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "decision": "negative",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    output_path = tmp_path / "canonical_dataset.csv"
    result = materialize_events(dataset_path, events_path, output_path)

    assert result.loc[0, "label"] == "conflict"
    assert result.loc[0, "has_conflict"] == "True"
    # already trusted-tier -- that's what made it a conflict; nothing to upgrade, must stay put.
    assert result.loc[0, "label_confidence"] == "human_curated"


def test_materialize_events_flags_conflict_when_undeterminable_contradicts_trusted_prior(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="registry_confirmed"
    )
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "decision": "undeterminable",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "canonical_dataset.csv")
    assert result.loc[0, "label"] == "conflict"


def test_materialize_events_flags_conflict_when_skipped_contradicts_trusted_prior(tmp_path):
    # A real, confirmed bug this guards against: skipping an already-trusted positive/negative
    # record (e.g. Step 19's Original Cohort Review, where every record has a trusted prior label
    # by construction) must NOT silently overwrite `label` to "skipped" -- it must surface as a
    # conflict, exactly like a contradicting positive/negative/undeterminable decision does.
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "decision": "skipped",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "canonical_dataset.csv")
    assert result.loc[0, "label"] == "conflict"
    assert result.loc[0, "has_conflict"] == "True"


def test_materialize_events_skip_on_unlabeled_record_still_applies_cleanly(tmp_path):
    # The common case (skipping a genuinely unlabeled record on the main queue) must be
    # unaffected by the fix above -- prior_label isn't positive/negative here, so no conflict.
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="unlabeled", label_confidence="unscored"
    )
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "rec1",
                "decision": "skipped",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "canonical_dataset.csv")
    assert result.loc[0, "label"] == "skipped"
    assert result.loc[0, "has_conflict"] == "False"


def _write_resolution_event(path, record_id="rec1", decision="negative", **overrides):
    row = {
        "record_id": record_id,
        "decision": decision,
        "tag": "",
        "notes": "",
        "features": "",
        "curator": "alice",
        "timestamp": "2026-01-01T00:00:00+00:00",
        **overrides,
    }
    pd.DataFrame([row]).to_csv(path, index=False)


_LLM_EVENT_COLUMNS = ["record_id", "model_tier", "mode", "classification", "timestamp"]


def _write_llm_events(path, rows):
    defaults = {"mode": "primary"}
    df = pd.DataFrame([{**defaults, **row} for row in rows], columns=_LLM_EVENT_COLUMNS)
    df.to_csv(path, index=False)


def test_materialize_cross_curate_resolutions_applies_final_decision_not_conflict_string(tmp_path):
    """The exact bug materialize_events() would have caused: a trusted prior label ("positive")
    with a final decision that genuinely differs ("negative") must become the real new label, not
    the literal string "conflict" -- this is the whole point of the fix."""
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, decision="negative", notes="agree with second curator")
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(llm_events_path, [])

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "label"] == "negative"
    assert result.loc[0, "label"] != "conflict"
    assert result.loc[0, "label_confidence"] == "human_curated"


def test_materialize_cross_curate_resolutions_preserves_prior_label_in_new_column(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, decision="negative")
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(llm_events_path, [])

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "cross_curate_prior_label"] == "positive"
    assert result.loc[0, "cross_curate_prior_label_confidence"] == "human_curated"
    assert result.loc[0, "cross_curate_final_label"] == "negative"


def test_materialize_cross_curate_resolutions_has_conflict_true_only_when_actually_relabeled(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, decision="negative")  # differs from prior "positive"
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(llm_events_path, [])

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "has_conflict"] == "True"


def test_materialize_cross_curate_resolutions_has_conflict_false_when_original_upheld(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, decision="positive")  # same as prior -- upheld, not a conflict
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(llm_events_path, [])

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "has_conflict"] == "False"
    assert result.loc[0, "label"] == "positive"


def test_materialize_cross_curate_resolutions_records_both_tiers_original_classification(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, decision="negative")
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(
        llm_events_path,
        [
            {"record_id": "rec1", "model_tier": "flash", "classification": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "rec1", "model_tier": "pro", "classification": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
        ],
    )

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "cross_curate_flash_classification"] == "negative"
    assert result.loc[0, "cross_curate_pro_classification"] == "negative"


def test_materialize_cross_curate_resolutions_uses_latest_llm_event_per_tier(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(
        tmp_path / "canonical_dataset.csv", label="positive", label_confidence="human_curated"
    )
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, decision="positive")
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(
        llm_events_path,
        [
            {"record_id": "rec1", "model_tier": "flash", "classification": "negative", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"record_id": "rec1", "model_tier": "flash", "classification": "positive", "timestamp": "2026-01-02T00:00:00+00:00"},
        ],
    )

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "cross_curate_flash_classification"] == "positive"  # the later event wins


def test_materialize_cross_curate_resolutions_skips_unmatched_record_id_gracefully(tmp_path, capsys):
    dataset_path = _write_canonical_dataset_with_label(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "cross_curate_resolution_events.csv"
    _write_resolution_event(events_path, record_id="does_not_exist", decision="negative")
    llm_events_path = tmp_path / "llm_classification_events.csv"
    _write_llm_events(llm_events_path, [])

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert "WARNING" in capsys.readouterr().out
    assert result.loc[0, "record_id"] == "rec1"
    assert result.loc[0, "label"] == "positive"  # untouched -- the event was for a different record


def test_materialize_cross_curate_resolutions_missing_events_file_returns_dataset_unchanged(tmp_path):
    dataset_path = _write_canonical_dataset_with_label(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "does_not_exist.csv"
    llm_events_path = tmp_path / "llm_classification_events.csv"

    result = materialize_cross_curate_resolutions(
        dataset_path, events_path, llm_events_path, tmp_path / "canonical_dataset.csv"
    )

    assert result.loc[0, "label"] == "positive"
    assert "cross_curate_final_label" not in result.columns or pd.isna(result.loc[0, "cross_curate_final_label"])


def _write_dataset_for_filters(path):
    df = pd.DataFrame(
        {
            "record_id": ["rec1", "rec2", "rec3", "rec4"],
            "title": ["P1", "P2", "P3", "P4"],
            "abstract": ["a", "b", "c", "d"],
            "journal": ["Nature", "Nature", "Science", "Obscure Journal"],
            "year": ["2018", "2022", "2020", "2015"],
            "label": ["unlabeled"] * 4,
            "label_confidence": ["unscored"] * 4,
            "pmcid": ["PMC1", "PMC2", "PMC3", "PMC4"],
            "pmid": ["", "", "", ""],
            "doi": ["", "", "", ""],
        }
    )
    df.to_csv(path, index=False)
    return path


_FILTER_SCORE_LOOKUP = {
    "PMC1": (10.0, "negative"),
    "PMC2": (200.0, "positive"),
    "PMC3": (150.0, "positive"),
    "PMC4": (5.0, "negative"),
}
_FILTER_SCREENING_LOOKUP = {"PMC1": True, "PMC2": False, "PMC3": False, "PMC4": False}


def test_classification_filter_restricts_queue(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,
        classification=["positive"],
    )
    assert set(session.queue) == {"rec2", "rec3"}


def test_year_range_filter_restricts_queue(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        year_range=(2019, 2022),
    )
    assert set(session.queue) == {"rec2", "rec3"}  # 2022 and 2020; 2018/2015 excluded


def test_journal_filter_matches_exact_journal_names(tmp_path):
    # journals filters directly against the real `journal` column (not a top-N-or-"other"
    # bucket) -- a curator searching for a specific journal by name needs an exact match, and
    # must be able to select ANY journal, not just one from a pre-baked top-N shortlist.
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        journals=["Nature"],
    )
    assert set(session.queue) == {"rec1", "rec2"}

    obscure_session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        journals=["Obscure Journal"],
    )
    assert set(obscure_session.queue) == {"rec4"}

    multi_session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        journals=["Science", "Obscure Journal"],
    )
    assert set(multi_session.queue) == {"rec3", "rec4"}


def test_score_band_filter_splits_high_and_low(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,
        score_band=[1],  # top half: 150.0, 200.0 -> rec2, rec3
        n_score_bands=2,
    )
    assert set(session.queue) == {"rec2", "rec3"}


def test_needs_screening_only_filter(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        screening_lookup=_FILTER_SCREENING_LOOKUP,
        needs_screening_only=True,
    )
    assert set(session.queue) == {"rec1"}


def test_diversity_stats_counts_trusted_prior_labels(tmp_path):
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice")

    stats = session.diversity_stats()

    # rec1 (human_curated positive, J1) and rec2 (registry_confirmed negative, J2) count;
    # rec3 (heuristic_candidate, unreviewed) does not, even though its label is "positive".
    assert stats["n_journals_covered"] == 2
    assert stats["n_journals_total"] == 4
    assert stats["per_journal_counts"].loc["J1", "positive"] == 1
    assert stats["per_journal_counts"].loc["J2", "negative"] == 1
    assert "J3" not in stats["per_journal_counts"].index


def test_diversity_stats_reflects_live_session_decisions_before_materialize(tmp_path):
    # rec3 starts as heuristic_candidate/positive (not yet "confirmed") -- deciding it via the
    # app in THIS session must move the dashboard immediately, without running curate materialize.
    dataset_path = _write_dataset_with_mixed_trust(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice"
    )
    before = session.diversity_stats()
    assert "J3" not in before["per_journal_counts"].index

    while session.current_record() is not None and session.current_record()["record_id"] != "rec3":
        session.record_decision("skipped")
    session.record_decision("negative")  # decide rec3

    after = session.diversity_stats()
    assert after["per_journal_counts"].loc["J3", "negative"] == 1
    assert after["n_journals_covered"] == before["n_journals_covered"] + 1


def test_scored_pool_is_memoized_per_instance(tmp_path):
    # Real, live incident this guards against: _scored_pool() was being recomputed independently
    # by __post_init__, score_band_summary(), and year_bounds() -- up to ~6 full recomputations
    # per Curate-page rerun, each one expensive (see annotate_bulk_scores' own fix) -- directly
    # responsible for a 40-58s per-decision reload. Same object identity (not just equal values)
    # on repeat access proves it's genuinely cached, not just coincidentally fast.
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,
    )
    first = session._scored_pool()
    second = session._scored_pool()
    assert first is second
    # score_band_summary()/year_bounds() must reuse the same cached pool too, not their own copy
    session.score_band_summary()
    session.year_bounds()
    assert session._scored_pool() is first


def test_score_band_summary_reports_real_ranges_and_curated_counts(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,  # PMC1:10, PMC2:200, PMC3:150, PMC4:5
        n_score_bands=2,
    )

    summary = session.score_band_summary()

    assert [s["band"] for s in summary] == [0, 1]
    low, high = summary
    assert low["min_score"] == 5.0 and low["max_score"] == 10.0
    assert high["min_score"] == 150.0 and high["max_score"] == 200.0
    assert low["total"] == 2 and high["total"] == 2
    # nothing decided yet -- all zero
    assert low["confirmed"] == 0 and high["confirmed"] == 0


def test_score_band_summary_reflects_confirmed_decisions(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,
        n_score_bands=2,
    )
    session.record_decision("negative")  # decides rec1 (PMC1, score 10 -> low band)

    summary = session.score_band_summary()
    low = next(s for s in summary if s["band"] == 0)
    assert low["confirmed"] == 1


def test_score_band_summary_does_not_count_skipped_or_undeterminable_as_confirmed(tmp_path):
    # Real, live bug this fixes: Skip/Undeterminable explicitly mean "not yet assessed" /
    # "looked and couldn't tell" -- neither is "curated", so a freshly-opened session where you've
    # only skipped things must show 0 confirmed, not a nonzero count for that band.
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,
        n_score_bands=2,
    )
    session.record_decision("skipped")  # rec1 (PMC1, score 10 -> low band)
    session.record_decision("undeterminable")  # rec2 (PMC2, score 200 -> high band)

    summary = session.score_band_summary()
    low = next(s for s in summary if s["band"] == 0)
    high = next(s for s in summary if s["band"] == 1)
    assert low["confirmed"] == 0
    assert high["confirmed"] == 0


def test_year_bounds_reflects_filtered_population_not_whole_dataset(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")  # years 2015-2022
    session = CurationSession(
        dataset_path=dataset_path,
        events_path=tmp_path / "events.csv",
        curator="alice",
        classification=["positive"],  # rec2 (2022) and rec3 (2020) only, per _FILTER_SCORE_LOOKUP
        bulk_score_lookup=_FILTER_SCORE_LOOKUP,
    )
    assert session.year_bounds() == (2020, 2022)


def test_year_bounds_unfiltered_spans_whole_dataset(tmp_path):
    dataset_path = _write_dataset_for_filters(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice"
    )
    assert session.year_bounds() == (2015, 2022)


def test_go_back_and_go_forward_navigate_without_changing_remaining(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice"
    )

    assert not session.can_go_back()
    session.record_decision("positive")  # decides rec1, now on rec2
    assert session.current_record()["record_id"] == "rec2"
    assert session.stats() == {"total": 3, "decided": 1, "remaining": 2}

    assert session.can_go_back()
    session.go_back()
    assert session.current_record()["record_id"] == "rec1"
    # remaining/decided must NOT change just from looking backward
    assert session.stats() == {"total": 3, "decided": 1, "remaining": 2}

    assert session.can_go_forward()
    session.go_forward()
    assert session.current_record()["record_id"] == "rec2"
    # rec3 still exists ahead in the queue -- forward keeps working past the decided frontier,
    # browsing is not gated on having decided anything.
    assert session.can_go_forward()


def test_go_forward_works_before_anything_is_decided(tmp_path):
    # Real requirement, not hypothetical: forward/back must work regardless of whether you've
    # curated yet -- pure browsing, not gated on decision progress.
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice"
    )
    assert session.can_go_forward()
    session.go_forward()
    assert session.current_record()["record_id"] == "rec2"
    # remaining/decided must NOT move just from browsing -- nothing has actually been decided
    assert session.stats() == {"total": 3, "decided": 0, "remaining": 3}

    session.go_forward()
    assert session.current_record()["record_id"] == "rec3"
    assert not session.can_go_forward()  # at the end of the queue, nothing further to browse to

    session.go_back()
    session.go_back()
    assert session.current_record()["record_id"] == "rec1"
    assert not session.can_go_back()


def test_revisiting_and_redeciding_a_record_advances_past_the_original_frontier(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice"
    )
    session.record_decision("positive")  # rec1 -> positive, now on rec2
    session.go_back()  # back to rec1
    session.record_decision("negative")  # change mind on rec1 -> negative, advances to rec2 again

    assert session.current_record()["record_id"] == "rec2"
    events = pd.read_csv(tmp_path / "events.csv")
    rec1_events = events[events["record_id"] == "rec1"]
    assert list(rec1_events["decision"]) == ["positive", "negative"]  # both kept, append-only


def test_current_record_prior_decision_reflects_latest_this_session(tmp_path):
    dataset_path = _write_dataset(tmp_path / "canonical_dataset.csv")
    session = CurationSession(
        dataset_path=dataset_path, events_path=tmp_path / "events.csv", curator="alice"
    )
    assert session.current_record_prior_decision() is None  # never decided yet

    session.record_decision("positive")
    session.go_back()
    assert session.current_record_prior_decision() == "positive"

    session.record_decision("negative")  # overwrite while revisiting
    session.go_back()
    assert session.current_record_prior_decision() == "negative"


# ---------------------------------------------------------------------------------------------
# Bulk-pool browsing: dataset_df override, min_score/max_score, sort_by_score_desc -- the
# mechanics behind the Curate page's "Full AI/ML bulk pool" queue source (curate/bulk_pool.py
# builds the real dataset_df; these tests use a small in-memory frame with the same shape).
# ---------------------------------------------------------------------------------------------
def _write_bulk_pool_df():
    return pd.DataFrame(
        {
            "record_id": ["poolA", "poolB", "poolC", "poolD"],
            "title": ["Highest", "Second", "Third", "Lowest"],
            "abstract": ["a.", "b.", "c.", "d."],
            "journal": ["J1", "J1", "J2", "J2"],
            "year": ["2020", "2021", "2022", "2023"],
            "label": ["unlabeled", "unlabeled", "unlabeled", "unlabeled"],
            "label_confidence": [None, None, None, None],
            "pmcid": ["", "", "", ""],
            "bulk_match_score": [300.0, 200.0, 100.0, 50.0],
            "bulk_match_classification": ["positive", "positive", "negative", "negative"],
        }
    )


def test_dataset_df_override_is_used_instead_of_reading_dataset_path(tmp_path):
    # dataset_path points at a file that doesn't exist -- if __post_init__ tried to read it, this
    # would raise. It must not: dataset_df takes priority.
    session = CurationSession(
        dataset_path=tmp_path / "does_not_exist.csv",
        dataset_df=_write_bulk_pool_df(),
        events_path=tmp_path / "events.csv",
        curator="alice",
    )
    assert session.total() == 4
    assert session.current_record()["record_id"] == "poolA"


def test_sort_by_score_desc_orders_queue_by_bulk_match_score_descending(tmp_path):
    df = _write_bulk_pool_df().iloc[[3, 1, 0, 2]].reset_index(drop=True)  # shuffle on-disk order
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
        sort_by_score_desc=True,
    )
    assert session.queue == ["poolA", "poolB", "poolC", "poolD"]  # 300 -> 200 -> 100 -> 50


def test_without_sort_by_score_desc_queue_keeps_dataframe_order(tmp_path):
    df = _write_bulk_pool_df().iloc[[3, 1, 0, 2]].reset_index(drop=True)
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
    )
    assert session.queue == ["poolD", "poolB", "poolA", "poolC"]


def test_sort_by_score_asc_orders_queue_by_bulk_match_score_ascending(tmp_path):
    df = _write_bulk_pool_df().iloc[[3, 1, 0, 2]].reset_index(drop=True)
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
        sort_by_score_asc=True,
    )
    assert session.queue == ["poolD", "poolC", "poolB", "poolA"]  # 50 -> 100 -> 200 -> 300


def test_shuffle_seed_produces_a_deterministic_but_reordered_queue(tmp_path):
    df = _write_bulk_pool_df()
    session_a = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
        shuffle_seed=7,
    )
    session_b = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events2.csv",
        curator="alice",
        shuffle_seed=7,
    )
    assert sorted(session_a.queue) == ["poolA", "poolB", "poolC", "poolD"]  # nothing lost
    assert session_a.queue == session_b.queue  # same seed -> same order, reproducible
    assert session_a.queue != ["poolA", "poolB", "poolC", "poolD"]  # actually shuffled, not a no-op


def test_shuffle_seed_takes_priority_over_sort_by_score_desc(tmp_path):
    df = _write_bulk_pool_df()
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
        shuffle_seed=7,
        sort_by_score_desc=True,
    )
    assert session.queue != ["poolA", "poolB", "poolC", "poolD"]  # shuffle wins, not the score sort


def test_max_score_filter_excludes_records_scoring_above_it(tmp_path):
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=_write_bulk_pool_df(),
        events_path=tmp_path / "events.csv",
        curator="alice",
        max_score=150.0,
    )
    assert set(session.queue) == {"poolC", "poolD"}


def test_min_score_filter_excludes_records_scoring_below_it(tmp_path):
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=_write_bulk_pool_df(),
        events_path=tmp_path / "events.csv",
        curator="alice",
        min_score=150.0,
    )
    assert set(session.queue) == {"poolA", "poolB"}


def test_min_and_max_score_together_narrow_to_a_band(tmp_path):
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=_write_bulk_pool_df(),
        events_path=tmp_path / "events.csv",
        curator="alice",
        min_score=75.0,
        max_score=250.0,
    )
    assert set(session.queue) == {"poolB", "poolC"}


def test_bulk_pool_session_decision_writes_event_with_bulk_pool_record_id(tmp_path):
    # Confirms a decision made while browsing the bulk pool writes a real event -- the same event
    # log any other session writes to -- keyed on the bulk-pool row's own record_id.
    events_path = tmp_path / "events.csv"
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=_write_bulk_pool_df(),
        events_path=events_path,
        curator="alice",
        sort_by_score_desc=True,
    )
    session.record_decision("positive")

    events = pd.read_csv(events_path)
    assert events.iloc[0]["record_id"] == "poolA"
    assert events.iloc[0]["decision"] == "positive"


def test_current_record_uses_indexed_lookup_not_a_boolean_scan(tmp_path):
    # current_record() used to do `self.dataset.loc[self.dataset["record_id"] == record_id]` -- a
    # boolean-mask selection over the *full* wide frame. On the real ~745k-row bulk pool this
    # measured ~1.1GB of peak RSS on its first call and kept adding smaller-but-real,
    # never-reclaimed amounts on every subsequent click (see AGENTS.md's "Curate app performance"
    # section). Fixed via a `.set_index("record_id")`-backed lookup, memoized per instance. This
    # test can't reproduce the RSS difference on a tiny fixture, but pins the *correctness* of the
    # indexed path, including the duplicate-record_id case a plain unique-index lookup would break
    # on (two bulk-pool rows can legitimately share a record_id -- see bulk_pool.py).
    df = _write_bulk_pool_df()
    df.loc[len(df)] = df.loc[df["record_id"] == "poolB"].iloc[0]  # duplicate poolB's row
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
    )

    record = session.current_record()
    assert record["record_id"] == "poolA"

    session.go_forward()
    record = session.current_record()
    assert record["record_id"] == "poolB"
    assert record["title"] == "Second"


def test_current_record_returns_none_for_a_record_id_not_in_the_dataset(tmp_path):
    df = _write_bulk_pool_df()
    session = CurationSession(
        dataset_path=tmp_path / "unused.csv",
        dataset_df=df,
        events_path=tmp_path / "events.csv",
        curator="alice",
    )
    session.queue = ["not-a-real-record-id"]
    session._position = 0

    assert session.current_record() is None


# ---------------------------------------------------------------------------------------------
# materialize_events: inserting a decision made on a record that isn't in canonical_dataset.csv
# yet (reached by browsing the full bulk pool directly) -- the correctness-critical fix that
# makes bulk-pool curation safe. Without it these decisions would hit `if not mask.any():
# continue` and be silently discarded.
# ---------------------------------------------------------------------------------------------
def _write_bulk_pool_csv(path):
    from dome_triage.dedupe.keys import record_id_from_ids

    new_record_id = record_id_from_ids("PMC5000001", "", "")
    pd.DataFrame(
        {
            "source_name": ["bulk_match_2024"],
            "label": ["unlabeled"],
            "label_confidence": ["unscored"],
            "pmcid": ["PMC5000001"],
            "pmid": [""],
            "doi": [""],
            "title": ["A brand new bulk-pool paper"],
            "abstract": ["Applies a random forest to genomics data."],
            "journal": ["Bulk Journal"],
            "authors": ["Someone"],
            "year": ["2024"],
            "citation_count": ["0"],
            "mesh_headings": ["[]"],
            "pub_types": ['["Journal Article"]'],
            "is_open_access": ["True"],
            "keywords_author": ["[]"],
            "fulltext_available": ["False"],
        }
    ).to_csv(path, index=False)
    return path, new_record_id


def _write_empty_canonical_dataset(path):
    pd.DataFrame(
        {
            "record_id": ["rec1"],
            "title": ["Existing paper"],
            "abstract": ["Existing abstract."],
            "label": ["unlabeled"],
            "label_confidence": [None],
            "has_conflict": [False],
            "curation_tag": [None],
            "notes": [None],
            "updated_at": [None],
        }
    ).to_csv(path, index=False)
    return path


def test_materialize_events_inserts_a_new_row_for_a_bulk_pool_only_record(tmp_path):
    dataset_path = _write_empty_canonical_dataset(tmp_path / "canonical_dataset.csv")
    bulk_pool_path, new_record_id = _write_bulk_pool_csv(tmp_path / "bulk_candidates_scored.csv")
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": new_record_id,
                "decision": "positive",
                "tag": "",
                "notes": "found via bulk pool browsing",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "out.csv", bulk_pool_path=bulk_pool_path)

    assert len(result) == 2  # the original rec1, plus the newly-inserted record
    new_row = result[result["record_id"] == new_record_id].iloc[0]
    assert new_row["label"] == "positive"
    assert new_row["label_confidence"] == "human_curated"
    assert new_row["title"] == "A brand new bulk-pool paper"
    assert new_row["notes"] == "found via bulk pool browsing"


def test_materialize_events_new_row_record_id_matches_dedupe_pipeline(tmp_path):
    # The inserted row's record_id must be reproducible by a real `dedupe consolidate` run over
    # the same pmcid/pmid/doi later -- otherwise a subsequent Step 13 re-run could create a
    # duplicate row for the same paper instead of recognizing it as already present.
    from dome_triage.dedupe.keys import record_id_from_ids

    dataset_path = _write_empty_canonical_dataset(tmp_path / "canonical_dataset.csv")
    bulk_pool_path, new_record_id = _write_bulk_pool_csv(tmp_path / "bulk_candidates_scored.csv")
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": new_record_id,
                "decision": "negative",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "out.csv", bulk_pool_path=bulk_pool_path)

    assert new_record_id == record_id_from_ids("PMC5000001", "", "")
    assert new_record_id in set(result["record_id"])


def test_materialize_events_skips_gracefully_when_record_missing_from_both_dataset_and_bulk_pool(tmp_path, capsys):
    dataset_path = _write_empty_canonical_dataset(tmp_path / "canonical_dataset.csv")
    bulk_pool_path, _ = _write_bulk_pool_csv(tmp_path / "bulk_candidates_scored.csv")
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "totally-unknown-record-id",
                "decision": "positive",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "out.csv", bulk_pool_path=bulk_pool_path)

    assert len(result) == 1  # unchanged -- nothing to insert, nothing crashed
    assert "totally-unknown-record-id" not in set(result["record_id"])
    assert "WARNING" in capsys.readouterr().out  # the decision loss is surfaced, not silent


def test_materialize_events_without_bulk_pool_path_preserves_prior_silent_skip_behavior(tmp_path):
    # Backward compatibility: existing callers that don't pass bulk_pool_path (the default None)
    # must behave exactly as before this feature existed.
    dataset_path = _write_empty_canonical_dataset(tmp_path / "canonical_dataset.csv")
    events_path = tmp_path / "curation_events.csv"
    pd.DataFrame(
        [
            {
                "record_id": "some-record-not-in-dataset",
                "decision": "positive",
                "tag": "",
                "notes": "",
                "features": "",
                "curator": "alice",
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ]
    ).to_csv(events_path, index=False)

    result = materialize_events(dataset_path, events_path, tmp_path / "out.csv")

    assert len(result) == 1
