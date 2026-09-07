from dome_triage.dedupe.conflicts import has_conflict, merge_label
from dome_triage.dedupe.consolidate import conflicts_dataframe, consolidate
from dome_triage.schema import RawRecord


def _record(**kwargs) -> RawRecord:
    defaults = {"source_file": "test.csv", "label_confidence": "human_curated"}
    defaults.update(kwargs)
    return RawRecord(**defaults)


def test_conflicting_labels_are_never_silently_resolved():
    cluster = [
        _record(source_name="src_a", pmcid="PMC5000001", label="positive", title="Paper X"),
        _record(source_name="src_b", pmcid="PMC5000001", label="negative", title="Paper X"),
    ]

    assert has_conflict(cluster) is True
    label, confidence = merge_label(cluster)
    assert label == "conflict"
    assert confidence is None


def test_skipped_alongside_definitive_label_is_not_a_conflict():
    cluster = [
        _record(source_name="src_a", pmcid="PMC5000002", label="positive"),
        _record(source_name="src_b", pmcid="PMC5000002", label="skipped"),
    ]

    assert has_conflict(cluster) is False
    label, confidence = merge_label(cluster)
    assert label == "positive"


def test_consolidate_flags_conflict_and_preserves_both_rows_for_review():
    records = [
        _record(
            source_name="dome_top_curate_positive",
            pmcid="PMC5000001",
            doi="10.1000/conflict",
            title="Disputed paper",
            abstract="Some abstract",
            label="positive",
        ),
        _record(
            source_name="copilot_1012_negative",
            pmcid="PMC5000001",
            doi="10.1000/conflict",
            title="Disputed paper",
            label="negative",
        ),
    ]

    canonical = consolidate(records)
    assert len(canonical) == 1

    conflict_record = canonical[0]
    assert conflict_record.label == "conflict"
    assert conflict_record.has_conflict is True
    assert conflict_record.source_count == 2

    review_df = conflicts_dataframe(canonical)
    assert len(review_df) == 2  # both contributing rows preserved, neither dropped
    assert set(review_df["source_label"]) == {"positive", "negative"}


def test_consolidate_does_not_flag_agreeing_sources_as_conflict():
    records = [
        _record(source_name="src_a", pmcid="PMC5000003", label="positive", title="Agreed paper"),
        _record(source_name="src_b", pmcid="PMC5000003", label="positive", title="Agreed paper"),
    ]

    canonical = consolidate(records)
    assert len(canonical) == 1
    assert canonical[0].label == "positive"
    assert canonical[0].has_conflict is False
