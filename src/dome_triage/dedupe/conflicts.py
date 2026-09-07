"""Conflict detection. A cluster is a hard conflict iff it contains >=2 distinct *definitive*
labels (positive vs negative) contributed by different sources -- `skipped` alongside a
definitive label is extra provenance, not a conflict. Conflicting clusters are never silently
resolved by majority vote, recency, or source-confidence ranking; see AGENTS.md.
"""

from __future__ import annotations

from dome_triage.schema import Label, LabelConfidence, RawRecord

_DEFINITIVE = {"positive", "negative"}
_CONFIDENCE_RANK = {"registry_confirmed": 2, "human_curated": 1, "heuristic_candidate": 0}


def has_conflict(cluster: list[RawRecord]) -> bool:
    definitive_labels = {r.label for r in cluster if r.label in _DEFINITIVE}
    return len(definitive_labels) > 1


def merge_label(cluster: list[RawRecord]) -> tuple[Label, LabelConfidence | None]:
    """Returns (label, label_confidence) for a cluster. `label_confidence` is only ever a
    display hint (the strongest confidence tier among agreeing sources) -- it is None whenever
    label == "conflict", since no single confidence applies to a disputed cluster."""
    if has_conflict(cluster):
        return "conflict", None

    definitive = [r for r in cluster if r.label in _DEFINITIVE]
    if definitive:
        label = definitive[0].label
        best = max(definitive, key=lambda r: _CONFIDENCE_RANK[r.label_confidence])
        return label, best.label_confidence

    skipped = [r for r in cluster if r.label == "skipped"]
    if skipped:
        return "skipped", skipped[0].label_confidence

    return "unlabeled", None
