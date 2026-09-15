"""The latest usable verdict per record from a classification event log.

A staged batch carries no classification column: its verdicts live in the event log the classify
step writes (`incoming_new_classification_events.csv`). The EBI Search route covers positives only,
so its fetch and its merge read the verdicts from here.

This mirrors `mongo_landscape_export/scripts/build_staged_documents.py::load_classifications` (the
two folders do not import each other): the log is append-only and a re-run retries parse errors as
new rows, so the last row whose classification is a real verdict is the verdict. Change both
together.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

VALID_CLASSIFICATIONS = ("positive", "negative", "undeterminable")


def latest_verdicts(path: Path) -> dict[str, str]:
    """record_id -> classification, for every record with a usable verdict."""
    latest: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            record_id = (row.get("record_id") or "").strip()
            verdict = (row.get("classification") or "").strip()
            if record_id and verdict in VALID_CLASSIFICATIONS:
                latest[record_id] = verdict
    return latest
