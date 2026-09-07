"""Pure-Python term-review session state for the keyword lexicon curation checkpoint (Step 8) --
no Streamlit import, unit-testable (see tests/test_term_review_state.py). Parallel to
curate/state.py's CurationSession, not merged into it: term review keys on `term` (not
`record_id`), uses a three-way decision vocabulary with no conflict concept, and can classify
terms that were never auto-extracted at all -- folding that into CurationSession would mean
branchy code for a structurally different domain.

Every term gets exactly one of three final decisions, independent of which candidate pile
surfaced it:
- "positive": feeds keyword_lexicon.csv.
- "negative": an exclusionary/negative-tail term -- feeds keyword_lexicon_exclusionary.csv, which
  keywords/scoring.py's scorers can subtract as a penalty.
- "irrelevant": neither useful as a positive nor a negative signal -- feeds
  keyword_lexicon_irrelevant.csv, purely for audit (never re-surfaced for review).

Two ways a term reaches a decision:
- Browsing a queue built from the extracted candidate file (`record_decision`), pulled from one
  of two piles: "positive_candidates" (top of discriminative_score/document_frequency) or
  "exclusionary_candidates" (the negative tail of discriminative_score -- TF-IDF only, since
  KeyBERT never ran over the negative corpus, see pipeline/steps.py::step_keywords_keybert).
  Browsing a pile is just a navigation aid -- the decision recorded isn't constrained to match it,
  since a term surfaced as a "positive candidate" might turn out to be genuinely exclusionary, or
  irrelevant to both.
- Directly naming an arbitrary term (`record_manual_decision`), whether or not it exists in the
  extracted candidates file at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from dome_triage.curate.state import backup_file

DECISIONS = ("positive", "negative", "irrelevant")

TERM_EVENT_COLUMNS = [
    "term",
    "decision",
    "source_queue",
    "notes",
    "discriminative_score",
    "document_frequency",
    "source",
    "curator",
    "timestamp",
]


def build_review_queue(
    candidates: pd.DataFrame,
    queue_source: str,
    already_decided: set,
    min_discriminative_score: float = 0.0,
    min_document_frequency: float = 1,
    max_discriminative_score: float = -0.001,
    max_terms: int = 500,
) -> pd.DataFrame:
    """Filters + sorts the candidate lexicon into a bounded review queue -- a navigation aid, not
    a constraint on what decision can be recorded for a term found this way.

    "positive_candidates": qualifies via discriminative_score OR document_frequency -- either the
    TF-IDF or the KeyBERT signal is enough, matching keywords/lexicon.py::lexicon_stats' own
    semantics -- sorted best-first.
    "exclusionary_candidates": discriminative_score below a negative cutoff only, sorted
    worst-first (most negative / most exclusionary-looking first).
    """
    if queue_source == "positive_candidates":
        mask = (candidates["discriminative_score"] >= min_discriminative_score) | (
            candidates["document_frequency"] >= min_document_frequency
        )
        pool = candidates[mask].sort_values(
            ["discriminative_score", "document_frequency"], ascending=False
        )
    elif queue_source == "exclusionary_candidates":
        mask = candidates["discriminative_score"] <= max_discriminative_score
        pool = candidates[mask].sort_values("discriminative_score", ascending=True)
    else:
        raise ValueError(
            f"Unknown queue_source: {queue_source!r} (expected 'positive_candidates' or "
            "'exclusionary_candidates')"
        )

    pool = pool[~pool["term"].isin(already_decided)]
    return pool.head(max_terms).reset_index(drop=True)


@dataclass
class TermReviewSession:
    candidates_path: Path
    events_path: Path
    queue_source: str
    curator: str = "unknown"
    min_discriminative_score: float = 0.0
    min_document_frequency: float = 1
    max_discriminative_score: float = -0.001
    max_terms: int = 500

    candidates: pd.DataFrame = field(init=False, repr=False)
    events: pd.DataFrame = field(init=False, repr=False)
    queue: pd.DataFrame = field(init=False, repr=False)
    _position: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.candidates = pd.read_csv(self.candidates_path)
        self.events = (
            pd.read_csv(self.events_path, dtype=str)
            if Path(self.events_path).exists()
            else pd.DataFrame(columns=TERM_EVENT_COLUMNS)
        )
        # Decided is GLOBAL, not scoped to this queue_source -- a term decided while browsing one
        # pile must never resurface in the other, since the decision now applies regardless of
        # which pile surfaced it.
        already_decided = set(self.events["term"]) if not self.events.empty else set()
        self.queue = build_review_queue(
            self.candidates,
            self.queue_source,
            already_decided,
            min_discriminative_score=self.min_discriminative_score,
            min_document_frequency=self.min_document_frequency,
            max_discriminative_score=self.max_discriminative_score,
            max_terms=self.max_terms,
        )
        self._position = 0

    def total(self) -> int:
        return len(self.queue)

    def remaining(self) -> int:
        return len(self.queue) - self._position

    def current_term(self) -> Optional[pd.Series]:
        if self._position >= len(self.queue):
            return None
        return self.queue.iloc[self._position]

    def skip_current(self) -> None:
        """Advances past the current term WITHOUT writing to the event log -- deliberately not a
        logged decision, so it isn't misrepresented as permanent. The term reappears the next
        time a TermReviewSession is freshly constructed (new session, or a threshold change)."""
        if self._position < len(self.queue):
            self._position += 1

    def all_time_counts(self) -> dict:
        """positive/negative/irrelevant counts across the WHOLE event log -- global, not scoped
        to this session's queue_source or current threshold, since a decision applies to the term
        regardless of which pile it came from."""
        if self.events.empty:
            return {d: 0 for d in DECISIONS}
        counts = self.events["decision"].value_counts()
        return {d: int(counts.get(d, 0)) for d in DECISIONS}

    def record_decision(self, decision: str, notes: str = "") -> None:
        """Records a decision for the current queued term and advances the queue. `decision` must
        be one of DECISIONS -- not constrained to match `queue_source` (a "positive_candidates"
        pile term can still be recorded "negative" or "irrelevant")."""
        term_row = self.current_term()
        if term_row is None:
            return
        self._write_event(
            term=term_row["term"],
            decision=decision,
            source_queue=self.queue_source,
            notes=notes,
            discriminative_score=term_row.get("discriminative_score"),
            document_frequency=term_row.get("document_frequency"),
            source=term_row.get("source"),
        )
        self._position += 1

    def record_manual_decision(self, term: str, decision: str, notes: str = "") -> None:
        """Classifies an arbitrary term the curator typed in directly -- whether or not it exists
        in the extracted candidates file. Does not touch the queue/position. If the term IS a
        known candidate, its stats are snapshotted the same way a queued decision would."""
        candidate_row = self.candidates[self.candidates["term"] == term]
        if not candidate_row.empty:
            row = candidate_row.iloc[0]
            discriminative_score = row.get("discriminative_score")
            document_frequency = row.get("document_frequency")
            source = row.get("source")
        else:
            discriminative_score = None
            document_frequency = None
            source = "manual"
        self._write_event(
            term=term,
            decision=decision,
            source_queue="manual",
            notes=notes,
            discriminative_score=discriminative_score,
            document_frequency=document_frequency,
            source=source,
        )

    def _write_event(
        self,
        term: str,
        decision: str,
        source_queue: str,
        notes: str,
        discriminative_score,
        document_frequency,
        source,
    ) -> None:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")

        row = {
            "term": term,
            "decision": decision,
            "source_queue": source_queue,
            "notes": notes,
            "discriminative_score": discriminative_score,
            "document_frequency": document_frequency,
            "source": source,
            "curator": self.curator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        events_path = Path(self.events_path)
        backup_file(events_path)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not events_path.exists()
        pd.DataFrame([row]).to_csv(events_path, mode="a", header=header_needed, index=False)

        self.events = pd.concat([self.events, pd.DataFrame([row]).astype(str)], ignore_index=True)


def materialize_term_events(
    candidates_path: Path,
    events_path: Path,
    lexicon_path: Path,
    exclusionary_path: Path,
    irrelevant_path: Path,
) -> dict:
    """Folds keyword_review_events.csv into the final lexicon files: last decision per term wins
    -- never silently overwritten -- regardless of which queue_source produced it. Output rows are
    built entirely from the event log's own snapshotted fields (term, discriminative_score,
    document_frequency, source, notes), not re-joined against the candidates file, so manually
    added terms that were never auto-extracted come through identically to extracted ones. Writes
    (each backed up first):
    - lexicon_path: decision == "positive"
    - exclusionary_path: decision == "negative"
    - irrelevant_path: decision == "irrelevant"
    - candidates_path: rewritten in place with a synced review_status column (only for terms that
      exist there -- manual-only terms have no row to sync) -- a no-join glance view, never a
      second write path for decisions.
    Returns counts for the caller's provenance notes.
    """
    candidates_path = Path(candidates_path)
    events_path = Path(events_path)
    lexicon_path = Path(lexicon_path)
    exclusionary_path = Path(exclusionary_path)
    irrelevant_path = Path(irrelevant_path)

    candidates = pd.read_csv(candidates_path)
    empty_counts = {d: 0 for d in DECISIONS}
    if not events_path.exists():
        return empty_counts

    events = pd.read_csv(events_path, dtype=str)
    if events.empty:
        return empty_counts

    latest = events.sort_values("timestamp").groupby("term").last().reset_index()
    latest["discriminative_score"] = pd.to_numeric(latest["discriminative_score"], errors="coerce")
    latest["document_frequency"] = pd.to_numeric(latest["document_frequency"], errors="coerce")

    output_cols = ["term", "discriminative_score", "document_frequency", "source", "notes"]
    outputs = {
        "positive": (lexicon_path, latest[latest["decision"] == "positive"][output_cols]),
        "negative": (exclusionary_path, latest[latest["decision"] == "negative"][output_cols]),
        "irrelevant": (irrelevant_path, latest[latest["decision"] == "irrelevant"][output_cols]),
    }

    for path, _ in outputs.values():
        backup_file(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    backup_file(candidates_path)

    for path, df in outputs.values():
        df.to_csv(path, index=False)

    status_map = dict(zip(latest["term"], latest["decision"]))
    candidates["review_status"] = candidates["term"].map(status_map).fillna("pending")
    candidates.to_csv(candidates_path, index=False)

    return {decision: len(df) for decision, (_, df) in outputs.items()}
