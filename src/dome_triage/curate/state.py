"""Pure-Python curation session state -- no Streamlit/ipywidgets import, so it's unit-testable
without a browser (see tests/test_curate_state.py).

Resume behavior follows DOME_Top_Curate/curation.ipynb's proven pattern: "already decided" is
determined by presence in the event log and filtered out of the queue on load, rather than an
index checkpoint -- so a crashed session loses at most the one in-flight decision. The single
rolling backup-before-write (`backup_file`) also mirrors that notebook's `backup_file()` exactly.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from dome_triage.curate.bulk_scores import annotate_bulk_scores, annotate_screening
from dome_triage.curate.review_detector import annotate_non_methods
from dome_triage.dedupe.consolidate import to_dataframe
from dome_triage.dedupe.keys import canonical_key_from_ids, record_id_from_ids
from dome_triage.sampling.stratified import build_strata
from dome_triage.schema import CanonicalRecord, SourceProvenance

EVENT_COLUMNS = ["record_id", "decision", "tag", "notes", "features", "curator", "timestamp"]


def backup_file(path: Path) -> None:
    """Single rolling backup (overwrites any previous backup), matching curation.ipynb's
    backup_file(): copy the existing file aside before every write."""
    if path.exists():
        backup_path = path.with_name(f"{path.stem}_backup{path.suffix}")
        shutil.copy2(path, backup_path)


# A record already trusted at this confidence tier (from a prior curation round outside this
# session -- DOME_Top_Curate, the DOME registry API, etc.) is settled ground truth, not something
# that needs a fresh decision -- mirrors materialize_events' own "contradicts_trusted_prior" tier
# pairing below.
_TRUSTED_LABEL_CONFIDENCE = ("human_curated", "registry_confirmed")


@dataclass
class CurationSession:
    dataset_path: Path
    events_path: Path
    curator: str = "unknown"
    include_already_labeled: bool = False
    # Mutually exclusive with the *exclusion* half of include_already_labeled -- when True, the
    # queue is filtered DOWN to just the trusted records (for deliberately re-checking a past
    # decision) instead of either hiding or merely allowing them alongside everything else. Takes
    # precedence over include_already_labeled in _scored_pool() (see there).
    only_already_labeled: bool = False
    require_pmcid: bool = False

    # Pre-built dataset override -- when given, __post_init__ uses this DataFrame directly instead
    # of `pd.read_csv(dataset_path, dtype=str)`. `dataset_path` is still required and still shown
    # for provenance (e.g. set to the bulk pool's own path), it just isn't re-read from disk. This
    # is what lets a session browse `curate/bulk_pool.py::load_bulk_pool()`'s ~745k-row frame
    # (built once, cached by the caller) instead of always reading canonical_dataset.csv -- see
    # streamlit_helpers.py's `queue_source` param.
    dataset_df: Optional[pd.DataFrame] = None

    # Optional lookups (see curate/bulk_scores.py) joining Step 12/14b's bulk-file-only columns
    # onto this session's rows for filtering/display -- canonical_dataset.csv itself never gains
    # these columns (see bulk_scores.py's module docstring for why). Built once by the caller
    # (streamlit_helpers.py caches them independently of these filter params) and passed in by
    # reference, so CurationSession never has to know how to read a 744k-row file.
    bulk_score_lookup: Optional[dict] = None
    screening_lookup: Optional[dict] = None

    # Filters -- each None/False means "no restriction", matching include_already_labeled/
    # require_pmcid's existing on/off-toggle pattern. score_band values are the integer band
    # labels build_strata() produces (0 = lowest quartile ... n_score_bands-1 = highest).
    # `journals` is a list of exact journal names (matched directly against the `journal` column,
    # not the top-N-or-"other" journal_bucket concept -- a curator searching for a specific
    # journal by name needs an exact match against the *real* value, not a bucketed one).
    score_band: Optional[list] = None
    journals: Optional[list] = None
    year_range: Optional[tuple] = None
    classification: Optional[list] = None
    needs_screening_only: bool = False
    n_score_bands: int = 4
    top_n_journals: int = 15

    # Raw-score bounds -- distinct from `score_band` (which restricts to whole quartiles of
    # *whatever population is currently filtered*). These are an absolute BM25 figure the curator
    # picked directly (e.g. "show me everything scoring at most 250"), for browsing the full bulk
    # pool ranked by score rather than being confined to Step 13's pre-stratified queue. Usable in
    # either queue source, though only meaningful once a `bulk_match_score` column exists.
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    # When True, the queue is ordered by `bulk_match_score` descending (highest-confidence AI/ML
    # match first) instead of the dataset's on-disk row order. Requires `bulk_match_score` to
    # already be a native column of `dataset`/`dataset_df` at construction time (true for
    # `bulk_pool.py::load_bulk_pool()`'s output; NOT guaranteed for canonical_dataset.csv, whose
    # score column is only populated inside `_scored_pool()` after a lookup join -- so this flag
    # is meaningful for bulk-pool sessions specifically).
    sort_by_score_desc: bool = False
    # Same as sort_by_score_desc but lowest-scoring first -- a separate flag rather than a single
    # "asc"/"desc" enum, so existing callers passing sort_by_score_desc=True keep working
    # unchanged. Mutually exclusive with sort_by_score_desc in practice (a caller sets at most
    # one); if both are somehow set, sort_by_score_desc wins (checked first below).
    sort_by_score_asc: bool = False
    # Randomizes queue order (Fisher-Yates via pandas' `.sample(frac=1, random_state=seed)`,
    # NumPy's Mersenne-Twister-backed PCG64 generator under the hood) -- takes priority over both
    # score-sort flags above when set, since "give me a random order" and "give me a score order"
    # are contradictory requests and random is the more specific ask when both are passed. None
    # (the default) leaves the dataset's on-disk row order untouched, matching every existing
    # caller's current behavior exactly.
    shuffle_seed: Optional[int] = None

    dataset: pd.DataFrame = field(init=False, repr=False)
    events: pd.DataFrame = field(init=False, repr=False)
    queue: list = field(init=False, repr=False)
    _reviewable_ids: set = field(init=False, repr=False)
    _position: int = field(init=False, default=0)
    # Furthest point the queue has ever reached -- distinct from `_position`, which can move
    # backward/forward freely once you're revisiting records you've already decided this session
    # (see go_back()/go_forward()). `stats()`/`remaining()` are driven by `_frontier`, not
    # `_position`, so browsing back through past decisions doesn't make "remaining" fluctuate.
    _frontier: int = field(init=False, default=0)
    # Memoizes _scored_pool() per instance -- it's called from __post_init__,
    # score_band_summary(), and year_bounds() independently, and was measured recomputing the
    # same result up to ~6 times in a single Curate-page rerun before this existed (see
    # _scored_pool()'s own docstring for the incident).
    _scored_pool_cache: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)
    # Memoizes _dataset_by_id() -- see that method's docstring for the incident this fixes.
    _dataset_by_id_cache: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.dataset = (
            self.dataset_df if self.dataset_df is not None else pd.read_csv(self.dataset_path, dtype=str)
        )
        self.events = (
            pd.read_csv(self.events_path, dtype=str)
            if Path(self.events_path).exists()
            else pd.DataFrame(columns=EVENT_COLUMNS)
        )
        decided = set(self.events["record_id"]) if not self.events.empty else set()

        reviewable = self._scored_pool()

        if self.score_band is not None:
            band_col = "match_score_band__bulk_match_score"
            reviewable = reviewable[
                reviewable[band_col].isin(self.score_band) if band_col in reviewable.columns else False
            ]
        if self.journals is not None:
            reviewable = reviewable[reviewable["journal"].isin(self.journals)]
        if self.year_range is not None:
            year_numeric = pd.to_numeric(reviewable["year"], errors="coerce")
            lo, hi = self.year_range
            reviewable = reviewable[(year_numeric >= lo) & (year_numeric <= hi)]

        self._reviewable_ids = set(reviewable["record_id"])

        if self.shuffle_seed is not None:
            # `.sample(frac=1, ...)` on a single Series (record_id only), never the full wide
            # frame -- same "don't copy the wide frame" reasoning as the score-sort branch below.
            ordered_ids = self.dataset["record_id"].sample(
                frac=1, random_state=self.shuffle_seed
            ).reset_index(drop=True)
        elif (
            self.sort_by_score_desc or self.sort_by_score_asc
        ) and "bulk_match_score" in self.dataset.columns:
            # Sort via a throwaway 2-column frame (record_id + score), never `self.dataset.assign
            # (...)` -- `.assign()` on the *full* wide frame copies every column, including
            # `abstract` (the ~745k-row bulk pool's single biggest column, ~1.1GB of the ~1.4GB
            # total). A real, measured incident: this ran on every `__post_init__` call (every
            # rerun in bulk-pool mode, since `build_probe_session()` -- see streamlit_helpers.py
            # -- rebuilds fresh each time), and each copy's peak RSS isn't reclaimed afterward
            # (glibc doesn't return freed heap arenas to the OS) -- the compounding cost across a
            # real browsing session, not any single call, is what eventually OOM-killed the
            # container. See `_scored_pool()`'s own docstring for the same lesson applied there.
            sort_frame = pd.DataFrame(
                {
                    "record_id": self.dataset["record_id"],
                    "_score": pd.to_numeric(self.dataset["bulk_match_score"], errors="coerce"),
                }
            )
            ordered_ids = sort_frame.sort_values(
                "_score", ascending=self.sort_by_score_asc, na_position="last"
            )["record_id"]
        else:
            ordered_ids = self.dataset["record_id"]

        self.queue = [rid for rid in ordered_ids if rid in self._reviewable_ids and rid not in decided]
        self._position = 0
        self._frontier = 0

    # Every column `_scored_pool()`'s own logic, `build_strata()`, `score_band_summary()`, and
    # `year_bounds()` actually read -- deliberately excludes `title`/`abstract`/`mesh_headings`
    # (and anything else display-only), which `current_record()` reads directly from
    # `self.dataset` instead, never from this method's return value. See `_scored_pool()`'s
    # docstring for why this projection exists.
    _SCORED_POOL_COLUMNS = (
        "record_id",
        "journal",
        "year",
        "label",
        "label_confidence",
        "pmcid",
        "pmid",
        "doi",
        "bulk_match_score",
        "bulk_match_classification",
        "needs_screening",
    )

    def _scored_pool(self) -> pd.DataFrame:
        """Applies every filter *except* score_band/journals/year_range, and adds the score-band
        column (via build_strata, unconditionally -- not just when score_band filtering is
        active) -- used by __post_init__ (which goes on to apply the remaining three filters),
        score_band_summary(), and year_bounds(). Memoized on `_scored_pool_cache` -- a real,
        measured incident: before this cache existed, those three call sites each independently
        recomputed the same result (and `annotate_bulk_scores` was independently expensive too,
        see its own docstring), measured at ~6 full recomputations for a single Curate-page
        rerun, directly responsible for a 40-58s per-decision reload.

        Works on a *slim projection* of `self.dataset` (`_SCORED_POOL_COLUMNS`), not the full
        wide frame -- a second, related, real incident: on the ~745k-row bulk pool, every
        `reviewable[mask]` boolean-filter step below forces pandas to copy the full row width
        (title/abstract/mesh_headings included -- `abstract` alone is ~1.1GB of that frame's
        ~1.4GB), and that peak RSS is not reclaimed afterward. Since none of this method's own
        logic or any of its three callers ever reads a text column from its *return value*
        (`current_record()` reads those directly from `self.dataset`, never from here), there was
        never a reason to carry them through this filtering chain at all. Confirmed via
        `resource.getrusage` against the real file: this took the OOM-killed combined-cache
        scenario from ~8.7GB down further once combined with the other fixes in this area (see
        AGENTS.md's "Curate app performance" section for the full incident writeup)."""
        if self._scored_pool_cache is not None:
            return self._scored_pool_cache

        # Every `reviewable[mask]` line is its own full-frame row-selection copy -- on the
        # ~745k-row bulk pool, even the slim projection above, chaining N of them (one per active
        # filter) means N separate peak-RSS spikes, each left un-reclaimed by glibc afterward (see
        # `_scored_pool()`'s docstring and AGENTS.md's "Curate app performance" section). Combined
        # into two AND-ed masks -- one before the score/screening joins add their columns, one
        # after -- so this pays for at most 2 copies instead of up to 6, regardless of how many
        # filters are actually active this call.
        reviewable = self.dataset[[c for c in self._SCORED_POOL_COLUMNS if c in self.dataset.columns]]

        pre_join_mask = pd.Series(True, index=reviewable.index)
        if self.only_already_labeled or not self.include_already_labeled:
            trusted_mask = reviewable["label_confidence"].isin(_TRUSTED_LABEL_CONFIDENCE) & reviewable[
                "label"
            ].isin(["positive", "negative"])
            pre_join_mask &= trusted_mask if self.only_already_labeled else ~trusted_mask
        if self.require_pmcid:
            pre_join_mask &= reviewable["pmcid"].notna() & (reviewable["pmcid"] != "")
        if not pre_join_mask.all():
            reviewable = reviewable[pre_join_mask]

        if self.bulk_score_lookup:
            reviewable = annotate_bulk_scores(reviewable, self.bulk_score_lookup)
        if self.screening_lookup:
            reviewable = annotate_screening(reviewable, self.screening_lookup)

        post_join_mask = pd.Series(True, index=reviewable.index)
        if self.classification is not None:
            post_join_mask &= reviewable.get("bulk_match_classification", pd.Series(dtype=str)).isin(
                self.classification
            )
        if self.needs_screening_only:
            screening_col = reviewable.get("needs_screening", pd.Series(False, index=reviewable.index))
            post_join_mask &= screening_col.fillna(False)
        if self.min_score is not None or self.max_score is not None:
            score_numeric = pd.to_numeric(reviewable.get("bulk_match_score"), errors="coerce")
            if self.min_score is not None:
                post_join_mask &= score_numeric >= self.min_score
            if self.max_score is not None:
                post_join_mask &= score_numeric <= self.max_score
        if not post_join_mask.all():
            reviewable = reviewable[post_join_mask]

        has_scores = "bulk_match_score" in reviewable.columns and reviewable["bulk_match_score"].notna().any()
        self._scored_pool_cache = build_strata(
            reviewable,
            score_col="bulk_match_score" if has_scores else None,
            n_score_bands=self.n_score_bands,
            top_n_journals=self.top_n_journals,
        )
        return self._scored_pool_cache

    def _confirmed_record_ids(self, candidate_ids) -> set:
        """Which of `candidate_ids` have a genuine positive/negative decision -- trusted from a
        prior round, or the *latest* event this session actually landing on positive/negative
        (not skipped/undeterminable, which mean "not yet assessed"/"looked and couldn't tell",
        neither of which is "curated" in the sense this is reporting). Used by
        `score_band_summary()` -- a real, live bug this fixes: it used to count *any* event
        (including Skip) as "confirmed", so a band containing a record you'd only Skipped this
        session (or a trusted-but-skipped record from a prior round) showed a nonzero curated
        count before you'd actually decided anything. `diversity_stats()` has its own,
        differently-shaped but equally-correct version of this same check (it also needs each
        confirmed record's *effective* label for grouping, not just membership) -- deliberately
        left as-is rather than forced through this helper too, since it's already correct and
        already tested."""
        candidate_ids = set(candidate_ids)
        trusted = set(
            self.dataset.loc[
                self.dataset["record_id"].isin(candidate_ids)
                & self.dataset["label_confidence"].isin(_TRUSTED_LABEL_CONFIDENCE)
                & self.dataset["label"].isin(["positive", "negative"]),
                "record_id",
            ]
        )
        decided = set()
        if not self.events.empty:
            latest = self.events.sort_values("timestamp").groupby("record_id").last()
            decided = set(latest[latest["decision"].isin(["positive", "negative"])].index)
        return (trusted | decided) & candidate_ids

    def score_band_summary(self) -> list[dict]:
        """One entry per score band (lowest to highest), for the filter widget's labels: the
        real numeric score range within that band, and how many of its records are already
        confirmed (a genuine positive/negative decision -- trusted, or decided this session) out
        of how many total -- so "Q1"/"Q4" stop being opaque and show real numbers instead.
        Computed fresh from `_scored_pool()` (the pre-score-band/journal/year-filtered
        population), so it doesn't shift confusingly just because a *different* band is currently
        selected."""
        pool = self._scored_pool()
        band_col = "match_score_band__bulk_match_score"
        if band_col not in pool.columns or pool.empty:
            return []

        confirmed_ids = self._confirmed_record_ids(pool["record_id"])

        summary = []
        for band in sorted(pool[band_col].dropna().unique()):
            band_df = pool[pool[band_col] == band]
            summary.append(
                {
                    "band": int(band),
                    "min_score": float(band_df["bulk_match_score"].min()),
                    "max_score": float(band_df["bulk_match_score"].max()),
                    "total": int(len(band_df)),
                    "confirmed": int(band_df["record_id"].isin(confirmed_ids).sum()),
                }
            )
        return summary

    def year_bounds(self) -> Optional[tuple]:
        """(min_year, max_year) across the *currently filtered* population (same base as
        score_band_summary() -- every filter except score_band/journals/year_range itself), not
        the whole dataset -- so the year slider's own range reflects what's actually left to
        pick from once you've, say, already filtered to BM25-classification=positive, rather than
        always spanning the full corpus regardless of what else is selected. Returns None if the
        filtered population has no parseable years at all (an edge case, not the common path)."""
        pool = self._scored_pool()
        years = pd.to_numeric(pool["year"], errors="coerce").dropna()
        if years.empty:
            return None
        return (int(years.min()), int(years.max()))

    def diversity_stats(self) -> dict:
        """All-time/corpus-wide diversity of *confirmed* (trusted-source OR actually decided in
        this session's events, even before `curate materialize` runs) positive/negative
        decisions -- deliberately NOT scoped to whichever filter is currently active (matches
        term_review_state.py's all_time_counts(): a decision's diversity contribution doesn't
        depend on which lens you were viewing it through when you made it). Overlays this
        session's in-memory events on top of self.dataset (replicating materialize_events()'s
        "last event wins" logic transiently) so today's decisions move the numbers immediately,
        not only after a separate materialize step. Excludes not-yet-reviewed
        heuristic_candidate/unscored rows from "confirmed" -- otherwise merging a large unreviewed
        clear-negative batch (Step 14) would inflate coverage before a human looked at any of it."""
        effective = self.dataset[["record_id", "journal", "year", "label", "label_confidence"]].copy()
        reviewed_ids: set = set()
        if not self.events.empty:
            latest = self.events.sort_values("timestamp").groupby("record_id").last()
            reviewed_ids = set(latest.index)
            effective = effective.set_index("record_id")
            common = latest.index.intersection(effective.index)
            effective.loc[common, "label"] = latest.loc[common, "decision"]
            effective = effective.reset_index()

        confirmed_mask = effective["label_confidence"].isin(_TRUSTED_LABEL_CONFIDENCE) | effective[
            "record_id"
        ].isin(reviewed_ids)
        confirmed = effective[effective["label"].isin(["positive", "negative"]) & confirmed_mask]

        total_journals = self.dataset["journal"].nunique(dropna=True)
        covered_journals = confirmed["journal"].nunique(dropna=True)

        return {
            "journal_coverage_pct": (covered_journals / total_journals * 100) if total_journals else 0.0,
            "n_journals_covered": int(covered_journals),
            "n_journals_total": int(total_journals),
            "per_journal_counts": confirmed.groupby(["journal", "label"]).size().unstack(fill_value=0),
            "per_year_counts": confirmed.groupby(["year", "label"]).size().unstack(fill_value=0),
        }

    def total(self) -> int:
        return len(self._reviewable_ids)

    def remaining(self) -> int:
        return len(self.queue) - self._frontier

    def stats(self) -> dict:
        total = self.total()
        remaining = self.remaining()
        return {"total": total, "decided": total - remaining, "remaining": remaining}

    def _dataset_by_id(self) -> pd.DataFrame:
        """`self.dataset` indexed by `record_id` (hash-based `.loc[]` lookup), built once and
        memoized on `_dataset_by_id_cache` -- `current_record()`'s only consumer. A real, measured
        incident: the previous implementation, `self.dataset.loc[self.dataset["record_id"] ==
        record_id]`, is a boolean-mask selection over the *full* wide frame (title/abstract/
        mesh_headings included) -- on the ~745k-row bulk pool this cost ~1.1GB of peak RSS on its
        first call and kept adding smaller-but-real, never-reclaimed amounts on every subsequent
        Forward/Back click, because boolean-mask selection on an object-dtype-heavy frame doesn't
        scale with the (tiny, usually 1-row) *output*, only with the frame's full width -- the
        exact same class of bug already documented for `_scored_pool()` and the bulk-pool sort in
        `__post_init__` (see AGENTS.md's "Curate app performance" section). `.set_index()` still
        pays a real, comparable cost, but only *once* per session construction -- which is now
        itself cached (`get_session()`/`build_probe_session()` in streamlit_helpers.py) -- instead
        of repeating on every single click for the session's entire lifetime."""
        if self._dataset_by_id_cache is None:
            self._dataset_by_id_cache = self.dataset.set_index("record_id", drop=False)
        return self._dataset_by_id_cache

    def current_record(self) -> Optional[pd.Series]:
        if self._position >= len(self.queue):
            return None
        record_id = self.queue[self._position]
        by_id = self._dataset_by_id()
        if record_id not in by_id.index:
            return None
        match = by_id.loc[record_id]
        # A non-unique index (two bulk-pool rows genuinely sharing a record_id -- see
        # bulk_pool.py's docstring) makes `.loc[record_id]` return a DataFrame instead of a
        # Series; take the first, same as the old boolean-mask code's `matches.iloc[0]`.
        return match.iloc[0] if isinstance(match, pd.DataFrame) else match

    def current_record_prior_decision(self) -> Optional[str]:
        """The latest decision already recorded this session for the record currently on screen,
        if any -- lets the UI show "you already marked this X" when backtracking, rather than
        presenting a blank slate as if it were never reviewed."""
        record = self.current_record()
        if record is None or self.events.empty:
            return None
        matches = self.events[self.events["record_id"] == record["record_id"]]
        if matches.empty:
            return None
        return matches.sort_values("timestamp").iloc[-1]["decision"]

    def can_go_back(self) -> bool:
        return self._position > 0

    def can_go_forward(self) -> bool:
        return self._position < len(self.queue) - 1

    def go_back(self) -> None:
        """Moves the viewing cursor back one record -- to revisit and, if you want, change a
        decision you already made this session. Never touches the event log by itself; only
        `record_decision()` writes anything."""
        self._position = max(0, self._position - 1)

    def go_forward(self) -> None:
        """Moves the viewing cursor forward one record -- works regardless of whether the record
        you're leaving (or the one you're moving to) has been decided yet; browsing is not gated
        on deciding. Bounded only by the queue's actual length -- `_frontier` (used by
        `remaining()`/`stats()`) is untouched by this, and only ever moves via a real decision in
        `record_decision()`, so browsing forward through undecided territory doesn't silently
        mark it as progress."""
        self._position = min(len(self.queue) - 1, self._position + 1)

    def record_decision(
        self,
        decision: str,
        tag: Optional[str] = None,
        notes: str = "",
        features: Optional[dict] = None,
    ) -> None:
        """Appends one row to the event log (backup-before-write, so nothing is lost even if the
        container closes mid-session) and advances the queue. `decision` should be one of
        "positive"/"negative"/"undeterminable"/"skipped". Works the same whether the current
        record is brand new or one you've backtracked to revisit -- a repeat decision for the same
        record_id is just another append; `materialize_events()`'s "last event wins" logic (and
        `diversity_stats()`'s live overlay) already handle picking the latest one. `features`
        holds the structured curation-feature flags (configs/curation_features.yaml) as a dict,
        stored as JSON -- a living, extensible checklist, not a fixed schema."""
        record = self.current_record()
        if record is None:
            return

        row = {
            "record_id": record["record_id"],
            "decision": decision,
            "tag": tag or "",
            "notes": notes,
            "features": json.dumps(features) if features else "",
            "curator": self.curator,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        events_path = Path(self.events_path)
        backup_file(events_path)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not events_path.exists()
        pd.DataFrame([row]).to_csv(events_path, mode="a", header=header_needed, index=False)

        self.events = pd.concat([self.events, pd.DataFrame([row])], ignore_index=True)
        self._position += 1
        self._frontier = max(self._frontier, self._position)


_BULK_POOL_INSERT_COLS = [
    "source_name",
    "label",
    "label_confidence",
    "pmcid",
    "pmid",
    "doi",
    "title",
    "abstract",
    "journal",
    "authors",
    "year",
    "citation_count",
    "mesh_headings",
    "pub_types",
    "is_open_access",
    "keywords_author",
    "fulltext_available",
]


def _json_list_or_empty(value) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _int_or_none(value) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _str_or_none(value) -> Optional[str]:
    """`pd.read_csv(..., dtype=str)` still represents a genuinely empty cell as float `nan`, not
    `""` -- `dtype` only constrains *non-null* values. A plain `value or None` is NOT a safe guard
    against that: `nan` is truthy in Python (`bool(float('nan'))` is `True`), so `nan or None`
    evaluates to `nan`, not `None` -- a real bug caught by this module's own tests (pydantic
    correctly rejected the resulting float where a CanonicalRecord field expects `Optional[str]`).
    `pd.isna` is the only reliable way to catch this."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value).strip()
    return value or None


def _insert_missing_records_from_bulk_pool(
    dataset: pd.DataFrame, missing_record_ids: set, bulk_pool_path: Path
) -> tuple[pd.DataFrame, set]:
    """A decision can be recorded (via `CurationSession.record_decision`) against a record_id that
    doesn't exist in `dataset` yet -- browsing the full bulk pool (Step 12's `bulk_candidates_
    scored.csv`, see `curate/bulk_pool.py`) reaches records that were never sampled into
    `canonical_dataset.csv` by Step 13. Without this, that decision would hit the `if not
    mask.any(): continue` branch below and be **silently discarded** -- a direct violation of
    AGENTS.md's "human curation is never bypassed" rule, and the reason this function exists.

    Looks each missing record_id up in the bulk pool (recomputing the same
    `dedupe.keys.record_id_from_ids` hash per row -- there's no `record_id` column on the raw bulk
    file itself) and, for every one found, builds a real `CanonicalRecord` from that row via the
    exact same construction `dedupe consolidate` would eventually produce for it (same
    `canonical_key`/`record_id`, `label="unlabeled"`/`label_confidence=None` -- matching
    `dedupe.conflicts.merge_label`'s convention for a fresh unlabeled cluster, not the raw
    `"unscored"` string the bulk file happens to carry), then appends it to `dataset` via
    `dedupe.consolidate.to_dataframe` (the same JSON-encoding used for every other canonical row,
    not a hand-rolled second encoder). The caller's existing per-event loop then runs unchanged --
    the newly-inserted row now has a real `label`/`label_confidence` for that loop's
    conflict-detection logic to compare the event's decision against, exactly like any
    already-existing row.

    Returns `(dataset_with_inserts, still_missing)` -- `still_missing` is whatever's left of
    `missing_record_ids` that wasn't found in the bulk pool either (e.g. it came from a queue
    source outside both files, or the bulk pool file has since been regenerated with different
    contents) so the caller can warn about it rather than fail silently a second way."""
    bulk_pool_path = Path(bulk_pool_path)
    if not missing_record_ids or not bulk_pool_path.exists():
        return dataset, set(missing_record_ids)

    usecols = [c for c in _BULK_POOL_INSERT_COLS if c in pd.read_csv(bulk_pool_path, nrows=0).columns]
    # fillna("") immediately after reading -- `pd.read_csv(..., dtype=str)` still represents a
    # genuinely empty cell as float `nan`, not `""` (dtype only constrains non-null values). Doing
    # this once here, rather than guarding every downstream `value or None`/`if value:` check
    # individually, is what makes those checks correct below (nan is truthy in Python -- an
    # unguarded `nan or None` evaluates to `nan`, not `None`; see `_str_or_none`'s docstring for
    # the real failure this caused before this fix).
    bulk = pd.read_csv(bulk_pool_path, usecols=usecols, dtype=str).fillna("")
    bulk["record_id"] = [
        record_id_from_ids(pmcid, pmid, doi) for pmcid, pmid, doi in zip(bulk["pmcid"], bulk["pmid"], bulk["doi"])
    ]
    bulk = bulk[bulk["record_id"].isin(missing_record_ids)].drop_duplicates(subset="record_id", keep="first")
    if bulk.empty:
        return dataset, set(missing_record_ids)

    now = datetime.now(timezone.utc).isoformat()
    new_records = []
    for _, row in bulk.iterrows():
        pmcid, pmid, doi = _str_or_none(row.get("pmcid")), _str_or_none(row.get("pmid")), _str_or_none(row.get("doi"))
        canonical_key = canonical_key_from_ids(pmcid, pmid, doi)
        matched_on = ",".join(
            field for field, value in (("pmcid", pmcid), ("doi", doi), ("pmid", pmid)) if value
        )
        new_records.append(
            CanonicalRecord(
                record_id=row["record_id"],
                canonical_key=canonical_key,
                pmcid=pmcid,
                pmid=pmid,
                doi=doi,
                title=_str_or_none(row.get("title")),
                abstract=_str_or_none(row.get("abstract")),
                journal=_str_or_none(row.get("journal")),
                authors=_str_or_none(row.get("authors")),
                year=_int_or_none(row.get("year")),
                citation_count=_int_or_none(row.get("citation_count")),
                label="unlabeled",
                label_confidence=None,
                sources=[
                    SourceProvenance(
                        source_name=_str_or_none(row.get("source_name")) or "bulk_match",
                        source_label=_str_or_none(row.get("label")) or "unlabeled",
                        source_label_confidence=_str_or_none(row.get("label_confidence")) or "unscored",
                        source_file=str(bulk_pool_path),
                        matched_on=matched_on or None,
                    )
                ],
                source_count=1,
                has_conflict=False,
                mesh_headings=_json_list_or_empty(row.get("mesh_headings")),
                pub_types=_json_list_or_empty(row.get("pub_types")),
                is_open_access=(row.get("is_open_access") == "True") if row.get("is_open_access") else None,
                keywords_author=_json_list_or_empty(row.get("keywords_author")),
                fulltext_available=row.get("fulltext_available") == "True",
                created_at=now,
                updated_at=now,
            )
        )

    new_rows = to_dataframe(new_records)
    combined = pd.concat([dataset, new_rows], ignore_index=True)
    still_missing = missing_record_ids - set(bulk["record_id"])
    return combined, still_missing


def materialize_events(
    dataset_path: Path, events_path: Path, output_path: Path, bulk_pool_path: Optional[Path] = None
) -> pd.DataFrame:
    """Folds curation_events.csv into the canonical dataset (last decision per record_id wins),
    flagging -- never silently overwriting -- any contradiction with a prior human_curated or
    registry_confirmed label. See AGENTS.md's "human curation is never bypassed" rule.

    A decision that lands cleanly (no conflict) also upgrades that row's `label_confidence` to
    `human_curated` -- a decision recorded through the Curate app is a real human judgment, on par
    with any other human_curated source, not a lesser tier. Before this, a record that started as
    e.g. `heuristic_candidate` (a bulk-match/clear-negative candidate) stayed at that confidence
    forever even after a human reviewed it via the app, making `label_confidence` alone unable to
    answer "was this actually reviewed" -- this fixes that. Conflicted rows keep their prior
    confidence untouched (it's already trusted-tier by definition -- that's what made it a
    conflict in the first place; there's nothing to upgrade).

    `bulk_pool_path`, if given, recovers decisions made against a record_id that isn't in
    `dataset_path` yet -- see `_insert_missing_records_from_bulk_pool` above for why this matters
    (curating directly from the full bulk pool, not just Step 13's pre-sampled queue, reaches
    records `canonical_dataset.csv` has never seen)."""
    dataset = pd.read_csv(dataset_path, dtype=str)
    events_path = Path(events_path)
    if not events_path.exists():
        return dataset

    events = pd.read_csv(events_path, dtype=str)
    if events.empty:
        return dataset

    latest = events.sort_values("timestamp").groupby("record_id").last()

    missing_ids = set(latest.index) - set(dataset["record_id"])
    if missing_ids and bulk_pool_path is not None:
        dataset, still_missing = _insert_missing_records_from_bulk_pool(dataset, missing_ids, bulk_pool_path)
        for record_id in still_missing:
            print(
                f"materialize: WARNING -- record_id {record_id} has a decision in {events_path} but no "
                f"matching row in {dataset_path} or {bulk_pool_path}; that decision is being skipped.",
                flush=True,
            )
    elif missing_ids:
        for record_id in missing_ids:
            print(
                f"materialize: WARNING -- record_id {record_id} has a decision in {events_path} but no "
                f"matching row in {dataset_path} (no bulk_pool_path given to recover it); skipping.",
                flush=True,
            )

    for record_id, event in latest.iterrows():
        mask = dataset["record_id"] == record_id
        if not mask.any():
            continue

        prior_label = dataset.loc[mask, "label"].iloc[0]
        prior_confidence = dataset.loc[mask, "label_confidence"].iloc[0]
        new_label = event["decision"]

        contradicts_trusted_prior = (
            prior_confidence in ("human_curated", "registry_confirmed")
            and prior_label in ("positive", "negative")
            # "skipped" is included here (not just positive/negative/undeterminable) -- a real,
            # confirmed bug this fixes: without it, skipping an already-trusted positive/negative
            # record (e.g. Step 19's Original Cohort Review, where every record already has a
            # trusted prior label by construction) silently overwrote `label` to "skipped" with
            # `has_conflict` left False -- destroying the prior determination from `label` with
            # no flag anywhere that a human should look at it. Skipping an `unlabeled` record
            # (the common case on the main queue) is unaffected: prior_label isn't in
            # ("positive", "negative") there, so this check never engages for that case.
            and new_label in ("positive", "negative", "undeterminable", "skipped")
            and new_label != prior_label
        )

        dataset.loc[mask, "label"] = "conflict" if contradicts_trusted_prior else new_label
        dataset.loc[mask, "has_conflict"] = str(contradicts_trusted_prior)
        if not contradicts_trusted_prior:
            dataset.loc[mask, "label_confidence"] = "human_curated"
        dataset.loc[mask, "curation_tag"] = event.get("tag") or None
        dataset.loc[mask, "notes"] = event.get("notes") or None
        if event.get("features"):
            dataset.loc[mask, "curation_features"] = event["features"]
        dataset.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()

    output_path = Path(output_path)
    backup_file(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


ORIGINAL_COHORT_SECOND_REVIEW_COLUMNS = [
    "original_cohort_review_prior_label",
    "original_cohort_review_label",
    "original_cohort_review_agrees_with_prior",
    "original_cohort_review_curator",
    "original_cohort_review_timestamp",
    "original_cohort_review_tag",
    "original_cohort_review_notes",
]


def merge_original_cohort_second_review(
    dataset_path: Path, events_path: Path, output_path: Path
) -> pd.DataFrame:
    """Step 19b (additive variant, distinct from `materialize_events`): folds
    `original_cohort_review_events.csv` into `canonical_dataset.csv` as brand-new, clearly-named
    columns -- `label`, `label_confidence`, `has_conflict`, `sources`, `curation_tag`, `notes`, and
    `updated_at` are never read for writing and never touched by this function, deliberately.

    `materialize_events()` above (used by `curate materialize`/`curate
    materialize-original-cohort-review`) folds a decision straight into `label`, flagging --
    never silently overwriting -- any contradiction with a trusted prior label. For this specific
    re-review pass, the explicit instruction was stronger than that: don't mutate the original
    `label` at all, not even via the flag-a-conflict path -- keep the second review as its own,
    independently-visible set of columns, so both the original decision and this second-review
    decision sit side by side in the same row (e.g. for a future model-training step to decide how
    to reconcile them) and the original is provably untouched by this pass, not just
    conflict-guarded. `materialize_events()` is still there, unchanged, if/when a deliberate later
    decision is made to fold this into `label` -- this function makes no attempt at that.

    New columns (NaN/blank for every row this event log doesn't cover):
    - `original_cohort_review_prior_label`: this row's `label` value *at the moment of this
      merge*, snapshotted explicitly rather than left implicit in the (untouched) `label` column --
      so the pre-review value survives even if `label` is changed by some other process later
      (e.g. a future `materialize_events` run against this same event log).
    - `original_cohort_review_label`: the second-review decision itself
      (positive/negative/undeterminable/skipped).
    - `original_cohort_review_agrees_with_prior`: `"True"`/`"False"` string -- this project's
      existing CSV convention for boolean columns (see Step 13b's log for why a bare Python bool
      crashes this pandas build's `dtype=str` reads) -- whether the second-review decision matches
      the prior label.
    - `original_cohort_review_curator`, `_timestamp`, `_tag`, `_notes`: straight from the event
      log's last decision per `record_id` (same "last decision wins" semantics as
      `materialize_events`).
    """
    dataset = pd.read_csv(dataset_path, dtype=str)
    events_path = Path(events_path)
    if not events_path.exists():
        return dataset

    events = pd.read_csv(events_path, dtype=str)
    if events.empty:
        return dataset

    latest = events.sort_values("timestamp").groupby("record_id").last()

    for col in ORIGINAL_COHORT_SECOND_REVIEW_COLUMNS:
        if col not in dataset.columns:
            dataset[col] = pd.NA

    matched = 0
    unmatched: list[str] = []
    for record_id, event in latest.iterrows():
        mask = dataset["record_id"] == record_id
        if not mask.any():
            unmatched.append(str(record_id))
            continue

        prior_label = dataset.loc[mask, "label"].iloc[0]
        new_label = event["decision"]
        dataset.loc[mask, "original_cohort_review_prior_label"] = prior_label
        dataset.loc[mask, "original_cohort_review_label"] = new_label
        dataset.loc[mask, "original_cohort_review_agrees_with_prior"] = str(new_label == prior_label)
        dataset.loc[mask, "original_cohort_review_curator"] = event.get("curator") or None
        dataset.loc[mask, "original_cohort_review_timestamp"] = event.get("timestamp") or None
        dataset.loc[mask, "original_cohort_review_tag"] = event.get("tag") or None
        dataset.loc[mask, "original_cohort_review_notes"] = event.get("notes") or None
        matched += 1

    for record_id in unmatched:
        print(
            f"merge-original-cohort-second-review: WARNING -- record_id {record_id} has a decision "
            f"in {events_path} but no matching row in {dataset_path}; skipped.",
            flush=True,
        )

    output_path = Path(output_path)
    backup_file(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(
        f"merge-original-cohort-second-review: {matched} records matched and annotated with new "
        f"original_cohort_review_* columns ({len(unmatched)} skipped, no matching canonical row). "
        f"label/label_confidence/has_conflict/sources/curation_tag/notes/updated_at untouched.",
        flush=True,
    )
    return dataset


def flag_likely_reviews(dataset_path: Path, exclusionary_lexicon_path: Path, output_path: Path) -> pd.DataFrame:
    """Step 19d: runs the shared, NLTK-based non-methods detector
    (`curate/review_detector.py::annotate_non_methods`) over the WHOLE `canonical_dataset.csv` and
    adds two new, additive columns -- same "new columns only, original data untouched" pattern
    `merge_original_cohort_second_review` above established, applied dataset-wide rather than to
    one cohort's subset. This is the exact same detector `ingest/clear_negative_sampler.py::
    fetch_filtered_clear_negatives` uses to gate the new Step 19d negative batch -- "same
    classification" is literally the same function call, not two implementations that could drift.

    New columns:
    - `likely_review_or_non_methods`: `"True"`/`"False"` string (this project's established CSV
      boolean convention -- see Step 13b's log for why a bare Python bool crashes this pandas
      build's `dtype=str` reads).
    - `likely_review_or_non_methods_detail`: JSON string of `{"text_hits": [...],
      "pub_type_hits": [...]}`, for the same per-row transparency `match_metadata` already gives
      elsewhere in this dataset.

    `label`/`label_confidence`/`sources`/`has_conflict`/`curation_tag`/`notes`/`updated_at`/every
    other existing column: never read for writing, never touched -- this is purely a function of
    each row's own `title`/`abstract`/`pub_types`, no event log or human decision involved, so
    every row gets a value in one pass (no subset gating like the cohort-scoped merge above)."""
    dataset = pd.read_csv(dataset_path, dtype=str)
    flagged = annotate_non_methods(dataset, exclusionary_lexicon_path)
    flagged["likely_review_or_non_methods"] = flagged["likely_review_or_non_methods"].astype(str)
    flagged["likely_review_or_non_methods_detail"] = flagged["likely_review_or_non_methods_detail"].apply(json.dumps)

    output_path = Path(output_path)
    backup_file(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(output_path, index=False)

    n_flagged = int((flagged["likely_review_or_non_methods"] == "True").sum())
    print(
        f"flag-likely-reviews: {len(flagged)} rows scored, {n_flagged} flagged "
        f"likely_review_or_non_methods=True ({n_flagged / len(flagged) * 100:.1f}%). "
        f"label/label_confidence/sources/every other existing column untouched.",
        flush=True,
    )
    return flagged


CROSS_CURATE_RESOLVE_COLUMNS = [
    "cross_curate_prior_label",
    "cross_curate_prior_label_confidence",
    "cross_curate_flash_classification",
    "cross_curate_pro_classification",
    "cross_curate_final_label",
    "cross_curate_curator",
    "cross_curate_timestamp",
    "cross_curate_tag",
    "cross_curate_notes",
]


def _latest_primary_classification_by_tier(llm_events: pd.DataFrame) -> dict[str, pd.Series]:
    """{tier: Series of classification indexed by record_id}, latest PRIMARY-mode event per
    (record_id, tier) -- the same "last event wins" join `cohort_filters.py::
    build_disagreement_queue` already uses. Returns {} if `llm_events` is empty/has no primary
    rows (Step 20b can still run -- the two classification columns just stay blank)."""
    if llm_events.empty or "mode" not in llm_events.columns:
        return {}
    primary = llm_events[llm_events["mode"] == "primary"]
    result = {}
    for tier, group in primary.groupby("model_tier"):
        result[tier] = group.sort_values("timestamp").groupby("record_id").last()["classification"]
    return result


def materialize_cross_curate_resolutions(
    dataset_path: Path, events_path: Path, llm_events_path: Path, output_path: Path
) -> pd.DataFrame:
    """Step 20b: folds `cross_curate_resolution_events.csv` (the "Cross Curate Resolve" page's
    human tie-break decisions on Step 20 DeepSeek/human disagreements) into `canonical_dataset.csv`.

    **Deliberately NOT `materialize_events()`** -- that function is built for a *naive* second
    curation pass, where any decision contradicting a trusted prior label is inherently suspicious
    and gets flagged by overwriting `label` with the literal string `"conflict"`, never applying
    the new decision. Every record here already has a trusted prior label by construction (that's
    why it was in the Cross Curate Resolve queue at all) -- so `materialize_events()`'s rule would
    silently turn the *real, deliberate, informed* resolution of exactly that disagreement into the
    string `"conflict"` for any record whose final call differs from the original label, discarding
    the very decision this whole review pass exists to produce. A real, confirmed-live incident
    this avoids: verified directly that `materialize_events()` was never actually run for this
    event log for real (zero `label=="conflict"` rows exist anywhere in `canonical_dataset.csv`
    as of 2026-08-21) -- this function replaces the wrong one before first use, nothing to repair.

    Snapshots the pre-review state and both tiers' original classifications into new, additive
    `CROSS_CURATE_RESOLVE_COLUMNS` (same naming convention as `merge_original_cohort_second_review`
    above) -- `cross_curate_prior_label`/`_prior_label_confidence` preserve exactly what `label`/
    `label_confidence` were before this function ever touches them, `cross_curate_flash_
    classification`/`_pro_classification` record what each tier actually said (joined from
    `llm_classification_events.csv`, same join `cohort_filters.py::build_disagreement_queue`
    already performs -- blank if that tier never classified this record) -- then, unlike the purely
    additive `merge_original_cohort_second_review`, **this function deliberately DOES update the
    real `label`** to `cross_curate_final_label`: this is a genuine, informed, final human decision
    (the whole point of the review), not a second opinion that might spuriously conflict with the
    first. `label_confidence` is set to `"human_curated"` (a human just made a real judgment call
    here, upgraded the same way `materialize_events()` already upgrades a clean decision elsewhere
    -- see that function's own `heuristic_candidate` -> `human_curated` upgrade for the precedent).

    `has_conflict` is set to `"True"` only when `cross_curate_final_label != cross_curate_prior_
    label` -- i.e. a genuine relabeling happened here, worth a future glance -- `"False"` when the
    review confirmed the original label stood. **This is a different meaning of `has_conflict` than
    `materialize_events()` gives it elsewhere** (there, `"True"` means the label was NOT updated,
    held at the literal string `"conflict"` instead) -- worth remembering when reading this column
    on a cross-curate-reviewed row versus an ordinarily-materialized one."""
    dataset = pd.read_csv(dataset_path, dtype=str)
    events_path = Path(events_path)
    if not events_path.exists():
        return dataset

    events = pd.read_csv(events_path, dtype=str)
    if events.empty:
        return dataset

    llm_events_path = Path(llm_events_path)
    llm_events = pd.DataFrame()
    if llm_events_path.exists() and llm_events_path.stat().st_size > 0:
        llm_events = pd.read_csv(llm_events_path, dtype=str)
    classification_by_tier = _latest_primary_classification_by_tier(llm_events)

    latest = events.sort_values("timestamp").groupby("record_id").last()

    for col in CROSS_CURATE_RESOLVE_COLUMNS:
        if col not in dataset.columns:
            dataset[col] = pd.NA

    matched = 0
    relabeled = 0
    unmatched: list[str] = []
    for record_id, event in latest.iterrows():
        mask = dataset["record_id"] == record_id
        if not mask.any():
            unmatched.append(str(record_id))
            continue

        prior_label = dataset.loc[mask, "label"].iloc[0]
        prior_confidence = dataset.loc[mask, "label_confidence"].iloc[0]
        final_label = event["decision"]

        dataset.loc[mask, "cross_curate_prior_label"] = prior_label
        dataset.loc[mask, "cross_curate_prior_label_confidence"] = prior_confidence
        dataset.loc[mask, "cross_curate_flash_classification"] = classification_by_tier.get(
            "flash", pd.Series(dtype=str)
        ).get(record_id)
        dataset.loc[mask, "cross_curate_pro_classification"] = classification_by_tier.get(
            "pro", pd.Series(dtype=str)
        ).get(record_id)
        dataset.loc[mask, "cross_curate_final_label"] = final_label
        dataset.loc[mask, "cross_curate_curator"] = event.get("curator") or None
        dataset.loc[mask, "cross_curate_timestamp"] = event.get("timestamp") or None
        dataset.loc[mask, "cross_curate_tag"] = event.get("tag") or None
        dataset.loc[mask, "cross_curate_notes"] = event.get("notes") or None

        dataset.loc[mask, "label"] = final_label
        dataset.loc[mask, "label_confidence"] = "human_curated"
        is_relabel = final_label != prior_label
        dataset.loc[mask, "has_conflict"] = str(is_relabel)
        dataset.loc[mask, "updated_at"] = datetime.now(timezone.utc).isoformat()
        matched += 1
        if is_relabel:
            relabeled += 1

    for record_id in unmatched:
        print(
            f"materialize-cross-curate-resolutions: WARNING -- record_id {record_id} has a "
            f"decision in {events_path} but no matching row in {dataset_path}; skipped.",
            flush=True,
        )

    output_path = Path(output_path)
    backup_file(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(
        f"materialize-cross-curate-resolutions: {matched} records matched ({relabeled} actually "
        f"relabeled, {matched - relabeled} confirmed as-is), {len(unmatched)} skipped (no matching "
        f"canonical row). label/label_confidence updated to the final decision; prior state fully "
        f"preserved in the new cross_curate_* columns.",
        flush=True,
    )
    return dataset
