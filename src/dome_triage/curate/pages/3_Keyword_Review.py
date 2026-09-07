"""Keyword lexicon curation checkpoint -- one term at a time, not a table (the ~40k-row
data_editor this replaces made real judgment calls impractical past a handful of rows).

Every term gets exactly one of three final decisions -- Positive / Negative / Irrelevant --
regardless of which pile surfaced it (curate/term_review_state.py):
- Positive -> feeds keyword_lexicon.csv.
- Negative -> an exclusionary/negative-tail term -> feeds keyword_lexicon_exclusionary.csv, which
  keywords/scoring.py's BM25/TF-IDF-cosine scorers can subtract as a penalty.
- Irrelevant -> useful as neither -> feeds keyword_lexicon_irrelevant.csv, for audit only.

Two piles to browse for queued review ("Positive candidates" / "Exclusionary candidates" below),
but the decision recorded for a queued term is NOT constrained to match the pile it came from --
a term surfaced as a positive candidate can still be marked Negative or Irrelevant if that's what
it turns out to be. A separate "Add a term manually" section lets you classify any term directly,
whether or not it was ever auto-extracted -- TF-IDF/KeyBERT won't catch everything.

Decisions are logged immediately to keyword_review_events.csv as you go (never batched, backed up
before every write). Nothing becomes a final lexicon file until
`dome-triage keywords materialize-lexicon` is run -- that's a separate, provenance-logged step.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dome_triage.config import resolve_path
from dome_triage.curate.streamlit_helpers import get_config, get_term_review_session
from dome_triage.keywords.lexicon import lexicon_stats

st.set_page_config(page_title="Keyword Review", layout="wide")
st.title("Keyword Review")

if not st.session_state.get("curator_name"):
    st.warning("Set your curator name on the Home page first.")
    st.stop()

cfg = get_config()
candidates_path = resolve_path(cfg.pipeline["keywords"]["candidates"])

if not candidates_path.exists():
    st.info("No keyword_lexicon_candidates.csv yet -- run `dome-triage keywords build-lexicon` first.")
    st.stop()


@st.cache_data
def _load_candidates(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


candidates_df = _load_candidates(str(candidates_path), candidates_path.stat().st_mtime)
defaults = cfg.pipeline.get("keyword_review", {})

mode_label = st.radio(
    "Browse", ["Positive candidates", "Exclusionary candidates"], horizontal=True
)
queue_source = "positive_candidates" if mode_label.startswith("Positive") else "exclusionary_candidates"

if queue_source == "positive_candidates":
    col1, col2, col3 = st.columns(3)
    min_discriminative_score = col1.number_input(
        "Min discriminative_score",
        value=float(defaults.get("default_min_discriminative_score", 0.0)),
        step=0.001,
        format="%.4f",
    )
    min_document_frequency = col2.number_input(
        "Min document_frequency (KeyBERT)",
        value=int(defaults.get("default_min_document_frequency", 1)),
        step=1,
    )
    max_terms = col3.number_input(
        "Max terms this session",
        value=int(defaults.get("default_max_terms_per_session", 500)),
        step=50,
        min_value=1,
    )
    max_discriminative_score = float(defaults.get("default_max_discriminative_score", -0.001))
    live_mask = (candidates_df["discriminative_score"] >= min_discriminative_score) | (
        candidates_df["document_frequency"] >= min_document_frequency
    )
else:
    col1, col2 = st.columns(2)
    max_discriminative_score = col1.number_input(
        "Max discriminative_score (negative cutoff)",
        value=float(defaults.get("default_max_discriminative_score", -0.001)),
        step=0.001,
        format="%.4f",
    )
    max_terms = col2.number_input(
        "Max terms this session",
        value=int(defaults.get("default_max_terms_per_session", 500)),
        step=50,
        min_value=1,
    )
    min_discriminative_score = float(defaults.get("default_min_discriminative_score", 0.0))
    min_document_frequency = float(defaults.get("default_min_document_frequency", 1))
    st.caption(
        "Exclusionary candidates are TF-IDF-only -- KeyBERT never ran over the negative corpus, "
        "so there's no document_frequency dimension here."
    )
    live_mask = candidates_df["discriminative_score"] <= max_discriminative_score

st.caption(f"{int(live_mask.sum())} candidate terms currently meet this threshold")

with st.expander("See counts at other thresholds"):
    st.dataframe(lexicon_stats(candidates_df), width="stretch")

session = get_term_review_session(
    queue_source=queue_source,
    min_discriminative_score=min_discriminative_score,
    min_document_frequency=min_document_frequency,
    max_discriminative_score=max_discriminative_score,
    max_terms=int(max_terms),
)

counts = session.all_time_counts()
st.divider()
count_cols = st.columns(3)
count_cols[0].metric("Positive (all time)", counts["positive"])
count_cols[1].metric("Negative (all time)", counts["negative"])
count_cols[2].metric("Irrelevant (all time)", counts["irrelevant"])

st.subheader("Add a term manually")
st.caption("Classify any term directly -- doesn't need to have been auto-extracted by TF-IDF/KeyBERT.")
if "manual_term_counter" not in st.session_state:
    st.session_state["manual_term_counter"] = 0

manual_term = st.text_input(
    "Term",
    key=f"manual_term_input_{st.session_state['manual_term_counter']}",
    placeholder="e.g. a specific method name to include, or a generic word you want excluded",
    label_visibility="collapsed",
)
manual_col1, manual_col2, manual_col3 = st.columns(3)


def _add_manual(decision: str) -> None:
    term = manual_term.strip()
    if not term:
        st.warning("Type a term first.")
        return
    session.record_manual_decision(term, decision)
    st.session_state["manual_term_counter"] += 1
    st.rerun()


if manual_col1.button("Add as Positive", width="stretch"):
    _add_manual("positive")
if manual_col2.button("Add as Negative", width="stretch"):
    _add_manual("negative")
if manual_col3.button("Add as Irrelevant", width="stretch"):
    _add_manual("irrelevant")

st.divider()

st.caption(f"{session.remaining()} of {session.total()} remaining in this session's queue")

term_row = session.current_term()
if term_row is None:
    st.success("No terms left in this session's queue.")
    st.caption("Widen the threshold or raise the term cap above to see more candidates.")
    st.stop()


def _fmt(value, spec: str) -> str:
    return format(value, spec) if pd.notna(value) else "—"


st.subheader(term_row["term"])
stat_cols = st.columns(5)
stat_cols[0].metric("discriminative_score", _fmt(term_row.get("discriminative_score"), ".4f"))
stat_cols[1].metric("document_frequency", _fmt(term_row.get("document_frequency"), ".0f"))
stat_cols[2].metric("keybert_score_mean", _fmt(term_row.get("keybert_score_mean"), ".4f"))
stat_cols[3].write(f"**seed_category**  \n{term_row.get('seed_category') or '—'}")
stat_cols[4].write(f"**source**  \n{term_row.get('source') or '—'}")

canonical_path = cfg.path("canonical_dataset")
if canonical_path.exists():

    @st.cache_resource
    def _load_corpus_texts(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str)
        return pd.DataFrame({"_text": df["title"].fillna("") + ". " + df["abstract"].fillna("")})

    corpus_df = _load_corpus_texts(str(canonical_path))
    term_str = str(term_row["term"])
    matches = corpus_df[corpus_df["_text"].str.contains(term_str, case=False, na=False, regex=False)]
    with st.expander(f"Example usage ({len(matches)} found in canonical_dataset.csv)"):
        if matches.empty:
            st.caption("No example usage found.")
        else:
            for text in matches["_text"].head(2):
                snippet = text if len(text) <= 500 else text[:500] + "..."
                st.caption(snippet)
else:
    st.caption("No example usage found (canonical_dataset.csv not built yet).")

notes_key = f"notes__{queue_source}__{term_row['term']}"
notes = st.text_area("Notes (optional)", key=notes_key)

positive_col, negative_col, irrelevant_col, skip_col = st.columns(4)
if positive_col.button("Positive", type="primary", width="stretch"):
    session.record_decision("positive", notes=notes)
    st.rerun()
if negative_col.button("Negative", width="stretch"):
    session.record_decision("negative", notes=notes)
    st.rerun()
if irrelevant_col.button("Irrelevant", width="stretch"):
    session.record_decision("irrelevant", notes=notes)
    st.rerun()
if skip_col.button("Skip (not logged, will reappear)", width="stretch"):
    session.skip_current()
    st.rerun()
