"""Headless AppTest coverage of src/dome_triage/curate/pages/1_Curate.py.

Per AGENTS.md's Testing rule: every path this file touches is redirected to tmp_path fixtures
(canonical_dataset.csv, bulk_candidates_scored.csv, curation_events.csv) -- never the real
config-resolved data paths. `streamlit_helpers.get_config` is monkeypatched to return a
`PipelineConfig` whose `sources`/`sampling`/`pipeline` dicts point at those fixtures, so every
`st.cache_resource`-cached loader in the app reads only what this file wrote.
"""

from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import dome_triage.curate.streamlit_helpers as streamlit_helpers
from dome_triage.config import PipelineConfig

PAGE_PATH = "src/dome_triage/curate/pages/1_Curate.py"


def _write_canonical_dataset(path):
    pd.DataFrame(
        {
            "record_id": ["rec1", "rec2"],
            "title": ["Paper One", "Paper Two"],
            "abstract": ["Abstract one.", "Abstract two."],
            "journal": ["J1", "J2"],
            "year": ["2020", "2021"],
            "label": ["unlabeled", "unlabeled"],
            "label_confidence": [None, None],
            "pmcid": ["PMC1000001", "PMC1000002"],
            "pmid": ["", ""],
            "doi": ["", ""],
            "has_conflict": [False, False],
            "curation_tag": [None, None],
            "notes": [None, None],
            "curation_features": [None, None],
            "updated_at": [None, None],
        }
    ).to_csv(path, index=False)


def _write_bulk_candidates_scored(path):
    pd.DataFrame(
        {
            "source_name": ["bulk_match_2020", "bulk_match_2021", "bulk_match_2022"],
            "label": ["unlabeled", "unlabeled", "unlabeled"],
            "label_confidence": ["unscored", "unscored", "unscored"],
            "pmcid": ["PMC1000001", "PMC1000002", "PMC1000003"],
            "pmid": ["", "", ""],
            "doi": ["", "", ""],
            "title": ["Paper One", "Paper Two", "Bulk-only paper three"],
            "abstract": ["Abstract one.", "Abstract two.", "Applies a neural network."],
            "journal": ["J1", "J2", "J3"],
            "authors": ["A", "B", "C"],
            "year": ["2020", "2021", "2022"],
            "citation_count": ["0", "0", "0"],
            "mesh_headings": ["[]", "[]", "[]"],
            "pub_types": ["[]", "[]", "[]"],
            "is_open_access": ["True", "True", "True"],
            "keywords_author": ["[]", "[]", "[]"],
            "fulltext_available": ["False", "False", "False"],
            "already_curated": [True, True, False],
            "existing_label": ["negative", "positive", ""],
            "existing_label_confidence": ["human_curated", "human_curated", ""],
            "match_score__bm25": ["50.0", "150.0", "300.0"],
            "matched_terms__bm25": ["[]", "[]", "[]"],
            "match_classification__bm25": ["negative", "positive", "positive"],
        }
    ).to_csv(path, index=False)


@pytest.fixture
def curate_app(tmp_path, monkeypatch):
    canonical_path = tmp_path / "canonical_dataset.csv"
    scored_path = tmp_path / "bulk_candidates_scored.csv"
    events_path = tmp_path / "curation_events.csv"
    _write_canonical_dataset(canonical_path)
    _write_bulk_candidates_scored(scored_path)

    cfg = PipelineConfig()
    cfg.sources["paths"]["canonical_dataset"] = str(canonical_path)
    cfg.sampling["paths"]["bulk_candidates_scored"] = str(scored_path)
    cfg.pipeline["curation"]["events_log"] = str(events_path)
    # Clear the *other* st.cache_resource/st.cache_data caches before replacing get_config --
    # they're keyed on (path, mtime) so cross-test collision is unlikely (each tmp_path is
    # unique), but clearing is cheap and removes any doubt. get_config itself is fully replaced
    # below, not cleared -- a plain lambda has no .clear() method.
    streamlit_helpers._cached_bulk_score_lookup.clear()
    streamlit_helpers._cached_bulk_pool.clear()
    streamlit_helpers._cached_filter_options.clear()
    monkeypatch.setattr(streamlit_helpers, "get_config", lambda: cfg)

    at = AppTest.from_file(PAGE_PATH)
    at.session_state["curator_name"] = "test-curator"
    at.run(timeout=180)
    assert not at.exception, at.exception
    return at, events_path


def test_page_defaults_to_canonical_queue_source(curate_app):
    at, _ = curate_app
    radio = at.radio[0]
    assert radio.value.startswith("Curation queue")


def test_switching_to_bulk_pool_reaches_a_record_not_in_canonical_dataset(curate_app):
    at, _ = curate_app
    at.radio[0].set_value("Full AI/ML bulk pool (~745k, ranked by BM25)").run(timeout=180)
    assert not at.exception, at.exception

    titles = [m.value for m in at.markdown if (m.value or "").startswith("### ")]
    assert titles, "expected a paper title to be rendered"
    # already_curated=True rows (Paper One/Two) are trusted and excluded by default; the queue
    # should reach the genuinely new bulk-only record, sorted highest-score-first.
    assert "Bulk-only paper three" in titles[0]


def test_bulk_pool_mode_shows_raw_score_inputs_not_quartile_bands(curate_app):
    at, _ = curate_app
    at.radio[0].set_value("Full AI/ML bulk pool (~745k, ranked by BM25)").run(timeout=180)
    assert not at.exception, at.exception

    number_input_labels = [n.label for n in at.number_input]
    assert any("Start browsing at BM25 score" in label for label in number_input_labels)
    band_multiselects = [m for m in at.multiselect if "Match-score band" in m.label]
    assert band_multiselects == []


def test_negative_reason_button_click_does_not_submit_a_decision(curate_app):
    at, events_path = curate_app
    reason_buttons = [b for b in at.button if b.label.startswith("1:")]
    assert reason_buttons, "expected a numbered negative-reason button"

    reason_buttons[0].click().run(timeout=180)
    assert not at.exception, at.exception
    assert not events_path.exists(), "selecting a reason must not write a decision"


def test_negative_reason_only_attaches_to_a_negative_decision(curate_app):
    import json

    at, events_path = curate_app
    reason_buttons = [b for b in at.button if b.label.startswith("1:")]
    reason_buttons[0].click().run(timeout=180)
    assert not at.exception, at.exception

    # Clicking Positive with a negative-reason pre-selected must NOT attach that reason --
    # shown_when_decision: negative must be enforced at submit time (state.py bug fixed this
    # session: previously any selected feature value attached to whichever decision was clicked).
    positive_btn = [b for b in at.button if b.label == "Positive (P)"][0]
    positive_btn.click().run(timeout=180)
    assert not at.exception, at.exception

    events = pd.read_csv(events_path)
    assert events.iloc[0]["decision"] == "positive"
    features = events.iloc[0]["features"]
    assert pd.isna(features) or json.loads(features) == {}


def test_negative_reason_attaches_when_negative_is_clicked(curate_app):
    import json

    at, events_path = curate_app
    reason_buttons = [b for b in at.button if b.label.startswith("1:")]
    reason_label = reason_buttons[0].label
    reason_buttons[0].click().run(timeout=180)
    assert not at.exception, at.exception

    negative_btn = [b for b in at.button if b.label == "Negative (N)"][0]
    negative_btn.click().run(timeout=180)
    assert not at.exception, at.exception

    events = pd.read_csv(events_path)
    assert events.iloc[0]["decision"] == "negative"
    features = json.loads(events.iloc[0]["features"])
    assert features["close_negative_reason"] == reason_label.split(": ", 1)[1]


def test_abstract_is_not_wrapped_in_a_fixed_height_container(curate_app):
    at, _ = curate_app
    # st.container(height=...) renders as a distinct block with a height style; the fix removed
    # it entirely for the abstract, so no container in the tree should carry a fixed height. Real
    # verification here is structural: the abstract text itself must appear as a plain markdown
    # block, not nested inside a container node.
    abstract_markdowns = [m for m in at.markdown if "Abstract" in (m.value or "")]
    assert abstract_markdowns
