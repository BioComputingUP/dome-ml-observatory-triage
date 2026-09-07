"""Canonical record schema for the consolidated labeled dataset (Phase 1 output).

See ROADMAP.md Phase 1 and AGENTS.md's provenance section before changing these fields --
`sources` is the audit trail that lets a human trace any label back to the file it came from.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

Label = Literal["positive", "negative", "undeterminable", "skipped", "conflict", "unlabeled"]
LabelConfidence = Literal["human_curated", "registry_confirmed", "heuristic_candidate", "unscored"]
CurationTag = Literal["uncertain", "close_negative"]


class SourceProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_name: str
    source_label: Label
    source_label_confidence: LabelConfidence
    source_file: str
    matched_on: Optional[str] = None  # which normalized ID field this row shared with the cluster


class CanonicalRecord(BaseModel):
    """One row of data/processed/canonical_dataset.csv."""

    record_id: str
    canonical_key: str

    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None

    title: Optional[str] = None
    abstract: Optional[str] = None
    journal: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None

    label: Label = "unlabeled"
    label_confidence: Optional[LabelConfidence] = None

    sources: list[SourceProvenance] = []
    source_count: int = 0
    has_conflict: bool = False

    match_metadata: Optional[dict] = None

    # Populated from Europe PMC's `core` result type by ingest/bulk_match.py -- optional because
    # older sources (Adapters A-D) predate this capture and MeSH is not guaranteed on every
    # record even when present (see ontology/mesh.py).
    mesh_headings: list[str] = []
    pub_types: list[str] = []
    is_open_access: Optional[bool] = None
    keywords_author: list[str] = []

    fulltext_available: bool = False
    fulltext_manifest_ref: Optional[str] = None

    curation_tag: Optional[CurationTag] = None
    notes: Optional[str] = None
    # Config-driven structured feature flags captured during curation (configs/curation_features.yaml)
    # -- an extensible checklist, not a fixed schema; see curate/state.py::CurationSession.record_decision.
    curation_features: Optional[dict] = None

    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RawRecord(BaseModel):
    """One row loaded from a source file before dedup/consolidation, i.e. data/interim/raw_records*.csv."""

    source_name: str
    source_file: str
    label: Label
    label_confidence: LabelConfidence

    pmcid: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None

    title: Optional[str] = None
    abstract: Optional[str] = None
    journal: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None

    match_metadata: Optional[dict] = None

    mesh_headings: list[str] = []
    pub_types: list[str] = []
    is_open_access: Optional[bool] = None
    keywords_author: list[str] = []

    fulltext_available: bool = False
    fulltext_source_root: Optional[str] = None
