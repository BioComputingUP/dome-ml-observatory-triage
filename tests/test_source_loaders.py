from dome_triage.ingest.source_loaders import (
    load_curated_csv,
    load_dome_registry_api_json,
    load_id_pair_only,
    load_pdf_directory_gold,
)


def test_load_curated_csv_parses_schema_and_cleans_ids(fixtures_dir):
    source_cfg = {
        "name": "test_curated",
        "path": str(fixtures_dir / "curated_csv_sample.csv"),
        "label": "positive",
        "label_confidence": "human_curated",
    }
    records, unresolved = load_curated_csv(source_cfg)

    assert unresolved == []
    assert len(records) == 3

    first = records[0]
    assert first.pmid == "11111111"  # ".0" suffix stripped
    assert first.pmcid == "PMC1111111"
    assert first.doi == "10.1000/xyz111"
    assert first.year == 2022
    assert first.citation_count == 7
    assert first.label == "positive"
    assert first.label_confidence == "human_curated"
    assert first.match_metadata is not None
    assert first.match_metadata["matches"][0]["term"] == "learning"

    # third row has no PMID at all
    assert records[2].pmid is None
    assert records[2].pmcid == "PMC3333333"


def test_load_id_pair_only_handles_different_delimiters(fixtures_dir):
    tsv_cfg = {
        "name": "test_positive_pairs",
        "path": str(fixtures_dir / "id_pair_positive_sample.tsv"),
        "delimiter": "\t",
        "label": "positive",
        "label_confidence": "human_curated",
    }
    csv_cfg = {
        "name": "test_negative_pairs",
        "path": str(fixtures_dir / "id_pair_negative_sample.csv"),
        "delimiter": ",",
        "label": "negative",
        "label_confidence": "human_curated",
    }

    pos_records, pos_unresolved = load_id_pair_only(tsv_cfg)
    neg_records, neg_unresolved = load_id_pair_only(csv_cfg)

    assert pos_unresolved == [] and neg_unresolved == []
    assert len(pos_records) == 2
    assert len(neg_records) == 2
    assert pos_records[0].pmcid == "PMC1111111"
    assert pos_records[0].title is None  # no metadata yet -- needs enrich-metadata
    assert neg_records[0].label == "negative"


def test_load_pdf_directory_gold_flat_layout(fixtures_dir):
    source_cfg = {
        "name": "test_flat_gold",
        "path": str(fixtures_dir / "pdf_directory_gold_sample"),
        "pdf_layout": "flat_main_pdf",
        "label": "positive",
        "label_confidence": "registry_confirmed",
    }
    records, unresolved = load_pdf_directory_gold(source_cfg)

    assert unresolved == []
    pmcids = {r.pmcid for r in records}
    assert pmcids == {"PMC1111111", "PMC8888888"}
    assert all(r.fulltext_available for r in records)


def test_load_pdf_directory_gold_subdir_layout(fixtures_dir):
    source_cfg = {
        "name": "test_subdir_gold",
        "path": str(fixtures_dir / "pdf_directory_gold_subdir_sample"),
        "pdf_layout": "pmcid_subdir",
        "label": "positive",
        "label_confidence": "registry_confirmed",
    }
    records, unresolved = load_pdf_directory_gold(source_cfg)

    assert unresolved == []
    assert len(records) == 1
    assert records[0].pmcid == "PMC7777777"


def test_load_dome_registry_api_json_dedupes_overlapping_batches_and_routes_unresolved(
    fixtures_dir,
):
    source_cfg = {
        "name": "test_dome_api",
        "path": str(fixtures_dir / "dome_api_batches_sample.json"),
        "label": "positive",
        "label_confidence": "registry_confirmed",
    }
    records, unresolved = load_dome_registry_api_json(source_cfg)

    # "First Entry" and "First Two Entries" both contain entry_a -- must be deduped, not doubled.
    assert len(records) == 1
    assert records[0].pmid == "99999999"
    assert records[0].doi == "10.1000/aaa999"

    # entry_b has no usable pmid/doi -- routed to unresolved, not dropped or fuzzy-matched.
    assert len(unresolved) == 1
    assert unresolved[0]["_id"] == "entry_b"
