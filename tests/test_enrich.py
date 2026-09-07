import io

import pandas as pd

from dome_triage.ingest.enrich import enrich_missing_canonical_metadata


def _epmc_result(**overrides) -> dict:
    base = {
        "title": "Fetched title",
        "abstractText": "Fetched abstract.",
        "journalInfo": {"journal": {"title": "Fetched Journal"}},
        "authorString": "Someone A.",
        "pubYear": "2019",
        "doi": "10.1234/fetched",
        "pmcid": "PMC1234567",
        "pmid": "9999999",
        "meshHeadingList": {"meshHeading": [{"descriptorName": "Humans"}]},
        "pubTypeList": {"pubType": ["Journal Article"]},
        "keywordList": {"keyword": ["ai"]},
        "isOpenAccess": "Y",
        "inEPMC": "Y",
    }
    base.update(overrides)
    return base


class _FakeClient:
    """Mirrors EpmcClient.get_by_ids -- keyed lookup returning canned results, matching the
    fake-client pattern already used in test_clear_negative_sampler.py."""

    def __init__(self, by_type: dict[str, dict[str, dict]]):
        self.by_type = by_type
        self.calls: list[tuple[str, list[str]]] = []

    def get_by_ids(self, ids: list[str], id_type: str, show_progress: bool = False) -> dict[str, dict]:
        self.calls.append((id_type, list(ids)))
        found = self.by_type.get(id_type, {})
        return {i: found[i] for i in ids if i in found}


_COLUMNS = [
    "record_id", "pmcid", "pmid", "doi", "title", "abstract", "journal", "authors",
    "year", "mesh_headings", "pub_types", "keywords_author", "is_open_access",
    "fulltext_available", "updated_at",
]


def _row(**overrides) -> dict:
    base = dict.fromkeys(_COLUMNS)
    base["record_id"] = "r1"
    base.update(overrides)
    return base


def _canonical_df(rows: list[dict]) -> pd.DataFrame:
    """Round-trips through a real CSV, exactly like `step_ingest_backfill_canonical_metadata`
    does (`pd.read_csv(canonical_path, dtype=str)`) -- this pandas build backs `dtype=str`
    columns with pyarrow's strict ArrowStringArray, which rejects a bare Python int/bool
    assignment (`TypeError: Invalid value ... got 'int' instead`), a real crash hit live against
    canonical_dataset.csv. Building the test DataFrame directly via `pd.DataFrame(rows)` instead
    would default to a permissive `object` dtype and silently miss that failure mode."""
    csv_text = pd.DataFrame(rows, columns=_COLUMNS).to_csv(index=False)
    return pd.read_csv(io.StringIO(csv_text), dtype=str)


def test_fills_missing_abstract_via_pmcid_lookup():
    df = _canonical_df([_row(record_id="r1", pmcid="PMC1234567")])
    client = _FakeClient({"pmcid": {"PMC1234567": _epmc_result()}})

    enriched, stats = enrich_missing_canonical_metadata(df, client)

    assert stats == {"targeted": 1, "found": 1, "still_missing": 0}
    row = enriched.iloc[0]
    assert row["title"] == "Fetched title"
    assert row["abstract"] == "Fetched abstract."
    assert row["journal"] == "Fetched Journal"
    assert row["year"] == "2019"
    assert row["mesh_headings"] == '["Humans"]'
    assert row["is_open_access"] == "True"
    assert row["fulltext_available"] == "True"


def test_falls_back_to_pmid_then_doi_when_pmcid_lookup_misses():
    df = _canonical_df(
        [
            _row(record_id="r1", pmcid="PMC_NOT_FOUND", pmid="111"),
            _row(record_id="r2", pmid="222_NOT_FOUND", doi="10.1/found"),
        ]
    )
    client = _FakeClient(
        {
            "pmcid": {},
            "pmid": {"111": _epmc_result(title="Via PMID")},
            "doi": {"10.1/found": _epmc_result(title="Via DOI")},
        }
    )

    enriched, stats = enrich_missing_canonical_metadata(df, client)

    assert stats["found"] == 2
    assert stats["still_missing"] == 0
    titles = set(enriched["title"])
    assert titles == {"Via PMID", "Via DOI"}


def test_never_overwrites_an_already_populated_field():
    df = _canonical_df(
        [_row(record_id="r1", pmcid="PMC1234567", title="Existing title", abstract=None)]
    )
    client = _FakeClient({"pmcid": {"PMC1234567": _epmc_result(title="Should not be used")}})

    enriched, _ = enrich_missing_canonical_metadata(df, client)

    row = enriched.iloc[0]
    assert row["title"] == "Existing title"
    assert row["abstract"] == "Fetched abstract."


def test_rows_with_no_id_are_left_untouched_and_reported_still_missing():
    df = _canonical_df([_row(record_id="r1")])
    client = _FakeClient({})

    enriched, stats = enrich_missing_canonical_metadata(df, client)

    assert stats == {"targeted": 1, "found": 0, "still_missing": 1}
    assert pd.isna(enriched.iloc[0]["title"])


def test_rows_with_existing_title_and_abstract_are_never_targeted():
    df = _canonical_df([_row(record_id="r1", pmcid="PMC1234567", title="T", abstract="A")])
    client = _FakeClient({"pmcid": {"PMC1234567": _epmc_result()}})

    _, stats = enrich_missing_canonical_metadata(df, client)

    assert stats == {"targeted": 0, "found": 0, "still_missing": 0}
    assert client.calls == []
