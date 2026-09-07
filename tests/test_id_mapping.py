from dome_triage.ingest.id_mapping import clean_doi, clean_pmcid, clean_pmid


def test_clean_pmid_strips_pandas_float_suffix():
    # Confirmed real dirt: positive_entries.csv stores PMID as "35582656.0"
    assert clean_pmid("35582656.0") == "35582656"
    assert clean_pmid("35582656") == "35582656"


def test_clean_pmid_handles_missing_values():
    assert clean_pmid(None) is None
    assert clean_pmid("") is None
    assert clean_pmid("nan") is None
    assert clean_pmid("-") is None


def test_clean_doi_strips_url_and_scheme_prefixes():
    assert clean_doi("https://doi.org/10.1038/ABC123") == "10.1038/abc123"
    assert clean_doi("doi:10.1038/ABC123") == "10.1038/abc123"
    assert clean_doi("10.1038/abc123") == "10.1038/abc123"


def test_clean_doi_handles_missing_values():
    assert clean_doi(None) is None
    assert clean_doi("-") is None


def test_clean_pmcid_normalizes_prefix_and_case():
    assert clean_pmcid("pmc1234567") == "PMC1234567"
    assert clean_pmcid("1234567") == "PMC1234567"
    assert clean_pmcid("PMC1234567") == "PMC1234567"


def test_clean_pmcid_handles_missing_values():
    assert clean_pmcid(None) is None
    assert clean_pmcid("nan") is None
