from dome_triage.ingest.epmc_client import EpmcClient


class _FakeResponse:
    def __init__(self, json_data: dict):
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_data


def test_count_returns_hit_count_from_a_single_request(monkeypatch):
    client = EpmcClient()
    captured_params = {}

    def fake_get(url, params=None, timeout=None):
        captured_params.update(params)
        return _FakeResponse({"hitCount": 42})

    monkeypatch.setattr(client.session, "get", fake_get)

    result = client.count('"machine learning"')

    assert result == 42
    assert captured_params["query"] == '"machine learning"'
    assert captured_params["pageSize"] == 1
    assert captured_params["resultType"] == "idlist"


def test_count_defaults_to_zero_when_hit_count_missing(monkeypatch):
    client = EpmcClient()
    monkeypatch.setattr(client.session, "get", lambda url, params=None, timeout=None: _FakeResponse({}))

    assert client.count('"anything"') == 0


def test_get_by_ids_pmid_query_has_no_quotes_around_the_numeric_value(monkeypatch):
    """Regression test for a real, confirmed-live EPMC bug found while building Step 19c: a
    *quoted* EXT_ID value combined with `AND SRC:MED` silently returns zero hits when that chunk
    has exactly one id -- `(EXT_ID:"31501885") AND SRC:MED` returned 0 live hits against the real
    API even though the unquoted form and the unquoted-with-SRC:MED form both correctly found the
    record (BioBERT, PMID 31501885). See `get_by_ids`'s docstring for the full live-verified
    write-up. This locks in the fix: pmid values must never be quoted."""
    client = EpmcClient()
    captured_queries = []

    def fake_get(url, params=None, timeout=None):
        captured_queries.append(params["query"])
        return _FakeResponse({"resultList": {"result": []}})

    monkeypatch.setattr(client.session, "get", fake_get)

    client.get_by_ids(["31501885"], id_type="pmid")

    assert captured_queries == ["(EXT_ID:31501885) AND SRC:MED"]


def test_get_by_ids_pmid_query_batches_multiple_ids_unquoted(monkeypatch):
    client = EpmcClient()
    captured_queries = []

    def fake_get(url, params=None, timeout=None):
        captured_queries.append(params["query"])
        return _FakeResponse({"resultList": {"result": []}})

    monkeypatch.setattr(client.session, "get", fake_get)

    client.get_by_ids(["111", "222"], id_type="pmid")

    assert captured_queries == ["(EXT_ID:111 OR EXT_ID:222) AND SRC:MED"]


def test_get_by_ids_pmcid_and_doi_queries_still_quoted_and_unaffected(monkeypatch):
    """pmcid/doi lookups never append `AND SRC:MED`, so they never hit the specific broken
    combination above -- confirms the fix didn't touch their (already-working) query shape."""
    client = EpmcClient()
    captured_queries = []

    def fake_get(url, params=None, timeout=None):
        captured_queries.append(params["query"])
        return _FakeResponse({"resultList": {"result": []}})

    monkeypatch.setattr(client.session, "get", fake_get)

    client.get_by_ids(["PMC123"], id_type="pmcid")
    client.get_by_ids(["10.1038/s41586-021-03819-2"], id_type="doi")

    assert captured_queries == [
        'PMCID:"PMC123"',
        'DOI:"10.1038/s41586-021-03819-2"',
    ]


def test_get_by_ids_returns_found_records_keyed_by_id(monkeypatch):
    client = EpmcClient()

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse({"resultList": {"result": [{"pmid": "31501885", "title": "BioBERT"}]}})

    monkeypatch.setattr(client.session, "get", fake_get)

    found = client.get_by_ids(["31501885"], id_type="pmid")

    assert found == {"31501885": {"pmid": "31501885", "title": "BioBERT"}}
