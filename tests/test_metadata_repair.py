import json

import pandas as pd
import pytest

from dome_triage.ingest.metadata_repair import (
    _Checkpoint,
    _safe_call,
    merge_repairs,
    scan_missing_abstracts,
)
from dome_triage.ingest.ncbi_client import NcbiClient, chunked

_PUBMED_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID Version="1">111</PMID>
   <Article>
    <Journal><Title>J Test</Title>
      <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
    </Journal>
    <ArticleTitle>A <i>nested</i> title</ArticleTitle>
    <Abstract>
      <AbstractText Label="BACKGROUND">Part one.</AbstractText>
      <AbstractText Label="RESULTS">Part two.</AbstractText>
    </Abstract>
    <AuthorList>
      <Author><LastName>Smith</LastName><Initials>J</Initials></Author>
    </AuthorList>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
    <ArticleId IdType="doi">10.1/x</ArticleId>
    <ArticleId IdType="pmc">PMC999</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""


def test_parses_structured_abstract_keeping_every_labelled_section():
    # Real failure mode this guards: taking only the first <AbstractText> silently truncates a
    # structured abstract down to its BACKGROUND section.
    parsed = NcbiClient._parse_pubmed_xml(_PUBMED_XML)
    assert parsed["111"]["abstract"] == "BACKGROUND: Part one.\nRESULTS: Part two."


def test_parses_title_through_nested_markup():
    # PubMed wraps <i>/<sup> inside titles; .text alone stops at the first child tag.
    assert NcbiClient._parse_pubmed_xml(_PUBMED_XML)["111"]["title"] == "A nested title"


def test_parses_journal_year_authors_and_ids():
    record = NcbiClient._parse_pubmed_xml(_PUBMED_XML)["111"]
    assert record["journal"] == "J Test"
    assert record["year"] == "2021"
    assert record["authors"] == "Smith J."
    assert record["doi"] == "10.1/x"
    assert record["pmcid"] == "PMC999"


def test_malformed_xml_returns_empty_rather_than_raising():
    assert NcbiClient._parse_pubmed_xml(b"<not xml") == {}


def test_chunked_splits_to_requested_size_including_the_short_tail():
    assert list(chunked(["a", "b", "c"], size=2)) == [["a", "b"], ["c"]]


def test_scan_buckets_rows_by_which_identifier_can_repair_them(tmp_path):
    csv_path = tmp_path / "pool.csv"
    pd.DataFrame(
        [
            {"pmcid": "", "pmid": "111", "doi": "", "abstract": ""},        # direct pmid
            {"pmcid": "PMC2", "pmid": "", "doi": "", "abstract": ""},       # needs conversion
            {"pmcid": "", "pmid": "", "doi": "10.1/z", "abstract": ""},     # doi only
            {"pmcid": "", "pmid": "", "doi": "", "abstract": ""},           # unrepairable
            {"pmcid": "", "pmid": "999", "doi": "", "abstract": "Has one"},  # not missing
        ]
    ).to_csv(csv_path, index=False)

    scan = scan_missing_abstracts(csv_path)
    assert scan["n_rows"] == 5
    assert scan["n_missing_abstract"] == 4
    assert scan["pmids"] == ["111"]
    assert scan["pmcids_needing_pmid"] == ["PMC2"]
    assert scan["dois_needing_pmid"] == ["10.1/z"]
    assert scan["n_unrepairable"] == 1


def test_checkpoint_roundtrips_done_ids_and_survives_a_torn_line(tmp_path):
    path = tmp_path / "cp.jsonl"
    cp = _Checkpoint(path)
    with cp as handle:
        handle.write_many([{"query_id": "a", "found": True}, {"query_id": "b", "found": False}])
    # Simulate a kill -9 mid-write leaving a partial final line.
    with path.open("a") as fh:
        fh.write('{"query_id": "c"')
    assert _Checkpoint(path).load_done() == {"a", "b"}


def test_not_found_ids_are_recorded_so_a_resume_never_retries_them_forever(tmp_path):
    path = tmp_path / "cp.jsonl"
    with _Checkpoint(path) as cp:
        cp.write_many([{"query_id": "missing-in-pubmed", "found": False}])
    assert "missing-in-pubmed" in _Checkpoint(path).load_done()


def test_safe_call_returns_none_instead_of_propagating(capsys):
    # Real, confirmed incident: a single failing batch raised out of the worker and aborted the
    # entire unattended run. One bad batch must degrade to "these ids weren't found", never a crash.
    def boom():
        raise RuntimeError("400 Bad Request")

    assert _safe_call(boom, ["a", "b"], "convert") is None
    assert "failed" in capsys.readouterr().out


def test_safe_call_passes_a_successful_result_straight_through():
    assert _safe_call(lambda: {"a": 1}, ["a"], "efetch") == {"a": 1}


def _write_checkpoint(path, rows):
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_merge_fills_blank_abstract_by_pmid(tmp_path):
    csv_path = tmp_path / "pool.csv"
    pd.DataFrame(
        [{"pmcid": "", "pmid": "111", "doi": "", "title": "T", "abstract": "",
          "journal": "", "year": "", "authors": ""}]
    ).to_csv(csv_path, index=False)
    cp = tmp_path / "records.jsonl"
    _write_checkpoint(cp, [{"query_id": "111", "found": True, "abstract": "Recovered", "title": "T2"}])

    stats = merge_repairs(csv_path, cp)
    result = pd.read_csv(csv_path, dtype=str)
    assert stats["rows_repaired"] == 1
    assert result.loc[0, "abstract"] == "Recovered"


def test_merge_never_overwrites_an_already_populated_field(tmp_path):
    csv_path = tmp_path / "pool.csv"
    pd.DataFrame(
        [{"pmcid": "", "pmid": "111", "doi": "", "title": "Original title", "abstract": "",
          "journal": "", "year": "", "authors": ""}]
    ).to_csv(csv_path, index=False)
    cp = tmp_path / "records.jsonl"
    _write_checkpoint(cp, [{"query_id": "111", "found": True, "abstract": "New", "title": "Should not win"}])

    merge_repairs(csv_path, cp)
    result = pd.read_csv(csv_path, dtype=str)
    assert result.loc[0, "title"] == "Original title"
    assert result.loc[0, "abstract"] == "New"


def test_merge_falls_back_to_pmcid_when_the_row_has_no_pmid(tmp_path):
    csv_path = tmp_path / "pool.csv"
    pd.DataFrame(
        [{"pmcid": "PMC999", "pmid": "", "doi": "", "title": "T", "abstract": "",
          "journal": "", "year": "", "authors": ""}]
    ).to_csv(csv_path, index=False)
    cp = tmp_path / "records.jsonl"
    _write_checkpoint(
        cp, [{"query_id": "111", "found": True, "pmcid": "PMC999", "abstract": "Via pmcid"}]
    )

    merge_repairs(csv_path, cp)
    assert pd.read_csv(csv_path, dtype=str).loc[0, "abstract"] == "Via pmcid"


def test_merge_does_not_count_an_abstractless_match_as_a_repair(tmp_path):
    csv_path = tmp_path / "pool.csv"
    pd.DataFrame(
        [{"pmcid": "", "pmid": "111", "doi": "", "title": "T", "abstract": "",
          "journal": "", "year": "", "authors": ""}]
    ).to_csv(csv_path, index=False)
    cp = tmp_path / "records.jsonl"
    # PubMed had the record but it carries no abstract -- its ids/journal are still worth keeping,
    # but it must NOT be counted as a repaired abstract.
    _write_checkpoint(
        cp, [{"query_id": "111", "found": True, "abstract": None, "journal": "J Recovered"}]
    )

    stats = merge_repairs(csv_path, cp)
    assert stats["rows_repaired"] == 0
    assert stats["still_missing"] == 1
    # ...but the other real metadata it did return was kept rather than discarded.
    assert pd.read_csv(csv_path, dtype=str).loc[0, "journal"] == "J Recovered"


def test_merge_without_a_checkpoint_raises_rather_than_silently_doing_nothing(tmp_path):
    csv_path = tmp_path / "pool.csv"
    pd.DataFrame([{"pmid": "1", "abstract": ""}]).to_csv(csv_path, index=False)
    with pytest.raises(FileNotFoundError):
        merge_repairs(csv_path, tmp_path / "nope.jsonl")


def test_transient_errors_are_retried_on_resume_but_real_misses_are_not(tmp_path):
    # Real incident 2026-08-27: Crossref 429s were checkpointed as `found: false`, identical to
    # "this DOI genuinely has no abstract" -- so a resume permanently skipped ids that had only
    # been rate-limited. A transient error must never be treated as an answer.
    path = tmp_path / "cp.jsonl"
    with _Checkpoint(path) as cp:
        cp.write_many(
            [
                {"query_id": "rate-limited", "found": False, "error": True},
                {"query_id": "genuinely-absent", "found": False},
                {"query_id": "recovered", "found": True},
            ]
        )
    done = _Checkpoint(path).load_done()
    assert "rate-limited" not in done   # will be retried
    assert "genuinely-absent" in done   # answered; never ask again
    assert "recovered" in done
