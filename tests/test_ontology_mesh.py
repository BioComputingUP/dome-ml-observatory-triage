from dome_triage.ontology.mesh import extract_mesh_headings


def test_extract_mesh_headings_present():
    # Structure confirmed via a live Europe PMC resultType=core fetch.
    result = {
        "meshHeadingList": {
            "meshHeading": [
                {"majorTopic_YN": "N", "descriptorName": "Humans"},
                {"majorTopic_YN": "Y", "descriptorName": "Machine Learning"},
            ]
        }
    }
    assert extract_mesh_headings(result) == ["Humans", "Machine Learning"]


def test_extract_mesh_headings_absent():
    # Confirmed real: even MEDLINE-sourced records don't always have MeSH (indexing lag).
    result = {"pmid": "123", "title": "Some paper"}
    assert extract_mesh_headings(result) == []


def test_extract_mesh_headings_empty_list():
    result = {"meshHeadingList": {"meshHeading": []}}
    assert extract_mesh_headings(result) == []


def test_extract_mesh_headings_skips_entries_without_descriptor_name():
    result = {"meshHeadingList": {"meshHeading": [{"majorTopic_YN": "N"}]}}
    assert extract_mesh_headings(result) == []
