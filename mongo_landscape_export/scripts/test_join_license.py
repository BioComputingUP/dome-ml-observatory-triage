from __future__ import annotations

import csv

import pytest

from join_license import join_license, load_licensing

HEADER = ["pid", "pmid", "title", "is_open_access"]
ROWS = [
    ["p1", "1", "Paper one", "True"],   # licensed, license disclosed
    ["p2", "2", "Paper two", "True"],   # licensed, none disclosed (empty string from EPMC)
    ["p3", "3", "Paper three", "False"],  # genuine EPMC miss (not in licensing table)
    ["p4", "", "Paper four", "True"],   # no pmid at all -- can never be checked
]

LICENSING_ROWS = [
    {"pmid": "1", "license": "cc by", "is_open_access": "Y"},
    {"pmid": "2", "license": "", "is_open_access": "N"},
]


def _write_csv(path, header, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_licensing(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pmid", "license", "is_open_access"])
        writer.writeheader()
        writer.writerows(rows)


def test_joins_license_and_marks_checked_status(tmp_path):
    data_path = tmp_path / "data.csv"
    licensing_path = tmp_path / "licensing.csv"
    _write_csv(data_path, HEADER, ROWS)
    _write_licensing(licensing_path, LICENSING_ROWS)

    licensing = load_licensing(licensing_path)
    report = join_license(data_path, licensing)

    assert report == {"rows_total": 4, "rows_checked": 2, "rows_not_checked": 2}

    with data_path.open(newline="", encoding="utf-8") as f:
        rows = {r["pid"]: r for r in csv.DictReader(f)}

    assert rows["p1"]["license"] == "cc by"
    assert rows["p1"]["license_checked"] == "True"
    assert rows["p1"]["epmc_is_open_access"] == "Y"

    # Checked, EPMC disclosed no license -- distinguishable from "never checked" via the flag.
    assert rows["p2"]["license"] == ""
    assert rows["p2"]["license_checked"] == "True"
    assert rows["p2"]["epmc_is_open_access"] == "N"

    # Genuine EPMC miss -- has a pmid but no licensing entry.
    assert rows["p3"]["license_checked"] == "False"
    assert rows["p3"]["license"] == ""

    # No pmid at all.
    assert rows["p4"]["license_checked"] == "False"


def test_refuses_to_join_twice(tmp_path):
    data_path = tmp_path / "data.csv"
    licensing_path = tmp_path / "licensing.csv"
    _write_csv(data_path, HEADER, ROWS)
    _write_licensing(licensing_path, LICENSING_ROWS)

    licensing = load_licensing(licensing_path)
    join_license(data_path, licensing)

    with pytest.raises(ValueError, match="already present"):
        join_license(data_path, licensing)
