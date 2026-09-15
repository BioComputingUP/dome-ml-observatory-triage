"""Tests for the before/after comparison of two data-links builds."""

from __future__ import annotations

import csv
import json

import compare_data_links as cdl


def _block(links, resources, sources=("epmc_annotations",)):
    return json.dumps({"sources": list(sources),
                       "resources": [{"resource": r, "count": n, **({"routes": routes} if routes else {})}
                                     for r, n, routes in resources],
                       "links": [{"resource": r, "id": i} for r, i in links]})


def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pid", "data_links_json"])
        writer.writeheader()
        writer.writerows({"pid": pid, "data_links_json": cell} for pid, cell in rows)


OLD = [
    ("p1", _block([("arrayexpress", "E-GEOD-1"), ("pdb", "6VW1")],
                  [("arrayexpress", 1, None), ("pdb", 1, None)])),
    ("p2", _block([("pdb", "1ABC")], [("pdb", 1, None)])),
    ("p3", _block([("pride", "PXD000001")], [("pride", 1, None)])),
    ("p4", _block([("dbgap", "phs000310.v1.p1")], [("dbgap", 1, None)])),
]
NEW = [
    ("p1", _block([("geo", "GSE1"), ("pdb", "6vw1"), ("biotools", "suba3")],
                  [("geo", 1, ["ebisearch_domain", "tm_accession"]), ("pdb", 1, ["tm_accession"]),
                   ("biotools", 1, ["ebisearch_domain"])], sources=("epmc_annotations", "ebisearch"))),
    ("p2", ""),
    ("p3", _block([("iprox", "PXD000001")], [("iprox", 1, ["ebisearch_domain", "tm_accession"])])),
    ("p4", _block([("dbgap", "phs000310")], [("dbgap", 1, None)])),
]


def _check(result):
    assert result.documents["compared"] == 4 and result.documents["newly_withheld"] == 1
    assert result.documents["with_ebisearch"] == 1
    assert result.moves == {("arrayexpress", "geo"): 1, ("pride", "iprox"): 1, ("dbgap", "dbgap"): 1}
    assert result.gained_documents == {"geo": 1, "biotools": 1, "iprox": 1}
    assert result.lost_documents == {"arrayexpress": 1, "pride": 1}
    assert result.added_links == {"biotools": 1}
    assert result.removed_links == {} and result.removed_under_cap == {}
    assert result.confirmed_by_both == {"geo": 1, "iprox": 1}


def test_moves_additions_and_withheld_documents_are_told_apart(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    _write(old, OLD)
    _write(new, NEW)
    _check(cdl.compare(old, new))


def test_rows_out_of_step_are_matched_by_pid(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    _write(old, OLD)
    _write(new, list(reversed(NEW)))
    result = cdl.compare(old, new)
    _check(result)
    assert result.documents["only_in_old"] == result.documents["only_in_new"] == 0


def test_a_mirror_merged_into_a_link_already_held_is_not_a_loss(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    _write(old, [("p1", _block([("arrayexpress", "E-GEOD-7"), ("geo", "GSE7")],
                               [("arrayexpress", 1, None), ("geo", 1, None)]))])
    _write(new, [("p1", _block([("geo", "GSE7")], [("geo", 1, ["tm_accession"])]))])
    result = cdl.compare(old, new)
    assert result.merged_links == {"arrayexpress": 1}
    assert result.removed_links == {} and result.moves == {} and result.lost_documents == {"arrayexpress": 1}


def test_a_link_capped_out_of_the_detail_is_not_a_loss(tmp_path):
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    _write(old, [("p1", _block([("pdb", "1ABC"), ("pdb", "2ABC")], [("pdb", 2, None)]))])
    _write(new, [("p1", _block([("pdb", "1ABC")], [("pdb", 2, None)]))])
    result = cdl.compare(old, new)
    assert result.removed_under_cap == {"pdb": 1} and result.removed_links == {}
