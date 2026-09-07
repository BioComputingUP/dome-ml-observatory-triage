from __future__ import annotations

from schema import SCHEMA_VERSION, build_document
from test_schema import BASE_ROW
from write_schema_template import build_template


def _keys_by_path(doc: dict, prefix: str = "") -> set[str]:
    keys = set()
    for key, value in doc.items():
        path = f"{prefix}{key}"
        keys.add(path)
        if isinstance(value, dict):
            keys |= _keys_by_path(value, prefix=f"{path}.")
    return keys


def test_template_key_shape_matches_a_real_built_document():
    template = build_template()
    real = build_document(BASE_ROW)
    assert _keys_by_path(template) == _keys_by_path(real)


def test_template_schema_version_matches_the_real_constant():
    assert build_template()["schema_version"] == SCHEMA_VERSION


def test_template_id_is_null_not_a_placeholder_string():
    assert build_template()["_id"] is None
