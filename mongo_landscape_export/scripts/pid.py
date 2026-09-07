"""Step 24 phase 1, Step 3: mints a unique, deterministic PID for each landscape record.

Distinct from this project's existing internal `record_id`
(`src/dome_triage/dedupe/keys.py::record_id_from_ids`, a sha1 hash) on purpose -- that id
identifies rows in a different resource (`canonical_dataset.csv`, the human-curated trusted set).
This Mongo-facing landscape resource gets its own identifier space, confirmed with Gavin
(2026-08-28): a UUID5 minted from a fixed namespace constant plus the paper's identity, so the
format alone (a 36-character UUID string vs. a 40-character sha1 hex string) can never collide
with `record_id`, on top of using an entirely separate namespace/hash.

Deterministic by construction: the same paper (same pmcid/doi/pmid) always mints the same PID on
any future run. That is what makes a monthly incremental append safe -- reprocessing a paper
already in the database yields its existing PID again instead of a duplicate document.

Self-contained -- no `dome_triage` import, matching this folder's `epmc_licensing`-style
"standalone, runs outside Docker" precedent (see README.md) -- but the pmcid>doi>pmid priority
order and the blank/NaN-string guard below are deliberately mirrored from
`dedupe/keys.py::canonical_key_from_ids` / `_is_usable_id` for consistency with the rest of the
project, not reinvented.
"""

from __future__ import annotations

import uuid

# Fixed forever once chosen -- generated once via uuid.uuid4() on 2026-08-28. A distinct namespace
# from any other identifier scheme in this project, so a PID minted here can never collide with an
# id minted anywhere else even given identical input values.
LANDSCAPE_PID_NAMESPACE = uuid.UUID("9a4394bb-415f-4429-9bc5-a58cd78bdd51")

_ID_PRIORITY = ("pmcid", "doi", "pmid")
_KEY_PREFIX = {"pmcid": "PMCID", "doi": "DOI", "pmid": "PMID"}

# Same hard-won guard as dedupe/keys.py::_is_usable_id -- a missing id read via `pandas
# dtype=str` (or any str() round-trip of a pandas NaN) comes back as the literal string "nan",
# which is truthy under a bare `if value:` check. This has silently fused unrelated records
# together three separate times in this project already (2026-08-27, see dedupe/keys.py's own
# comment). Guarded here too even though this folder reads CSVs with the stdlib `csv` module
# (which yields a real blank string, not "nan") -- defense in depth against the same class of bug.
_NON_ID_STRINGS = {"nan", "none", "null", "na", "n/a", "<na>", "nat", ""}


def _is_usable_id(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in _NON_ID_STRINGS


def landscape_identity_key(pmcid: str | None, doi: str | None, pmid: str | None) -> str | None:
    """pmcid > doi > pmid priority, mirroring dedupe/keys.py::canonical_key_from_ids. Returns None
    when none of the three is usable. Confirmed live (2026-08-28) that this never actually happens
    across the full 827,061-row usable set -- every row has at least one -- but callers must not
    assume that holds forever (e.g. a future incremental batch)."""
    values = {"pmcid": pmcid, "doi": doi, "pmid": pmid}
    for field in _ID_PRIORITY:
        value = values[field]
        if _is_usable_id(value):
            return f"{_KEY_PREFIX[field]}:{value.strip()}"
    return None


def mint_landscape_pid(pmcid: str | None, doi: str | None, pmid: str | None) -> str:
    """Raises ValueError when no usable id is present at all -- never silently mints a placeholder
    PID for an unidentifiable record."""
    key = landscape_identity_key(pmcid, doi, pmid)
    if key is None:
        raise ValueError("mint_landscape_pid: no usable pmcid/doi/pmid to mint a PID from")
    return str(uuid.uuid5(LANDSCAPE_PID_NAMESPACE, key))
