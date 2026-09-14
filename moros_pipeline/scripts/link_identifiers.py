"""What a clean data-link identifier is -- defined once, used by the merge and by every writer.

Europe PMC's text mining records the string it matched in the article (`exact`), and PDF/HTML
extraction leaves that string carrying whatever surrounded it. Stored verbatim in the 2026-09-14
corpus build (872,144 links), these produced links that 404:

- trailing sentence punctuation and brackets: `10.5281/zenodo.18675888.`, `10.17632/bcmn9cxyzs.4]`;
- quote marks, non-breaking spaces and the words after them: `10.6084/m9.figshare.24123303”.`,
  `10.5061/dryad.040h9t7.<NBSP>Behavioural`;
- comma lists and thousands separators: `10.7910/DVN/UFC6B5,HarvardDataverse,V2`,
  `10.34740/kaggle/ds/6,979,862`;
- PDF glyphs inside the identifier: `10.17632/trvb5k4×5m.1`, `10.4121/...-41c9–8607-...`;
- placeholders: `10.5281/zenodo.XXXXX`;
- free text where an RRID was matched: `Alexa Fluor 647-conjugated goat anti-rabbit`,
  `RRID: AB_1658454`, `ORPHA 401777`.

`build_data_links.py` is the only place identifiers are chosen: it repairs what can be repaired
with the helpers below and confirms every DOI exists at doi.org. `load_fields.py --mode data_links`
and `load_documents.py` refuse any document whose links fail `malformed_links()`, and
`verify_corpus.py` asserts that `MONGO_MALFORMED_REGEX` matches nothing in moros. The raw fetch
outputs keep Europe PMC's strings verbatim, so this can always be re-applied without re-fetching.

Pure: no I/O, no network.
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote

# Characters that end an identifier embedded in running text: any Unicode whitespace (including the
# non-breaking and thin spaces PDF extraction leaves), double and typographic quotes, fullwidth
# punctuation, and the start of a closing HTML tag. `<` and `>` alone are deliberately NOT here:
# SICI DOIs contain them (`10.1002/(SICI)...36:4<471::AID-JBM4>3.0.CO;2-G`).
_TERMINATOR_RE = re.compile(r'[\s"“”„‘’«»（）［］｛｝，。；：、！？]|</')
_TRAILING = set(".,;:'\"!?")
_PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}
_OPENING = {opening: closing for closing, opening in _PAIRS.items()}
_LEADING_QUOTES = set("'\"")

# Glyphs PDF extraction substitutes inside identifiers.
_TIMES_RE = re.compile(r"\s*[×✕✖]\s*")
_DASH_RE = re.compile(r"[‐‑‒–—−]")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

DOI_START_RE = re.compile(r"10\.\d{4,9}/")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
RRID_RE = re.compile(r"^RRID:[A-Za-z]+[_:]\S+$")
PLACEHOLDER_RE = re.compile(r"XXX")
_NON_PRINTABLE_ASCII_RE = re.compile(r"[^\x21-\x7e]")
_PREFIXED_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)\s*:?\s+(?=\S)")
_IDENTIFIERS_ORG_RE = re.compile(r"^(.*?identifiers\.org/(?:[^/:]+/)?[^/:]+:)(.+)$")
_SCICRUNCH_RE = re.compile(r"scicrunch\.org/resolver/(RRID:.+)$")

_BAD_FIRST = "([{<'\""
_BAD_LAST = ".,;:]}>'\""

# The Mongo-expressible subset of `problems()`: any character outside printable ASCII, a leading
# bracket or quote, or trailing punctuation. `verify_corpus.py` counts documents whose
# `data_links.links.id` or `.url` match it; the number must be zero.
MONGO_MALFORMED_REGEX = r"[^\x21-\x7e]|^[(\[{<'\"]|[.,;:\]}>'\"]$"


def clean(value: str | None) -> str:
    """Cuts an identifier out of the text around it: stop at the first terminator, then strip
    trailing punctuation, unbalanced closing brackets, leading quotes and unbalanced opening
    brackets until none remain. Never changes characters inside the identifier."""
    text = (value or "").strip()
    match = _TERMINATOR_RE.search(text)
    if match:
        text = text[:match.start()]
    while text:
        last, first = text[-1], text[0]
        if last in _TRAILING or (last in _PAIRS and text.count(last) > text.count(_PAIRS[last])):
            text = text[:-1]
        elif first in _LEADING_QUOTES or (
                first in _OPENING and text.count(first) > text.count(_OPENING[first])):
            text = text[1:]
        else:
            break
    return text


def normalise_prefixed(value: str | None) -> str:
    """`RRID: AB_1` -> `RRID:AB_1`; `ORPHA 401777` -> `ORPHA:401777`; `ZFIN:<NBSP>ZDB-...` ->
    `ZFIN:ZDB-...`. A value with no spaced prefix is returned stripped and otherwise unchanged."""
    text = (value or "").strip()
    return _PREFIXED_RE.sub(lambda m: m.group(1) + ":", text, count=1)


def resolver_local_id(uri: str | None) -> str | None:
    """The identifier inside a resolver URL Europe PMC attached to the annotation:
    `https://scicrunch.org/resolver/RRID:AB_2535812` -> `RRID:AB_2535812`;
    `http://identifiers.org/orphanet:401777` -> `401777`;
    `http://identifiers.org/ebi/ena.embl:AY278488` -> `AY278488`."""
    text = (uri or "").strip()
    match = _SCICRUNCH_RE.search(text)
    if match:
        return match.group(1)
    match = _IDENTIFIERS_ORG_RE.match(text)
    return match.group(2) if match else None


def is_doi(value: str | None) -> bool:
    return bool(DOI_RE.match(value or ""))


def problems(link_id: str | None, resource: str | None = None) -> list[str]:
    """Why an identifier must not be stored; empty when it is clean."""
    text = link_id or ""
    if not text:
        return ["empty"]
    reasons: list[str] = []
    if _NON_PRINTABLE_ASCII_RE.search(text):
        reasons.append("whitespace or non-ASCII")
    if text[0] in _BAD_FIRST:
        reasons.append("leading punctuation")
    if text[-1] in _BAD_LAST:
        reasons.append("trailing punctuation")
    if any(text.count(closing) != text.count(opening) for closing, opening in _PAIRS.items()):
        reasons.append("unbalanced brackets")
    if text.startswith("10."):
        if not DOI_RE.match(text):
            reasons.append("not a DOI")
        elif PLACEHOLDER_RE.search(text):
            reasons.append("placeholder")
    if resource == "rrid" or text.upper().startswith("RRID:"):
        if not RRID_RE.match(text):
            reasons.append("not an RRID")
    return reasons


def url_problems(url: str | None) -> list[str]:
    """Why a link URL must not be stored; empty when it is clean."""
    text = url or ""
    reasons: list[str] = []
    if not text.startswith(("http://", "https://")):
        reasons.append("not an http(s) URL")
    if _NON_PRINTABLE_ASCII_RE.search(text):
        reasons.append("whitespace or non-ASCII")
    if text and text[-1] in _BAD_LAST:
        reasons.append("trailing punctuation")
    return reasons


def doi_candidates(raw: str | None) -> list[str]:
    """The DOIs a text-mined string may stand for, most likely first, each syntactically clean.
    Which one is real is not a syntax question -- `build_data_links.py` asks doi.org, in this order.

    1. PDF glyphs repaired (`×` -> `x`, typographic dashes -> `-`), then cleaned.
    2. Thousands separators removed (`.../ds/6,979,862` -> `.../ds/6979862`).
    3. Cut at the first comma (`10.7910/DVN/UFC6B5,HarvardDataverse,V2` -> `10.7910/DVN/UFC6B5`).

    The unrepaired string is deliberately not a candidate when a glyph was repaired: cutting
    `10.17632/k82<NBSP>×<NBSP>7czd87.1` at its space would offer `10.17632/k82`, a truncation that
    could exist as a different record."""
    text = (raw or "").strip()
    match = DOI_START_RE.search(text)
    if not match:
        return []
    token = _DASH_RE.sub("-", _TIMES_RE.sub("x", text[match.start():]))
    variants = (clean(token), clean(_THOUSANDS_RE.sub("", token)), clean(token.split(",")[0]))
    found: list[str] = []
    for variant in variants:
        if variant and variant not in found and not problems(variant):
            found.append(variant)
    return found


def doi_url(doi: str) -> str:
    return "https://doi.org/" + quote(doi, safe="/:;()._-~")


def canonical_url(resource: str | None, link_id: str, original_url: str | None) -> str | None:
    """The URL stored with a chosen identifier. Europe PMC's own URL is kept whenever it is clean
    (and, for a DOI or RRID, ends with the identifier); otherwise the canonical resolver is used,
    an identifiers.org URL gets its local part replaced by the cleaned one, and anything else is
    dropped rather than stored broken (the record page then falls back to Europe PMC)."""
    original = (original_url or "").strip()
    usable = bool(original) and not url_problems(original)
    if is_doi(link_id):
        if usable and unquote(original).lower().endswith(link_id.lower()):
            return original
        return doi_url(link_id)
    if resource == "rrid" or link_id.upper().startswith("RRID:"):
        if usable and unquote(original).endswith(link_id):
            return original
        return "https://scicrunch.org/resolver/" + quote(link_id, safe=":_-.")
    if usable:
        return original
    match = _IDENTIFIERS_ORG_RE.match(original)
    if match:
        local = clean(match.group(2))
        rebuilt = match.group(1) + local
        if local and not problems(local) and not url_problems(rebuilt):
            return rebuilt
    return None


def malformed_links(detail: dict | None) -> list[tuple[str | None, str | None, str | None, list[str]]]:
    """(resource, id, url, reasons) for every link in a `data_links` block that must not be written.
    The gate every writer applies: an empty list is the only acceptable answer."""
    found = []
    for link in (detail or {}).get("links") or []:
        link_id, url = link.get("id"), link.get("url")
        reasons = problems(link_id, link.get("resource"))
        if url is not None:
            reasons += [f"url {reason}" for reason in url_problems(url)]
        if reasons:
            found.append((link.get("resource"), link_id, url, reasons))
    return found
