"""The one definition of what a dataset citation is in Luxar.

A citation credits whoever produced the data a scene shows. It lives in two
places — a demo's ``DEMO_META`` and a scene's zarr root attributes — and both
go through the validator here, so "what counts as a citation" is stated once
rather than drifting between the authoring registry and the file format.

``None`` is a legitimate value: it means "procedurally generated, nothing to
credit". That is a deliberate statement, not a missing field, which is why it
is spelled out rather than expressed by leaving the key off.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping, Optional

#: A DOI is ``10.<registrant>/<suffix>``; the registrant is 4+ digits and the
#: suffix is non-empty. Deliberately permissive about the suffix, which the DOI
#: spec allows to contain almost anything.
_DOI_SHAPE = re.compile(r"10\.\d{4,9}/\S+")

#: Keys a citation may carry. ``short`` is the only required one: it is what a
#: gallery tile and the viewer render ("Schlegel et al. 2024"), so a citation
#: that cannot be displayed is not a citation.
CITATION_KEYS = ("short", "doi", "license", "url")

#: Schemes a citation URL may use. A credit's URL is the one field a UI turns
#: into a link, and a citation travels inside data that is copied, published and
#: opened by whoever receives it -- so a ``javascript:`` or ``data:`` payload has
#: no business being storable here.
_URL_SCHEMES = ("http://", "https://")


def _reject_unprintable(field: str, text: str) -> None:
    """Refuse line breaks and control/format characters in a citation field.

    "Single line" has to mean every line break and control character, not just
    ``\\n``: a lone ``\\r`` still breaks a line in plenty of renderers, a tab
    wrecks a tile's alignment, and a bidi override (U+202E) can make a rendered
    credit read differently from the string that was stored -- which in an
    attribution field is a spoofing vector, not a cosmetic issue. That reasoning
    is not specific to ``short``: a licence, a DOI and a URL are all rendered,
    and a DOI's suffix in particular is otherwise free to contain anything
    non-whitespace.
    """
    bad = {ch for ch in text if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp")}
    if bad:
        raise ValueError(
            f"citation.{field} must be a single line of printable text; it "
            "contains " + ", ".join(sorted("U+%04X" % ord(ch) for ch in bad))
        )


def validate_citation(value: Any) -> Optional[dict[str, str]]:
    """Validate a citation payload and return a plain, copied dict.

    Args:
        value: ``None`` (procedural, no credit owed) or a mapping with a
            non-empty single-line ``short`` and optional ``doi``, ``license``
            and ``url``.

    Returns:
        A new dict with the same entries, or ``None``. Copying keeps a caller's
        mutable mapping from aliasing into scene attributes or registry state.

    Raises:
        ValueError: If the payload is not a valid citation. The message names
            the offending field so callers can prefix it with their own context.
    """
    if value is None:
        return None

    if not isinstance(value, Mapping):
        raise ValueError(
            f"citation must be None or a mapping, got {type(value).__name__}"
        )

    extra = set(value.keys()) - set(CITATION_KEYS)
    if extra:
        raise ValueError(
            f"citation has unknown keys {sorted(extra)} "
            f"(allowed: {sorted(CITATION_KEYS)})"
        )

    short = value.get("short")
    if not isinstance(short, str) or not short.strip():
        raise ValueError("citation.short must be a non-empty string")
    _reject_unprintable("short", short)

    for key in ("doi", "license", "url"):
        if key not in value:
            continue
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"citation.{key} must be a non-empty string when present")
        _reject_unprintable(key, value[key])

    url = value.get("url")
    # Scheme allowlist, not a full URL parse: the point is to keep an executable
    # or inline-payload scheme out of a field a viewer will render as a link.
    if url is not None and not url.strip().lower().startswith(_URL_SCHEMES):
        raise ValueError(
            f"citation.url {url!r} must be an http:// or https:// URL "
            "(it is rendered as a link)"
        )

    doi = value.get("doi")
    # Shape only: a DOI is ``10.<registrant>/<suffix>``. Whether it RESOLVES is an
    # authoring-time question, not something to ask on every scene write -- but
    # the shape is worth enforcing in full, because the near-misses are the ones
    # people actually paste: a URL, a ``doi:`` prefix, or a bare registrant with
    # the suffix lost.
    if doi is not None and not _DOI_SHAPE.fullmatch(doi):
        raise ValueError(
            f"citation.doi {doi!r} must be a bare DOI of the form "
            "'10.<registrant>/<suffix>' (no https://doi.org/ or 'doi:' prefix)"
        )

    # Strip on the way in: a stray space in a demo literal would otherwise be
    # rendered verbatim after the tile's em-dash.
    return {key: value[key].strip() for key in CITATION_KEYS if key in value}
