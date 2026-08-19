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

from typing import Any, Mapping, Optional

#: Keys a citation may carry. ``short`` is the only required one: it is what a
#: gallery tile and the viewer render ("Schlegel et al. 2024"), so a citation
#: that cannot be displayed is not a citation.
CITATION_KEYS = ("short", "doi", "license", "url")


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
    if "\n" in short:
        raise ValueError("citation.short must be a single line")

    for key in ("doi", "license", "url"):
        if key in value and (not isinstance(value[key], str) or not value[key].strip()):
            raise ValueError(f"citation.{key} must be a non-empty string when present")

    doi = value.get("doi")
    # Shape only: a DOI is ``10.<registrant>/<suffix>``. Whether it resolves is
    # an authoring-time question, not something to ask on every scene write.
    if doi is not None and not doi.startswith("10."):
        raise ValueError(
            f"citation.doi {doi!r} must be a bare DOI starting with '10.', not a URL"
        )

    return {key: value[key] for key in CITATION_KEYS if key in value}
