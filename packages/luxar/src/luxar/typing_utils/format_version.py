"""On-disk format-version policy shared by every Luxar reader.

One rule, applied identically to a compiled scene (``.luxar.zarr``) and a
standalone gsplats store (``.gsplats.zarr``), and mirrored line for line by the
viewer's ``src/data/format-version.ts``:

* **supported** — the version is in this build's ``supported`` allowlist: load
  silently.
* **newer-minor** — same MAJOR as the current writer version, higher MINOR
  (a ``0.3`` scene read by a ``0.2`` build; a ``3.5`` gsplats store read by a
  ``3.4`` build): load, but WARN. A minor bump is additive by policy, so the
  reader can still make sense of the store; it just cannot see what the newer
  writer added.
* **refuse** — anything else: an OLDER version that has fallen out of
  ``supported`` (``0.0``, ``2.0``), a NEWER MAJOR (``9.9``), or a value that is
  not ``MAJOR.MINOR`` at all (``abc``). The error names the version, the
  supported set and the remedy (a rebuild for a scene, ``luxar gsplat
  migrate-format`` for a gsplats store).

A scene root may also carry NO version. Scene 0.1 wrote only the legacy
``luxar_version`` key (:data:`LEGACY_SCENE_VERSION_ATTR`), which readers fall
back to; a root that declares a ``format_type`` but no version is malformed and
is refused, while a root with neither is tolerated (external / hand-written
stores that predate any header — today's behaviour).

Shared case table (the viewer's ``format-version.test.ts`` uses the same ids so
the two suites can be diffed):

===============================  ======================  ==============
id                               input                   outcome
===============================  ======================  ==============
``supported``                    scene 0.1 / 0.2         SUPPORTED
                                 gsplats 3.0 / 3.4       SUPPORTED
``newer-minor``                  scene 0.3               NEWER_MINOR
                                 gsplats 3.5             NEWER_MINOR
``older-unsupported``            scene 0.0               REFUSE
                                 gsplats 2.0             REFUSE
``newer-major``                  scene 9.9               REFUSE
                                 gsplats 9.9             REFUSE
``unparsable``                   ``abc``                 REFUSE
``missing-with-format-type``     ``{format_type: …}``    REFUSE
``missing-without-format-type``  ``{type: scene}``       tolerated (None)
``legacy-key-fallback``          ``{luxar_version: 0.1}``  SUPPORTED, reads 0.1
===============================  ======================  ==============
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

from ._format_contract import (
    FORMAT_TYPE_GSPLATS,
    GSPLATS_FORMAT_VERSION,
    LEGACY_SCENE_VERSION_ATTR,
    SCENE_FORMAT_VERSION,
    SUPPORTED_GSPLATS_VERSIONS,
    SUPPORTED_SCENE_VERSIONS,
)

__all__ = [
    "FormatKind",
    "FormatVersionOutcome",
    "UnsupportedFormatVersionError",
    "parse_format_version",
    "check_format_version",
    "read_scene_format_version",
    "enforce_scene_format_version",
    "enforce_gsplats_format_version",
]

#: Which on-disk format a version string belongs to. Decides the remedy text.
FormatKind = Literal["scene", "gsplats"]


class FormatVersionOutcome(Enum):
    """What a reader should do with a store of a given format version."""

    #: In the allowlist — load silently.
    SUPPORTED = "supported"
    #: Same major, newer minor — load, but warn that newer content is invisible.
    NEWER_MINOR = "newer-minor"
    #: Older-unsupported, newer-major or unparsable — do not load.
    REFUSE = "refuse"


class UnsupportedFormatVersionError(ValueError):
    """Raised when a store's format version must be refused.

    A ``ValueError`` subclass so every existing ``except ValueError`` around a
    reader keeps working; the message already names the version, the supported
    set and the remedy.
    """


def parse_format_version(version: Any) -> Optional[Tuple[int, int]]:
    """Parse a ``MAJOR.MINOR`` string into ``(major, minor)``; ``None`` otherwise.

    Deliberately strict: no patch component, no leading ``v``, no whitespace.
    Version strings are written by Luxar itself from the contract, so anything
    else is a foreign or corrupt header, not a spelling to be lenient about.
    """
    if not isinstance(version, str):
        return None
    major, sep, minor = version.partition(".")
    if not sep or not major.isdigit() or not minor.isdigit():
        return None
    return int(major), int(minor)


def _remedy(kind: FormatKind, outcome: str) -> str:
    if kind == "gsplats":
        if outcome == "older":
            return (
                "Convert it with `luxar gsplat migrate-format <input> "
                "<output.gsplats.zarr>`."
            )
        return "Upgrade Luxar to a build that supports this format."
    if outcome == "older":
        return "Rebuild the scene with the current Luxar release."
    return "Upgrade Luxar to a build that supports this format."


def check_format_version(
    kind: FormatKind,
    version: Any,
    current: str,
    supported: Sequence[str],
) -> Tuple[FormatVersionOutcome, str]:
    """Classify ``version`` against this build's ``current`` / ``supported``.

    Returns ``(outcome, message)``; ``message`` is empty for
    :attr:`FormatVersionOutcome.SUPPORTED`, the warning text for
    :attr:`~FormatVersionOutcome.NEWER_MINOR`, and the error text for
    :attr:`~FormatVersionOutcome.REFUSE`.

    Args:
        kind: ``"scene"`` or ``"gsplats"`` — picks the noun and the remedy.
        version: The on-disk value (any type; non-strings are unparsable).
        current: The version this build writes.
        supported: The versions this build has read end-to-end.
    """
    label = "scene" if kind == "scene" else ".gsplats.zarr"
    if isinstance(version, str) and version in supported:
        return FormatVersionOutcome.SUPPORTED, ""

    parsed = parse_format_version(version)
    supported_text = ", ".join(supported)
    if parsed is None:
        return FormatVersionOutcome.REFUSE, (
            f"Unsupported format_version: {version!r} for a {label} store "
            f"(not a MAJOR.MINOR version; supported: {supported_text}). "
            + _remedy(kind, "unparsable")
        )

    cur = parse_format_version(current)
    assert cur is not None, f"current version {current!r} is not MAJOR.MINOR"
    major, minor = parsed
    if major == cur[0] and minor > cur[1]:
        return FormatVersionOutcome.NEWER_MINOR, (
            f"This {label} store is format {version}, newer than the {current} "
            f"this Luxar build writes. Loading anyway; content added by the "
            f"newer format is not visible. Upgrade Luxar to read it fully."
        )
    if (major, minor) < cur:
        return FormatVersionOutcome.REFUSE, (
            f"Unsupported format_version: {version!r} for a {label} store "
            f"(too old; supported: {supported_text}). " + _remedy(kind, "older")
        )
    return FormatVersionOutcome.REFUSE, (
        f"Unsupported format_version: {version!r} for a {label} store "
        f"(newer major version; this build writes {current} and reads "
        f"{supported_text}). " + _remedy(kind, "newer-major")
    )


def read_scene_format_version(attrs: Mapping[str, Any]) -> Optional[str]:
    """Return a scene root's version string, or ``None`` when it carries none.

    ``format_version`` (scene 0.2+) wins; the 0.1 legacy key
    :data:`LEGACY_SCENE_VERSION_ATTR` is the fallback. Values are coerced to
    ``str`` so a YAML-ish ``0.2`` float still compares against the allowlist.
    """
    value = attrs.get("format_version")
    if value is None:
        value = attrs.get(LEGACY_SCENE_VERSION_ATTR)
    if value is None:
        return None
    return str(value)


def enforce_scene_format_version(
    attrs: Mapping[str, Any], *, stacklevel: int = 2
) -> Optional[FormatVersionOutcome]:
    """Apply the policy to a scene root's attrs; raise, warn or return.

    * REFUSE → :class:`UnsupportedFormatVersionError`.
    * NEWER_MINOR → ``warnings.warn(..., UserWarning)`` and return the outcome.
    * SUPPORTED → return the outcome.
    * No version but a ``format_type`` → REFUSE (a 0.2+ header with the version
      stripped is corrupt, not legacy).
    * No version and no ``format_type`` → ``None`` (tolerated; a pre-header or
      hand-written store).
    """
    version = read_scene_format_version(attrs)
    if version is None:
        format_type = attrs.get("format_type")
        if format_type is not None:
            raise UnsupportedFormatVersionError(
                f"Scene root declares format_type={format_type!r} but no "
                f"format_version. A Luxar scene header carries both; this store "
                f"is corrupt or was written by a foreign tool. Rebuild it with "
                f"the current Luxar release."
            )
        return None
    outcome, message = check_format_version(
        "scene", version, SCENE_FORMAT_VERSION, SUPPORTED_SCENE_VERSIONS
    )
    if outcome is FormatVersionOutcome.REFUSE:
        raise UnsupportedFormatVersionError(message)
    if outcome is FormatVersionOutcome.NEWER_MINOR:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)
    return outcome


def enforce_gsplats_format_version(
    attrs: Mapping[str, Any], *, stacklevel: int = 2
) -> FormatVersionOutcome:
    """Apply the policy to a standalone gsplats root's attrs.

    The root must already have been identified as a gsplats store
    (``format_type == FORMAT_TYPE_GSPLATS``); a missing ``format_version`` is
    refused because every gsplats format since 1.0 has written one.
    """
    if attrs.get("format_type") != FORMAT_TYPE_GSPLATS:
        raise UnsupportedFormatVersionError(
            f"Invalid format_type: {attrs.get('format_type')!r}, expected "
            f"{FORMAT_TYPE_GSPLATS!r}."
        )
    outcome, message = check_format_version(
        "gsplats",
        attrs.get("format_version"),
        GSPLATS_FORMAT_VERSION,
        SUPPORTED_GSPLATS_VERSIONS,
    )
    if outcome is FormatVersionOutcome.REFUSE:
        raise UnsupportedFormatVersionError(message)
    if outcome is FormatVersionOutcome.NEWER_MINOR:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)
    return outcome
