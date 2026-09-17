"""Deprecation helpers for the post-release compatibility window.

Before the first PyPI release a rename is a hard cut — the old spelling is
rejected with a pointer to the new one (see ``_reject_renamed_method_flags`` in
``luxar.cli.lod`` and ``LEGACY_RECIPE_NAMES`` in ``luxar.gsplats.lod.recipes``).
After it, a public name that changes keeps working for the window the
compatibility policy promises (``docs/guides/user/COMPATIBILITY_POLICY.md``:
two releases or six months, whichever is longer) while warning, and these
helpers are how every such alias says so in one voice.

Two entry points:

* :func:`warn_deprecated` emits the standard sentence as a
  :class:`DeprecationWarning`. Call it from inside the deprecated function,
  property, or alias so the default ``stacklevel`` points at the *user's* call
  site rather than at Luxar.
* :func:`deprecated_kwarg_alias` handles the commonest case, a renamed keyword
  argument: it moves the old key onto the new one in the caller's ``**kwargs``
  and warns, so the function body only ever reads the new name.

The CLI counterpart, ``luxar.cli.utils.deprecated_option``, prints to stderr
instead of warning: Python hides :class:`DeprecationWarning` by default outside
``__main__``, which is the right default for a library and the wrong one for a
command someone just typed.

Every message carries the release the deprecation started in and the point
after which the alias may go, so a reader of the warning never has to look the
window up.
"""

from __future__ import annotations

import warnings
from typing import MutableMapping

__all__ = [
    "deprecated_kwarg_alias",
    "deprecation_message",
    "warn_deprecated",
]


def deprecation_message(
    old: str,
    new: str | None,
    *,
    since: str,
    remove_after: str,
) -> str:
    """Build the one sentence every Luxar deprecation notice uses.

    Args:
        old: The deprecated spelling as the user would type it (``"optimise"``,
            ``"--reveal-centre"``, ``"reveal_centre="``).
        new: The replacement, or ``None`` when the feature is going away with no
            successor.
        since: The Luxar release (CalVer, ``YYYY.MM.DD``) that introduced the
            deprecation.
        remove_after: The earliest release or date after which the alias may be
            removed, per the compatibility policy.

    Returns:
        The formatted notice, ending in a full stop.
    """
    replacement = f"; use {new} instead" if new else ""
    return (
        f"{old} is deprecated since Luxar {since} and will be removed after "
        f"{remove_after}{replacement}."
    )


def warn_deprecated(
    old: str,
    new: str | None = None,
    *,
    since: str,
    remove_after: str,
    stacklevel: int = 3,
) -> None:
    """Emit the standard deprecation notice as a :class:`DeprecationWarning`.

    Args:
        old: The deprecated spelling.
        new: The replacement spelling, or ``None`` when there is none.
        since: Release that introduced the deprecation (``YYYY.MM.DD``).
        remove_after: Earliest release or date the alias may be removed after.
        stacklevel: Frames to skip so the warning is attributed to the user's
            code. The default of ``3`` is right when this is called directly
            from inside the deprecated function: one frame for this helper, one
            for the deprecated function, landing on its caller.

    Example:
        >>> def optimise_store(*args, **kwargs):  # the alias
        ...     warn_deprecated(
        ...         "luxar.io.optimise_store",
        ...         "luxar.io.optimize_store",
        ...         since="2026.10.01",
        ...         remove_after="2027.04.01",
        ...     )
        ...     return optimize_store(*args, **kwargs)
    """
    warnings.warn(
        deprecation_message(old, new, since=since, remove_after=remove_after),
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated_kwarg_alias(
    kwargs: MutableMapping[str, object],
    old: str,
    new: str,
    *,
    since: str,
    remove_after: str,
    stacklevel: int = 3,
) -> MutableMapping[str, object]:
    """Forward a renamed keyword argument from its old name to its new one.

    Mutates ``kwargs`` in place: when ``old`` is present it is popped, its value
    stored under ``new``, and :func:`warn_deprecated` fires. When ``old`` is
    absent nothing happens and nothing is emitted, so the call is free on the
    modern path. Passing both spellings is an error — silently preferring one
    would hide a real mistake in the caller.

    Args:
        kwargs: The ``**kwargs`` mapping of the function accepting the alias.
        old: The deprecated keyword name.
        new: The current keyword name.
        since: Release that introduced the deprecation (``YYYY.MM.DD``).
        remove_after: Earliest release or date the alias may be removed after.
        stacklevel: As for :func:`warn_deprecated`; the default attributes the
            warning to the caller of the function that owns ``kwargs``.

    Returns:
        The same mapping, for callers that prefer an expression.

    Raises:
        TypeError: If both ``old`` and ``new`` are present in ``kwargs``.

    Example:
        >>> def add_points(name, positions, **kwargs):
        ...     deprecated_kwarg_alias(
        ...         kwargs, "reveal_centre", "reveal_center",
        ...         since="2026.10.01", remove_after="2027.04.01",
        ...     )
        ...     reveal_center = kwargs.pop("reveal_center", None)
    """
    if old not in kwargs:
        return kwargs
    if new in kwargs:
        raise TypeError(f"got both {old!r} (deprecated) and {new!r}; pass only {new!r}")
    warn_deprecated(
        f"{old}=",
        f"{new}=",
        since=since,
        remove_after=remove_after,
        stacklevel=stacklevel,
    )
    kwargs[new] = kwargs.pop(old)
    return kwargs
