"""Turn Luxar's console output down, or off.

Luxar's library layers narrate what they are doing through arbol — 547 ``aprint``
calls below the CLI, in ``gsplats`` (314), ``io`` (146) and ``core`` (75). That
is the right default for a long CLI run, and the wrong one for a notebook cell
or a napari plugin, where the same tree scrolls past the result you asked for.
Until this module there was no way to say so: no ``verbose`` or ``quiet``
parameter on ``Scene.save()`` or ``fit_gaussian_splats()``, and no global switch
that was part of Luxar's own API (audit finding ``A15-08``).

Two entry points, one mechanism::

    import luxar

    luxar.set_verbosity("silent")     # process-wide, until changed again
    with luxar.verbosity("summary"):  # scoped, restores on exit
        scene.save()

Levels, and exactly what each one does:

======================  ===================================================
``"silent"``            Normal arbol narration is suppressed; Python warnings
                        still surface via their standard display.
``"summary"``           Top-level lines and one level of nesting; deeper
                        sections become truncation notices (depth 1).
``"normal"``            Three levels of nesting (depth 3).
``"full"``              Everything. **The default** — unchanged behaviour.
an ``int``              That many levels of nesting. ``0`` is not silent (see
                        below); use ``"silent"``.
======================  ===================================================

Implementation, stated plainly because it constrains how you can use this: these
functions set ``arbol.Arbol.enable_output`` and ``arbol.Arbol.max_depth``, which
are **class attributes**. So:

* The setting is **process-global**, not per-call and not per-object. A
  ``verbosity()`` block around one ``save()`` also quiets anything else running
  concurrently.
* It is **not thread-safe**. Two threads entering different ``verbosity()``
  blocks will interleave and the loser's restore will win. The switches and
  arbol's depth counter are shared class state.
* It changes the shared settings for **all** arbol users in the process,
  including other libraries. There is no Luxar-only namespace to scope it to.

A per-call ``verbosity=`` argument on the public entry points, and routing
library messages through ``logging`` so a handler could filter by module, are
both possible later; neither is needed to make the output silenceable, which is
what was actually missing.

``max_depth=0`` is deliberately *not* the silent level: arbol still prints
depth-0 lines plus a "(log tree truncated here)" notice for each section it
truncates at the cap, so a caller asking for silence and getting truncation
notices would be worse than the status quo. Measured, not assumed — see
``tests/test_verbosity.py::TestTheDocumentedEffectsAreReal::test_depth_zero_is_not_silence``.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Literal, Tuple, Union

from arbol import Arbol, aprint

__all__ = [
    "VerbosityLevel",
    "aprint_heartbeat",
    "get_verbosity",
    "quiet_narration",
    "set_verbosity",
    "verbosity",
]

VerbosityLevel = Literal["silent", "summary", "normal", "full"]

#: Level name -> (``Arbol.enable_output``, ``Arbol.max_depth``).
#:
#: ``"full"`` must stay exactly arbol's own defaults: it is what every existing
#: caller gets, so a change here is a change in behaviour for code that never
#: asked for one.
_LEVELS: dict[str, Tuple[bool, float]] = {
    "silent": (False, math.inf),
    "summary": (True, 1),
    "normal": (True, 3),
    "full": (True, math.inf),
}


def _resolve(level: Union[VerbosityLevel, int]) -> Tuple[bool, float]:
    """Map a level name or explicit depth onto the two arbol switches."""
    if isinstance(level, bool):
        # `bool` is an `int` subclass, and `set_verbosity(False)` reads as
        # "quiet" while resolving to depth 0, which is not silent. Refuse it
        # rather than do the surprising thing.
        raise TypeError(
            "set_verbosity() takes a level name or an int depth, not a bool. "
            'Use set_verbosity("silent") or set_verbosity("full").'
        )
    if isinstance(level, int):
        if level < 0:
            raise ValueError(f"Verbosity depth must be >= 0, got {level}")
        return (True, level)
    try:
        return _LEVELS[level]
    except KeyError:
        raise ValueError(
            f"Unknown verbosity level {level!r}. "
            f"Expected one of {sorted(_LEVELS)} or an int depth."
        ) from None


def set_verbosity(level: Union[VerbosityLevel, int]) -> None:
    """Set how much Luxar narrates, process-wide.

    Args:
        level: ``"silent"``, ``"summary"``, ``"normal"``, ``"full"``, or an int
            giving the maximum section nesting depth to show. ``0`` shows
            top-level lines only and is **not** silence — pass ``"silent"``.

    Raises:
        ValueError: If ``level`` is an unknown name or a negative depth.
        TypeError: If ``level`` is a bool.

    Example:
        >>> import luxar
        >>> luxar.set_verbosity("silent")
        >>> luxar.set_verbosity("full")  # back to the default

    See the module docstring for the scope and thread-safety caveats: this
    writes process-global arbol state.
    """
    enable_output, max_depth = _resolve(level)
    if _QUIET_OUTER:
        # Inside quiet_narration(): the level governs what its heartbeats (and
        # everything after the block) show; the block's narration stays muted.
        _QUIET_OUTER[-1] = (enable_output, max_depth, _QUIET_OUTER[-1][2])
        Arbol.max_depth = max_depth
        return
    Arbol.enable_output = enable_output
    Arbol.max_depth = max_depth


def get_verbosity() -> Union[VerbosityLevel, int]:
    """Return the current verbosity as a level name, or an int depth.

    Output disabled reports ``"silent"`` regardless of the current depth.
    Otherwise a name is returned when the live arbol settings match a named
    level exactly, and the raw depth is returned when they do not. Consequently,
    ``set_verbosity(get_verbosity())`` is lossy for a manually disabled finite
    depth; :func:`verbosity` still restores the exact switch pair.

    Returns:
        One of the level names, or the current maximum depth as an int.
    """
    current = (Arbol.enable_output, Arbol.max_depth)
    for name, settings in _LEVELS.items():
        if current == settings:
            return name  # type: ignore[return-value]
    if not Arbol.enable_output:
        return "silent"
    return int(Arbol.max_depth)


@contextmanager
def verbosity(level: Union[VerbosityLevel, int]) -> Iterator[None]:
    """Set the verbosity for the duration of a block, then restore it.

    Restores the exact previous ``enable_output`` / ``max_depth`` pair rather
    than the level name it resolves to, so a caller who had set
    ``Arbol.max_depth`` by hand gets their own value back.

    Args:
        level: As :func:`set_verbosity`.

    Yields:
        Nothing.

    Example:
        >>> import luxar
        >>> with luxar.verbosity("silent"):
        ...     pass  # nothing this block does will print

    Not thread-safe — see the module docstring.
    """
    previous = (Arbol.enable_output, Arbol.max_depth)
    quiet_top = (len(_QUIET_OUTER), _QUIET_OUTER[-1]) if _QUIET_OUTER else None
    set_verbosity(level)
    try:
        yield
    finally:
        Arbol.enable_output, Arbol.max_depth = previous
        if quiet_top is not None and len(_QUIET_OUTER) == quiet_top[0]:
            _QUIET_OUTER[-1] = quiet_top[1]


#: Arbol ``(enable_output, max_depth, depth)`` outside each active
#: :func:`quiet_narration` block, innermost last.
_QUIET_OUTER: list[tuple[bool, float, int]] = []


@contextmanager
def quiet_narration() -> Iterator[None]:
    """Silence arbol for a block, except lines sent through :func:`aprint_heartbeat`.

    For a driver that runs many inner operations whose own narration would bury
    the run (a content-tiled fit's per-box fits), but still wants a few progress
    lines out of them. Python warnings keep surfacing through their standard
    display. Same process-global, not-thread-safe caveats as :func:`verbosity`.

    A :func:`set_verbosity` / :func:`verbosity` call made INSIDE the block sets
    the level its heartbeats print at (so a nested ``verbosity("silent")``
    silences them too) while the block's own narration stays muted.

    Yields:
        Nothing.
    """
    _QUIET_OUTER.append(
        (Arbol.enable_output, Arbol.max_depth, int(getattr(Arbol, "_depth", 0)))
    )
    Arbol.enable_output = False
    try:
        yield
    finally:
        Arbol.enable_output, Arbol.max_depth, _ = _QUIET_OUTER.pop()


def aprint_heartbeat(message: str) -> None:
    """``aprint`` that an enclosing :func:`quiet_narration` block does not silence.

    Inside such a block the line is printed under the settings in force OUTSIDE
    it, at its tree depth — so ``luxar.set_verbosity("silent")`` still silences
    it, and the line sits beside the driver's own lines rather than under inner
    sections that never printed. Outside any block it is a plain ``aprint``.

    Args:
        message: The line to print.
    """
    if not _QUIET_OUTER:
        aprint(message)
        return
    enable_output, max_depth, depth = _QUIET_OUTER[-1]
    inner = (Arbol.enable_output, Arbol.max_depth, int(getattr(Arbol, "_depth", 0)))
    Arbol.enable_output, Arbol.max_depth = enable_output, max_depth
    Arbol._depth = depth  # arbol keeps counting depth while output is off
    try:
        aprint(message)
    finally:
        Arbol.enable_output, Arbol.max_depth = inner[0], inner[1]
        Arbol._depth = inner[2]
