"""Decide whether Luxar's arbol narration may use ANSI colour.

arbol colours every line by default, so a run redirected to a file
(``luxar demo run cloud > log 2>&1``) used to fill the log with escape codes,
and ``NO_COLOR`` was ignored. The policy, applied at ``import luxar`` and again
at the start of every CLI command:

1. ``FORCE_COLOR`` set to anything but empty or ``0`` keeps colour on.
2. ``NO_COLOR`` set to anything non-empty turns it off (https://no-color.org).
3. Otherwise colour is turned off when stdout is a REDIRECTED stream: a
   standard text stream (``io.TextIOWrapper``) that is not a terminal. A
   notebook's output stream is not one of those, so Jupyter keeps arbol's
   colours, which it renders.

Rich (Typer's help and usage panels) already follows the same environment
variables on its own, so only arbol needs this.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any, Optional

from arbol import Arbol

__all__ = ["apply_arbol_color_policy", "color_env_override"]


def color_env_override() -> Optional[bool]:
    """``True`` for ``FORCE_COLOR``, ``False`` for ``NO_COLOR``, else ``None``.

    ``FORCE_COLOR`` wins when both are set: it is the more specific request.
    """
    force = os.environ.get("FORCE_COLOR", "").strip()
    if force and force != "0":
        return True
    if os.environ.get("NO_COLOR", ""):
        return False
    return None


def _is_redirected(stream: Any) -> bool:
    """Whether ``stream`` is a plain text stream that is not a terminal."""
    if not isinstance(stream, io.TextIOWrapper):
        return False
    try:
        return not stream.isatty()
    except (ValueError, OSError):  # closed or detached
        return True


def apply_arbol_color_policy(stream: Any = None) -> None:
    """Set ``arbol.Arbol.colorful`` from the environment and ``stream``.

    Args:
        stream: The stream arbol writes to; ``sys.stdout`` when omitted.

    Only ever turns colour OFF, except under ``FORCE_COLOR``, so a caller that
    disabled it explicitly (``Arbol.colorful = False``) is never overridden.
    """
    override = color_env_override()
    if override is not None:
        Arbol.colorful = override
    elif _is_redirected(sys.stdout if stream is None else stream):
        Arbol.colorful = False
