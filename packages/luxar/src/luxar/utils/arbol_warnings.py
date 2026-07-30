"""Route Python warning *display* through arbol console output.

Python's default ``warnings.showwarning`` writes
``/abs/path/module.py:299: UserWarning: ...`` straight to stderr, which lands
mid-tree and out of place in arbol's hierarchical console output. This module
re-routes only the *display* of warnings to :func:`arbol.aprint` — the
``warnings.warn`` machinery itself is untouched, so filters, ``-W error``,
``warnings.catch_warnings`` and ``pytest.warns`` all behave exactly as before.

Two entry points:

- :func:`install_arbol_warnings` — process-wide install for application entry
  points (the ``luxar`` CLI callback).
- :func:`arbol_warnings` — a context manager / decorator that scopes the
  override to a block (used on the compiler's public write methods so
  Python-API scene builds get arbol-styled warnings too).

Both engage only when Python's *default* display path is active. If anything
else owns warning display — a user-installed ``showwarning`` hook, or a
recorder such as ``warnings.catch_warnings(record=True)`` / ``pytest.warns``
(which capture via ``warnings._showwarnmsg_impl`` while leaving
``showwarning`` set to the default) — the override steps aside so recorded
warnings are never diverted away from their catcher.
"""

from __future__ import annotations

import os.path
import warnings
from contextlib import contextmanager
from typing import Iterator, Optional, TextIO, Type

from arbol import aprint

__all__ = ["arbol_warnings", "install_arbol_warnings"]


def _arbol_showwarning(
    message: Warning | str,
    category: Type[Warning],
    filename: str,
    lineno: int,
    file: Optional[TextIO] = None,
    line: Optional[str] = None,
) -> None:
    """``warnings.showwarning`` replacement that prints via arbol."""
    location = f"{os.path.basename(filename)}:{lineno}"
    aprint(f"⚠️  {category.__name__}: {message} [{location}]")


def _default_display_active() -> bool:
    """True when Python's stock warning display is in effect.

    Both checks are needed: a custom hook rebinds ``warnings.showwarning``,
    but ``catch_warnings(record=True)`` (and therefore ``pytest.warns`` and
    pytest's per-test capture) records via ``warnings._showwarnmsg_impl``
    while resetting ``showwarning`` to the default — overriding
    ``showwarning`` there would steal warnings from the recorder.
    """
    # Python <= 3.12 defines the stock displayers in `warnings`; 3.13+ moved
    # the implementation to `_py_warnings` (re-exported from `warnings`).
    stock_modules = {"warnings", "_py_warnings"}
    if getattr(warnings.showwarning, "__module__", None) not in stock_modules:
        return False
    impl = getattr(warnings, "_showwarnmsg_impl", None)
    if impl is not None and getattr(impl, "__module__", None) not in stock_modules:
        return False
    return True


@contextmanager
def arbol_warnings() -> Iterator[None]:
    """Display warnings raised inside the block via :func:`arbol.aprint`.

    Usable as a decorator (``@arbol_warnings()``). No-op when warning display
    is already owned by someone else (see :func:`_default_display_active`).
    """
    if not _default_display_active():
        yield
        return
    previous = warnings.showwarning
    warnings.showwarning = _arbol_showwarning
    try:
        yield
    finally:
        # Restore only if nobody re-bound it underneath us (e.g. a nested
        # catch_warnings that exits after this block would restore itself).
        if warnings.showwarning is _arbol_showwarning:
            warnings.showwarning = previous


def install_arbol_warnings() -> None:
    """Process-wide install for application entry points (the luxar CLI).

    Skipped when a recorder or custom hook already owns warning display, so
    in-process CLI test harnesses (``CliRunner`` under ``pytest.warns``) keep
    capturing warnings normally.
    """
    if _default_display_active():
        warnings.showwarning = _arbol_showwarning
