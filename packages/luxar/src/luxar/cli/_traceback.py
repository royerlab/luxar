"""One way for a CLI command to fail, with a way to get the traceback back.

CLI handlers use this helper to print a one-line message, then
`raise typer.Exit(1)`. That is the right default — a stack trace is noise when
the cause is "file not found" — while a genuine bug still needs a way to expose
its origin. The first seventeen routed sites included eight that discarded the
original chain entirely (audit finding ``A9-02``); the remaining interactive
handlers previously printed tracebacks unconditionally (#2553).

Setting ``LUXAR_TRACEBACK=1`` re-raises the original exception, traceback
intact, and the message itself says so — a hint nobody reads in the docs is a
hint nobody has. Fatal lines fall back to stderr when arbol verbosity hides
normal narration, so the quiet path cannot become a silent exit 1.

An environment variable rather than a ``--traceback`` flag, deliberately: a
Typer callback option has to precede the subcommand
(``luxar --traceback gsplat fit …``), which is a poor shape for something you
reach for *after* a command has already failed. ``LUXAR_TRACEBACK=1 luxar
gsplat fit …`` re-runs the exact command you just typed.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn

import typer
from arbol import Arbol, aprint

__all__ = [
    "TRACEBACK_ENV_VAR",
    "exit_with_error",
    "report_error",
    "traceback_requested",
]

#: Set to any value other than the falsey spellings below to get a traceback.
TRACEBACK_ENV_VAR = "LUXAR_TRACEBACK"

_FALSEY = frozenset({"", "0", "false", "no", "off"})


def _report_line(message: str) -> None:
    """Print a fatal line through arbol, or stderr when arbol hides it."""
    captured = getattr(Arbol._thread_local, "captured", False)
    arbol_will_display = (
        Arbol.passthrough
        or captured
        or (Arbol.enable_output and Arbol._depth <= Arbol.max_depth)
    )
    if arbol_will_display:
        aprint(message)
    else:
        print(message, file=sys.stderr)


def traceback_requested() -> bool:
    """Whether the caller asked for tracebacks via the environment.

    Returns:
        True unless :data:`TRACEBACK_ENV_VAR` is unset, empty, or one of
        ``0`` / ``false`` / ``no`` / ``off`` (case-insensitively). The falsey
        spellings matter because ``LUXAR_TRACEBACK=0`` in a shell profile
        should mean off, not "set, therefore on".
    """
    return os.environ.get(TRACEBACK_ENV_VAR, "").strip().lower() not in _FALSEY


def report_error(message: str, error: BaseException) -> None:
    """Report a recoverable command failure — or re-raise for a traceback.

    Args:
        message: The one-line explanation, already formatted (including any
            emoji prefix the surrounding command uses). The traceback path
            does not print it, so the exception itself must carry any context
            essential to diagnosing the failure.
        error: The caught exception.

    Raises:
        BaseException: ``error`` itself, unchanged, when
            :func:`traceback_requested`.
    """
    if traceback_requested():
        raise error
    _report_line(message)
    _report_line(f"   (set {TRACEBACK_ENV_VAR}=1 and re-run for the full traceback)")


def exit_with_error(message: str, error: BaseException) -> NoReturn:
    """Report a command failure and exit 1 — or re-raise for a traceback.

    Args:
        message: Passed to :func:`report_error`.
        error: Passed to :func:`report_error`, then chained onto the
            ``typer.Exit`` so ``__cause__`` survives on the quiet path.

    Raises:
        BaseException: ``error`` itself, unchanged, when
            :func:`traceback_requested`.
        typer.Exit: Otherwise, with code 1.

    Example:
        >>> try:  # doctest: +SKIP
        ...     do_the_thing()
        ... except Exception as e:
        ...     exit_with_error(f"❌ Error doing the thing: {e}", e)
    """
    report_error(message, error)
    raise typer.Exit(1) from error
