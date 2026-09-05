"""One way for a CLI command to fail, with a way to get the traceback back.

Fifteen `except Exception` blocks across the CLI did the same three things:
print a one-line message, then `raise typer.Exit(1)`. That is the right default
— a stack trace is noise when the cause is "file not found" — but there was no
way to opt out of it, so a genuine bug inside the library surfaced as one line
with nowhere to go next. Five of the fifteen did not even chain the original via
``from``, so the ``__cause__`` was gone too (audit finding ``A9-02``).

Two things change. Setting ``LUXAR_TRACEBACK=1`` re-raises the original
exception, traceback intact, and the message itself now says so — a hint nobody
reads in the docs is a hint nobody has.

An environment variable rather than a ``--traceback`` flag, deliberately: a
Typer callback option has to precede the subcommand
(``luxar --traceback gsplat fit …``), which is a poor shape for something you
reach for *after* a command has already failed. ``LUXAR_TRACEBACK=1 luxar
gsplat fit …`` re-runs the exact command you just typed.
"""

from __future__ import annotations

import os
from typing import NoReturn

import typer
from arbol import aprint

__all__ = [
    "TRACEBACK_ENV_VAR",
    "exit_with_error",
    "traceback_requested",
]

#: Set to any value other than the falsey spellings below to get a traceback.
TRACEBACK_ENV_VAR = "LUXAR_TRACEBACK"

_FALSEY = frozenset({"", "0", "false", "no", "off"})


def traceback_requested() -> bool:
    """Whether the caller asked for tracebacks via the environment.

    Returns:
        True unless :data:`TRACEBACK_ENV_VAR` is unset, empty, or one of
        ``0`` / ``false`` / ``no`` / ``off`` (case-insensitively). The falsey
        spellings matter because ``LUXAR_TRACEBACK=0`` in a shell profile
        should mean off, not "set, therefore on".
    """
    return os.environ.get(TRACEBACK_ENV_VAR, "").strip().lower() not in _FALSEY


def exit_with_error(message: str, error: BaseException) -> NoReturn:
    """Report a command failure and exit 1 — or re-raise for a traceback.

    Args:
        message: The one-line explanation, already formatted (including any
            emoji prefix the surrounding command uses).
        error: The caught exception. Chained onto the ``typer.Exit`` via
            ``from``, so ``__cause__`` survives even on the quiet path.

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
    if traceback_requested():
        raise error
    aprint(message)
    aprint(f"   (set {TRACEBACK_ENV_VAR}=1 and re-run for the full traceback)")
    raise typer.Exit(1) from error
