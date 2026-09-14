"""Command-line flag parsers shared by Luxar demos."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, overload

from arbol import aprint

from ....utils.hosts import is_loopback_host


def parse_demo_flags() -> dict:
    """Parse common GSplat demo command-line flags from ``sys.argv``.

    Returns a dict with keys: ``recompute``, ``no_serve``, ``serve_only``,
    ``keep_stale``.
    """
    return {
        "recompute": "--recompute" in sys.argv,
        "no_serve": "--no-serve" in sys.argv,
        "serve_only": "--serve-only" in sys.argv,
        "keep_stale": "--keep-stale" in sys.argv,
    }


def _flag_token(name: str) -> str:
    """``--name`` for a flag called ``name``, tolerating pre-written dashes.

    :func:`parse_int_arg` and :func:`parse_path_arg` prepend the ``--``
    themselves, so a caller that passes ``"--points"`` used to make them search
    for ``----points`` — a flag that never matches, silently ignored, every run
    at the default. Normalising here removes that silent no-op for both helpers.
    """
    return f"--{name.lstrip('-')}"


@overload
def parse_int_arg(name: str, default: int, argv: Optional[list[str]] = ...) -> int: ...
@overload
def parse_int_arg(
    name: str, default: None, argv: Optional[list[str]] = ...
) -> Optional[int]: ...
def parse_int_arg(
    name: str, default: Optional[int], argv: Optional[list[str]] = None
) -> Optional[int]:
    """Parse an integer ``--name=VALUE`` or ``--name VALUE`` flag from argv.

    A tiny shared replacement for the ad-hoc ``sys.argv`` scanning every demo
    re-implements (``--points``, ``--sample``, ``--grid``, ``--frames``,
    ``--resolution``, …). Accepts both ``--name=8000`` and ``--name 8000``.
    Returns ``default`` when the flag is absent. The first occurrence is
    decisive, even if malformed: a malformed/unparseable value (e.g.
    ``--points=abc``) warns and returns ``default`` rather than raising — the
    shared helpers never abort a run over a mistyped flag.
    """
    args = list(sys.argv if argv is None else argv)
    flag = _flag_token(name)
    for i, arg in enumerate(args):
        raw: Optional[str] = None
        if arg.startswith(flag + "="):
            raw = arg.split("=", 1)[1]
        elif arg == flag and i + 1 < len(args):
            raw = args[i + 1]
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                aprint(f"Ignoring malformed {flag}={raw!r}; using {default}")
                return default
    return default


def parse_path_arg(name: str, argv: Optional[list[str]] = None) -> Optional[Path]:
    """Parse a path ``--name=PATH`` or ``--name PATH`` flag from argv.

    Sibling of :func:`parse_int_arg` for path-valued flags (``--cache-dir``,
    ``--data``). Expands a leading ``~``. Returns ``None`` when the flag is
    absent. An empty value (``--data=``) is not treated as a hit — the scan
    skips it and keeps looking, so a lone ``--data=`` reads as absent instead of
    resolving to the current directory (and a later non-empty occurrence wins).

    In the space form the next token must not itself look like an option:
    ``--data --no-tsp`` is a missing value, not a path named ``--no-tsp``, so it
    warns and keeps scanning rather than handing the demo a bogus file to open.
    The ``=`` form stays literal (``--data=--odd`` really does mean that path).
    """
    args = list(sys.argv if argv is None else argv)
    flag = _flag_token(name)
    for i, arg in enumerate(args):
        raw: Optional[str] = None
        if arg.startswith(flag + "="):
            raw = arg.split("=", 1)[1]
        elif arg == flag and i + 1 < len(args):
            if args[i + 1].startswith("--"):
                aprint(f"Ignoring {flag}: followed by {args[i + 1]!r}, not a path")
                continue
            raw = args[i + 1]
        if raw:
            return Path(raw).expanduser()
    return None


def parse_str_arg(name: str, argv: Optional[list[str]] = None) -> Optional[str]:
    """Parse a string ``--name=VALUE`` or ``--name VALUE`` flag from argv.

    Sibling of :func:`parse_int_arg` for free-text values (a shared secret, a
    host). Same conventions: first non-empty occurrence wins, and in the space
    form the next token must not look like an option, so ``--control-token
    --host`` reads as a missing value rather than a token spelled ``--host``.
    """
    args = list(sys.argv if argv is None else argv)
    flag = _flag_token(name)
    for i, arg in enumerate(args):
        raw: Optional[str] = None
        if arg.startswith(flag + "="):
            raw = arg.split("=", 1)[1]
        elif arg == flag and i + 1 < len(args):
            if args[i + 1].startswith("--"):
                aprint(f"Ignoring {flag}: followed by {args[i + 1]!r}, not a value")
                continue
            raw = args[i + 1]
        if raw:
            return raw
    return None


def control_serve_args(argv: Optional[list[str]] = None) -> list[str]:
    """``luxar serve`` arguments for a demo's opt-in remote-control panel.

    Returns ``[]`` unless ``--control`` is present, so remote control is OFF
    for every demo by default and no demo gains a listening socket a user did
    not ask for. Forward the result through ``launch_viewer(serve_args=...)``.

    Generic on purpose — it takes no demo-specific argument and lives beside
    the other shared flag parsers, so any demo opts in with one call rather
    than each re-deriving the argv scanning and the LAN advice.

    Recognised flags:

    ``--control``
        Expose the hub and the touch panel.
    ``--control-token TOKEN``
        Require ``TOKEN`` on every control socket. Strongly advised with
        ``--host``, which is the only configuration reachable from another
        machine.
    ``--host HOST``
        Bind address. Defaults to loopback, which no tablet can reach; pass
        ``0.0.0.0`` for a kiosk on a LAN.
    """
    args = list(sys.argv if argv is None else argv)
    if "--control" not in args:
        return []

    serve_args = ["--control"]
    token = parse_str_arg("control-token", args)
    if token:
        serve_args.extend(["--control-token", token])
    host = parse_str_arg("host", args)
    if host:
        serve_args.extend(["--host", host])
        # Only when the bind actually reaches the network. Warning on
        # `--host 127.0.0.1` would be a false alarm, and a warning that cries
        # wolf is one a kiosk operator learns to scroll past.
        if not token and not is_loopback_host(host):
            aprint(
                "⚠️  --control with --host and no --control-token: anything that "
                "can reach this machine on the network can drive the display. "
                "Pass --control-token to restrict it."
            )
    return serve_args
