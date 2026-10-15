"""``luxar stop`` — stop every running Luxar server and demo.

A top-level command rather than ``luxar serve stop``: ``serve`` takes a
positional path, so a ``stop`` subcommand there would be read as a directory
named "stop". The discovery and kill engine is :mod:`luxar.cli.demo_runs`,
whose registry ``luxar serve``, ``luxar viewer``, ``luxar gsplat view`` and
``luxar export --open`` join as ``kind="server"`` entries; demos keep their own
``luxar demo stop``, which this command is a superset of.
"""

from __future__ import annotations

import typer
from arbol import aprint

from . import demo_runs
from .demo_commands import _translate_sweep_keys, list_confirm_stop


def register_stop_command(app: typer.Typer) -> None:
    """Attach ``luxar stop`` to the top-level CLI."""
    app.command("stop")(stop)


def stop(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List running servers and demos; stop nothing."
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip the confirmation prompt (required when stdin is not a terminal).",
    ),
) -> None:
    """Stop running Luxar servers and demos, freeing their ports.

    Covers servers started by `luxar serve`, `luxar viewer`, `luxar gsplat
    view` and `luxar export --open`, and every demo `luxar demo stop` sees. A
    server is signalled by PID (SIGINT, then SIGTERM, then SIGKILL), so nothing
    else in the terminal or script that started it is touched; a demo's whole
    process group goes, its own server included. Without a terminal, pass
    --yes (or --dry-run).
    """
    runs = _translate_sweep_keys(demo_runs.discover_runs(include_servers=True))
    if not runs:
        aprint("✅ No running Luxar servers or demos found.")
        raise typer.Exit(0)
    list_confirm_stop(
        runs,
        dry_run=dry_run,
        yes=yes,
        noun=("server/demo", "servers/demos"),
        done_message="✅ All stopped; their ports are free again.",
    )
