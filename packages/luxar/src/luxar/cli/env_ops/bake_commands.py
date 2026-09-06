"""``luxar env bake`` — capture a scene's environment headlessly and attach it."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint

# Only the constant at import time: `luxar.environment.bake` imports `luxar.cli.serving`,
# and the CLI package imports this module, so the bake function is imported in
# the command body to keep the cycle open.
from ...environment.container import DEFAULT_RESOLUTION


def bake_command(
    store: Path = typer.Argument(..., help="The .luxar.zarr scene to bake for."),
    probe: str = typer.Option(
        "auto",
        "--probe",
        help="Where the capture looks out from: auto | node:<path> | x,y,z.",
    ),
    resolution: int = typer.Option(
        DEFAULT_RESOLUTION,
        "--resolution",
        "-r",
        help="Cube face size in pixels (16-1024).",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Keep the container file here (default: temporary)."
    ),
    attach: bool = typer.Option(
        True, "--attach/--no-attach", help="Attach the result to the store when done."
    ),
    force: bool = typer.Option(
        False, "--force", help="Attach even if the bake's scene digest is stale."
    ),
    build: bool = typer.Option(
        False, "--build", help="Rebuild a stale viewer dist first (never by default)."
    ),
    timeout: float = typer.Option(
        300.0, "--timeout", help="Seconds to wait for the viewer to settle and bake."
    ),
) -> None:
    """Capture the scene-derived environment and store it as environment/faces.

    Serves the store and the built viewer, drives a headless browser to
    ?bake-env, collects the six half-float cube faces and (by default) attaches
    them with `luxar env attach`. Needs a development checkout: the driver is the
    viewer's Playwright (packages/luxar-viewer/scripts/bake-env.mjs).
    """
    from ...environment.bake import bake_environment

    try:
        report = bake_environment(
            store,
            probe=probe,
            resolution=resolution,
            out=out,
            attach=attach,
            force=force,
            build=build,
            timeout=timeout,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1) from exc
    if report.attach is not None:
        aprint(f"✓ {report.attach.status}: environment/{report.attach.array_name}")
    else:
        aprint(f"✓ container written to {report.container}")


def register_bake_commands(app: typer.Typer) -> None:
    """Attach the ``bake`` command to the env CLI."""
    app.command("bake")(bake_command)
