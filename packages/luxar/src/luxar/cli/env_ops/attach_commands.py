"""``luxar env attach`` — write a baked environment container into a scene."""

from __future__ import annotations

from pathlib import Path

import typer
from arbol import aprint

from ...environment.attach import attach_environment


def attach_command(
    store: Path = typer.Argument(..., help="The .luxar.zarr scene to attach to."),
    faces: Path = typer.Argument(..., help="The .env.bin container a bake produced."),
    force: bool = typer.Option(
        False,
        "--force",
        help="Attach even if the bake records a different scene content_hash.",
    ),
) -> None:
    """Attach a baked environment (the manual half of `luxar env bake`).

    Writes the six cube faces as environment/faces-<digest> (uint16 half bits)
    plus the capture header as attrs. The scene content_hash does not change,
    attaching the same map twice writes nothing, and a stale bake is refused
    unless --force.
    """
    try:
        report = attach_environment(store, faces, force=force)
    except (ValueError, FileNotFoundError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1) from exc
    aprint(
        f"✓ {report.status}: environment/{report.array_name} ({report.resolution}px)"
    )


def register_attach_commands(app: typer.Typer) -> None:
    """Attach the ``attach`` command to the env CLI."""
    app.command("attach")(attach_command)
