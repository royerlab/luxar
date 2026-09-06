"""The ``luxar env`` command group.

A pure registration surface, matching ``mesh_commands.py``: the Typer object plus
``register_*_commands`` calls. Command bodies live in ``cli/env_ops/``.
"""

import typer

from .env_ops import register_attach_commands, register_bake_commands

app_env = typer.Typer(
    help=(
        "Scene environments for material='physical' meshes — bake the scene's own "
        "light into the store.\n\n"
        "Getting started:\n"
        "  luxar env bake scene.luxar.zarr                       # capture + attach\n"
        "  luxar env bake scene.luxar.zarr --probe node:shell    # capture from a node\n"
        "  luxar env attach scene.luxar.zarr scene.env.bin       # the manual half\n"
    )
)

register_bake_commands(app_env)  # bake
register_attach_commands(app_env)  # attach

__all__ = ["app_env"]
