"""The ``luxar mesh`` command group.

A pure registration surface, matching ``gsplat_commands.py``: the Typer object plus a
series of ``register_*_commands`` calls. Command bodies live in ``cli/mesh_ops/``.
"""

import typer

from .mesh_ops import register_import_commands

app_mesh = typer.Typer(
    help=(
        "Triangle-mesh tools — bring classical surface files into Luxar.\n\n"
        "Getting started:\n"
        "  luxar mesh import bunny.ply bunny.luxar.zarr   # PLY/OBJ/STL/glTF -> a scene\n"
        "  luxar serve bunny.luxar.zarr --viewer          # open it in the browser\n"
    )
)

register_import_commands(app_mesh)  # import

__all__ = ["app_mesh"]
