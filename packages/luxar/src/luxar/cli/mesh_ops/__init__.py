"""Thematic command groups for ``luxar mesh``.

Mirrors ``cli/gsplat_ops/``: each module exposes a ``register_*_commands(app)`` that
attaches its commands to the shared ``app_mesh`` Typer, so registration order in
``mesh_commands.py`` is also ``--help`` display order.
"""

from .import_commands import register_import_commands
from .lod_commands import register_lod_commands

__all__ = ["register_import_commands", "register_lod_commands"]
