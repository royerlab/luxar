"""Aggregator for the ``luxar gsplat`` command group.

The command implementations live in ``cli/gsplat_ops/`` — one module per thematic
group (inspect / transforms / fitting / scene / benchmark / batch), plus the
``lod`` recipe command in ``cli/lod.py``. This module builds the ``app_gsplat``
Typer and registers each group onto it, keeping itself a thin registration
surface rather than a multi-thousand-line god-file.

A few symbols are re-exported for external consumers that import them from here:
``main.py`` mounts ``app_gsplat``; tests import ``info_dataset`` /
``batch_validate_cmd`` / ``_validate_tile``; ``_resolve_encoding_mode``
historically lived in this module.
"""

from __future__ import annotations

import typer

from .gsplat_ops.batch import _validate_tile, app_batch, batch_validate_cmd
from .gsplat_ops.benchmark import register_benchmark_commands
from .gsplat_ops.encoding import _resolve_encoding_mode
from .gsplat_ops.fitting import register_fitting_commands
from .gsplat_ops.inspect import info_dataset, register_inspect_commands
from .gsplat_ops.planner import register_planner_commands
from .gsplat_ops.scene import register_scene_commands
from .gsplat_ops.transforms import register_transforms_commands
from .lod import register_lod_command

app_gsplat = typer.Typer(help="Gaussian splat tools")

# Register each thematic command group (source-order preserved for --help).
register_inspect_commands(app_gsplat)
register_transforms_commands(app_gsplat)
register_fitting_commands(app_gsplat)
register_planner_commands(app_gsplat)
register_scene_commands(app_gsplat)
register_benchmark_commands(app_gsplat)
register_lod_command(app_gsplat)
app_gsplat.add_typer(app_batch, name="batch")

__all__ = [
    "app_gsplat",
    "info_dataset",
    "batch_validate_cmd",
    "_validate_tile",
    "_resolve_encoding_mode",
]
