"""Aggregator for the ``luxar gsplat`` command group.

The command implementations live in ``cli/gsplat_ops/`` — one module per thematic
group (inspect / transforms / fitting / scene / benchmark / batch), plus the
``lod`` recipe command in ``cli/lod.py``. This module builds the ``app_gsplat``
Typer and registers each group onto it, keeping itself a thin registration
surface rather than a multi-thousand-line god-file.

``main.py`` mounts ``app_gsplat``.
"""

from __future__ import annotations

import typer

from .gsplat_ops.batch_commands import app_batch
from .gsplat_ops.benchmark import register_benchmark_commands
from .gsplat_ops.fitting import register_fitting_commands
from .gsplat_ops.inspect_commands import register_inspect_commands
from .gsplat_ops.scene_commands import register_scene_commands
from .gsplat_ops.transforms_commands import register_transforms_commands
from .lod import register_lod_command

app_gsplat = typer.Typer(
    help=(
        "Gaussian splat tools — fit nD images/volumes (.tiff/.npy/.npz/.zarr) to "
        "oriented Gaussians and view them.\n\n"
        "Getting started:\n"
        "  luxar demo                                   # quick end-to-end demo\n"
        "  luxar gsplat fit stack.tiff out.gsplats.zarr # fit a volume (auto settings)\n"
        "  luxar gsplat view out.gsplats.zarr           # open it in the web viewer\n\n"
        "Non-canonical axis order? pass --axes (e.g. 'z,c,y,x') to fit/cal. "
        "Whole timelapse / large data? see `batch-fit` (`run` = local multi-GPU, "
        "`submit` = Slurm)."
    )
)

# Register each thematic command group. Registration order = --help display
# order, so lead with the core workflow (fit -> cal -> lod -> scene/view ->
# inspect), then editing ops, then cluster/benchmark.
register_fitting_commands(app_gsplat)  # fit, cal, render, denoise
register_lod_command(app_gsplat)  # lod
register_scene_commands(app_gsplat)  # convert, migrate-format
register_inspect_commands(app_gsplat)  # info, view, napari, compare
register_transforms_commands(
    app_gsplat
)  # transform, cull, filter, slice, merge, partition
register_benchmark_commands(app_gsplat)  # benchmark
app_gsplat.add_typer(app_batch, name="batch-fit")  # run (local) + submit (Slurm)

__all__ = [
    "app_gsplat",
]
