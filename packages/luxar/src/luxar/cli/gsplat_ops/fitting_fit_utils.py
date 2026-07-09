"""Shared utility helpers for gsplat fit command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import typer
from arbol import aprint

from ..utils import format_memory_size
from .fitting_recipe_args import (
    build_fit_recipe_params as _build_fit_recipe_params_impl,
)

if TYPE_CHECKING:
    pass


# Whole-volume voxel budget for ``--tiling auto``: below this a volume fits
# whole (no tiling), avoiding spurious tile seams on small/medium stacks. ~4x a
# 256^3 tile — a 28M-voxel neuromast stack fits whole; gigavoxel volumes tile.
_AUTO_WHOLE_VOLUME_VOXELS = 64_000_000


def resolve_tiling(
    tiling: str, shape: "tuple[int, ...]", tile_size: int, has_density: bool
) -> str:
    """Resolve ``--tiling`` to a concrete strategy: ``none | uniform | content``.

    ``auto`` fits the whole volume unless it is genuinely large — i.e. some
    dimension exceeds ``tile_size`` AND the total voxel count exceeds
    :data:`_AUTO_WHOLE_VOLUME_VOXELS`. A single dim over ``tile_size`` is not
    enough on its own (that needlessly tiled small stacks and produced visible
    background seams). When tiling IS selected, ``content`` is used when a
    transferable density (``--cal`` / ``--k-star-ref`` / ``--plan``) is
    available, else ``uniform``.
    """
    t = tiling.lower()
    if t not in ("auto", "none", "uniform", "content"):
        raise typer.BadParameter(
            f"--tiling must be one of auto|none|uniform|content, got {tiling!r}"
        )
    if t != "auto":
        return t
    n_voxels = 1
    for s in shape:
        n_voxels *= int(s)
    exceeds_dim = any(int(s) > int(tile_size) for s in shape)
    large = exceeds_dim and n_voxels > _AUTO_WHOLE_VOLUME_VOXELS
    if not large:
        return "none"
    return "content" if has_density else "uniform"


def build_fit_recipe_params(
    recipe: str,
    *,
    n_lods: Optional[int],
    additive_method: Optional[str],
    breakpoints: Optional[str],
    target_ms: Optional[float] = None,
    bandwidth_mbps: Optional[float] = None,
    bytes_per_splat: Optional[float] = None,
    compression_factor: Optional[int],
    levels: Optional[int],
    substitutive_method: Optional[str],
    coarsen_dims: Optional[str],
    device: Optional[str],
    volume_ndim: int,
) -> "Any":
    """Build optional fit-time LOD recipe parameters from CLI arguments."""
    return _build_fit_recipe_params_impl(
        recipe,
        n_lods=n_lods,
        additive_method=additive_method,
        breakpoints=breakpoints,
        target_ms=target_ms,
        bandwidth_mbps=bandwidth_mbps,
        bytes_per_splat=bytes_per_splat,
        compression_factor=compression_factor,
        levels=levels,
        substitutive_method=substitutive_method,
        coarsen_dims=coarsen_dims,
        device=device,
        volume_ndim=volume_ndim,
    )


def save_fit_output(
    result: Any,
    output_path: Path,
    *,
    compress: "Optional[Literal['zip', 'tar.gz']]",
    verbose: bool,
) -> int:
    """Save a flat ``GSplatData`` leaf or a ``kind=partition`` tree node.

    Returns the splat count for the summary line.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    if isinstance(result, GSplatData):
        result.save(output_path, compress=compress)
        n = int(result.n_splats)
    else:  # a partition / tree node has no flat-matrix equivalent
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        write_gsplats_tree(output_path, result, compress=compress)
        n = int(getattr(result, "n_splats", 0))
    if verbose:
        aprint(f"Saved {n:,} splats")
        if output_path.exists():
            aprint(f"File size: {format_memory_size(output_path.stat().st_size)}")
    return n
