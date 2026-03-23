# fit_tiled_gsplats.py
"""Tiled fitting for large volumes that exceed GPU memory.

Splits a volume into overlapping tiles with cosine (Hann) apodization,
fits Gaussian splats independently per tile, and concatenates results.
The Hann partition-of-unity property ensures seamless blending without
post-merge pruning.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np
from arbol import aprint, asection

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD
from luxar.gsplats.tiling import TileSpec, compute_tile_specs, cosine_window


def fit_tile(
    volume: Any,
    spec: TileSpec,
    voxel_size: Optional[Sequence[float] | float] = None,
    output_space: str = "real",
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    **fit_kwargs: Any,
) -> GSplatData:
    """Fit Gaussian splats on a single tile of a larger volume.

    Extracts the tile subvolume, applies cosine apodization, fits splats,
    and translates centers to global volume coordinates. This is the atomic
    unit for tiled fitting — each call is independent and Slurm-ready.

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume (or lazy zarr array). Only the tile's slice is materialized
        into memory via ``volume[spec.slices]``.
    spec : TileSpec
        Tile specification from :func:`compute_tile_specs`. Contains the
        per-face overlap sizes used for cosine window construction.
    voxel_size : float or sequence of float, optional
        Physical voxel spacing. Passed through to :func:`fit_gaussian_splats`
        and used for correct center translation when ``output_space="real"``.
    output_space : str, default "real"
        Coordinate space for output centers (``"real"`` or ``"voxel"``).
    progressive : bool, default False
        If True, use progressive fitting (multiple passes on residuals)
        instead of standard single-pass fitting. Produces multi-LOD result.
    max_splats_per_pass : int, default 5000
        Maximum splats per progressive pass (ignored if progressive=False).
    psnr_patience : float, default 0.5
        Stop progressive passes if ΔPSNR < this value in dB.
    max_passes : int, optional
        Maximum number of progressive passes (None = unlimited).
    **fit_kwargs
        All other keyword arguments forwarded to the fitting function.

    Returns
    -------
    GSplatData
        Fit result with centers in global volume coordinates.
        Multi-LOD if progressive=True.

    Raises
    ------
    ValueError
        If ``seeds`` in fit_kwargs is an explicit np.ndarray (not supported
        with tiled fitting; use int, float, or None instead).
    """
    # Reject explicit seed arrays — they don't make sense per-tile
    seeds = fit_kwargs.get("seeds")
    if isinstance(seeds, np.ndarray):
        raise ValueError(
            "Explicit seed coordinate arrays are not supported with tiled fitting. "
            "Use an integer (count per tile), float (compression ratio), or None (auto)."
        )

    # 1. Extract tile subvolume (materializes from zarr if needed)
    tile_data = np.asarray(volume[spec.slices], dtype=np.float32)

    # 2. Apply cosine apodization window
    window = cosine_window(spec)
    tile_data = tile_data * window

    # 3. Skip fitting if tile has negligible signal (e.g., windowed to near-zero)
    if tile_data.max() < 1e-8:
        from luxar.gsplats.utils.trils import tril_size

        ndim = tile_data.ndim
        result = GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros((0,), dtype=np.float32),
            cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
            stats={"time_seconds": 0.0, "skipped": True},
        )
    elif progressive:
        from luxar.gsplats.fit_progressive_gsplats import (
            fit_progressive_gaussian_splats,
        )

        # Use seeds as max_splats budget for progressive fitting (don't pop — shared dict)
        prog_max_splats = fit_kwargs.get("seeds", 5000)
        if isinstance(prog_max_splats, float):
            # Compression ratio — let progressive handle it as seed count
            prog_max_splats = max(100, int(prog_max_splats * np.prod(tile_data.shape)))
        elif prog_max_splats is None:
            prog_max_splats = 5000

        # Progressive fitting always works in voxel space internally
        # (it pops output_space/voxel_size from kwargs), so we don't
        # pass them here. The tile offset translation handles coordinate
        # space conversion.
        # Map n_iters → iters_per_pass (progressive uses its own param name)
        # Check iters_per_pass first (direct callers), then n_iters (CLI path)
        prog_iters = fit_kwargs.pop("iters_per_pass", None)
        if prog_iters is None:
            prog_iters = fit_kwargs.pop("n_iters", 1000)
        result = fit_progressive_gaussian_splats(
            tile_data,
            max_splats=prog_max_splats,
            max_splats_per_pass=max_splats_per_pass,
            iters_per_pass=prog_iters,
            psnr_patience=psnr_patience,
            max_passes=max_passes,
            **fit_kwargs,
        )
    else:
        result = fit_gaussian_splats(
            tile_data,
            voxel_size=voxel_size,
            output_space=output_space,
            **fit_kwargs,
        )

    # 4. Translate centers from tile-local to global coordinates
    if result.n_splats > 0:
        origin = np.array(spec.origin, dtype=np.float32)
        # Progressive fitting always returns voxel-space centers (it pops
        # output_space/voxel_size internally), so the offset must be in
        # voxel space regardless of the caller's output_space.
        if not progressive and output_space == "real" and voxel_size is not None:
            vs = np.broadcast_to(
                np.asarray(voxel_size, dtype=np.float32), (len(origin),)
            )
            offset = origin * vs
        else:
            offset = origin

        if result.n_lods > 1:
            # LOD-aware translate: translate each LOD individually to
            # preserve the multi-LOD structure (translate() flattens LODs).
            translated_lods = [
                GSplatLOD(
                    centers=lod.centers + offset,
                    amplitudes=lod.amplitudes,
                    cholesky_factors=lod.cholesky_factors,
                    colors=lod.colors,
                    stats=dict(lod.stats),
                )
                for lod in result.lods
            ]
            result = GSplatData.from_lods(translated_lods, stats=dict(result.stats))
        else:
            result = result.translate(offset)

    # Tag tile info in stats
    result.stats["tile_index"] = spec.index
    result.stats["tile_grid_index"] = spec.grid_index
    result.stats["tile_origin"] = spec.origin

    return result


def fit_tiled(
    volume: Any,
    tile_size: int | Sequence[int] = 256,
    overlap: int | Sequence[int] = 32,
    voxel_size: Optional[Sequence[float] | float] = None,
    output_space: str = "real",
    verbose: bool = True,
    progressive: bool = False,
    max_splats_per_pass: int = 5000,
    psnr_patience: float = 0.5,
    max_passes: Optional[int] = None,
    **fit_kwargs: Any,
) -> GSplatData:
    """Fit Gaussian splats to a large volume using tiled decomposition.

    Splits the volume into overlapping tiles with cosine apodization
    (Hann window), fits each tile independently, and merges results.
    The Hann partition-of-unity property guarantees seamless blending.

    When ``progressive=True``, each tile is fitted using progressive
    residual decomposition, producing a multi-LOD result where LODs are
    merged across tiles (LOD 0 = all tiles' coarse splats, etc.).

    Parameters
    ----------
    volume : np.ndarray or zarr.Array
        Full volume. Can be a lazy zarr array for out-of-core processing —
        only one tile at a time is materialized in memory.
    tile_size : int or tuple of int, default 256
        Tile size per axis in voxels. Scalar is broadcast to all axes.
    overlap : int or tuple of int, default 32
        Overlap width per axis in voxels. Scalar is broadcast.
    voxel_size : float or sequence of float, optional
        Physical voxel spacing, forwarded to per-tile fitting.
    output_space : str, default "real"
        Coordinate space for output centers (``"real"`` or ``"voxel"``).
    verbose : bool, default True
        Print per-tile progress with arbol.
    progressive : bool, default False
        Use progressive fitting per tile (multi-LOD output).
    max_splats_per_pass : int, default 5000
        Maximum splats per progressive pass (ignored if progressive=False).
    psnr_patience : float, default 0.5
        Stop progressive passes if ΔPSNR < this value in dB.
    max_passes : int, optional
        Maximum number of progressive passes (None = unlimited).
    **fit_kwargs
        All other keyword arguments forwarded to the per-tile fitting function
        (e.g. ``seeds``, ``n_iters``, ``preset``, ``device``).

    Returns
    -------
    GSplatData
        Merged result with all splats in global coordinates.
        Multi-LOD if progressive=True.

    Notes
    -----
    **GPU utilization with progressive**: When ``progressive=True``, each
    per-pass fit uses fewer splats (``max_splats_per_pass``), which may
    under-saturate the GPU.  For batch/Slurm jobs, combine
    ``--progressive`` with ``--parallel`` to run multiple tiles concurrently
    on the same GPU and improve throughput.
    """
    # Propagate verbose to per-tile fit_gaussian_splats unless caller
    # explicitly provided a different value in fit_kwargs.
    fit_kwargs.setdefault("verbose", verbose)

    volume_shape = tuple(volume.shape)
    specs = compute_tile_specs(volume_shape, tile_size, overlap)

    results: list[GSplatData] = []
    total_splats = 0
    t0 = time.perf_counter()

    with asection(
        f"Tiled fitting: {len(specs)} tiles, tile_size={tile_size}, overlap={overlap}"
    ):
        if verbose:
            aprint(f"Volume shape: {volume_shape}")
            aprint(f"Tile grid: {_grid_shape(specs, len(volume_shape))}")

        for spec in specs:
            label = f"Tile {spec.index + 1}/{len(specs)} grid={spec.grid_index}"
            with asection(label):
                tile_result = fit_tile(
                    volume,
                    spec,
                    voxel_size=voxel_size,
                    output_space=output_space,
                    progressive=progressive,
                    max_splats_per_pass=max_splats_per_pass,
                    psnr_patience=psnr_patience,
                    max_passes=max_passes,
                    **fit_kwargs,
                )
                n = tile_result.n_splats
                total_splats += n
                if verbose:
                    t_tile = tile_result.stats.get("time_seconds", 0)
                    aprint(f"{n:,} splats ({t_tile:.1f}s)")
                results.append(tile_result)

    # Concatenate all tile results
    if len(results) == 0:
        ndim = len(volume_shape)
        from luxar.gsplats.utils.trils import tril_size

        return GSplatData(
            centers=np.zeros((0, ndim), dtype=np.float32),
            amplitudes=np.zeros((0,), dtype=np.float32),
            cholesky_factors=np.zeros((0, tril_size(ndim)), dtype=np.float32),
            stats={},
        )

    # Merge tile results — LOD-aware if progressive
    has_lods = any(r.n_lods > 1 for r in results)
    if has_lods:
        merged = _merge_lods_across_tiles(results)
    else:
        merged = GSplatData.concatenate(results)

    # Build merged stats
    elapsed = time.perf_counter() - t0
    merged.stats.update(
        {
            "tiled_fitting": True,
            "progressive": progressive,
            "num_tiles": len(specs),
            "tile_size": tile_size,
            "overlap": overlap,
            "volume_shape": volume_shape,
            "time_seconds": elapsed,
            "splats_per_tile": [r.n_splats for r in results],
        }
    )

    if verbose:
        lod_info = f", {merged.n_lods} LODs" if has_lods else ""
        aprint(
            f"Total: {merged.n_splats:,} splats from {len(specs)} tiles "
            f"in {elapsed:.1f}s{lod_info}"
        )

    return merged


def _merge_lods_across_tiles(results: list[GSplatData]) -> GSplatData:
    """Merge LODs across tiles: LOD N = concat of all tiles' LOD N.

    Pads to the maximum LOD count — tiles that stopped early simply
    contribute nothing to higher LODs.
    """
    max_lods = max(r.n_lods for r in results)
    merged_lods: list[GSplatLOD] = []

    for level in range(max_lods):
        # Collect this level's LOD from each tile that has it
        level_lods = [r.at_lod(level) for r in results if level < r.n_lods]

        centers = np.concatenate([lod.centers for lod in level_lods], axis=0)
        amplitudes = np.concatenate([lod.amplitudes for lod in level_lods])
        cholesky = np.concatenate(
            [lod.cholesky_factors for lod in level_lods], axis=0
        )

        # Handle colors (None/mixed)
        has_colors = [lod.colors is not None for lod in level_lods]
        if all(has_colors):
            colors = np.concatenate([lod.colors for lod in level_lods], axis=0)
        elif not any(has_colors):
            colors = None
        else:
            parts = []
            for lod in level_lods:
                if lod.colors is not None:
                    parts.append(lod.colors)
                else:
                    parts.append(np.ones((lod.n_splats, 3), dtype=np.float32))
            colors = np.concatenate(parts, axis=0)

        merged_lods.append(
            GSplatLOD(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=colors,
                stats={
                    "lod_level": level,
                    "n_tiles": len(level_lods),
                },
            )
        )

    return GSplatData.from_lods(merged_lods)


def _grid_shape(specs: list[TileSpec], ndim: int) -> tuple[int, ...]:
    """Compute the grid shape from tile specs."""
    if not specs:
        return tuple([0] * ndim)
    max_grid = [0] * ndim
    for spec in specs:
        for d in range(ndim):
            max_grid[d] = max(max_grid[d], spec.grid_index[d] + 1)
    return tuple(max_grid)
