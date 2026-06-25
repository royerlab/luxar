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
from luxar.gsplats.gsplat_data import GSplatData
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

    # 1b. Denoise tile (if requested via fit_kwargs)
    # Use pop to remove denoise keys before forwarding to fitting functions
    _denoise_h = fit_kwargs.pop("_denoise_h", None)
    _denoise_params = fit_kwargs.pop("_denoise_params", None)
    if _denoise_h is not None and _denoise_params is not None:
        from arbol import asection as _asection

        from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

        with _asection(f"Denoising tile {spec.index} (h={_denoise_h:.4f})"):
            tile_data = denoise_volume_array(tile_data, h=_denoise_h, **_denoise_params)

    # Pop cull_retention — per-tile culling is disabled (fit_tiled culls the merged result)
    fit_kwargs.pop("cull_retention", None)

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
            cull_retention=None,  # Disable per-tile; fit_tiled culls the merged result
            **fit_kwargs,
        )
    else:
        result = fit_gaussian_splats(
            tile_data,
            voxel_size=voxel_size,
            output_space=output_space,
            cull_retention=None,  # Disable per-tile; fit_tiled culls the merged result
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
    cull_retention: float | None = 0.95,
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
    cull_retention : float or None, default=0.95
        Post-fit cumulative culling on the merged result.  Keeps the top
        splats that account for this fraction of total amplitude (0--1).
        Per-tile culling is disabled automatically; only the merged result
        is culled.  Set to ``None`` to disable.
    **fit_kwargs
        All other keyword arguments forwarded to the per-tile fitting function
        (e.g. ``seeds``, ``n_iters``, ``preset``, ``device``,
        ``residual_pass_min_iters`` when ``progressive=True``).

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
                if verbose:
                    t_tile = tile_result.stats.get("time_seconds", 0)
                    aprint(f"{n:,} splats ({t_tile:.1f}s)")
                results.append(tile_result)

    elapsed = time.perf_counter() - t0
    return merge_tile_results(
        results,
        volume_shape=volume_shape,
        tile_size=tile_size,
        overlap=overlap,
        num_tiles=len(specs),
        progressive=progressive,
        cull_retention=cull_retention,
        elapsed=elapsed,
        verbose=verbose,
    )


def merge_tile_results(
    results: list[GSplatData],
    *,
    volume_shape: tuple[int, ...],
    tile_size: int | Sequence[int],
    overlap: int | Sequence[int],
    num_tiles: int,
    progressive: bool,
    cull_retention: float | None,
    elapsed: float,
    verbose: bool = True,
) -> GSplatData:
    """Merge per-tile fit results into a single (optionally multi-LOD) dataset.

    Shared by both the sequential :func:`fit_tiled` loop and the parallel
    orchestrator in ``fit_tiled_parallel``.  Concatenates tile results
    (LOD-aware when ``progressive``), stamps tiled-fitting stats, and applies
    a single post-fit cumulative cull on the merged result.

    Parameters
    ----------
    results : list of GSplatData
        Per-tile fit results, already translated to global coordinates. May
        be empty.
    volume_shape : tuple of int
        Full (possibly downscaled) volume shape, recorded in stats and used to
        build an empty result when ``results`` is empty.
    tile_size, overlap : int or sequence of int
        Tiling geometry, recorded in stats.
    num_tiles : int
        Number of tiles in the grid (``len(specs)``).
    progressive : bool
        Whether tiles were fit progressively (selects LOD-aware merge).
    cull_retention : float or None
        Post-fit cumulative culling fraction on the merged result (0--1).
        ``None`` or outside (0, 1) disables culling.
    elapsed : float
        Wall-clock seconds for the fitting stage, recorded in stats.
    verbose : bool, default True
        Print a summary line via arbol.

    Returns
    -------
    GSplatData
        Merged result. Multi-LOD if ``progressive`` and tiles carry sublods.
    """
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
    has_lods = any(r.n_additive_sublods > 1 for r in results)
    if has_lods:
        merged = _merge_lods_across_tiles(results)
    else:
        merged = GSplatData.concatenate(results)

    # Build merged stats
    merged.stats.update(
        {
            "tiled_fitting": True,
            "progressive": progressive,
            "num_tiles": num_tiles,
            "tile_size": tile_size,
            "overlap": overlap,
            "volume_shape": volume_shape,
            "time_seconds": elapsed,
            "splats_per_tile": [r.n_splats for r in results],
        }
    )

    if verbose:
        lod_info = f", {merged.n_additive_sublods} LODs" if has_lods else ""
        aprint(
            f"Total: {merged.n_splats:,} splats from {num_tiles} tiles "
            f"in {elapsed:.1f}s{lod_info}"
        )

    # Post-fit cumulative culling on merged result
    if cull_retention is not None and 0 < cull_retention < 1.0 and merged.n_splats > 0:
        n_before = merged.n_splats
        merged = merged.cull(method="cumulative", retention=cull_retention)
        if verbose and merged.n_splats < n_before:
            aprint(
                f"Post-fit culling: {n_before} -> {merged.n_splats} splats "
                f"(retained {cull_retention * 100:.0f}% of amplitude)"
            )

    return merged


def _merge_lods_across_tiles(results: list[GSplatData]) -> GSplatData:
    """Merge LODs across tiles: LOD N = concat of all tiles' LOD N.

    Pads to the maximum LOD count — tiles that stopped early simply
    contribute nothing to higher LODs.
    """
    # Delegates to LOD-aware concatenate() which handles per-LOD merging
    # and mixed LOD counts automatically.
    return GSplatData.concatenate(results)


def _grid_shape(specs: list[TileSpec], ndim: int) -> tuple[int, ...]:
    """Compute the grid shape from tile specs."""
    if not specs:
        return tuple([0] * ndim)
    max_grid = [0] * ndim
    for spec in specs:
        for d in range(ndim):
            max_grid[d] = max(max_grid[d], spec.grid_index[d] + 1)
    return tuple(max_grid)
