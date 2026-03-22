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
    **fit_kwargs
        All other keyword arguments forwarded to :func:`fit_gaussian_splats`.

    Returns
    -------
    GSplatData
        Fit result with centers in global volume coordinates.

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
        if output_space == "real" and voxel_size is not None:
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
    **fit_kwargs: Any,
) -> GSplatData:
    """Fit Gaussian splats to a large volume using tiled decomposition.

    Splits the volume into overlapping tiles with cosine apodization
    (Hann window), fits each tile independently with :func:`fit_gaussian_splats`,
    and concatenates all results. The Hann partition-of-unity property guarantees
    seamless blending at tile boundaries.

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
    **fit_kwargs
        All other keyword arguments forwarded to :func:`fit_gaussian_splats`
        for each tile (e.g. ``seeds``, ``n_iters``, ``preset``, ``device``).

    Returns
    -------
    GSplatData
        Concatenated result with all splats in global coordinates.
        Hilbert curve resorting happens automatically on ``save()``.
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

    merged = GSplatData.concatenate(results)

    # Build merged stats
    elapsed = time.perf_counter() - t0
    merged.stats.update(
        {
            "tiled_fitting": True,
            "num_tiles": len(specs),
            "tile_size": tile_size,
            "overlap": overlap,
            "volume_shape": volume_shape,
            "time_seconds": elapsed,
            "splats_per_tile": [r.n_splats for r in results],
        }
    )

    if verbose:
        aprint(
            f"Total: {merged.n_splats:,} splats from {len(specs)} tiles "
            f"in {elapsed:.1f}s"
        )

    return merged


def _grid_shape(specs: list[TileSpec], ndim: int) -> tuple[int, ...]:
    """Compute the grid shape from tile specs."""
    if not specs:
        return tuple([0] * ndim)
    max_grid = [0] * ndim
    for spec in specs:
        for d in range(ndim):
            max_grid[d] = max(max_grid[d], spec.grid_index[d] + 1)
    return tuple(max_grid)
