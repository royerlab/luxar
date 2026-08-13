# tiling.py
"""Tile geometry and cosine apodization for large-volume fitting.

Pure NumPy module with no fitting dependencies. Computes overlapping tile
specifications and Hann (raised cosine) apodization windows that satisfy
the partition-of-unity property: overlapping windows sum to 1.0.
"""

from __future__ import annotations

import itertools
import numbers
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """Specification for a single tile within a larger volume.

    Attributes
    ----------
    index : int
        Flat index in [0, N) where N is total tile count.
        Used for ``--tile N/M`` CLI addressing.
    grid_index : tuple of int
        Position in the tile grid, e.g. ``(iz, iy, ix)`` for 3D.
    slices : tuple of slice
        Index slices into the full volume to extract this tile.
    origin : tuple of float
        Global coordinate offset of this tile's ``[0, 0, ...]`` corner.
        Equal to the start of each slice, as float for voxel_size compatibility.
    shape : tuple of int
        Expected tile shape after slicing (may be smaller at volume edges).
    border_low : tuple of bool
        Per-axis: True if the tile is the first on this axis (no taper on low side).
    border_high : tuple of bool
        Per-axis: True if the tile is the last on this axis (no taper on high side).
    overlap_low : tuple of int
        Per-axis: actual overlap in voxels with the preceding tile on low side.
        Zero for the first tile on each axis.
    overlap_high : tuple of int
        Per-axis: actual overlap in voxels with the following tile on high side.
        Zero for the last tile on each axis.
    """

    index: int
    grid_index: tuple[int, ...]
    slices: tuple[slice, ...]
    origin: tuple[float, ...]
    shape: tuple[int, ...]
    border_low: tuple[bool, ...]
    border_high: tuple[bool, ...]
    overlap_low: tuple[int, ...]
    overlap_high: tuple[int, ...]


def _broadcast_to_ndim(
    value: int | Sequence[int], ndim: int, name: str
) -> tuple[int, ...]:
    """Broadcast a scalar or sequence to a tuple of length ndim."""
    if isinstance(value, (int, numbers.Integral)):
        return tuple([int(value)] * ndim)
    # At this point value must be a Sequence
    result = tuple(int(v) for v in value)
    if len(result) != ndim:
        raise ValueError(
            f"{name} has length {len(result)} but volume has {ndim} dimensions"
        )
    return result


def compute_tile_specs(
    volume_shape: tuple[int, ...],
    tile_size: int | Sequence[int],
    overlap: int | Sequence[int],
) -> list[TileSpec]:
    """Compute a deterministic grid of overlapping tiles covering a volume.

    The grid uses a stride of ``tile_size - overlap`` per axis. Edge tiles
    are clamped to the volume boundary and may be smaller than ``tile_size``.
    Each tile stores its actual overlap with neighbors (which may differ from
    the ``overlap`` parameter at volume edges) to ensure correct windowing.

    Parameters
    ----------
    volume_shape : tuple of int
        Shape of the full volume, e.g. ``(500, 2048, 2048)``.
    tile_size : int or sequence of int
        Tile size per axis. Scalar is broadcast to all axes.
    overlap : int or sequence of int
        Overlap width per axis. Scalar is broadcast to all axes.
        Must satisfy ``0 <= overlap <= tile_size // 2`` on each axis.
        Overlaps larger than half the tile size cause triple tile overlap,
        which breaks the Hann partition-of-unity guarantee.

    Returns
    -------
    list of TileSpec
        Tile specifications in row-major order. The list is deterministic:
        identical inputs always produce identical output (critical for Slurm).

    Raises
    ------
    ValueError
        If overlap > tile_size // 2, tile_size <= 0, or overlap < 0 on any axis.
    """
    ndim = len(volume_shape)
    ts = _broadcast_to_ndim(tile_size, ndim, "tile_size")
    ov = _broadcast_to_ndim(overlap, ndim, "overlap")

    for d in range(ndim):
        if ts[d] <= 0:
            raise ValueError(f"tile_size must be > 0, got {ts[d]} on axis {d}")
        if ov[d] < 0:
            raise ValueError(f"overlap must be >= 0, got {ov[d]} on axis {d}")
        if ov[d] >= ts[d]:
            raise ValueError(
                f"overlap ({ov[d]}) must be < tile_size ({ts[d]}) on axis {d}"
            )
        if ov[d] * 2 > ts[d]:
            raise ValueError(
                f"overlap ({ov[d]}) must be <= tile_size/2 ({ts[d] // 2}) on axis {d}. "
                f"Larger overlaps cause triple tile overlap which breaks the "
                f"Hann partition-of-unity property."
            )

    # Compute grid starts and ends per axis
    strides = tuple(t - o for t, o in zip(ts, ov))
    grid_starts: list[list[int]] = []
    grid_ends: list[list[int]] = []
    for d in range(ndim):
        starts: list[int] = []
        ends: list[int] = []
        pos = 0
        while pos < volume_shape[d]:
            starts.append(pos)
            ends.append(min(pos + ts[d], volume_shape[d]))
            pos += strides[d]
        grid_starts.append(starts)
        grid_ends.append(ends)

    # Enumerate tiles in row-major order
    specs: list[TileSpec] = []
    flat_idx = 0
    for grid_index in itertools.product(*(range(len(s)) for s in grid_starts)):
        per_axis_starts = tuple(grid_starts[d][grid_index[d]] for d in range(ndim))
        per_axis_ends = tuple(grid_ends[d][grid_index[d]] for d in range(ndim))
        shape = tuple(per_axis_ends[d] - per_axis_starts[d] for d in range(ndim))
        slices = tuple(slice(per_axis_starts[d], per_axis_ends[d]) for d in range(ndim))
        origin = tuple(float(per_axis_starts[d]) for d in range(ndim))

        is_first = tuple(grid_index[d] == 0 for d in range(ndim))
        is_last = tuple(grid_index[d] == len(grid_starts[d]) - 1 for d in range(ndim))

        # Compute actual overlap with neighbors per axis
        overlap_low: list[int] = []
        overlap_high: list[int] = []
        for d in range(ndim):
            if is_first[d]:
                overlap_low.append(0)
            else:
                # Overlap with predecessor: predecessor's end - our start
                prev_end = grid_ends[d][grid_index[d] - 1]
                ov_lo = max(0, prev_end - per_axis_starts[d])
                overlap_low.append(ov_lo)

            if is_last[d]:
                overlap_high.append(0)
            else:
                # Overlap with successor: our end - successor's start
                next_start = grid_starts[d][grid_index[d] + 1]
                ov_hi = max(0, per_axis_ends[d] - next_start)
                overlap_high.append(ov_hi)

        specs.append(
            TileSpec(
                index=flat_idx,
                grid_index=grid_index,
                slices=slices,
                origin=origin,
                shape=shape,
                border_low=is_first,
                border_high=is_last,
                overlap_low=tuple(overlap_low),
                overlap_high=tuple(overlap_high),
            )
        )
        flat_idx += 1

    return specs


def grid_bsp_tree(specs: Sequence[TileSpec]) -> dict | None:
    """Split-plane tree over a uniform tile grid, in the serialized ``bsp_tree`` form.

    Lets the viewer order uniform-tiled partition parts back-to-front by
    painter's algorithm instead of by part centroid, which is not a valid order
    and flips discretely as the camera moves (the seam popping of issue #1555).

    APPROXIMATE, unlike a content plan's tree. A content box crops its splats to
    the core box, so those parts are exactly disjoint; a uniform tile keeps every
    splat of the *apodized* tile, overlap band included, so neighbouring tiles
    genuinely share space and no exact part order exists.
    :func:`compute_tile_specs` caps the halo at ``2 * overlap <= tile_size``, so
    at most two tiles meet on any axis and the honest cut is the MIDPLANE of
    their shared band. Misordering is then confined to that band rather than
    whole tiles swapping — the same second-order residual as splats whose own
    Gaussian straddles a seam.

    Leaves carry ``TileSpec.index`` (the flat, row-major tile index) VERBATIM, so
    a caller can prune with the same keep-set it uses for the fitted regions
    (:func:`~luxar.core.group.partition.prune_serialized_bsp_tree`). The labels
    are explicit rather than DFS-implied because the median split below does not
    visit tiles in flat order.

    Parameters
    ----------
    specs : sequence of TileSpec
        A full grid as returned by :func:`compute_tile_specs`.

    Returns
    -------
    dict or None
        The serialized tree, or ``None`` when ``specs`` is empty or the grid
        subdivides an axis beyond the third — the serialized format admits split
        axes ``0``/``1``/``2`` only.
    """
    if not specs:
        return None

    ndim = len(specs[0].grid_index)
    n_cells = [max(s.grid_index[d] for s in specs) + 1 for d in range(ndim)]
    if any(n_cells[d] > 1 for d in range(3, ndim)):
        return None

    # Per-axis grid-line coordinates, and the flat index of each grid cell.
    lo_coord: list[dict[int, float]] = [{} for _ in range(ndim)]
    hi_coord: list[dict[int, float]] = [{} for _ in range(ndim)]
    for spec in specs:
        for d, k in enumerate(spec.grid_index):
            lo_coord[d][k] = float(spec.origin[d])
            hi_coord[d][k] = float(spec.origin[d]) + float(spec.shape[d])
    flat_of_cell = {tuple(s.grid_index): int(s.index) for s in specs}

    def build(ranges: list[tuple[int, int]]) -> dict:
        widths = [hi - lo for lo, hi in ranges]
        if all(w == 1 for w in widths):
            return {"part": flat_of_cell[tuple(lo for lo, _ in ranges)]}
        # Split the axis with the most cells left (ties -> lowest axis, so the
        # tree is deterministic); only axes 0/1/2 are representable, and an
        # all-widths-1 grid already returned above.
        axis = max(range(min(3, ndim)), key=lambda d: widths[d])
        k0, k1 = ranges[axis]
        kmid = (k0 + k1) // 2
        # Midplane of the band shared by cells kmid-1 and kmid. The two are
        # adjacent by construction, so hi_coord[kmid - 1] >= lo_coord[kmid].
        split = 0.5 * (lo_coord[axis][kmid] + hi_coord[axis][kmid - 1])
        low = list(ranges)
        low[axis] = (k0, kmid)
        high = list(ranges)
        high[axis] = (kmid, k1)
        return {
            "axis": axis,
            "split": split,
            "left": build(low),
            "right": build(high),
        }

    return build([(0, n_cells[d]) for d in range(ndim)])


def _cosine_ramp(length: int) -> np.ndarray:
    """Half-cosine ramp from ~0 to 1 over ``length`` samples.

    Uses ``w(k) = 0.5 * (1 - cos(pi * k / (length + 1)))`` for k in [1, length].
    This avoids exact zeros at the boundary while maintaining the partition-of-unity
    property: two overlapping ramps (rising + falling) sum to 1.0 at every sample.

    Parameters
    ----------
    length : int
        Number of samples in the ramp.

    Returns
    -------
    np.ndarray, shape (length,), dtype float32
        Monotonically increasing values in (0, 1).
    """
    if length <= 0:
        return np.array([], dtype=np.float32)
    k = np.arange(1, length + 1, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(np.pi * k / (length + 1)))


def cosine_window(
    spec: TileSpec,
) -> np.ndarray:
    """Build an nD cosine (Hann) apodization window for a tile.

    The window is a separable product of 1D half-cosine ramps. Boundary faces
    (first/last tile on an axis) stay at 1.0. Interior faces are tapered over
    the actual overlap with the neighboring tile.

    Two overlapping windows from adjacent tiles sum to exactly 1.0 in the
    overlap zone (Hann partition-of-unity property).

    Parameters
    ----------
    spec : TileSpec
        Tile specification with shape, border flags, and actual overlap sizes.

    Returns
    -------
    np.ndarray, dtype float32
        Window array of tile shape with values in (0, 1].
    """
    ndim = len(spec.shape)

    # Build 1D window per axis
    windows_1d: list[np.ndarray] = []
    for d in range(ndim):
        w = np.ones(spec.shape[d], dtype=np.float32)

        # Rising ramp on low side (unless first tile on this axis)
        ov_lo = spec.overlap_low[d]
        if not spec.border_low[d] and ov_lo > 0:
            ramp_len = min(ov_lo, spec.shape[d])
            w[:ramp_len] = _cosine_ramp(ramp_len)

        # Falling ramp on high side (unless last tile on this axis)
        ov_hi = spec.overlap_high[d]
        if not spec.border_high[d] and ov_hi > 0:
            ramp_len = min(ov_hi, spec.shape[d])
            w[-ramp_len:] = _cosine_ramp(ramp_len)[::-1]

        windows_1d.append(w)

    # Separable product via broadcasting
    for d in range(ndim):
        broadcast_shape = [1] * ndim
        broadcast_shape[d] = spec.shape[d]
        windows_1d[d] = windows_1d[d].reshape(broadcast_shape)

    window: np.ndarray = windows_1d[0].copy()
    for d in range(1, ndim):
        window = window * windows_1d[d]

    return window
