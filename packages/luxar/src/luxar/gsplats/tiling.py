# tiling.py
"""Tile geometry and cosine apodization for large-volume fitting.

Pure NumPy module with no fitting dependencies. Computes overlapping tile
specifications and Hann (raised cosine) apodization windows that satisfy
the partition-of-unity property: overlapping windows sum to 1.0.
"""

from __future__ import annotations

import itertools
import math
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
    *,
    fold_slivers: bool = False,
) -> list[TileSpec]:
    """Compute a deterministic grid of overlapping tiles covering a volume.

    The grid uses a stride of ``tile_size - overlap`` per axis. Edge tiles
    are clamped to the volume boundary and may be smaller than ``tile_size``.
    With ``fold_slivers=True``, a trailing tile whose unique coverage is smaller
    than the overlap is folded into its predecessor instead of creating an
    overlap-dominated sliver.
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
        if fold_slivers and len(starts) > 1:
            trailing_unique = volume_shape[d] - ends[-2]
            if trailing_unique < ov[d]:
                starts.pop()
                ends.pop()
                ends[-1] = volume_shape[d]
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


def resolve_grid_scale(
    ndim: int,
    *,
    downscale_factors: Sequence[int] | None = None,
    voxel_size: Sequence[float] | float | None = None,
    output_space: str = "real",
) -> tuple[float, ...] | None:
    """Combine the two factors that separate the tile grid's frame from the splats'.

    :func:`compute_tile_specs` works in VOXELS of the array that was tiled, but
    the fitted splats need not live in that frame (issue #1587), for two
    independent and MULTIPLICATIVE reasons:

    * ``--downscale``: the grid is computed on the decimated shape while each
      worker rescales its splats back to full resolution, so a tile origin
      ``o`` lands at ``o * f``.
    * ``voxel_size`` with ``output_space="real"``: the fit emits physical
      coordinates, so a tile origin ``o`` lands at ``o * voxel_size``.

    Both at once (a downscaled parallel fit with a ``voxel_size`` from
    ``--config``) gives ``o * f * voxel_size``, which is why one combined
    factor is resolved here rather than each caller applying its own.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the tiled array (the length of the result).
    downscale_factors : sequence of int, optional
        Per-axis ``--downscale`` factors the grid was decimated by, or ``None``
        when the grid is at full resolution. A scalar is broadcast.
    voxel_size : float or sequence of float, optional
        Physical voxel spacing the fit was given. A scalar is broadcast;
        ``None`` means unit spacing.
    output_space : str, default "real"
        The fit's output space, ``"real"`` or ``"voxel"``. The ``voxel_size``
        term applies only for ``"real"`` — with ``"voxel"`` the centers stay in
        voxel coordinates and multiplying by the spacing would move the planes
        off the parts.

    Returns
    -------
    tuple of float or None
        Per-axis factor for :func:`grid_bsp_tree`'s ``scale``, or ``None`` when
        every factor is 1 (the two frames already agree).

    Raises
    ------
    ValueError
        On an ``output_space`` outside ``("real", "voxel")``, or a
        ``voxel_size`` that is neither a scalar nor a length-``ndim`` sequence.
        Both are refused rather than absorbed: an unrecognised
        ``output_space`` would silently DROP the ``voxel_size`` term (the exact
        #1587 mismatch this function exists to close), and a wrong-length
        spacing would broadcast a partial answer.
    """
    # Same vocabulary as `luxar.gsplats.fitting.validation`, checked here too
    # because this is a public entry point that a producer can reach without
    # ever going through the fitter.
    if output_space not in ("real", "voxel"):
        raise ValueError(
            f"output_space must be 'real' or 'voxel', got {output_space!r}"
        )
    factors = np.ones(ndim, dtype=np.float64)
    if downscale_factors is not None:
        factors *= np.broadcast_to(
            np.asarray(downscale_factors, dtype=np.float64), (ndim,)
        )
    if output_space == "real" and voxel_size is not None:
        spacing = np.asarray(voxel_size, dtype=np.float64)
        if spacing.ndim == 0:
            spacing = np.full(ndim, float(spacing))
        elif spacing.shape != (ndim,):
            raise ValueError(
                f"voxel_size must have length {ndim} to match the tile grid's "
                f"dimensions, got length {spacing.size}"
            )
        factors *= spacing
    # Both terms are user input (a YAML `voxel_size:` / `downscale:`), and
    # neither the fitter's own `<= 0` check nor a `== 1.0` comparison rejects a
    # NaN — which would go on to produce NaN split planes rather than an error.
    # Refuse anything the tree cannot be stated in, here, where the offending
    # term is still nameable.
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
        raise ValueError(
            f"the tile grid's scale must be finite and strictly positive, got "
            f"{tuple(float(f) for f in factors)} from downscale_factors="
            f"{downscale_factors!r} and voxel_size={voxel_size!r}"
        )
    if np.all(factors == 1.0):
        return None
    return tuple(float(f) for f in factors)


def _validated_scale(scale: Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    """``scale`` as a per-axis float tuple, or all-ones for ``None``.

    Raises
    ------
    ValueError
        On a wrong length, or an entry that is not finite and positive: a ``0``
        collapses every plane onto the origin, a negative factor mirrors the
        frame (so the tree would order the parts backwards), and a
        ``NaN``/``inf`` — which a bare ``<= 0`` test lets through — would put a
        non-number in the serialized tree. None can be meant, so refuse rather
        than emit a silently useless tree.
    """
    if scale is None:
        return (1.0,) * ndim
    factors = tuple(float(f) for f in scale)
    if len(factors) != ndim:
        raise ValueError(
            f"scale has length {len(factors)} but the tile grid has {ndim} dimensions"
        )
    if any(not math.isfinite(f) or f <= 0.0 for f in factors):
        raise ValueError(
            f"scale entries must be finite and strictly positive, got {factors}"
        )
    return factors


def grid_bsp_tree(
    specs: Sequence[TileSpec], *, scale: Sequence[float] | None = None
) -> dict | None:
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
    scale : sequence of float, optional
        Per-axis factor mapping the specs' VOXEL frame onto the frame the
        SPLATS live in. ``None`` (the default) means the two frames agree.
        Every entry must be strictly positive: ``0`` would collapse the planes
        onto the origin and a negative factor would mirror the ordering the
        tree encodes. Use :func:`resolve_grid_scale` to build it.

        Two independent terms can put the splats in a different frame from the
        grid, and they COMPOSE (issue #1587):

        * ``--downscale``. The parallel tiled path deliberately computes its
          grid on the POST-downscale shape — that is how the parent and its
          ``fit --tile i/M`` workers agree on the tile count M — while each
          worker rescales its own splats back to full resolution before
          writing.
        * ``voxel_size`` with ``output_space="real"``. The specs are voxel
          coordinates, but a fit asked for real-space output emits centers in
          physical units (see :func:`~luxar.gsplats.fit_tiled_gsplats.fit_tile`,
          which offsets a tile by ``origin * voxel_size``).

        Without a factor here, every plane of the resulting ``kind=partition``
        would be a factor too small and would no longer lie between the parts it
        separates, so the viewer's back-to-front part ordering (#1555) would be
        computed against nonsense. Passing the resolved factors maps a tile
        spanning ``[origin[d], origin[d] + shape[d])`` to
        ``[origin[d] * f[d], (origin[d] + shape[d]) * f[d])``, which is exactly
        the convention :func:`~luxar.gsplats.fitting.downscale.rescale_centers`
        (and the ``origin * voxel_size`` offset) applies to the centers.

    Returns
    -------
    dict or None
        The serialized tree, or ``None`` when ``specs`` is empty or the grid
        subdivides an axis beyond the third. Those axes are stacked
        time/channel barriers and are never displayed.

    Raises
    ------
    ValueError
        If ``scale`` is given with a length other than the grid's ndim, or with
        a non-positive entry.
    """
    if not specs:
        return None

    ndim = len(specs[0].grid_index)
    factors = _validated_scale(scale, ndim)
    n_cells = [max(s.grid_index[d] for s in specs) + 1 for d in range(ndim)]
    if any(n_cells[d] > 1 for d in range(3, ndim)):
        return None

    # Per-axis grid-line coordinates, and the flat index of each grid cell.
    lo_coord: list[dict[int, float]] = [{} for _ in range(ndim)]
    hi_coord: list[dict[int, float]] = [{} for _ in range(ndim)]
    for spec in specs:
        for d, k in enumerate(spec.grid_index):
            lo_coord[d][k] = float(spec.origin[d]) * factors[d]
            hi_coord[d][k] = (float(spec.origin[d]) + float(spec.shape[d])) * factors[d]
    flat_of_cell = {tuple(s.grid_index): int(s.index) for s in specs}

    def build(ranges: list[tuple[int, int]]) -> dict:
        widths = [hi - lo for lo, hi in ranges]
        if all(w == 1 for w in widths):
            return {"part": flat_of_cell[tuple(lo for lo, _ in ranges)]}
        # Split the axis with the most cells left (ties -> lowest axis, so the
        # tree is deterministic). Axes past the third are stacked barriers that
        # are never displayed, and an all-widths-1 grid already returned above.
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
