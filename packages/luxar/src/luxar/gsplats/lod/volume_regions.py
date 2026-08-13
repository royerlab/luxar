"""Give one volume re-fit exactly the sub-volume its splats are responsible for.

:func:`~luxar.gsplats.lod.volume_refit.volume_refine_splats` re-fits a coarse
level against the source volume, which is only sound while the seed's support and
the volume coincide. Two structures break that, and both break it the same way —
by making each seed responsible for a *part* of the volume:

``barrier dims``
    A stacked timelapse groups splats by timepoint and must never coarsen across
    that axis. Each group owns ONE index along the barrier axes and the whole
    extent of the coarsened ones.
``partition parts``
    A BSP tile's splats only explain their own tile. Each part owns a BOX of the
    coarsened axes and the whole extent of the barrier ones.

They compose: one part of one timepoint of a tiled timelapse owns a box within a
slice, which is why this is a single selector rather than two special cases.

Why the barrier dims are sliced away rather than frozen: the fitting stack has no
freeze mechanism at all. ``fit_gaussian_splats`` re-parameterises centers as
``sigmoid(raw) * (shape - 1)`` over EVERY axis and floors every per-axis sigma at
``sqrt(1/12)``, so a re-fit handed a 4D volume would drag splats off their
timepoint and widen them along time no matter what penalty it was given. Removing
the barrier axis from the fit makes that unrepresentable instead of merely
discouraged, and the barrier coordinate is then restored verbatim from the seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.utils.trils import pack_tril, permute_cholesky_packed, unpack_tril


@dataclass(frozen=True)
class SubVolume:
    """One re-fit's slice of the source volume, in the splats' own dim order.

    Attributes
    ----------
    array:
        The materialised sub-volume, ``float32``, with axes ordered to match
        ``dims`` (i.e. the re-fit's center columns), NOT the source volume's
        axis order.
    origin:
        Offset to SUBTRACT from the seed's centers before re-fitting, one entry
        per entry of ``dims``. The re-fit renders on an origin-anchored voxel
        grid and silently clips warm-start centers into it, so a cropped volume
        is only usable in crop-local coordinates.
    dims:
        Center-column indices the sub-volume spans, ascending.
    """

    array: np.ndarray
    origin: np.ndarray
    dims: Tuple[int, ...]


def select_sub_volume(
    # Deliberately array-LIKE, not ndarray: the whole point is that a lazy store
    # (a zarr array) can be passed and only the slice is realised.
    volume: Any,
    *,
    ndim: int,
    barrier_dims: Sequence[int] = (),
    barrier_coords: Sequence[float] = (),
    box: Optional[Sequence[Tuple[float, float]]] = None,
    volume_axes: Optional[Sequence[int]] = None,
) -> SubVolume:
    """Materialise the sub-volume for one barrier group and/or one tile.

    ``volume`` is only ever *sliced*, never coerced whole, so a lazy store (a
    zarr array) is a valid argument and stays lazy. That is not a nicety: a
    253-timepoint 407x2048x2048 uint16 timelapse is 431 GB, while one timepoint
    is 3.4 GB.

    Parameters
    ----------
    volume:
        Array-like supporting numpy basic indexing. Its axes correspond to the
        splats' center columns through ``volume_axes``.
    ndim:
        The splats' dimensionality.
    barrier_dims, barrier_coords:
        Center dims to index away, and the coordinate to index them at. The
        coordinate is rounded to the nearest voxel: a barrier coordinate is a
        discrete label, and quantized storage can leave it a fraction of a step
        off an integer (uint16 over a 0-252 range has a step of ~0.0038).
    box:
        Per-retained-dim ``(low, high)`` bounds in center coordinates, inclusive
        of ``high``. Rounded OUTWARD so a splat on the boundary keeps its
        support, and clamped to the volume.
    volume_axes:
        ``volume_axes[i]`` is the volume axis holding center dim ``i``. Defaults
        to the identity, which is what a whole-volume 3D re-fit has always
        assumed. A stacked timelapse needs it explicitly: Luxar puts spatial
        dims first and the stacked axis LAST, while the source array is
        typically ``(t, z, y, x)`` with time FIRST.
    """
    bset = {int(d) for d in barrier_dims}
    free_dims = tuple(d for d in range(ndim) if d not in bset)
    if volume_axes is None:
        volume_axes = tuple(range(ndim))
    # A permutation, not merely the right length: a duplicate or out-of-range
    # entry otherwise surfaces as numpy's "repeated axis in transpose" or a bare
    # IndexError, neither of which names the argument at fault.
    if sorted(int(a) for a in volume_axes) != list(range(ndim)):
        raise ValueError(
            f"volume_axes must be a permutation of 0..{ndim - 1} (one volume "
            f"axis per center dim, each used once); got {tuple(volume_axes)!r}"
        )
    vshape = tuple(int(s) for s in volume.shape)
    if len(vshape) != ndim:
        raise ValueError(
            f"volume is {len(vshape)}D but the splats are {ndim}D; a re-fit "
            "target must span every center dim (slice the barrier dims via "
            "barrier_coords, not by pre-reducing the array)"
        )

    index: list = [slice(None)] * len(vshape)
    for d, coord in zip(barrier_dims, barrier_coords):
        ax = int(volume_axes[int(d)])
        i = int(round(float(coord)))
        if not 0 <= i < vshape[ax]:
            raise ValueError(
                f"barrier coordinate {coord!r} on center dim {d} maps to index "
                f"{i} on volume axis {ax} of extent {vshape[ax]}; the splats "
                "and the volume disagree about this axis"
            )
        index[ax] = i

    origin = np.zeros(len(free_dims), dtype=np.float64)
    if box is not None:
        if len(box) != len(free_dims):
            raise ValueError(
                f"box must have one (low, high) per retained dim "
                f"({len(free_dims)}); got {len(box)}"
            )
        for k, d in enumerate(free_dims):
            ax = int(volume_axes[int(d)])
            lo_f, hi_f = box[k]
            # A BSP cell records CUTS, so the outer faces of the root box come
            # back infinite; the volume's own extent is the missing bound.
            lo = 0 if not np.isfinite(lo_f) else max(0, int(np.floor(float(lo_f))))
            hi = (
                vshape[ax]
                if not np.isfinite(hi_f)
                else min(vshape[ax], int(np.ceil(float(hi_f))) + 1)
            )
            if hi <= lo:  # a degenerate box would give the fit nothing to see
                lo, hi = 0, vshape[ax]
            index[ax] = slice(lo, hi)
            origin[k] = float(lo)

    arr = np.asarray(volume[tuple(index)], dtype=np.float32)
    # Integer-indexed axes are gone; the survivors are in VOLUME order, so
    # transpose them into center-dim order before the fit ever sees them.
    survivors = [ax for ax in range(len(vshape)) if not isinstance(index[ax], int)]
    desired = [int(volume_axes[int(d)]) for d in free_dims]
    if desired != survivors:
        arr = np.transpose(arr, [survivors.index(ax) for ax in desired])
    return SubVolume(array=arr, origin=origin, dims=free_dims)


def project_to_dims(data: GSplatData, dims: Sequence[int]) -> GSplatData:
    """Drop every center column outside ``dims``, keeping the marginal covariance.

    The retained covariance is the true marginal ``Sigma[dims, dims]`` (permute
    then take the leading block), which is what the re-fit should see: the
    splat's extent within the sub-volume it is being fitted against.
    """
    dims = tuple(int(d) for d in dims)
    d_total = data.ndim
    if dims == tuple(range(d_total)):
        return data
    centers = np.asarray(data.centers)[:, list(dims)].astype(np.float32)
    d_free = len(dims)
    perm = list(dims) + [d for d in range(d_total) if d not in set(dims)]
    L_perm = unpack_tril(
        permute_cholesky_packed(np.asarray(data.cholesky_factors), d_total, perm),
        d_total,
    )
    # Leading block of a lower-triangular factor IS the marginal's factor:
    # Sigma[:f, :f] = A A^T with A = L[:f, :f], since L's first f rows have no
    # entries beyond column f.
    packed = pack_tril(np.ascontiguousarray(L_perm[:, :d_free, :d_free]))
    return GSplatData(
        centers=centers,
        amplitudes=np.asarray(data.amplitudes),
        cholesky_factors=packed.astype(np.float32),
        colors=data.colors,
        truncation_radius=data.truncation_radius,
    )


def restore_dims(
    refit: GSplatData, seed: GSplatData, dims: Sequence[int]
) -> GSplatData:
    """Lift ``refit`` back to ``seed``'s dimensionality, row-for-row.

    The re-fit is identity-preserving, so row ``i`` of ``refit`` is row ``i`` of
    ``seed``; the dropped columns are restored from the seed verbatim.

    The covariance is recombined in FACTOR space, not in ``Sigma`` space. With
    the free dims permuted first, ``L = [[A, 0], [B, C]]``; substituting the
    re-fit's ``A`` leaves ``L`` lower-triangular with a positive diagonal, so
    ``Sigma = L L^T`` stays positive-definite by construction. Recombining
    ``Sigma`` blocks directly offers no such guarantee. For a genuine barrier
    dim — one the splats have no extent along — ``B`` is zero and the lift is
    exact.
    """
    dims = tuple(int(d) for d in dims)
    d_total = seed.ndim
    if dims == tuple(range(d_total)):
        return refit
    if refit.n_splats != seed.n_splats:
        raise ValueError(
            f"restore_dims expects a row-aligned re-fit; got {refit.n_splats} "
            f"splats against a {seed.n_splats}-splat seed"
        )
    d_free = len(dims)
    rest = [d for d in range(d_total) if d not in set(dims)]
    perm = list(dims) + rest

    centers = np.asarray(seed.centers).copy()
    centers[:, list(dims)] = np.asarray(refit.centers)

    L_perm = unpack_tril(
        permute_cholesky_packed(np.asarray(seed.cholesky_factors), d_total, perm),
        d_total,
    )
    L_perm[:, :d_free, :d_free] = unpack_tril(
        np.asarray(refit.cholesky_factors), d_free
    )
    inverse = np.argsort(np.asarray(perm)).tolist()
    packed = permute_cholesky_packed(
        pack_tril(np.ascontiguousarray(L_perm)), d_total, inverse
    )
    return GSplatData(
        centers=centers.astype(np.float32),
        amplitudes=np.asarray(refit.amplitudes),
        cholesky_factors=packed.astype(np.float32),
        colors=seed.colors,
        truncation_radius=seed.truncation_radius,
    )


def merge_volume_refit_stats(sink: Dict, group: Dict, *, weight: int) -> None:
    """Fold one group's/part's volume-refit stats into a per-level aggregate.

    The single-refit dict (``volume_refine_splats``' return) is a per-level
    record; once a level is refined in pieces the level's stats have to describe
    all of them. Counts and wall time SUM, the MSEs are count-weighted means
    (each is a mean over that piece's voxels, so pieces of different size must
    not be given equal say), and the booleans become fractions — reporting a
    single ``improved`` for 253 timepoints would throw away exactly the
    information a caller wants.

    ``weight`` is the piece's splat count. Kept flat and JSON-safe: the zarr
    writer and the stats tests both require that.
    """
    sink["n_pieces"] = sink.get("n_pieces", 0) + 1
    sink["n_seed"] = sink.get("n_seed", 0) + int(group.get("n_seed", 0))
    sink["n_refit"] = sink.get("n_refit", 0) + int(group.get("n_refit", 0))
    sink["wall_s"] = sink.get("wall_s", 0.0) + float(group.get("wall_s", 0.0))
    sink["iters"] = max(sink.get("iters", 0), int(group.get("iters", 0)))
    for key in ("improved", "seed_won", "frame_mismatch", "mass_pinned", "tile_escape"):
        if bool(group.get(key)):
            sink[f"n_{key}"] = sink.get(f"n_{key}", 0) + 1
    w = max(int(weight), 0)
    if "mse_seed" in group and "mse_refit" in group:
        prev_w = sink.get("_mse_w", 0)
        # `mse_stored` describes what was KEPT. Averaging `mse_refit` alone is
        # misleading once a level is refined in pieces: it folds in the MSE of
        # candidates that lost their never-worse comparison and were discarded,
        # so the aggregate can read worse than the seed even though every piece
        # stored the better of the two. `mse_stored <= mse_seed` is the
        # never-worse property, and it must survive aggregation to be checkable.
        #
        # Read the VERDICT rather than re-deriving it as min(seed, refit): the
        # two agree only while MSE is the sole arbiter. A re-fit rejected for
        # leaving its tile can hold the LOWER MSE and still not be what was
        # stored, and min() would then credit the level with an error it never
        # achieved — precisely in the case the containment guard exists for.
        seed_kept = bool(group.get("seed_won")) or bool(group.get("tile_escape"))
        contribution = {
            "mse_seed": float(group["mse_seed"]),
            "mse_refit": float(group["mse_refit"]),
            "mse_stored": float(group["mse_seed"] if seed_kept else group["mse_refit"]),
        }
        if prev_w + w > 0:
            for key, value in contribution.items():
                sink[key] = (sink.get(key, 0.0) * prev_w + value * w) / (prev_w + w)
        sink["_mse_w"] = prev_w + w


def finalize_volume_refit_stats(sink: Dict) -> Dict:
    """Turn the accumulated counts into fractions and drop bookkeeping keys."""
    sink.pop("_mse_w", None)
    pieces = max(int(sink.get("n_pieces", 0)), 1)
    for key in ("improved", "seed_won", "frame_mismatch", "mass_pinned", "tile_escape"):
        sink[f"{key}_frac"] = float(sink.pop(f"n_{key}", 0)) / pieces
    # `improved` stays present and boolean-ish for callers (and the verbose
    # line) that only ask "did this level get better anywhere?".
    sink["improved"] = bool(sink["improved_frac"] > 0.0)
    sink["frame_mismatch"] = bool(sink["frame_mismatch_frac"] > 0.0)
    sink["seed_won"] = bool(sink["seed_won_frac"] >= 1.0)
    sink["tile_escape"] = bool(sink["tile_escape_frac"] > 0.0)
    return sink
