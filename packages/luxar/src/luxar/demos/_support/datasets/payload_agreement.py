"""Payload-to-splat correspondence checks for fitted demo data."""

from __future__ import annotations

from typing import Optional

import numpy as np

# |center| at or above this has no int64 voxel index (the cast would overflow),
# so such a row is excluded from the agreement rather than cast.
_VOXEL_KEY_LIMIT = 2.0**62


def voxel_sampled_payload_agreement(
    centers: np.ndarray,
    payload: np.ndarray,
    *,
    min_pairs: int = 1024,
) -> Optional[float]:
    """Fraction of same-voxel splat pairs that carry an identical payload row.

    Several demos ship a per-splat payload (an organ label, a sampled RGB) as a
    SEPARATE sidecar file, indexed positionally against a ``.gsplats.zarr`` fit.
    The data manifest marks these companions with ``positional_pair`` so fetches
    keep their generations aligned; this check validates their actual row order.
    Nothing in either file records the correspondence, so a sidecar written in a
    different splat order than the store — ``GSplatData.save`` applies a spatial
    ordering, so the stored order is NOT the in-memory one — loads silently and
    renders plausible nonsense. This is the cheap check that catches it.

    The invariant it tests: the payload was produced by NEAREST-VOXEL sampling of
    a volume at the splat centers (``np.round`` the center, index the volume), so
    two splats whose centers round to the SAME voxel necessarily read the SAME
    value. Aligned data satisfies that exactly (agreement 1.0); a permuted
    sidecar pairs each voxel with unrelated rows and scores at the chance level
    of the payload's own value distribution.

    HOW STRONG THE VERDICT IS depends on the payload's own value diversity, not
    on this function: a shuffled sidecar still scores at that payload's chance
    level ``Σ p_v²`` (measured on the two shipped payloads: 0.027 for the CT's
    117 organ labels, 1.4e-05 for the Visible Human's sampled uint8 RGB), so a
    payload that is NEARLY CONSTANT scores near 1.0 however badly it is
    permuted. A caller whose payload has little diversity must not rely on this
    check. NaN is likewise invisible to it: ``NaN == NaN`` is False, so a
    float payload using NaN as "no data" scores ~0 even when perfectly aligned
    (neither Luxar caller can hit that — int labels and uint8-derived colours).

    PRECONDITION — ``centers`` must be in the voxel coordinates the payload was
    sampled in, i.e. straight off ``GSplatData.load``, before any centring,
    scaling or other transform. Recentred centers round to different voxels and
    the collision structure the test relies on is lost.

    Args:
        centers: ``(N, D)`` splat centers. EVERY column takes part in the voxel
            key: a stacked/nD fit puts the spatial dims first and the stacked
            axis LAST, so keying on three columns alone would fold every
            timepoint of a voxel together and reject an aligned sidecar. A row
            that has no integer voxel — a non-finite center, or a magnitude at
            or above ``_VOXEL_KEY_LIMIT`` — is excluded (it cannot be judged).
        payload: ``(N,)`` or ``(N, C)`` per-splat values sampled at those centers.
        min_pairs: Minimum number of same-voxel pairs required to return a
            verdict. Clamped to at least 1: with zero pairs there is nothing to
            divide by, so "no evidence" must stay ``None`` rather than raise
            ``ZeroDivisionError``.

    Returns:
        The agreement fraction in ``[0, 1]``, or ``None`` when fewer than
        ``min_pairs`` same-voxel pairs exist — too little evidence to judge, which
        a caller must treat as "unverifiable", NOT as a failure.

    Raises:
        ValueError: if ``centers`` and ``payload`` have different lengths, or
            ``centers`` is not 2-D, or ``centers`` has no columns.
    """
    centers = np.asarray(centers)
    payload = np.asarray(payload)
    if len(centers) != len(payload):
        raise ValueError(
            f"centers and payload length mismatch: {len(centers)} != {len(payload)}"
        )
    if centers.ndim != 2:
        raise ValueError(f"centers must be 2-D (N, D), got shape {centers.shape}")
    if centers.shape[1] == 0:
        # No columns is no voxel key at all. Rejected explicitly because the
        # lexsort below raises a bare `TypeError: need sequence of keys with
        # len > 0` there, which reads as an internal bug rather than as the
        # caller's malformed input.
        raise ValueError(f"centers must have at least one column, got {centers.shape}")
    # A caller-supplied floor of 0 would let a pair-free input reach the final
    # division; one pair is the least that can be judged.
    min_pairs = max(int(min_pairs), 1)

    # A center that has no int64 voxel — NaN, ±inf, or a magnitude that overflows
    # the cast — must be excluded BEFORE the cast below: `astype(np.int64)` warns
    # bare on such a row ("invalid value encountered in cast", fatal under
    # `-W error`) and collapses every one of them onto ONE sentinel voxel,
    # inventing collisions between splats that share nothing. Dropping them costs
    # nothing: they are unjudgeable, and finite in-range data is unaffected. This
    # test itself is warning-free (`isfinite`/`abs` on a float array are total),
    # so it needs no `errstate` of its own.
    judgeable = np.isfinite(centers) & (np.abs(centers) < _VOXEL_KEY_LIMIT)
    keep = np.flatnonzero(judgeable.all(axis=1))

    # Sort by voxel index so same-voxel splats become adjacent (O(N log N)).
    # `voxels.T[::-1]` makes column 0 the primary lexsort key, for any D.
    voxels = np.rint(centers[keep]).astype(np.int64)
    sort = np.lexsort(voxels.T[::-1])
    voxels = voxels[sort]
    order = keep[sort]
    collides = np.flatnonzero((voxels[1:] == voxels[:-1]).all(axis=1))
    n_pairs = int(collides.size)
    if n_pairs < min_pairs:
        return None

    # Index only the colliding pairs — the payload can be wide and long.
    a = payload[order[collides]]
    b = payload[order[collides + 1]]
    equal = a == b if a.ndim == 1 else (a == b).all(axis=tuple(range(1, a.ndim)))
    return float(np.count_nonzero(equal)) / float(n_pairs)
