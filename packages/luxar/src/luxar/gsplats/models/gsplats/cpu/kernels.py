"""Numba kernels for the fused 3D CPU splatting forward and analytic backward.

Importing this module imports numba and declares the kernels; nothing compiles
until the first call (``cache=True`` then stores the machine code, so later
processes load it instead of recompiling). Only :mod:`.backend` imports it, and
only once a CPU fit has selected the native backend.

The math is the PyTorch reference renderer's
(:func:`luxar.gsplats.models.gsplats.rendering_core._render_gaussians_3d`),
voxel for voxel, so the two agree to float rounding:

* samples at integer voxel coordinates, axes ``[Z, Y, X]``;
* the same AABB: ``radius_i = max(1, ceil(T * sqrt(Sigma_ii)))``, shrunk by the
  amplitude-aware intensity floor, ``[floor(c - r), ceil(c + r)]`` clipped;
* ``D^2 = ||y||^2`` with ``y = L~^-1 (x - mu)`` solved by forward
  substitution, where ``L~`` is ``L`` with its diagonal clamped to >= 1e-6;
* the shifted Gaussian ``a * scale * max(0, exp(-D^2 / 2) - C)``, with
  ``C = exp(-T^2 / 2)`` and ``scale = 1 / (1 - C)``.

What the kernels add is fusion and pruning, not different numbers: a voxel the
truncated ellipsoid does not reach contributes exactly zero (value and
gradient), so each row's x-range is clipped to the ellipsoid analytically
before the loop, and the voxel loop is one ``exp`` plus a few multiply-adds
because ``y0`` is constant per plane and ``y1`` per row.

Parallelism needs no atomics. The forward splits the OUTPUT into tiles (one
z-plane by a band of rows) and each tile loops over the splats whose box covers
its plane, so every voxel has exactly one writer and the sum order is fixed
(the result is deterministic). The backward's per-splat gradients are
independent given dL/dV, so it is a parallel loop over splats that only reads
the gradient volume. Per-splat sums accumulate in float64.

The backward never forms the conic: with ``S_y = sum s y`` and
``S_yy = sum s y y^T`` (``s = dLoss/dD^2`` per voxel), the chain rule through
the triangular solve gives ``dmu = -2 L~^-T S_y`` and
``dL = lower(-2 L~^-T S_yy)``, the latter masked to zero on a clamped
diagonal entry -- exactly the gradient PyTorch's autograd produces for the
reference forward.
"""

from __future__ import annotations

import math

import numba
import numpy as np
from numba import prange

from .backend import kernel_cache_enabled

#: ``cache=True`` unless the backend found no writable cache location.
_CACHE = kernel_cache_enabled()

#: Diagonal clamp of the forward substitution (``fwd_norm2_3d``'s ``min=1e-6``).
_DIAG_CLAMP = 1e-6


@numba.njit(cache=_CACHE, fastmath=False)  # type: ignore[misc]
def _splat_box(  # type: ignore[no-untyped-def]
    centers, Ls, amps, k, shape0, shape1, shape2, truncate, floor_on, floor_value
):
    """``(lo0, lo1, lo2, hi0, hi1, hi2)`` of splat *k* (hi exclusive).

    The reference ``compute_aabb_with_intensity_floor``, one splat at a time.
    """
    shift_c = math.exp(-0.5 * truncate * truncate)
    scale = 1.0 / (1.0 - shift_c)
    tmax = truncate
    if floor_on:
        a = max(float(amps[k]), 1e-12)
        threshold = floor_value / (a * scale) + shift_c
        threshold = min(threshold, 1.0 - 1e-7)
        tmax = math.sqrt(max(-2.0 * math.log(threshold), 0.0))
    los = np.empty(3, np.int64)
    his = np.empty(3, np.int64)
    shape = (shape0, shape1, shape2)
    for i in range(3):
        sigma_sq = 0.0
        for j in range(3):
            v = float(Ls[k, i, j])
            sigma_sq += v * v
        sigma = math.sqrt(max(sigma_sq, 1e-8))
        radius = max(1, int(math.ceil(truncate * sigma)))
        if floor_on:
            radius = min(radius, max(1, int(math.ceil(tmax * sigma))))
        c = float(centers[k, i])
        los[i] = max(int(math.floor(c - radius)), 0)
        his[i] = min(int(math.ceil(c + radius)) + 1, shape[i])
    return los[0], los[1], los[2], his[0], his[1], his[2]


@numba.njit(cache=_CACHE, fastmath=False, inline="always")  # type: ignore[misc]
def _splat_is_finite(centers, Ls, amps, k):  # type: ignore[no-untyped-def]
    """True when every input of splat *k* is finite."""
    if not math.isfinite(float(amps[k])):
        return False
    for i in range(3):
        if not math.isfinite(float(centers[k, i])):
            return False
        for j in range(3):
            if not math.isfinite(float(Ls[k, i, j])):
                return False
    return True


@numba.njit(parallel=True, cache=_CACHE, fastmath=False)  # type: ignore[misc]
def compute_boxes(  # type: ignore[no-untyped-def]
    centers, Ls, amps, shape0, shape1, shape2, truncate, floor_on, floor_value
):
    """Per-splat AABBs ``(N, 6)`` int64 as ``[lo0, lo1, lo2, hi0, hi1, hi2]``.

    An empty box (any ``hi <= lo``) is stored as all zeros, so every consumer
    can skip it with one test on its z-extent. A splat with a non-finite
    center, factor or amplitude gets an empty box too (it renders nothing and
    receives zero gradient) and is flagged in the returned ``nonfinite`` mask.
    """
    n = centers.shape[0]
    boxes = np.zeros((n, 6), np.int64)
    nonfinite = np.zeros(n, np.uint8)
    for k in prange(n):
        if not _splat_is_finite(centers, Ls, amps, k):
            nonfinite[k] = 1
            continue
        lo0, lo1, lo2, hi0, hi1, hi2 = _splat_box(
            centers,
            Ls,
            amps,
            k,
            shape0,
            shape1,
            shape2,
            truncate,
            floor_on,
            floor_value,
        )
        if hi0 > lo0 and hi1 > lo1 and hi2 > lo2:
            boxes[k, 0] = lo0
            boxes[k, 1] = lo1
            boxes[k, 2] = lo2
            boxes[k, 3] = hi0
            boxes[k, 4] = hi1
            boxes[k, 5] = hi2
    return boxes, nonfinite


@numba.njit(cache=_CACHE, fastmath=False)  # type: ignore[misc]
def plane_lists(boxes, depth):  # type: ignore[no-untyped-def]
    """CSR ``(start, splats)``: the splats whose box covers each z-plane.

    ``splats[start[z]:start[z + 1]]`` lists, in index order, every splat with
    ``lo0 <= z < hi0``. Serial and O(sum of z-extents).
    """
    n = boxes.shape[0]
    counts = np.zeros(depth + 1, np.int64)
    for k in range(n):
        for z in range(boxes[k, 0], boxes[k, 3]):
            counts[z + 1] += 1
    for z in range(depth):
        counts[z + 1] += counts[z]
    fill = counts[:-1].copy()
    splats = np.empty(counts[depth], np.int64)
    for k in range(n):
        for z in range(boxes[k, 0], boxes[k, 3]):
            splats[fill[z]] = k
            fill[z] += 1
    return counts, splats


@numba.njit(cache=_CACHE, fastmath=False, inline="always")  # type: ignore[misc]
def _inverse_diagonal(Ls, k):  # type: ignore[no-untyped-def]
    """Reciprocals of the clamped diagonal of ``Ls[k]``."""
    i00 = 1.0 / max(float(Ls[k, 0, 0]), _DIAG_CLAMP)
    i11 = 1.0 / max(float(Ls[k, 1, 1]), _DIAG_CLAMP)
    i22 = 1.0 / max(float(Ls[k, 2, 2]), _DIAG_CLAMP)
    return i00, i11, i22


@numba.njit(cache=_CACHE, fastmath=False, inline="always")  # type: ignore[misc]
def _row_x_range(xoff, half_width, lo2, hi2):  # type: ignore[no-untyped-def]
    """Inclusive x-range of a row the truncated ellipsoid can reach.

    One voxel of slack on each side (``floor``/``ceil``) keeps the analytic cut
    conservative; the exact test is the ``value > 0`` check in the voxel loop.
    """
    xa = max(lo2, int(math.floor(xoff - half_width)))
    xb = min(hi2 - 1, int(math.ceil(xoff + half_width)))
    return xa, xb


@numba.njit(parallel=True, cache=_CACHE, fastmath=False)  # type: ignore[misc]
def forward_3d(  # type: ignore[no-untyped-def]
    centers, Ls, amps, boxes, plane_start, plane_splats, out, truncate, band
):
    """Accumulate every splat into ``out`` (zeroed, shape ``(D, H, W)``).

    Work item ``t`` owns z-plane ``t // n_bands`` rows ``[b * band, (b+1) *
    band)``; it is the only writer of those voxels.
    """
    depth, height, _width = out.shape
    t_sq = truncate * truncate
    shift_c = math.exp(-0.5 * t_sq)
    scale = 1.0 / (1.0 - shift_c)
    n_bands = (height + band - 1) // band
    for item in prange(depth * n_bands):
        z = item // n_bands
        row_lo = (item % n_bands) * band
        row_hi = min(height, row_lo + band)
        for p in range(plane_start[z], plane_start[z + 1]):
            k = plane_splats[p]
            y_lo = max(boxes[k, 1], row_lo)
            y_hi = min(boxes[k, 4], row_hi)
            if y_hi <= y_lo:
                continue
            i00, i11, i22 = _inverse_diagonal(Ls, k)
            l22 = 1.0 / i22
            l10 = float(Ls[k, 1, 0])
            l20 = float(Ls[k, 2, 0])
            l21 = float(Ls[k, 2, 1])
            cz = float(centers[k, 0])
            cy = float(centers[k, 1])
            cx = float(centers[k, 2])
            amp_scale = float(amps[k]) * scale
            y0 = (z - cz) * i00
            q0 = y0 * y0
            if q0 >= t_sq:
                continue
            for y in range(y_lo, y_hi):
                y1 = ((y - cy) - l10 * y0) * i11
                q1 = q0 + y1 * y1
                if q1 >= t_sq:
                    continue
                xoff = cx + l20 * y0 + l21 * y1
                xa, xb = _row_x_range(
                    xoff, math.sqrt(t_sq - q1) * l22, boxes[k, 2], boxes[k, 5]
                )
                for x in range(xa, xb + 1):
                    y2 = (x - xoff) * i22
                    value = math.exp(-0.5 * (q1 + y2 * y2)) - shift_c
                    if value > 0.0:
                        out[z, y, x] += amp_scale * value


@numba.njit(cache=_CACHE, fastmath=False)  # type: ignore[misc]
def _splat_sums(  # type: ignore[no-untyped-def]
    grad, centers, Ls, boxes, k, t_sq, shift_c
):
    """One splat's gradient moments over its truncated ellipsoid.

    Returns ``(sum g * value, S_y[3], S_yy[6])`` with ``s = g * exp(-D^2/2)``
    (the ``-a * scale / 2`` factor of dLoss/dD^2 is applied by the caller) and
    ``S_yy`` packed ``[00, 01, 02, 11, 12, 22]``.
    """
    i00, i11, i22 = _inverse_diagonal(Ls, k)
    l22 = 1.0 / i22
    l10 = float(Ls[k, 1, 0])
    l20 = float(Ls[k, 2, 0])
    l21 = float(Ls[k, 2, 1])
    cz = float(centers[k, 0])
    cy = float(centers[k, 1])
    cx = float(centers[k, 2])
    g_value = 0.0
    sy = np.zeros(3)
    syy = np.zeros(6)
    for z in range(boxes[k, 0], boxes[k, 3]):
        y0 = (z - cz) * i00
        q0 = y0 * y0
        if q0 >= t_sq:
            continue
        for y in range(boxes[k, 1], boxes[k, 4]):
            y1 = ((y - cy) - l10 * y0) * i11
            q1 = q0 + y1 * y1
            if q1 >= t_sq:
                continue
            xoff = cx + l20 * y0 + l21 * y1
            xa, xb = _row_x_range(
                xoff, math.sqrt(t_sq - q1) * l22, boxes[k, 2], boxes[k, 5]
            )
            row_s = 0.0
            row_sy2 = 0.0
            row_sy22 = 0.0
            for x in range(xa, xb + 1):
                g = float(grad[z, y, x])
                y2 = (x - xoff) * i22
                e = math.exp(-0.5 * (q1 + y2 * y2))
                value = e - shift_c
                if value > 0.0:
                    g_value += g * value
                    s = g * e
                    row_s += s
                    row_sy2 += s * y2
                    row_sy22 += s * y2 * y2
            sy[0] += y0 * row_s
            sy[1] += y1 * row_s
            sy[2] += row_sy2
            syy[0] += y0 * y0 * row_s
            syy[1] += y0 * y1 * row_s
            syy[2] += y0 * row_sy2
            syy[3] += y1 * y1 * row_s
            syy[4] += y1 * row_sy2
            syy[5] += row_sy22
    return g_value, sy, syy


@numba.njit(cache=_CACHE, fastmath=False, inline="always")  # type: ignore[misc]
def _solve_lower_transpose(l00, l10, l11, l20, l21, l22, b0, b1, b2):  # type: ignore[no-untyped-def]
    """Solve ``L^T u = b`` for lower-triangular ``L`` (back substitution)."""
    u2 = b2 / l22
    u1 = (b1 - l21 * u2) / l11
    u0 = (b0 - l10 * u1 - l20 * u2) / l00
    return u0, u1, u2


@numba.njit(parallel=True, cache=_CACHE, fastmath=False)  # type: ignore[misc]
def backward_3d(  # type: ignore[no-untyped-def]
    grad, centers, Ls, amps, boxes, d_centers, d_Ls, d_amps, truncate
):
    """Write dLoss/d(centers, Ls, amps) for every splat (outputs pre-zeroed)."""
    n = centers.shape[0]
    t_sq = truncate * truncate
    shift_c = math.exp(-0.5 * t_sq)
    scale = 1.0 / (1.0 - shift_c)
    for k in prange(n):
        if boxes[k, 3] <= boxes[k, 0]:
            continue
        g_value, sy, syy = _splat_sums(grad, centers, Ls, boxes, k, t_sq, shift_c)
        d_amps[k] = scale * g_value
        # dLoss/dD^2 = g * a * scale * exp(-D^2/2) * (-1/2); grad of D^2 = ||y||^2
        # w.r.t. y is 2y, hence the factor -a * scale on S_y and S_yy.
        f = -float(amps[k]) * scale
        raw00 = float(Ls[k, 0, 0])
        raw11 = float(Ls[k, 1, 1])
        raw22 = float(Ls[k, 2, 2])
        l00 = max(raw00, _DIAG_CLAMP)
        l11 = max(raw11, _DIAG_CLAMP)
        l22 = max(raw22, _DIAG_CLAMP)
        l10 = float(Ls[k, 1, 0])
        l20 = float(Ls[k, 2, 0])
        l21 = float(Ls[k, 2, 1])
        # dmu = -L~^-T (2 S_y)
        u0, u1, u2 = _solve_lower_transpose(
            l00, l10, l11, l20, l21, l22, f * sy[0], f * sy[1], f * sy[2]
        )
        d_centers[k, 0] = -u0
        d_centers[k, 1] = -u1
        d_centers[k, 2] = -u2
        # dL = lower(-L~^-T (2 S_yy)), one column of the symmetric S_yy at a time.
        m00 = f * syy[0]
        m01 = f * syy[1]
        m02 = f * syy[2]
        m11 = f * syy[3]
        m12 = f * syy[4]
        m22 = f * syy[5]
        a0, a1, a2 = _solve_lower_transpose(l00, l10, l11, l20, l21, l22, m00, m01, m02)
        _b0, b1, b2 = _solve_lower_transpose(
            l00, l10, l11, l20, l21, l22, m01, m11, m12
        )
        _c0, _c1, c2 = _solve_lower_transpose(
            l00, l10, l11, l20, l21, l22, m02, m12, m22
        )
        d_Ls[k, 0, 0] = -a0 if raw00 >= _DIAG_CLAMP else 0.0
        d_Ls[k, 1, 0] = -a1
        d_Ls[k, 2, 0] = -a2
        d_Ls[k, 1, 1] = -b1 if raw11 >= _DIAG_CLAMP else 0.0
        d_Ls[k, 2, 1] = -b2
        d_Ls[k, 2, 2] = -c2 if raw22 >= _DIAG_CLAMP else 0.0
