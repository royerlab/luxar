"""Shared 3D vector-field helpers.

Used by demos that build a smoothed Cartesian vector field over a cubic
domain and integrate streamlines through it (currently the PPI flow-field
demo and the zebrahub RNA-velocity-streamlines demo).

The helpers here are deliberately demo-agnostic — they operate on
``FlowField`` plus raw points/scenes — so each demo can keep its own
binning, smoothing, caching, and streamline-seeding policy.

Public API
----------
- :class:`FlowField` — frozen dataclass representing a cubic vector field
  with regular voxel spacing.
- :func:`cubic_bounds` — symmetric cubic AABB around point clouds.
- :func:`trilinear_vector` — sample a ``FlowField`` at world points.
- :func:`unit_flow` — direction-only sample (NaN where the field is zero
  or out-of-bounds).
- :func:`rk4_step` — vectorized 4-stage RK4 advection step.
- :func:`add_reference_cube_to_scene` — visualize the field's cubic
  domain as a 12-edge wire cube on a Luxar scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..typing_utils.constants import DEFAULT_BLENDING_MODE


@dataclass(frozen=True)
class FlowField:
    """Cubic vector field on a regular grid.

    Attributes:
        vectors: ``(n, n, n, 3)`` float32. Vectors[i, j, k] holds the
            field value at grid point ``grid_min + (i, j, k) * spacing``.
        grid_min: ``(3,)`` float32, world-space min corner.
        grid_max: ``(3,)`` float32, world-space max corner.
        spacing: Voxel size (uniform along all 3 axes).
        cache_key: Free-form identifier used by callers for on-disk cache
            invalidation.
    """

    vectors: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    spacing: float
    cache_key: str


def cubic_bounds(
    coords: np.ndarray, pad_fraction: float = 0.06
) -> tuple[np.ndarray, np.ndarray]:
    """Return a symmetric cubic AABB around ``coords``.

    The cube is centered on the input cloud and sized so the longest
    axis-aligned side fits, plus a fractional padding. The minimum
    half-side is clamped to 1.0 to avoid degenerate domains for tiny
    point clouds.

    Args:
        coords: ``(N, 3)`` array of world-space points.
        pad_fraction: Extra padding as a fraction of the longest side.

    Returns:
        ``(grid_min, grid_max)`` — both shape ``(3,)`` float32.
    """
    lo = coords.min(axis=0).astype(np.float32)
    hi = coords.max(axis=0).astype(np.float32)
    center = (lo + hi) * 0.5
    side = float(np.max(hi - lo))
    half = max(side * (0.5 + pad_fraction), 1.0)
    return (center - half).astype(np.float32), (center + half).astype(np.float32)


def trilinear_vector(field: FlowField, points: np.ndarray) -> np.ndarray:
    """Trilinearly interpolate ``field.vectors`` at world ``points``.

    Out-of-bounds points return NaN rows so the caller can detect them
    via ``np.isfinite``.

    Args:
        field: The flow field.
        points: ``(M, 3)`` float32 world-space points.

    Returns:
        ``(M, 3)`` float32 interpolated vectors. NaN rows for any sample
        whose lower corner index is outside ``[0, n-2]`` along any axis.
    """
    arr = field.vectors
    n = arr.shape[0]
    idx = (points - field.grid_min[None, :]) / np.float32(field.spacing)
    x = idx[:, 0]
    y = idx[:, 1]
    z = idx[:, 2]

    ix0 = np.floor(x).astype(np.int32)
    iy0 = np.floor(y).astype(np.int32)
    iz0 = np.floor(z).astype(np.int32)
    valid = (
        (ix0 >= 0)
        & (iy0 >= 0)
        & (iz0 >= 0)
        & (ix0 < n - 1)
        & (iy0 < n - 1)
        & (iz0 < n - 1)
    )

    out = np.full((len(points), 3), np.nan, dtype=np.float32)
    if not np.any(valid):
        return out

    ix = ix0[valid]
    iy = iy0[valid]
    iz = iz0[valid]
    dx = (x[valid] - ix).astype(np.float32)
    dy = (y[valid] - iy).astype(np.float32)
    dz = (z[valid] - iz).astype(np.float32)

    c000 = arr[ix, iy, iz]
    c100 = arr[ix + 1, iy, iz]
    c010 = arr[ix, iy + 1, iz]
    c110 = arr[ix + 1, iy + 1, iz]
    c001 = arr[ix, iy, iz + 1]
    c101 = arr[ix + 1, iy, iz + 1]
    c011 = arr[ix, iy + 1, iz + 1]
    c111 = arr[ix + 1, iy + 1, iz + 1]

    c00 = c000 * (1.0 - dx[:, None]) + c100 * dx[:, None]
    c10 = c010 * (1.0 - dx[:, None]) + c110 * dx[:, None]
    c01 = c001 * (1.0 - dx[:, None]) + c101 * dx[:, None]
    c11 = c011 * (1.0 - dx[:, None]) + c111 * dx[:, None]
    c0 = c00 * (1.0 - dy[:, None]) + c10 * dy[:, None]
    c1 = c01 * (1.0 - dy[:, None]) + c11 * dy[:, None]
    out[valid] = c0 * (1.0 - dz[:, None]) + c1 * dz[:, None]
    return out.astype(np.float32)


def unit_flow(field: FlowField, points: np.ndarray) -> np.ndarray:
    """Return unit-length flow direction at ``points``.

    NaN rows for out-of-bounds samples and for zero-magnitude vectors
    (numerical floor 1e-7).
    """
    vectors = trilinear_vector(field, points)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-7)
    out = np.full_like(vectors, np.nan)
    out[valid] = vectors[valid] / norms[valid, None]
    return out


def rk4_step(points: np.ndarray, step_size: float, field: FlowField) -> np.ndarray:
    """Vectorized 4-stage Runge-Kutta step for ``dx/ds = unit_flow(x)``.

    Returns ``(N, 3)`` advected points; rows that fall out of bounds at
    any RK stage are NaN-filled so the caller can stop integrating those
    streamlines.
    """
    out = np.full_like(points, np.nan)

    k1 = unit_flow(field, points)
    valid = np.isfinite(k1).all(axis=1)
    if not np.any(valid):
        return out

    src = np.flatnonzero(valid)
    p = points[src]
    kk1 = k1[src]

    k2 = unit_flow(field, p + 0.5 * step_size * kk1)
    valid = np.isfinite(k2).all(axis=1)
    if not np.any(valid):
        return out
    src = src[valid]
    p = p[valid]
    kk1 = kk1[valid]
    kk2 = k2[valid]

    k3 = unit_flow(field, p + 0.5 * step_size * kk2)
    valid = np.isfinite(k3).all(axis=1)
    if not np.any(valid):
        return out
    src = src[valid]
    p = p[valid]
    kk1 = kk1[valid]
    kk2 = kk2[valid]
    kk3 = k3[valid]

    k4 = unit_flow(field, p + step_size * kk3)
    valid = np.isfinite(k4).all(axis=1)
    if not np.any(valid):
        return out
    src = src[valid]
    p = p[valid]
    kk1 = kk1[valid]
    kk2 = kk2[valid]
    kk3 = kk3[valid]
    kk4 = k4[valid]

    out[src] = p + (step_size / 6.0) * (kk1 + 2.0 * kk2 + 2.0 * kk3 + kk4)
    return out.astype(np.float32)


def add_reference_cube_to_scene(
    scene: Any,
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    *,
    name: str = "Reference cube",
    widths: float = 0.006,
    opacity: float = 0.18,
    intensity: float = 0.35,
    color: tuple[float, float, float] = (0.55, 0.62, 0.80),
    sharpness: float = 0.8,
    blending_mode: str = DEFAULT_BLENDING_MODE,
    layer: bool = True,
    visible: bool = False,
) -> None:
    """Add a 12-edge wire cube as Lines geometry to a Luxar scene.

    Used by flow-field demos to visualize the cubic field domain. Defaults
    are tuned for a faint, off-by-default reference. Override kwargs to
    match your scene's intensity/exposure scaling.
    """
    x0, y0, z0 = grid_min.tolist()
    x1, y1, z1 = grid_max.tolist()
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # vertical edges
        ],
        dtype=np.uint32,
    )
    scene.add_lines(
        name,
        vertices=corners,
        widths=widths,
        colors=color,
        sharpness=sharpness,
        indices=edges.ravel(),
        line_type="indexed",
        opacity=opacity,
        intensity=intensity,
        blending_mode=blending_mode,
        layer=layer,
        visible=visible,
    )
