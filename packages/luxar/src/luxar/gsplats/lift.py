"""Lift Points into isotropic Gaussian splats (point -> gsplat).

This is the bridge that lets the mature gsplat *substitutive* LOD pipeline
(:func:`luxar.gsplats.lod.substitutive.make_substitutive_lod`) coarsen a point
cloud: each point becomes one isotropic Gaussian, the pipeline synthesises
fewer-but-larger representatives, and those become the coarse levels of a points
LOD ladder (the finest level stays the original Points node).

The seed formulas below were calibrated against the viewer shaders (point
``materials/point/shader-glsl.ts`` and gsplat ``materials/gsplat/shader-glsl.ts``)
so a single lifted Gaussian renders like the point it came from:

* **Footprint match.** A point's super-Gaussian sprite truncates to zero at its
  1% iso-contour (``rho = 1``, the sprite edge); a Gaussian truncated at ``T``
  sigmas has its visible edge at ``T * sigma``. The point's screen radius is
  ``R * pointSizeFactor / (2 z)`` and the gsplat's is ``T * f * sigma_world / z``
  with ``pointSizeFactor / f = 4`` (``2 resY/tan`` vs ``resY/(2 tan)``), so
  matching the two screen radii gives, view-independently::

      sigma_world = (pointSizeFactor / (2 f)) * R / T = 2 R / T

  where ``R = radius * radius_scale`` is the point's world radius and ``T`` is the
  gsplat truncation radius (default 3.0). At ``T = 3`` the gsplat Gaussian matches
  the point super-Gaussian profile to 0.45% relative L2 (the kernels coincide
  exactly at ``T* = sqrt(2 ln 100) ≈ 3.035``).

* **Brightness match.** A single isotropic gsplat's peak screen intensity is
  ``a * sigma_world * uRayIntegralFactor`` (the ray-integral boost
  ``vAmplitude2D = a * sigmaRay * uRIF`` with ``sigmaRay = sigma_world`` for an
  isotropic covariance, and ``uInvOneMinusC * (1 - uShiftC) = 1`` at the centre).
  A point's peak alpha is ``opacity``. Equating::

      a_lift = opacity / (uRayIntegralFactor(T) * sigma_world)

The lift is **strictly isotropic** on purpose: ``sigmaRay`` equals ``sigma_world``
only for isotropic covariances, so anisotropy would make brightness view-dependent
and break the seam match. (Lines, which lift to anisotropic Gaussians, are a
separate future problem.)

Sharpness/``beta`` is intentionally NOT used: the point kernel is a *truncated*
super-Gaussian, and an (untruncated) moment-match to ``beta = 2`` overspreads it
badly; ``sigma = 2 R / T`` is the right footprint-preserving choice for all
sharpness. The single-point seam mismatch for non-default sharpness is hidden in
practice because the finest LOD level is the real Points node and coarse levels
merge many points (per-point shape washes out).
"""

from __future__ import annotations

from typing import Any, List, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from .gsplat_data import GSplatData
from .utils.trils import unpack_tril

__all__ = [
    "coarse_substitutive_levels",
    "compute_ray_integral_factor",
    "lift_points_to_gsplats",
    "render_light",
]


def compute_ray_integral_factor(truncation_radius: float) -> float:
    """Shifted-Gaussian ray-integral factor for a truncation of ``T`` sigmas.

    Port of ``materials/gsplat/math.ts::computeRayIntegralFactor`` (Abramowitz &
    Stegun erf approximation, max error ~1.5e-7) so the Python lift and the
    TypeScript renderer agree. Returns
    ``sqrt(2 pi) * erf(T / sqrt 2) - 2 T * exp(-T^2 / 2)`` (≈ 2.433 at ``T = 3``).
    """
    T = float(truncation_radius)
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    x = T / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    erf = 1.0 - t * (
        0.254829592
        + t
        * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    ) * np.exp(-x * x)
    if x < 0:
        erf = -erf
    return float(sqrt_2pi * erf - 2.0 * T * np.exp(-0.5 * T * T))


def lift_points_to_gsplats(
    positions: NDArray,
    radii: Union[NDArray, float],
    colors: Union[NDArray, None] = None,
    opacity: float = 1.0,
    *,
    radius_scale: float = 1.0,
    truncation_radius: float = 3.0,
) -> GSplatData:
    """Lift a point cloud to a single-level :class:`GSplatData` of isotropic Gaussians.

    Each point ``i`` becomes a Gaussian with centre ``positions[i]``, isotropic
    covariance ``sigma_i^2 I`` where ``sigma_i = 2 * radii[i] * radius_scale /
    truncation_radius``, and peak amplitude ``opacity / (uRIF * sigma_i)`` (see the
    module docstring for the calibration). The result is a flat (single
    substitutive level, single additive sub-LOD) ``GSplatData`` ready to feed to
    :func:`luxar.gsplats.lod.substitutive.make_substitutive_lod`.

    Parameters
    ----------
    positions : array, shape (N, d)
        Point centres (any spatial dimensionality ``d``).
    radii : array (N,) or float
        Per-point world radius (the 1% iso-contour radius), before ``radius_scale``.
    colors : array (N, 3) or None
        Per-point RGB (float32, 0..1 or HDR). ``None`` leaves colours unset.
    opacity : float
        Node opacity baked into the lifted amplitude (peak match).
    radius_scale : float
        Mirrors the shader ``radiusScale`` dtype normalisation (e.g. 1/255 for
        uint8 radii). Default 1.0.
    truncation_radius : float
        Gaussian truncation ``T`` in sigmas (default 3.0, the gsplat default).

    Returns
    -------
    GSplatData
        A flat dataset with ``n_splats == N`` valid splats (zero-radius points,
        e.g. from nD slicing, are dropped).
    """
    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2:
        raise ValueError(f"positions must be (N, d); got shape {pos.shape}")
    n, d = pos.shape

    radii_arr = np.broadcast_to(np.asarray(radii, dtype=np.float64), (n,)).astype(
        np.float64
    )
    T = float(truncation_radius)
    if T <= 0:
        raise ValueError(f"truncation_radius must be > 0; got {T}")

    # World radius -> isotropic sigma (footprint match): sigma = 2 R / T.
    sigma = (2.0 * radii_arr * float(radius_scale) / T).astype(np.float64)

    # Drop degenerate (zero-radius) points — they render nothing and would make
    # a singular covariance.
    valid = sigma > 0.0
    if not np.all(valid):
        pos = pos[valid]
        sigma = sigma[valid]
        if colors is not None:
            colors = np.asarray(colors)[valid]
    n = pos.shape[0]

    # Peak match: a = opacity / (uRIF * sigma).
    uRIF = compute_ray_integral_factor(T)
    amplitudes = (float(opacity) / (uRIF * sigma)).astype(np.float32)

    # Isotropic Cholesky L = sigma * I, packed lower-triangular.
    eye = np.eye(d, dtype=np.float64)
    chol_full = np.einsum("n,ij->nij", sigma, eye)
    from .utils.trils import pack_tril

    cholesky_factors = pack_tril(chol_full).astype(np.float32)

    colors_arr = None if colors is None else np.asarray(colors, dtype=np.float32)

    return GSplatData(
        centers=pos,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        colors=colors_arr,
        truncation_radius=T,
    )


def coarse_substitutive_levels(
    lifted: GSplatData,
    *,
    compression_factor: int = 4,
    levels: int = 3,
    method: str = "auto",
    device: Any = "auto",
    seed: Optional[int] = None,
) -> "List[GSplatData]":
    """Coarse substitutive levels of a lifted point cloud (render-light conserved).

    Runs :func:`luxar.gsplats.lod.substitutive.make_substitutive_lod` on
    ``lifted``, **drops level 0** (the 1:1 lifted set — the original Points node
    is the finest LOD level, so a 1:1 gsplat copy would double-render at the
    seam), and **rescales each remaining level's amplitudes** so its
    :func:`render_light` equals the finest (lifted) level's. The substitutive
    L2-optimal amplitude otherwise undershoots total light by ~10-20% over a few
    levels, which would read as zoom-out dimming; the rescale removes it.

    Returns the coarse levels **finest → coarsest** (substitutive index 1..L),
    each a flat :class:`GSplatData`. Empty if the pyramid has no coarser level.
    """
    from .lod.substitutive import make_substitutive_lod

    pyramid = make_substitutive_lod(
        lifted,
        compression_factor=int(compression_factor),
        levels=int(levels),
        method=cast(Any, str(method)),
        device=device,
        seed=seed,
    )
    light0 = render_light(pyramid.at_substitutive(0))
    out: List[GSplatData] = []
    for s in range(1, pyramid.n_substitutive):
        lvl = pyramid.at_substitutive(s).flattened()
        ls = render_light(lvl)
        scale = (light0 / ls) if ls > 0 else 1.0
        out.append(
            GSplatData(
                centers=np.asarray(lvl.centers, dtype=np.float32),
                amplitudes=(
                    np.asarray(lvl.amplitudes, dtype=np.float64) * scale
                ).astype(np.float32),
                cholesky_factors=np.asarray(lvl.cholesky_factors, dtype=np.float32),
                colors=(
                    None if lvl.colors is None else np.asarray(lvl.colors, np.float32)
                ),
                truncation_radius=float(lifted.truncation_radius),
            )
        )
    return out


def render_light(data: GSplatData) -> float:
    """Total emitted light of a (flattened) gsplat set under sum/linear projection.

    For an additive (linear-sum) projection the screen-integrated intensity of one
    isotropic splat is proportional to ``a * sigma^3`` (amplitude * 3D volume
    integral, ``D = 3`` convention), so the conserved "no zoom-dimming" quantity is
    ``sum_i a_i * sigma_geo_i^D`` with ``sigma_geo = det(L)^(1/D)`` the
    geometric-mean sigma. Used by the points-substitutive builder to rescale each
    coarse level's amplitudes so total light is conserved across LOD levels (the
    substitutive L2-optimal amplitude otherwise undershoots by ~9% over 3 levels).
    """
    flat = data.flattened()
    amps = np.asarray(flat.amplitudes, dtype=np.float64)
    chol = np.asarray(flat.cholesky_factors, dtype=np.float64)
    if amps.size == 0:
        return 0.0
    d = int(flat.ndim)
    L = unpack_tril(chol, d)
    sigma_geo = np.abs(np.linalg.det(L)) ** (1.0 / d)
    return float(np.sum(amps * sigma_geo**d))
