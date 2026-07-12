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
and break the seam match. (Lines lift to a *string of isotropic beads* for the very
same reason — never one elongated anisotropic Gaussian — see
:func:`lift_lines_to_gsplats` below.)

Sharpness/``beta`` is intentionally NOT used: the point kernel is a *truncated*
super-Gaussian, and an (untruncated) moment-match to ``beta = 2`` overspreads it
badly; ``sigma = 2 R / T`` is the right footprint-preserving choice for all
sharpness. The single-point seam mismatch for non-default sharpness is hidden in
practice because the finest LOD level is the real Points node and coarse levels
merge many points (per-point shape washes out).

nD note: a point radius is a single isotropic spatial scalar, so the lift assigns
``sigma = 2 R / T`` to **every** axis of ``positions``. For a 3D cloud that is
exactly right. For an nD scene where a non-spatial axis (e.g. a continuous time
coordinate filled in via ``dim_order``) is part of ``positions``, the lifted
Gaussian gains a spurious extent along that axis — the coarse gsplat levels then
blur across it. Use ``extend_to_all`` for such axes (the common case), or restrict
``positions`` to the spatial subspace, until a dim-aware lift lands.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Union, cast

import numpy as np
from numpy.typing import NDArray

from .gsplat_data import GSplatData

__all__ = [
    "coarse_substitutive_levels",
    "compute_ray_integral_factor",
    "lift_lines_to_gsplats",
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
    if T <= 0:
        raise ValueError(f"truncation_radius must be > 0; got {T}")
    sqrt_2pi = np.sqrt(2.0 * np.pi)
    x = T / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    erf = 1.0 - t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
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

    # Peak match: a = opacity / (uRIF * sigma). Surface non-finite amplitudes
    # (a float32 overflow only happens for absurdly tiny radii, < ~1e-39) rather
    # than silently writing inf into the gsplat.
    uRIF = compute_ray_integral_factor(T)
    amplitudes = (float(opacity) / (uRIF * sigma)).astype(np.float32)
    if n and not np.all(np.isfinite(amplitudes)):
        raise ValueError(
            "lift_points_to_gsplats produced non-finite amplitudes — some radii "
            "are too small (sigma underflow). Filter or clamp tiny radii first."
        )

    # Isotropic Cholesky L = sigma * I, packed lower-triangular directly (no
    # dense (N, d, d) intermediate — that is ~d^2/((d+1)/2) larger and mostly
    # zeros, a real memory hazard at the 10M-splat scale). pack_tril order is
    # [L00, L10, L11, ...]; np.tril_indices yields (row, col) in that same order,
    # so the diagonal entries (row == col) carry sigma and the rest stay zero.
    k = d * (d + 1) // 2
    rows, cols = np.tril_indices(d)
    cholesky_factors = np.zeros((n, k), dtype=np.float32)
    cholesky_factors[:, rows == cols] = sigma[:, None].astype(np.float32)

    # Colour dtype normalisation — mirror the point shader's radiusScale/dtype
    # handling: integer RGB (uint8 0..255, uint16 0..65535) must be scaled to
    # float [0, 1], otherwise the gsplat colour writer classifies values > 1 as
    # HDR and the coarse gsplat LOD levels render ~255x too bright (an SDR/HDR
    # flip across the LOD seam vs the SDR-decoded uint8 Points node).
    colors_arr: Optional[NDArray] = None
    if colors is not None:
        c = np.asarray(colors)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(
                "colors must be (N, 3) RGB; gsplats carry no alpha channel — got "
                f"shape {c.shape}. Pass RGB (drop the alpha column) before lifting."
            )
        if np.issubdtype(c.dtype, np.integer):
            colors_arr = c.astype(np.float32) / float(np.iinfo(c.dtype).max)
        else:
            colors_arr = c.astype(np.float32)

    return GSplatData(
        centers=pos,
        amplitudes=amplitudes,
        cholesky_factors=cholesky_factors,
        colors=colors_arr,
        truncation_radius=T,
    )


def _segment_pairs(
    n_vertices: int, line_type: str, indices: Optional[NDArray]
) -> NDArray:
    """Vertex-index pairs ``(E, 2)`` for each edge of a line set."""
    if line_type == "segments":
        m = n_vertices - (n_vertices % 2)
        return np.arange(m, dtype=np.intp).reshape(-1, 2)
    if line_type in ("polyline", "loop"):
        if n_vertices < 2:
            return np.empty((0, 2), dtype=np.intp)
        a = np.arange(n_vertices - 1, dtype=np.intp)
        pairs = np.stack([a, a + 1], axis=1)
        if line_type == "loop":
            pairs = np.vstack([pairs, [[n_vertices - 1, 0]]])
        return pairs
    if line_type == "indexed":
        if indices is None:
            raise ValueError("line_type='indexed' requires an indices edge list")
        return np.asarray(indices, dtype=np.intp).reshape(-1, 2)
    raise ValueError(
        f"line_type must be 'segments'/'polyline'/'loop'/'indexed'; got {line_type!r}"
    )


#: Max beads synthesised per segment — caps the lift-time allocation so a thin
#: (tiny-width) or zero-width segment can't explode ``ceil(L/σ)`` into an OOM.
#: A segment longer than ``MAX_BEADS_PER_SEGMENT * σ`` is under-sampled (beads
#: spaced > σ → a slightly gappy tube), which is acceptable for such extreme
#: aspect ratios; a ``UserWarning`` is emitted when the clamp fires.
MAX_BEADS_PER_SEGMENT: int = 4096

#: Max beads synthesised across the WHOLE line set in one lift — a hard ceiling
#: on the total allocation. Each segment is individually capped by
#: ``MAX_BEADS_PER_SEGMENT``, but their *sum* is not: many long, thin segments
#: could still sum to an OOM. When the requested total exceeds this, bead spacing
#: is widened uniformly to fit the budget (a slightly gappier tube) and a
#: ``UserWarning`` is emitted. 8M keeps the pre-reduction cloud within the
#: project's 10M-element interactive scale target.
MAX_TOTAL_BEADS: int = 8_000_000


def lift_lines_to_gsplats(
    vertices: NDArray,
    widths: Union[NDArray, float],
    line_type: str = "polyline",
    indices: Optional[NDArray] = None,
    colors: Union[NDArray, None] = None,
    opacity: float = 1.0,
    *,
    scalars: Union[NDArray, None] = None,
    colormap: Union[str, NDArray, None] = None,
    radius_scale: float = 1.0,
    truncation_radius: float = 3.0,
    bead_spacing_factor: float = 1.0,
) -> GSplatData:
    """Lift a line set to a flat :class:`GSplatData` of **isotropic bead** Gaussians.

    Each segment is sampled into a string of overlapping isotropic "bead"
    Gaussians spaced ``bead_spacing_factor * σ_perp`` along it (``σ_perp = 2 w /
    T`` per the C0 calibration — same constant as the point lift). Beads are used
    instead of one elongated anisotropic Gaussian per segment because the gsplat
    ray-integral is **view-dependent** for anisotropic covariances (a single
    elongated Gaussian is ``~L/(4w)`` brighter end-on than broadside); isotropic
    beads are view-independent and sum to a smooth tube.

    Bead amplitude conserves the line's centreline brightness: each bead's
    amplitude is divided by the **per-segment** Gaussian-comb sum evaluated at the
    segment midpoint (the sum of all the segment's beads' unit peaks there), so a
    long segment's tube and a short segment's single bead both peak at ``opacity``
    — the asymptotic ``√(2π)`` only applies in the long-segment limit.

    ``scalars`` + ``colormap`` (per-vertex scalar field): the scalar is
    interpolated per bead and then mapped through the colormap LUT
    (interpolate-then-LUT, matching the line shader) — pass these instead of
    pre-baked ``colors`` so non-linear colormaps get correct mid-segment colours.

    Parameters otherwise mirror :func:`lift_points_to_gsplats` plus ``line_type`` /
    ``indices`` (how vertices form edges) and ``bead_spacing_factor``.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim != 2:
        raise ValueError(f"vertices must be (N, d); got shape {verts.shape}")
    n_vertices, d = verts.shape
    T = float(truncation_radius)
    if T <= 0:
        raise ValueError(f"truncation_radius must be > 0; got {T}")

    def _empty() -> GSplatData:
        return lift_points_to_gsplats(
            np.empty((0, d), np.float32),
            np.empty((0,), np.float32),
            colors=None,
            opacity=opacity,
            radius_scale=radius_scale,
            truncation_radius=T,
        )

    pairs = _segment_pairs(n_vertices, line_type, indices)
    if pairs.shape[0] == 0:
        return _empty()  # no edges (e.g. a single-vertex polyline)

    w_arr = np.broadcast_to(np.asarray(widths, dtype=np.float64), (n_vertices,)).astype(
        np.float64
    )
    p0 = verts[pairs[:, 0]].astype(np.float64)  # (E, d)
    p1 = verts[pairs[:, 1]].astype(np.float64)
    w0 = w_arr[pairs[:, 0]]
    w1 = w_arr[pairs[:, 1]]
    seg_len = np.linalg.norm(p1 - p0, axis=1)  # (E,)

    # Per-segment isotropic sigma; drop degenerate (zero/negative-width) segments
    # — they render nothing AND would blow ceil(L/σ) up to an OOM (σ=0 → ∞ beads).
    wbar = 0.5 * (w0 + w1)
    sigma = 2.0 * wbar * float(radius_scale) / T  # (E,)
    keep = sigma > 0.0
    if not np.any(keep):
        return _empty()
    pairs, p0, p1, w0, w1, seg_len, sigma = (
        pairs[keep],
        p0[keep],
        p1[keep],
        w0[keep],
        w1[keep],
        seg_len[keep],
        sigma[keep],
    )

    # Bead count per segment, CAPPED so a tiny-but-positive width can't OOM.
    spacing = float(bead_spacing_factor) * sigma
    raw_n = np.maximum(1, np.ceil(seg_len / spacing).astype(np.intp))
    n_i = np.minimum(raw_n, MAX_BEADS_PER_SEGMENT)  # (E,)
    n_clamped = int(np.count_nonzero(raw_n > MAX_BEADS_PER_SEGMENT))
    if n_clamped:
        warnings.warn(
            f"{n_clamped} segment(s) exceed MAX_BEADS_PER_SEGMENT="
            f"{MAX_BEADS_PER_SEGMENT} (extreme length:width ratio) and were "
            "under-sampled; their tube may look slightly gappy.",
            UserWarning,
            stacklevel=2,
        )

    # TOTAL bead budget: even with each segment individually capped, a large edge
    # count can sum to an OOM. If the total exceeds MAX_TOTAL_BEADS, widen spacing
    # uniformly (thin every segment proportionally, >=1 bead each) so the
    # allocation fits — a slightly gappier tube, never an OOM.
    total = int(n_i.sum())
    if total > MAX_TOTAL_BEADS:
        factor = total / MAX_TOTAL_BEADS
        n_i = np.maximum(1, (n_i / factor).astype(np.intp))
        warnings.warn(
            f"line lift requested {total} beads (> MAX_TOTAL_BEADS="
            f"{MAX_TOTAL_BEADS}); bead spacing widened ~{factor:.1f}x to fit the "
            "budget. Partition or coarsen the line set for full bead resolution.",
            UserWarning,
            stacklevel=2,
        )

    # Ragged expansion: one row per bead, at interval centres t=(k+0.5)/n_i.
    n_seg = pairs.shape[0]
    seg_idx = np.repeat(np.arange(n_seg, dtype=np.intp), n_i)
    starts = np.repeat(np.cumsum(n_i) - n_i, n_i)
    within = np.arange(seg_idx.shape[0], dtype=np.intp) - starts
    t = (within + 0.5) / n_i[seg_idx]  # (B,) in (0, 1)

    bead_centers = (p0[seg_idx] + t[:, None] * (p1[seg_idx] - p0[seg_idx])).astype(
        np.float32
    )
    bead_widths = (w0[seg_idx] + t * (w1[seg_idx] - w0[seg_idx])).astype(np.float32)

    # Per-bead colour. Prefer interpolate-the-scalar-then-LUT (matches the line
    # shader's interpolate-then-LUT order for non-linear colormaps); else
    # interpolate per-vertex colours.
    bead_colors: Optional[NDArray] = None
    if scalars is not None and colormap is not None:
        from luxar.colormaps import scalars_to_colors

        s_arr = np.broadcast_to(
            np.asarray(scalars, dtype=np.float64).reshape(-1), (n_vertices,)
        )
        s0 = s_arr[pairs[:, 0]][seg_idx]
        s1 = s_arr[pairs[:, 1]][seg_idx]
        bead_scalars = s0 + t * (s1 - s0)
        # Normalise over the FULL field range (vmin/vmax from all vertices) so the
        # beads share the finest node's scalar_data_range, not a per-segment one.
        bead_colors = scalars_to_colors(
            bead_scalars, colormap, vmin=float(s_arr.min()), vmax=float(s_arr.max())
        )
    elif colors is not None:
        c = np.asarray(colors)
        c0 = c[pairs[:, 0]].astype(np.float64)[seg_idx]
        c1 = c[pairs[:, 1]].astype(np.float64)[seg_idx]
        interp = c0 + t[:, None] * (c1 - c0)
        bead_colors = (
            interp.round().astype(c.dtype)
            if np.issubdtype(c.dtype, np.integer)
            else interp.astype(c.dtype)
        )

    # Per-segment comb overlap at the midpoint: sum over the segment's beads of
    # exp(-½ (Δ/σ)²), Δ = (t-½)·L. For a long segment this → √(2π) (the comb sum,
    # smooth tube); for a single bead it is 1 (so a lone bead keeps full opacity
    # instead of being √(2π)≈2.5× too dim). bincount reduces per segment.
    off_over_sigma = (t - 0.5) * seg_len[seg_idx] / sigma[seg_idx]
    contrib = np.exp(-0.5 * off_over_sigma * off_over_sigma)
    overlap = np.bincount(seg_idx, weights=contrib, minlength=n_seg)  # (E,)
    overlap_per_bead = np.maximum(overlap[seg_idx], 1e-9)

    # Build via the isotropic point lift at opacity=1 (no beads dropped — all
    # kept segments have σ>0 → positive bead widths), then scale each bead's
    # amplitude by opacity / overlap_i so every segment's tube peaks at opacity.
    lifted = lift_points_to_gsplats(
        bead_centers,
        bead_widths,
        colors=bead_colors,
        opacity=1.0,
        radius_scale=radius_scale,
        truncation_radius=T,
    )
    flat = lifted.flattened()
    if int(flat.n_splats) != bead_centers.shape[0]:  # defensive: alignment broke
        return lifted
    amps = (
        np.asarray(flat.amplitudes, dtype=np.float64)
        * (float(opacity) / overlap_per_bead)
    ).astype(np.float32)
    return GSplatData(
        centers=np.asarray(flat.centers, dtype=np.float32),
        amplitudes=amps,
        cholesky_factors=np.asarray(flat.cholesky_factors, dtype=np.float32),
        colors=None if flat.colors is None else np.asarray(flat.colors, np.float32),
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
    coarsen_dims: Optional[Sequence[int]] = None,
) -> "List[GSplatData]":
    """Coarse substitutive levels of a lifted point cloud (render-light conserved).

    Runs :func:`luxar.gsplats.lod.substitutive.make_substitutive_lod` on
    ``lifted``, **drops level 0** (the 1:1 lifted set — the original Points node
    is the finest LOD level, so a 1:1 gsplat copy would double-render at the
    seam), and **rescales each remaining level's amplitudes** so its
    :func:`render_light` equals the finest (lifted) level's. The substitutive
    L2-optimal amplitude otherwise undershoots total light (~9% over 3 levels at
    K=4), which would read as zoom-out dimming; the rescale removes it.

    Returns the coarse levels **finest → coarsest** (substitutive index 1..L),
    each a flat :class:`GSplatData` — one entry per synthesised coarser level.
    Empty only if the pyramid collapsed to level 0 alone; for degenerate input
    the coarsest entry (``[-1]``) may itself have ``n_splats == 0`` (callers
    treat both as "no usable coarse levels" — see the adders' degenerate guards).
    """
    from .lod.substitutive import make_substitutive_lod

    pyramid = make_substitutive_lod(
        lifted,
        compression_factor=int(compression_factor),
        levels=int(levels),
        method=cast(Any, str(method)),
        device=device,
        seed=seed,
        coarsen_dims=coarsen_dims,
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

    ``sigma_geo^D == |det(L)|`` and ``L`` is lower-triangular, so the determinant
    is the product of the packed diagonal — no ``unpack_tril`` or general LU
    needed (avoids a second dense ``(N, d, d)`` transient at scale).
    """
    flat = data.flattened()
    amps = np.asarray(flat.amplitudes, dtype=np.float64)
    if amps.size == 0:
        return 0.0
    d = int(flat.ndim)
    # Packed lower-tri diagonal positions: [0, 2, 5, ...] = cumsum(1..d) - 1.
    # Index the diagonal columns on the float32 source FIRST, then widen only
    # that (N, d) slice — avoids casting the whole packed (N, k) array to float64.
    diag_idx = np.cumsum(np.arange(1, d + 1)) - 1
    chol_diag = np.asarray(flat.cholesky_factors)[:, diag_idx].astype(np.float64)
    det = np.abs(np.prod(chol_diag, axis=1))  # |det(L)| == sigma_geo^D
    return float(np.sum(amps * det))
