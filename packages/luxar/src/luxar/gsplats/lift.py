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
:func:`lift_lines_to_gsplats` below.) The substitutive merge would silently
re-introduce anisotropy — Morton bins chunk a bead string into elongated
representatives whose aspect grows ~K× per level — so
:func:`coarse_substitutive_levels` caps each coarse splat's aspect at
``max_aspect`` (default 3, mass-preserving; see :func:`_cap_aspect`) and uses
per-bin mass-preserving amplitudes, keeping every level's brightness and hue
view-coherent with the finest one.

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
    "LIFT_TRUNCATION_RADIUS",
    "coarse_point_levels",
    "coarse_substitutive_levels",
    "compute_ray_integral_factor",
    "lift_lines_to_gsplats",
    "lift_points_to_gsplats",
    "render_light",
]


def coarse_point_levels(
    order: NDArray[np.integer],
    energy: NDArray[np.floating],
    *,
    compression_factor: int,
    levels: int,
) -> List[tuple[NDArray[np.intp], float]]:
    """Build exact-count same-type point levels from a spatial sampling order.

    The returned levels are finest→coarsest. Each entry contains the selected
    original-row indices and the float gain that exactly conserves the finest
    level's :func:`~luxar.core.group.lod.points.compute_points_energy` sum.
    Callers apply that gain only for additive/luminous projection; preserving
    summed light keeps the original point radius rather than inflating coverage.
    """
    permutation = np.asarray(order, dtype=np.intp).reshape(-1)
    weights = np.asarray(energy, dtype=np.float64).reshape(-1)
    if permutation.size != weights.size:
        raise ValueError(
            f"order has {permutation.size} entries but energy has {weights.size}"
        )
    n_points = int(permutation.size)
    if n_points < 2:
        return []

    reference_energy = float(np.sum(weights))
    out: List[tuple[NDArray[np.intp], float]] = []
    previous_count = n_points
    for level in range(1, int(levels) + 1):
        count = max(1, n_points // (int(compression_factor) ** level))
        if count >= previous_count:
            continue
        indices = permutation[:count].copy()
        selected_energy = float(np.sum(weights[indices]))
        gain = reference_energy / selected_energy if selected_energy > 0.0 else 1.0
        out.append((indices, gain))
        previous_count = count
    return out


#: Truncation radius ``T`` used by the lift — deliberately NOT
#: :data:`luxar.typing_utils.constants.DEFAULT_TRUNCATION_RADIUS` (2.75).
#:
#: Here ``T`` is a *profile-matching* parameter, not a render default. The point
#: super-Gaussian sprite truncates at its 1% iso-contour, so it coincides with a
#: truncated Gaussian exactly at ``T* = sqrt(2 ln 100) = 3.0349``; 3.0 is the
#: nearby round value the seed formulas were calibrated against.
#:
#: Moving this to 2.75 would degrade the profile match by ~8.5x. The exact
#: percentage depends on the norm convention (the module docstring above quotes
#: 0.45% at ``T = 3`` under its own normalisation), but the *ratio* is stable
#: across every weighting — e.g. radial-weighted (``u^2``) relative L2 gives
#: 0.00% / 1.96% / 16.91% at ``T*`` / 3.0 / 2.75, and the unweighted 1D profile
#: gives 0.00% / 0.91% / 7.60%.
#:
#: So the divergence from the codebase default is intentional and load-bearing.
LIFT_TRUNCATION_RADIUS: float = 3.0


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


#: Integer colour dtypes the leaf write accepts (``encoding._encoders.base``:
#: "Integer COLOR arrays must use dtype uint8 or uint16"). Floating point is
#: accepted at any width; everything else — a wider/signed integer, bool — is
#: refused by the leaf.
_LEAF_COLOR_INT_DTYPES = (np.dtype("uint8"), np.dtype("uint16"))


def _reject_unwritable_color_dtype(
    colors: Union[NDArray, Sequence[float], None], context: str
) -> None:
    """Refuse a colour dtype the leaf write would reject, before anything is WRITTEN.

    ``None`` is a no-op (no colours, nothing to refuse), so callers can run this
    unconditionally.

    Same rule as the leaf, deliberately not a new one. Without this the lift
    happily normalises e.g. an ``int64`` colour by ``iinfo(int64).max`` (≈1e-19,
    a near-black coarse level), the coarse gsplat children are written, and only
    the FINEST child — written last — trips the encoder's dtype check: a
    ``kind=lod`` node stranded half-written on disk, which is exactly the class
    the #1437 pre-split gate exists to prevent. Raising here keeps the STORE
    untouched, for the uniform ``(1, c)`` row and the per-element ``(N, c)``
    array alike. (Not "before anything is computed": on the per-element points
    path this runs after sigma, the zero-radius mask and the Cholesky
    allocation — all in-memory work, none of it on disk.)
    """
    if colors is None:
        return
    dtype = np.asarray(colors).dtype
    if np.issubdtype(dtype, np.floating) or dtype in _LEAF_COLOR_INT_DTYPES:
        return
    raise ValueError(
        f"{context} has dtype {dtype}; a COLOR array must be floating point, or "
        "integer uint8/uint16 (the leaf writer's rule). Cast the colours, or "
        "pass a uniform colour as an RGB(A) tuple."
    )


def _expand_uniform_colors(
    colors: Union[NDArray, Sequence[float]], n_elements: int
) -> "tuple[Any, bool]":
    """Expand a uniform (broadcast) RGB(A) colour to a per-element ``(n, c)`` array.

    Returns ``(colors, expanded)``. A bare RGB(A) list/tuple and a ``(1, c)`` row
    are legal, documented colour forms on every other path (the flat write,
    ``partition=``, ``additive_lod=``), so the lift honours them too: a uniform
    colour is exactly the case a coarse level can carry trivially, every merged
    representative being that same colour (#1444). Anything else — a per-element
    ``(N, c)`` array, a list of triples, a malformed value — is returned
    UNCHANGED with ``expanded=False`` so the caller's own shape check produces
    its usual message.

    ``expanded`` is what lets the caller accept **four** columns without opening
    the door to a genuine per-element RGBA (which stays refused: the substitutive
    merge is untested on a varying alpha). A uniform alpha is preserved all the
    way to the coarse levels because all three shaders consume per-element alpha
    as a linear intensity scale (``-ln(1-a)`` in volumetric), so dropping it
    would make the node jump ``1/alpha`` brighter the moment the ladder switches
    off the finest child — the very LOD seam this module's mass/aspect/light
    machinery exists to keep flat.

    Two rules come from the leaf writers rather than being invented here:

    * **Value scale.** A list/tuple is ALWAYS the uniform form and its components
      are taken at FACE VALUE, exactly like the leaf writer's tuple branch
      (``dataset_writers.colors.write_colors`` classifies HDR by value and never
      divides by 255). Hence the float32 conversion here — it keeps an integer
      tuple such as ``(255, 0, 0)`` out of the callers' integer-dtype
      normalisation, which applies to arrays only (where ``uint8`` really does
      mean 0..255, as it does at the leaf).
    * **Row dtype.** A ``(1, c)`` ARRAY is held to the leaf's dtype rule
      (:func:`_reject_unwritable_color_dtype`) *here*, before the bead expansion
      on the Lines path, rather than only at the shared normalisation block
      further down.

    The list/tuple admission test is a deliberate MIRROR of
    :func:`luxar.core.group.compositing.is_broadcast_color` (importing it would
    invert the core → gsplats layering). It is only the type/shape half: unlike
    :func:`~luxar.io._compiler.node_common.validate_broadcast_color` it does not
    range-check the components, so a negative / non-finite / ``alpha > 1`` tuple
    is expanded here and left to the caller's own gate — which, via the scene
    API, is the #1437 pre-split gate, and it runs before this function.
    """
    if isinstance(colors, (list, tuple)):
        if len(colors) not in (3, 4) or not all(
            isinstance(c, (int, float, np.integer, np.floating)) for c in colors
        ):
            return colors, False
        row = np.asarray(colors, dtype=np.float32).reshape(1, -1)
    else:
        # Return the CONVERTED array on the pass-through branch: the caller
        # re-asarray's it anyway, and this way a non-ndarray input is converted
        # once and its real shape appears in the caller's error message.
        arr = np.asarray(colors)
        if arr.ndim != 2 or arr.shape[0] != 1 or arr.shape[1] not in (3, 4):
            return arr, False
        _reject_unwritable_color_dtype(arr, "uniform colors row")
        row = arr
    # Read-only 0-stride view on purpose: the consumers all fancy-index or
    # astype it, each of which copies, so materialising here would only add a
    # SECOND (N, c) transient — not avoid one.
    return np.broadcast_to(row, (n_elements, row.shape[1])), True


def _resolve_uniform_colors(
    colors: Union[NDArray, Sequence[float], None],
    n_elements: int,
    declared: Optional[bool],
) -> "tuple[Any, bool]":
    """Settle whether ``colors`` is the uniform form, expanding it if so.

    ``declared`` is a caller's already-final verdict (see the ``_uniform_colors``
    parameter of :func:`lift_points_to_gsplats`): supplied, it is taken verbatim
    and ``colors`` is passed through untouched — no re-classification, no
    expansion. Otherwise the verdict is :func:`_expand_uniform_colors`'s.
    """
    if declared is not None:
        return colors, bool(declared)
    if colors is None:
        return colors, False
    return _expand_uniform_colors(colors, n_elements)


def lift_points_to_gsplats(
    positions: NDArray,
    radii: Union[NDArray, float],
    colors: Union[NDArray, Sequence[float], None] = None,
    opacity: float = 1.0,
    *,
    radius_scale: float = 1.0,
    truncation_radius: float = LIFT_TRUNCATION_RADIUS,
    _uniform_colors: Optional[bool] = None,
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
    colors : array (N, 3), uniform RGB(A), or None
        Per-point RGB (float32, 0..1 or HDR). A **uniform** colour — an RGB(A)
        list/tuple or a ``(1, c)`` row — is broadcast to all N points, ALPHA
        INCLUDED: gsplats carry per-splat alpha end to end (``GSplatData.colors``
        is ``(N, 3)`` or ``(N, 4)``, and every shader scales intensity by it), so
        a uniform ``(r, g, b, a)`` keeps rendering like the node it coarsens.
        Per-element ``(N, 4)`` RGBA is refused — the substitutive merge is
        untested on a VARYING alpha, and only the uniform case is trivially
        exact. ``None`` leaves colours unset.
    opacity : float
        Node opacity baked into the lifted amplitude (peak match).
    radius_scale : float
        Mirrors the shader ``radiusScale`` dtype normalisation (e.g. 1/255 for
        uint8 radii). Default 1.0.
    truncation_radius : float
        Gaussian truncation ``T`` in sigmas. Defaults to
        :data:`LIFT_TRUNCATION_RADIUS` (3.0) — NOT the codebase-wide
        ``DEFAULT_TRUNCATION_RADIUS``; see that constant for why.
    _uniform_colors : bool or None
        PRIVATE. ``None`` (the default) means "nobody has classified ``colors``
        yet" and this function resolves it itself. Supplying a bool means the
        CALLER already resolved uniformity, and its verdict is final: no
        re-classification happens here, and the bool alone decides whether a 4th
        (alpha) column is admitted. Only :func:`lift_lines_to_gsplats` supplies
        it — it classifies per VERTEX and then interpolates per bead, and
        re-classifying the bead array would misread a one-bead ``(1, 4)`` result
        as the uniform form and let a genuine per-element RGBA through with an
        invented (averaged) alpha. Supplying a bool therefore also ASSERTS that
        ``colors`` is already one row per element: the expansion is skipped
        along with the classification, and an unexpanded ``(1, c)`` row would
        reach the zero-radius mask below (which indexes per element).

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

    # Uniform colours BEFORE the zero-radius mask below (which indexes `colors`
    # per point and would corrupt a 3/4-component broadcast row). A caller that
    # already classified them (`_uniform_colors` supplied) is trusted verbatim —
    # re-classifying here would read a caller's one-row per-element array as the
    # uniform form and admit an alpha column it must not.
    colors, uniform_colors = _resolve_uniform_colors(colors, n, _uniform_colors)

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
        # A 4th (alpha) column is admitted only for a colour that came from the
        # UNIFORM form: gsplats carry per-splat alpha fine, but the substitutive
        # merge is untested on a varying one, whereas a constant survives every
        # merge trivially.
        allowed_channels = (3, 4) if uniform_colors else (3,)
        if c.ndim != 2 or c.shape[1] not in allowed_channels:
            raise ValueError(
                "colors must be (N, 3) RGB, or a UNIFORM RGB(A) colour (an RGB(A) "
                "tuple or a (1, c) row) broadcast to every element; per-element "
                f"RGBA is not supported by the lift — got shape {c.shape}. Pass "
                "RGB (drop the alpha column) before lifting."
            )
        # Same leaf dtype rule as the uniform row, applied to EVERY colour that
        # reaches the store: a per-element int64/uint32/int8 array would
        # otherwise normalise to near-black here and be refused by the encoder
        # only at the finest child, stranding the coarse levels on disk.
        _reject_unwritable_color_dtype(c, "colors")
        # order="C" is cheap insurance, not a fix for a reproduced bug: `c` may
        # be a 0-stride broadcast view, and the default order="K" astype turns
        # that into a channel-major (Fortran) array — an odd layout for one
        # input class only. No downstream breakage was observed without it (the
        # uniform path collapses to a (1, c) row on disk), and the copy happens
        # either way, so pin the layout every other input already has.
        if np.issubdtype(c.dtype, np.integer):
            colors_arr = c.astype(np.float32, order="C") / float(np.iinfo(c.dtype).max)
        else:
            colors_arr = c.astype(np.float32, order="C")

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
        indices_arr = np.asarray(indices)
        if not np.issubdtype(indices_arr.dtype, np.integer):
            raise ValueError(
                f"Indices must be an integer array, got dtype {indices_arr.dtype}"
            )
        if indices_arr.size > 0:
            min_index = int(np.min(indices_arr))
            max_index = int(np.max(indices_arr))
            if min_index < 0:
                raise ValueError(f"Index {min_index} < 0 (indices must be >= 0)")
            if max_index >= n_vertices:
                raise ValueError(f"Index {max_index} >= n_vertices {n_vertices}")
        return indices_arr.astype(np.intp, copy=False).reshape(-1, 2)
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
    colors: Union[NDArray, Sequence[float], None] = None,
    opacity: float = 1.0,
    *,
    scalars: Union[NDArray, None] = None,
    colormap: Union[str, NDArray, None] = None,
    radius_scale: float = 1.0,
    truncation_radius: float = LIFT_TRUNCATION_RADIUS,
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
    ``indices`` (how vertices form edges) and ``bead_spacing_factor`` —
    ``colors`` included, so a uniform RGB(A) list/tuple or ``(1, c)`` row is
    broadcast to every vertex, alpha included, before the per-bead
    interpolation (interpolating a constant alpha yields that same constant, so
    the beads stay uniformly transparent). Uniformity is decided ONCE, per
    vertex, and forwarded to the inner point lift: the bead array must never be
    re-classified, or a line set that collapses to a single bead would present a
    per-element RGBA as a ``(1, 4)`` "uniform" row and slip an averaged alpha
    into the coarse levels.
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim != 2:
        raise ValueError(f"vertices must be (N, d); got shape {verts.shape}")
    n_vertices, d = verts.shape
    T = float(truncation_radius)
    if T <= 0:
        raise ValueError(f"truncation_radius must be > 0; got {T}")

    # Uniform colours BEFORE the per-vertex gather below (`c[pairs[:, 0]]`),
    # which would otherwise read a 3/4-component broadcast row as if its
    # components were vertex rows and raise a bare IndexError (#1444).
    colors, uniform_colors = _resolve_uniform_colors(colors, n_vertices, None)
    # Leaf dtype rule up front, so a per-element int64 colour fails before the
    # (potentially huge) bead expansion rather than inside it.
    _reject_unwritable_color_dtype(colors, "colors")

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
        # Over the FINITE vertices only, as `scalars_to_colors` does for its own
        # default bounds: a single Inf would otherwise collapse the whole tube to
        # LUT[0], and a NaN/-Inf bound turns every bead index into garbage. The
        # scene API refuses non-finite scalars upstream; a direct caller of this
        # function does not go through that gate.
        finite = np.isfinite(s_arr)
        vmin = float(s_arr[finite].min()) if finite.any() else 0.0
        vmax = float(s_arr[finite].max()) if finite.any() else 1.0
        bead_colors = scalars_to_colors(bead_scalars, colormap, vmin=vmin, vmax=vmax)
        # A LUT lookup is per-bead by construction, whatever `colors` was: the
        # vertex verdict does not describe this array. Inert while colormap LUTs
        # are RGB-only, but an RGBA LUT would otherwise re-open exactly the
        # per-element-alpha door the vertex classification closes.
        uniform_colors = False
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
        # The beads inherit the vertices' uniformity: a uniform colour
        # interpolates to itself, so a 4th (alpha) column is still the safe
        # uniform case and must not be read as per-element RGBA.
        _uniform_colors=uniform_colors,
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


#: Rows per chunk in :func:`_cap_aspect`'s eigendecomposition loop — bounds the
#: float64 ``(chunk, d, d)`` transients (a lifted level can reach ~2M splats).
_CAP_ASPECT_CHUNK: int = 1_000_000


def _cap_aspect(
    level: GSplatData,
    coarsen_dims: Optional[Sequence[int]],
    max_aspect: float,
) -> GSplatData:
    """Cap per-splat anisotropy (mass-preservingly) on the coarsened dims.

    The lift is *strictly isotropic* (module docstring): ``sigmaRay ==
    sigma_world`` only for isotropic covariances, so an anisotropic splat's
    ray integral — hence its rendered brightness — is view-dependent (an
    elongated Gaussian is ``sigma_max/sigma_perp`` brighter end-on than
    broadside). The substitutive merge of a 1D bead string violates that
    invariant: Morton bins chunk the string, so representatives elongate
    ~K× more per level, and the resulting per-splat, per-orientation flares
    read as haphazard brightness/hue pops between LOD levels. This helper
    restores the invariant up to ``max_aspect``: each splat's covariance is
    eigendecomposed on the ``coarsen_dims`` submatrix and the small axes are
    fattened so ``sigma_i >= sigma_max / max_aspect``, with the amplitude
    rescaled by ``det_old/det_new`` so the splat's integral (its additive
    X-ray light) is exactly unchanged — coarse levels become fatter, softer,
    proportionally dimmer tubes (image-pyramid semantics) instead of flarey
    elongated shards.

    Barrier dims (not in ``coarsen_dims``) are left bitwise untouched: they
    carry near-delta widths on sliced nD data, and fattening them would bleed
    geometry across slices. The block edit is exact because barrier↔coarsen
    cross-covariances are zero on the lifted path (beads have diagonal
    isotropic covariances and the grouped reduction keeps barrier coordinates
    constant per bin, so neither the intra nor inter moment term develops
    cross entries). This helper is private to the lift path on purpose:
    anisotropic representatives are *correct* for fitted volumetric gsplats.

    ``coarsen_dims=None`` means all dims. Splats whose capped covariance
    fails to re-factorize even after a proportional ridge are left uncapped
    (never crash the build).
    """
    flat = level.flattened()
    n = int(flat.n_splats)
    d = int(flat.ndim)
    if n == 0 or max_aspect is None:
        return level
    tau = float(max_aspect)
    if tau < 1.0:
        raise ValueError(f"max_aspect must be >= 1 (or None to disable); got {tau}")

    cd = (
        tuple(range(d))
        if coarsen_dims is None
        else tuple(sorted({int(i) for i in coarsen_dims}))
    )
    if len(cd) <= 1:
        return level  # 1x1 submatrix: aspect is identically 1

    from .utils.trils import pack_tril, unpack_tril

    chol = np.asarray(flat.cholesky_factors, dtype=np.float32)
    amps = np.asarray(flat.amplitudes, dtype=np.float64).copy()
    out_chol = chol.copy()
    cd_idx = np.asarray(cd, dtype=np.intp)
    changed_any = False

    for start in range(0, n, _CAP_ASPECT_CHUNK):
        sl = slice(start, min(n, start + _CAP_ASPECT_CHUNK))
        L = unpack_tril(chol[sl].astype(np.float64), d)  # (m, d, d)
        Sig = L @ np.swapaxes(L, 1, 2)
        sub = Sig[:, cd_idx][:, :, cd_idx]  # (m, c, c)
        ev, evec = np.linalg.eigh(sub)  # ascending eigenvalues
        s = np.sqrt(np.maximum(ev, 0.0))
        s_new = np.maximum(s, s[:, -1:] / tau)
        needs = np.any(s_new > s * (1.0 + 1e-12), axis=1)
        if not np.any(needs):
            continue
        changed_any = True
        idx = np.nonzero(needs)[0]
        sub_new = (evec[idx] * (s_new[idx] ** 2)[:, None, :]) @ np.swapaxes(
            evec[idx], 1, 2
        )
        Sig_new = Sig[idx]
        Sig_new[:, cd_idx[:, None], cd_idx[None, :]] = sub_new

        # Re-factorize; on failure add a proportional ridge on the coarsened
        # diagonal and retry once; still-failing splats stay uncapped.
        ok = np.ones(idx.shape[0], dtype=bool)
        L_new = np.empty_like(Sig_new)
        try:
            L_new = np.linalg.cholesky(Sig_new)
        except np.linalg.LinAlgError:
            for j in range(idx.shape[0]):
                try:
                    L_new[j] = np.linalg.cholesky(Sig_new[j])
                except np.linalg.LinAlgError:
                    ridge = np.zeros(d, dtype=np.float64)
                    ridge[cd_idx] = 1e-9 * np.diagonal(Sig_new[j])[cd_idx]
                    try:
                        L_new[j] = np.linalg.cholesky(Sig_new[j] + np.diag(ridge))
                    except np.linalg.LinAlgError:
                        ok[j] = False
        keep = np.nonzero(ok)[0]
        if keep.size == 0:
            continue
        rows = np.asarray(sl.indices(n)[0] + idx[keep], dtype=np.intp)
        # Mass preservation: a' = a * det_old/det_new. The submatrix det ratio
        # equals the full-matrix one (barrier block + zero cross terms are
        # untouched), so both render_light and the sliced per-barrier mass
        # are conserved exactly.
        det_old = np.prod(s[idx[keep]], axis=1)
        det_new = np.prod(s_new[idx[keep]], axis=1)
        amps[rows] *= det_old / np.maximum(det_new, 1e-300)
        out_chol[rows] = pack_tril(L_new[keep]).astype(np.float32)

    if not changed_any:
        return level
    return GSplatData(
        centers=np.asarray(flat.centers, dtype=np.float32),
        amplitudes=amps.astype(np.float32),
        cholesky_factors=out_chol,
        colors=(None if flat.colors is None else np.asarray(flat.colors, np.float32)),
        truncation_radius=float(flat.truncation_radius),
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
    max_aspect: Optional[float] = 3.0,
    quality_stamps: bool = True,
) -> "List[GSplatData]":
    """Coarse substitutive levels of a lifted point cloud (render-light conserved).

    Runs :func:`luxar.gsplats.lod.substitutive.make_substitutive_lod` on
    ``lifted`` with **per-bin mass-preserving amplitudes** (``amplitude="mass"``
    — every bin's merged splat carries exactly its members' summed
    ``a·|det L|`` mass, so per-channel colored light is conserved bin-by-bin
    together with the mass-weighted mean colors), **drops level 0** (the 1:1
    lifted set — the original Points/Lines node is the finest LOD level, so a
    1:1 gsplat copy would double-render at the seam), **caps each level's
    per-splat anisotropy** at ``max_aspect`` (see :func:`_cap_aspect` — the
    merge would otherwise elongate representatives level over level, whose
    view-dependent ray integrals read as haphazard brightness/hue pops between
    LOD levels; ``None`` disables), and finally **rescales each level's
    amplitudes** so its :func:`render_light` equals the finest (lifted)
    level's (a near-no-op safety net under mass amplitudes; it also absorbs
    the cull of zero-amplitude bins).

    Returns the coarse levels **finest → coarsest** (substitutive index 1..L),
    each a flat :class:`GSplatData` — one entry per synthesised coarser level.
    When ``quality_stamps`` is true and the reference is non-empty, every level
    carries its measured value in ``stats["quality"]``. Empty only if the pyramid
    collapsed to level 0 alone; for degenerate input the coarsest entry (``[-1]``)
    may itself have ``n_splats == 0`` (callers treat both as "no usable coarse
    levels" — see the adders' degenerate guards).
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
        amplitude="mass",
    )
    light0 = render_light(pyramid.at_substitutive(0))
    out: List[GSplatData] = []
    for s in range(1, pyramid.n_substitutive):
        lvl = pyramid.at_substitutive(s).flattened()
        if max_aspect is not None:
            lvl = _cap_aspect(lvl, coarsen_dims, float(max_aspect)).flattened()
        ls = render_light(lvl)
        scale = (light0 / ls) if ls > 0 else 1.0
        final_level = GSplatData(
            centers=np.asarray(lvl.centers, dtype=np.float32),
            amplitudes=(np.asarray(lvl.amplitudes, dtype=np.float64) * scale).astype(
                np.float32
            ),
            cholesky_factors=np.asarray(lvl.cholesky_factors, dtype=np.float32),
            colors=(None if lvl.colors is None else np.asarray(lvl.colors, np.float32)),
            truncation_radius=float(lifted.truncation_radius),
        )
        if quality_stamps and lifted.n_splats > 0:
            from .lod.quality import mixture_quality

            final_level.stats["quality"] = mixture_quality(
                final_level, lifted, device=device
            ).quality
        out.append(final_level)
    return out


def render_light(data: GSplatData) -> float:
    """Total emitted light of a (flattened) gsplat set under sum/linear projection.

    For an additive (linear-sum) projection the screen-integrated intensity of one
    isotropic splat is proportional to ``a * sigma^3`` (amplitude * 3D volume
    integral, ``D = 3`` convention), so the conserved "no zoom-dimming" quantity is
    ``sum_i a_i * sigma_geo_i^D`` with ``sigma_geo = det(L)^(1/D)`` the
    geometric-mean sigma. Used by :func:`coarse_substitutive_levels` to rescale
    each coarse level's amplitudes so total light is conserved across LOD levels
    (a near-no-op safety net now that the lift path merges with per-bin
    mass-preserving amplitudes; it still absorbs cull/ridge/float32 residue).

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
