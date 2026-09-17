"""luxar.core.group.lod.points – Additive-LOD helpers for the Points leaf type.

Provides the ordering + ladder-construction primitives that the
``add_points(..., additive_lod=...)`` convenience kwarg consumes — one
of three peer implementations of the same additive-LOD pattern, alongside
``core/group/lod/lines.py`` (per-polyline) and ``gsplats/lod/additive.py``
(Gaussian-energy ordering).

Five ordering methods, all geometry-agnostic-ish:

* ``random``     — uniform-random permutation (with optional seed).
* ``salience``   — sort by radii descending (largest points first).
* ``spatial-uniform`` — stratified-grid sampling
  (:func:`luxar.core.group.lod.spatial_uniform.stratified_grid_order`).
* ``poisson-disk`` — blue-noise sampling
  (:func:`luxar.core.group.lod.poisson_disk.poisson_disk_order`).
* ``radial``     — concentric shells around the node's own bbox centre, so a
  streaming prefix grows outward from the middle (the reveal). The only
  ascending sort, and the only one whose ladder carries no energy stamps —
  see :func:`luxar.core.group.lod.reveal.radial_element_score`.

The breakpoints API mirrors the gsplats one in vocabulary but without
the gsplats-only ``energy:`` variant:

* ``n_lods: int`` — equal-count split into this many LODs.
* ``counts: list[int]`` — cumulative element-count breakpoints (sized for
  HTTP-range-streaming the coarsest levels first).
* ``counts: "equi-energy:<n>"`` — ``n`` rungs at equal shares of cumulative
  perceptual energy (``luminance × radius³``) along the ordering, commit-capped;
  pair with ``method="salience", salience_kind="energy"`` for a heaviest-first
  ladder whose first rung is small and whose late rungs are fat.

If both are passed, ``counts`` wins. If the dataset has fewer elements
than the requested ``n_lods``, the writer emits ``min(n_lods, n)``
non-empty levels and silently drops the trailing (empty) levels.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .group import (
    ADDITIVE_METHODS,
    DEFAULT_ADDITIVE_METHOD,
    DEFAULT_ADDITIVE_N_LODS,
    resolve_additive_axis,
)
from .poisson_disk import poisson_disk_order
from .reveal import radial_element_score
from .spatial_uniform import stratified_grid_order

#: Ordering methods supported on Points additive LOD.
PointsMethodName = Literal[
    "random", "salience", "spatial-uniform", "poisson-disk", "radial"
]


# Default for ``additive_lod=True`` and ``additive_lod=dict()``. Aliases of the
# shared resolver's constants so the kwarg defaults below and
# ``resolve_additive_axis_points`` always agree.
DEFAULT_N_LODS: int = DEFAULT_ADDITIVE_N_LODS
DEFAULT_METHOD: PointsMethodName = DEFAULT_ADDITIVE_METHOD

# Defaults for ``substitutive_lod=True`` / ``substitutive_lod=dict()`` keep the
# legacy synthesised-gsplat path. Points owns the optional ``coarse="points"``
# vocabulary; Lines deliberately still delegates only to the shared lift
# resolver until its same-type arm lands.


def _resolve_points_representation(kwargs: dict) -> tuple[str, Union[str, float]]:
    """Pop and validate the Points-only substitutive representation keys."""
    from .group import resolve_same_type_representation

    return resolve_same_type_representation(
        kwargs,
        geometry="Points",
        same_type="points",
        inapplicable_reasons={
            "truncation_radius": (
                "it controls the Gaussian footprint used by the lift, and "
                "same-type point levels are not lifted"
            ),
            "max_aspect": (
                "it caps anisotropy on merged Gaussian levels, and same-type "
                "point levels contain no Gaussians"
            ),
            "method": (
                "it selects the Gaussian clustering algorithm, and same-type "
                "point levels are spatially stratified instead"
            ),
            "device": (
                "it selects where Gaussian clustering runs, and same-type "
                "point levels use the CPU point sampler"
            ),
            "coarsen_dims": (
                "it selects Gaussian merge dimensions, and same-type point "
                "levels preserve discrete hidden coordinates automatically"
            ),
        },
    )


def resolve_substitutive_axis_points(spec: Any) -> Optional[dict]:
    """Translate the ``substitutive_lod=`` kwarg value into a normalized dict.

    The default coarsens by **synthesising gsplats**. ``coarse="points"`` instead
    writes spatially stratified point subsamples, with explicit blending-aware brightness
    compensation, while keeping the finest level as the original Points node.

    The lift vocabulary remains delegated to the shared Points/Lines resolver;
    the same-type representation keys are parsed here so they do not become
    accidentally valid for Lines.
    """
    from .group import resolve_substitutive_axis

    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        return resolve_substitutive_axis(spec, "Points")

    kwargs = dict(spec)
    coarse, brightness = _resolve_points_representation(kwargs)
    resolved = resolve_substitutive_axis(
        kwargs,
        "Points",
        extra_valid_keys=("coarse", "brightness_compensation"),
    )
    assert resolved is not None
    resolved["coarse"] = coarse
    resolved["brightness_compensation"] = brightness
    return resolved


# ─────────────────────────────────────────────────────────────────────
# Ordering — per-element permutation
# ─────────────────────────────────────────────────────────────────────


def compute_additive_order_points(
    positions: NDArray,
    radii: Optional[NDArray] = None,
    method: PointsMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    seed: Optional[int] = None,
    reveal_center: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> Tuple[NDArray[np.intp], List[int]]:
    """Compute an additive ordering permutation over Points.

    Args:
        positions: ``(N, d)`` array. ``d >= 3`` for ``spatial-uniform``;
            other methods don't care about ``d``.
        radii: ``(N,)`` array or ``None``. Required for ``salience``.
        method: One of ``random`` / ``salience`` / ``spatial-uniform`` /
            ``poisson-disk`` / ``radial``.
        n_lods: Only consulted by ``spatial-uniform`` / ``poisson-disk`` for
            deciding how many grid levels to iterate. ``random``, ``salience``
            and ``radial`` return a single permutation; the slicing into LOD
            levels happens later in :func:`make_additive_lod_points`.
        seed: For ``random``; ignored by others.
        reveal_center: ``radial`` only — centre of the shells, defaulting to the
            spatial bounding-box centre (NOT the scene origin, so a dataset far
            from the origin still grows from its own middle). One coordinate per
            spatial axis.
        spatial_dims: ``radial`` only — position columns the distance is measured
            over, defaulting to the columns with non-zero extent. That drops a
            *constant* time/channel column but not a *stacked* one (it varies
            like a spatial axis), so pass it explicitly for stacked data — or go
            through ``add_points``, which fills it from the scene's displayed
            dims (:func:`~luxar.core.group.lod.reveal.resolve_reveal_spatial_dims`).

    Returns:
        ``(permutation, per_level_counts)`` — same shape as
        :func:`stratified_grid_order`. For ``random`` and ``salience``,
        ``per_level_counts`` is an empty list (caller decides slicing).
        For ``spatial-uniform``, ``per_level_counts`` describes the
        natural per-LOD partition.
    """
    n = positions.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.intp), []

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(n).astype(np.intp), []

    if method == "salience":
        if radii is None:
            raise ValueError("salience ordering requires per-element radii; got None")
        if radii.shape[0] != n:
            raise ValueError(
                f"radii length ({radii.shape[0]}) must match positions length ({n})"
            )
        # Sort by radii descending; stable so identical scores keep
        # their original index order (deterministic).
        score = np.asarray(radii, dtype=np.float64)
        perm = np.argsort(-score, kind="stable").astype(np.intp)
        return perm, []

    if method in ("spatial-uniform", "poisson-disk"):
        # One branch for both samplers, sharing the d >= 3 guard their grids
        # need — mirroring the Lines equivalent, which has always been shaped
        # this way. (Two branches with a copy-pasted guard each is what pushed
        # this function past the C901 ratchet when `radial` was added.)
        if positions.shape[1] < 3:
            raise ValueError(
                f"{method} ordering needs positions with d >= 3; "
                f"got shape {positions.shape}"
            )
        if method == "poisson-disk":
            return poisson_disk_order(positions, n_lods, seed=seed or 0)
        return stratified_grid_order(positions, n_lods)

    if method == "radial":
        # ASCENDING, unlike `salience` above: the score is a DISTANCE, so the
        # nearest is revealed first and the prefixes grow outward as concentric
        # shells. Returns an EMPTY natural partition, deliberately — unlike
        # `spatial-uniform` / `poisson-disk`, whose per-level counts make
        # `make_additive_lod_points` bypass the `counts:` / `stream:` / `energy:`
        # breakpoint vocabularies entirely. A reveal must stay compatible with
        # those, so the caller does the slicing.
        return (
            np.argsort(
                radial_element_score(positions, reveal_center, spatial_dims),
                kind="stable",
            ).astype(np.intp),
            [],
        )

    # Unreachable for a well-typed caller — the branches above are exhaustive
    # over PointsMethodName — but `method` arrives as a plain string from the
    # resolver and from user code, so the runtime guard stays.
    raise ValueError(
        f"method must be one of {' / '.join(repr(m) for m in ADDITIVE_METHODS)}; "
        f"got {method!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# Public LOD-ladder construction
# ─────────────────────────────────────────────────────────────────────


def _perceptual_luminance(colors: NDArray) -> NDArray[np.float64]:
    """Rec.709 perceptual luminance from RGB (per row).

    Accepts ``(N, 3)`` or ``(N, 4)`` color arrays. Values can be in any
    range — the returned luminance is in the same scale, since the
    weighted-sum is linear. Returns ``(N,)`` float64.
    """
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError(f"colors must be (N, 3) or (N, 4); got shape {colors.shape}")
    c = colors[:, :3].astype(np.float64, copy=False)
    return 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]


def compute_points_energy(
    n: int,
    radii: Optional[NDArray],
    colors: Optional[NDArray],
    scalars: Optional[NDArray],
) -> NDArray[np.float64]:
    """Per-point energy score: ``luminance × volume`` ≈ perceptual mass.

    - **Luminance**: from ``colors`` via Rec.709; from ``scalars`` if
      colors are absent; constant 1.0 if neither.
    - **Volume**: ``radius**3`` (sphere volume up to a constant).
      ``radii=None`` → constant radius 1.

    Returns ``(N,)`` non-negative float64.
    """
    if radii is None:
        radii_arr = np.ones(n, dtype=np.float64)
    else:
        radii_arr = np.asarray(radii, dtype=np.float64).reshape(-1)
        if radii_arr.size != n:
            raise ValueError(f"radii has {radii_arr.size} entries but expected {n}")
    # Squared then cubed via multiply to avoid abs / pow rounding.
    volume = radii_arr * radii_arr * radii_arr

    if colors is not None:
        lum = _perceptual_luminance(np.asarray(colors))
    elif scalars is not None:
        lum = np.asarray(scalars, dtype=np.float64).reshape(-1)
        if lum.size != n:
            raise ValueError(f"scalars has {lum.size} entries but expected {n}")
    else:
        lum = np.ones(n, dtype=np.float64)

    # Non-negative — luminance may be slightly negative on HDR overshoot
    # or signed scalars; clip to keep cumulative-sum monotone.
    return np.clip(lum, 0.0, None) * np.clip(volume, 0.0, None)


def _energy_breakpoints_to_counts(
    energy: NDArray[np.float64],
    perm: NDArray[np.intp],
    fractions: List[float],
) -> List[int]:
    """Convert cumulative energy fractions to cumulative element counts.

    Walks the permutation in order, accumulating energy. Returns the
    index at which the cumulative energy first crosses each fraction's
    threshold. Used by both Points and Lines so the cut shape is
    consistent across types.
    """
    if not fractions:
        raise ValueError("energy: fractions must be non-empty")
    total = float(energy.sum())
    if total <= 0:
        # Degenerate (zero radii, zero luminance, etc.) — fall back to
        # equal-count cuts so the user still gets a useful ladder.
        n = perm.size
        return [min(n, int(round(f * n))) for f in sorted(fractions) if 0 < f < 1]
    cumulative = np.cumsum(energy[perm])
    counts: List[int] = []
    sorted_fracs = sorted(fractions)
    cursor = 0
    for f in sorted_fracs:
        if f <= 0:
            continue
        if f >= 1:
            counts.append(perm.size)
            break
        threshold = f * total
        # First index where cumulative meets the fraction.
        idx = int(np.searchsorted(cumulative, threshold, side="left")) + 1
        idx = max(cursor + 1, min(idx, perm.size))
        counts.append(idx)
        cursor = idx
    return counts


def _parse_breakpoints_spec(
    spec: Any,
    total: int,
    energy: Optional[NDArray[np.float64]] = None,
    perm: Optional[NDArray[np.intp]] = None,
) -> Optional[List[int]]:
    """Parse the ``counts``/``breakpoints`` kwarg into cumulative integer counts.

    Vocabulary:

    - ``None`` → caller falls back to equal-count split.
    - ``List[int]`` → cumulative counts (legacy ``counts=`` shape).
    - ``"energy:0.5,0.9,0.99,1.0"`` → cumulative energy fractions. Needs
      both ``energy`` and ``perm`` to be supplied by the caller.
    - ``"equi-energy:4"`` → 4 rungs at EQUAL shares of cumulative energy along
      ``perm`` (few heavy elements first, fatter rungs later), commit-capped
      (:func:`luxar.utils.lod_breakpoints.equi_energy_cuts`). Needs ``energy``
      and ``perm`` too.
    - ``"stream:40000"`` → bandwidth-derived geometric ladder
      ``[c, 2c, 4c, …, total]``, resolved against THIS ``total`` so one spec
      adapts to every level/part. Identical cut geometry to the GSplats ladder
      (shared :mod:`luxar.utils.lod_breakpoints` helper).

    Returns the breakpoint list (cumulative counts, strictly increasing,
    ``<= total``), or ``None`` for the no-op default.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec.startswith("stream:"):
            from ....utils.lod_breakpoints import parse_stream_chunk, stream_cuts

            return stream_cuts(total, parse_stream_chunk(spec))
        if spec.startswith("equi-energy:"):
            from ....utils.lod_breakpoints import (
                equi_energy_cuts,
                parse_equi_energy_rungs,
            )

            if energy is None or perm is None:
                raise ValueError(
                    "equi-energy: breakpoints require both energy and perm; "
                    "internal error"
                )
            return equi_energy_cuts(
                np.asarray(energy, dtype=np.float64)[perm],
                parse_equi_energy_rungs(spec),
            )
        if not spec.startswith("energy:"):
            raise ValueError(
                f"unrecognized breakpoints string {spec!r}; expected "
                "'energy:<fractions>' (e.g. 'energy:0.5,0.9,1.0'), "
                "'equi-energy:<n>' (e.g. 'equi-energy:4') or "
                "'stream:<c>' (e.g. 'stream:40000')"
            )
        if energy is None or perm is None:
            raise ValueError(
                "energy: breakpoints require both energy and perm; internal error"
            )
        fracs = [float(s) for s in spec[len("energy:") :].split(",") if s.strip()]
        return _energy_breakpoints_to_counts(energy, perm, fracs)
    return _validate_counts(list(spec), total)


def make_additive_lod_points(
    positions: NDArray,
    radii: Optional[NDArray] = None,
    *,
    method: PointsMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    counts: Optional[Any] = None,
    seed: Optional[int] = None,
    colors: Optional[NDArray] = None,
    scalars: Optional[NDArray] = None,
    salience_kind: Literal["size", "energy"] = "size",
    reveal_center: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> List[NDArray[np.intp]]:
    """Compute per-LOD-level index arrays for Points.

    Each entry in the returned list is a 1-D array of indices into the
    original ``positions``: concatenating them in order gives the full
    permutation. The caller (``_write_points_multi_lod`` in
    ``io/compiler.py``) slices the data arrays accordingly.

    Args:
        positions: ``(N, d)`` array.
        radii: ``(N,)`` array; required for ``salience`` (and for
            ``energy:`` breakpoints / ``salience_kind='energy'``).
        method: Ordering method.
        n_lods: Target level count (default 4). Ignored if ``counts``
            is provided.
        counts: Cumulative-element-count breakpoints. Vocabulary:

            * ``None`` → equal-count splits (default).
            * ``List[int]`` → cumulative counts
              (e.g. ``[1500, 8000, 40000]``).
            * ``"energy:0.5,0.9,0.99,1.0"`` → cumulative perceptual-
              energy fractions. The energy of element *i* is
              ``luminance_i × radius_i**3``; fractions partition the
              total energy of the ordering. Requires ``colors`` or
              ``scalars`` for the luminance term; ``radii`` for volume.
            * ``"stream:40000"`` → bandwidth-derived geometric ladder
              ``[c, 2c, 4c, …, N]``: first paint costs ``c`` elements,
              then each refinement doubles. Resolved against the actual
              N, so one spec adapts to every level of a tree. This is
              the shape that makes a large leaf paint progressively —
              prefer it over ``n_lods`` on anything big, since an
              equal-count split still ends in an N/n_lods-sized commit.
            * ``"equi-energy:4"`` → 4 rungs at EQUAL shares of cumulative
              perceptual energy along the ordering, any increment above
              the shared commit cap split into capped steps. Under an
              energy-first ordering the first rung is few heavy elements
              and the late rungs fat — fast first paint, slow rungs that
              matter least. Uses the same energy as ``energy:``.
        seed: For ``random``; ignored otherwise.
        colors: Optional ``(N, 3)`` or ``(N, 4)`` per-point colors —
            used for the luminance term of ``salience_kind='energy'``
            and ``"energy:"`` breakpoints. If absent, ``scalars`` is
            used; if both are absent, luminance defaults to 1.
        scalars: Optional ``(N,)`` per-point scalar — fallback
            luminance when ``colors`` are absent.
        salience_kind: For ``method='salience'``. ``'size'`` (default)
            sorts by radius alone (legacy). ``'energy'`` sorts by
            ``luminance × radius**3`` — the same per-element score the
            ``energy:`` breakpoints accumulate against.
        reveal_center: For ``method='radial'`` — centre of the concentric
            shells, defaulting to the spatial bounding-box centre.
        spatial_dims: For ``method='radial'`` — the position columns the
            shell distance is measured over, defaulting to the columns with
            non-zero extent (which excludes a *constant* time/channel column,
            but not a *stacked* one — see
            :func:`~luxar.core.group.lod.reveal.radial_element_score`).

    Returns:
        List of per-level index arrays, length
        ``min(n_lods, n_elements)`` for the non-empty-tail case
        (trailing empty levels are dropped silently).
    """
    n = positions.shape[0]
    if n == 0:
        return []

    # Salience-energy: compute the per-element energy score once and
    # use it as both the permutation key and the source for energy:
    # breakpoint resolution below.
    energy: Optional[NDArray[np.float64]] = None
    if method == "salience" and salience_kind == "energy":
        energy = compute_points_energy(n, radii, colors, scalars)
        perm = np.argsort(-energy, kind="stable").astype(np.intp)
        natural_counts: List[int] = []
    else:
        perm, natural_counts = compute_additive_order_points(
            positions,
            radii=radii,
            method=method,
            n_lods=n_lods,
            seed=seed,
            reveal_center=reveal_center,
            spatial_dims=spatial_dims,
        )

    if method in ("spatial-uniform", "poisson-disk"):
        # Both samplers already partition the permutation into per-
        # level groups (coarse-to-fine). Respect that partition rather
        # than re-slicing — it's what makes the cumulative density
        # approximately uniform.
        out: List[NDArray[np.intp]] = []
        cursor = 0
        for count in natural_counts:
            if count > 0:
                out.append(perm[cursor : cursor + count])
            cursor += count
        return out

    # For random / salience, slice the permutation by breakpoints.
    if (
        isinstance(counts, str)
        and counts.startswith(("energy:", "equi-energy:"))
        and energy is None
    ):
        # Compute energy on demand for energy-based breakpoints under any
        # ordering method.
        energy = compute_points_energy(n, radii, colors, scalars)

    breakpoints = _parse_breakpoints_spec(counts, n, energy=energy, perm=perm)
    if breakpoints is None:
        # Equal-count split into n_lods levels.
        per_level = max(1, (n + n_lods - 1) // n_lods)  # ceil-divide
        breakpoints = [min(n, (i + 1) * per_level) for i in range(n_lods - 1)]

    # Slice the permutation. Trailing empty levels collapse silently.
    out = []
    start = 0
    for bp in breakpoints:
        if bp > start:
            out.append(perm[start:bp])
            start = bp
    if start < n:
        out.append(perm[start:])
    return out


def _validate_counts(counts: List[int], n: int) -> List[int]:
    """Normalize a user-supplied ``counts`` breakpoint list.

    Returns the breakpoint list clamped to ``<= n`` and strictly
    increasing. The final breakpoint at ``n`` is implicit (handled by
    the slicer's tail).
    """
    if not counts:
        raise ValueError("counts must be a non-empty list of integers")
    breakpoints: List[int] = []
    prev = 0
    for c in counts:
        c = int(c)
        if c <= prev:
            raise ValueError(
                f"counts must be strictly increasing cumulative breakpoints; "
                f"got {counts!r}"
            )
        clamped = min(c, n)
        if clamped > prev:
            breakpoints.append(clamped)
            prev = clamped
        if clamped == n:
            break
    return breakpoints


# ─────────────────────────────────────────────────────────────────────
# Convenience-kwarg resolver
# ─────────────────────────────────────────────────────────────────────


def resolve_additive_axis_points(
    spec: Any,
) -> Optional[dict]:
    """Translate the ``additive_lod=`` kwarg value into a normalized dict.

    Vocabulary (same shape as the GSplats peer
    :func:`luxar.core.group.lod.gsplats.resolve_additive_axis_gsplats`,
    but with Points semantics):

    * ``None``    → no multi-LOD ladder; caller writes a flat node.
    * ``True``    → use defaults (``DEFAULT_N_LODS`` levels, ``DEFAULT_METHOD``).
    * ``False``   → no-op (treat as None — there's no stored multi-LOD
      source state on Points the way gsplats has multi-substitutive).
    * ``dict()``  → use defaults.
    * ``dict(...)`` → user-supplied parameters (``method``, ``n_lods``,
      ``counts``, ``seed``). Unrecognized keys raise.
    * ``dict(recompute=True)`` → tolerated for API symmetry with gsplats,
      ignored on Points (no stored ladder to recompute against).

    Returns ``None`` for no-op, or a dict with normalized keys
    ``method`` / ``n_lods`` / ``counts`` / ``seed``.

    Thin wrapper over the shared
    :func:`luxar.core.group.lod.group.resolve_additive_axis` (one
    implementation shared with Lines so the two can't drift).
    """
    return resolve_additive_axis(spec, "Points")


# Tolerate the export name used by gsplats's resolver convention.
PointsAdditiveSpec = Union[None, bool, dict]
