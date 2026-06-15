"""luxar.core.group.lod.points – Additive-LOD helpers for the Points leaf type.

Provides the ordering + ladder-construction primitives that the
``add_points(..., additive_lod=...)`` convenience kwarg consumes — one
of three peer implementations of the same additive-LOD pattern, alongside
``core/group/lod/lines.py`` (per-polyline) and ``gsplats/lod/additive.py``
(Gaussian-energy ordering).

Three ordering methods, all geometry-agnostic-ish:

* ``random``     — uniform-random permutation (with optional seed).
* ``salience``   — sort by radii descending (largest points first).
* ``spatial-uniform`` — stratified-grid sampling
  (:func:`luxar.core.group.lod.spatial_uniform.stratified_grid_order`).

The breakpoints API mirrors the gsplats one in vocabulary but without
the gsplats-only ``energy:`` variant:

* ``n_lods: int`` — equal-count split into this many LODs.
* ``counts: list[int]`` — cumulative element-count breakpoints (the
  shape PR #301 used for HTTP-range-streaming-sized gsplats demos).

If both are passed, ``counts`` wins. If the dataset has fewer elements
than the requested ``n_lods``, the writer emits ``min(n_lods, n)``
non-empty levels and the trailing levels are dropped (silent — matches
the user's preference in the plan).
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .poisson_disk import poisson_disk_order
from .spatial_uniform import stratified_grid_order

#: Ordering methods supported on Points additive LOD.
PointsMethodName = Literal["random", "salience", "spatial-uniform", "poisson-disk"]


# Default for ``additive_lod=True`` and ``additive_lod=dict()``.
DEFAULT_N_LODS: int = 4
DEFAULT_METHOD: PointsMethodName = "random"

# Defaults for ``substitutive_lod=True`` / ``substitutive_lod=dict()`` — the
# coarse levels are synthesised gsplats (each point lifted to an isotropic
# Gaussian, then reduced by the gsplat substitutive pipeline). The substitutive
# vocabulary/defaults are shared with Lines — see
# :func:`luxar.core.group.lod.group.resolve_substitutive_axis`.


def resolve_substitutive_axis_points(spec: Any) -> Optional[dict]:
    """Translate the ``substitutive_lod=`` kwarg value into a normalized dict.

    The substitutive axis coarsens a point cloud by **synthesising gsplats**:
    each point is lifted to an isotropic Gaussian and the gsplat substitutive
    pipeline builds fewer-but-larger representative levels (mass-preserving),
    which become the coarse levels of a points LOD ladder (the finest level
    stays the original Points node).

    Thin wrapper over the shared
    :func:`luxar.core.group.lod.group.resolve_substitutive_axis` (one
    implementation shared with Lines so the two can't drift). See it for the
    full value vocabulary.
    """
    from .group import resolve_substitutive_axis

    return resolve_substitutive_axis(spec, "Points")


# ─────────────────────────────────────────────────────────────────────
# Ordering — per-element permutation
# ─────────────────────────────────────────────────────────────────────


def compute_additive_order_points(
    positions: NDArray,
    radii: Optional[NDArray] = None,
    method: PointsMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.intp], List[int]]:
    """Compute an additive ordering permutation over Points.

    Args:
        positions: ``(N, d)`` array. ``d >= 3`` for ``spatial-uniform``;
            other methods don't care about ``d``.
        radii: ``(N,)`` array or ``None``. Required for ``salience``.
        method: One of ``random`` / ``salience`` / ``spatial-uniform``.
        n_lods: Only consulted by ``spatial-uniform`` for deciding how
            many grid levels to iterate. ``random`` and ``salience``
            return a single permutation; the slicing into LOD levels
            happens later in :func:`make_additive_lod_points`.
        seed: For ``random``; ignored by others.

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

    if method == "spatial-uniform":
        if positions.shape[1] < 3:
            raise ValueError(
                "spatial-uniform ordering needs positions with d >= 3; "
                f"got shape {positions.shape}"
            )
        return stratified_grid_order(positions, n_lods)

    if method == "poisson-disk":
        if positions.shape[1] < 3:
            raise ValueError(
                "poisson-disk ordering needs positions with d >= 3; "
                f"got shape {positions.shape}"
            )
        return poisson_disk_order(positions, n_lods, seed=seed or 0)

    raise ValueError(
        "method must be one of 'random' / 'salience' / 'spatial-uniform' / "
        f"'poisson-disk'; got {method!r}"
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


def _compute_points_energy(
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

    Returns the breakpoint list (cumulative counts, strictly increasing,
    ``<= total``), or ``None`` for the no-op default.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if not spec.startswith("energy:"):
            raise ValueError(
                f"unrecognized breakpoints string {spec!r}; expected "
                "'energy:<fractions>' (e.g. 'energy:0.5,0.9,1.0')"
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
        energy = _compute_points_energy(n, radii, colors, scalars)
        perm = np.argsort(-energy, kind="stable").astype(np.intp)
        natural_counts: List[int] = []
    else:
        perm, natural_counts = compute_additive_order_points(
            positions, radii=radii, method=method, n_lods=n_lods, seed=seed
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
    if isinstance(counts, str) and counts.startswith("energy:") and energy is None:
        # Compute energy on demand for energy: breakpoints under any
        # ordering method.
        energy = _compute_points_energy(n, radii, colors, scalars)

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
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        return {
            "method": DEFAULT_METHOD,
            "n_lods": DEFAULT_N_LODS,
            "counts": None,
            "seed": None,
            "salience_kind": "size",
        }
    if not isinstance(spec, dict):
        raise TypeError(
            f"additive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)
    # ``recompute`` is gsplats-only; tolerate but don't act on it.
    kwargs.pop("recompute", None)

    method = kwargs.pop("method", DEFAULT_METHOD)
    if method not in (
        "random",
        "salience",
        "spatial-uniform",
        "poisson-disk",
    ):
        raise ValueError(
            "method must be one of 'random' / 'salience' / 'spatial-uniform' "
            f"/ 'poisson-disk'; got {method!r}"
        )

    n_lods = int(kwargs.pop("n_lods", DEFAULT_N_LODS))
    if n_lods < 1:
        raise ValueError(f"n_lods must be >= 1, got {n_lods}")

    # Accept ``breakpoints`` as an alias for ``counts`` (the energy:
    # vocabulary reads more naturally as "breakpoints") — but only one.
    counts = kwargs.pop("counts", None)
    breakpoints = kwargs.pop("breakpoints", None)
    if counts is not None and breakpoints is not None:
        raise ValueError(
            "additive_lod: pass either 'counts' OR 'breakpoints', not both"
        )
    if breakpoints is not None:
        counts = breakpoints
    if counts is not None and not isinstance(counts, str):
        counts = [int(c) for c in counts]

    seed = kwargs.pop("seed", None)
    if seed is not None:
        seed = int(seed)

    salience_kind = kwargs.pop("salience_kind", "size")
    if salience_kind not in ("size", "energy"):
        raise ValueError(
            f"salience_kind must be 'size' or 'energy'; got {salience_kind!r}"
        )

    if kwargs:
        raise ValueError(
            f"additive_lod for Points: unrecognized keys "
            f"{sorted(kwargs)}. Valid keys: method, n_lods, counts, "
            f"breakpoints, seed, salience_kind, recompute."
        )

    return {
        "method": method,
        "n_lods": n_lods,
        "counts": counts,
        "seed": seed,
        "salience_kind": salience_kind,
    }


# Tolerate the export name used by gsplats's resolver convention.
PointsAdditiveSpec = Union[None, bool, dict]
