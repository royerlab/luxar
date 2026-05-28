"""luxar.core.lod_points – Additive-LOD helpers for the Points leaf type.

Mirrors the role that ``gsplats/lod/additive.py`` plays for GSplats:
provides the ordering + ladder-construction primitives that the
``add_points(..., additive_lod=...)`` convenience kwarg consumes.

Three ordering methods, all geometry-agnostic-ish:

* ``random``     — uniform-random permutation (with optional seed).
* ``salience``   — sort by radii descending (largest points first).
* ``spatial-uniform`` — stratified-grid sampling
  (:func:`luxar.core._spatial_uniform.stratified_grid_order`).

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

from ._spatial_uniform import stratified_grid_order

#: Ordering methods supported on Points additive LOD.
PointsMethodName = Literal["random", "salience", "spatial-uniform"]


# Default for ``additive_lod=True`` and ``additive_lod=dict()``.
DEFAULT_N_LODS: int = 4
DEFAULT_METHOD: PointsMethodName = "random"


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
            raise ValueError(
                "salience ordering requires per-element radii; got None"
            )
        if radii.shape[0] != n:
            raise ValueError(
                f"radii length ({radii.shape[0]}) must match positions "
                f"length ({n})"
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

    raise ValueError(
        f"method must be one of 'random' / 'salience' / 'spatial-uniform'; "
        f"got {method!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# Public LOD-ladder construction
# ─────────────────────────────────────────────────────────────────────


def make_additive_lod_points(
    positions: NDArray,
    radii: Optional[NDArray] = None,
    *,
    method: PointsMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    counts: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> List[NDArray[np.intp]]:
    """Compute per-LOD-level index arrays for Points.

    Each entry in the returned list is a 1-D array of indices into the
    original ``positions``: concatenating them in order gives the full
    permutation. The caller (``_write_points_multi_lod`` in
    ``io/compiler.py``) slices the data arrays accordingly.

    Args:
        positions: ``(N, d)`` array.
        radii: ``(N,)`` array; required for ``salience``.
        method: Ordering method.
        n_lods: Target level count (default 4). Ignored if ``counts``
            is provided.
        counts: Cumulative element-count breakpoints (e.g.
            ``[1500, 8000, 40000]`` → L0 ≤ 1500, L1 ≤ 8000-1500, etc.).
            ``None`` falls back to equal-count splits of size
            ``ceil(N / n_lods)``.
        seed: For ``random``; ignored otherwise.

    Returns:
        List of per-level index arrays, length
        ``min(n_lods, n_elements)`` for the non-empty-tail case
        (trailing empty levels are dropped silently).
    """
    n = positions.shape[0]
    if n == 0:
        return []

    perm, natural_counts = compute_additive_order_points(
        positions, radii=radii, method=method, n_lods=n_lods, seed=seed
    )

    if method == "spatial-uniform":
        # The stratified-grid sampler already partitions the
        # permutation into per-level groups. Respect that partition
        # rather than re-slicing — it's what makes the cumulative
        # density approximately uniform.
        out: List[NDArray[np.intp]] = []
        cursor = 0
        for count in natural_counts:
            if count > 0:
                out.append(perm[cursor : cursor + count])
            cursor += count
        return out

    # For random / salience, slice the permutation by breakpoints.
    if counts is not None:
        breakpoints = _validate_counts(counts, n)
    else:
        # Equal-count split into n_lods levels.
        per_level = max(1, (n + n_lods - 1) // n_lods)  # ceil-divide
        breakpoints = [
            min(n, (i + 1) * per_level) for i in range(n_lods - 1)
        ]

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

    Vocabulary (mirrors gsplats's :func:`luxar.core.lod.resolve_additive_axis`
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
        }
    if not isinstance(spec, dict):
        raise TypeError(
            f"additive_lod must be None, bool, or dict; got "
            f"{type(spec).__name__}"
        )
    kwargs = dict(spec)
    # ``recompute`` is gsplats-only; tolerate but don't act on it.
    kwargs.pop("recompute", None)

    method = kwargs.pop("method", DEFAULT_METHOD)
    if method not in ("random", "salience", "spatial-uniform"):
        raise ValueError(
            f"method must be one of 'random' / 'salience' / 'spatial-uniform'; "
            f"got {method!r}"
        )

    n_lods = int(kwargs.pop("n_lods", DEFAULT_N_LODS))
    if n_lods < 1:
        raise ValueError(f"n_lods must be >= 1, got {n_lods}")

    counts = kwargs.pop("counts", None)
    if counts is not None:
        counts = [int(c) for c in counts]

    seed = kwargs.pop("seed", None)
    if seed is not None:
        seed = int(seed)

    if kwargs:
        raise ValueError(
            f"additive_lod for Points: unrecognized keys "
            f"{sorted(kwargs)}. Valid keys: method, n_lods, counts, "
            f"seed, recompute."
        )

    return {
        "method": method,
        "n_lods": n_lods,
        "counts": counts,
        "seed": seed,
    }


# Tolerate the export name used by gsplats's resolver convention.
PointsAdditiveSpec = Union[None, bool, dict]
