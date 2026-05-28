"""luxar.core.group.lod.lines – Additive-LOD helpers for the Lines leaf type.

Mirrors ``core/group/lod/points.py`` in shape but operates **per-polyline**:
each ``additive_<i>/`` subgroup carries whole polylines (vertices +
their segments) so that segment topology stays valid during partial
loads.

Polyline-identification per ``line_type``:

* ``segments`` — each consecutive pair of vertices is its own polyline
  of length 2. N/2 polylines.
* ``indexed``  — connected-components walk over the explicit segments;
  one component = one polyline.
* ``polyline`` / ``loop`` — ONE polyline encompassing all vertices. A
  multi-LOD ladder over a single polyline is a no-op (would require
  vertex-subsampling, which breaks the "polyline-level, no topology
  damage" choice). For these the helper logs a warning and emits a
  single LOD level.

Three ordering methods mirror the Points helper:

* ``random``           — uniform-random per-polyline permutation.
* ``salience``         — sort polylines by ``length × max_width`` desc.
* ``spatial-uniform``  — stratified-grid sampling on per-polyline bbox
  centers (:func:`luxar.core.group.lod.spatial_uniform.stratified_grid_order`).
"""

from __future__ import annotations

import warnings
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .poisson_disk import poisson_disk_order
from .spatial_uniform import stratified_grid_order

#: Ordering methods supported on Lines additive LOD.
LinesMethodName = Literal[
    "random", "salience", "spatial-uniform", "poisson-disk"
]

DEFAULT_N_LODS: int = 4
DEFAULT_METHOD: LinesMethodName = "random"


# ─────────────────────────────────────────────────────────────────────
# Polyline identification
# ─────────────────────────────────────────────────────────────────────


def identify_polylines(
    n_vertices: int,
    line_type: str,
    indices: Optional[NDArray] = None,
) -> List[NDArray[np.intp]]:
    """Return per-polyline vertex-index arrays.

    Args:
        n_vertices: Total vertices in the lines node.
        line_type: One of ``segments`` / ``polyline`` / ``loop`` /
            ``indexed``.
        indices: Required for ``indexed``; ignored otherwise.

    Returns:
        List of per-polyline ``(K_p,)`` int arrays into the vertex
        array. The list's length is the number of distinct polylines.

        * ``polyline`` / ``loop``: returns a single array
          ``[0, 1, ..., n-1]``.
        * ``segments``: returns ``n // 2`` arrays of shape ``(2,)``.
        * ``indexed``: walks the explicit segments to find connected
          components.
    """
    if n_vertices == 0:
        return []

    if line_type == "polyline" or line_type == "loop":
        return [np.arange(n_vertices, dtype=np.intp)]

    if line_type == "segments":
        if n_vertices % 2 != 0:
            raise ValueError(
                f"line_type='segments' requires even n_vertices; got "
                f"{n_vertices}"
            )
        pairs: List[NDArray[np.intp]] = []
        for i in range(0, n_vertices, 2):
            pairs.append(np.array([i, i + 1], dtype=np.intp))
        return pairs

    if line_type == "indexed":
        if indices is None:
            raise ValueError(
                "line_type='indexed' requires an indices array"
            )
        return _indexed_connected_components(
            n_vertices, np.asarray(indices, dtype=np.intp).reshape(-1, 2)
        )

    raise ValueError(
        f"line_type must be one of 'segments' / 'polyline' / 'loop' / "
        f"'indexed'; got {line_type!r}"
    )


def _indexed_connected_components(
    n_vertices: int,
    segments: NDArray[np.intp],
) -> List[NDArray[np.intp]]:
    """Union-Find connected-components walk on indexed segments.

    Returns per-component vertex-index arrays. Isolated vertices (with
    no incident segment) become single-element polylines so every
    vertex appears in exactly one polyline.
    """
    # Union-Find on vertices.
    parent = np.arange(n_vertices, dtype=np.intp)

    def find(x: int) -> int:
        # Iterative path-halving.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in segments:
        union(int(a), int(b))

    # Group vertices by root.
    roots = np.array([find(i) for i in range(n_vertices)], dtype=np.intp)
    unique_roots = np.unique(roots)
    components: List[NDArray[np.intp]] = []
    for root in unique_roots:
        members = np.nonzero(roots == root)[0].astype(np.intp)
        components.append(members)
    return components


# ─────────────────────────────────────────────────────────────────────
# Per-polyline ordering
# ─────────────────────────────────────────────────────────────────────


def compute_additive_order_lines(
    vertices: NDArray,
    polylines: List[NDArray[np.intp]],
    widths: Optional[NDArray] = None,
    method: LinesMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    seed: Optional[int] = None,
) -> Tuple[NDArray[np.intp], List[int]]:
    """Compute an additive ordering permutation over **polylines** (not vertices).

    Args:
        vertices: ``(N, d)`` vertex positions.
        polylines: Per-polyline vertex-index arrays (from
            :func:`identify_polylines`).
        widths: ``(N,)`` per-vertex widths; required for ``salience``.
        method: ``random`` / ``salience`` / ``spatial-uniform``.
        n_lods: Consulted only by ``spatial-uniform``.
        seed: For ``random``.

    Returns:
        ``(polyline_permutation, per_level_polyline_counts)``. The
        permutation indexes into ``polylines`` (not into vertices).
        For ``random`` / ``salience`` the per-level counts list is
        empty (caller slices); for ``spatial-uniform`` it describes
        the natural per-LOD partition of polylines.
    """
    p = len(polylines)
    if p == 0:
        return np.empty(0, dtype=np.intp), []

    if method == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(p).astype(np.intp), []

    if method == "salience":
        if widths is None:
            raise ValueError(
                "salience ordering requires per-vertex widths; got None"
            )
        score = np.empty(p, dtype=np.float64)
        for i, members in enumerate(polylines):
            if members.size < 2:
                # Length-0 segment (degenerate): use width alone.
                score[i] = float(widths[members].max()) if members.size else 0.0
                continue
            # Polyline length = sum of consecutive vertex distances. For
            # segments-type pairs this is just the single segment length.
            pts = vertices[members, :3].astype(np.float64)
            seg_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            total_len = float(seg_lens.sum())
            max_w = float(widths[members].max())
            score[i] = total_len * max_w
        perm = np.argsort(-score, kind="stable").astype(np.intp)
        return perm, []

    if method in ("spatial-uniform", "poisson-disk"):
        if vertices.shape[1] < 3:
            raise ValueError(
                f"{method} ordering needs vertices with d >= 3; "
                f"got shape {vertices.shape}"
            )
        # Representative spatial point per polyline = bbox center.
        reps = np.empty((p, 3), dtype=np.float64)
        for i, members in enumerate(polylines):
            pts = vertices[members, :3].astype(np.float64)
            reps[i] = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        if method == "poisson-disk":
            return poisson_disk_order(reps, n_lods, seed=seed or 0)
        return stratified_grid_order(reps, n_lods)

    raise ValueError(
        "method must be one of 'random' / 'salience' / 'spatial-uniform' "
        f"/ 'poisson-disk'; got {method!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# Public LOD-ladder construction
# ─────────────────────────────────────────────────────────────────────


def _compute_lines_energy(
    vertices: NDArray,
    polylines: List[NDArray[np.intp]],
    widths: Optional[NDArray],
    colors: Optional[NDArray],
    scalars: Optional[NDArray],
) -> NDArray[np.float64]:
    """Per-polyline energy score: ``mean_luminance × Σ (seg_length × width²)``.

    The tube-volume analog of points-energy: ``width**2`` sweeps along
    each segment's length so we get a perceptual mass per polyline.
    Returns ``(P,)`` float64 non-negative.

    - **Mean luminance**: Rec.709 over the polyline's per-vertex colors;
      mean of ``scalars`` if colors absent; 1.0 if neither.
    - **Width**: per-vertex widths, defaulting to 1.0 when absent.
    - **Length**: sum of consecutive vertex distances (single segment
      for ``segments`` line_type pairs).
    """
    p = len(polylines)
    if p == 0:
        return np.empty(0, dtype=np.float64)

    pts3 = vertices[:, :3].astype(np.float64, copy=False)
    has_colors = colors is not None and (
        isinstance(colors, np.ndarray)
        and colors.ndim == 2
        and colors.shape[1] >= 3
    )
    has_scalars = scalars is not None and (
        isinstance(scalars, np.ndarray)
        and scalars.shape == (vertices.shape[0],)
    )
    if has_colors:
        c = np.asarray(colors)[:, :3].astype(np.float64, copy=False)
        vertex_lum = 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]
    elif has_scalars:
        vertex_lum = np.asarray(scalars, dtype=np.float64).reshape(-1)
    else:
        vertex_lum = None

    if widths is not None and isinstance(widths, np.ndarray) and widths.shape == (vertices.shape[0],):
        vertex_w = widths.astype(np.float64, copy=False)
    else:
        vertex_w = None

    out = np.empty(p, dtype=np.float64)
    for i, members in enumerate(polylines):
        if members.size == 0:
            out[i] = 0.0
            continue
        pts = pts3[members]
        if members.size < 2:
            seg_lens = np.zeros(0)
            w_seg = np.zeros(0)
        else:
            seg_lens = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            if vertex_w is not None:
                # Per-segment width = mean of endpoint widths.
                w_seg = 0.5 * (vertex_w[members[1:]] + vertex_w[members[:-1]])
            else:
                w_seg = np.ones_like(seg_lens)
        tube_volume = float((seg_lens * (w_seg * w_seg)).sum())
        if vertex_lum is not None:
            mean_lum = float(np.clip(vertex_lum[members], 0.0, None).mean())
        else:
            mean_lum = 1.0
        out[i] = max(0.0, mean_lum) * max(0.0, tube_volume)
    return out


def make_additive_lod_lines(
    vertices: NDArray,
    *,
    line_type: str = "polyline",
    indices: Optional[NDArray] = None,
    widths: Optional[NDArray] = None,
    method: LinesMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    counts: Optional[Any] = None,
    seed: Optional[int] = None,
    colors: Optional[NDArray] = None,
    scalars: Optional[NDArray] = None,
    salience_kind: Literal["size", "energy"] = "size",
) -> List[List[NDArray[np.intp]]]:
    """Compute per-LOD-level polyline groupings for Lines.

    Each entry in the returned list is a list of per-polyline vertex
    arrays — the polylines assigned to that LOD level. The caller
    (``_write_lines_multi_lod`` in ``io/compiler.py``) walks each
    level's polylines, gathers their vertices + widths + (per-vertex)
    color / scalar arrays, builds segment indices LOCAL to the
    subgroup, and writes the subgroup.

    Args:
        vertices: ``(N, d)`` array.
        line_type: One of ``segments`` / ``polyline`` / ``loop`` / ``indexed``.
        indices: Required for ``indexed``.
        widths: ``(N,)`` per-vertex widths; required for ``salience``.
        method / n_lods / counts / seed: see Points equivalent. Same
          semantics.

    Returns:
        List of LOD-level entries. Each entry is a list of per-polyline
        vertex-index arrays.

        For ``polyline`` / ``loop`` (single polyline): returns a single
        LOD level containing that one polyline (no-op LOD; a warning is
        logged). Multi-LOD ladder over a single polyline would require
        breaking the polyline-level invariant.
    """
    n = vertices.shape[0]
    if n == 0:
        return []

    polylines = identify_polylines(n, line_type, indices)
    p = len(polylines)
    if p == 0:
        return []

    if p == 1 and line_type in ("polyline", "loop") and n_lods > 1:
        warnings.warn(
            f"line_type={line_type!r} produces a single polyline; "
            f"polyline-level additive LOD is a no-op. Emitting 1 LOD "
            "level. Use line_type='segments' or 'indexed' for "
            "multi-polyline data if you want a multi-level ladder.",
            UserWarning,
            stacklevel=2,
        )
        return [[polylines[0]]]

    # salience_kind='energy' overrides the default radii-based salience
    # with a per-polyline luminance × tube-volume score.
    energy: Optional[NDArray[np.float64]] = None
    if method == "salience" and salience_kind == "energy":
        energy = _compute_lines_energy(
            vertices, polylines, widths, colors, scalars
        )
        perm = np.argsort(-energy, kind="stable").astype(np.intp)
        natural_counts: List[int] = []
    else:
        perm, natural_counts = compute_additive_order_lines(
            vertices,
            polylines,
            widths=widths,
            method=method,
            n_lods=n_lods,
            seed=seed,
        )

    if method in ("spatial-uniform", "poisson-disk"):
        # Both samplers return a natural per-level partition; respect it.
        out: List[List[NDArray[np.intp]]] = []
        cursor = 0
        for count in natural_counts:
            if count > 0:
                level_polylines = [polylines[int(i)] for i in perm[cursor : cursor + count]]
                out.append(level_polylines)
            cursor += count
        return out

    # random / salience: slice the polyline permutation by breakpoints.
    if isinstance(counts, str) and counts.startswith("energy:") and energy is None:
        energy = _compute_lines_energy(
            vertices, polylines, widths, colors, scalars
        )

    if isinstance(counts, str):
        # Energy: fractions → cumulative counts (over polylines).
        from .points import _energy_breakpoints_to_counts  # shared helper

        if not counts.startswith("energy:"):
            raise ValueError(
                f"unrecognized breakpoints string {counts!r}; expected "
                "'energy:<fractions>'"
            )
        if energy is None:
            energy = _compute_lines_energy(
                vertices, polylines, widths, colors, scalars
            )
        fracs = [float(s) for s in counts[len("energy:"):].split(",") if s.strip()]
        breakpoints = _energy_breakpoints_to_counts(energy, perm, fracs)
    elif counts is not None:
        breakpoints = _validate_counts(counts, p)
    else:
        per_level = max(1, (p + n_lods - 1) // n_lods)
        breakpoints = [
            min(p, (i + 1) * per_level) for i in range(n_lods - 1)
        ]

    out = []
    start = 0
    for bp in breakpoints:
        if bp > start:
            level_polylines = [polylines[int(i)] for i in perm[start:bp]]
            out.append(level_polylines)
            start = bp
    if start < p:
        level_polylines = [polylines[int(i)] for i in perm[start:p]]
        out.append(level_polylines)
    return out


def _validate_counts(counts: List[int], total: int) -> List[int]:
    """Normalize cumulative element-count breakpoints; same shape as
    :func:`luxar.core.group.lod.points._validate_counts`."""
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
        clamped = min(c, total)
        if clamped > prev:
            breakpoints.append(clamped)
            prev = clamped
        if clamped == total:
            break
    return breakpoints


# ─────────────────────────────────────────────────────────────────────
# Resolver for the ``additive_lod=`` convenience kwarg on ``add_lines``
# ─────────────────────────────────────────────────────────────────────


def resolve_additive_axis_lines(spec: Any) -> Optional[dict]:
    """Translate the ``additive_lod=`` kwarg into a normalized dict.

    Same vocabulary as :func:`luxar.core.group.lod.points.resolve_additive_axis_points`.
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
            f"additive_lod must be None, bool, or dict; got "
            f"{type(spec).__name__}"
        )
    kwargs = dict(spec)
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
            f"additive_lod for Lines: unrecognized keys "
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


LinesAdditiveSpec = Union[None, bool, dict]
