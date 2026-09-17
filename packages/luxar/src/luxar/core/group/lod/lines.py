"""luxar.core.group.lod.lines – Additive-LOD helpers for the Lines leaf type.

Mirrors ``core/group/lod/points.py`` in shape but operates **per-polyline**:
each ``additive_<i>/`` subgroup carries whole polylines (vertices +
their segments) so that segment topology stays valid during partial
loads.

Polyline-identification per ``line_type``:

* ``segments`` — each consecutive pair of vertices is its own polyline
  of length 2. N/2 polylines.
* ``indexed``  — connected-components walk over the explicit segments;
  one component = one polyline. When a multi-level additive ladder is
  emitted, each component's edge multiset must equal its consecutive-vertex chain.
* ``polyline`` / ``loop`` — ONE polyline encompassing all vertices. A
  multi-LOD ladder over a single polyline is a no-op (would require
  vertex-subsampling, which breaks the "polyline-level, no topology
  damage" choice). For these the helper logs a warning and emits a
  single LOD level.

Five ordering methods mirror the Points helper:

* ``random``           — uniform-random per-polyline permutation.
* ``salience``         — sort polylines by ``length × max_width`` desc.
* ``spatial-uniform``  — stratified-grid sampling on per-polyline bbox
  centers (:func:`luxar.core.group.lod.spatial_uniform.stratified_grid_order`).
* ``poisson-disk``     — blue-noise sampling on the same bbox centers.
* ``radial``           — concentric shells around the node's own bbox centre,
  ordering WHOLE polylines by their own centre's distance, so a streaming
  prefix grows outward from the middle with segment topology intact (the
  reveal). See :func:`luxar.core.group.lod.reveal.radial_element_score`.
"""

from __future__ import annotations

import warnings
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
from .reveal import radial_element_score, resolve_reveal_center
from .spatial_uniform import stratified_grid_order

#: Ordering methods supported on Lines additive LOD.
LinesMethodName = Literal[
    "random", "salience", "spatial-uniform", "poisson-disk", "radial"
]

# Aliases of the shared resolver's constants so the kwarg defaults below and
# ``resolve_additive_axis_lines`` always agree.
DEFAULT_N_LODS: int = DEFAULT_ADDITIVE_N_LODS
DEFAULT_METHOD: LinesMethodName = DEFAULT_ADDITIVE_METHOD


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
                f"line_type='segments' requires even n_vertices; got {n_vertices}"
            )
        pairs: List[NDArray[np.intp]] = []
        for i in range(0, n_vertices, 2):
            pairs.append(np.array([i, i + 1], dtype=np.intp))
        return pairs

    if line_type == "indexed":
        if indices is None:
            raise ValueError("line_type='indexed' requires an indices array")
        indices = np.asarray(indices)
        # Reject non-integer indices before the intp cast below silently
        # TRUNCATES a float (1.7 -> 1) into an edge the user never authored.
        # The plain single-leaf path guards this in write_lines, but the
        # partition / additive-LOD branches consume the raw indices through
        # this function first, so the guard has to live at this shared
        # chokepoint too (#886). The substitutive path has an equivalent gate
        # in gsplats.lift._segment_pairs.
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"Indices must be an integer array, got dtype {indices.dtype}"
            )
        if indices.size > 0:
            min_index = int(np.min(indices))
            max_index = int(np.max(indices))
            if min_index < 0:
                raise ValueError(f"Index {min_index} < 0 (indices must be >= 0)")
            if max_index >= n_vertices:
                raise ValueError(f"Index {max_index} >= n_vertices {n_vertices}")
        return _indexed_connected_components(
            n_vertices, indices.astype(np.intp, copy=False).reshape(-1, 2)
        )

    raise ValueError(
        f"line_type must be one of 'segments' / 'polyline' / 'loop' / "
        f"'indexed'; got {line_type!r}"
    )


def _indexed_connected_components(
    n_vertices: int,
    segments: NDArray[np.intp],
) -> List[NDArray[np.intp]]:
    """Return connected components of an indexed graph in vertex order.

    The union step hooks component roots in bulk with ``np.minimum.at`` and
    pointer-jumps all parents between passes. Grouping then stable-sorts the
    final root labels once and slices that permutation at label boundaries.
    This avoids both Python work per edge and the former full ``roots == root``
    scan per component, which was quadratic for ribbon-heavy indexed Lines.

    Components are ordered by their smallest vertex index; members within a
    component retain ascending vertex order. Isolated vertices become
    single-element components so every vertex appears exactly once.
    """
    if n_vertices == 0:
        return []

    parent = np.arange(n_vertices, dtype=np.intp)
    if segments.size > 0:
        endpoint_a = segments[:, 0]
        endpoint_b = segments[:, 1]

        while True:
            # Compress every current forest to roots. Pointer jumping halves
            # path lengths each pass and is vectorized over all vertices.
            while True:
                grandparents = parent[parent]
                if np.array_equal(grandparents, parent):
                    break
                parent = grandparents

            root_a = parent[endpoint_a]
            root_b = parent[endpoint_b]
            if np.array_equal(root_a, root_b):
                break

            # Hook every larger root to the smallest adjacent root. Duplicate
            # writes are reduced by minimum, so edge order cannot change the
            # result and parent links always decrease (cycles are impossible).
            # When the roots differ, at least one root strictly decreases;
            # non-negative parent values therefore guarantee termination.
            high_roots = np.maximum(root_a, root_b)
            np.minimum(root_a, root_b, out=root_a)
            np.minimum.at(parent, high_roots, root_a)

    # ``parent`` is fully compressed on the terminating pass above. A stable
    # sort groups equal roots while preserving ascending vertex order inside
    # each component; one boundary scan replaces C full scans of V roots.
    order = np.argsort(parent, kind="stable").astype(np.intp, copy=False)
    sorted_roots = parent[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.flatnonzero(sorted_roots[1:] != sorted_roots[:-1]).astype(
                np.intp, copy=False
            )
            + 1,
        )
    )
    # ``np.split`` returns zero-copy views into the shared permutation.
    return [chunk for chunk in np.split(order, starts[1:])]


def _indexed_ladder_edge_multisets(
    indices: NDArray,
    polylines: List[NDArray[np.intp]],
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return canonical undirected edge multisets with multiplicity preserved."""
    authored = np.asarray(indices, dtype=np.intp).reshape(-1, 2)
    if polylines:
        lengths = np.fromiter(
            (members.size for members in polylines),
            dtype=np.intp,
            count=len(polylines),
        )
        members = np.concatenate(polylines)
        if members.size >= 2:
            keep = np.ones(members.size - 1, dtype=bool)
            keep[np.cumsum(lengths[:-1]) - 1] = False
            rebuilt = np.column_stack((members[:-1][keep], members[1:][keep]))
        else:
            rebuilt = np.empty((0, 2), dtype=np.intp)
    else:
        rebuilt = np.empty((0, 2), dtype=np.intp)

    def canonical_edges(edges: NDArray[np.intp]) -> NDArray[np.intp]:
        if edges.size == 0:
            return np.empty((0, 2), dtype=np.intp)
        canonical = np.sort(edges, axis=1)
        return canonical[np.lexsort((canonical[:, 1], canonical[:, 0]))]

    return canonical_edges(authored), canonical_edges(rebuilt)


def _indexed_ladder_preserves_edges(
    indices: NDArray,
    polylines: List[NDArray[np.intp]],
) -> bool:
    """Compare undirected edge multisets, ignoring edge direction and row order."""
    authored, rebuilt = _indexed_ladder_edge_multisets(indices, polylines)
    return np.array_equal(authored, rebuilt)


def indexed_components_are_chains(
    n_vertices: int,
    segments: NDArray[np.intp],
) -> bool:
    """Whether component chains preserve the authored undirected edge multiset.

    The indexed additive writer discards the explicit edge list and rebuilds it
    by chaining each connected component in ascending vertex order. The authored
    and rebuilt edge multisets must therefore match exactly, including duplicate
    multiplicity; edge direction and row order do not matter.

    Branches, cycles, self-loops, chords, non-ascending paths, and duplicate edges
    are rejected because the rebuilt chains would change their topology or
    additive brightness. An interior numbering gap is safe because it creates
    separate connected components that are chained independently.
    """
    segments = np.asarray(segments, dtype=np.intp).reshape(-1, 2)
    if n_vertices == 0 or segments.size == 0:
        return True
    components = _indexed_connected_components(n_vertices, segments)
    return _indexed_ladder_preserves_edges(segments, components)


def _validate_indexed_ladder_edges(
    line_type: str,
    indices: Optional[NDArray],
    polylines: List[NDArray[np.intp]],
    levels: List[List[NDArray[np.intp]]],
) -> None:
    """Reject multi-level indexed ladders that would change authored edges."""
    if len(levels) <= 1 or line_type != "indexed":
        return
    assert indices is not None
    if _indexed_ladder_preserves_edges(indices, polylines):
        return

    authored, rebuilt = _indexed_ladder_edge_multisets(indices, polylines)
    combined = np.concatenate((authored, rebuilt), axis=0)
    unique_edges, inverse = np.unique(combined, axis=0, return_inverse=True)
    split = authored.shape[0]
    authored_counts = np.bincount(inverse[:split], minlength=unique_edges.shape[0])
    rebuilt_counts = np.bincount(inverse[split:], minlength=unique_edges.shape[0])
    mismatch = int(np.flatnonzero(authored_counts > rebuilt_counts)[0])
    missing_edge = tuple(map(int, unique_edges[mismatch]))
    raise ValueError(
        "line_type='indexed' additive LOD cannot preserve the explicit edge list: "
        "every connected component's undirected edge multiset must equal its consecutive "
        "vertex pairs (edge direction and row order do not matter). "
        f"Authored {authored.shape[0]} edges but the component chains produce "
        f"{rebuilt.shape[0]}; offending authored edge {missing_edge} is not matched "
        f"by the component chains (authored multiplicity {authored_counts[mismatch]}, "
        f"chain multiplicity {rebuilt_counts[mismatch]}). "
        "Remove additive_lod= and use partition= alone, or re-author the edges with "
        "line_type='segments'."
    )


# ─────────────────────────────────────────────────────────────────────
# Per-polyline ordering
# ─────────────────────────────────────────────────────────────────────


def polyline_bbox_centers(
    vertices: NDArray,
    polylines: List[NDArray[np.intp]],
    ncols: Optional[int] = None,
) -> NDArray:
    """One representative coordinate per polyline: its own bbox centre.

    The spatial stand-in for a polyline in any ordering that treats it as a
    single element. ``ncols`` limits the columns considered — the samplers use
    ``3`` because their grids are 3-D — while ``None`` (the default) uses every
    column, which is what ``radial`` needs so that
    :func:`~luxar.core.group.lod.reveal.radial_element_score` can see, and
    therefore exclude, a stacked time or channel column.
    """
    cols = vertices.shape[1] if ncols is None else min(ncols, vertices.shape[1])
    reps = np.empty((len(polylines), cols), dtype=np.float64)
    for i, members in enumerate(polylines):
        pts = vertices[members, :cols].astype(np.float64)
        reps[i] = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    return reps


def compute_additive_order_lines(
    vertices: NDArray,
    polylines: List[NDArray[np.intp]],
    widths: Optional[NDArray] = None,
    method: LinesMethodName = DEFAULT_METHOD,
    n_lods: int = DEFAULT_N_LODS,
    seed: Optional[int] = None,
    reveal_center: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> Tuple[NDArray[np.intp], List[int]]:
    """Compute an additive ordering permutation over **polylines** (not vertices).

    Args:
        vertices: ``(N, d)`` vertex positions.
        polylines: Per-polyline vertex-index arrays (from
            :func:`identify_polylines`).
        widths: ``(N,)`` per-vertex widths; required for ``salience``.
        method: ``random`` / ``salience`` / ``spatial-uniform`` /
            ``poisson-disk`` / ``radial``.
        n_lods: Consulted only by ``spatial-uniform`` / ``poisson-disk``.
        seed: For ``random``.
        reveal_center: ``radial`` only — centre of the shells, defaulting to the
            spatial bounding-box centre of ``vertices`` (the node's own middle —
            NOT of the polyline representatives, whose bounding box weighs a
            long polyline and a short one equally).
        spatial_dims: ``radial`` only — columns the distance is measured over,
            defaulting to the columns with non-zero extent (which drops a
            *constant* time/channel column but not a *stacked* one — see
            :func:`~luxar.core.group.lod.reveal.radial_element_score`).

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
            raise ValueError("salience ordering requires per-vertex widths; got None")
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
        reps = polyline_bbox_centers(vertices, polylines, ncols=3)
        if method == "poisson-disk":
            return poisson_disk_order(reps, n_lods, seed=seed or 0)
        return stratified_grid_order(reps, n_lods)

    if method == "radial":
        # ASCENDING, unlike `salience` above: the score is a DISTANCE, so the
        # nearest polyline is revealed first and the prefixes grow outward as
        # concentric shells. A polyline is revealed WHOLE — the ordering is over
        # polylines, not vertices — so segment topology survives every prefix.
        #
        # ALL columns, not just the first three: the shell axes are chosen from
        # the full column set (by `spatial_dims`, or by non-zero extent), so
        # truncating to 3 would silently pick the wrong ones on nD data.
        #
        # Returns an EMPTY natural partition, deliberately — see the Points
        # equivalent: per-level counts would make `make_additive_lod_lines`
        # bypass the caller's `counts:` / `stream:` breakpoints entirely.
        # The default centre comes from `vertices`, the NODE's own bbox — not
        # from the bbox of `reps`, in which a long polyline and a short one weigh
        # the same and pull the origin off the geometry's middle.
        reps = polyline_bbox_centers(vertices, polylines)
        return (
            np.argsort(
                radial_element_score(
                    reps,
                    resolve_reveal_center(reveal_center, vertices, reps, spatial_dims),
                    spatial_dims,
                ),
                kind="stable",
            ).astype(np.intp),
            [],
        )

    raise ValueError(
        f"method must be one of {' / '.join(repr(m) for m in ADDITIVE_METHODS)}; "
        f"got {method!r}"
    )


# ─────────────────────────────────────────────────────────────────────
# Public LOD-ladder construction
# ─────────────────────────────────────────────────────────────────────


def compute_lines_energy(
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
        isinstance(colors, np.ndarray) and colors.ndim == 2 and colors.shape[1] >= 3
    )
    has_scalars = scalars is not None and (
        isinstance(scalars, np.ndarray) and scalars.shape == (vertices.shape[0],)
    )
    if has_colors:
        c = np.asarray(colors)[:, :3].astype(np.float64, copy=False)
        vertex_lum = 0.2126 * c[:, 0] + 0.7152 * c[:, 1] + 0.0722 * c[:, 2]
    elif has_scalars:
        vertex_lum = np.asarray(scalars, dtype=np.float64).reshape(-1)
    else:
        vertex_lum = None

    if (
        widths is not None
        and isinstance(widths, np.ndarray)
        and widths.shape == (vertices.shape[0],)
    ):
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


def _energy_string_polyline_cuts(
    counts: str,
    energy: NDArray[np.float64],
    perm: NDArray[np.intp],
    polylines: List[NDArray[np.intp]],
) -> List[int]:
    """Polyline-count cuts for the two energy-based string specs.

    ``energy:<fractions>`` → cumulative energy fractions (shared with Points);
    ``equi-energy:<n>`` → equal shares of cumulative polyline energy along the
    order, with the commit cap measured in VERTICES (the payload currency, as
    for ``stream:``). Any other string is refused naming all three vocabularies.
    """
    if counts.startswith("equi-energy:"):
        from ....utils.lod_breakpoints import equi_energy_cuts, parse_equi_energy_rungs

        poly_lengths = [int(polylines[int(i)].shape[0]) for i in perm]
        return equi_energy_cuts(
            np.asarray(energy, dtype=np.float64)[perm],
            parse_equi_energy_rungs(counts),
            weights=poly_lengths,
        )
    if not counts.startswith("energy:"):
        raise ValueError(
            f"unrecognized breakpoints string {counts!r}; expected "
            "'energy:<fractions>' (e.g. 'energy:0.5,0.9,1.0'), "
            "'equi-energy:<n>' (e.g. 'equi-energy:4') or "
            "'stream:<c>' (e.g. 'stream:40000')"
        )
    from .points import _energy_breakpoints_to_counts  # shared helper

    fracs = [float(s) for s in counts[len("energy:") :].split(",") if s.strip()]
    return _energy_breakpoints_to_counts(energy, perm, fracs)


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
    reveal_center: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> List[List[NDArray[np.intp]]]:
    """Compute per-LOD-level polyline groupings for Lines.

    Each entry in the returned list is a list of per-polyline vertex
    arrays — the polylines assigned to that LOD level. The caller
    (``write_lines_multi_lod`` in ``io/compiler.py``) walks each
    level's polylines, gathers their vertices + widths + (per-vertex)
    color / scalar arrays, builds segment indices LOCAL to the
    subgroup, and writes the subgroup.

    Args:
        vertices: ``(N, d)`` array.
        line_type: One of ``segments`` / ``polyline`` / ``loop`` / ``indexed``.
        indices: Required for ``indexed``.
        widths: ``(N,)`` per-vertex widths; required for ``salience``.
        method / n_lods / counts / seed: see Points equivalent. Same
          semantics, with one currency note: a ``"stream:<c>"`` spec sizes
          ``c`` in VERTICES (so the same spec means the same payload for
          every geometry). Each geometric vertex target ``[c, 2c, 4c, …]`` is
          mapped to the first whole-polyline boundary whose CUMULATIVE vertex
          count (along the additive order) reaches it, so cuts stay on
          whole-polyline boundaries while honouring the vertex budget even when
          polyline lengths are highly skewed.
        reveal_center: For ``method='radial'`` — centre of the concentric
            shells, defaulting to the node's own vertex bounding-box centre.
        spatial_dims: For ``method='radial'`` — the vertex columns the shell
            distance is measured over, defaulting to the columns with non-zero
            extent (which excludes a *constant* time/channel column, but not a
            *stacked* one — see
            :func:`~luxar.core.group.lod.reveal.radial_element_score`).

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
        energy = compute_lines_energy(vertices, polylines, widths, colors, scalars)
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
            reveal_center=reveal_center,
            spatial_dims=spatial_dims,
        )

    if method in ("spatial-uniform", "poisson-disk"):
        # Both samplers return a natural per-level partition; respect it.
        out: List[List[NDArray[np.intp]]] = []
        cursor = 0
        for count in natural_counts:
            if count > 0:
                level_polylines = [
                    polylines[int(i)] for i in perm[cursor : cursor + count]
                ]
                out.append(level_polylines)
            cursor += count
        _validate_indexed_ladder_edges(line_type, indices, polylines, out)
        return out

    # random / salience: slice the polyline permutation by breakpoints.
    if (
        isinstance(counts, str)
        and counts.startswith(("energy:", "equi-energy:"))
        and energy is None
    ):
        energy = compute_lines_energy(vertices, polylines, widths, colors, scalars)

    if isinstance(counts, str) and counts.startswith("stream:"):
        # `stream:C` is sized in VERTICES — the payload currency, symmetric with
        # Points and GSplats — but this ladder can only cut on whole-polyline
        # boundaries. Converting C to a fixed polyline count via the MEAN length
        # misses the vertex budget badly when polyline lengths are skewed (one
        # giant `indexed` component plus many tiny ones), so instead size the
        # cuts against the ACTUAL cumulative vertex count along the additive
        # order: each geometric vertex target `[C, 2C, 4C, …]` becomes the first
        # whole-polyline boundary whose cumulative vertices reach it. Cuts stay
        # on whole-polyline boundaries, preserving the segment-topology
        # invariant, and every level honours the vertex budget as closely as
        # indivisible polylines allow.
        from ....utils.lod_breakpoints import parse_stream_chunk, stream_cuts

        chunk_verts = parse_stream_chunk(counts)
        poly_lengths = np.fromiter(
            (polylines[int(i)].shape[0] for i in perm), dtype=np.int64, count=p
        )
        cum_verts = np.cumsum(poly_lengths)
        breakpoints: List[int] = []
        for target in stream_cuts(n, chunk_verts)[:-1]:
            # first whole-polyline boundary whose cumulative vertices reach target
            bp = min(p, int(np.searchsorted(cum_verts, target, side="left")) + 1)
            if bp > (breakpoints[-1] if breakpoints else 0):
                breakpoints.append(bp)
    elif isinstance(counts, str):
        if energy is None:
            energy = compute_lines_energy(vertices, polylines, widths, colors, scalars)
        breakpoints = _energy_string_polyline_cuts(counts, energy, perm, polylines)
    elif counts is not None:
        breakpoints = _validate_counts(counts, p)
    else:
        per_level = max(1, (p + n_lods - 1) // n_lods)
        breakpoints = [min(p, (i + 1) * per_level) for i in range(n_lods - 1)]

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
    _validate_indexed_ladder_edges(line_type, indices, polylines, out)
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

    Thin wrapper over the shared
    :func:`luxar.core.group.lod.group.resolve_additive_axis` (one
    implementation shared with Points so the two can't drift).
    """
    return resolve_additive_axis(spec, "Lines")


LinesAdditiveSpec = Union[None, bool, dict]


def resolve_substitutive_axis_lines(spec: Any) -> Optional[dict]:
    """Translate the ``substitutive_lod=`` kwarg value into a normalized dict.

    The substitutive axis coarsens a line set by **synthesising gsplats**: each
    segment is lifted to a string of isotropic "bead" Gaussians (view-independent,
    summing to a smooth tube) and the gsplat substitutive pipeline builds
    fewer-but-larger representative levels, which become the coarse levels of a
    lines LOD ladder (the finest level stays the original Lines node).

    Thin wrapper over the shared
    :func:`luxar.core.group.lod.group.resolve_substitutive_axis` (one
    implementation shared with Points so the two can't drift). See it for the
    full value vocabulary.
    """
    from .group import resolve_substitutive_axis

    return resolve_substitutive_axis(spec, "Lines")
