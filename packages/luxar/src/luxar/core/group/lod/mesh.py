"""Mesh LOD axis resolvers — substitutive (decimation) and additive (reveal).

The Mesh sibling of :mod:`luxar.core.group.lod.points` /
:mod:`luxar.core.group.lod.lines` — and, unlike those two, **not** a thin
wrapper over the shared :func:`luxar.core.group.lod.group.resolve_substitutive_axis`.

## Why mesh needs its own resolver

Points and Lines share one implementation because they coarsen the same way:
both LIFT their elements to gsplats and run the gsplat substitutive pipeline. So
they share its whole vocabulary, including four keys that exist only because of
that lift —

* ``truncation_radius`` — feeds ``lift_points_to_gsplats`` / ``lift_lines_to_gsplats``
* ``max_aspect`` — caps per-splat anisotropy on the merged coarse levels
* ``device`` / ``seed`` — the mixture reduction's compute placement and RNG

Mesh does not lift. It coarsens by DECIMATION: merge vertices, reindex the faces,
drop the triangles that collapsed. None of those four keys names anything the
decimator can do, and there is no Gaussian mixture for ``method="kmeans"`` to
reduce. Widening the shared resolver would therefore have meant accepting five
words that quietly do nothing — which is exactly the class of bug the geometry
capability table exists to prevent, one layer down.

What mesh *does* share is the part that is genuinely geometry-agnostic:
``compression_factor`` / ``levels`` / ``coverage_fractions`` / ``coarsen_dims``
mean the same thing here as anywhere, and
:func:`luxar.core.group.lod.group.coverage_fractions` takes plain element counts.
Those are reused verbatim; only the vocabulary around them is mesh's own.

## ``coarsen_dims`` is the shared name for the decimator's ``spatial_dims``

They are the same concept: the dimensions the reduction may merge across, with
the complement acting as hard barriers so a coarse element never blends across a
timepoint or a channel. The decimator's parameter is spelled ``spatial_dims``
because that is what a grid is built over; the *authoring* vocabulary stays
``coarsen_dims`` so a mesh reads like the other three geometries. This module is
where the two names meet.

@module luxar.core.group.lod.mesh
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .group import (
    DEFAULT_MESH_SUBSTITUTIVE_METHOD,
    DEFAULT_SUBSTITUTIVE_K,
    DEFAULT_SUBSTITUTIVE_LEVELS,
    MAX_COVERAGE_FRACTION,
    MESH_SUBSTITUTIVE_METHODS,
    _assert_strict_ascending,
    _validate_coarsen_dims_spec,
)

#: Keys the shared (lift-to-gsplats) vocabulary accepts and mesh cannot. Named
#: individually so the error says WHY rather than "unrecognized key" — each of
#: these is a reasonable thing to have tried, having read the Points docs.
_LIFT_ONLY_KEYS: Dict[str, str] = {
    "truncation_radius": (
        "it is the Gaussian truncation radius used when lifting elements to "
        "gsplats, and a mesh is not lifted"
    ),
    "max_aspect": (
        "it caps per-splat anisotropy on merged coarse levels, and a mesh's "
        "coarse levels are triangles, not splats"
    ),
    "device": (
        "the decimator is pure NumPy on the CPU (deliberately — a scipy or torch "
        "dependency would put `luxar mesh lod` behind an optional extra)"
    ),
    "seed": (
        "vertex clustering is deterministic: there is no seeding, no "
        "initialization and nothing to converge, so there is no RNG to fix"
    ),
}


def _reject_lift_only_keys(kwargs: Dict[str, Any]) -> None:
    """Refuse the four :data:`_LIFT_ONLY_KEYS` with the reason each cannot apply.

    Run BEFORE the generic unknown-key sweep at the end of the resolver so these
    get their specific explanation rather than being lumped into a list of typos.
    Extracted from :func:`resolve_substitutive_axis_mesh` so the mesh-specific
    vocabulary refusal reads as one step (and to keep the resolver under the C901
    limit the complexity ratchet enforces); insertion order of the table decides
    which key a multi-key call is told about.
    """
    for key, why in _LIFT_ONLY_KEYS.items():
        if key in kwargs:
            raise ValueError(
                f"substitutive_lod for Mesh: {key!r} does not apply to a mesh — "
                f"{why}. It is valid for Points/Lines/GSplats, which coarsen by "
                "reducing a Gaussian mixture; a mesh coarsens by decimation."
            )


def _validate_coverage_fractions_spec(value: Any) -> Optional[List[float]]:
    """Coerce and shape-check an explicit ``coverage_fractions`` list (or ``None``).

    ``None`` passes through as ``None`` (the ladder derives its own thresholds from
    the per-level element counts). Anything else must be a non-empty,
    strictly-ascending sequence of values in ``[0, MAX_COVERAGE_FRACTION]``,
    coarsest→finest — the same
    contract :func:`luxar.core.group.lod.group.resolve_substitutive_axis` enforces
    for the lifting geometries, and named after its ``_validate_coarsen_dims_spec``
    peer. The upper bound is NOT ``1.0``: a mesh ladder under a hand-built
    ``kind=partition`` DERIVES a finest of exactly
    :data:`~luxar.core.group.lod.group.MAX_COVERAGE_FRACTION` via
    ``derive_coverage_fractions``, so an author must be able to write the same
    ladder by hand. The LENGTH is not checked here: it must match the ladder that
    actually got written, which is only known once levels that could not reduce the
    surface have been dropped, so the adder's wrapper checks it.
    """
    if value is None:
        return None
    explicit_coverage = [float(m) for m in value]
    if not explicit_coverage:
        raise ValueError(
            "substitutive_lod=dict(coverage_fractions=...) must be non-empty "
            f"(one strictly-ascending value in [0, {MAX_COVERAGE_FRACTION:g}] "
            "per LOD level)"
        )
    _assert_strict_ascending(
        explicit_coverage, "substitutive_lod=dict(coverage_fractions=...)"
    )
    if explicit_coverage[0] < 0.0 or explicit_coverage[-1] > MAX_COVERAGE_FRACTION:
        raise ValueError(
            "substitutive_lod=dict(coverage_fractions=...): values must lie in "
            f"[0, {MAX_COVERAGE_FRACTION:g}] (coarsest→finest); got "
            f"{explicit_coverage}. An explicit list keeps the legacy "
            "selector='coverage' diagonal units, whose upper bound is "
            "SCREEN_FILL_DIAGONAL_RATIO/FILL_FACTOR — roughly the metric a "
            "screen-filling object produces. (Omit the list for the derived "
            "screen-area ladder.)"
        )
    return explicit_coverage


def resolve_substitutive_axis_mesh(spec: Any) -> Optional[Dict[str, Any]]:
    """Normalize a mesh ``substitutive_lod=`` kwarg into a spec dict (or ``None``).

    Vocabulary:

    * ``None`` / ``False`` → no-op (the caller writes a plain mesh leaf).
    * ``True`` / ``dict()`` → defaults (K=4, levels=3, method="auto").
    * ``dict(...)`` → keys ``compression_factor`` (alias ``K``), ``levels``
      (alias ``n_lods``), ``method``, ``coverage_fractions``, ``coarsen_dims``.

    ``method`` accepts :data:`MESH_SUBSTITUTIVE_METHODS`. ``auto`` resolves to
    topology-preserving ``qem`` through 10,000 vertices and vectorized ``cluster``
    above that performance envelope.

    Raises:
        TypeError: If ``spec`` is neither ``None``, a bool, nor a dict.
        ValueError: On an out-of-range value, an unknown key, or one of the
            lift-only keys in :data:`_LIFT_ONLY_KEYS` — the latter with the
            reason it cannot apply to a surface, since every one of them is
            valid for Points and Lines and so a reasonable thing to have tried.
    """
    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(
            f"substitutive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)

    # Checked BEFORE the generic unknown-key sweep so these get their specific
    # explanation rather than being lumped into a list of typos.
    _reject_lift_only_keys(kwargs)

    compression_factor = int(
        kwargs.pop("compression_factor", kwargs.pop("K", DEFAULT_SUBSTITUTIVE_K))
    )
    if compression_factor < 2:
        raise ValueError(f"compression_factor must be >= 2, got {compression_factor}")

    levels = int(
        kwargs.pop("levels", kwargs.pop("n_lods", DEFAULT_SUBSTITUTIVE_LEVELS))
    )
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    method = str(kwargs.pop("method", DEFAULT_MESH_SUBSTITUTIVE_METHOD)).replace(
        "-", "_"
    )
    if method not in MESH_SUBSTITUTIVE_METHODS:
        raise ValueError(
            f"substitutive_lod for Mesh: method must be one of "
            f"{sorted(MESH_SUBSTITUTIVE_METHODS)}; got {method!r}. The "
            "Gaussian-mixture reducers (kmeans, greedy, ...) coarsen a mixture of "
            "independent elements and have no meaning for a connected surface, "
            "which is decimated instead."
        )

    explicit_coverage = _validate_coverage_fractions_spec(
        kwargs.pop("coverage_fractions", None)
    )

    # Shape/type only here; names and the "display" default are resolved against
    # the scene in the adder, exactly as the other three geometries do it.
    coarsen_dims = _validate_coarsen_dims_spec(kwargs.pop("coarsen_dims", None))

    if kwargs:
        raise ValueError(
            f"substitutive_lod for Mesh: unrecognized keys {sorted(kwargs)}. "
            "Valid keys: compression_factor (K), levels (n_lods), method, "
            "coverage_fractions, coarsen_dims. (A mesh's vocabulary is SHORTER "
            "than Points/Lines/GSplats — see this module's docstring for the "
            "four lift-only keys it deliberately omits.)"
        )

    return {
        "compression_factor": compression_factor,
        "levels": levels,
        "method": method,
        "coverage_fractions": explicit_coverage,
        "coarsen_dims": coarsen_dims,
    }


#: The additive orderings a mesh accepts — reveals, and only reveals.
#:
#: This is the shortest vocabulary in the codebase and the restriction is the
#: whole design, so it is worth stating why rather than treating it as
#: unfinished work.
#:
#: **A prefix of a face set is a surface with holes in it.** For Points and Lines
#: a prefix is a sparser SAMPLE of the same object — fewer stars, fewer
#: streamlines — which is an honest coarse approximation, so `random` and
#: `salience` are meaningful there. Take a random half of a mesh's triangles and
#: you do not get a coarser surface; you get confetti. The one family of orderings
#: that yields a presentable prefix is the spatially coherent one: a reveal, whose
#: every prefix is a contiguous partial object at full brightness that grows.
#: That is a statement about surfaces, not about how much work has been done —
#: `luxar.mesh.decimate` is where "coarser surface" lives, and it is the
#: substitutive axis above.
#:
#: Two independent consequences fall out, which is the tell that the restriction
#: is the right cut rather than a convenient one:
#:
#: 1. **No energy stamps, by construction.**
#:    :func:`~luxar.core.group.lod.group.additive_level_stats` already suppresses
#:    them for reveal methods, so a mesh ladder cannot reach the hazard §9.1 of
#:    MESH_NODE_SPEC warns about — the viewer's ``1/e(k)`` brightness
#:    compensation is gated on the BLENDING MODE and never on geometry type,
#:    which would blow out an inner shell and then dim it as the surface
#:    completes. Nothing here has to remember to suppress anything; restricting
#:    the method set is the enforcement.
#: 2. **Vertex duplication stays far below the unwelded worst case.** Each level
#:    re-indexes its own faces (:func:`luxar.mesh.split.split_mesh_by_faces`), so
#:    a vertex on a level boundary is stored once per level that touches it. A
#:    connected patch has ONE boundary curve, so the duplication scales with that
#:    curve rather than with the face count; a random order duplicates almost every
#:    interior vertex and approaches the 3x ceiling of a fully unwelded soup.
#:    MEASURED at 4 levels, reveal vs a random order of the same faces: 288-triangle
#:    plane **1.66 vs 2.95**, 320-face icosphere **2.67 vs 3.31**, 1280-face
#:    icosphere **1.69 vs 3.25**. The ratio improves as the mesh gets finer relative
#:    to the level count, which is the boundary-to-area ratio falling.
#:
#:    This argument is downstream of the connectivity one and was NOT free: under a
#:    plain radius sort the prefixes interleaved instead of nesting, there was no
#:    single boundary curve, and the sphere numbers were 2.81 / 2.17 — much closer to
#:    random. Growing through adjacency is what makes both claims true at once.
MESH_ADDITIVE_METHODS: frozenset[str] = frozenset({"radial"})

#: Unlike the element geometries — whose default is ``random`` — a mesh's default
#: IS the only accepted value, so ``additive_lod=True`` means "reveal outward from
#: the surface's own bbox centre".
DEFAULT_MESH_ADDITIVE_METHOD: str = "radial"

#: Keys the element (Points/Lines) additive vocabulary accepts and mesh cannot,
#: each with the reason. Same shape and purpose as :data:`_LIFT_ONLY_KEYS` above:
#: every one of these is a reasonable thing to have tried after reading the Points
#: docs, so the error names the mechanism rather than saying "unrecognized key".
_ELEMENT_ONLY_ADDITIVE_KEYS: Dict[str, str] = {
    "salience_kind": (
        "it chooses whether an element's rank comes from its size or its "
        "radiometric energy, and both are properties of an independently-emitting "
        "element; a triangle's contribution is its share of a surface, and "
        "ranking triangles by it produces holes rather than a dimmer surface"
    ),
    "seed": (
        "the only ordering a mesh accepts is deterministic — a reveal sorts by "
        "distance from a centre, so there is no sampling and no RNG to fix"
    ),
}


def _reject_element_only_additive_keys(kwargs: Dict[str, Any]) -> None:
    """Refuse :data:`_ELEMENT_ONLY_ADDITIVE_KEYS` with the reason each cannot apply.

    Run BEFORE the generic unknown-key sweep so these get their explanation rather
    than being lumped in with typos — the same ordering, and the same reason for
    it, as :func:`_reject_lift_only_keys`.
    """
    for key, why in _ELEMENT_ONLY_ADDITIVE_KEYS.items():
        if key in kwargs:
            raise ValueError(
                f"additive_lod for Mesh: {key!r} does not apply to a mesh — "
                f"{why}. It is valid for Points/Lines, whose elements emit "
                "independently; a mesh reveals a connected surface."
            )


def _pop_mesh_breakpoints(kwargs: Dict[str, Any]) -> Any:
    """Pop and validate the ``counts``/``breakpoints`` alias pair for a mesh ladder.

    Extracted from :func:`resolve_additive_axis_mesh` rather than inlined for the
    same reason :func:`~luxar.core.group.lod.reveal.pop_reveal_knobs` was extracted
    from the shared resolver: the block is four branches of self-contained
    validation and it pushed the resolver to C901 11, which the complexity ratchet
    counts as a new regression (the gate is "no worse", not "under the limit").

    Mutates ``kwargs``, so the caller's leftover-keys sweep still catches typos.
    """
    from ....utils.lod_breakpoints import validate_element_breakpoints

    if "counts" in kwargs and "breakpoints" in kwargs:
        raise ValueError(
            "additive_lod for Mesh: pass either 'counts' or 'breakpoints', not both "
            "(they are aliases)"
        )
    counts = kwargs.pop("counts", kwargs.pop("breakpoints", None))
    if isinstance(counts, str) and counts.strip().startswith("energy:"):
        # Refused rather than let through: the parser only honours energy
        # fractions when it is HANDED an energy array, and a mesh has none — the
        # same absence that restricts the method set. Left alone this would fall
        # back to equal-count splits and silently ignore the fractions asked for.
        raise ValueError(
            "additive_lod for Mesh: 'energy:' breakpoints need a per-element "
            "energy to integrate, and a mesh has none — a triangle's brightness "
            "is a property of the surface it belongs to, not of the triangle. Use "
            "'stream:<c>' for a geometric ladder, an explicit count list, or "
            "n_lods for equal-count levels."
        )
    if counts is not None and not isinstance(counts, str):
        counts = [int(c) for c in counts]
    if counts is not None:
        # At RESOLVE time, not write time: under a substitutive ladder or a
        # partition the wrapper group is already on disk before its children are
        # written, so a late raise leaves a partial group behind. Same reason
        # `pop_reveal_knobs` validates early.
        validate_element_breakpoints(counts)
    return counts


def resolve_additive_axis_mesh(spec: Any) -> Optional[Dict[str, Any]]:
    """Normalize a mesh ``additive_lod=`` kwarg into a spec dict (or ``None``).

    Vocabulary:

    * ``None`` / ``False`` → no-op (the caller writes a plain mesh leaf).
    * ``True`` / ``dict()`` → defaults (``method="radial"``, 4 levels).
    * ``dict(...)`` → keys ``method``, ``n_lods``, ``counts`` (alias
      ``breakpoints``), ``reveal_center``, ``spatial_dims``.

    ``method`` accepts only :data:`MESH_ADDITIVE_METHODS` — see its docstring for
    why that is one name and not an omission.

    Deliberately NOT a wrapper over
    :func:`~luxar.core.group.lod.group.resolve_additive_axis`, for the same reason
    :func:`resolve_substitutive_axis_mesh` is not one over its shared peer:
    widening the shared vocabulary would mean accepting words that quietly do
    nothing. Two keys are refused by name and three of the five methods are
    refused with the surface argument, so the overlap with the shared resolver is
    the mechanical part (breakpoints, reveal knobs) and those helpers are called
    directly rather than copied.

    Raises:
        TypeError: If ``spec`` is neither ``None``, a bool, nor a dict.
        ValueError: On an out-of-range value, an unknown key, a non-reveal
            ``method``, or one of :data:`_ELEMENT_ONLY_ADDITIVE_KEYS`.
    """
    from .group import DEFAULT_ADDITIVE_N_LODS
    from .reveal import pop_reveal_knobs

    if spec is None or spec is False:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise TypeError(
            f"additive_lod must be None, bool, or dict; got {type(spec).__name__}"
        )
    kwargs = dict(spec)

    _reject_element_only_additive_keys(kwargs)

    method = str(kwargs.pop("method", DEFAULT_MESH_ADDITIVE_METHOD)).replace("_", "-")
    if method not in MESH_ADDITIVE_METHODS:
        raise ValueError(
            f"additive_lod for Mesh: method must be one of "
            f"{sorted(MESH_ADDITIVE_METHODS)}; got {method!r}. A prefix of a face "
            "set is a surface with HOLES in it, not a coarser surface, so only a "
            "spatially coherent reveal yields a presentable prefix — 'random' and "
            "'salience' would scatter triangles, and the samplers "
            "('spatial-uniform', 'poisson-disk') thin an element cloud, which a "
            "connected surface is not. To make a mesh genuinely coarser, use "
            "substitutive_lod= (decimation)."
        )

    n_lods = int(kwargs.pop("n_lods", DEFAULT_ADDITIVE_N_LODS))
    if n_lods < 1:
        raise ValueError(f"n_lods must be >= 1, got {n_lods}")

    counts = _pop_mesh_breakpoints(kwargs)

    reveal_center, spatial_dims = pop_reveal_knobs(kwargs, method)

    if kwargs:
        raise ValueError(
            f"additive_lod for Mesh: unrecognized keys {sorted(kwargs)}. "
            "Valid keys: method, n_lods, counts, breakpoints, reveal_center, "
            "spatial_dims. (A mesh's vocabulary is SHORTER than Points/Lines — "
            "see MESH_ADDITIVE_METHODS for why only a reveal applies to a "
            "surface.)"
        )

    return {
        "method": method,
        "n_lods": n_lods,
        "counts": counts,
        "reveal_center": reveal_center,
        "spatial_dims": spatial_dims,
    }


def _slice_at_cuts(perm: "NDArray", cuts: "List[int]", n: int) -> "List[NDArray]":
    """Slice ``perm`` at CUMULATIVE ``cuts``, appending the tail past the last cut.

    The single place the ladder turns cut positions into disjoint level groups, so
    the cumulative-vs-increment confusion cannot recur in two spellings. The tail
    matters: ``counts=[10, 30, 60]`` on 100 faces is four levels, not three — the
    last cut is a cut, not an end. Empty runs are dropped so a duplicated or
    clamped cut cannot emit a zero-face level, which
    :func:`luxar.mesh.split.split_mesh_by_faces` would count as a part.
    """
    levels: "List[NDArray]" = []
    start = 0
    for cut in list(cuts) + [n]:
        stop = min(int(cut), n)
        if stop > start:
            levels.append(perm[start:stop])
            start = stop
    return levels


def _face_adjacency(faces: "NDArray", n_faces: int) -> tuple:
    """Face-to-face adjacency over shared EDGES, as a CSR pair ``(offsets, nbrs)``.

    Built with one ``np.unique`` over the ``3F`` sorted edge keys rather than a
    Python dict, so it stays vectorized: faces sharing an edge key are neighbours.

    An edge shared by more than two faces (a non-manifold seam — which the
    decimator itself produces, see :mod:`luxar.mesh.decimate`) is linked as a
    CHAIN through its owners rather than as every ordered pair. Both consumers need
    only REACHABILITY — :func:`_component_seeds` floods the CSR, and the reveal
    frontier admits any face touching an admitted one — and a chain keeps the whole
    group connected with ``O(k)`` links where a clique costs ``k*(k-1)``. A
    manifold edge (``k == 2``) is identical either way; a degenerate one (many
    faces on a single edge, which nothing rejects today) no longer expands
    quadratically. The chain is also a subgraph of the true adjacency, so a prefix
    grown through it is still connected in the mesh.
    """
    import numpy as np

    f = faces.astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    owner = np.tile(np.arange(n_faces, dtype=np.int64), 3)
    _keys, inverse = np.unique(e, axis=0, return_inverse=True)
    inverse = inverse.ravel()

    # Group face owners by edge id, then link CONSECUTIVE owners within each group
    # (both directions) — the chain the docstring describes. Fully vectorized: a
    # pair exists wherever two adjacent entries of the sorted list share an edge id,
    # so no per-group Python loop is needed at all.
    order = np.argsort(inverse, kind="stable")
    grouped_edge = inverse[order]
    grouped_face = owner[order]
    same_edge = grouped_edge[:-1] == grouped_edge[1:]
    left = grouped_face[:-1][same_edge]
    right = grouped_face[1:][same_edge]
    if left.size == 0:
        return np.zeros(n_faces + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    src_arr = np.concatenate([left, right])
    dst_arr = np.concatenate([right, left])
    sort_idx = np.argsort(src_arr, kind="stable")
    nbrs = dst_arr[sort_idx]
    counts = np.bincount(src_arr, minlength=n_faces)
    offsets = np.zeros(n_faces + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, nbrs


def _component_seeds(
    offsets: "NDArray", nbrs: "NDArray", scores: "NDArray", n_faces: int
) -> List[int]:
    """One seed per edge-connected component: its face NEAREST the reveal center.

    Labels the components once off the adjacency CSR
    :func:`_face_adjacency` already built, so the reveal can seed every component
    UP FRONT and let a single heap interleave them by radius (#1514). Visiting
    faces in ascending score means the first face of each component encountered is
    also its closest to the centre, so no per-component minimum pass is needed.

    A flood fill rather than a union-find: the CSR is already the adjacency a fill
    walks, and each face is pushed and popped exactly once, so this is O(F + E)
    with no parent-chain bookkeeping. On the overwhelmingly common CONNECTED mesh
    it returns after one fill of length F and yields a single seed.
    """
    import numpy as np

    labelled = np.zeros(n_faces, dtype=bool)
    seeds: List[int] = []
    for start in np.argsort(scores, kind="stable"):
        start_i = int(start)
        if labelled[start_i]:
            continue
        seeds.append(start_i)
        labelled[start_i] = True
        stack = [start_i]
        while stack:
            face = stack.pop()
            for k in range(offsets[face], offsets[face + 1]):
                nb = int(nbrs[k])
                if not labelled[nb]:
                    labelled[nb] = True
                    stack.append(nb)
    return seeds


def compute_additive_order_mesh(
    vertices: "NDArray",
    faces: "NDArray",
    *,
    method: str = DEFAULT_MESH_ADDITIVE_METHOD,
    reveal_center: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> "NDArray":
    """Order a mesh's FACES for a reveal, returning a permutation of face indices.

    One representative per face — its centroid — scored by
    :func:`~luxar.core.group.lod.reveal.radial_element_score` and sorted
    ASCENDING, so the nearest shell to the centre comes first. Ascending is
    correct and unusual: the score is a distance, not a contribution to maximise.

    Follows the LINES call pattern rather than the Points one: centroids are
    DERIVED representatives whose bounding box is not the vertex bounding box (it
    is strictly inside it), so the centre has to be resolved against the vertices
    up front via :func:`~luxar.core.group.lod.reveal.resolve_reveal_center`. Left
    to the scorer's own default, a reveal would grow from the centre of the
    centroid cloud instead of the centre of the surface — close on a symmetric
    mesh and visibly off on an asymmetric one.

    Args:
        vertices: ``(V, D)`` vertex coordinates.
        faces: ``(F, 3)`` triangle indices.
        method: Must be in :data:`MESH_ADDITIVE_METHODS`.
        reveal_center: Explicit centre, ``D`` values (or the scored subset).
        spatial_dims: Which columns the distance is measured over.

    Returns:
        ``(F,)`` face permutation, ``np.intp``.
    """
    import numpy as np

    from ....mesh.split import face_centroids
    from .reveal import radial_element_score, resolve_reveal_center

    if method not in MESH_ADDITIVE_METHODS:
        raise ValueError(
            f"unknown mesh additive method {method!r}; expected one of "
            f"{sorted(MESH_ADDITIVE_METHODS)}"
        )

    faces_arr = np.asarray(faces).reshape(-1, 3)
    if faces_arr.shape[0] == 0:
        return np.empty(0, dtype=np.intp)

    centroids = face_centroids(np.asarray(vertices), faces_arr.astype(np.uint32))
    centre = resolve_reveal_center(
        reveal_center, np.asarray(vertices), centroids, spatial_dims
    )
    scores = radial_element_score(centroids, centre, spatial_dims)
    n_faces = int(faces_arr.shape[0])

    # ── best-first growth through face ADJACENCY, keyed on radius ──
    #
    # NOT `argsort(scores)`. Sorting by radius alone does not keep a prefix
    # connected, and the failure is worst on exactly the data this targets: on a
    # closed surface every centroid sits at nearly the same radius, so the order is
    # decided by small variations spread over the whole shell and the "shells"
    # interleave instead of nesting. MEASURED on an icosphere with `n_lods=4`,
    # edge-connected components of each cumulative prefix: 20 / 20 / 1 / 1 under a
    # plain radial sort — a half-loaded sphere as twenty separate patches of lace,
    # which is the failure mode this whole axis refuses `random` to avoid.
    #
    # Growing through shared edges instead makes the guarantee structural: the
    # frontier only ever admits a face touching one already admitted, so every
    # prefix is a connected patch on ANY topology — sphere, torus, or a non-convex
    # dumbbell. The radius key is what makes it a *reveal* rather than an arbitrary
    # flood: among all faces currently reachable, the nearest to the centre goes
    # next. It degenerates to the radial sort exactly when the mesh is convex and
    # every shell is reachable, which is the case the sort already handled.
    #
    # COST, stated rather than hidden: this is a heap loop in Python, ~O(F log F)
    # with a per-face constant far above a vectorized argsort — order a second per
    # 100k faces. It runs once at authoring time, next to a decimator that is also
    # a Python loop, so the trade (a true guarantee for authoring seconds) is the
    # right one; a vectorized reformulation would be welcome and is not needed yet.
    # EVERY component is seeded UP FRONT, not opened when the frontier runs dry.
    #
    # That difference is the whole of #1514. Refilling the heap only once it
    # emptied meant one component was exhausted before the next opened — and on a
    # STACKED mesh (several timepoints or channels held in one vertex array, a
    # first-class authoring shape) the components ARE the timepoints, so the
    # ladder came out sequenced by time: two 320-face icospheres stacked as
    # t=0/t=1 with `n_lods=4` gave 160/0, 160/0, 0/160, 0/160 faces per level. A
    # viewer parked on the last timepoint then renders NOTHING until the whole
    # ladder has arrived, which is the opposite of what a streaming ladder is for.
    #
    # It also defeated the machinery that exists to prevent exactly this:
    # `resolve_reveal_spatial_dims` already keeps the stacked column out of the
    # SCORE, and the traversal reintroduced the same effect through connectivity.
    #
    # Seeding all components at once puts one heap over all of them, so they
    # interleave by radius while each still grows only through shared edges. Both
    # guarantees survive, and the docstring's "one patch per component" becomes
    # literally true rather than "per component reached so far".
    import heapq

    offsets, nbrs = _face_adjacency(faces_arr, n_faces)
    component_seeds = _component_seeds(offsets, nbrs, scores, n_faces)

    visited = np.zeros(n_faces, dtype=bool)
    order: list = []
    heap: list = []
    for seed in component_seeds:
        visited[seed] = True
        heapq.heappush(heap, (float(scores[seed]), seed))
    while heap:
        _radius, face = heapq.heappop(heap)
        order.append(face)
        for k in range(offsets[face], offsets[face + 1]):
            nb = int(nbrs[k])
            if not visited[nb]:
                visited[nb] = True
                heapq.heappush(heap, (float(scores[nb]), nb))
    return np.asarray(order, dtype=np.intp)


def make_additive_lod_mesh(
    vertices: "NDArray",
    faces: "NDArray",
    *,
    method: str = DEFAULT_MESH_ADDITIVE_METHOD,
    n_lods: int = 4,
    counts: Any = None,
    reveal_center: Optional[List[float]] = None,
    spatial_dims: Optional[List[int]] = None,
) -> "List[NDArray]":
    """Split a mesh's faces into additive levels — a reveal ladder.

    Returns DISJOINT face-index groups, coarsest first, whose union is every face
    exactly once. That is deliberately the same contract
    :func:`luxar.mesh.split.split_mesh_by_faces` enforces, so each level can be
    re-indexed by it without a second partition check: an additive ladder's levels
    ARE a partition of the faces, and the viewer forms level *i* by concatenating
    groups 0..i.

    Levels are cumulative *when concatenated*, not individually — level 1 holds
    only the faces level 0 does not, exactly as the Points and Lines ladders slice
    their permutations.

    An empty mesh returns ``[]`` and a single level returns one group; both let the
    caller fall through to a plain leaf rather than writing a one-level ladder.
    """
    import numpy as np

    # Cross-module private import, matching `lines.py`'s reuse of
    # `_energy_breakpoints_to_counts`: the breakpoint vocabulary is shared and
    # copying it is how the five `--method` help strings rotted.
    from .points import _parse_breakpoints_spec

    faces_arr = np.asarray(faces).reshape(-1, 3)
    n_faces = int(faces_arr.shape[0])
    if n_faces == 0:
        return []

    perm = compute_additive_order_mesh(
        vertices,
        faces_arr,
        method=method,
        reveal_center=reveal_center,
        spatial_dims=spatial_dims,
    )

    # `_parse_breakpoints_spec` returns CUMULATIVE cut positions, not per-level
    # increments — `points.py::_validate_counts` enforces strictly-increasing and
    # clamped, which only makes sense for cuts. Reading them as increments (the bug
    # this comment replaces) lost a level and doubled every `stream:` chunk after
    # the first, and `n_lods=` masked it because its own fallback below is the one
    # place increments ARE the natural form. So: cuts here, increments there, and
    # `_cuts_to_slices` is the single place the two meet.
    #
    # No `energy=`/`perm=` arguments: energy breakpoints are refused up front by
    # `resolve_additive_axis_mesh`, so nothing here can need them.
    cuts = _parse_breakpoints_spec(counts, n_faces)
    if cuts is not None:
        return _slice_at_cuts(perm, cuts, n_faces)

    # Equal-count fallback: the one place per-level increments are the natural form,
    # converted to cuts immediately so there is exactly one slicer.
    base, extra = divmod(n_faces, max(1, n_lods))
    increments = [base + (1 if i < extra else 0) for i in range(max(1, n_lods))]
    running = 0
    equal_cuts = []
    for inc in increments:
        running += inc
        equal_cuts.append(running)
    return _slice_at_cuts(perm, equal_cuts, n_faces)


__all__ = [
    "DEFAULT_MESH_ADDITIVE_METHOD",
    "compute_additive_order_mesh",
    "make_additive_lod_mesh",
    "MESH_ADDITIVE_METHODS",
    "resolve_additive_axis_mesh",
    "resolve_substitutive_axis_mesh",
]
