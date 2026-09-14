"""The stamped ``selector`` must describe the stamped thresholds — all four adders.

A ``kind=lod`` group writes TWO things that are really one decision: the per-child
``coverage_fraction`` thresholds, and the group-level ``selector`` that names the
UNITS those thresholds are in. Get the pairing wrong and nothing raises — both
selector spellings are individually valid — but every switch point the viewer
computes is on the wrong scale.

The rule, which :func:`luxar.core.group.lod.group.resolve_lod_ladder` decides once
for the four SCENE ADDERS:

* an explicit ``coverage_fractions=[...]`` list is used verbatim and keeps the
  LEGACY units it was authored in (``LEGACY_LOD_SELECTOR``, ceiling
  ``MAX_COVERAGE_FRACTION``);
* otherwise the screen-area halving ladder is derived and stamped
  ``DERIVED_LOD_SELECTOR`` — finest ``WHOLE_OBJECT_FINEST_ANCHOR`` for a
  whole-object ladder, ``PARTITION_FINEST_AREA`` for a partition-bound one.

Every geometry that can build such a group is exercised here in one place —
``add_points`` / ``add_lines`` / ``add_mesh`` ``substitutive_lod=`` and
``add_gsplats_from_data`` ``lod_group=`` — because the invariant is
cross-geometry, and because two of the four (mesh ``substitutive_lod=`` and
gsplats ``lod_group=``) stamped a selector that no test had ever read back.
Per-geometry ladder mechanics stay in ``test_substitutive_points.py`` /
``test_substitutive_lines.py`` / ``test_gsplats.py`` /
``core/tests/test_mesh_substitutive_lod.py`` — including the INTERIOR spacing of a
derived ladder (``[0, 0.25, 0.5]`` and friends), which this module deliberately
does not re-assert.

**Scope, stated honestly.** These are regression fences on what the library
WRITES FROM NOW ON, and they were written against already-correct adders — the
behavioural half passes on the pre-refactor revision too. They neither detect nor
repair the 209 already-shipped ladders that carry ``selector="coverage"`` over
derived-looking thresholds (issue #1727); migrating an existing store is the
separate, explicitly opt-in ``luxar restamp-lod`` pass
(:func:`luxar.io.lod_restamp.restamp_lod_store`), which reads a STORE rather than
building one and so is out of scope for the routing guards below too — it calls
neither ``add_lod_group`` nor :data:`_RESOLVER`, and pairs the two ladder
functions with :data:`DERIVED_LOD_SELECTOR` itself because it has no live scene
``Node`` to hand either helper. What the suite buys is that the pairing cannot
drift back apart silently, in any of the four adders, and that a FIFTH producer
has to make a deliberate choice (see
``test_no_unrouted_producer_builds_a_lod_group``).

**Also out of scope: the detached-tree paths**, which are a different question
(what a *stored* tree already claims about its own thresholds, with no ``explicit``
argument to branch on) answered by
:func:`luxar.gsplats.tree.gate_authored_selector`, whose arms are: a fully
authored ladder keeps its own stored selector verbatim (legacy only when it
carries none), a partially- or un-authored one is re-derived and stamped
screen-area, and an out-of-vocabulary selector raises — shared by
``io/_compiler/gsplat_tree.write_gsplat_node`` and
``gsplats_pipeline/from_io.graft_gsplat_node``. The latter is itself a scene door
(``add_gsplats_from_file`` on a non-matrix-shaped subtree) that derives thresholds
directly and pairs them with that gate's selector, so it is exempted by name from
the routing guards below rather than being an oversight.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest

from luxar._zarr_compat import read_consolidated_attrs
from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.group import (
    MAX_COVERAGE_FRACTION,
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
)
from luxar.io.compiler import LuxarZarrCompiler
from luxar.typing_utils.constants import (
    DERIVED_LOD_SELECTOR,
    LEGACY_LOD_SELECTOR,
    LOD_SELECTORS,
)

#: The four adders that can build a ``kind=lod`` group, by short geometry name.
GEOMETRIES = ("points", "lines", "gsplats", "mesh")

#: How a ladder is reached: at the scene root (whole-object anchor), under a
#: hand-built two-part ``kind=partition`` (tile anchor), or with an explicit
#: legacy list.
VARIANTS = ("derived", "partition", "explicit")
POINTS_COMBINED_VARIANT = "combined"


# ────────────────────────────────────────────────────────────────────────
# The invariant, as a reusable assertion
# ────────────────────────────────────────────────────────────────────────


def assert_selector_describes_thresholds(
    selector: Any,
    thresholds: Sequence[float],
    *,
    partition_bound: bool,
    where: str,
) -> None:
    """Assert a written ladder's ``selector`` agrees with its own thresholds.

    This is the consistency invariant, not a re-derivation: it reads only what
    was stamped on disk and asks whether the two halves can both be true.

    * ``selector="screen-area"`` — thresholds are literal screen-area fractions,
      so they live in ``[0, PARTITION_FINEST_AREA]`` and the FINEST one sits
      exactly on the anchor the insertion point implies
      (:data:`PARTITION_FINEST_AREA` under a partition, else
      :data:`WHOLE_OBJECT_FINEST_ANCHOR`). A derived ladder that ended anywhere
      else would mean the selector describes a ladder nobody derived.
    * ``selector="coverage"`` — the legacy diagonal metric, whose only bound is
      :data:`MAX_COVERAGE_FRACTION`. Nothing is asserted about the finest value:
      an author picks it.

    Both cases share the format-level ladder shape (coarsest exactly ``0.0``,
    strictly ascending), because a selector cannot rescue a ladder the viewer
    cannot walk.

    **What this does NOT check: the INTERIOR spacing.** Endpoints, ascent and range
    are the parts that depend on the selector, and they are all it looks at — a
    ladder whose halving base changed from 2 to 3 (say ``[0, 1/9, 1/3, 1]`` in
    place of ``[0, 1/4, 1/2, 1]``) keeps both endpoints and passes here. That is
    deliberate: the interior spacing belongs to the derivation, and is pinned
    literally next to it — ``test_lod_group.py::test_area_halving_spacing``
    (``[0, A/4, A/2, A]`` straight off ``coverage_fractions``) plus, per geometry,
    the literal ``[0, 0.25, 0.5]`` in ``test_substitutive_lines.py`` and the
    whole-ladder equality against ``partitioned_coverage_fractions(counts)`` in
    ``test_substitutive_points.py``. This helper's job is only the
    SELECTOR/THRESHOLD pairing.

    Args:
        selector: The group's stamped ``selector`` attr.
        thresholds: Per-child ``coverage_fraction``, coarsest→finest.
        partition_bound: Whether the ladder's insertion point sits under a
            ``kind=partition`` (which anchor a derived ladder must show).
        where: Identifying text for assertion messages.
    """
    assert selector in LOD_SELECTORS, f"{where}: unknown selector {selector!r}"
    assert thresholds, f"{where}: ladder has no thresholds"
    assert thresholds[0] == pytest.approx(0.0), (
        f"{where}: coarsest threshold must be the 0.0 always-eligible floor, "
        f"got {list(thresholds)}"
    )
    assert all(b > a for a, b in zip(thresholds, thresholds[1:])), (
        f"{where}: thresholds must be strictly ascending, got {list(thresholds)}"
    )
    if selector == DERIVED_LOD_SELECTOR:
        anchor = (
            PARTITION_FINEST_AREA if partition_bound else WHOLE_OBJECT_FINEST_ANCHOR
        )
        assert all(0.0 <= f <= PARTITION_FINEST_AREA for f in thresholds), (
            f"{where}: selector={selector!r} means literal screen-area "
            f"fractions, which cannot exceed {PARTITION_FINEST_AREA:g}; got "
            f"{list(thresholds)}"
        )
        assert thresholds[-1] == pytest.approx(anchor), (
            f"{where}: selector={selector!r} claims a DERIVED screen-area "
            f"ladder, whose finest level must land exactly on its anchor "
            f"{anchor:g}; got {thresholds[-1]}. Either the ladder was authored "
            "(then stamp LEGACY_LOD_SELECTOR) or the anchor is wrong."
        )
    else:
        assert all(0.0 <= f <= MAX_COVERAGE_FRACTION for f in thresholds), (
            f"{where}: selector={selector!r} is the legacy diagonal metric, "
            f"bounded by {MAX_COVERAGE_FRACTION:g}; got {list(thresholds)}"
        )


# ────────────────────────────────────────────────────────────────────────
# One written ladder per (geometry, variant) — read back from disk
# ────────────────────────────────────────────────────────────────────────


def _octahedron_sphere(subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """A closed, welded sphere — a mesh the decimator can actually coarsen."""
    verts: List[np.ndarray] = [
        np.array(v, dtype=np.float64)
        for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    for _ in range(subdivisions):
        midpoint: Dict[Tuple[int, int], int] = {}

        def split(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                midpoint[key] = len(verts) - 1
            return midpoint[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = split(a, b), split(b, c), split(c, a)
            refined += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = refined
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def _add_points_ladder(target: Any, name: str, part: int, **spec: Any) -> None:
    rng = np.random.default_rng(part)
    positions = rng.normal(0, 20, (400, 3)).astype(np.float32)
    target.add_points(
        name,
        positions,
        radii=0.5,
        substitutive_lod=dict(levels=1, device="cpu", seed=0, **spec),
    )


def _add_lines_ladder(target: Any, name: str, part: int, **spec: Any) -> None:
    rng = np.random.default_rng(part)
    starts = rng.uniform(0, 60, (120, 3))
    vertices = np.empty((240, 3), dtype=np.float32)
    vertices[0::2] = starts
    vertices[1::2] = starts + rng.normal(0, 5, (120, 3))
    target.add_lines(
        name,
        vertices,
        0.8,
        line_type="segments",
        substitutive_lod=dict(levels=1, device="cpu", seed=0, **spec),
    )


def _add_gsplats_ladder(target: Any, name: str, part: int, **spec: Any) -> None:
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod import make_substitutive_lod

    rng = np.random.default_rng(part)
    n = 64
    chol = (rng.standard_normal((n, 6)) * 0.1).astype(np.float32)
    chol[:, [0, 2, 5]] = np.abs(chol[:, [0, 2, 5]]) + 0.4
    flat = GSplatData(
        centers=(rng.standard_normal((n, 3)) * 1.5).astype(np.float32),
        amplitudes=(np.abs(rng.standard_normal(n)) + 0.5).astype(np.float32),
        cholesky_factors=chol,
    )
    pyramid = make_substitutive_lod(flat, levels=2, device="cpu")
    # ``lod_group=True`` auto-derives; a dict carries the explicit list.
    target.add_gsplats_from_data(name, pyramid, lod_group=spec or True)


def _add_mesh_ladder(target: Any, name: str, part: int, **spec: Any) -> None:
    verts, faces = _octahedron_sphere(3)
    shifted = verts.copy()
    shifted[:, 0] += 3.0 * part
    target.add_mesh(name, shifted, faces, substitutive_lod={"levels": 1, **spec})


#: ``(adder, partition display_type)`` per geometry.
_LADDER_ADDERS: Dict[str, Tuple[Callable[..., None], str]] = {
    "points": (_add_points_ladder, "points"),
    "lines": (_add_lines_ladder, "lines"),
    "gsplats": (_add_gsplats_ladder, "gsplats"),
    "mesh": (_add_mesh_ladder, "mesh"),
}


def _write_ladders(
    store: Path, geometry: str, *, partitioned: bool, **spec: Any
) -> List[str]:
    """Write TWO ladders (``part_0`` / ``part_1``) and return their node paths.

    Two on purpose for the partition variant: a ONE-part ``kind=partition`` is
    the degenerate shape ``partitioned_coverage_fractions`` documents as too
    coarse (its "tile" IS the whole object), so it must not be the fixture the
    anchor half of the invariant is argued from. The root variant writes the same
    pair so only the insertion point differs.
    """
    add, display_type = _LADDER_ADDERS[geometry]
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        wrapper = (
            scene.add_partition_group(
                "tiled", display_type=display_type, max_elements=100_000
            )
            if partitioned
            else scene
        )
        for part in (0, 1):
            add(wrapper, f"part_{part}", part, **spec)
    prefix = "tiled/" if partitioned else ""
    return [f"{prefix}part_{part}" for part in (0, 1)]


def _coarsest_to_finest(
    children: List[Tuple[str, Dict[str, Any]]],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Order lod-group children by their ``child_index`` attr, coarsest→finest.

    The canonical insertion order, and what
    ``io/_compiler/finalize/lod_backfill.py`` orders by. Deliberately NOT a
    name-sort: ``sorted`` puts ``child_10`` before ``child_2``, so a name-sort
    would silently reorder any ladder of ten or more levels — and a reordered
    ladder would fail the ascending/anchor checks for the wrong reason, or (worse)
    pass them by luck. Extracted as a function so that property can be pinned on
    synthetic input, without paying for a ten-level compiler run.
    """
    return sorted(children, key=lambda kv: int(kv[1]["child_index"]))


def test_children_are_ordered_by_child_index_not_by_name() -> None:
    """The ``_coarsest_to_finest`` claim, on a ladder long enough to expose it.

    Ten-plus levels is where a name-sort and an index-sort diverge; the compiled
    fixtures below top out at three, so this pins the property on a synthetic
    mapping instead of building a real ten-level ladder.
    """
    children = [(f"g/child_{i}", {"child_index": i}) for i in range(12)]
    shuffled = [children[i] for i in (7, 0, 11, 2, 10, 1, 9, 3, 8, 4, 6, 5)]

    assert [p for p, _ in _coarsest_to_finest(shuffled)] == [p for p, _ in children]
    # And the failure it avoids: by NAME, child_10/11 sort between 1 and 2.
    by_name = [p for p, _ in sorted(shuffled)]
    assert by_name != [p for p, _ in children]
    assert by_name.index("g/child_10") < by_name.index("g/child_2")


def _read_ladder(store: Path, group_path: str) -> Tuple[Any, List[float]]:
    """``(selector, thresholds coarsest→finest)`` as WRITTEN, for one lod group.

    Children are ordered by ``child_index`` via :func:`_coarsest_to_finest` — see
    there for why not by name.
    """
    nodes = dict(read_consolidated_attrs(store))
    assert group_path in nodes, f"{group_path} not in {sorted(nodes)}"
    group = nodes[group_path]
    assert group.get("kind") == "lod", f"{group_path} is not a kind=lod group"
    children = [
        (path, dict(attrs))
        for path, attrs in nodes.items()
        if path.startswith(f"{group_path}/child_") and "coverage_fraction" in attrs
    ]
    assert children, f"{group_path} has no ladder children"
    return group.get("selector"), [
        float(attrs["coverage_fraction"]) for _, attrs in _coarsest_to_finest(children)
    ]


def _legacy_list(n_levels: int) -> List[float]:
    """A strictly ascending explicit ladder that is ONLY legal in legacy units.

    For ``n_levels >= 2`` it tops out at ``MAX_COVERAGE_FRACTION`` = 4.0, four
    times the screen-area ceiling, so a store that mislabels it ``screen-area``
    is caught by the invariant rather than merely by an equality check on the
    values. The degenerate ``n_levels == 1`` branch canNOT carry that property —
    ``[0.0]`` is a valid ladder under BOTH selectors — which is why
    :func:`ladders` asserts every built ladder is at least two levels long
    rather than letting a 1-level one pass vacuously.
    """
    if n_levels == 1:
        return [0.0]
    step = MAX_COVERAGE_FRACTION / (n_levels - 1)
    return [0.0] + [step * i for i in range(1, n_levels)]


@pytest.fixture(scope="module")
def ladders(tmp_path_factory) -> Dict[Tuple[str, str], Tuple[Any, List[float]]]:
    """``{(geometry, variant): (selector, thresholds)}`` — every ladder, once.

    Module-scoped because each entry is a real compiler run; the tests below are
    pure assertions over what those runs wrote. The ``explicit`` variant reuses
    the ``derived`` ladder's LENGTH, since a mesh level that could not reduce the
    surface is dropped and an explicit list must match the ladder it lands on.

    Every built ladder is asserted to be at least TWO levels long, because a
    1-level one would make two of the tests below vacuous rather than red: the
    only 1-level ladder is ``[0.0]``, which is legal under BOTH selectors, so
    ``_legacy_list`` loses its off-scale 4.0 and the consistency invariant has
    nothing left to discriminate. Dropping a mesh level is the realistic trigger.
    """
    out: Dict[Tuple[str, str], Tuple[Any, List[float]]] = {}
    root = tmp_path_factory.mktemp("selector_contract")
    for geometry in GEOMETRIES:
        derived = _write_ladders(
            root / f"{geometry}_root.luxar.zarr", geometry, partitioned=False
        )
        out[(geometry, "derived")] = _read_ladder(
            root / f"{geometry}_root.luxar.zarr", derived[0]
        )
        tiled = _write_ladders(
            root / f"{geometry}_tiled.luxar.zarr", geometry, partitioned=True
        )
        out[(geometry, "partition")] = _read_ladder(
            root / f"{geometry}_tiled.luxar.zarr", tiled[0]
        )
        explicit = _legacy_list(len(out[(geometry, "derived")][1]))
        authored = _write_ladders(
            root / f"{geometry}_explicit.luxar.zarr",
            geometry,
            partitioned=False,
            coverage_fractions=explicit,
        )
        out[(geometry, "explicit")] = _read_ladder(
            root / f"{geometry}_explicit.luxar.zarr", authored[0]
        )
        out[(geometry, "explicit_input")] = (None, explicit)
        for variant in VARIANTS:
            n_levels = len(out[(geometry, variant)][1])
            assert n_levels >= 2, (
                f"{geometry}/{variant}: ladder came out {n_levels} level(s) "
                "long. The tests below need at least two, in both directions: a "
                "1-level ladder is [0.0], which carries no anchor, so the "
                "`explicit` (legacy) variant would pass while asserting nothing "
                "(see _legacy_list), and a derived variant would instead fail "
                "the invariant for a reason that has nothing to do with the "
                "selector. Give this geometry "
                "a fixture its coarsener can actually reduce — for mesh that "
                "means enough subdivisions in _octahedron_sphere."
            )
    combined_store = root / "points_combined.luxar.zarr"
    with LuxarZarrCompiler(combined_store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        positions = np.random.default_rng(0).normal(0, 20, (400, 3)).astype(np.float32)
        scene.add_points(
            "points",
            positions,
            radii=0.5,
            partition={"max_elements": 100},
            substitutive_lod=dict(levels=1, device="cpu", seed=0),
            additive_lod=False,
        )
    out[("points", POINTS_COMBINED_VARIANT)] = _read_ladder(combined_store, "points")
    return out


# ────────────────────────────────────────────────────────────────────────
# (a)/(b) A DERIVED ladder stamps the screen-area selector
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_a_derived_ladder_stamps_the_screen_area_selector(ladders, geometry) -> None:
    """Unasserted before #1727 for mesh (`substitutive_lod=`) and gsplats
    (`lod_group=`): both wrote the selector and no test read it back."""
    selector, thresholds = ladders[(geometry, "derived")]
    assert selector == DERIVED_LOD_SELECTOR, (
        f"{geometry}: a derived ladder must stamp {DERIVED_LOD_SELECTOR!r}, "
        f"got {selector!r} over thresholds {thresholds}"
    )


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_a_partition_bound_derived_ladder_still_stamps_screen_area(
    ladders, geometry
) -> None:
    """The tile anchor moves the VALUES, never the units."""
    selector, thresholds = ladders[(geometry, "partition")]
    assert selector == DERIVED_LOD_SELECTOR, (
        f"{geometry}: a partition-bound derived ladder is still screen-area, "
        f"got {selector!r} over thresholds {thresholds}"
    )


def test_points_combined_spelling_routes_selector_with_thresholds(ladders) -> None:
    selector, thresholds = ladders[("points", POINTS_COMBINED_VARIANT)]
    assert selector == DERIVED_LOD_SELECTOR
    assert_selector_describes_thresholds(
        selector,
        thresholds,
        partition_bound=True,
        where="points/combined",
    )


# ────────────────────────────────────────────────────────────────────────
# (c) The other half of the rule: an EXPLICIT list keeps the legacy units
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_an_explicit_list_stamps_the_legacy_selector(ladders, geometry) -> None:
    """An authored list was tuned against the legacy diagonal metric, so
    re-labelling it ``screen-area`` would move every switch point the author
    chose. Nothing asserted this for ANY geometry before #1727."""
    selector, thresholds = ladders[(geometry, "explicit")]
    expected = ladders[(geometry, "explicit_input")][1]
    assert selector == LEGACY_LOD_SELECTOR, (
        f"{geometry}: an explicit coverage_fractions list must stamp "
        f"{LEGACY_LOD_SELECTOR!r}, got {selector!r}"
    )
    assert thresholds == pytest.approx(expected), (
        f"{geometry}: an explicit list must be written verbatim"
    )


# ────────────────────────────────────────────────────────────────────────
# (d) The consistency invariant — selector vs its own thresholds
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_written_ladder_is_self_consistent(ladders, geometry, variant) -> None:
    """The class of breakage #1727 is about: a selector that does not describe
    its own thresholds. Applied to a ladder from each of the four adders, in each
    of the three shapes they can produce."""
    selector, thresholds = ladders[(geometry, variant)]
    assert_selector_describes_thresholds(
        selector,
        thresholds,
        partition_bound=(variant == "partition"),
        where=f"{geometry}/{variant}",
    )


def test_the_invariant_helper_rejects_a_mismatched_pair() -> None:
    """The helper must actually fail on the shipped-store failure mode: derived-
    looking thresholds labelled legacy, and legacy values labelled derived.

    Without this the invariant above could be vacuously true.
    """
    # A legacy-bounded ladder cannot be screen-area: 4.0 is off that scale.
    with pytest.raises(AssertionError, match="screen-area"):
        assert_selector_describes_thresholds(
            DERIVED_LOD_SELECTOR,
            [0.0, 2.0, MAX_COVERAGE_FRACTION],
            partition_bound=False,
            where="synthetic",
        )
    # In range, but not on either anchor — so no derivation produced it.
    with pytest.raises(AssertionError, match="anchor"):
        assert_selector_describes_thresholds(
            DERIVED_LOD_SELECTOR,
            [0.0, 0.3, 0.7],
            partition_bound=False,
            where="synthetic",
        )
    # The whole-object anchor under a partition-bound ladder is the wrong anchor.
    with pytest.raises(AssertionError, match="anchor"):
        assert_selector_describes_thresholds(
            DERIVED_LOD_SELECTOR,
            [0.0, WHOLE_OBJECT_FINEST_ANCHOR],
            partition_bound=True,
            where="synthetic",
        )
    # Legacy units still have a ceiling.
    with pytest.raises(AssertionError, match="legacy"):
        assert_selector_describes_thresholds(
            LEGACY_LOD_SELECTOR,
            [0.0, MAX_COVERAGE_FRACTION + 1.0],
            partition_bound=False,
            where="synthetic",
        )


# ────────────────────────────────────────────────────────────────────────
# (e) The producers route through the shared helper, not their own literal
# ────────────────────────────────────────────────────────────────────────

#: The name every routed producer must reach.
_RESOLVER = "resolve_lod_ladder"
_DERIVATION_HELPERS = ("coverage_fractions", "partitioned_coverage_fractions")

#: Every module that builds a ``kind=lod`` group from a scene adder, relative to
#: the ``luxar`` package root. Kept in step with ``_LOD_GROUP_CALLERS`` below,
#: whose discovery guard is what makes a FIFTH producer impossible to add
#: silently.
_LADDER_PRODUCERS = (
    "core/group/adders/points.py",
    "core/group/adders/lines.py",
    "core/group/adders/mesh.py",
    "core/group/gsplats_pipeline/lod_dispatch.py",
)

#: Production modules that call ``Node.add_lod_group(...)`` and are NOT routed
#: through :data:`_RESOLVER`, each with the reason it is exempt. Every entry here
#: is a deliberate decision; a new key must earn one (see
#: :func:`test_no_unrouted_producer_builds_a_lod_group`).
_EXEMPT_LOD_GROUP_CALLERS = {
    # The detached-tree graft (``add_gsplats_from_file`` on a non-matrix-shaped
    # subtree): its selector answers "what does this STORED tree already claim?",
    # which has no ``explicit`` argument to branch on, so it pairs
    # ``coverage_fractions`` / ``partitioned_coverage_fractions`` with
    # ``gsplats.tree.gate_authored_selector`` instead of with ``_RESOLVER``.
    "core/group/gsplats_pipeline/from_io.py": (
        "detached-tree graft; selector comes from gsplats.tree."
        "gate_authored_selector (stored-tree question, not explicit-vs-derived)"
    ),
    # This demo hand-builds a ladder with no user-supplied explicit-threshold
    # branch; it always derives screen-area thresholds, choosing the anchor from
    # the realized BSP part count so a one-part partition remains whole-object.
    "demos/demo_ocean_currents_earth.py": (
        "hand-built demo ladder; thresholds are always derived and its anchor "
        "depends on the realized BSP part count"
    ),
    # One whole-object ladder per tract of subsampled STREAMLINES (the interim
    # form of #2679's `coarse="lines"`); thresholds always derived through
    # `coverage_fractions`, no user-supplied explicit-threshold branch.
    "demos/demo_dmri_tractography.py": (
        "hand-built demo ladder of subsampled streamlines; thresholds are "
        "always derived (whole-object anchor) via coverage_fractions"
    ),
    # Same shape for the protein-universe backdrop: 64 BSP tiles, each a
    # two-level POINTS ladder whose thresholds always come from
    # ``partitioned_coverage_fractions`` (no explicit-threshold branch exists).
    "demos/demo_esm_protein_universe.py": (
        "hand-built demo ladder (per-tile two-level points ladder); thresholds "
        "are always derived from partitioned_coverage_fractions"
    ),
}

#: The full expected set of production ``add_lod_group(...)`` call sites.
_LOD_GROUP_CALLERS = frozenset(_LADDER_PRODUCERS) | frozenset(_EXEMPT_LOD_GROUP_CALLERS)

#: How many ``add_lod_group(...)`` CALLS each of those modules makes. Asserted as
#: well as the key set, because a SECOND call added inside an already-listed
#: module changes no key and would otherwise slip past every structural guard
#: (see :func:`test_no_unrouted_producer_builds_a_lod_group`). Every listed
#: module has exactly one call today.
_EXPECTED_CALLS_PER_MODULE: Dict[str, int] = {rel: 1 for rel in _LOD_GROUP_CALLERS}


def _luxar_root() -> Path:
    import luxar

    return Path(luxar.__file__).resolve().parent


def _aliases_of(tree: ast.Module, name: str) -> set:
    """Every local name ``name`` is reachable under in ``tree``.

    ``name`` itself, plus any ``from … import name as alias`` alias. Without this
    an aliased import would make the routing guard fail on correct code.
    """
    names = {name}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == name and alias.asname:
                    names.add(alias.asname)
    return names


def _called_names(tree: ast.Module) -> set:
    """Every callee spelling in ``tree``, as a bare name.

    Accepts BOTH ``f(...)`` (``ast.Name``) and ``mod.f(...)`` /
    ``pkg.mod.f(...)`` (``ast.Attribute``, whose ``attr`` is the function name) —
    a qualified call routes the decision just as well as a bare one, so the guard
    must not mistake the idiom for the property.
    """
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called.add(node.func.attr)
    return called


def _calls_resolver(path: Path, *, filename: str) -> bool:
    """Does ``path`` call :data:`_RESOLVER` under any of its local spellings?"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
    return bool(_called_names(tree) & _aliases_of(tree, _RESOLVER))


def test_no_producer_spells_a_selector_out_for_itself() -> None:
    """No ladder producer may contain a selector STRING LITERAL.

    The rule "explicit ⇒ legacy units, derived ⇒ screen-area" was written out
    four times and enforced nowhere — four correct copies, but four places to
    drift. The four now share ``lod.group.resolve_lod_ladder``, which returns the
    selector alongside the thresholds it describes, so a producer has no reason to
    name a selector at all and naming one is the re-divergence this guard exists to
    catch. (The detached-tree writers pair their own selector through
    ``gsplats.tree.gate_authored_selector`` and are not scanned here — see the
    module docstring.) AST
    rather than grep so a mention in a comment stays free (comments are not
    ``ast.Constant`` nodes) while a real literal is caught; the same
    production-explicitness style as
    ``tests/test_zarr_compat.py::test_production_create_array_calls_pass_a_compressor``.
    """
    root = _luxar_root()
    offenders: List[str] = []
    for rel in _LADDER_PRODUCERS:
        path = root / rel
        assert path.exists(), f"ladder producer moved: {rel}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in LOD_SELECTORS
            ):
                offenders.append(f"{rel}:{node.lineno}")

    assert not offenders, (
        "these ladder producers name a LOD selector themselves instead of "
        "taking it from lod.group.resolve_lod_ladder (which pairs it with the "
        f"thresholds it describes): {offenders}. The scan is by VALUE, so it "
        f"flags ANY exact {sorted(LOD_SELECTORS)} string in these files — if "
        "yours is an unrelated coincidence (a dict key, an attr name that happens "
        "to be spelled 'coverage'), the fix is to define it as a named constant in "
        "a module OUTSIDE this list and import it, not to relax the guard. If it "
        "really is the selector, take it from resolve_lod_ladder; if it is a "
        "genuinely new producer that cannot route (see "
        "_EXEMPT_LOD_GROUP_CALLERS), move it out of _LADDER_PRODUCERS and "
        "document the reason there instead."
    )


@pytest.mark.parametrize("rel", _LADDER_PRODUCERS)
def test_every_producer_calls_the_shared_resolver(rel: str) -> None:
    """The other half of the guard above: absence of a literal is not presence of
    the helper. A producer could stamp nothing at all and inherit the
    ``add_lod_group`` legacy default over a derived ladder — exactly the
    mislabelling shape #1727 is about, and invisible to a literal scan. Each
    producer must CALL ``resolve_lod_ladder``.

    Three call spellings count, so the guard tests the property and not one
    idiom: a bare ``resolve_lod_ladder(...)``, a qualified
    ``group.resolve_lod_ladder(...)``, and an aliased import
    (``from ..lod.group import resolve_lod_ladder as _resolve``, resolved back
    through the module's ``ImportFrom`` nodes). Any of them routes the decision
    through the shared helper, which is the whole assertion.

    The same scan also forbids either derivation helper directly. Without that
    half, a second arm in an already-listed producer can call
    ``partitioned_coverage_fractions`` and hand-stamp the selector while another
    arm keeps this file-level resolver-presence check green.

    **What it does NOT prove, stated honestly.** This is a NAME check: a producer
    that defined its own local ``def resolve_lod_ladder(...)`` implementing the
    rule WRONGLY passes this guard and the two structural ones beside it. So the
    guard defends the DE-DUPLICATION convention (one shared implementation, no
    per-producer copies to drift); the behavioural assertions at the top of this
    module are what defend correctness — demonstrated: a deliberately wrong local
    copy leaves all three structural guards green and turns three behavioural
    tests red.
    """
    path = _luxar_root() / rel
    assert _calls_resolver(path, filename=rel), (
        f"{rel} builds a kind=lod group but does not call resolve_lod_ladder"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
    called = _called_names(tree)
    direct = sorted(
        helper for helper in _DERIVATION_HELPERS if called & _aliases_of(tree, helper)
    )
    assert not direct, (
        f"{rel} calls derivation helpers directly ({direct}) instead of routing "
        "every ladder arm through resolve_lod_ladder"
    )


def test_no_unrouted_producer_builds_a_lod_group() -> None:
    """DISCOVERY guard: every ``add_lod_group(...)`` call site inside the ``luxar``
    package.

    The two guards above take :data:`_LADDER_PRODUCERS` as given, so a FIFTH scene
    adder that builds a ``kind=lod`` group and stamps its own selector would trip
    nothing at all — the hardcoded list is the hole. This walks the whole package
    instead and requires the discovered set — AND each module's call COUNT — to
    equal the four routed producers plus the explicitly rationalised exemptions,
    so a new producer forces a decision whether it lands in a new file or beside
    an existing call.

    Scope is the installed ``luxar`` package root only. ``examples/`` and
    ``packages/luxar-viewer/tests/fixtures/generate_test_data.py`` also call
    ``add_lod_group``, legitimately and hand-authored; they are outside the
    library whose convention this pins, so they are out of scope here.

    By AST, not a text grep: ``add_lod_group`` also appears as a ``def`` (the
    ``Node`` method itself), in docstring examples and in prose comments, none of
    which is a call. Tests are excluded — they exercise the hand-built ladder path
    on purpose.
    """
    root = _luxar_root()
    found: Dict[str, List[int]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "/tests/" in f"/{rel}":  # covers a top-level ``tests/`` too
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except SyntaxError as exc:  # pragma: no cover - would be a broken tree
            pytest.fail(f"cannot parse {rel}: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name == "add_lod_group":
                found.setdefault(rel, []).append(node.lineno)

    assert set(found) == set(_LOD_GROUP_CALLERS), (
        "the set of production add_lod_group(...) call sites changed.\n"
        f"  discovered: { {k: v for k, v in sorted(found.items())} }\n"
        f"  expected:   {sorted(_LOD_GROUP_CALLERS)}\n"
        "A module that builds a kind=lod group also stamps its `selector`, so it "
        "must either (a) take BOTH thresholds and selector from "
        f"lod.group.{_RESOLVER} and be added to _LADDER_PRODUCERS, or (b) be added "
        "to _EXEMPT_LOD_GROUP_CALLERS with a one-line reason why the "
        "explicit-vs-derived rule does not apply to it. Removing a call site means "
        "deleting its entry. Do not delete this assertion."
    )
    # And the per-module COUNT, not just the key set: a SECOND add_lod_group(...)
    # inside an ALREADY-LISTED module leaves the keys unchanged, adds no selector
    # literal, and satisfies the routing guard through that module's EXISTING
    # routed call — so without this the discovery guard closes only the new-FILE
    # hole, not the new-CALL one.
    counts = {rel: len(lines) for rel, lines in sorted(found.items())}
    assert counts == dict(sorted(_EXPECTED_CALLS_PER_MODULE.items())), (
        "the number of production add_lod_group(...) calls per module changed.\n"
        f"  discovered: {counts}  (lines: { {k: v for k, v in sorted(found.items())} })\n"
        f"  expected:   {dict(sorted(_EXPECTED_CALLS_PER_MODULE.items()))}\n"
        "Most listed modules build exactly one kind=lod group today; "
        "_EXPECTED_CALLS_PER_MODULE carries the exceptions. If you added "
        f"another, make sure it takes BOTH thresholds and selector from lod.group."
        f"{_RESOLVER} (or belongs in _EXEMPT_LOD_GROUP_CALLERS for a stated "
        "reason), then raise that module's count in _EXPECTED_CALLS_PER_MODULE "
        "deliberately. Do not delete this assertion."
    )
    # Each routed producer must actually route; each exemption must actually be
    # unrouted (an exemption that quietly started routing is a stale rationale).
    for rel in _LADDER_PRODUCERS:
        assert _calls_resolver(root / rel, filename=rel), (
            f"{rel} is listed as ROUTED but does not call {_RESOLVER}"
        )
    for rel in _EXEMPT_LOD_GROUP_CALLERS:
        assert not _calls_resolver(root / rel, filename=rel), (
            f"{rel} is exempt from {_RESOLVER} but now calls it — move it to "
            "_LADDER_PRODUCERS and drop the exemption"
        )


def test_the_selector_constants_are_the_documented_vocabulary() -> None:
    """The two constants must BE the vocabulary, not a third opinion beside it."""
    assert {DERIVED_LOD_SELECTOR, LEGACY_LOD_SELECTOR} == set(LOD_SELECTORS)
    assert DERIVED_LOD_SELECTOR != LEGACY_LOD_SELECTOR
    # The VALUES, not just their distinctness. A constant's value is normally an
    # implementation detail, but these two are WIRE spellings: the viewer compares
    # the raw string, and in
    # ``data/scene-loader/nodes/load-lod-group-node.ts`` anything that is not
    # exactly 'screen-area' falls back to the legacy metric
    # (``attrs.selector === 'screen-area' ? 'screen-area' : 'coverage'``). So
    # SWAPPING the two constants keeps every other assertion in this module green
    # — the vocabulary set is unchanged and each arm still returns "its" constant —
    # while every derived ladder on disk would silently be read on the legacy
    # diagonal scale. Pin the spellings.
    assert (DERIVED_LOD_SELECTOR, LEGACY_LOD_SELECTOR) == ("screen-area", "coverage")


def test_resolve_lod_ladder_returns_the_constants(tmp_path) -> None:
    """The helper is what makes the constants live: it must return them, so a
    literal typo in either arm cannot pass."""
    from luxar.core.group.lod.group import resolve_lod_ladder

    def boom(n_explicit: int, n_levels: int) -> str:
        return f"{n_explicit} != {n_levels}"

    with LuxarZarrCompiler(tmp_path / "helper.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        derived, derived_selector = resolve_lod_ladder(
            None, [10, 40], scene, name="ladder", length_error=boom
        )
        authored, authored_selector = resolve_lod_ladder(
            [0.0, 4.0], [10, 40], scene, name="ladder", length_error=boom
        )
        with pytest.raises(ValueError, match="3 != 2"):
            resolve_lod_ladder(
                [0.0, 1.0, 2.0], [10, 40], scene, name="ladder", length_error=boom
            )
    assert derived_selector == DERIVED_LOD_SELECTOR
    assert derived[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
    assert authored_selector == LEGACY_LOD_SELECTOR
    assert authored == [0.0, 4.0]
