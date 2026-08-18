"""The stamped ``selector`` must describe the stamped thresholds — all four adders.

A ``kind=lod`` group writes TWO things that are really one decision: the per-child
``coverage_fraction`` thresholds, and the group-level ``selector`` that names the
UNITS those thresholds are in. Get the pairing wrong and nothing raises — both
selector spellings are individually valid — but every switch point the viewer
computes is on the wrong scale. That is not hypothetical: 209 shipped ladders
carry ``selector="coverage"`` over thresholds that look derived (issue #1727).

The rule, which :func:`luxar.core.group.lod.group.resolve_lod_ladder` is now the
single implementation of:

* an explicit ``coverage_fractions=[...]`` list is used verbatim and keeps the
  LEGACY units it was authored in (``LEGACY_LOD_SELECTOR``, ceiling
  ``MAX_COVERAGE_FRACTION``);
* otherwise the screen-area halving ladder is derived and stamped
  ``DERIVED_LOD_SELECTOR`` — finest ``WHOLE_OBJECT_FINEST_ANCHOR`` for a
  whole-object ladder, ``PARTITION_FINEST_AREA`` for a partition-bound one.

Every geometry that can build such a group is exercised here in one place —
``add_points`` / ``add_lines`` / ``add_mesh`` ``substitutive_lod=`` and
``add_gsplats_from_data`` ``lod_group=`` — because the invariant is
cross-geometry: it is exactly the thing that drifted while each adder restated
the rule for itself. Per-geometry ladder mechanics (level counts, ordering,
colours, anchors) stay in ``test_substitutive_points.py`` /
``test_substitutive_lines.py`` / ``test_gsplats.py`` /
``core/tests/test_mesh_substitutive_lod.py``.
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


def _read_ladder(store: Path, group_path: str) -> Tuple[Any, List[float]]:
    """``(selector, thresholds coarsest→finest)`` as WRITTEN, for one lod group.

    Children are ordered by their ``child_index`` attr — the canonical insertion
    order, and what ``io/_compiler/finalize/lod_backfill.py`` orders by — not by
    name: a name-sort puts ``child_10`` before ``child_2`` and would silently
    reorder any ladder of ten or more levels.
    """
    nodes = dict(read_consolidated_attrs(store))
    assert group_path in nodes, f"{group_path} not in {sorted(nodes)}"
    group = nodes[group_path]
    assert group.get("kind") == "lod", f"{group_path} is not a kind=lod group"
    children = [
        (path, attrs)
        for path, attrs in nodes.items()
        if path.startswith(f"{group_path}/child_") and "coverage_fraction" in attrs
    ]
    assert children, f"{group_path} has no ladder children"
    children.sort(key=lambda kv: int(kv[1]["child_index"]))
    return group.get("selector"), [
        float(attrs["coverage_fraction"]) for _, attrs in children
    ]


def _legacy_list(n_levels: int) -> List[float]:
    """A strictly ascending explicit ladder that is ONLY legal in legacy units.

    It tops out at ``MAX_COVERAGE_FRACTION`` = 4.0, four times the screen-area
    ceiling, so a store that mislabels it ``screen-area`` is caught by the
    invariant rather than merely by an equality check on the values.
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

#: Every module that builds a ``kind=lod`` group from a scene adder, relative to
#: the ``luxar`` package root.
_LADDER_PRODUCERS = (
    "core/group/adders/points.py",
    "core/group/adders/lines.py",
    "core/group/adders/mesh.py",
    "core/group/gsplats_pipeline/lod_dispatch.py",
)


def _luxar_root() -> Path:
    import luxar

    return Path(luxar.__file__).resolve().parent


def test_no_producer_spells_a_selector_out_for_itself() -> None:
    """No ladder producer may contain a selector STRING LITERAL.

    The rule "explicit ⇒ legacy units, derived ⇒ screen-area" was written out
    four times and enforced nowhere, which is how 209 shipped ladders ended up
    mislabelled. It now lives once, in
    ``lod.group.resolve_lod_ladder``, which returns the selector alongside the
    thresholds it describes — so a producer has no reason to name a selector at
    all, and naming one is the re-divergence this guard exists to catch. AST
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
        f"thresholds it describes): {offenders}"
    )


@pytest.mark.parametrize("rel", _LADDER_PRODUCERS)
def test_every_producer_calls_the_shared_resolver(rel: str) -> None:
    """The other half of the guard above: absence of a literal is not presence of
    the helper (a producer could stamp nothing at all, and inherit the
    ``add_lod_group`` legacy default over a derived ladder — the exact 209-store
    bug). Each producer must CALL ``resolve_lod_ladder``."""
    tree = ast.parse(
        (_luxar_root() / rel).read_text(encoding="utf-8"), filename=str(rel)
    )
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "resolve_lod_ladder" in called, (
        f"{rel} builds a kind=lod group but does not call resolve_lod_ladder"
    )


def test_the_selector_constants_are_the_documented_vocabulary() -> None:
    """The two constants must BE the vocabulary, not a third opinion beside it."""
    assert {DERIVED_LOD_SELECTOR, LEGACY_LOD_SELECTOR} == set(LOD_SELECTORS)
    assert DERIVED_LOD_SELECTOR != LEGACY_LOD_SELECTOR


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
