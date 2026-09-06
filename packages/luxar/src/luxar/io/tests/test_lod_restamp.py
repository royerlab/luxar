"""``luxar.io.lod_restamp`` — re-deriving a stored LOD ladder in place.

The pass rewrites ATTRS only, so every assertion here reads the store back from
disk (through the consolidated index the viewer actually fetches, unless a test
says otherwise) rather than trusting the returned report alone. The report is
asserted too, because it is the CLI's and a caller's only structured view.

Fixtures come from the real writers wherever the shape allows it: a compiled
`.luxar.zarr` scene with explicit legacy `coverage_fractions=[...]` lists is
byte-for-byte the shape the shipped corpus has (``selector="coverage"`` over a
ladder nobody can prove was authored), and a real `gsplat lod --recipe levels`
tree covers the standalone `.gsplats.zarr` root. The hand-built stores below are
only for shapes no current writer produces — an out-of-vocabulary selector, a
leaf with no element count, a ten-plus-level ladder — since fabricating those
through a writer would mean asserting against a fiction anyway.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import (
    close,
    consolidate,
    create_root_group,
    is_consolidated,
    open_group,
    read_consolidated_attrs,
)
from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    coverage_fractions,
)
from luxar.io import lod_restamp
from luxar.io.compiler import LuxarZarrCompiler
from luxar.io.lod_restamp import (
    HASH_RESTAMPED,
    HASH_UNCHANGED,
    HASH_UNSTAMPABLE,
    RestampedGroup,
    RestampReport,
    _verify,
    restamp_lod_store,
)
from luxar.typing_utils.constants import (
    DERIVED_LOD_SELECTOR,
    ENVIRONMENT_GROUP,
    LEGACY_LOD_SELECTOR,
)

#: The legacy ladder every compiled fixture below is authored with. Off the
#: screen-area scale entirely (4.0 is ``MAX_COVERAGE_FRACTION``), so a test that
#: sees the derived values back cannot be reading the input by accident.
LEGACY_LADDER = [0.0, 4.0]


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────


def _points(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 20, (400, 3)).astype(np.float32)


def _add_legacy_ladder(target: Any, name: str, seed: int) -> None:
    """A two-level points ladder stamped with the LEGACY selector."""
    target.add_points(
        name,
        _points(seed),
        radii=0.5,
        substitutive_lod=dict(
            levels=1, device="cpu", seed=0, coverage_fractions=LEGACY_LADDER
        ),
    )


def _add_derived_ladder(target: Any, name: str, seed: int) -> None:
    """A two-level points ladder the writer derives (already ``screen-area``)."""
    target.add_points(
        name,
        _points(seed),
        radii=0.5,
        substitutive_lod=dict(levels=1, device="cpu", seed=0),
    )


def _add_legacy_lines_ladder(target: Any, name: str, seed: int) -> None:
    """A two-level LINES ladder stamped with the LEGACY selector.

    Sized by ``n_vertices`` — the finest child is a real ``type="lines"`` node,
    so this is the only coverage of that row of ``_ELEMENT_COUNT_ATTR``. (The
    coarse level is gsplats: a substitutive lines ladder lifts to splats.)
    """
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 60, (120, 3))
    vertices = np.empty((240, 3), dtype=np.float32)
    vertices[0::2] = starts
    vertices[1::2] = starts + rng.normal(0, 5, (120, 3))
    target.add_lines(
        name,
        vertices,
        0.8,
        line_type="segments",
        substitutive_lod=dict(
            levels=1, device="cpu", seed=0, coverage_fractions=LEGACY_LADDER
        ),
    )


def _grid_mesh(side: int = 12) -> tuple:
    """A ``(vertices, faces)`` height-field grid — small, but really decimatable."""
    xs, ys = np.meshgrid(
        np.linspace(0, 30, side), np.linspace(0, 30, side), indexing="ij"
    )
    zs = np.sin(xs / 5.0) * 3.0
    vertices = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float32)
    faces = [
        (i * side + j, i * side + j + 1, (i + 1) * side + j)
        for i in range(side - 1)
        for j in range(side - 1)
    ] + [
        (i * side + j + 1, (i + 1) * side + j + 1, (i + 1) * side + j)
        for i in range(side - 1)
        for j in range(side - 1)
    ]
    return vertices, np.asarray(faces, dtype=np.uint32)


def _add_legacy_mesh_ladder(target: Any, name: str) -> None:
    """A two-level MESH ladder stamped with the LEGACY selector.

    Both levels are ``type="mesh"``, so this covers the fourth
    ``_ELEMENT_COUNT_ATTR`` row — and covers it at BOTH ladder positions, where
    ``n_vertices`` and ``n_faces`` differ (36/44 and 144/242), so reading the
    wrong one is visible in the reported counts.
    """
    vertices, faces = _grid_mesh()
    target.add_mesh(
        name,
        vertices,
        faces,
        substitutive_lod={"levels": 1, "coverage_fractions": LEGACY_LADDER},
    )


@pytest.fixture(scope="module")
def legacy_scene_template(tmp_path_factory) -> Path:
    """A compiled scene holding the ladder shapes that matter, all legacy.

    * ``pts`` — a plain whole-object ladder at the scene root.
    * ``curves`` / ``surf`` — the same shape for LINES and MESH, the two
      geometries whose element-count attr nothing else here exercises.
    * ``tiled/part_{0,1}`` — under a REAL (two-part) partition, so tile-anchored.
    * ``lonely/part_0`` — under a ONE-part partition, which is not a tiling: its
      single part's bbox IS the whole object, so it must keep the whole-object
      anchor. The shape ``to_spatial_partition`` produces routinely.

    Module-scoped because a compiler run is not free; each test copies it, so no
    test can see another's writes.
    """
    store = tmp_path_factory.mktemp("restamp") / "legacy.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        _add_legacy_ladder(scene, "pts", 0)
        _add_legacy_lines_ladder(scene, "curves", 4)
        _add_legacy_mesh_ladder(scene, "surf")
        tiled = scene.add_partition_group(
            "tiled", display_type="points", max_elements=100_000
        )
        for part in (0, 1):
            _add_legacy_ladder(tiled, f"part_{part}", part + 1)
        lonely = scene.add_partition_group(
            "lonely", display_type="points", max_elements=100_000
        )
        _add_legacy_ladder(lonely, "part_0", 3)
    return store


#: Every ``kind=lod`` path in :func:`legacy_scene_template`.
LEGACY_SCENE_LADDERS = frozenset(
    {"pts", "curves", "surf", "tiled/part_0", "tiled/part_1", "lonely/part_0"}
)


@pytest.fixture
def legacy_scene(legacy_scene_template: Path, tmp_path: Path) -> Path:
    """A per-test writable copy of :func:`legacy_scene_template`."""
    destination = tmp_path / "scene.luxar.zarr"
    shutil.copytree(legacy_scene_template, destination)
    return destination


def _attrs(store: Path) -> Dict[str, Dict[str, Any]]:
    """Every node's attrs from the CONSOLIDATED index — what the viewer reads."""
    return dict(read_consolidated_attrs(store))


def _ladder(store: Path, group_path: str) -> List[float]:
    """A lod group's thresholds, coarsest→finest by ``child_index``.

    DIRECT children only (``"/" not in`` the remainder), and the store root is
    spelled ``"/"``: a nested ladder or partition under one of these levels
    carries thresholds of its own, and folding those in would silently read a
    four-entry ladder off a two-level group.
    """
    nodes = _attrs(store)
    prefix = "" if group_path == "/" else f"{group_path}/"
    children = [
        (int(attrs["child_index"]), float(attrs["coverage_fraction"]))
        for path, attrs in nodes.items()
        if path != group_path
        and path.startswith(prefix)
        and "/" not in path[len(prefix) :]
        and "coverage_fraction" in attrs
        and "child_index" in attrs
    ]
    return [value for _, value in sorted(children)]


def _hash(store: Path) -> Any:
    return _attrs(store)["/"].get("content_hash")


def _synthetic_scene(path: Path) -> zarr.Group:
    """An empty on-disk scene root to hand-build ladders under."""
    root = create_root_group(zarr.storage.LocalStore(str(path)))
    root.attrs["type"] = "scene"
    return root


def _synthetic_ladder(
    parent: zarr.Group,
    name: str,
    children: List[Dict[str, Any]],
    *,
    selector: Optional[str] = LEGACY_LOD_SELECTOR,
) -> zarr.Group:
    """A ``kind=lod`` group whose children are given verbatim as attr dicts."""
    lod = parent.create_group(name)
    lod.attrs["type"] = "group"
    lod.attrs["kind"] = "lod"
    lod.attrs["display_type"] = "points"
    if selector is not None:
        lod.attrs["selector"] = selector
    for index, spec in enumerate(children):
        child = lod.create_group(spec.pop("name", f"child_{index}"))
        child.attrs["type"] = spec.pop("type", "points")
        child.attrs["child_index"] = spec.pop("child_index", index)
        for key, value in spec.items():
            child.attrs[key] = value
    return lod


# ────────────────────────────────────────────────────────────────────────
# The core re-derivation, per anchor
# ────────────────────────────────────────────────────────────────────────


def test_a_whole_object_ladder_is_re_derived_and_relabelled(legacy_scene: Path) -> None:
    report = restamp_lod_store(legacy_scene)

    assert _attrs(legacy_scene)["pts"]["selector"] == DERIVED_LOD_SELECTOR
    assert _ladder(legacy_scene, "pts") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]
    entry = next(g for g in report.restamped if g.path == "pts")
    assert entry.old_selector == LEGACY_LOD_SELECTOR
    assert entry.old_thresholds == LEGACY_LADDER
    assert entry.partition_bound is False
    assert report.clean


@pytest.mark.parametrize(
    "path, counts",
    [
        # lines → n_vertices (240), NOT n_segments (120); the coarse level of a
        # substitutive lines ladder is lifted to gsplats, hence n_splats there.
        ("curves", [475, 240]),
        # mesh → n_vertices at both levels, NOT n_faces (44 / 242).
        ("surf", [36, 144]),
    ],
)
def test_a_lines_and_a_mesh_ladder_are_sized_by_their_own_count_attr(
    legacy_scene: Path, path: str, counts: List[int]
) -> None:
    """Every row of ``_ELEMENT_COUNT_ATTR`` is a store this command exists for.

    A wrong attr here does not fail loudly: the count reads back as ``None`` and
    the whole ladder is silently skipped as ``unresolved-finest-count``, i.e.
    exactly the stores the migration was written for go un-migrated. So the
    reported counts are asserted, not just the derived ladder — the derivation
    reads only the ladder's LENGTH and the finest entry's positivity, so a
    ``n_segments``/``n_faces`` mix-up would leave the thresholds unchanged.
    """
    report = restamp_lod_store(legacy_scene)

    entry = next(g for g in report.restamped if g.path == path)
    assert entry.element_counts == counts
    assert _ladder(legacy_scene, path) == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]
    assert _attrs(legacy_scene)[path]["selector"] == DERIVED_LOD_SELECTOR


def test_a_multi_part_partition_binds_its_ladders_to_the_tile_anchor(
    legacy_scene: Path,
) -> None:
    """Two parts IS a tiling, so each part's ladder anchors at fills-screen."""
    report = restamp_lod_store(legacy_scene)

    for part in (0, 1):
        path = f"tiled/part_{part}"
        assert _ladder(legacy_scene, path) == [0.0, PARTITION_FINEST_AREA]
        assert next(g for g in report.restamped if g.path == path).partition_bound


def test_a_one_part_partition_keeps_the_whole_object_anchor(
    legacy_scene: Path,
) -> None:
    """The producers' rule, mirrored: a lone part's bbox IS the whole object.

    Getting this wrong is not a rounding error — the tile anchor is 2x too coarse
    in screen-area units, so the finest level would be held back until the object
    OVERFILLS the viewport (#1361's blur, re-introduced by a migration).

    Note what the fixture's input is: ``[0, 4.0]``, i.e. the LEGACY tile anchor
    (``MAX_COVERAGE_FRACTION``) under a one-part partition — the exact shape
    ``warn_one_part_partition_anchors`` reports and deliberately cannot repair.
    This pass is where it gets repaired, because this pass is the opt-in.
    """
    report = restamp_lod_store(legacy_scene)

    assert _ladder(legacy_scene, "lonely/part_0") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]
    entry = next(g for g in report.restamped if g.path == "lonely/part_0")
    assert entry.partition_bound is False


def test_a_one_part_partition_inside_a_real_tiling_is_still_tile_bound(
    tmp_path: Path,
) -> None:
    """The flag ORs in going down and is never cleared.

    ``partition(2 parts) → partition(1 part) → lod`` is still inside one tile, so
    the inner ladder is correctly tile-anchored — the same rule both gsplat tree
    writers thread (``under_partition or len(children) > 1``).
    """
    store = tmp_path / "nested.luxar.zarr"
    root = _synthetic_scene(store)
    outer = root.create_group("tiled")
    outer.attrs.update({"type": "group", "kind": "partition"})
    for part in (0, 1):
        holder = outer.create_group(f"part_{part}")
        holder.attrs.update({"type": "group", "kind": "partition"})
        _synthetic_ladder(
            holder,
            "only_part",
            [{"coverage_fraction": 0.0}, {"coverage_fraction": 4.0, "n_points": 400}],
        )
    consolidate(root)
    close(root)

    restamp_lod_store(store)

    for part in (0, 1):
        assert _ladder(store, f"tiled/part_{part}/only_part") == [
            0.0,
            PARTITION_FINEST_AREA,
        ]


def test_a_single_level_ladder_gets_the_lone_always_eligible_floor(
    tmp_path: Path,
) -> None:
    """One level has no switch to make: the ladder is just the 0.0 floor."""
    store = tmp_path / "single.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(root, "solo", [{"coverage_fraction": 2.5, "n_points": 400}])
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert _ladder(store, "solo") == [0.0]
    assert report.clean and len(report.restamped) == 1


def test_children_are_ordered_by_child_index_not_by_name(tmp_path: Path) -> None:
    """A twelve-level ladder is where a name-sort and an index-sort diverge.

    ``child_10`` sorts before ``child_2`` alphabetically, so a name-ordered pass
    would hand the finest threshold to the wrong child and read the wrong finest
    element count. Twelve levels because the compiled fixtures top out at two.
    """
    store = tmp_path / "deep.luxar.zarr"
    root = _synthetic_scene(store)
    # Created in a shuffled order too, so insertion order cannot rescue a
    # name-ordered implementation either.
    order = [7, 0, 11, 2, 10, 1, 9, 3, 8, 4, 6, 5]
    lod = root.create_group("deep")
    lod.attrs.update(
        {
            "type": "group",
            "kind": "lod",
            "display_type": "points",
            "selector": LEGACY_LOD_SELECTOR,
        }
    )
    for index in order:
        child = lod.create_group(f"child_{index}")
        child.attrs.update(
            {
                "type": "points",
                "child_index": index,
                "coverage_fraction": index / 3.0,
                # Only the FINEST count gates the derivation; make the ladder's
                # own indices legible so a mis-ordered read is obvious.
                "n_points": 100 * (index + 1),
            }
        )
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    expected = coverage_fractions([100 * (i + 1) for i in range(12)])
    assert _ladder(store, "deep") == pytest.approx(expected)
    assert report.restamped[0].element_counts == [100 * (i + 1) for i in range(12)]


def test_a_wrapper_child_is_sized_by_summing_its_leaves(tmp_path: Path) -> None:
    """A ladder level that is a plain wrapper carries no single count.

    Deliberately NOT the ``overview`` shape (see the next test): a plain
    ``type="group"`` level whose parts all render together is exactly where
    SUMMING the leaves is the right answer, and it must not drag the anchor
    along with it.
    """
    store = tmp_path / "wrapper.luxar.zarr"
    root = _synthetic_scene(store)
    lod = _synthetic_ladder(
        root,
        "mixed",
        [{"coverage_fraction": 0.0, "n_points": 50}],
    )
    fine = lod.create_group("child_1")
    fine.attrs.update({"type": "group", "child_index": 1, "coverage_fraction": 4.0})
    for part, count in enumerate((300, 400)):
        leaf = fine.create_group(f"part_{part}")
        leaf.attrs.update({"type": "points", "child_index": part, "n_points": count})
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert report.restamped[0].element_counts == [50, 700]
    assert report.restamped[0].partition_bound is False
    assert _ladder(store, "mixed") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]


def test_a_partition_CHILD_binds_the_ladder_to_the_tile_anchor(tmp_path: Path) -> None:
    """The writers' SECOND binding clause: the ``overview`` cap.

    ``[coarse_leaf, fine_partition]`` is what ``lod --recipe overview`` writes,
    and both tree writers anchor it at fills-screen via ``any(isinstance(c,
    GSplatPartition) for c in on_disk)`` — nothing about ancestry. That anchor is
    a documented product CONTRACT, not geometry (see
    ``partitioned_coverage_fractions``): the coarse cap is what the opening
    framing shows, and the fine partition is the zoom-in branch. Re-deriving it
    at the whole-object anchor halves the threshold, so the viewer pulls the
    ENTIRE dataset at the opening framing — the one cost the recipe exists to
    avoid, on the biggest stores in the corpus.
    """
    store = tmp_path / "overview_shaped.luxar.zarr"
    root = _synthetic_scene(store)
    lod = _synthetic_ladder(
        root,
        "cap",
        [{"coverage_fraction": 0.0, "n_points": 50}],
    )
    fine = lod.create_group("child_1")
    fine.attrs.update(
        {
            "type": "group",
            "kind": "partition",
            "child_index": 1,
            "coverage_fraction": 4.0,
        }
    )
    for part, count in enumerate((300, 400)):
        leaf = fine.create_group(f"part_{part}")
        leaf.attrs.update({"type": "points", "child_index": part, "n_points": count})
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert report.restamped[0].partition_bound is True
    assert _ladder(store, "cap") == [0.0, PARTITION_FINEST_AREA]


def test_the_binding_from_a_partition_child_reaches_a_nested_ladder(
    tmp_path: Path,
) -> None:
    """The writers pass ``under_partition=partition_bound`` INTO their children.

    So a ladder nested under an overview-shaped cap is inside that cap's tiling
    and must be tile-anchored too, even though no ``kind=partition`` sits between
    it and the root.
    """
    store = tmp_path / "nested_cap.luxar.zarr"
    root = _synthetic_scene(store)
    lod = _synthetic_ladder(
        root,
        "cap",
        [{"coverage_fraction": 0.0, "n_points": 50}],
    )
    fine = lod.create_group("child_1")
    fine.attrs.update(
        {
            "type": "group",
            "kind": "partition",
            "child_index": 1,
            "coverage_fraction": 4.0,
        }
    )
    part = fine.create_group("part_0")
    part.attrs.update({"type": "group", "child_index": 0})
    _synthetic_ladder(
        part,
        "inner",
        [{"coverage_fraction": 0.0}, {"coverage_fraction": 4.0, "n_points": 400}],
    )
    consolidate(root)
    close(root)

    restamp_lod_store(store)

    assert _ladder(store, "cap") == [0.0, PARTITION_FINEST_AREA]
    assert _ladder(store, "cap/child_1/part_0/inner") == [0.0, PARTITION_FINEST_AREA]


def test_a_nested_lod_child_is_sized_by_its_own_FINEST_level(tmp_path: Path) -> None:
    """A nested ladder's levels are ALTERNATIVES, so they must not be summed.

    Only one of them is ever drawn. Summing reports a level count that exists
    nowhere, in the very audit trail an operator uses to sanity-check a rewrite
    they cannot undo.
    """
    store = tmp_path / "lod_in_lod.luxar.zarr"
    root = _synthetic_scene(store)
    outer = _synthetic_ladder(
        root,
        "outer",
        [{"coverage_fraction": 0.0, "n_points": 50}],
    )
    inner = _synthetic_ladder(
        outer,
        "child_1",
        [
            {"coverage_fraction": 0.0, "n_points": 100},
            {"coverage_fraction": 4.0, "n_points": 400},
        ],
    )
    inner.attrs["child_index"] = 1
    inner.attrs["coverage_fraction"] = 4.0
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    outer_entry = next(g for g in report.restamped if g.path == "outer")
    assert outer_entry.element_counts == [50, 400], "not 500 — the levels are not parts"
    assert _ladder(store, "outer") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]
    assert _ladder(store, "outer/child_1") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]


# ────────────────────────────────────────────────────────────────────────
# Skips — each one loudly reported, never silent
# ────────────────────────────────────────────────────────────────────────


def test_a_screen_area_group_is_skipped_in_a_mixed_store(tmp_path: Path) -> None:
    """A store part-way through the migration must not be re-derived twice."""
    store = tmp_path / "mixed.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        _add_legacy_ladder(scene, "old", 0)
        _add_derived_ladder(scene, "new", 1)
    before = _attrs(store)["new"]

    report = restamp_lod_store(store)

    assert [g.path for g in report.restamped] == ["old"]
    assert [g.path for g in report.already_current] == ["new"]
    assert _attrs(store)["new"] == before, "an up-to-date ladder must not be touched"
    assert report.clean


def test_an_unsupported_selector_is_reported_and_left_alone(tmp_path: Path) -> None:
    """``pixel_size`` (the pre-v3.2 gsplats spelling) is not convertible here."""
    store = tmp_path / "legacy_selector.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "old",
        [{"coverage_fraction": 0.0}, {"coverage_fraction": 4.0, "n_points": 400}],
        selector="pixel_size",
    )
    consolidate(root)
    close(root)
    before = _hash(store)

    report = restamp_lod_store(store)

    assert not report.restamped
    assert [g.path for g in report.unsupported] == ["old"]
    assert "migrate-format" in report.unsupported[0].detail
    assert not report.clean, "the caller must be able to see this in the exit code"
    assert _attrs(store)["old"]["selector"] == "pixel_size"
    assert _ladder(store, "old") == [0.0, 4.0]
    assert _hash(store) == before


def test_an_absent_selector_is_treated_as_legacy(tmp_path: Path) -> None:
    """``add_lod_group``'s historical default wrote no selector at all."""
    store = tmp_path / "no_selector.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "old",
        [{"coverage_fraction": 0.0}, {"coverage_fraction": 4.0, "n_points": 400}],
        selector=None,
    )
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert report.restamped[0].old_selector is None
    assert _attrs(store)["old"]["selector"] == DERIVED_LOD_SELECTOR
    assert _ladder(store, "old") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]


def test_a_descending_stored_ladder_is_refused_rather_than_inverted(
    tmp_path: Path,
) -> None:
    """The store's own two signals disagree, so the pass must not pick one.

    Neither ``child_index`` nor a ``child_<i>`` name is present here, so
    ``_lod_children`` falls back to sorted NAME — which puts ``level_hi``
    (10,000 elements, stored at the finest threshold 4.0) before ``level_lo``
    (100 elements, stored at 0.0). Writing the derived ascending ladder onto
    that order inverts it: the 100-element level would show at half-screen and
    the 10,000-element one only once the object is tiny. The incoming ladder
    says so plainly by descending, so the group is refused and named.
    """
    store = tmp_path / "descending.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "backwards",
        [
            {"name": "level_hi", "coverage_fraction": 4.0, "n_points": 10_000},
            {"name": "level_lo", "coverage_fraction": 0.0, "n_points": 100},
        ],
    )
    # No child_index anywhere: that is what forces the name-sorted fallback.
    lod = root["backwards"]
    for name in lod.group_keys():
        del lod[str(name)].attrs["child_index"]
    consolidate(root)
    close(root)
    before = _attrs(store)

    report = restamp_lod_store(store)

    assert not report.restamped
    assert [g.path for g in report.unresolved] == ["backwards"]
    assert report.unresolved[0].reason == "descending-ladder"
    assert not report.clean
    assert _attrs(store) == before, "a refused group must be left exactly as found"


def test_a_kind_lod_group_with_no_children_at_all_is_unresolved(
    tmp_path: Path,
) -> None:
    """An empty ladder is broken, not "already current" — the exit code must say so."""
    store = tmp_path / "childless.luxar.zarr"
    root = _synthetic_scene(store)
    lod = root.create_group("empty")
    lod.attrs.update(
        {
            "type": "group",
            "kind": "lod",
            "display_type": "points",
            "selector": LEGACY_LOD_SELECTOR,
        }
    )
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert not report.restamped and not report.already_current
    assert [g.reason for g in report.unresolved] == ["no-children"]
    assert "no child groups at all" in report.unresolved[0].detail
    assert not report.clean


def test_unclassifiable_children_get_their_own_message(tmp_path: Path) -> None:
    """ "No ladder children" and "children nothing can classify" are different bugs.

    Reporting the second as the first sends the operator looking for a missing
    write when the real defect is a missing ``type`` stamp on children that are
    right there.
    """
    store = tmp_path / "untyped.luxar.zarr"
    root = _synthetic_scene(store)
    lod = root.create_group("untyped")
    lod.attrs.update(
        {
            "type": "group",
            "kind": "lod",
            "display_type": "points",
            "selector": LEGACY_LOD_SELECTOR,
        }
    )
    for index in (0, 1):
        child = lod.create_group(f"child_{index}")
        child.attrs.update({"child_index": index, "coverage_fraction": float(index)})
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert [g.reason for g in report.unresolved] == ["unclassifiable-children"]
    detail = report.unresolved[0].detail
    assert "child_0, child_1" in detail and "'type'" in detail
    assert not report.clean


def test_a_ladder_child_the_node_filter_drops_is_refused_not_half_written(
    tmp_path: Path,
) -> None:
    """A rung the filter cannot see must refuse the GROUP, not be skipped over.

    ``_child_nodes`` keeps only children carrying a scene-node ``type``, so a
    middle rung without one vanishes from the resolved ladder while keeping its
    own stored ``coverage_fraction``. Deriving over the survivors then writes a
    PARTIAL ladder — measured on this exact store, ``[child_0=0.0,
    child_1=2.0, child_2=0.5]`` under ``selector="screen-area"``: non-monotonic,
    with a 2.0 rung above the screen-area ceiling of 1.0 that the viewer's
    clipped area metric can never satisfy. The viewer re-sorts it with a warning
    and thereby SWAPS the two finest levels, stranding the real finest one. No
    Python producer writes a type-less lod child, but hand-authored and
    third-party stores are exactly what this command is for.
    """
    store = tmp_path / "holed.luxar.zarr"
    root = _synthetic_scene(store)
    lod = _synthetic_ladder(
        root,
        "holed",
        [
            {"coverage_fraction": 0.0, "n_points": 100},
            {"coverage_fraction": 2.0, "n_points": 200},
            {"coverage_fraction": 4.0, "n_points": 300},
        ],
    )
    del lod["child_1"].attrs["type"]
    consolidate(root)
    close(root)
    before = _attrs(store)

    report = restamp_lod_store(store)

    assert not report.restamped
    assert [g.reason for g in report.unresolved] == ["unclassifiable-ladder-child"]
    assert "child_1" in report.unresolved[0].detail
    assert not report.clean
    assert _attrs(store) == before, "a refused group must be left exactly as found"


def test_an_unresolvable_finest_count_is_a_reported_skip(tmp_path: Path) -> None:
    """Never fabricate a positive count.

    ``coverage_fractions`` raises on a finest count of 0 precisely so a ladder
    whose finest level was culled fails loudly. Passing an unknown count as 0
    would trip that guard on a healthy store; passing it as 1 would DISABLE the
    guard on a broken one. So the group is skipped and named.
    """
    store = tmp_path / "countless.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "unknown",
        [{"coverage_fraction": 0.0, "n_points": 50}, {"coverage_fraction": 4.0}],
    )
    consolidate(root)
    close(root)
    before = _hash(store)

    report = restamp_lod_store(store)

    assert not report.restamped
    assert [g.path for g in report.unresolved] == ["unknown"]
    assert report.unresolved[0].reason == "unresolved-finest-count"
    assert not report.clean
    assert _ladder(store, "unknown") == [0.0, 4.0]
    assert _hash(store) == before


def test_an_empty_finest_level_is_a_reported_skip(tmp_path: Path) -> None:
    """A finest level with 0 elements is a broken ladder, not one to re-anchor."""
    store = tmp_path / "empty_finest.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "broken",
        [
            {"coverage_fraction": 0.0, "n_points": 50},
            {"coverage_fraction": 4.0, "n_points": 0},
        ],
    )
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert [g.path for g in report.unresolved] == ["broken"]
    assert report.unresolved[0].reason == "empty-finest-level"
    assert not report.clean


def test_an_unresolvable_COARSER_count_does_not_block_the_re_derivation(
    tmp_path: Path,
) -> None:
    """Only the finest count is consumed, so a missing coarse one is harmless.

    The derivation reads ``element_counts`` for its LENGTH plus the finest
    entry's positivity; a coarse level whose count the store never recorded
    cannot affect either. It is reported honestly as ``None`` rather than
    silently becoming a real-looking number.
    """
    store = tmp_path / "partial_counts.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "partial",
        [
            {"coverage_fraction": 0.0},
            {"coverage_fraction": 2.0},
            {"coverage_fraction": 4.0, "n_points": 400},
        ],
    )
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert report.clean
    assert report.restamped[0].element_counts == [None, None, 400]
    assert _ladder(store, "partial") == [0.0, 0.25, WHOLE_OBJECT_FINEST_ANCHOR]


# ────────────────────────────────────────────────────────────────────────
# Cache invalidation, idempotency, and the no-op store
# ────────────────────────────────────────────────────────────────────────


def test_a_real_run_moves_the_content_hash(legacy_scene: Path) -> None:
    """An attrs-only change must still invalidate a warm viewer cache."""
    before = _hash(legacy_scene)

    report = restamp_lod_store(legacy_scene)

    after = _hash(legacy_scene)
    assert before and after and before != after
    assert report.content_hash == after


def test_a_real_run_keeps_the_baked_environment_current(legacy_scene: Path) -> None:
    """An attrs-only scene rewrite must keep its excluded sidecar accepted."""
    before = _hash(legacy_scene)
    root = open_group(legacy_scene, mode="r+")
    environment = root.require_group(ENVIRONMENT_GROUP)
    environment.attrs["scene_content_hash"] = before
    consolidate(root)
    close(root)

    report = restamp_lod_store(legacy_scene)

    after = _hash(legacy_scene)
    assert report.content_hash == after
    root = open_group(legacy_scene, mode="r")
    try:
        assert dict(root[ENVIRONMENT_GROUP].attrs)["scene_content_hash"] == after
    finally:
        close(root)


def test_a_second_run_changes_nothing_at_all(legacy_scene: Path) -> None:
    """Idempotency down to the hash: re-running must not re-publish the store."""
    restamp_lod_store(legacy_scene)
    after_first = _attrs(legacy_scene)

    report = restamp_lod_store(legacy_scene)

    assert not report.restamped
    assert len(report.already_current) == len(LEGACY_SCENE_LADDERS)
    assert report.content_hash is None
    assert report.content_hash_status == HASH_UNCHANGED
    assert _attrs(legacy_scene) == after_first


def test_a_store_with_no_lod_group_is_left_completely_alone(tmp_path: Path) -> None:
    """No ladder means no change — and specifically NO spurious hash restamp."""
    store = tmp_path / "plain.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", _points(0), radii=0.5)
    before = _attrs(store)

    report = restamp_lod_store(store)

    assert not report.restamped and report.clean
    assert report.content_hash is None
    assert _attrs(store) == before


def test_a_store_with_no_digest_to_move_says_so(tmp_path: Path) -> None:
    """``content_hash is None`` means two opposite things; the report must not.

    A bare ``kind=lod`` root with no ``content_hash`` and no scene ``type`` IS a
    Luxar node — so the ladder is restamped — but there is no digest to move, and
    a warm viewer cache will therefore keep serving the old ladder. That is the
    caller's problem to know about, and it is indistinguishable from "nothing
    changed" if all they get back is ``None``.
    """
    store = tmp_path / "hashless.gsplats.zarr"
    root = create_root_group(zarr.storage.LocalStore(str(store)))
    root.attrs.update(
        {
            "type": "group",
            "kind": "lod",
            "display_type": "points",
            "selector": LEGACY_LOD_SELECTOR,
        }
    )
    for index, count in enumerate((100, 400)):
        child = root.create_group(f"child_{index}")
        child.attrs.update(
            {
                "type": "points",
                "child_index": index,
                "coverage_fraction": 4.0 * index,
                "n_points": count,
            }
        )
    consolidate(root)
    close(root)

    report = restamp_lod_store(store)

    assert [g.path for g in report.restamped] == ["/"]
    assert report.content_hash is None
    assert report.content_hash_status == HASH_UNSTAMPABLE
    assert report.clean is False
    assert _ladder(store, "/") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]


def test_a_rewrite_no_warm_cache_can_see_is_not_a_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unstampable digest is a result the caller must act on, not a warning.

    A ``kind=partition`` ROOT is a Luxar node — ``optimise._is_luxar_store``
    admits it via ``_LUXAR_NODE_KINDS`` — but carries neither the scene ``type``
    that selects the value-hashing branch nor a ``.gsplats.zarr``
    ``content_hash`` to re-stamp. So both ladders are rewritten and NOTHING moves
    that a client keys on. At zarr format 2, which is what the legacy corpus this
    command exists for is written in, even the viewer's documented
    ``zattrs-hash`` fallback digests the raw root ``.zattrs`` bytes — and editing
    a child's ladder does not move those, as asserted below. Exiting 0 here would
    report success on precisely the stale-cache failure the pass exists to
    prevent; the only fix is to republish under a new URL prefix, which the
    caller can only do if the run says so.
    """
    from luxar import _zarr_compat

    monkeypatch.setattr(_zarr_compat, "ZARR_FORMAT", 2)
    store = tmp_path / "parts.luxar.zarr"
    root = create_root_group(zarr.storage.LocalStore(str(store)))
    root.attrs.update({"kind": "partition", "display_type": "points"})
    for part in (0, 1):
        _synthetic_ladder(
            root,
            f"part_{part}",
            [
                {"coverage_fraction": 0.0, "n_points": 100},
                {"coverage_fraction": 4.0, "n_points": 400},
            ],
        )
    consolidate(root)
    close(root)
    assert int(open_group(store, mode="r").metadata.zarr_format) == 2
    root_document = (store / ".zattrs").read_bytes()

    report = restamp_lod_store(store)

    assert {g.path for g in report.restamped} == {"part_0", "part_1"}
    assert _ladder(store, "part_0") == [0.0, PARTITION_FINEST_AREA]
    assert report.content_hash is None
    assert report.content_hash_status == HASH_UNSTAMPABLE
    assert report.clean is False, "a rewrite no cache can see must not report success"
    assert not (report.unsupported or report.unresolved or report.residual), (
        "clean must be False because of the unstampable digest alone"
    )
    assert (store / ".zattrs").read_bytes() == root_document, (
        "the zattrs-hash fallback digests exactly these bytes, and they did not "
        "move — which is why the run has to report the problem itself"
    )


def test_a_dry_run_reports_everything_and_writes_nothing(legacy_scene: Path) -> None:
    before = _attrs(legacy_scene)

    report = restamp_lod_store(legacy_scene, dry_run=True)

    assert report.dry_run
    assert {g.path for g in report.restamped} == set(LEGACY_SCENE_LADDERS)
    assert next(g for g in report.restamped if g.path == "tiled/part_0").new_thresholds
    assert _attrs(legacy_scene) == before, "a dry run must not touch the store"
    assert report.content_hash is None
    assert report.content_hash_status == HASH_UNCHANGED


# ────────────────────────────────────────────────────────────────────────
# --group
# ────────────────────────────────────────────────────────────────────────


def test_group_restricts_the_pass_to_the_named_ladders(legacy_scene: Path) -> None:
    report = restamp_lod_store(legacy_scene, groups=["pts", "tiled/part_1"])

    assert {g.path for g in report.restamped} == {"pts", "tiled/part_1"}
    assert _attrs(legacy_scene)["tiled/part_0"]["selector"] == LEGACY_LOD_SELECTOR
    assert _ladder(legacy_scene, "tiled/part_0") == LEGACY_LADDER
    assert _ladder(legacy_scene, "tiled/part_1") == [0.0, PARTITION_FINEST_AREA]


def test_group_accepts_a_leading_slash(legacy_scene: Path) -> None:
    """A user copying a path out of a tree listing should not have to guess."""
    report = restamp_lod_store(legacy_scene, groups=["/pts"])

    assert [g.path for g in report.restamped] == ["pts"]


def test_an_unmatched_group_is_an_error_that_writes_nothing(
    legacy_scene: Path,
) -> None:
    """And it fires BEFORE the walk, so the store is not left half-restamped."""
    before = _attrs(legacy_scene)

    with pytest.raises(ValueError, match="not a kind=lod group"):
        restamp_lod_store(legacy_scene, groups=["pts", "nope/at/all"])

    assert _attrs(legacy_scene) == before


def test_a_group_that_exists_but_is_not_a_ladder_is_an_error(
    legacy_scene: Path,
) -> None:
    """``tiled`` is a real node — a partition — but has no ladder to restamp."""
    with pytest.raises(ValueError, match="tiled"):
        restamp_lod_store(legacy_scene, groups=["tiled"])


# ────────────────────────────────────────────────────────────────────────
# Refusals
# ────────────────────────────────────────────────────────────────────────


def test_a_compressed_store_is_refused_with_the_way_out(
    legacy_scene: Path, tmp_path: Path
) -> None:
    archive = tmp_path / "scene.zarr.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for item in legacy_scene.rglob("*"):
            if item.is_file():
                zf.write(item, item.relative_to(legacy_scene).as_posix())

    with pytest.raises(ValueError, match="unpack"):
        restamp_lod_store(archive)


def test_a_non_luxar_zarr_is_refused(tmp_path: Path) -> None:
    store = tmp_path / "foreign.zarr"
    root = create_root_group(zarr.storage.LocalStore(str(store)))
    root.attrs["something"] = "else"
    consolidate(root)
    close(root)

    with pytest.raises(ValueError, match="does not look like a Luxar"):
        restamp_lod_store(store)


def test_a_missing_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uncompressed"):
        restamp_lod_store(tmp_path / "nowhere.luxar.zarr")


# ────────────────────────────────────────────────────────────────────────
# The standalone .gsplats.zarr root (a different content-hash branch)
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def legacy_gsplats_template(tmp_path_factory) -> tuple:
    """A REAL ``gsplat lod --recipe levels`` tree, downgraded to legacy stamps.

    Written by the production writer, then walked back to what an older Luxar
    put on disk (``selector="coverage"`` over the same ladder, rehashed the way
    that writer would have) — which is precisely how the shipped corpus got
    there. Nothing is mocked: the store is real, only its stamps are aged.

    Returns ``(path, pristine_hash)``: the hash the CURRENT writer stamped
    before the ageing, so a test can assert the restamp lands back on it and
    thereby that the digest really does cover these attrs.
    """
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

    rng = np.random.default_rng(0)
    n = 400
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    data = GSplatData(
        centers=rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )
    tree = build_recipe(
        data,
        "levels",
        # No stream ladders: this fixture is about the substitutive levels'
        # selector stamp, and a per-level additive branch only adds nodes.
        RecipeParams(
            compression_factor=4, levels=2, device="cpu", additive_ladders=False
        ),
    )
    from luxar.gsplats.io.save_gsplats import _stamp_content_hash

    store = tmp_path_factory.mktemp("restamp_gsplats") / "levels.gsplats.zarr"
    # The same write path `luxar gsplat lod` takes for a matrix (levels) recipe.
    tree.save(store, ordering="none")
    pristine = _hash(store)

    # Age it: legacy selector over a legacy-scaled ladder, rehashed and
    # re-consolidated exactly as the writer of the day would have left it.
    root = open_group(store, mode="r+")
    assert root.attrs["selector"] == DERIVED_LOD_SELECTOR, "writer changed shape"
    root.attrs["selector"] = LEGACY_LOD_SELECTOR
    for name in root.group_keys():
        child = root[str(name)]
        if "coverage_fraction" in dict(child.attrs):
            child.attrs["coverage_fraction"] = 4.0 * float(
                child.attrs["coverage_fraction"]
            )
    _stamp_content_hash(root)
    consolidate(root)
    close(root)
    return store, pristine


@pytest.fixture
def legacy_gsplats(legacy_gsplats_template: tuple, tmp_path: Path) -> tuple:
    template, pristine = legacy_gsplats_template
    destination = tmp_path / "levels.gsplats.zarr"
    shutil.copytree(template, destination)
    return destination, pristine


def test_a_standalone_gsplats_root_is_restamped_and_rehashed(
    legacy_gsplats: tuple,
) -> None:
    """The root IS the lod group here, and the hash branch is the gsplats one.

    ``_restamp_content_hash`` dispatches on the root: a scene gets the value
    walk, a standalone ``.gsplats.zarr`` its own metadata-only stamp. Landing
    back on the PRISTINE digest is the strong form of "the stamp covers these
    attrs" — a digest blind to them would have matched the aged store instead.
    """
    store, pristine = legacy_gsplats
    aged = _hash(store)
    assert aged != pristine, "the ageing must move the digest, or nothing is proven"

    report = restamp_lod_store(store)

    assert [g.path for g in report.restamped] == ["/"]
    assert report.clean and not report.residual
    attrs = _attrs(store)
    assert attrs["/"]["selector"] == DERIVED_LOD_SELECTOR
    assert attrs["/"]["content_hash"] == pristine != aged
    finest = max(
        (a for a in attrs.values() if "coverage_fraction" in a),
        key=lambda a: int(a["child_index"]),
    )
    assert finest["coverage_fraction"] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)


def test_the_gsplats_root_can_be_named_as_a_group(legacy_gsplats: tuple) -> None:
    store, _ = legacy_gsplats

    report = restamp_lod_store(store, groups=["/"])

    assert [g.path for g in report.restamped] == ["/"]


# ────────────────────────────────────────────────────────────────────────
# The verifier itself must not be vacuous
# ────────────────────────────────────────────────────────────────────────


def test_the_verifier_reports_a_ladder_that_did_not_land(legacy_scene: Path) -> None:
    """``_verify`` is what makes a silent consolidation mistake loud, so it has
    to actually fail when the store disagrees with the report.

    Driven directly with a report claiming thresholds the store does not carry —
    the observable shape of a stale index — and both readers must object: the
    per-node documents AND the consolidated index the viewer fetches.
    """
    restamp_lod_store(legacy_scene)
    lying = RestampReport(path=str(legacy_scene), dry_run=False)
    lying.restamped = [
        RestampedGroup(
            path="pts",
            partition_bound=False,
            old_selector=LEGACY_LOD_SELECTOR,
            old_thresholds=LEGACY_LADDER,
            new_thresholds=[0.0, 0.125],
            element_counts=[100, 400],
        )
    ]

    residual = _verify(legacy_scene, lying)

    assert any("the store" in message for message in residual)
    assert any("consolidated index" in message for message in residual)


def test_the_verifier_is_silent_on_a_store_that_agrees(legacy_scene: Path) -> None:
    report = restamp_lod_store(legacy_scene)

    assert _verify(legacy_scene, report) == []


def test_a_stale_index_makes_the_run_dirty_end_to_end(
    legacy_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verify → report → exit-code chain, driven by a REAL failure.

    Not a hand-built lying report: the store is genuinely left with a stale
    consolidated index (``consolidate`` neutered for this one run), which is the
    exact silent failure the verifier exists for — the per-node documents are
    correct while the only thing the viewer ever fetches is not. Both links must
    hold: ``_verify`` has to be called at all, and ``clean`` has to read its
    result.
    """
    monkeypatch.setattr(lod_restamp, "consolidate", lambda group: None)

    report = restamp_lod_store(legacy_scene)

    assert report.restamped, "the run must actually have rewritten something"
    assert report.residual, "a stale index has to surface as a residual"
    assert any("consolidated index" in message for message in report.residual)
    assert report.clean is False
    assert not (report.unsupported or report.unresolved), (
        "clean must be False because of the residual alone"
    )


# ────────────────────────────────────────────────────────────────────────
# Atomicity: a failed write leaves the store exactly as it was
# ────────────────────────────────────────────────────────────────────────


def _snapshot_tree(store: Path) -> Dict[str, bytes]:
    """Every file in the store, by relative path → bytes."""
    return {
        str(item.relative_to(store)): item.read_bytes()
        for item in sorted(store.rglob("*"))
        if item.is_file()
    }


def test_a_failed_write_rolls_the_whole_store_back(
    legacy_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn ladder is worse than no migration, so a failure must unwind.

    Without a rollback the store is left with (a) some ladders restamped, (b) the
    failing one TORN — a screen-area threshold under ``selector="coverage"``, the
    thresholds and the selector disagreeing about their units, which nothing
    downstream can detect — and (c) a consolidated index describing neither.
    ``optimise`` is all-or-nothing for exactly this reason; this pass writes in
    place and so has to unwind instead of staging.
    """
    before = _snapshot_tree(legacy_scene)
    assert is_consolidated(legacy_scene)
    calls: List[str] = []
    real_apply = lod_restamp._apply_one

    def explode(root: Any, plan: Any, undo: Any, cache: Any) -> None:
        calls.append(plan.entry.path)
        if len(calls) == 2:
            raise PermissionError("simulated read-only child directory")
        real_apply(root, plan, undo, cache)

    monkeypatch.setattr(lod_restamp, "_apply_one", explode)

    with pytest.raises(PermissionError, match="simulated"):
        restamp_lod_store(legacy_scene)

    assert len(calls) == 2, "the failure must land part-way through, not first"
    assert _snapshot_tree(legacy_scene) == before, "every byte must be restored"
    assert is_consolidated(legacy_scene), "the store must keep exactly one index"
    for path in LEGACY_SCENE_LADDERS:
        assert _attrs(legacy_scene)[path]["selector"] == LEGACY_LOD_SELECTOR
        assert _ladder(legacy_scene, path) == LEGACY_LADDER


def test_the_rollback_error_says_what_it_undid(
    legacy_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The raised error has to name the state it left behind, or nobody can act."""
    real_apply = lod_restamp._apply_one
    calls: List[str] = []

    def explode(root: Any, plan: Any, undo: Any, cache: Any) -> None:
        calls.append(plan.entry.path)
        if len(calls) == 2:
            raise OSError("disk gone")
        real_apply(root, plan, undo, cache)

    monkeypatch.setattr(lod_restamp, "_apply_one", explode)

    with pytest.raises(OSError) as caught:
        restamp_lod_store(legacy_scene)

    notes = getattr(caught.value, "__notes__", [])
    assert any("rolled back" in note for note in notes), notes
    assert any("exactly as it was" in note for note in notes), notes


def test_a_failure_in_the_hash_pass_also_rolls_back(
    legacy_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ladders are written BEFORE the hash walk, which reads every array.

    That walk is the step most likely to fail on a real store (a truncated chunk,
    a permissions problem), and by then every ladder is already rewritten — so it
    has to be inside the same unwind.
    """
    before = _snapshot_tree(legacy_scene)

    def explode(root: Any) -> str:
        raise RuntimeError("simulated hash failure")

    monkeypatch.setattr(lod_restamp, "_restamp_content_hash", explode)

    with pytest.raises(RuntimeError, match="simulated hash failure"):
        restamp_lod_store(legacy_scene)

    assert _snapshot_tree(legacy_scene) == before
    assert is_consolidated(legacy_scene)


def test_a_rollback_removes_a_threshold_that_was_absent_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restoring an attr that never existed means DELETING it, not zeroing it.

    A partially-authored ladder (a coarse level with no stored threshold) is a
    real shape — it is what ``test_an_unresolvable_COARSER_count_...`` covers —
    so the undo has to handle "absent" as a value.
    """
    store = tmp_path / "partial.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "partial",
        [{}, {"coverage_fraction": 4.0, "n_points": 400}],
    )
    consolidate(root)
    close(root)
    before = _snapshot_tree(store)
    assert "coverage_fraction" not in _attrs(store)["partial/child_0"]

    def explode(root: Any) -> str:
        raise RuntimeError("simulated failure after the ladder was written")

    monkeypatch.setattr(lod_restamp, "_restamp_content_hash", explode)

    with pytest.raises(RuntimeError, match="simulated failure"):
        restamp_lod_store(store)

    assert "coverage_fraction" not in _attrs(store)["partial/child_0"]
    assert _snapshot_tree(store) == before


def _node_attrs(store: Path) -> Dict[str, Dict[str, Any]]:
    """Every node's attrs from the per-node DOCUMENTS, index bypassed.

    The counterpart of :func:`_attrs`, for the stores below where the index is
    either absent by design or the very thing under test.
    """
    out: Dict[str, Dict[str, Any]] = {}

    def walk(group: zarr.Group) -> None:
        out[group.path or "/"] = dict(group.attrs)
        for name in group.group_keys():
            walk(group[str(name)])

    root = open_group(store, mode="r")
    try:
        walk(root)
    finally:
        close(root)
    return out


def test_a_rollback_that_never_wrote_the_root_leaves_the_index_alone(
    legacy_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failing at attr write #1 must not put the store's index at risk.

    At that instant the ROOT document is still pristine, so the format-3
    consolidated index — which lives inside it, and which the viewer builds its
    whole scene graph from with no directory-walk fallback — is still valid. A
    rollback that restamps the ``content_hash`` writes the root, destroys that
    index, and then depends on ``consolidate`` succeeding to put it back: a
    failure needing no root write at all is upgraded into a published store that
    loads as an EMPTY scene. Nothing about the recovery may touch the root, and
    nothing may re-consolidate.
    """
    assert (legacy_scene / "zarr.json").exists(), (
        "this is the format-3 hazard: v2 keeps its index in a separate document"
    )
    before = _snapshot_tree(legacy_scene)
    assert is_consolidated(legacy_scene)

    real_write = lod_restamp._write_attr
    writes: List[str] = []

    def explode(node: Any, path: str, key: str, value: Any, undo: Any) -> None:
        real_write(node, path, key, value, undo)
        writes.append(path)
        # The SECOND threshold of the first ladder: still a child group, so the
        # root document is untouched, but a write that really moved a value.
        if len(writes) == 2:
            raise PermissionError("simulated read-only child directory")

    def no_index(root: Any) -> None:
        raise OSError("simulated consolidate failure")

    monkeypatch.setattr(lod_restamp, "_write_attr", explode)
    monkeypatch.setattr(lod_restamp, "consolidate", no_index)

    with pytest.raises(PermissionError, match="simulated"):
        restamp_lod_store(legacy_scene)

    # The first ladder's two child thresholds, whichever ladder the walk reached
    # first: both are nested groups, so the root document is still untouched.
    assert len(writes) == 2 and all("/" in path for path in writes), writes
    assert is_consolidated(legacy_scene), (
        "the index was valid when the write failed and must still be"
    )
    assert _snapshot_tree(legacy_scene) == before, "every byte must be restored"


def test_a_failed_run_restores_a_digest_it_could_not_have_recomputed(
    legacy_scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rollback RESTORES the store's digests; it does not re-derive them.

    A recompute only lands on the stored value when the stored value already was
    this walk's answer. A legacy store hashed by an older walk, a hand-edited one,
    or a scene whose inner groups carry no per-group hash — all of them shapes
    this command exists to migrate — would instead have their digests silently
    rewritten by a FAILED run, under a headline promising the store is exactly as
    it was.
    """

    def strip(group: zarr.Group) -> None:
        for name in list(group.group_keys()):
            child = group[str(name)]
            if "content_hash" in dict(child.attrs):
                del child.attrs["content_hash"]
            strip(child)

    root = open_group(legacy_scene, mode="r+")
    strip(root)
    root.attrs["content_hash"] = "LEGACY-DIGEST"
    root.require_group(ENVIRONMENT_GROUP).attrs["scene_content_hash"] = "LEGACY-DIGEST"
    consolidate(root)
    close(root)
    stale = _node_attrs(legacy_scene)
    assert stale["/"]["content_hash"] == "LEGACY-DIGEST"
    assert stale[ENVIRONMENT_GROUP]["scene_content_hash"] == "LEGACY-DIGEST"
    assert "content_hash" not in stale["pts"]

    def no_index(root: Any) -> None:
        raise OSError("simulated consolidate failure")

    # Fails AFTER the hash pass, so the digests really were overwritten and the
    # rollback has to put every one of them back.
    monkeypatch.setattr(lod_restamp, "consolidate", no_index)

    with pytest.raises(OSError, match="simulated consolidate failure"):
        restamp_lod_store(legacy_scene)

    after = _node_attrs(legacy_scene)
    assert after["/"]["content_hash"] == "LEGACY-DIGEST"
    assert [path for path, attrs in after.items() if "content_hash" in attrs] == ["/"]
    assert after == stale, "an attrs-level rollback has to be exact everywhere"


def test_a_clean_run_does_not_give_an_unconsolidated_store_an_index(
    tmp_path: Path,
) -> None:
    """An index is rebuilt, never introduced.

    ``is_consolidated`` is ``batch-fit``'s "this tile finished" sentinel, so
    consolidating a store that arrived without one would mark an interrupted tile
    complete. The rollback path already knows this (``was_consolidated``); the
    success path must agree — and the verifier must not then report the absent
    index it deliberately did not write.
    """
    store = tmp_path / "interrupted.luxar.zarr"
    root = _synthetic_scene(store)
    _synthetic_ladder(
        root,
        "pts",
        [
            {"coverage_fraction": 0.0, "n_points": 100},
            {"coverage_fraction": 4.0, "n_points": 400},
        ],
    )
    close(root)
    assert not is_consolidated(store)

    report = restamp_lod_store(store)

    assert not is_consolidated(store), "an interrupted tile must not read finished"
    assert report.was_consolidated is False
    assert report.residual == [], "a deliberately absent index is not a residual"
    assert report.clean
    nodes = _node_attrs(store)
    assert nodes["pts"]["selector"] == DERIVED_LOD_SELECTOR
    assert nodes["pts/child_1"]["coverage_fraction"] == WHOLE_OBJECT_FINEST_ANCHOR


# ────────────────────────────────────────────────────────────────────────
# Per-recipe round trip — the invariant the whole command rests on
# ────────────────────────────────────────────────────────────────────────


def _recipe_store(recipe: str, store: Path) -> None:
    """Write a real ``gsplat lod --recipe <recipe>`` tree to ``store``."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

    rng = np.random.default_rng(0)
    n = 800
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    data = GSplatData(
        centers=rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )
    built = build_recipe(
        data,
        recipe,  # type: ignore[arg-type]
        RecipeParams(
            max_elements=250,
            compression_factor=4,
            levels=2,
            device="cpu",
            seed=0,
            # Bare leaves: a stream ladder adds `additive_<i>` nodes, which are
            # not `kind=lod` groups and so are noise for this test.
            additive_ladders=False,
        ),
    )
    if isinstance(built, GSplatData):
        built.save(store, ordering="none")
    else:
        write_gsplats_tree(store, built, ordering="none")


def _all_ladders(store: Path) -> Dict[str, List[float]]:
    """Every ``kind=lod`` group's thresholds, coarsest→finest, keyed by path."""
    return {
        path: _ladder(store, path)
        for path, attrs in _attrs(store).items()
        if attrs.get("kind") == "lod"
    }


def _age_every_ladder(store: Path, scale: float = 4.0) -> None:
    """Walk a freshly written store back to what an older Luxar left on disk.

    ``selector="coverage"`` over the same ladder scaled off the screen-area
    range — the same ageing the ``legacy_gsplats`` fixture does, applied to every
    ``kind=lod`` group in the tree. Order-preserving, so nothing but the units
    changes.
    """
    root = open_group(store, mode="r+")

    def visit(group: zarr.Group) -> None:
        if dict(group.attrs).get("kind") == "lod":
            group.attrs["selector"] = LEGACY_LOD_SELECTOR
            for name in group.group_keys():
                child = group[str(name)]
                stored = dict(child.attrs).get("coverage_fraction")
                if stored is not None:
                    child.attrs["coverage_fraction"] = scale * float(stored)
        for name in group.group_keys():
            visit(group[str(name)])

    visit(root)
    consolidate(root)
    close(root)


#: Per recipe: how many ``kind=lod`` groups it writes, and the finest threshold
#: each of them carries. ``tiles`` has NO substitutive ladder at all (its tiles
#: are bare leaves), which is itself worth pinning: the pass must not invent one.
_RECIPE_SHAPE = {
    "levels": (1, WHOLE_OBJECT_FINEST_ANCHOR),
    "tiles": (0, None),
    "overview": (1, PARTITION_FINEST_AREA),
    "adaptive": (4, PARTITION_FINEST_AREA),
}


@pytest.mark.parametrize("recipe", sorted(_RECIPE_SHAPE))
def test_a_restamped_store_matches_what_the_writer_would_have_written(
    recipe: str, tmp_path: Path
) -> None:
    """The invariant the whole command rests on, per recipe.

    Build a real tree with the production writer, record the ladder IT chose,
    age every group to legacy stamps, restamp — and the ladder must come back
    bit-for-bit. Anything the pass reads differently from the writers (the
    anchor rule, the child ordering, the element counts) shows up here as a
    mismatch on the recipe that exercises it, which is the only way to keep the
    two rules from drifting apart.
    """
    store = tmp_path / f"{recipe}.gsplats.zarr"
    _recipe_store(recipe, store)
    expected_count, expected_finest = _RECIPE_SHAPE[recipe]

    written = _all_ladders(store)
    assert len(written) == expected_count, f"{recipe} wrote {sorted(written)}"
    for ladder in written.values():
        assert ladder[-1] == pytest.approx(expected_finest)

    _age_every_ladder(store)
    aged = _all_ladders(store)
    assert (aged != written) or not written, "the ageing must move something"

    report = restamp_lod_store(store)

    assert _all_ladders(store) == pytest.approx(written)
    assert {g.path for g in report.restamped} == set(written)
    assert report.clean and not report.residual
    for path in written:
        assert _attrs(store)[path]["selector"] == DERIVED_LOD_SELECTOR


# ────────────────────────────────────────────────────────────────────────
# zarr format 2
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("write_format", [2, 3])
def test_the_pass_works_at_either_zarr_format(
    write_format: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Format 2 and 3 disagree about where attrs and the index LIVE.

    Format 2 writes ``.zattrs`` per node plus a separate ``.zmetadata``; format 3
    puts both inside one ``zarr.json`` per node, with the consolidated index
    nested in the root document. This pass edits attrs and re-consolidates, so it
    touches every one of those differences — and the shipped legacy corpus this
    command exists for is format 2.
    """
    from luxar import _zarr_compat

    monkeypatch.setattr(_zarr_compat, "ZARR_FORMAT", write_format)
    with zarr.config.set({"default_zarr_format": write_format}):
        store = tmp_path / f"v{write_format}.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            _add_legacy_ladder(scene, "pts", 0)
        assert int(open_group(store, mode="r").metadata.zarr_format) == write_format
        before = _hash(store)

        report = restamp_lod_store(store)

        assert [g.path for g in report.restamped] == ["pts"]
        assert report.clean and not report.residual
        assert report.content_hash_status == HASH_RESTAMPED
        assert _attrs(store)["pts"]["selector"] == DERIVED_LOD_SELECTOR
        assert _ladder(store, "pts") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]
        assert _hash(store) not in (None, before)
