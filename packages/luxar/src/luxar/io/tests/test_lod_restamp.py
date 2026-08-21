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
    open_group,
    read_consolidated_attrs,
)
from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    coverage_fractions,
)
from luxar.io.compiler import LuxarZarrCompiler
from luxar.io.lod_restamp import (
    RestampedGroup,
    RestampReport,
    _verify,
    restamp_lod_store,
)
from luxar.typing_utils.constants import DERIVED_LOD_SELECTOR, LEGACY_LOD_SELECTOR

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


@pytest.fixture(scope="module")
def legacy_scene_template(tmp_path_factory) -> Path:
    """A compiled scene holding the four ladder shapes that matter, all legacy.

    * ``pts`` — a plain whole-object ladder at the scene root.
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


@pytest.fixture
def legacy_scene(legacy_scene_template: Path, tmp_path: Path) -> Path:
    """A per-test writable copy of :func:`legacy_scene_template`."""
    destination = tmp_path / "scene.luxar.zarr"
    shutil.copytree(legacy_scene_template, destination)
    return destination


def _attrs(store: Path) -> Dict[str, Dict[str, Any]]:
    """Every node's attrs from the CONSOLIDATED index — what the viewer reads."""
    return dict(read_consolidated_attrs(store))


def _ladder(store: Path, group_path: str) -> List[Optional[float]]:
    """A lod group's thresholds, coarsest→finest by ``child_index``."""
    nodes = _attrs(store)
    children = [
        (path, attrs)
        for path, attrs in nodes.items()
        if path.startswith(f"{group_path}/") and "coverage_fraction" in attrs
    ]
    ordered = sorted(children, key=lambda kv: int(kv[1]["child_index"]))
    return [float(attrs["coverage_fraction"]) for _, attrs in ordered]


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
    """A ladder level that is itself a partition carries no single count."""
    store = tmp_path / "wrapper.luxar.zarr"
    root = _synthetic_scene(store)
    lod = _synthetic_ladder(
        root,
        "mixed",
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

    assert report.restamped[0].element_counts == [50, 700]
    assert _ladder(store, "mixed") == [0.0, WHOLE_OBJECT_FINEST_ANCHOR]


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


def test_a_second_run_changes_nothing_at_all(legacy_scene: Path) -> None:
    """Idempotency down to the hash: re-running must not re-publish the store."""
    restamp_lod_store(legacy_scene)
    after_first = _attrs(legacy_scene)

    report = restamp_lod_store(legacy_scene)

    assert not report.restamped
    assert len(report.already_current) == 4
    assert report.content_hash is None
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


def test_a_dry_run_reports_everything_and_writes_nothing(legacy_scene: Path) -> None:
    before = _attrs(legacy_scene)

    report = restamp_lod_store(legacy_scene, dry_run=True)

    assert report.dry_run
    assert {g.path for g in report.restamped} == {
        "pts",
        "tiled/part_0",
        "tiled/part_1",
        "lonely/part_0",
    }
    assert next(g for g in report.restamped if g.path == "tiled/part_0").new_thresholds
    assert _attrs(legacy_scene) == before, "a dry run must not touch the store"
    assert report.content_hash is None


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
