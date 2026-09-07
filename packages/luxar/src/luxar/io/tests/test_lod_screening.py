"""Tests for the opening-shot LOD screen (:mod:`luxar.io.lod_screening`).

The projection/metric primitives are pinned against values derived ANALYTICALLY
in each test — a unit cube at a known distance under a known FOV has a closed
form for its NDC rect, so nothing here snapshots what the implementation happens
to produce. The two selectors' metrics are exercised on a case where they
disagree about which level they pick, because keeping their units apart is the
whole point of the module.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pytest

from luxar._zarr_compat import close, consolidate, create_array, open_group
from luxar.core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    coverage_fractions,
)
from luxar.io.lod_screening import (
    DEFAULT_ASPECTS,
    DEGENERATE_RECT_HALF_EXTENT,
    VERDICT_ALREADY_CURRENT,
    VERDICT_FRAGILE,
    VERDICT_NO_OP,
    VERDICT_OFF_SCREEN,
    VERDICT_SKIPPED,
    VERDICT_WIN,
    Box3,
    calculate_camera_distance,
    frustum_intersects_box,
    frustum_planes,
    legacy_coverage_metric,
    mat4_from_column_major,
    perspective_matrix,
    pick_child_with_hysteresis,
    print_screen_report,
    project_bounds_to_display_dims,
    project_box_area_fraction,
    project_box_diagonal_px,
    project_box_ndc_rect,
    screen_lod_store,
    screen_stores,
    transform_box,
    view_matrix,
)

#: A 90-degree vertical FOV makes ``tan(fov/2) == 1``, so every expected NDC
#: coordinate below is a plain ``coordinate / depth`` and can be written out.
FOV90 = 90.0


def _proj_view(distance: float, aspect: float = 1.0, fov: float = FOV90) -> np.ndarray:
    """``P·V`` for a camera at ``(0, 0, distance)`` looking down ``-Z``."""
    projection = perspective_matrix(fov, aspect, 0.001, 10_000.0)
    return projection @ view_matrix((0.0, 0.0, distance))


def _unit_cube() -> Box3:
    """The cube spanning ``[-0.5, 0.5]`` on every axis."""
    return Box3((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


# --------------------------------------------------------------------------- #
# Projection primitives, against hand-derived values.
# --------------------------------------------------------------------------- #


def test_ndc_rect_of_a_unit_cube_matches_the_closed_form() -> None:
    """At ``tan(fov/2) == 1`` the rect is ``±0.5 / (distance − 0.5)``.

    The camera sits at ``z = 2``; the cube's NEAREST face is at ``z = 0.5``, so
    the widest corners are at view depth ``w = 1.5`` and project to
    ``0.5 / 1.5 = 1/3``. The far face (``w = 2.5``) projects inside that, so it
    never sets the bounds.
    """
    rect = project_box_ndc_rect(_unit_cube(), _proj_view(2.0))
    assert rect is not None
    expected = 0.5 / 1.5
    assert rect.min_x == pytest.approx(-expected)
    assert rect.max_x == pytest.approx(expected)
    assert rect.min_y == pytest.approx(-expected)
    assert rect.max_y == pytest.approx(expected)
    assert rect.half_w == pytest.approx(expected)
    assert rect.half_h == pytest.approx(expected)


def test_aspect_ratio_scales_only_the_x_half_span() -> None:
    """``x`` is divided by ``aspect·tan``, ``y`` only by ``tan`` — so only X moves."""
    square = project_box_ndc_rect(_unit_cube(), _proj_view(2.0, aspect=1.0))
    wide = project_box_ndc_rect(_unit_cube(), _proj_view(2.0, aspect=2.0))
    assert square is not None and wide is not None
    assert wide.half_w == pytest.approx(square.half_w / 2.0)
    assert wide.half_h == pytest.approx(square.half_h)


def test_area_fraction_of_a_unit_cube_is_the_half_span_product() -> None:
    """Fully on screen, the metric is exactly ``halfW × halfH``."""
    half = 0.5 / 1.5
    assert project_box_area_fraction(_unit_cube(), _proj_view(2.0)) == pytest.approx(
        half * half
    )
    # Aspect 2 halves the X span, and therefore the area.
    assert project_box_area_fraction(
        _unit_cube(), _proj_view(2.0, aspect=2.0)
    ) == pytest.approx(half / 2.0 * half)


def test_area_fraction_saturates_at_exactly_one() -> None:
    """A box far larger than the frustum reads 1.0 — the viewport clamp, not 1e8.

    Without the ``min(…, 1) / max(…, -1)`` intersection the raw half-spans here
    are ~1e4 each and the product would be ~1e8, which no threshold could ever
    bound.
    """
    huge = Box3((-1e4, -1e4, -1.0), (1e4, 1e4, 1.0))
    assert project_box_area_fraction(huge, _proj_view(2.0)) == 1.0


def test_off_screen_box_reads_zero_area_and_fails_the_frustum_gate() -> None:
    """The two independent off-screen paths, checked independently.

    ``projectBoxAreaFraction`` zeroes on an INVERTED clipped interval; the
    world-space frustum gate is a separate, earlier test the registry applies
    before any metric is taken. A box parked far off ``+X`` must fail both.
    """
    proj_view = _proj_view(2.0)
    away = Box3((1e4, -1.0, -1.0), (1e4 + 1.0, 1.0, 1.0))
    assert project_box_area_fraction(away, proj_view) == 0.0
    assert frustum_intersects_box(frustum_planes(proj_view), away) is False
    # …and the same gate says yes to a box the camera is actually looking at.
    assert frustum_intersects_box(frustum_planes(proj_view), _unit_cube()) is True


def test_frustum_gate_keeps_a_box_that_only_partly_leaves_the_view() -> None:
    """The POSITIVE-vertex test: partial overlap is still inside.

    A box reaching from the middle of the view out to ``x = 1e4`` has corners on
    both sides of the right plane. Testing the near (negative) vertex instead
    would call it outside and hide a node that is half on screen — so this is
    the case that distinguishes the two, not the wholly-outside one above.
    """
    proj_view = _proj_view(2.0)
    straddling = Box3((-0.5, -0.5, -0.5), (1e4, 0.5, 0.5))
    assert frustum_intersects_box(frustum_planes(proj_view), straddling) is True
    assert project_box_area_fraction(straddling, proj_view) > 0.0


def test_near_plane_straddle_saturates_both_metrics_and_forces_the_finest() -> None:
    """A corner behind the camera aborts the projection → ``+inf`` → finest level."""
    straddling = Box3((-1.0, -1.0, -1.0), (1.0, 1.0, 3.0))  # camera sits at z = 2
    proj_view = _proj_view(2.0)
    assert project_box_ndc_rect(straddling, proj_view) is None
    assert project_box_area_fraction(straddling, proj_view) == math.inf
    assert project_box_diagonal_px(straddling, proj_view, 1920, 1080) == math.inf
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5], 0, math.inf) == 2


def test_zero_thickness_rect_ramps_to_its_full_linear_span() -> None:
    """A flat (edge-on) box reads its clipped LINEAR span, not a zero area.

    ``y`` is identically 0, so ``rawThin == 0``, the ramp factor is 1, and the
    metric is the X half-span ``0.5 / 1.5``. Without the ramp this is 0 and the
    node is pinned to the coarsest level forever.
    """
    flat = Box3((-0.5, 0.0, -0.5), (0.5, 0.0, 0.5))
    assert project_box_area_fraction(flat, _proj_view(2.0)) == pytest.approx(0.5 / 1.5)


def test_degenerate_ramp_is_continuous_at_half_the_floor() -> None:
    """Half-way down the ramp the metric is half the span, not the area.

    ``y = ±7.5e-4`` at the nearest depth 1.5 gives ``rawThin = 5e-4``, exactly
    half of :data:`DEGENERATE_RECT_HALF_EXTENT`, so the ramp factor is 0.5 and
    the metric is ``span × 0.5 = (1/3) × 0.5``. The plain area product would be
    ``(1/3) × 5e-4 ≈ 1.7e-4``, three orders of magnitude smaller — the ramp is
    doing all the work here and a dropped ``max(...)`` would show it.
    """
    thickness = 1.5 * DEGENERATE_RECT_HALF_EXTENT / 2.0
    thin = Box3((-0.5, -thickness, -0.5), (0.5, thickness, 0.5))
    span = 0.5 / 1.5
    metric = project_box_area_fraction(thin, _proj_view(2.0))
    assert metric == pytest.approx(span * 0.5)
    assert metric > span * DEGENERATE_RECT_HALF_EXTENT * 10  # ≫ the area product


def test_degenerate_ramp_switches_off_above_the_floor() -> None:
    """Thicker than the floor, the metric is exactly the area product again."""
    thickness = 1.5 * DEGENERATE_RECT_HALF_EXTENT * 2.0
    thick = Box3((-0.5, -thickness, -0.5), (0.5, thickness, 0.5))
    half_w = 0.5 / 1.5
    half_h = thickness / 1.5
    assert project_box_area_fraction(thick, _proj_view(2.0)) == pytest.approx(
        half_w * half_h
    )


def test_diagonal_px_and_the_legacy_metric_are_pixel_quantities() -> None:
    """``hypot(halfW·W, halfH·H)`` over ``FILL_FACTOR × min(W, H)``.

    The diagonal is UNCLIPPED and the denominator is the SHORT axis, which is
    what makes the legacy metric run to ~4 rather than saturating at 1.
    """
    half = 0.5 / 1.5
    proj_view = _proj_view(2.0)
    assert project_box_diagonal_px(
        _unit_cube(), proj_view, 1000, 1000
    ) == pytest.approx(math.hypot(half * 1000, half * 1000))
    assert legacy_coverage_metric(_unit_cube(), proj_view, 1000, 1000) == pytest.approx(
        math.hypot(half * 1000, half * 1000) / (0.5 * 1000)
    )
    # Unclipped: the saturating box above reads 1.0 in area but blows past 1 here.
    huge = Box3((-1e4, -1e4, -1.0), (1e4, 1e4, 1.0))
    assert legacy_coverage_metric(huge, proj_view, 1000, 1000) > 1000.0


def test_the_two_metrics_disagree_about_the_bucket() -> None:
    """The reason the units are never interchangeable, as a worked example.

    At ``distance = 1.8`` the unit cube's half-span is ``0.5 / 1.3 ≈ 0.3846``:

    * legacy ``= 0.3846 × 2√2 ≈ 1.088`` — at or past the whole-object legacy
      anchor 1.0, so a ``[0, 0.5, 1.0]`` ladder picks the FINEST level;
    * area ``= 0.3846² ≈ 0.1479`` — below the whole-object area anchor 0.5 and
      below the next rung 0.25, so a ``[0, 0.25, 0.5]`` ladder picks the
      COARSEST.

    Feeding either metric into the other's ladder inverts the answer, which is
    exactly the mistake a single shared "coverage" number would make.
    """
    half = 0.5 / 1.3
    proj_view = _proj_view(1.8)
    legacy = legacy_coverage_metric(_unit_cube(), proj_view, 1000, 1000)
    area = project_box_area_fraction(_unit_cube(), proj_view)
    assert legacy == pytest.approx(half * 2.0 * math.sqrt(2.0))
    assert area == pytest.approx(half * half)

    legacy_ladder = [0.0, 0.5, 1.0]
    area_ladder = [0.0, 0.25, WHOLE_OBJECT_FINEST_ANCHOR]
    assert pick_child_with_hysteresis(legacy_ladder, 0, legacy) == 2
    assert pick_child_with_hysteresis(area_ladder, 0, area) == 0
    # Cross the units and both answers flip.
    assert pick_child_with_hysteresis(legacy_ladder, 0, area) == 0
    assert pick_child_with_hysteresis(area_ladder, 0, legacy) == 2


# --------------------------------------------------------------------------- #
# The pick, against the TypeScript behaviour clause by clause.
# --------------------------------------------------------------------------- #


def test_pick_returns_minus_one_for_an_empty_ladder() -> None:
    """``thresholds.length === 0`` → ``-1``."""
    assert pick_child_with_hysteresis([], 0, 1.0) == -1


def test_pick_breaks_at_the_first_threshold_above_the_metric() -> None:
    """The scan BREAKS; it does not take the max over all satisfied thresholds.

    ``[0, 0.5, 0.2, 0.9]`` is non-monotone, and index 2 (0.2) IS below the
    metric — but the loop has already stopped at index 1. A ``max``-style
    implementation would answer 2 and silently reorder a broken store's ladder.
    """
    assert pick_child_with_hysteresis([0.0, 0.5, 0.2, 0.9], 0, 0.3) == 0


def test_pick_upgrades_immediately_with_no_hysteresis() -> None:
    """A finer natural pick wins outright, even one exactly on its threshold."""
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5], 0, 0.5) == 2
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5], 0, 0.25) == 1


def test_pick_downgrade_requires_clearing_the_spacing_aware_margin() -> None:
    """The margin is 10% of the gap to the ADJACENT coarser threshold.

    From index 2 of ``[0, 0.25, 0.5]`` the band is ``0.5 − 0.1×(0.5−0.25) =
    0.475``: 0.48 holds, 0.47 drops. A margin taken from the threshold in
    isolation would be ``0.45`` and would let 0.47 hold too.
    """
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5], 2, 0.48) == 2
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5], 2, 0.47) == 1
    # Exactly ON the band edge holds: the TS test is ``metric < threshold −
    # margin``, so a ``<=`` here would drop a level early.
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5], 2, 0.475) == 2


def test_pick_downgrade_from_the_bottom_real_level_uses_the_whole_threshold() -> None:
    """At ``currentIdx == 1`` the coarser threshold is 0, so the margin is 10%.

    Also guards the Python-specific hazard: ``thresholds[currentIdx - 1]`` with
    ``currentIdx == 0`` would wrap to the LAST element. It cannot be reached —
    a downgrade implies ``currentIdx >= 1`` — and this pins the boundary case.
    """
    assert pick_child_with_hysteresis([0.0, 0.5], 1, 0.46) == 1
    assert pick_child_with_hysteresis([0.0, 0.5], 1, 0.44) == 0


def test_pick_jumps_several_levels_when_the_metric_falls_far() -> None:
    """A multi-level drop is not suppressed by a one-gap margin."""
    assert pick_child_with_hysteresis([0.0, 0.25, 0.5, 0.75], 3, 0.1) == 0


# --------------------------------------------------------------------------- #
# Camera fit and matrix plumbing.
# --------------------------------------------------------------------------- #


def test_camera_distance_matches_the_viewer_formula() -> None:
    """``nearestDepth + fitRadius / fitRatio / tan(halfFov)``, maxed over the axes.

    For ``[-1, 1]³`` at ``tan(halfFov) == 1`` and ``fitRatio = 0.75``:
    ``nearestDepth = 1``, ``fitRadius = 1``, so the vertical fit is
    ``1 + 4/3``. The horizontal fit divides by ``aspect`` as well, so it only
    wins when ``aspect < 1``.
    """
    box = Box3((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    vertical = 1.0 + (1.0 / 0.75)
    assert calculate_camera_distance(box, FOV90, 1.0) == pytest.approx(vertical)
    assert calculate_camera_distance(box, FOV90, 2.0) == pytest.approx(vertical)
    assert calculate_camera_distance(box, FOV90, 0.5) == pytest.approx(
        1.0 + (1.0 / 0.75) / 0.5
    )


def test_camera_distance_falls_back_to_half_the_depth_for_a_flat_box() -> None:
    """Zero screen-plane radius → half the Z extent keeps the camera off the geometry."""
    flat = Box3((0.0, 0.0, -2.0), (0.0, 0.0, 2.0))
    # target = centre = (0, 0, 0) → nearestDepth 2, fitRadius = |4| / 2 = 2.
    assert calculate_camera_distance(flat, FOV90, 1.0) == pytest.approx(
        2.0 + (2.0 / 0.75)
    )


def test_column_major_transform_puts_translation_at_indices_12_13_14() -> None:
    """The THREE.js convention the store writes, converted to a math matrix."""
    flat = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 7, 8, 9, 1]
    matrix = mat4_from_column_major(flat)
    assert matrix[0, 3] == 7
    assert matrix[1, 3] == 8
    assert matrix[2, 3] == 9
    with pytest.raises(ValueError):
        mat4_from_column_major([1, 2, 3])


def test_transform_box_rotates_the_corner_set_not_the_min_max_pair() -> None:
    """A 45-degree Z rotation: the ANTI-diagonal corners set both X bounds.

    45 degrees, not 90: a right-angle rotation of an axis-aligned box maps the
    ``(min, max)`` pair onto the new AABB's own corners, so mapping just those
    two gives the right answer by accident and both the corner-enumeration bug
    and the map-only-the-pair bug survive. At 45 degrees ``(x, y) →
    ((x − y)/√2, (x + y)/√2)`` and the X bounds come from ``(0, 1)`` and
    ``(2, 0)`` — neither of which is a corner of the input pair.
    """
    half = math.sqrt(0.5)
    rotation = np.array(
        [
            [half, -half, 0.0, 0.0],
            [half, half, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    box = Box3((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    out = transform_box(box, rotation)
    # min X from (0, 1) → −1/√2; max X from (2, 0) → 2/√2. The diagonal pair
    # alone would give [0, 1/√2] on X and miss both.
    assert out.min == pytest.approx((-half, 0.0, 0.0))
    assert out.max == pytest.approx((2.0 * half, 3.0 * half, 1.0))


def test_project_bounds_to_display_dims_maps_and_defaults() -> None:
    """Reordered axes map through; unmapped or out-of-range axes stay at 0."""
    lo = [0.0, 10.0, 20.0, 30.0]
    hi = [1.0, 11.0, 22.0, 33.0]
    box = project_bounds_to_display_dims(lo, hi, [2, 0, 1])
    assert box.min == pytest.approx((20.0, 0.0, 10.0))
    assert box.max == pytest.approx((22.0, 1.0, 11.0))
    # Only two displayed dims → Z collapses to 0.
    flat = project_bounds_to_display_dims(lo, hi, [0, 1])
    assert flat.min[2] == 0.0 and flat.max[2] == 0.0
    # An index past the end of the bounds arrays is ignored, not an error.
    guarded = project_bounds_to_display_dims(lo, hi, [9, 0, 1])
    assert guarded.min[0] == 0.0 and guarded.max[0] == 0.0


# --------------------------------------------------------------------------- #
# Store fixtures.
# --------------------------------------------------------------------------- #


def _blob(rng: np.random.Generator, n: int, half: float) -> np.ndarray:
    """``n`` points uniformly inside the cube of half-extent ``half``."""
    return rng.uniform(-half, half, size=(n, 3)).astype(np.float32)


@pytest.fixture()
def win_scene(tmp_path: Path) -> Path:
    """A scene whose LOD group spans a big-but-not-screen-filling share of the shot.

    ``spacer`` widens the scene root to ``x ∈ [±100]`` so the fitted camera pulls
    back to frame it; the ladder's own blob is a cube of half-extent 30 at the
    origin. At that framing the blob's projected half-span is ~0.22, which is
    * past the LEGACY 0.5 rung (metric ≈ 0.64 → level 1), and
    * short of the RE-DERIVED 0.25 rung (area ≈ 0.05 → level 0).

    So the re-derived ladder is strictly coarser at every aspect: a ``win``.
    """
    from luxar import Dimensions, LuxarZarrCompiler

    path = tmp_path / "win.luxar.zarr"
    rng = np.random.default_rng(0)
    blob = _blob(rng, 4000, 30.0)
    spacer = _blob(rng, 200, 1.0) + np.array([100.0, 0.0, 0.0], dtype=np.float32)
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("whole")
        lod.add_points("l0", blob[::16], radii=0.5, coverage_fraction=0.0)
        lod.add_points("l1", blob[::4], radii=0.5, coverage_fraction=0.5)
        lod.add_points("l2", blob, radii=0.5, coverage_fraction=1.0)
        scene.add_points("spacer", spacer, radii=0.5)
    return path


@pytest.fixture()
def fragile_scene(tmp_path: Path) -> Path:
    """A lone blob that IS the whole scene — the aspect-sensitive regime.

    Fitted alone, the blob's half-span is ``0.75`` of the vertical NDC axis (the
    ``fitRatio``). At 1:1 that is an area of ``0.5625`` — past the whole-object
    anchor 0.5, so the re-derived ladder still picks the finest level and nothing
    is deferred. Widen the window and the SAME object covers less area
    (``0.75 × 0.75/aspect``): ``0.316`` at 16:9 and ``0.241`` at 21:9, each one
    rung coarser. The legacy metric is aspect-invariant under this fit and stays
    on the finest level throughout — so the verdict genuinely depends on the
    window shape, which is what :data:`VERDICT_FRAGILE` is for.
    """
    from luxar import Dimensions, LuxarZarrCompiler

    path = tmp_path / "fragile.luxar.zarr"
    blob = _blob(np.random.default_rng(3), 4000, 30.0)
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("whole")
        lod.add_points("l0", blob[::16], radii=0.5, coverage_fraction=0.0)
        lod.add_points("l1", blob[::4], radii=0.5, coverage_fraction=0.5)
        lod.add_points("l2", blob, radii=0.5, coverage_fraction=1.0)
    return path


@pytest.fixture()
def partition_scene(tmp_path: Path) -> Path:
    """A ``kind=partition`` of two ``kind=lod`` groups — the tile-anchored case."""
    from luxar import Dimensions, LuxarZarrCompiler

    path = tmp_path / "partition.luxar.zarr"
    rng = np.random.default_rng(1)
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        outer = scene.add_partition_group(
            "tiled", display_type="points", max_elements=10_000
        )
        for index, offset in enumerate((-20.0, 20.0)):
            points = _blob(rng, 2000, 10.0) + np.array(
                [offset, 0.0, 0.0], dtype=np.float32
            )
            part = outer.add_lod_group(f"part_{index}")
            part.add_points("coarse", points[::8], radii=0.2, coverage_fraction=0.0)
            part.add_points("fine", points, radii=0.2, coverage_fraction=1.0)
    return path


@pytest.fixture()
def already_current_scene(tmp_path: Path) -> Path:
    """A ladder already on ``screen-area`` carrying exactly the derived thresholds."""
    from luxar import Dimensions, LuxarZarrCompiler

    path = tmp_path / "current.luxar.zarr"
    rng = np.random.default_rng(2)
    blob = _blob(rng, 3200, 25.0)
    counts = [len(blob[::16]), len(blob[::4]), len(blob)]
    ladder = coverage_fractions(counts)
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("whole", selector="screen-area")
        lod.add_points("l0", blob[::16], radii=0.5, coverage_fraction=ladder[0])
        lod.add_points("l1", blob[::4], radii=0.5, coverage_fraction=ladder[1])
        lod.add_points("l2", blob, radii=0.5, coverage_fraction=ladder[2])
    return path


def _handmade_store(
    path: Path,
    *,
    group_bounds: dict,
    root_max: float = 5.0,
    lod_attrs: dict | None = None,
) -> Path:
    """A minimal scene document with one ``kind=lod`` group, written attr by attr.

    Some states a real compile can never produce still have to be screened — a
    group parked outside the scene's own bounds (the off-screen gate), a child
    with no ``coverage_fraction`` (a skip). The authoring API refuses to build
    those, so they are written directly. Everything here goes through the
    bi-format facade, so the fixture works at zarr format 2 and 3 alike.
    """
    root = open_group(path, mode="w")
    root.attrs.update(
        {
            "type": "scene",
            "position_bounds": {
                "min": [-root_max, -root_max, -root_max],
                "max": [root_max, root_max, root_max],
            },
        }
    )
    lod = root.create_group("lod")
    lod.attrs.update(
        {"kind": "lod", "type": "group", "default_level": 0, **(lod_attrs or {})}
    )
    for name, child in group_bounds.items():
        node = lod.create_group(name)
        node.attrs.update(child)
        # A real leaf carries data; the screen reads only attrs, but writing one
        # tiny array keeps the store loadable by anything that expects arrays.
        create_array(
            node, "positions", data=np.zeros((1, 3), np.float32), compressor=None
        )
    consolidate(root)
    return path


# --------------------------------------------------------------------------- #
# The screen, end to end.
# --------------------------------------------------------------------------- #


def test_payload_files_do_not_warn_during_the_scene_walk(tmp_path: Path) -> None:
    path = _handmade_store(
        tmp_path / "payload.luxar.zarr",
        group_bounds={
            "coarse": {
                "type": "points",
                "n_points": 10,
                "child_index": 0,
                "coverage_fraction": 0.0,
                "position_bounds": {
                    "min": [-1.0, -1.0, -1.0],
                    "max": [1.0, 1.0, 1.0],
                },
            },
            "fine": {
                "type": "points",
                "n_points": 100,
                "child_index": 1,
                "coverage_fraction": 1.0,
                "position_bounds": {
                    "min": [-1.0, -1.0, -1.0],
                    "max": [1.0, 1.0, 1.0],
                },
            },
        },
    )
    root = open_group(path, mode="a")
    sound = root.create_group("sound")
    sound.attrs.update({"type": "sound", "audio_file": "audio.mp3"})
    (path / "sound" / "audio.mp3").write_bytes(b"ID3payload")
    close(root)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scene = screen_lod_store(path)

    assert scene.skipped_reason == ""
    assert [group.path for group in scene.groups] == ["lod"]


def test_whole_object_group_is_a_win_at_every_aspect(win_scene: Path) -> None:
    """The headline case: legacy says finest-ish, re-derived says coarser."""
    scene = screen_lod_store(win_scene)
    assert scene.skipped_reason == ""
    assert [group.path for group in scene.groups] == ["whole"]
    group = scene.groups[0]

    assert group.verdict == VERDICT_WIN
    assert group.selector == "coverage"
    assert group.partition_bound is False
    # Coarsest is the always-eligible 0.0 floor, then one area-halving per level
    # up to the whole-object anchor.
    assert group.rederived_thresholds == pytest.approx([0.0, 0.25, 0.5])
    assert group.rederived_thresholds[-1] == WHOLE_OBJECT_FINEST_ANCHOR

    labels = [m.label for m in group.measurements]
    assert labels == [label for label, _ in DEFAULT_ASPECTS]
    for m in group.measurements:
        assert m.off_screen is False
        assert m.rederived_index < m.today_index
        assert m.rederived_elements is not None and m.today_elements is not None
        assert m.rederived_elements < m.today_elements
        # The two metrics are in different units, so the numbers must differ.
        assert m.today_metric != pytest.approx(m.area_metric)


def test_a_verdict_that_flips_across_the_aspect_sweep_is_fragile(
    fragile_scene: Path,
) -> None:
    """Coarser at 16:9 and 21:9 but NOT at 1:1 — filed as fragile, not as a win.

    This is the case a single-aspect screen would have mis-reported, in either
    direction: measured only at 1:1 it looks like a no-op, measured only at 21:9
    like a clean win.
    """
    group = screen_lod_store(fragile_scene).groups[0]
    assert group.verdict == VERDICT_FRAGILE
    by_label = {m.label: m for m in group.measurements}
    # Every aspect agrees on TODAY (the legacy metric is aspect-invariant under
    # this fit) — only the re-derived, area-based pick moves.
    assert {m.today_index for m in group.measurements} == {2}
    assert by_label["1:1"].coarser is False
    assert by_label["1:1"].rederived_index == 2
    assert by_label["16:9"].rederived_index == 1
    assert by_label["21:9"].rederived_index == 0
    assert by_label["1:1"].area_metric > by_label["16:9"].area_metric
    assert by_label["16:9"].area_metric > by_label["21:9"].area_metric


def test_partition_bound_groups_anchor_at_one_and_report_the_reason(
    partition_scene: Path,
) -> None:
    """A ladder under a real (>1-part) partition is derived at the tile anchor."""
    scene = screen_lod_store(partition_scene)
    assert sorted(group.path for group in scene.groups) == [
        "tiled/part_0",
        "tiled/part_1",
    ]
    for group in scene.groups:
        assert group.partition_bound is True
        assert "kind=partition ancestor" in group.anchor_reason
        assert group.rederived_thresholds[-1] == PARTITION_FINEST_AREA
        # Two levels, tile-anchored: the whole-object ladder doubled.
        assert group.rederived_thresholds == pytest.approx([0.0, 1.0])
        # A tile occupying a fraction of the shot never reaches its own anchor,
        # so neither ladder leaves level 0 — no-op, not a win.
        assert group.verdict == VERDICT_NO_OP
        for m in group.measurements:
            assert m.today_index == 0 and m.rederived_index == 0


def test_partition_anchor_is_twice_the_whole_object_one(
    win_scene: Path, partition_scene: Path
) -> None:
    """The two anchors differ by exactly the ratio the derivations document."""
    whole = screen_lod_store(win_scene).groups[0]
    tile = screen_lod_store(partition_scene).groups[0]
    assert whole.rederived_thresholds[-1] == WHOLE_OBJECT_FINEST_ANCHOR
    assert tile.rederived_thresholds[-1] == PARTITION_FINEST_AREA
    assert tile.rederived_thresholds[-1] == 2 * whole.rederived_thresholds[-1]


def test_already_current_ladder_is_its_own_bucket(already_current_scene: Path) -> None:
    """``screen-area`` + the exact derived ladder → nothing to compare."""
    group = screen_lod_store(already_current_scene).groups[0]
    assert group.verdict == VERDICT_ALREADY_CURRENT
    assert group.selector == "screen-area"
    assert group.stored_thresholds == pytest.approx(group.rederived_thresholds)
    # Both sides read the SAME (area) metric here, unlike the legacy scenes.
    for m in group.measurements:
        assert m.today_metric == pytest.approx(m.area_metric)
        assert m.today_index == m.rederived_index


def test_off_screen_group_is_bucketed_without_taking_a_metric(tmp_path: Path) -> None:
    """The frustum gate fires and both ladders hold the coarsest ready level."""
    path = _handmade_store(
        tmp_path / "away.luxar.zarr",
        group_bounds={
            "c0": {
                "type": "points",
                "n_points": 10,
                "child_index": 0,
                "coverage_fraction": 0.0,
                "position_bounds": {
                    "min": [1e5, -1.0, -1.0],
                    "max": [1e5 + 1, 1.0, 1.0],
                },
            },
            "c1": {
                "type": "points",
                "n_points": 100,
                "child_index": 1,
                "coverage_fraction": 1.0,
                "position_bounds": {
                    "min": [1e5, -1.0, -1.0],
                    "max": [1e5 + 1, 1.0, 1.0],
                },
            },
        },
    )
    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_OFF_SCREEN
    assert all(m.off_screen for m in group.measurements)
    assert all(
        m.today_index == 0 and m.rederived_index == 0 for m in group.measurements
    )


def test_a_child_without_a_threshold_is_skipped_with_a_reason(tmp_path: Path) -> None:
    """What the viewer picks TODAY is undecidable, so no verdict is invented."""
    path = _handmade_store(
        tmp_path / "partial.luxar.zarr",
        group_bounds={
            "c0": {
                "type": "points",
                "n_points": 10,
                "child_index": 0,
                "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            },
            "c1": {
                "type": "points",
                "n_points": 100,
                "child_index": 1,
                "coverage_fraction": 1.0,
                "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            },
        },
    )
    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_SKIPPED
    assert "coverage_fraction" in group.reason
    assert group.measurements == []


def test_an_empty_finest_level_is_skipped_like_restamp_lod_skips_it(
    tmp_path: Path,
) -> None:
    """``coverage_fractions`` refuses a zero finest count; so does the screen."""
    path = _handmade_store(
        tmp_path / "empty.luxar.zarr",
        group_bounds={
            "c0": {
                "type": "points",
                "n_points": 10,
                "child_index": 0,
                "coverage_fraction": 0.0,
                "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            },
            "c1": {
                "type": "points",
                "n_points": 0,
                "child_index": 1,
                "coverage_fraction": 1.0,
                "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            },
        },
    )
    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_SKIPPED
    assert "0 elements" in group.reason


def test_a_scene_with_an_authored_camera_pose_is_skipped_not_guessed(
    tmp_path: Path,
) -> None:
    """The screen models ONE pose; a scene that opens elsewhere says so."""
    path = _handmade_store(
        tmp_path / "posed.luxar.zarr",
        group_bounds={
            "c0": {
                "type": "points",
                "n_points": 10,
                "child_index": 0,
                "coverage_fraction": 0.0,
                "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            }
        },
    )
    root = open_group(path, mode="r+")
    root.attrs["viewer_config"] = {"camera": {"target": [1.0, 2.0, 3.0]}}
    consolidate(root)

    scene = screen_lod_store(path)
    assert scene.groups == []
    assert "viewer_config.camera.target" in scene.skipped_reason


def test_a_group_transform_moves_the_world_box(tmp_path: Path) -> None:
    """A ``transform`` on the lod group itself is inside its ``matrixWorld``.

    The same ladder is screened twice — once at the origin, once translated far
    off ``+X`` by a column-major ``transform`` attr. Only the second falls out
    of the frustum, which is only possible if the transform was applied.
    """
    bounds = {
        "c0": {
            "type": "points",
            "n_points": 10,
            "child_index": 0,
            "coverage_fraction": 0.0,
            "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        },
        "c1": {
            "type": "points",
            "n_points": 100,
            "child_index": 1,
            "coverage_fraction": 1.0,
            "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        },
    }
    here = _handmade_store(tmp_path / "here.luxar.zarr", group_bounds=bounds)
    assert screen_lod_store(here).groups[0].verdict != VERDICT_OFF_SCREEN

    away = _handmade_store(tmp_path / "away.luxar.zarr", group_bounds=bounds)
    root = open_group(away, mode="r+")
    # Column-major identity with the translation in slots 12/13/14.
    root["lod"].attrs["transform"] = [
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1e5,
        0,
        0,
        1,
    ]
    consolidate(root)
    assert screen_lod_store(away).groups[0].verdict == VERDICT_OFF_SCREEN


def _lod_bounds_store(path: Path, lod_bounds: dict | None) -> Path:
    """A ±0.5 cube ladder, optionally carrying robust ``lod_bounds``."""
    raw = {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}
    child = {"type": "points", "position_bounds": raw}
    if lod_bounds is not None:
        child["lod_bounds"] = lod_bounds
    return _handmade_store(
        path,
        root_max=0.5,
        group_bounds={
            "c0": {**child, "n_points": 10, "child_index": 0, "coverage_fraction": 0.0},
            "c1": {**child, "n_points": 40, "child_index": 1, "coverage_fraction": 1.0},
        },
    )


def test_lod_bounds_size_the_metric_but_may_only_reduce_it(tmp_path: Path) -> None:
    """The registry's ``min(areaFraction(lodBox), areaFraction(rawBox))`` rule.

    Fitted alone, the ±0.5 cube's half-span is ``fitRatio`` = 0.75, so the raw
    area is ``0.75² ≈ 0.5625``. Two robust boxes, opposite effects:

    * NARROWER in X (±0.25) genuinely shrinks the metric to ``0.375 × 0.75 ≈
      0.281`` — proof the robust bounds are read at all;
    * FLAT in Y is where the thin-rect ramp turns non-monotone: on its own that
      box reads its full linear span, ``0.75``, which is LARGER than the raw
      box's area. The ``min`` clamps it back to 0.5625. Drop the clamp and
      trimming outliers would make a node look BIGGER.
    """
    plain = screen_lod_store(_lod_bounds_store(tmp_path / "plain.luxar.zarr", None))
    narrow = screen_lod_store(
        _lod_bounds_store(
            tmp_path / "narrow.luxar.zarr",
            {"min": [-0.25, -0.5, -0.5], "max": [0.25, 0.5, 0.5]},
        )
    )
    flat = screen_lod_store(
        _lod_bounds_store(
            tmp_path / "flat.luxar.zarr",
            {"min": [-0.5, 0.0, -0.5], "max": [0.5, 0.0, 0.5]},
        )
    )
    square = next(m for m in plain.groups[0].measurements if m.label == "1:1")
    narrow_square = next(m for m in narrow.groups[0].measurements if m.label == "1:1")
    flat_square = next(m for m in flat.groups[0].measurements if m.label == "1:1")

    assert square.area_metric == pytest.approx(0.75 * 0.75, rel=1e-3)
    assert narrow_square.area_metric == pytest.approx(0.375 * 0.75, rel=1e-3)
    assert flat_square.area_metric == pytest.approx(square.area_metric)


@pytest.mark.parametrize(
    "lod_bounds",
    [
        {"min": [-0.25, -0.5], "max": [0.25, 0.5]},
        {"min": ["-0.25", -0.5, -0.5], "max": [0.25, 0.5, 0.5]},
        {"min": [False, -0.5, -0.5], "max": [0.25, 0.5, 0.5]},
        {"min": [float("nan"), -0.5, -0.5], "max": [0.25, 0.5, 0.5]},
        {"min": [0.25, -0.5, -0.5], "max": [-0.25, 0.5, 0.5]},
        {"min": [-0.75, -0.5, -0.5], "max": [-0.25, 0.5, 0.5]},
    ],
    ids=[
        "wrong-dimensions",
        "non-numeric",
        "boolean",
        "non-finite",
        "unordered",
        "not-contained",
    ],
)
def test_invalid_lod_bounds_fall_back_to_position_bounds(
    tmp_path: Path, lod_bounds: dict
) -> None:
    """The offline screen rejects every robust-bound shape the viewer rejects."""
    plain = screen_lod_store(_lod_bounds_store(tmp_path / "plain.luxar.zarr", None))
    invalid = screen_lod_store(
        _lod_bounds_store(tmp_path / "invalid.luxar.zarr", lod_bounds)
    )

    assert [m.area_metric for m in invalid.groups[0].measurements] == pytest.approx(
        [m.area_metric for m in plain.groups[0].measurements]
    )


def _default_level_store(path: Path, default_level: int) -> Path:
    """A ±5 cube ladder, thresholds ``[0, 2.2, 4]``, at a chosen ``default_level``."""
    child = {
        "type": "points",
        "position_bounds": {"min": [-5.0, -5.0, -5.0], "max": [5.0, 5.0, 5.0]},
    }
    return _handmade_store(
        path,
        lod_attrs={"default_level": default_level},
        group_bounds={
            "c0": {**child, "n_points": 10, "child_index": 0, "coverage_fraction": 0.0},
            "c1": {**child, "n_points": 40, "child_index": 1, "coverage_fraction": 2.2},
            "c2": {
                **child,
                "n_points": 160,
                "child_index": 2,
                "coverage_fraction": 4.0,
            },
        },
    )


def test_default_level_anchors_the_hysteresis_on_frame_one(tmp_path: Path) -> None:
    """``default_level`` is the level in force on frame one, so it can hold a pick.

    The group IS the whole scene, so the fit puts its half-span at ``fitRatio``
    = 0.75 of the vertical NDC axis and the legacy metric is
    ``0.75 × 2√2 ≈ 2.12`` at every aspect. Against the ladder
    ``[0, 2.2, 4.0]`` the NATURAL pick is level 0 (2.12 < 2.2). Starting from
    ``default_level = 1`` the downgrade must first clear
    ``2.2 − 0.1 × 2.2 = 1.98``, which 2.12 does not — so the viewer holds level
    1. Ignoring ``default_level`` (or reading it as 0) gives level 0 instead.
    """
    at_zero = screen_lod_store(_default_level_store(tmp_path / "d0.luxar.zarr", 0))
    at_one = screen_lod_store(_default_level_store(tmp_path / "d1.luxar.zarr", 1))
    assert at_zero.groups[0].default_level == 0
    assert at_one.groups[0].default_level == 1
    for m in at_zero.groups[0].measurements:
        assert m.today_metric == pytest.approx(0.75 * 2.0 * math.sqrt(2.0), rel=1e-3)
        assert m.today_index == 0
    for m in at_one.groups[0].measurements:
        assert m.today_index == 1


def test_screen_stores_records_an_unreadable_store_instead_of_raising(
    tmp_path: Path, win_scene: Path
) -> None:
    """One broken path must not hide the screening of every other store."""
    report = screen_stores([tmp_path / "nope.luxar.zarr", win_scene])
    assert "cannot read" in report.scenes[0].skipped_reason
    assert report.scenes[1].groups[0].verdict == VERDICT_WIN
    assert report.tally[VERDICT_WIN] == 1


def test_tally_covers_every_bucket_and_sums_the_scenes(
    win_scene: Path, partition_scene: Path
) -> None:
    """The report tally is over every scene and names all six buckets."""
    report = screen_stores([win_scene, partition_scene])
    tally = report.tally
    assert set(tally) == {
        VERDICT_WIN,
        VERDICT_FRAGILE,
        VERDICT_NO_OP,
        VERDICT_OFF_SCREEN,
        VERDICT_ALREADY_CURRENT,
        VERDICT_SKIPPED,
    }
    assert tally[VERDICT_WIN] == 1
    assert tally[VERDICT_NO_OP] == 2
    assert sum(tally.values()) == 3


def test_aspect_ratio_can_flip_the_verdict(fragile_scene: Path) -> None:
    """A single aspect cannot answer the question — which is why three are used.

    The SAME store screened at one aspect each: at 1:1 the group still fills
    enough of the shot that the re-derived ladder keeps the finest level (a
    ``no-op``), at 21:9 it does not (a ``win``). Asserting only that the two
    numbers differ would pass with the comparison inverted, so the DIRECTION is
    pinned too — a wider window can only ever show less area.
    """
    narrow = screen_lod_store(fragile_scene, aspects=[("1:1", 1.0)]).groups[0]
    ultrawide = screen_lod_store(fragile_scene, aspects=[("21:9", 21 / 9)]).groups[0]
    assert narrow.measurements[0].area_metric > ultrawide.measurements[0].area_metric
    assert narrow.verdict == VERDICT_NO_OP
    assert ultrawide.verdict == VERDICT_WIN


def test_no_op_can_include_a_more_expensive_finer_opening_pick(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``no-op`` means no coarser pick, not necessarily the same opening cost."""
    path = _handmade_store(
        tmp_path / "finer-no-op.luxar.zarr",
        root_max=1.0,
        group_bounds={
            "c0": _plain_child(0, 250, 0.0),
            "c1": _plain_child(1, 1_000, 3.9),
            "c2": _plain_child(2, 4_000, 4.0),
        },
    )

    report = screen_stores([path])
    group = report.scenes[0].groups[0]
    assert group.verdict == VERDICT_NO_OP
    assert [(m.today_index, m.rederived_index) for m in group.measurements] == [
        (0, 2),
        (0, 1),
        (0, 0),
    ]
    assert (
        group.measurements[0].rederived_elements > group.measurements[0].today_elements
    )

    print_screen_report(report)
    output = capsys.readouterr().out
    assert "L0 (250)" in output
    assert "L2 (4,000) [FINER]" in output


def test_the_legacy_metric_reads_the_viewport_shape_and_the_area_one_cannot() -> None:
    """Why ``--viewport-long`` exists, and why it is not needed for ``screen-area``.

    The legacy metric divides an UNCLIPPED pixel diagonal — which grows with
    BOTH viewport axes — by only the SHORT one, so squashing the viewport at a
    fixed projection changes it. ``projectBoxAreaFraction`` takes no viewport
    argument at all: it is a pure NDC ratio.
    """
    proj_view = _proj_view(2.0)
    square = legacy_coverage_metric(_unit_cube(), proj_view, 1000, 1000)
    squashed = legacy_coverage_metric(_unit_cube(), proj_view, 2000, 1000)
    assert squashed > square
    assert squashed == pytest.approx(
        math.hypot(0.5 / 1.5 * 2000, 0.5 / 1.5 * 1000) / (0.5 * 1000)
    )


def test_viewport_pixel_count_alone_moves_neither_metric(win_scene: Path) -> None:
    """At a FIXED aspect, scaling the viewport up cancels out of both metrics.

    The area metric is size-independent by construction; the legacy one is
    normalised by the short axis, so a uniform scale cancels there too. Only the
    integer rounding of the derived short axis (1920/2.333 → 823) survives, hence
    the loose relative tolerance — the point is that neither metric MOVES.
    """
    small = screen_lod_store(win_scene, viewport_long_px=800).groups[0]
    large = screen_lod_store(win_scene, viewport_long_px=3840).groups[0]
    for a, b in zip(small.measurements, large.measurements):
        assert a.area_metric == pytest.approx(b.area_metric)
        assert a.today_metric == pytest.approx(b.today_metric, rel=1e-3)
        assert a.today_index == b.today_index


def test_render_fov_is_separate_from_the_fit_fov(win_scene: Path) -> None:
    """A wider first-frame FOV shrinks the projected area at the SAME distance.

    The cinematic preset does exactly this: it overrides the FOV only after the
    camera has been fitted at 47, so the opening frame is wider than the fit
    assumed and every group reads a smaller occupancy.
    """
    default = screen_lod_store(win_scene).groups[0]
    cinematic = screen_lod_store(win_scene, render_fov=63.0).groups[0]
    for a, b in zip(default.measurements, cinematic.measurements):
        assert b.area_metric < a.area_metric


def test_a_non_scene_store_is_skipped_with_a_reason(tmp_path: Path) -> None:
    """A ``.gsplats.zarr`` tree has no opening framing, so there is nothing to screen."""
    path = tmp_path / "bare.zarr"
    root = open_group(path, mode="w")
    root.attrs.update({"kind": "lod"})
    consolidate(root)
    scene = screen_lod_store(path)
    assert scene.groups == []
    assert "not a compiled scene" in scene.skipped_reason


def test_a_missing_store_raises_but_a_file_is_rejected_clearly(tmp_path: Path) -> None:
    """Genuine argument errors DO raise — only verdicts are non-fatal."""
    with pytest.raises(FileNotFoundError):
        screen_lod_store(tmp_path / "absent.luxar.zarr")
    plain = tmp_path / "plain.txt"
    plain.write_text("not a store")
    with pytest.raises(ValueError, match="DIRECTORY"):
        screen_lod_store(plain)


def test_the_screen_writes_nothing(win_scene: Path) -> None:
    """A report must not mutate the store it reports on."""

    def fingerprint() -> List[tuple]:
        return sorted(
            (str(p.relative_to(win_scene)), p.stat().st_size)
            for p in win_scene.rglob("*")
            if p.is_file()
        )

    before = fingerprint()
    screen_lod_store(win_scene)
    assert fingerprint() == before


# --------------------------------------------------------------------------- #
# Parity with `restamp-lod`: the screen must never predict a rewrite that the
# real command refuses to make. Each test asserts the SCREEN's bucket and the
# matching `restamp_lod_store(dry_run=True)` classification side by side.
# --------------------------------------------------------------------------- #


def _plain_child(index: int, n_points: int, threshold: float | None) -> dict:
    """One ±1 cube ladder child at ``child_index``, optionally stamped."""
    child = {
        "type": "points",
        "n_points": n_points,
        "child_index": index,
        "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
    }
    if threshold is not None:
        child["coverage_fraction"] = threshold
    return child


def test_a_screen_area_group_is_already_current_whatever_its_ladder_says(
    tmp_path: Path,
) -> None:
    """``restamp-lod`` skips on the SELECTOR ALONE, so the screen must too.

    The ladder here (``[0, 0.02, 0.04]``) is nothing like the one
    ``coverage_fractions`` would derive, and read as area fractions its rungs are
    so low that the finest level is picked at any framing — which under a
    ladder-aware ``already-current`` test made this store screen as a ``win``
    (L2 → L0). ``lod_restamp._plan_lod`` returns before it has read a single
    threshold, so a real run writes NOTHING here, and a report that says
    otherwise is a false positive.
    """
    from luxar.io.lod_restamp import restamp_lod_store

    path = _handmade_store(
        tmp_path / "stamped.luxar.zarr",
        lod_attrs={"selector": "screen-area"},
        group_bounds={
            "c0": _plain_child(0, 10, 0.0),
            "c1": _plain_child(1, 40, 0.02),
            "c2": _plain_child(2, 160, 0.04),
        },
    )

    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_ALREADY_CURRENT
    # The evidence is still there: the ladder is NOT the derived one…
    assert group.stored_thresholds == [0.0, 0.02, 0.04]
    assert group.rederived_thresholds == pytest.approx(
        coverage_fractions([10, 40, 160])
    )
    assert group.stored_thresholds != pytest.approx(group.rederived_thresholds)

    # …and this is exactly what the command it predicts would do: nothing.
    report = restamp_lod_store(path, dry_run=True)
    assert [entry.path for entry in report.already_current] == ["lod"]
    assert report.restamped == []


def test_a_screen_area_group_with_missing_threshold_is_still_already_current(
    tmp_path: Path,
) -> None:
    """Selector-only parity survives hygiene details the command never reads."""
    from luxar.io.lod_restamp import restamp_lod_store

    path = _handmade_store(
        tmp_path / "stamped-missing-threshold.luxar.zarr",
        lod_attrs={"selector": "screen-area"},
        group_bounds={
            "c0": _plain_child(0, 10, None),
            "c1": _plain_child(1, 40, 1.0),
        },
    )

    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_ALREADY_CURRENT
    assert "coverage_fraction" in group.reason
    assert group.measurements == []

    report = restamp_lod_store(path, dry_run=True)
    assert [entry.path for entry in report.already_current] == ["lod"]
    assert report.restamped == []


def test_already_current_report_distinguishes_ladder_evidence_from_hygiene(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a derived mismatch gets a note; an undecidable ladder gets detail."""
    mismatched = _handmade_store(
        tmp_path / "mismatched.luxar.zarr",
        lod_attrs={"selector": "screen-area"},
        group_bounds={
            "c0": _plain_child(0, 10, 0.0),
            "c1": _plain_child(1, 40, 0.02),
            "c2": _plain_child(2, 160, 0.04),
        },
    )
    hygiene = _handmade_store(
        tmp_path / "hygiene.luxar.zarr",
        lod_attrs={"selector": "screen-area"},
        group_bounds={
            "c0": _plain_child(0, 10, 0.05),
            "c1": _plain_child(1, 40, 0.02),
        },
    )

    print_screen_report(screen_stores([mismatched, hygiene]))
    output = capsys.readouterr().out

    assert output.count("stored ladder is NOT the one restamp-lod would derive") == 1
    assert "detail:" in output
    assert "DESCENDS" in output
    assert "skipped:" not in output


def test_a_descending_stored_ladder_is_skipped_exactly_as_restamp_lod_skips_it(
    tmp_path: Path,
) -> None:
    """``restamp-lod`` refuses to invert a ladder; the screen refuses to score it."""
    from luxar.io.lod_restamp import restamp_lod_store

    path = _handmade_store(
        tmp_path / "descending.luxar.zarr",
        group_bounds={
            "c0": _plain_child(0, 10, 0.05),
            "c1": _plain_child(1, 40, 0.02),
        },
    )

    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_SKIPPED
    assert "DESCENDS" in group.reason
    assert group.measurements == []
    assert group.rederived_thresholds == []

    report = restamp_lod_store(path, dry_run=True)
    assert report.restamped == []
    assert [entry.reason for entry in report.unresolved] == ["descending-ladder"]


def test_an_orphan_ladder_child_is_skipped_exactly_as_restamp_lod_skips_it(
    tmp_path: Path,
) -> None:
    """A ``coverage_fraction`` child with no scene-node ``type`` blocks both passes.

    ``restamp-lod`` refuses because re-deriving over the survivors alone strands
    that rung on its legacy threshold, leaving a partial, non-monotonic ladder.
    The screen cannot see the rung either, so any verdict it produced would be
    about a ladder the store does not have.
    """
    from luxar.io.lod_restamp import restamp_lod_store

    path = _handmade_store(
        tmp_path / "orphan.luxar.zarr",
        group_bounds={
            "c0": _plain_child(0, 10, 0.0),
            "c1": _plain_child(1, 40, 1.0),
            # No `type`, so the node filter drops it — but it IS a ladder rung.
            "sidecar": {"coverage_fraction": 0.5},
        },
    )

    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_SKIPPED
    assert "sidecar" in group.reason
    assert group.measurements == []

    report = restamp_lod_store(path, dry_run=True)
    assert report.restamped == []
    assert [entry.reason for entry in report.unresolved] == [
        "unclassifiable-ladder-child"
    ]


def test_all_unclassifiable_children_report_the_same_reason_as_restamp_lod(
    tmp_path: Path,
) -> None:
    """The empty resolved ladder takes precedence over the orphan-rung detail."""
    from luxar.io.lod_restamp import restamp_lod_store

    path = _handmade_store(
        tmp_path / "unclassifiable.luxar.zarr",
        group_bounds={
            "c0": {"coverage_fraction": 0.0},
            "c1": {"coverage_fraction": 1.0},
        },
    )

    group = screen_lod_store(path).groups[0]
    report = restamp_lod_store(path, dry_run=True)

    assert group.verdict == VERDICT_SKIPPED
    assert [entry.reason for entry in report.unresolved] == ["unclassifiable-children"]
    assert group.reason == report.unresolved[0].detail


def test_an_empty_aspect_sweep_is_refused_rather_than_a_free_win(
    win_scene: Path,
) -> None:
    """``all([])`` is True, so no aspects would make every group a vacuous win."""
    with pytest.raises(ValueError, match="at least one aspect"):
        screen_lod_store(win_scene, aspects=[])
    # Raised, NOT recorded per store: `screen_stores` checks before its own
    # one-broken-store guard, which would otherwise blame the stores.
    with pytest.raises(ValueError, match="at least one aspect"):
        screen_stores([win_scene], aspects=[])


def test_an_unknown_selector_and_an_out_of_range_default_level_are_skipped(
    tmp_path: Path,
) -> None:
    """The two ``_preflight`` refusals nothing else covers, each with its reason."""
    bounds = {"c0": _plain_child(0, 10, 0.0), "c1": _plain_child(1, 40, 1.0)}
    unknown = screen_lod_store(
        _handmade_store(
            tmp_path / "pixel.luxar.zarr",
            lod_attrs={"selector": "pixel_size"},
            group_bounds=bounds,
        )
    ).groups[0]
    assert unknown.verdict == VERDICT_SKIPPED
    assert "pixel_size" in unknown.reason
    assert "migrate-format" in unknown.reason

    out_of_range = screen_lod_store(
        _handmade_store(
            tmp_path / "level9.luxar.zarr",
            lod_attrs={"default_level": 9},
            group_bounds=bounds,
        )
    ).groups[0]
    assert out_of_range.verdict == VERDICT_SKIPPED
    assert "default_level=9" in out_of_range.reason


def test_a_present_but_falsy_selector_is_unsupported_not_legacy(
    tmp_path: Path,
) -> None:
    """``_plan_lod`` gates on ``selector is not None``, so an empty-string
    selector is unsupported there and ``restamp-lod`` writes nothing. Defaulting
    it to ``"coverage"`` here would score it against the legacy metric and could
    report a win for a rewrite that never happens."""
    bounds = {"c0": _plain_child(0, 10, 0.0), "c1": _plain_child(1, 40, 1.0)}
    falsy = screen_lod_store(
        _handmade_store(
            tmp_path / "empty-selector.luxar.zarr",
            lod_attrs={"selector": ""},
            group_bounds=bounds,
        )
    ).groups[0]
    assert falsy.verdict == VERDICT_SKIPPED
    assert "migrate-format" in falsy.reason


def test_an_off_screen_group_holds_its_default_level_not_index_zero(
    tmp_path: Path,
) -> None:
    """``coarsestReadyIndex`` is the first READY child, and on frame one that is
    the eagerly-committed ``default_level`` — every finer level sits behind a
    deferred loader. Index 0 is only the answer when ``default_level`` is 0."""
    away = {"min": [1e5, -1.0, -1.0], "max": [1e5 + 1, 1.0, 1.0]}
    group_bounds = {
        name: {
            **_plain_child(index, 10 * 4**index, float(index)),
            "position_bounds": away,
        }
        for index, name in enumerate(("c0", "c1", "c2"))
    }
    path = _handmade_store(
        tmp_path / "away-default2.luxar.zarr",
        lod_attrs={"default_level": 2},
        group_bounds=group_bounds,
    )
    group = screen_lod_store(path).groups[0]
    assert group.verdict == VERDICT_OFF_SCREEN
    assert all(m.off_screen for m in group.measurements)
    assert all(
        m.today_index == 2 and m.rederived_index == 2 for m in group.measurements
    )
    assert all(m.today_elements == 160 for m in group.measurements)


def test_a_cinematic_scene_is_skipped_unless_the_render_fov_is_supplied(
    tmp_path: Path,
) -> None:
    """The preset FOV lives in the viewer's TS table, so it is asked for, not guessed.

    Screening a 63-degree opening frame at the fit FOV 47 reads roughly twice the
    area — a whole halving of the derived ladder, enough to flip win↔no-op.
    """
    path = _handmade_store(
        tmp_path / "cinematic.luxar.zarr",
        group_bounds={"c0": _plain_child(0, 10, 0.0), "c1": _plain_child(1, 40, 1.0)},
    )
    root = open_group(path, mode="r+")
    root.attrs["viewer_config"] = {"cinematic_mode": True}
    consolidate(root)

    scene = screen_lod_store(path)
    assert scene.groups == []
    assert "cinematic_mode" in scene.skipped_reason
    assert "--screen-render-fov" in scene.skipped_reason
    # An explicit render FOV is the answer the message asks for, so it screens.
    assert screen_lod_store(path, render_fov=63.0).groups != []


def test_a_fov_preset_scene_is_skipped_but_a_numeric_fov_still_screens(
    tmp_path: Path,
) -> None:
    """``camera.fov_preset`` names a table entry; ``camera.fov`` is a number we honour."""
    path = _handmade_store(
        tmp_path / "preset.luxar.zarr",
        group_bounds={"c0": _plain_child(0, 10, 0.0), "c1": _plain_child(1, 40, 1.0)},
    )
    root = open_group(path, mode="r+")
    root.attrs["viewer_config"] = {"camera": {"fov_preset": "35mm"}}
    consolidate(root)
    assert "fov_preset" in screen_lod_store(path).skipped_reason

    # An author-set numeric fov wins whole (the bridge's CINEMATIC_FOV_PAIR
    # rule), so it is read as the render FOV rather than blocking the screen.
    root = open_group(path, mode="r+")
    root.attrs["viewer_config"] = {
        "cinematic_mode": True,
        "camera": {"fov_preset": "35mm", "fov": 63.0},
    }
    consolidate(root)
    numeric = screen_lod_store(path)
    assert numeric.skipped_reason == ""
    assert numeric.groups != []

    # A corpus-wide fallback must not replace the store's authored lens. The
    # bridge's CINEMATIC_FOV_PAIR rule still makes the numeric FOV authoritative.
    fallback = screen_lod_store(path, render_fov=47.0)
    assert [m.area_metric for m in fallback.groups[0].measurements] == pytest.approx(
        [m.area_metric for m in numeric.groups[0].measurements]
    )


def test_a_scene_with_one_displayed_dimension_is_skipped(tmp_path: Path) -> None:
    """``evaluatePerFrame`` bails at ``displayDims.length < 2``, before any group."""
    path = _handmade_store(
        tmp_path / "one-dim.luxar.zarr",
        group_bounds={"c0": _plain_child(0, 10, 0.0), "c1": _plain_child(1, 40, 1.0)},
    )
    root = open_group(path, mode="r+")
    root.attrs["scene_dimensions"] = {
        "dimensions": [
            {"name": "x", "display": True},
            {"name": "y", "display": False},
            {"name": "z", "display": False},
        ]
    }
    consolidate(root)
    scene = screen_lod_store(path)
    assert scene.groups == []
    assert "bails below 2" in scene.skipped_reason


def test_a_store_wide_skip_is_counted_in_the_tally(tmp_path: Path) -> None:
    """A ``❔`` line the footer counted nowhere read as ``0 skipped``."""
    report = screen_stores([tmp_path / "gone-a.luxar.zarr", tmp_path / "gone-b.zarr"])
    assert all("cannot read" in scene.skipped_reason for scene in report.scenes)
    assert report.tally[VERDICT_SKIPPED] == 2
    assert sum(report.tally.values()) == 2


def test_every_mirrored_constant_holds_its_typescript_value() -> None:
    """Each constant, pinned by LITERAL value against the TS twin it cites.

    The module header calls these "pinned by the colocated tests", and four of
    them were not: every ramp test scales WITH
    :data:`DEGENERATE_RECT_HALF_EXTENT` (its thicknesses are written as
    multiples of it), and the fit FOV, the viewport long axis and the ``w``
    epsilon appeared nowhere. A divergence in any of them silently makes the
    whole screen wrong, so the numbers are written out here rather than derived.
    """
    from luxar.io import lod_screening

    # scene/lod-selector-math.ts
    assert lod_screening.DEGENERATE_RECT_HALF_EXTENT == 1e-3
    assert lod_screening.W_EPSILON == 1e-6
    assert lod_screening.HYSTERESIS_RATIO == 0.1
    # scene/lod-group-registry.ts
    assert lod_screening.FILL_FACTOR == 0.5
    # config/sections/scene/data.ts :: config.scene.defaultFitRatio
    assert lod_screening.DEFAULT_FIT_RATIO == 0.75
    # config/sections/rendering-controls :: renderingControls.defaults.fov
    assert lod_screening.DEFAULT_FIT_FOV == 47.0
    # This module's own choice, but a report knob the docs quote.
    assert lod_screening.DEFAULT_VIEWPORT_LONG_PX == 1920
    assert lod_screening.DEFAULT_ASPECTS == (
        ("1:1", 1.0),
        ("16:9", 16.0 / 9.0),
        ("21:9", 21.0 / 9.0),
    )


def _union_store(path: Path, coarse_half: float, fine_half: float) -> Path:
    """A two-level ladder whose children carry independently sized bounds."""

    def child(index: int, n_points: int, half: float) -> dict:
        return {
            "type": "points",
            "n_points": n_points,
            "child_index": index,
            "coverage_fraction": float(index),
            "position_bounds": {
                "min": [-half, -half, -half],
                "max": [half, half, half],
            },
        }

    return _handmade_store(
        path,
        root_max=3.0,
        group_bounds={"c0": child(0, 10, coarse_half), "c1": child(1, 40, fine_half)},
    )


def test_the_group_box_is_the_union_over_children_not_the_last_one(
    tmp_path: Path,
) -> None:
    """A COARSER child's bounds can exceed the finest child's, and must still count.

    ``computeEntryWorldBox`` unions every usable child box. Folding the children
    with "last one wins" instead would silently read only the finest level's
    extent — which is the SMALLER one on a decimated-outlier ladder — so the
    group would be measured as if part of it were not on screen. All three
    stores are framed identically (the scene root is ±3 in each).
    """
    small = screen_lod_store(_union_store(tmp_path / "small.luxar.zarr", 1.0, 1.0))
    mixed = screen_lod_store(_union_store(tmp_path / "mixed.luxar.zarr", 3.0, 1.0))
    both = screen_lod_store(_union_store(tmp_path / "both.luxar.zarr", 3.0, 3.0))

    area = lambda scene: scene.groups[0].measurements[0].area_metric  # noqa: E731
    # "Last child wins" would make the mixed store read exactly like the small
    # one; the union makes it read exactly like the all-large one.
    assert area(mixed) == pytest.approx(area(both))
    assert area(mixed) > area(small)


def _nested_store(path: Path, *, parts: int) -> Path:
    """A ``kind=partition`` of ``parts`` ``kind=lod`` groups, written by hand."""
    root = open_group(path, mode="w")
    root.attrs.update(
        {
            "type": "scene",
            "position_bounds": {"min": [-5.0, -5.0, -5.0], "max": [5.0, 5.0, 5.0]},
        }
    )
    partition = root.create_group("tiled")
    partition.attrs.update({"kind": "partition", "type": "group"})
    for part in range(parts):
        lod = partition.create_group(f"part_{part}")
        lod.attrs.update({"kind": "lod", "type": "group", "default_level": 0})
        for index, count in enumerate((10, 40)):
            node = lod.create_group(f"c{index}")
            node.attrs.update(_plain_child(index, count, float(index)))
            create_array(
                node, "positions", data=np.zeros((1, 3), np.float32), compressor=None
            )
    consolidate(root)
    return path


def test_a_one_part_partition_does_not_tile_bind_its_ladder(tmp_path: Path) -> None:
    """The writers' ancestry clause needs a REAL tiling (>1 part), not a wrapper.

    A one-part partition's single part IS the whole object, so its ladder anchors
    at whole-object 0.5 like any other. Binding it would derive the ladder at the
    fills-screen anchor and report an anchor ``restamp-lod`` never uses — the
    exact state ``warn_one_part_partition_anchors`` exists to flag.
    """
    lone = screen_lod_store(_nested_store(tmp_path / "one.luxar.zarr", parts=1))
    assert [group.path for group in lone.groups] == ["tiled/part_0"]
    assert lone.groups[0].partition_bound is False
    assert "no partition ancestor" in lone.groups[0].anchor_reason
    assert lone.groups[0].rederived_thresholds[-1] == WHOLE_OBJECT_FINEST_ANCHOR

    real = screen_lod_store(_nested_store(tmp_path / "two.luxar.zarr", parts=2))
    assert all(group.partition_bound for group in real.groups)
    assert real.groups[0].rederived_thresholds[-1] == PARTITION_FINEST_AREA


def test_an_overview_cap_binds_on_its_own_partition_child(tmp_path: Path) -> None:
    """The writers' SECOND clause: a ``kind=partition`` among the ladder's children.

    That is the ``overview`` recipe's ``[coarse_leaf, fine_partition]`` cap, and
    missing it is the worst failure available here: derived at the whole-object
    anchor instead, the cap selects the fine partition at half-screen occupancy
    and loads the WHOLE dataset on frame one — the cost the recipe exists to
    avoid, on the largest stores there are.
    """
    path = tmp_path / "overview.luxar.zarr"
    root = open_group(path, mode="w")
    root.attrs.update(
        {
            "type": "scene",
            "position_bounds": {"min": [-5.0, -5.0, -5.0], "max": [5.0, 5.0, 5.0]},
        }
    )
    lod = root.create_group("overview")
    lod.attrs.update({"kind": "lod", "type": "group", "default_level": 0})

    coarse = lod.create_group("c0")
    coarse.attrs.update(_plain_child(0, 10, 0.0))
    create_array(
        coarse, "positions", data=np.zeros((1, 3), np.float32), compressor=None
    )

    fine = lod.create_group("c1")
    fine.attrs.update(
        {
            "type": "group",
            "kind": "partition",
            "child_index": 1,
            "coverage_fraction": 1.0,
            "position_bounds": {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
        }
    )
    for part, count in enumerate((30, 50)):
        node = fine.create_group(f"part_{part}")
        node.attrs.update(_plain_child(part, count, None))
        create_array(
            node, "positions", data=np.zeros((1, 3), np.float32), compressor=None
        )
    consolidate(root)

    group = screen_lod_store(path).groups[0]
    assert group.partition_bound is True
    assert "kind=partition ladder child" in group.anchor_reason
    assert group.rederived_thresholds[-1] == PARTITION_FINEST_AREA
    # The cap's fine branch is sized by the SUM over its parts, not one of them.
    assert group.element_counts == [10, 80]
