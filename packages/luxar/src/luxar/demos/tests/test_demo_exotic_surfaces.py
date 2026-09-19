"""Tests for the exotic-surfaces demo.

Eighteen hand-transcribed equations is eighteen chances to fat-finger a
coefficient, and a wrong one does not crash — it quietly renders a different
(often plausible-looking) surface. So the tests check each surface actually has a
zero set of the right size, that the numerical gradient is trustworthy, and that
the sprite and layer appearance that made the previous version of this demo
illegible cannot come back.
"""

import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import zarr

from luxar.demos import demo_exotic_surfaces as _demo
from luxar.demos.demo_exotic_surfaces import (
    AO_RADIUS_FRACTION,
    AO_STRENGTH,
    CELL_PITCH,
    CELL_SIZE,
    DETAIL_FONT_SIZE,
    DETAIL_LEADING,
    DETAIL_TOP,
    FAMILY_COLORS,
    FAMILY_DISPLAY_MAXIMA,
    FAMILY_NAMES,
    GRID,
    OCCLUDER,
    SPRITE_OVERLAP,
    SUBTITLES,
    SURFACES,
    Surface,
    _detail_lines,
    _gyroid,
    cell_offset,
    generate_exotic_surfaces,
    implicit_normals,
    sample_surface,
)
from luxar.shading import bake_ambient_occlusion

RESOLUTION = 44


def _gyroid_analytic_gradient(p: np.ndarray) -> np.ndarray:
    """The one gradient in this demo that is known in closed form."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return np.stack(
        [
            np.cos(x) * np.cos(y) - np.sin(z) * np.sin(x),
            -np.sin(x) * np.sin(y) + np.cos(y) * np.cos(z),
            -np.sin(y) * np.sin(z) + np.cos(z) * np.cos(x),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# The numerical gradient everything else leans on
# ---------------------------------------------------------------------------


def test_numerical_normals_match_the_one_analytic_gradient_we_have():
    """Validates the shared routine so eighteen hand-derived ones are not needed.

    If this holds for the gyroid it holds for the rest: the routine has no
    per-surface behaviour, and every field here is smooth on its domain.
    """
    rng = np.random.default_rng(0)
    points = rng.uniform(-3.0, 3.0, size=(256, 3))

    numeric = implicit_normals(_gyroid, points, 1e-5)
    analytic = _gyroid_analytic_gradient(points)
    analytic /= np.linalg.norm(analytic, axis=1, keepdims=True)

    np.testing.assert_allclose(numeric, analytic, atol=1e-6)


def test_surface_sampling_is_stable_across_python_hash_seeds():
    """The baked dataset must not move between Python processes."""
    script = """
import hashlib
from luxar.demos.demo_exotic_surfaces import SURFACES, sample_surface
positions, _, _ = sample_surface(SURFACES[6], 44)
print(hashlib.sha256(positions.tobytes()).hexdigest())
"""
    digests = []
    for hash_seed in ("1", "2"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        digests.append(result.stdout.strip())

    assert digests[0] == digests[1]


# ---------------------------------------------------------------------------
# Every surface is a real, non-degenerate surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.key)
def test_surface_has_a_usable_zero_set(surface: Surface):
    """A mistyped coefficient usually shows up as an empty or space-filling set.

    Both failure modes are silent at render time — an empty cell just looks like
    a cell that was not authored — so they are caught here instead.
    """
    positions, normals, spacing = sample_surface(surface, RESOLUTION)

    assert len(positions) > 200, f"{surface.key}: nearly empty zero set"

    # A SHELL, not a solid — asserted on how the retained fraction SCALES rather
    # than against an absolute cap. The shell is a fixed number of spacings
    # thick, so it keeps ~area/spacing^2 of ~1/spacing^3 samples and the fraction
    # falls linearly with resolution: `fraction * resolution` is then a
    # resolution-independent shape index, measured at 10.7-11.2 for every surface
    # here across resolutions 32-88. A solid would sit at `resolution` itself
    # (44 here), so 20 separates the two cleanly at any resolution. An absolute
    # cap instead just encodes whichever resolution the test was written at, and
    # one surface was already sitting on it.
    shape_index = len(positions) / RESOLUTION**2
    assert shape_index < 20.0, f"{surface.key}: fills its box ({shape_index:.1f})"
    assert shape_index > 0.5, f"{surface.key}: too sparse ({shape_index:.2f})"
    assert spacing > 0.0
    assert np.all(np.isfinite(positions))
    # Normals are unit length wherever the gradient does not vanish; a nodal
    # singularity legitimately has a zero gradient, so some rows may be zero.
    lengths = np.linalg.norm(normals, axis=1)
    assert np.all((np.abs(lengths - 1.0) < 1e-9) | (lengths == 0.0))
    assert (lengths > 0).mean() > 0.9, f"{surface.key}: mostly degenerate normals"


@pytest.mark.parametrize("surface", SURFACES, ids=lambda s: s.key)
def test_surface_is_normalized_into_its_cell(surface: Surface):
    """Every surface must occupy the same cell, or the grid reads as a jumble."""
    positions, _, _ = sample_surface(surface, RESOLUTION)

    span = positions.max(axis=0) - positions.min(axis=0)
    assert np.all(span <= CELL_SIZE + 1e-5), f"{surface.key}: {span} exceeds the cell"
    # The longest axis is scaled TO the cell, so it must actually reach it.
    assert span.max() == pytest.approx(CELL_SIZE, abs=1e-5)
    # Centred on the origin before placement.
    centre = 0.5 * (positions.max(axis=0) + positions.min(axis=0))
    np.testing.assert_allclose(centre, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# The collection itself
# ---------------------------------------------------------------------------


def test_two_families_of_exactly_nine():
    counts = [sum(1 for s in SURFACES if s.family == f) for f in (0, 1)]
    assert counts == [GRID * GRID, GRID * GRID]
    assert len(FAMILY_NAMES) == len(FAMILY_COLORS) == len(FAMILY_DISPLAY_MAXIMA) == 2


def test_every_surface_is_credited_and_described():
    """The demo's premise is that each surface has a stateable property."""
    for surface in SURFACES:
        assert surface.credit.strip(), f"{surface.key} has no credit"
        assert len(surface.note) > 25, f"{surface.key}'s note says too little"
        # A credit should name a year, which is what makes it a citation rather
        # than a label. "Banchoff" is the one deliberate exception.
        assert any(c.isdigit() for c in surface.credit) or surface.key == "tanglecube"


def test_detail_lines_fit_on_one_line():
    """Each detail is its own overlay, and an overlay cannot break its own lines.

    Detail overlays have no `width`, so `overlay-manager.ts` sets `nowrap`; the
    lines are stacked separately and capped so they do not overflow their
    right-anchored edge.
    """
    for family in (0, 1):
        lines = _detail_lines(family)
        assert len(lines) == GRID * GRID
        for line in lines:
            assert "\n" not in line, "a newline here would render as a space"
            assert len(line) <= 95, f"too long to sit on one line: {line!r}"


def test_overlay_explains_detail_order():
    assert all("reading order" in subtitle for subtitle in SUBTITLES)


def test_detail_stack_clears_the_caption_and_the_nav_panel():
    """Nine hand-stacked lines must not collide with the rest of the UI."""
    bottom = DETAIL_TOP + (GRID * GRID - 1) * DETAIL_LEADING
    assert bottom < 0.75, f"detail stack reaches {bottom:.2f}, into the nav panel"
    assert DETAIL_LEADING > DETAIL_FONT_SIZE, "lines would overlap each other"


def test_surface_keys_and_titles_are_unique():
    assert len({s.key for s in SURFACES}) == len(SURFACES)
    assert len({s.title for s in SURFACES}) == len(SURFACES)


def test_cells_tile_without_overlapping():
    """Nine distinct centres, spaced so normalized cells cannot intersect."""
    offsets = [cell_offset(i) for i in range(GRID * GRID)]

    assert len(set(offsets)) == GRID * GRID
    assert CELL_PITCH > CELL_SIZE, "cells would overlap"
    for i, a in enumerate(offsets):
        for b in offsets[i + 1 :]:
            gap = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            assert gap >= CELL_PITCH - 1e-9

    # Reading order: first cell top-left, last bottom-right.
    assert offsets[0][0] < offsets[2][0]
    assert offsets[0][1] > offsets[6][1]


# ---------------------------------------------------------------------------
# The appearance bugs that made the previous demo illegible
# ---------------------------------------------------------------------------


def test_generated_layers_pin_the_authored_volumetric_appearance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        _demo,
        "bake_ambient_occlusion",
        lambda points, **_kwargs: np.ones(len(points), dtype=np.float32),
    )

    expected_counts = [len(sample_surface(surface, 8)[0]) for surface in SURFACES]
    assert all(expected_counts), "every surface must reach the write path"

    output = tmp_path / "exotic.luxar.zarr"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert generate_exotic_surfaces(output, resolution=8) == sum(expected_counts)

    root = zarr.open_group(str(output), mode="r")
    for family_name, display_max in zip(FAMILY_NAMES, (2.177, 2.085), strict=True):
        attrs = dict(root[family_name].attrs)
        assert attrs["blending_mode"] == "volumetric"
        assert attrs["opacity"] == pytest.approx(0.60)
        assert attrs["absorption"] == pytest.approx(1.0)
        assert attrs["gamma"] == pytest.approx(1.0)
        assert attrs["intensity"] == pytest.approx(1.0 / display_max)
        assert attrs.get("offset", 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("surface", SURFACES[:3] + SURFACES[9:12], ids=lambda s: s.key)
def test_sprites_overlap_so_no_surface_renders_as_a_dot_screen(surface: Surface):
    """Render radius must span the sample spacing.

    Below that the sprites never touch and the surface renders as stipple, whose
    per-pixel on/off contrast drowns the occlusion gradient entirely — which is
    exactly what went wrong before this demo was rebuilt.
    """
    c_kd_tree = pytest.importorskip("scipy.spatial").cKDTree
    positions, _, spacing = sample_surface(surface, RESOLUTION)
    distances, _ = c_kd_tree(positions).query(positions, k=2)
    actual_spacing = float(np.median(distances[:, 1]))
    diameter = 2.0 * SPRITE_OVERLAP * spacing

    assert diameter > actual_spacing, f"{surface.key} would stipple"
    assert diameter < 3.0 * actual_spacing, f"{surface.key} would smear"


# ---------------------------------------------------------------------------
# The occlusion actually does something on these surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", [SURFACES[0], SURFACES[9]], ids=lambda s: s.key)
def test_occlusion_spans_a_visible_range(surface: Surface):
    """A near-constant multiplier would look identical to no bake at all."""
    positions, normals, _ = sample_surface(surface, 64)

    shade = bake_ambient_occlusion(
        positions,
        normals=normals,
        occluder=OCCLUDER,
        radius=AO_RADIUS_FRACTION * CELL_SIZE,
        strength=AO_STRENGTH,
    )

    spread = float(np.percentile(shade, 95) - np.percentile(shade, 5))
    assert spread > 0.25, f"{surface.key}: only {spread:.3f} of range used"
    assert shade.max() <= 1.0
    assert shade.min() >= 0.0


# ---------------------------------------------------------------------------
# The family ladder
#
# Two nodes on a two-stop categorical axis, each carrying only its own
# coordinate. `family_ladder` therefore applies the 1/8 resident share BY HAND
# rather than through `slices=`, which would also relax the commit ceiling to
# 1,800,000 on a node with no second slice to spend it on. Every number below is
# measured: the first two against the built store, the third against a
# `--resolution=172` point count.
# ---------------------------------------------------------------------------


def _ladder_increments(counts: list[int]) -> list[int]:
    return [
        cut - previous for previous, cut in zip([0, *counts[:-1]], counts, strict=True)
    ]


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (857_226, [107_154, 214_308, 428_616, 857_226]),  # Minimal surfaces
        (464_073, [58_010, 116_020, 232_040, 464_073]),  # Algebraic surfaces
    ],
)
def test_family_ladder_opens_on_an_eighth_of_the_wall(
    points: int, expected: list[int]
) -> None:
    """The shipped cuts at the authored RESOLUTION, and why they are those.

    The unsliced download budget would open on 39,062 points — 4.56% of the
    minimal family and 8.42% of the algebraic one, both under the ladder gate's
    10% rung-0 share floor for a node the viewer slices.
    """
    counts = _demo.family_ladder(points)["counts"]

    assert counts == expected
    assert counts[0] == -(-points // 8) > 39_062


@pytest.mark.parametrize("points", [857_226, 2_023_609, 6_000_000])
def test_family_ladder_stays_under_the_gates_absolute_commit_cap(
    points: int,
) -> None:
    """A high-resolution build must not author a level the gate fails.

    Counts scale about as ``resolution**2`` (measured: the minimal family is
    857,226 points at 112 and 2,023,609 at 172), and each node holds ONE hidden
    coordinate — so the gate's largest-coordinate-fetch measurement is the whole
    level and its 1,000,000 cap applies to these increments directly. Borrowing
    ``slices=2`` for the share floor also doubled the commit ceiling to
    1,800,000, which authors a 1,011,801-element increment at 172.
    """
    increments = _ladder_increments(_demo.family_ladder(points)["counts"])

    assert max(increments) <= 900_000


def test_family_ladder_leaves_a_tiny_node_flat() -> None:
    """The `resolution=8` build path: nothing to stream, so no rungs."""
    assert _demo.family_ladder(100)["counts"] == [100]
