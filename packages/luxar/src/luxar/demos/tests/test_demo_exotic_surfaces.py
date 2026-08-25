"""Tests for the exotic-surfaces demo.

Eighteen hand-transcribed equations is eighteen chances to fat-finger a
coefficient, and a wrong one does not crash — it quietly renders a different
(often plausible-looking) surface. So the tests check each surface actually has a
zero set of the right size, that the numerical gradient is trustworthy, and that
the sprite/exposure arithmetic that made the previous version of this demo
illegible cannot come back.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

from luxar.demos.demo_exotic_surfaces import (
    AO_RADIUS_FRACTION,
    AO_STRENGTH,
    CELL_PITCH,
    CELL_SIZE,
    FAMILY_COLORS,
    FAMILY_NAMES,
    GRID,
    OCCLUDER,
    SPRITE_OVERLAP,
    SURFACES,
    TARGET_PEAK,
    Surface,
    _gyroid,
    auto_exposure,
    cell_offset,
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
    # A shell, not a solid: well under the whole sampled volume.
    assert len(positions) < 0.25 * RESOLUTION**3, f"{surface.key}: fills its box"
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
    assert len(FAMILY_NAMES) == len(FAMILY_COLORS) == 2


def test_every_surface_is_credited_and_described():
    """The demo's premise is that each surface has a stateable property."""
    for surface in SURFACES:
        assert surface.credit.strip(), f"{surface.key} has no credit"
        assert len(surface.note) > 40, f"{surface.key}'s note says too little"
        # A credit should name a year, which is what makes it a citation rather
        # than a label. "Banchoff" is the one deliberate exception.
        assert any(c.isdigit() for c in surface.credit) or surface.key == "tanglecube"


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
# The two arithmetic bugs that made the previous demo illegible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", SURFACES[:3] + SURFACES[9:12], ids=lambda s: s.key)
def test_sprites_overlap_so_no_surface_renders_as_a_dot_screen(surface: Surface):
    """Render radius must span the sample spacing.

    Below that the sprites never touch and the surface renders as stipple, whose
    per-pixel on/off contrast drowns the occlusion gradient entirely — which is
    exactly what went wrong before this demo was rebuilt.
    """
    positions, _, spacing = sample_surface(surface, RESOLUTION)
    distances, _ = cKDTree(positions).query(positions, k=2)
    actual_spacing = float(np.median(distances[:, 1]))
    diameter = 2.0 * SPRITE_OVERLAP * spacing

    assert diameter > actual_spacing, f"{surface.key} would stipple"
    assert diameter < 3.0 * actual_spacing, f"{surface.key} would smear"


def test_auto_exposure_holds_the_deepest_sightline_under_white():
    positions, _, spacing = sample_surface(SURFACES[0], RESOLUTION)
    radius = SPRITE_OVERLAP * spacing
    intensity, deepest = auto_exposure(positions, radius)

    expected_counts: dict[tuple[int, int], int] = {}
    for x, y in positions[:, :2]:
        key = (
            int(np.floor(x / (2.0 * radius))),
            int(np.floor(y / (2.0 * radius))),
        )
        expected_counts[key] = expected_counts.get(key, 0) + 1
    assert deepest == max(expected_counts.values())

    peak = deepest * intensity * float(max(c.max() for c in FAMILY_COLORS))
    assert peak == pytest.approx(TARGET_PEAK, abs=1e-6)
    assert peak < 1.0


def test_auto_exposure_dims_as_the_surface_gets_denser():
    sparse, _, sp_s = sample_surface(SURFACES[0], 40)
    dense, _, sp_d = sample_surface(SURFACES[0], 72)

    assert (
        auto_exposure(dense, SPRITE_OVERLAP * sp_d)[0]
        < auto_exposure(sparse, SPRITE_OVERLAP * sp_s)[0]
    )


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
