"""Tests for the ambient-occlusion A/B demo.

The demo's job is to make one comparison legible, so the tests pin the things
that would silently break the comparison rather than merely break the run: the
surface being the real gyroid, the normals being exact, the auto-exposure
actually holding the render under white, and the hemisphere panel genuinely
carrying more contrast than the sphere panel (which is the point of showing both).
"""

import numpy as np
from scipy.spatial import cKDTree

from luxar.demos.demo_ambient_occlusion import (
    AO_RADIUS,
    AO_STRENGTH,
    BASE_COLOR,
    OCCLUDER,
    TARGET_PEAK,
    auto_exposure,
    gyroid_field,
    point_radius,
    sample_gyroid_surface,
)
from luxar.shading import bake_ambient_occlusion

RESOLUTION = 48


def test_gyroid_gradient_matches_a_finite_difference() -> None:
    """The normals are advertised as exact; check them against numerics."""
    rng = np.random.default_rng(0)
    points = rng.uniform(-6.0, 6.0, size=(64, 3))

    _, analytic = gyroid_field(points)

    eps = 1e-6
    numeric = np.empty_like(analytic)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = eps
        plus, _ = gyroid_field(points + offset)
        minus, _ = gyroid_field(points - offset)
        numeric[:, axis] = (plus - minus) / (2.0 * eps)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_sampled_points_lie_on_the_gyroid_surface() -> None:
    positions, normals = sample_gyroid_surface(RESOLUTION)

    assert len(positions) > 0
    values, gradients = gyroid_field(positions.astype(np.float64))
    distance = np.abs(values) / np.linalg.norm(gradients, axis=1)
    # Every retained point is within the shell half-thickness of the zero set.
    assert distance.max() < 0.07
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)


def test_normals_are_perpendicular_to_the_surface() -> None:
    """A normal must be parallel to the gradient, i.e. the level-set normal."""
    positions, normals = sample_gyroid_surface(RESOLUTION)
    _, gradients = gyroid_field(positions.astype(np.float64))
    unit = gradients / np.linalg.norm(gradients, axis=1, keepdims=True)

    np.testing.assert_allclose(np.abs(np.sum(normals * unit, axis=1)), 1.0, atol=1e-9)


def test_sprites_overlap_so_the_surface_is_not_a_dot_screen() -> None:
    """The regression that made the whole demo pointless.

    A render radius below half the sample spacing leaves the sprites not even
    touching, and the surface renders as stipple. That per-pixel on/off contrast
    is far stronger than the smooth occlusion gradient beneath it, so the eye
    reads noise and no amount of extra AO strength helps. Diameter must exceed
    the spacing, at every resolution.
    """
    for resolution in (40, 80, 120):
        positions, _ = sample_gyroid_surface(resolution)
        distances, _ = cKDTree(positions).query(positions, k=2)
        spacing = float(np.median(distances[:, 1]))
        diameter = 2.0 * point_radius(resolution)
        assert diameter > spacing, (
            f"resolution {resolution}: sprite diameter {diameter:.4f} does not "
            f"span the {spacing:.4f} sample spacing — surface will stipple"
        )
        # And not so large that the fine channel walls smear together.
        assert diameter < 3.0 * spacing


def test_auto_exposure_keeps_the_deepest_sightline_under_white() -> None:
    """The guard that stops the render clipping the shading flat.

    A fixed gain cannot do this: the deepest column grows with resolution, so the
    check runs at two resolutions and both must land on the target.
    """
    for resolution in (40, 64):
        positions, _ = sample_gyroid_surface(resolution)
        radius = point_radius(resolution)
        intensity, deepest = auto_exposure(positions, radius)

        expected_counts: dict[tuple[int, int], int] = {}
        for x, y in positions[:, :2]:
            key = (
                int(np.floor(x / (2.0 * radius))),
                int(np.floor(y / (2.0 * radius))),
            )
            expected_counts[key] = expected_counts.get(key, 0) + 1
        assert deepest == max(expected_counts.values())

        peak = deepest * intensity * float(BASE_COLOR.max())
        assert peak < 1.0, f"resolution {resolution} clips at {peak:.2f}"
        assert np.isclose(peak, TARGET_PEAK, atol=1e-6)


def test_auto_exposure_dims_as_the_surface_gets_denser() -> None:
    sparse, _ = sample_gyroid_surface(40)
    dense, _ = sample_gyroid_surface(80)

    assert (
        auto_exposure(dense, point_radius(80))[0]
        < auto_exposure(sparse, point_radius(40))[0]
    )


def test_hemisphere_panel_carries_more_contrast_than_the_sphere_panel() -> None:
    """Why the demo shows both panels rather than one.

    On a thin shell the full sphere is diluted by the in-plane material every
    point shares. If this ever stopped holding, the third panel would be
    redundant and the demo's caption would be wrong.
    """
    positions, normals = sample_gyroid_surface(64)

    sphere = bake_ambient_occlusion(
        positions,
        radius=AO_RADIUS,
        strength=AO_STRENGTH,
        occluder=OCCLUDER,
    )
    hemisphere = bake_ambient_occlusion(
        positions,
        normals=normals,
        radius=AO_RADIUS,
        strength=AO_STRENGTH,
        occluder=OCCLUDER,
    )

    assert hemisphere.std() > 1.5 * sphere.std()
    # Both must still be genuine multipliers, not a constant darkening.
    assert sphere.std() > 0.01
    assert hemisphere.max() <= 1.0
    assert sphere.min() > 0.0


def test_occlusion_varies_across_the_surface() -> None:
    """A bake that returned near-constant values would look identical to no bake."""
    positions, normals = sample_gyroid_surface(64)

    shade = bake_ambient_occlusion(
        positions,
        normals=normals,
        radius=AO_RADIUS,
        strength=AO_STRENGTH,
        occluder=OCCLUDER,
    )

    spread = float(np.percentile(shade, 95) - np.percentile(shade, 5))
    assert spread > 0.2, f"only {spread:.3f} of range used"
