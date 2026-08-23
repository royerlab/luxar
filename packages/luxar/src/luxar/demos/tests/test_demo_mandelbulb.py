import numpy as np

from luxar.demos.demo_mandelbulb import (
    _mandelbulb_distance_and_orbit_trap,
    _mandelbulb_surface_appearance,
    mandelbulb_distance_estimate,
)


def _surface_sample(resolution: int = 36) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.linspace(-1.3, 1.3, resolution)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    distances, _, orbit_trap = _mandelbulb_distance_and_orbit_trap(points)
    near_surface = (distances > 0.0) & (distances < 0.01)
    return points[near_surface], distances[near_surface], orbit_trap[near_surface]


def test_distance_estimate_api_remains_backward_compatible() -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.2, 0.1, -0.3]])

    distances, iterations = mandelbulb_distance_estimate(points)

    assert distances.shape == (2,)
    assert iterations.shape == (2,)


def test_surface_appearance_has_broad_colour_and_lighting_range() -> None:
    positions, distances, orbit_trap = _surface_sample()

    colors, lighting, ambient_occlusion = _mandelbulb_surface_appearance(
        positions,
        distances,
        orbit_trap,
    )

    dominant_share = np.bincount(np.argmax(colors, axis=1), minlength=3) / len(colors)
    distinct_colors = len(np.unique(np.round(colors, 3), axis=0))

    assert len(colors) > 500
    assert dominant_share.max() < 0.75
    assert distinct_colors > 128
    assert np.ptp(lighting) > 0.45
    assert np.ptp(ambient_occlusion) > 0.1
    assert np.all(np.isfinite(colors))
    assert np.all((colors >= 0.0) & (colors <= 1.0))


def test_ambient_occlusion_darkens_more_enclosed_surface_points() -> None:
    positions, distances, orbit_trap = _surface_sample()

    colors, lighting, ambient_occlusion = _mandelbulb_surface_appearance(
        positions,
        distances,
        orbit_trap,
    )

    exposed = ambient_occlusion >= np.quantile(ambient_occlusion, 0.9)
    enclosed = ambient_occlusion <= np.quantile(ambient_occlusion, 0.1)
    assert lighting[enclosed].mean() < lighting[exposed].mean()
    assert colors[enclosed].mean() < colors[exposed].mean()
