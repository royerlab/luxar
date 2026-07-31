"""Invariant tests for the ocean-currents demo's pure helpers.

The advection maths and the polyline index construction are where this demo can
be silently wrong (a ribbon wandering onto land, a segment bridging two
unrelated ribbons, a Lines node over the viewer's vertex ceiling), so those are
pinned here. No network and no IO.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.demos.demo_ocean_currents_earth import (
    MAX_LINE_VERTICES,
    N_SEEDS,
    N_STEPS,
    RADIUS,
    LonLatField,
    advect_streamlines,
    build_lut,
    fibonacci_sphere,
    globe_camera,
    lonlat_to_xyz,
    polyline_segment_indices,
    sample_equirect,
    seed_ocean_points,
)


def _uniform_eastward_field(nlat: int = 41, nlon: int = 80) -> LonLatField:
    """A wet-everywhere field flowing due east at 1 m/s."""
    lat = np.linspace(-80.0, 80.0, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    u = np.ones((nlat, nlon), dtype=np.float32)
    v = np.zeros((nlat, nlon), dtype=np.float32)
    return LonLatField(u, v, lon, lat)


def _half_land_field(nlat: int = 41, nlon: int = 80) -> LonLatField:
    """Eastward flow, but everything east of 180 deg is land (NaN)."""
    lat = np.linspace(-80.0, 80.0, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    u = np.ones((nlat, nlon), dtype=np.float32)
    v = np.zeros((nlat, nlon), dtype=np.float32)
    u[:, nlon // 2 :] = np.nan
    v[:, nlon // 2 :] = np.nan
    return LonLatField(u, v, lon, lat)


# --------------------------------------------------------------------- geometry


def test_lonlat_to_xyz_puts_points_on_the_sphere() -> None:
    lon = np.array([0.0, 90.0, -90.0, 180.0])
    lat = np.array([0.0, 45.0, -45.0, 80.0])
    xyz = lonlat_to_xyz(lon, lat, np.zeros(4))
    assert np.allclose(np.linalg.norm(xyz, axis=1), RADIUS, atol=1e-3)


def test_lonlat_to_xyz_is_right_handed_not_mirrored() -> None:
    """East x North must point outward, or the globe renders mirrored."""
    east = lonlat_to_xyz(np.array([1.0]), np.array([0.0]), np.zeros(1))[0]
    north = lonlat_to_xyz(np.array([0.0]), np.array([1.0]), np.zeros(1))[0]
    origin = lonlat_to_xyz(np.array([0.0]), np.array([0.0]), np.zeros(1))[0]
    outward = np.cross(east - origin, north - origin)
    assert float(np.dot(outward, origin)) > 0.0


def test_relief_scales_radius() -> None:
    xyz = lonlat_to_xyz(np.array([10.0]), np.array([20.0]), np.array([0.5]))
    assert np.linalg.norm(xyz[0]) == pytest.approx(RADIUS * 1.5, rel=1e-5)


def test_fibonacci_sphere_covers_both_hemispheres_uniformly() -> None:
    lon, lat = fibonacci_sphere(20_000, jitter=False)
    assert len(lon) == len(lat) == 20_000
    assert lat.min() < -80.0 and lat.max() > 80.0
    # area-uniform on a sphere => sin(lat) is uniform, so mean(sin) ~ 0
    assert abs(float(np.mean(np.sin(np.radians(lat))))) < 0.02


def test_fibonacci_jitter_is_deterministic_and_perturbs() -> None:
    a = fibonacci_sphere(5_000, jitter=True, seed=7)
    b = fibonacci_sphere(5_000, jitter=True, seed=7)
    plain = fibonacci_sphere(5_000, jitter=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    # jitter must actually move points (this is the anti-moire mechanism)
    assert not np.allclose(a[1], plain[1])
    # ...but by about one cell, not wildly
    cell = np.degrees(np.sqrt(4.0 * np.pi / 5_000))
    assert float(np.max(np.abs(a[1] - plain[1]))) < cell


def test_fibonacci_sphere_rejects_empty() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        fibonacci_sphere(0)


def test_globe_camera_looks_at_the_requested_point() -> None:
    cam = globe_camera(-84.0, 25.0)
    surface = lonlat_to_xyz(np.array([-84.0]), np.array([25.0]), np.zeros(1))[0]
    pos = np.array(cam.position)
    target = np.array(cam.target)
    # camera is outside the globe, target inside, both on the surface normal
    assert np.linalg.norm(pos) > RADIUS
    assert np.linalg.norm(target) < RADIUS
    unit_surface = surface / np.linalg.norm(surface)
    assert np.allclose(pos / np.linalg.norm(pos), unit_surface, atol=1e-6)


# ---------------------------------------------------------------------- texture


def test_sample_equirect_reads_expected_corners() -> None:
    """Row 0 is +90 lat and column 0 is -180 lon; a mix-up flips the globe."""
    tex = np.zeros((4, 8, 3), dtype=np.uint8)
    tex[0, 0] = (255, 0, 0)  # north-west
    tex[3, 0] = (0, 255, 0)  # south-west
    north_west = sample_equirect(tex, np.array([-179.0]), np.array([89.0]))[0]
    south_west = sample_equirect(tex, np.array([-179.0]), np.array([-89.0]))[0]
    assert north_west[0] > north_west[1]
    assert south_west[1] > south_west[0]


def test_sample_equirect_wraps_longitude() -> None:
    tex = np.random.default_rng(0).integers(0, 255, (8, 16, 3), dtype=np.uint8)
    a = sample_equirect(tex, np.array([179.9]), np.array([0.0]))
    b = sample_equirect(tex, np.array([179.9 + 360.0]), np.array([0.0]))
    assert np.allclose(a, b)


def test_sample_equirect_output_is_normalized() -> None:
    tex = np.full((4, 4, 3), 255, dtype=np.uint8)
    out = sample_equirect(tex, np.array([0.0, 45.0]), np.array([0.0, 10.0]))
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_build_lut_shape_and_endpoints() -> None:
    lut = build_lut([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])
    assert lut.shape == (256, 3) and lut.dtype == np.float32
    assert np.allclose(lut[0], 0.0) and np.allclose(lut[-1], 1.0)


# ------------------------------------------------------------------------ field


def test_field_rejects_mismatched_grids() -> None:
    lat = np.linspace(-10.0, 10.0, 5)
    lon = np.linspace(0.0, 350.0, 36)
    with pytest.raises(ValueError, match="shape mismatch"):
        LonLatField(
            np.zeros((5, 36), np.float32), np.zeros((5, 35), np.float32), lon, lat
        )
    with pytest.raises(ValueError, match="does not match"):
        LonLatField(
            np.zeros((4, 36), np.float32), np.zeros((4, 36), np.float32), lon, lat
        )


def test_field_sample_recovers_a_uniform_flow() -> None:
    field = _uniform_eastward_field()
    u, v = field.sample(np.array([12.0, 200.0]), np.array([0.0, -30.0]))
    assert np.allclose(u, 1.0, atol=1e-5)
    assert np.allclose(v, 0.0, atol=1e-5)


def test_field_land_is_dry_and_interpolates_without_nan() -> None:
    field = _half_land_field()
    assert bool(field.is_wet(np.array([10.0]), np.array([0.0]))[0]) is True
    assert bool(field.is_wet(np.array([300.0]), np.array([0.0]))[0]) is False
    # land must not poison interpolation with NaN
    u, v = field.sample(np.array([300.0]), np.array([0.0]))
    assert np.isfinite(u).all() and np.isfinite(v).all()


# -------------------------------------------------------------------- advection


def test_advection_step_length_is_fixed_arc_length() -> None:
    """The whole aesthetic depends on equal-length ribbons; pin the step size."""
    field = _uniform_eastward_field()
    step_km = 20.0
    lon, lat, _ = advect_streamlines(
        field, np.array([10.0]), np.array([0.0]), 6, step_km
    )
    xyz = lonlat_to_xyz(lon[0], lat[0], np.zeros(lon.shape[1]))
    seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    expected = RADIUS * np.radians(np.degrees(step_km / 6371.0))
    assert np.allclose(seg, expected, rtol=2e-3)


def test_advection_follows_the_flow_direction() -> None:
    field = _uniform_eastward_field()
    lon, lat, speed = advect_streamlines(
        field, np.array([10.0]), np.array([0.0]), 5, 25.0
    )
    assert np.all(np.diff(lon[0]) > 0.0)  # eastward
    assert np.allclose(lat[0], 0.0, atol=1e-6)  # no meridional drift
    assert np.allclose(speed, 1.0, atol=1e-5)


def test_longitude_step_grows_with_latitude() -> None:
    """A fixed arc length must span more degrees of longitude near the poles."""
    field = _uniform_eastward_field()
    lon_eq, _, _ = advect_streamlines(field, np.array([0.0]), np.array([0.0]), 1, 40.0)
    lon_hi, _, _ = advect_streamlines(field, np.array([0.0]), np.array([60.0]), 1, 40.0)
    assert (lon_hi[0, 1] - lon_hi[0, 0]) > 1.8 * (lon_eq[0, 1] - lon_eq[0, 0])


def test_streamlines_never_walk_onto_land() -> None:
    """Seeded in open water heading at a coast, no vertex may end up on land."""
    field = _half_land_field()
    seed_lon = np.linspace(120.0, 170.0, 40)  # wet, flowing east into the coast
    seed_lat = np.zeros(40)
    assert field.is_wet(seed_lon, seed_lat).all(), "fixture seeds must start wet"
    lon, lat, _ = advect_streamlines(field, seed_lon, seed_lat, 40, 30.0)
    assert field.is_wet(lon.ravel(), lat.ravel()).all()


def test_beached_streamlines_freeze_rather_than_drift() -> None:
    field = _half_land_field()
    lon, lat, _ = advect_streamlines(
        field, np.array([170.0]), np.array([0.0]), 60, 30.0
    )
    # once frozen, every later vertex is identical (zero-length segments)
    assert lon[0, -1] == pytest.approx(lon[0, -2])
    assert lat[0, -1] == pytest.approx(lat[0, -2])
    assert field.is_wet(lon.ravel(), lat.ravel()).all()


def test_dry_seeds_raise_instead_of_emitting_land_ribbons() -> None:
    """A land seed would freeze at its start point and draw a ribbon on land.

    Only the *next* position is masked per step, so this has to be rejected up
    front or it renders as a plausible-looking but wrong streak over a continent.
    """
    field = _half_land_field()
    with pytest.raises(ValueError, match="on land"):
        advect_streamlines(field, np.array([10.0, 300.0]), np.zeros(2), 5, 20.0)


def test_advection_shapes_and_dtypes() -> None:
    field = _uniform_eastward_field()
    lon, lat, speed = advect_streamlines(field, np.zeros(7), np.zeros(7), 11, 15.0)
    for arr in (lon, lat, speed):
        assert arr.shape == (7, 12) and arr.dtype == np.float32


# ------------------------------------------------------------------ line topology


def test_segment_indices_never_bridge_two_ribbons() -> None:
    """A segment crossing a path boundary would draw a line across the ocean."""
    n_paths, n_vertices = 5, 4
    idx = polyline_segment_indices(n_paths, n_vertices).reshape(-1, 2)
    assert len(idx) == n_paths * (n_vertices - 1)
    assert np.all(idx[:, 1] == idx[:, 0] + 1)
    # no pair may straddle a multiple of n_vertices
    assert not np.any(idx[:, 1] % n_vertices == 0)
    assert idx.max() == n_paths * n_vertices - 1


def test_segment_indices_are_uint32_for_the_writer() -> None:
    assert polyline_segment_indices(3, 3).dtype == np.uint32


def test_segment_indices_reject_degenerate_paths() -> None:
    with pytest.raises(ValueError, match="n_vertices must be >= 2"):
        polyline_segment_indices(4, 1)


def test_configured_budget_respects_the_viewer_vertex_ceiling() -> None:
    """N_SEEDS x (N_STEPS+1) must stay under the loader's 2**24 Set limit.

    Above it the Lines node renders nothing at all (royerlab/luxar#1049), so this
    guards the demo's own configuration, not just the helper.
    """
    assert N_SEEDS * (N_STEPS + 1) <= MAX_LINE_VERTICES


# --------------------------------------------------------------------- seeding


def test_seeds_land_only_in_moving_water() -> None:
    field = _half_land_field()
    lon, lat = seed_ocean_points(field, 500, seed=3)
    assert len(lon) == len(lat) == 500
    assert field.is_wet(lon, lat).all()
    u, v = field.sample(lon, lat)
    assert np.all(np.hypot(u, v) > 0.0)


def test_seeds_are_area_uniform_not_pole_clustered() -> None:
    field = _uniform_eastward_field()
    _, lat = seed_ocean_points(field, 20_000, seed=5)
    # uniform in sin(lat) => mean(sin(lat)) ~ 0 for a symmetric band
    assert abs(float(np.mean(np.sin(np.radians(lat))))) < 0.03


def test_seeding_is_deterministic() -> None:
    field = _uniform_eastward_field()
    a = seed_ocean_points(field, 200, seed=11)
    b = seed_ocean_points(field, 200, seed=11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
