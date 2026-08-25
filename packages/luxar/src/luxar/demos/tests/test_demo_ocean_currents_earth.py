"""Invariant tests for the ocean-currents demo's pure helpers.

The advection maths and the polyline index construction are where this demo can
be silently wrong (a ribbon wandering onto land, a segment bridging two
unrelated ribbons, a Lines node overflowing the viewer's per-segment element
texture), so those are pinned here. No network and no IO.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from luxar._zarr_compat import consolidate, open_group
from luxar.demos import demo_global_rivers_earth, demo_ocean_currents_earth
from luxar.demos._globe_common import surface_point_radius
from luxar.demos.demo_ocean_currents_earth import (
    CURRENT_TILE_RIBBONS,
    FLOW_LIFT,
    GLOBE_COARSEST_RADII,
    GLOBE_RADII,
    GLOBE_TILE_POINTS,
    LINE_OPACITY,
    LOD_COMPRESSION,
    LOD_LEVELS,
    N_GLOBE,
    N_SEEDS,
    N_STEPS,
    RADIUS,
    SHELL_SEAL_MARGIN,
    LonLatField,
    advect_streamlines,
    build_lut,
    check_tile_budget,
    fibonacci_sphere,
    globe_camera,
    level_seed,
    level_subset,
    lod_counts,
    lonlat_to_xyz,
    polyline_segment_indices,
    sample_equirect,
    seal_margin,
    seed_ocean_points,
    tile_coverage,
    write_current_parts,
    write_globe_parts,
)
from luxar.typing_utils.constants import (
    MAX_POINTS_PER_POINTS_NODE,
    MAX_SEGMENTS_PER_LINES_NODE,
)


@pytest.mark.parametrize(
    "demo_module,download_name",
    [
        (demo_ocean_currents_earth, "download_sources"),
        (demo_global_rivers_earth, "_download_sources"),
    ],
)
def test_scene_gate_rebuilds_only_for_a_stale_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    demo_module: ModuleType,
    download_name: str,
) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    group = open_group(output_path, mode="w")
    group.attrs["builder_fingerprint"] = "older-builder"
    consolidate(group)

    reached_download = False

    def stop_at_download() -> tuple[Path, Path]:
        nonlocal reached_download
        reached_download = True
        raise RuntimeError("download reached")

    monkeypatch.setattr(demo_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(demo_module, "RECOMPUTE", False)
    monkeypatch.setattr(demo_module, "KEEP_STALE", False)
    monkeypatch.setattr(demo_module, download_name, stop_at_download)

    with pytest.raises(RuntimeError, match="download reached"):
        demo_module.load_or_build_scene(output_path)
    assert reached_download

    group = open_group(output_path, mode="a")
    group.attrs["builder_fingerprint"] = demo_module.FINGERPRINT
    consolidate(group)
    reached_download = False

    assert demo_module.load_or_build_scene(output_path) == output_path
    assert not reached_download


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


def _dry_band_field() -> tuple[LonLatField, float]:
    """Fine (~0.08 deg) grid, northward 1 m/s flow, a one-cell dry band at lat 0.

    The band is a SINGLE latitude row — narrower than the 14 km STEP_KM arc — so
    an endpoint-only wetness check would hop clean over it. Returns the field and
    the band's centre latitude.
    """
    nlat, nlon = 401, 16
    lat = np.linspace(-16.0, 16.0, nlat)  # dlat = 0.08 deg
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    u = np.zeros((nlat, nlon), dtype=np.float32)
    v = np.ones((nlat, nlon), dtype=np.float32)  # due north
    j0 = int(np.argmin(np.abs(lat)))
    u[j0, :] = np.nan
    v[j0, :] = np.nan
    return LonLatField(u, v, lon, lat), float(lat[j0])


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


def test_surface_point_radius_tracks_spherical_sample_spacing() -> None:
    base = surface_point_radius(500_000, 1.0, overlap=1.0)
    assert surface_point_radius(2_000_000, 1.0, overlap=1.0) == pytest.approx(
        base / 2.0
    )
    assert surface_point_radius(2_000_000, 1.0, overlap=1.0) == pytest.approx(
        0.0025, rel=0.01
    )


def test_surface_point_radius_rejects_empty() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        surface_point_radius(0, 1.0)


def test_globe_camera_looks_at_the_requested_point() -> None:
    cam = globe_camera(-84.0, 25.0)
    surface = lonlat_to_xyz(np.array([-84.0]), np.array([25.0]), np.zeros(1))[0]
    pos = np.array(cam.position)
    # The camera stays on the requested surface normal at the pre-cinematic
    # opening distance after the 42° -> 63° pull-in.
    assert np.linalg.norm(pos) == pytest.approx(1.62 * RADIUS, rel=1e-3)
    unit_surface = surface / np.linalg.norm(surface)
    assert np.allclose(pos / np.linalg.norm(pos), unit_surface, atol=1e-6)
    assert cam.fov is None


def test_globe_camera_targets_the_centre_of_the_earth() -> None:
    """The opening target is the origin, so the globe starts centred.

    The scene is a sphere centred on the origin. Targeting a point just under
    the near surface (``normal * RADIUS * 0.9``) framed it off-centre and put
    the orbit pivot on the near face, so the first drag swung the planet about
    a surface point instead of its axis.
    """
    for lon, lat in ((-84.0, 25.0), (0.0, 0.0), (140.0, -60.0)):
        assert globe_camera(lon, lat).target == (0.0, 0.0, 0.0)


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


def test_sample_equirect_float_texture_is_not_rescaled() -> None:
    """A float texture is already in [0, 1]; dividing by 255 would render ~black."""
    tex8 = np.random.default_rng(1).integers(0, 256, (8, 16, 3), dtype=np.uint8)
    texf = tex8.astype(np.float32) / 255.0
    lon = np.array([-120.0, 0.0, 45.5])
    lat = np.array([10.0, -33.3, 71.2])
    assert np.allclose(
        sample_equirect(tex8, lon, lat), sample_equirect(texf, lon, lat), atol=1e-6
    )


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


def test_streamlines_do_not_hop_over_a_narrow_dry_band() -> None:
    """A one-cell dry band must stop a ribbon, not be jumped over.

    With STEP_KM (~14 km / ~0.126 deg) larger than the 0.08 deg cell, an
    endpoint-only wetness check lands wet-to-wet across the band and draws a
    current over land. Checking wetness along the segment must freeze the ribbon
    on the south side instead.
    """
    from luxar.demos.demo_ocean_currents_earth import STEP_KM

    field, band_lat = _dry_band_field()
    # seed exactly one cell south of the band, in wet water
    seed_lat = np.array([band_lat - 0.08])
    seed_lon = np.array([100.0])
    assert field.is_wet(seed_lon, seed_lat).all(), "seed must start wet"
    lon, lat, _ = advect_streamlines(field, seed_lon, seed_lat, 52, STEP_KM)
    # every vertex stays wet AND on the south side of the band (never bridged)
    assert field.is_wet(lon.ravel(), lat.ravel()).all()
    assert float(np.max(lat)) < band_lat


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


def test_tile_budgets_stay_under_the_element_texture_bounds() -> None:
    """A tile's FINEST level must fit one node, and tiling must actually engage.

    The viewer packs per-segment line data at 6 texels and per-point data at 3
    into an element texture whose width is capped at 4096 texels, so one node
    holds at most 682 x maxTextureSize segments (2,793,472 on the conservative
    4096-class GPU) or 1365 x maxTextureSize points (5,591,040). Exceeding it
    clamps the tail SILENTLY, and because geometry is stored in spatial order
    the loss is one contiguous lobe — a clean-edged wedge, which is how #1957
    erased the North Atlantic. Both tile sizes must sit under their floor, and
    both totals must exceed one tile or the partition would never engage.
    """
    assert MAX_SEGMENTS_PER_LINES_NODE == 682 * 4096  # the derivation above
    assert MAX_POINTS_PER_POINTS_NODE == 1365 * 4096
    assert CURRENT_TILE_RIBBONS * N_STEPS <= MAX_SEGMENTS_PER_LINES_NODE
    assert GLOBE_TILE_POINTS <= MAX_POINTS_PER_POINTS_NODE
    assert N_SEEDS > CURRENT_TILE_RIBBONS  # tiling engages
    assert N_GLOBE > GLOBE_TILE_POINTS


def test_check_tile_budget_rejects_an_oversized_tile() -> None:
    """The guard must fail the BUILD, not leave the clamp to the viewer."""
    check_tile_budget("points", 100, 100)  # at the cap is fine
    with pytest.raises(ValueError, match="silently clamped"):
        check_tile_budget("points", 101, 100)


def test_lod_counts_are_a_geometric_ladder_ending_at_n() -> None:
    """Coarsest first, finest exactly n, each step a factor of the compression."""
    counts = lod_counts(160_000, levels=3, compression=4)
    assert counts == [10_000, 40_000, 160_000]
    assert lod_counts(160_000, levels=3, compression=2) == [40_000, 80_000, 160_000]
    assert counts[-1] == 160_000  # the finest level is the whole tile
    assert counts == sorted(counts)


def test_lod_counts_collapse_instead_of_repeating_on_a_tiny_tile() -> None:
    """A tile too small for the full depth gets a SHORTER ladder, not dupes.

    Repeated counts would make the viewer cross-fade between two identical
    levels — both resident, no visual difference, pure cost.
    """
    counts = lod_counts(2, levels=4, compression=4)
    assert counts == sorted(set(counts))
    assert counts[-1] == 2
    assert all(c >= 1 for c in counts)


def test_lod_counts_reject_degenerate_parameters() -> None:
    for kwargs in (dict(n=0), dict(n=10, levels=0), dict(n=10, compression=1)):
        with pytest.raises(ValueError):
            lod_counts(**{"n": 10, **kwargs})


def test_level_subset_is_deterministic_sorted_and_a_real_subset() -> None:
    a = level_subset(2000, 200, seed=7)
    b = level_subset(2000, 200, seed=7)
    assert np.array_equal(a, b)  # deterministic: rebuilds are reproducible
    assert a.size == 200
    assert np.array_equal(a, np.sort(a))  # keeps chunk-coherent ordering
    assert len(set(a.tolist())) == 200  # no repeats
    assert level_subset(2000, 5000, seed=7).size == 2000  # count >= n keeps all


def test_level_subset_beats_the_bounding_box_stratified_sampler() -> None:
    """Records WHY the shared `spatial-uniform` sampler is not used here.

    `stratified_grid_order` walks a doubling grid over the BOUNDING BOX. That
    is right for a volumetric cloud and wrong for a hollow shell: a cubic grid
    cuts a sphere into cells of very unequal shell area, so its prefix is
    uniform per CELL and lumpy per unit surface. Measured on this exact
    geometry it is WORSE than plain random by the metric that matters (the
    worst nearest-neighbour gap = the widest hole in the shell), which is the
    non-obvious fact this test exists to keep true — a future "cleanup" that
    routes this through the shared sampler makes the globe porous.
    """
    from luxar.core.group.lod.spatial_uniform import stratified_grid_order

    lon, lat = fibonacci_sphere(4000)
    pos = lonlat_to_xyz(lon, lat, np.zeros(4000)).astype(np.float64)
    keep = 250

    def worst_gap(idx: np.ndarray) -> float:
        p = pos[idx]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).max())

    perm, _ = stratified_grid_order(pos, 5)
    gridded = worst_gap(np.sort(perm[:keep].astype(np.int64)))
    random_gap = float(
        np.median([worst_gap(level_subset(4000, keep, seed=s)) for s in range(5)])
    )
    assert random_gap < gridded


def test_seal_margin_leaves_the_finest_level_alone_and_inflates_coarse_ones() -> None:
    """The finest level is an even lattice already — inflating it only blurs.

    Coarse levels get ``sqrt(thinning)`` (mean spacing on a sphere goes as
    1/sqrt(N)) times a margin for the Poisson gaps a random subset leaves past
    the mean. Dropping either factor fails here.
    """
    assert seal_margin(1000, 1000) == 1.0  # finest: exactly untouched
    assert seal_margin(2000, 1000) == 1.0  # and never shrinks
    assert seal_margin(250, 1000) == pytest.approx(SHELL_SEAL_MARGIN * 2.0)
    assert seal_margin(1000, 16000) == pytest.approx(SHELL_SEAL_MARGIN * 4.0)
    assert SHELL_SEAL_MARGIN > 1.0  # a pure sqrt law leaves holes


def test_level_seeds_never_collide_across_layers_tiles_and_levels() -> None:
    """Correlated subsets would make the two layers thin in the SAME places."""
    seeds = [
        level_seed(layer, tile, level)
        for layer in range(2)
        for tile in range(64)
        for level in range(LOD_LEVELS)
    ]
    assert len(set(seeds)) == len(seeds)


def test_ribbons_clear_the_coarsest_globe_level_not_the_finest() -> None:
    """A lift tuned to the FINEST radius is buried when a tile switches down.

    The coarsest globe level thins by ``LOD_COMPRESSION ** (LOD_LEVELS - 1)``
    and inflates its point radius by the square root of that, so it reaches
    much further off the sphere than the finest level does. Clearing only the
    finest radius means the ribbons over a tile vanish into the shell the
    moment that tile drops a level — the exact "vanishing layer" class this
    topology is supposed to remove, reintroduced by a constant.
    """
    thinning = LOD_COMPRESSION ** (LOD_LEVELS - 1)
    assert GLOBE_COARSEST_RADII == pytest.approx(
        GLOBE_RADII * SHELL_SEAL_MARGIN * thinning**0.5
    )
    assert GLOBE_COARSEST_RADII > GLOBE_RADII  # the whole point
    lift_units = FLOW_LIFT * RADIUS
    assert lift_units > GLOBE_COARSEST_RADII  # clears the COARSEST shell
    # ...and still reads as draped on the surface, not orbiting it.
    assert lift_units < 0.02 * RADIUS


def test_one_part_partition_uses_the_whole_object_anchor() -> None:
    """A single "tile" IS the whole object — the fills-screen anchor is wrong.

    ``to_spatial_partition`` wraps even one BSP leaf in a ``kind=partition``, so
    the shape alone does not prove a tiling. Anchoring a one-part ladder at
    fills-screen holds its finest level back until the object OVERFILLS the
    viewport — the #1361 blur — and the compiler warns about exactly this.
    """
    counts = [100, 400, 1600]
    assert tile_coverage(counts, 1)[-1] == pytest.approx(0.5)  # whole-object
    assert tile_coverage(counts, 2)[-1] == pytest.approx(1.0)  # fills-screen
    assert tile_coverage(counts, 16)[-1] == pytest.approx(1.0)


def _tiny_globe(n: int = 4000) -> tuple:
    lon, lat = fibonacci_sphere(n)
    pos = lonlat_to_xyz(lon, lat, np.zeros(n))
    colors = np.tile(np.float32([0.2, 0.4, 0.8]), (n, 1))
    return pos, colors


def _tiny_ribbons(n_paths: int = 400, n_vertices: int = 6) -> tuple:
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, n_paths)[:, None] + np.arange(n_vertices)[None, :]
    lat = rng.uniform(-60, 60, n_paths)[:, None] + np.zeros(n_vertices)[None, :]
    verts = lonlat_to_xyz(lon.ravel(), lat.ravel(), np.full(lon.size, FLOW_LIFT))
    colors = np.tile(np.float32([0.5, 0.7, 1.0, 1.0]), (lon.size, 1))
    return verts, colors, n_paths, n_vertices


def _write(tmp_path: Path, fn) -> "object":
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    dims = Dimensions([Dimension(a, unit="", display=True) for a in "xyz"])
    out = tmp_path / "mini.luxar.zarr"
    with LuxarZarrCompiler(out) as c:
        scene = c.create_scene(dimensions=dims)
        fn(scene)
    return out


def _payload_groups(level) -> list:
    additive = sorted(k for k in level.keys() if k.startswith("additive_"))
    return [level[name] for name in additive] if additive else [level]


def _payload_count(level, array_name: str) -> int:
    return sum(int(group[array_name].shape[0]) for group in _payload_groups(level))


def _payload_scalar(level, array_name: str) -> float:
    values = {
        float(np.asarray(group[array_name]).ravel()[0])
        for group in _payload_groups(level)
    }
    assert len(values) == 1
    return values.pop()


def test_globe_writer_builds_partition_of_lod(tmp_path: Path) -> None:
    """The shape is a kind=partition of per-tile kind=lod ladders.

    Not one or the other: `partition=` alone keeps every part resident (the
    viewer only frustum-culls) and a lod ladder alone still overflows one
    node's element texture. Only the nesting bounds BOTH (#2155).
    """
    pos, colors = _tiny_globe()
    holder: dict = {}

    def build(scene):
        holder["result"] = write_globe_parts(scene, pos, colors, tile_size=500)

    out = _write(tmp_path, build)
    n_tiles, coarsest = holder["result"]
    assert n_tiles >= 2  # a real tiling, or the fills-screen anchor is wrong
    assert coarsest < 0.3 * len(pos)

    root = open_group(str(out), mode="r")
    wrapper = root["earth"]
    assert wrapper.attrs["kind"] == "partition"
    assert wrapper.attrs["layer"] is True
    assert wrapper.attrs["blending_mode"] == "opaque"
    parts = [k for k in wrapper.keys() if k.startswith("part_")]
    assert len(parts) == n_tiles
    for name in parts:
        part = wrapper[name]
        assert part.attrs["kind"] == "lod"
        assert part.attrs["selector"] == "screen-area"
        children = sorted(k for k in part.keys() if k.startswith("child_"))
        assert len(children) >= 2  # a ladder, not a lone leaf
        covers = [part[c].attrs["coverage_fraction"] for c in children]
        assert covers == sorted(covers)  # coarsest first
        assert covers[-1] == pytest.approx(1.0)  # fills-screen anchor for a tile


def test_globe_coarse_levels_seal_the_shell_with_a_sqrt_radius(
    tmp_path: Path,
) -> None:
    """Thinning by K needs radius x sqrt(K), or the globe goes see-through.

    Spacing on a fixed sphere goes as 1/sqrt(N), so a coarse level that keeps
    the finest radius leaves gaps the far side of the planet shows through.
    Mutating the exponent to 1.0 or dropping it entirely fails here.
    """
    pos, colors = _tiny_globe()
    holder: dict = {}
    out = _write(
        tmp_path,
        lambda s: holder.__setitem__(
            "r", write_globe_parts(s, pos, colors, tile_size=500)
        ),
    )
    root = open_group(str(out), mode="r")
    part = root["earth"]["part_0"]
    children = sorted(k for k in part.keys() if k.startswith("child_"))
    finest = part[children[-1]]
    n_finest = _payload_count(finest, "positions")
    r_finest = _payload_scalar(finest, "radii")
    for name in children[:-1]:
        level = part[name]
        n = _payload_count(level, "positions")
        r = _payload_scalar(level, "radii")
        assert r == pytest.approx(r_finest * seal_margin(n, n_finest), rel=1e-5)
        assert r > r_finest  # coarser really is fatter


def test_currents_writer_keeps_ribbons_atomic_and_widens_linearly(
    tmp_path: Path,
) -> None:
    """Ribbons stay whole across tiles and levels; width holds the ink constant.

    Two things are pinned. Every level's vertex count is a whole multiple of
    the per-ribbon vertex count — a ribbon split across a tile boundary or
    truncated by a level would be a half-curve, not a coarser one. And width
    scales LINEARLY with the thinning (not sqrt): the line shader floors
    sub-pixel lines at 1.5px and dims by pixelWidth/1.5, so ink goes as
    count x width and only the linear exponent keeps a switch from popping.
    """
    verts, colors, n_paths, n_vertices = _tiny_ribbons()
    holder: dict = {}
    out = _write(
        tmp_path,
        lambda s: holder.__setitem__(
            "r",
            write_current_parts(s, verts, colors, n_paths, n_vertices, tile_size=50),
        ),
    )
    n_tiles, coarsest = holder["r"]
    assert n_tiles >= 2
    assert coarsest < 0.3 * n_paths * (n_vertices - 1)

    root = open_group(str(out), mode="r")
    wrapper = root["currents"]
    assert wrapper.attrs["kind"] == "partition"
    assert wrapper.attrs["layer"] is True
    assert wrapper.attrs["blending_mode"] == "normal"
    assert wrapper.attrs["opacity"] == pytest.approx(LINE_OPACITY)
    total_finest = 0
    for name in (k for k in wrapper.keys() if k.startswith("part_")):
        part = wrapper[name]
        assert part.attrs["kind"] == "lod"
        children = sorted(k for k in part.keys() if k.startswith("child_"))
        finest = part[children[-1]]
        n_v_finest = _payload_count(finest, "vertices")
        assert n_v_finest % n_vertices == 0  # whole ribbons only
        w_finest = _payload_scalar(finest, "widths")
        total_finest += n_v_finest // n_vertices
        for child in children[:-1]:
            level = part[child]
            n_v = _payload_count(level, "vertices")
            assert n_v % n_vertices == 0  # whole ribbons at every level too
            w = _payload_scalar(level, "widths")
            ribbons_finest = n_v_finest // n_vertices
            ribbons = n_v // n_vertices
            assert w == pytest.approx(w_finest * ribbons_finest / ribbons, rel=1e-5)
    assert total_finest == n_paths  # the partition conserves every ribbon


def test_writers_specialize_streaming_for_coarser_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored ladders start near their coarser sibling, not one shared base."""
    monkeypatch.setattr(demo_ocean_currents_earth, "LOD_STREAM_CHUNK", 100)
    pos, colors = _tiny_globe()
    vertices, line_colors, n_paths, n_vertices = _tiny_ribbons()

    def build(scene) -> None:
        write_globe_parts(scene, pos, colors, tile_size=len(pos))
        write_current_parts(
            scene,
            vertices,
            line_colors,
            n_paths,
            n_vertices,
            tile_size=n_paths,
        )

    out = _write(tmp_path, build)
    root = open_group(str(out), mode="r")

    part = root["earth"]["part_0"]
    children = sorted(k for k in part.keys() if k.startswith("child_"))
    first_chunks = []
    for child in children:
        level = part[child]
        assert "positions" not in level
        first_chunks.append(int(level["additive_0"].attrs["n_points"]))

    assert first_chunks == [100, 500, 1000]

    part = root["currents"]["part_0"]
    children = sorted(k for k in part.keys() if k.startswith("child_"))
    first_chunks = [
        int(part[child]["additive_0"].attrs["n_vertices"]) for child in children
    ]
    assert first_chunks == [102, 300, 600]


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
