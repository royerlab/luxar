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

# `fibonacci_sphere` / `sample_equirect` are imported from their real home now
# that the globe is a mesh and this demo no longer re-exports them. They are still
# shared helpers used by the other Earth demos, so the tests below keep their
# coverage — just against the module that owns them.
from luxar.demos._globe_common import (
    fibonacci_sphere,
    sample_equirect,
    surface_point_radius,
)
from luxar.demos.demo_ocean_currents_earth import (
    CURRENT_TILE_RIBBONS,
    FLOW_LIFT,
    LINE_OPACITY,
    LOD_LEVELS,
    N_SEEDS,
    N_STEPS,
    RADIUS,
    LonLatField,
    advect_streamlines,
    build_lut,
    check_tile_budget,
    globe_camera,
    level_seed,
    level_subset,
    lod_counts,
    lonlat_to_xyz,
    polyline_segment_indices,
    seed_ocean_points,
    tile_coverage,
    write_current_parts,
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


def test_current_tile_budget_stays_under_the_segment_texture_bound() -> None:
    """Each finest current tile stays below the conservative Lines cap."""
    assert CURRENT_TILE_RIBBONS * N_STEPS <= MAX_SEGMENTS_PER_LINES_NODE
    assert MAX_SEGMENTS_PER_LINES_NODE == 682 * 4096  # the derivation above
    assert N_SEEDS > CURRENT_TILE_RIBBONS


def test_textured_globe_no_longer_needs_the_point_texture_budget() -> None:
    assert MAX_POINTS_PER_POINTS_NODE == 1365 * 4096  # the derivation above


def test_the_globe_is_a_textured_mesh_not_a_point_cloud() -> None:
    """The globe must be a Mesh with UVs and a texture, and NOT partitioned.

    Pinned as source text for the same reason the partition call was: the
    difference is invisible in a rendered frame that "looks like an Earth". A
    regression to `add_points` would still render a globe — just a stippled one
    needing 240x the elements, which is the whole problem this replaced.

    The two ABSENCES matter as much as the presences. `partition=` and
    `additive_lod=` both existed to work around the point cloud's scale, and a
    mesh refuses the latter outright for a non-radial method — a prefix of an
    arbitrarily ordered index buffer is a holed surface, not a coarser one — so
    carrying them over would have failed at write time rather than silently.
    """
    source = Path(demo_ocean_currents_earth.__file__).read_text()
    assert "scene.add_points(" not in source, "the globe should no longer be points"
    # The globe goes through the SHARED tiled helper rather than a direct
    # `add_mesh`, because a 16384-wide basemap exceeds the per-axis texture limit
    # and has to be split across nodes. The helper owns geometry, slicing,
    # transcoding and the `uvs`/`normal_dims`/`double_sided` wiring, so what this
    # test pins is that the demo REACHES it with the arguments that matter.
    globe_call = source.split("build_earth(")[1].split("write_current_parts(")[0]
    for token in ("basemap=basemap", "tiles=GLOBE_TILES", "radius=RADIUS"):
        assert token in globe_call, f"missing {token}"
    assert "partition=" not in globe_call
    assert "additive_lod=" not in globe_call


def test_the_globe_is_unlit_so_its_colours_stay_comparable() -> None:
    """`shading="none"` on the basemap, paired with the pinned tone mapping.

    Not a style preference. This basemap is a REFERENCE for the current speeds
    drawn over it, so a view-anchored diffuse key would darken the limb as the
    camera moved and the same ocean would read as a different colour depending
    on where you looked from. It is the same argument
    `test_demos_tone_mapping_policy.py` makes for pinning `tone_mapping="None"`
    here, and the two only work as a pair — an unlit surface run through ACES is
    still reshaded, and a lit surface with an exact passthrough is still lit.
    """
    source = Path(demo_ocean_currents_earth.__file__).read_text()
    globe_call = source.split("build_earth(")[1].split("write_current_parts(")[0]
    assert 'shading="none"' in globe_call


def test_layer_appearance_matches_the_authored_intent() -> None:
    """The globe is an OPAQUE backdrop; the ribbons are translucent over it.

    `opaque` is the only mode that leaves the viewer's sorted transparent set
    and the only one that unconditionally depth-writes, so it is the only one
    that reliably composites *under* the ribbons in front of it. The ribbons
    stay `normal` (additive ignores depth, so far-side currents would bleed
    across the continents) at the tuned opacity.
    """
    source = Path(demo_ocean_currents_earth.__file__).read_text()
    globe_call = source.split("build_earth(")[1].split("write_current_parts(")[0]
    lines_call = source.split("def write_current_parts(")[1].split("def build_scene(")[
        0
    ]
    assert 'blending_mode="opaque"' in globe_call
    # `luminous`, not `normal`. The distinction that matters is between
    # `luminous` and plain `additive`, not between additive and `normal` — which
    # is how this was originally reasoned. `luminous` is additive AND depth-tested,
    # so it keeps what `normal` was protecting (the far-side network stays hidden
    # behind the opaque globe) and gains two things: overlapping ribbons ACCUMULATE
    # (informative — a boundary current concentrates flow, so it brightens), and
    # the composition is commutative, so nothing depends on getting depth order
    # right across 11M segments and a cloud shell.
    assert 'blending_mode="luminous"' in lines_call
    assert LINE_OPACITY == pytest.approx(0.77)


def test_current_lod_helpers_preserve_the_ladder_contract() -> None:
    assert lod_counts(1600) == [400, 800, 1600]
    assert lod_counts(2, levels=4) == [1, 2]
    assert tile_coverage([100, 400, 1600], 1)[-1] == pytest.approx(0.5)
    assert tile_coverage([100, 400, 1600], 2)[-1] == pytest.approx(1.0)
    subset = level_subset(2000, 200, seed=7)
    assert np.array_equal(subset, level_subset(2000, 200, seed=7))
    assert np.array_equal(subset, np.sort(subset))
    seeds = {
        level_seed(1, tile, level) for tile in range(32) for level in range(LOD_LEVELS)
    }
    assert len(seeds) == 32 * LOD_LEVELS
    with pytest.raises(ValueError):
        check_tile_budget("lines", 11, 10)


def _tiny_ribbons(n_paths: int = 400, n_vertices: int = 6) -> tuple:
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, n_paths)[:, None] + np.arange(n_vertices)[None, :]
    lat = rng.uniform(-60, 60, n_paths)[:, None] + np.zeros(n_vertices)[None, :]
    vertices = lonlat_to_xyz(lon.ravel(), lat.ravel(), np.full(lon.size, FLOW_LIFT))
    colors = np.tile(np.float32([0.5, 0.7, 1.0, 1.0]), (lon.size, 1))
    return vertices, colors, n_paths, n_vertices


def _write_currents(tmp_path: Path, tile_size: int = 50) -> Path:
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    vertices, colors, n_paths, n_vertices = _tiny_ribbons()
    output = tmp_path / "currents.luxar.zarr"
    dims = Dimensions([Dimension(axis, unit="", display=True) for axis in "xyz"])
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        write_current_parts(
            scene, vertices, colors, n_paths, n_vertices, tile_size=tile_size
        )
    return output


def _payload_groups(level) -> list:
    additive = sorted(name for name in level.keys() if name.startswith("additive_"))
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


def test_current_writer_builds_partition_of_lod_with_atomic_ribbons(
    tmp_path: Path,
) -> None:
    output = _write_currents(tmp_path)
    wrapper = open_group(output, mode="r")["currents"]
    assert wrapper.attrs["kind"] == "partition"
    assert wrapper.attrs["blending_mode"] == "luminous"
    total_finest = 0
    n_vertices = 6
    for part_name in (name for name in wrapper.keys() if name.startswith("part_")):
        part = wrapper[part_name]
        assert part.attrs["kind"] == "lod"
        children = sorted(name for name in part.keys() if name.startswith("child_"))
        assert len(children) == LOD_LEVELS
        finest = part[children[-1]]
        finest_count = _payload_count(finest, "vertices")
        assert finest_count % n_vertices == 0
        total_finest += finest_count // n_vertices
        finest_width = _payload_scalar(finest, "widths")
        for child_name in children[:-1]:
            child = part[child_name]
            child_count = _payload_count(child, "vertices")
            assert child_count % n_vertices == 0
            child_width = _payload_scalar(child, "widths")
            assert child_width == pytest.approx(
                finest_width * finest_count / child_count, rel=1e-5
            )
    assert total_finest == 400


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
