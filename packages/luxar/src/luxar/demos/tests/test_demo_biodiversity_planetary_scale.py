"""Tests for the pure helpers of ``demo_biodiversity_planetary_scale``.

Everything exercised here is network-free and IO-free: coordinate mapping,
taxonomy resolution, jitter scaling, great-circle densification, index builders
and the sampling/tiling arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from luxar.demos.demo_biodiversity_planetary_scale import (
    ALL_LIFE_SLOT,
    DEFAULT_N_POINTS,
    GBIF_USABLE_ROWS_PER_PART,
    JITTER_CEIL_M,
    JITTER_FLOOR_M,
    N_PERIODS,
    OVERSAMPLE_FACTOR,
    PERIOD_ALL_SLOT,
    PERIOD_CATEGORIES,
    R_EARTH_KM,
    RADIUS,
    TAXON_CATEGORIES,
    TAXON_GROUP_COLORS,
    TAXON_GROUP_NAMES,
    BottomKSampler,
    chain_segment_indices,
    globe_camera,
    great_circle_resample,
    jitter_sigma_deg,
    lonlat_to_xyz,
    parts_needed_for,
    period_slot,
    polyline_segment_indices,
    stratified_cap,
    taxon_slot,
    tile_count_for,
)

# ---------------------------------------------------------------------------
# lonlat_to_xyz
# ---------------------------------------------------------------------------


def test_lonlat_to_xyz_poles_and_origin():
    xyz = lonlat_to_xyz(
        np.array([0.0, 0.0, 0.0]),
        np.array([90.0, -90.0, 0.0]),
        np.zeros(3),
    )
    assert xyz.dtype == np.float32
    np.testing.assert_allclose(xyz[0], [0.0, RADIUS, 0.0], atol=1e-3)
    np.testing.assert_allclose(xyz[1], [0.0, -RADIUS, 0.0], atol=1e-3)
    np.testing.assert_allclose(xyz[2], [RADIUS, 0.0, 0.0], atol=1e-3)


def test_lonlat_to_xyz_is_right_handed_not_mirrored():
    """East x North must point OUTWARD, or the globe renders mirrored.

    This is the invariant that separates this mapping from the older
    ``demo_earthquakes_3d`` one (which omits the ``-z`` and compensates by
    negating longitude at the call site).
    """
    lon, lat = 30.0, 20.0
    p = lonlat_to_xyz(np.array([lon]), np.array([lat]), np.zeros(1))[0].astype(
        np.float64
    )
    d = 1e-3
    east = (
        lonlat_to_xyz(np.array([lon + d]), np.array([lat]), np.zeros(1))[0].astype(
            np.float64
        )
        - p
    )
    north = (
        lonlat_to_xyz(np.array([lon]), np.array([lat + d]), np.zeros(1))[0].astype(
            np.float64
        )
        - p
    )
    outward = np.cross(east, north)
    # Positive projection onto the radial direction == outward == not mirrored.
    assert float(np.dot(outward, p)) > 0.0


def test_lonlat_to_xyz_relief_is_fractional():
    r0 = np.linalg.norm(lonlat_to_xyz(np.array([12.0]), np.array([5.0]), np.zeros(1)))
    r1 = np.linalg.norm(
        lonlat_to_xyz(np.array([12.0]), np.array([5.0]), np.full(1, 0.01))
    )
    assert r0 == pytest.approx(RADIUS, abs=1e-3)
    assert r1 == pytest.approx(RADIUS * 1.01, abs=1e-3)


def test_lonlat_to_xyz_longitude_is_periodic():
    a = lonlat_to_xyz(np.array([-180.0]), np.array([0.0]), np.zeros(1))
    b = lonlat_to_xyz(np.array([180.0]), np.array([0.0]), np.zeros(1))
    np.testing.assert_allclose(a, b, atol=1e-3)


def test_globe_camera_looks_at_origin_from_outside():
    cam = globe_camera(-40.0, 25.0, distance=2.6)
    assert cam.target == (0.0, 0.0, 0.0)
    assert np.linalg.norm(cam.position) == pytest.approx(RADIUS * 2.6, rel=1e-6)
    # The camera must sit over the requested surface point.
    surface = lonlat_to_xyz(np.array([-40.0]), np.array([25.0]), np.zeros(1))[0]
    cos = float(np.dot(cam.position / np.linalg.norm(cam.position), surface / RADIUS))
    assert cos == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# taxon_group_of
# ---------------------------------------------------------------------------


def test_nine_groups_each_with_a_colour():
    assert len(TAXON_GROUP_NAMES) == 9
    assert len(set(TAXON_GROUP_NAMES)) == 9
    assert TAXON_GROUP_COLORS.shape == (9, 3)
    assert TAXON_GROUP_COLORS.dtype == np.float32
    assert np.all((TAXON_GROUP_COLORS >= 0.0) & (TAXON_GROUP_COLORS <= 1.0))


def test_categories_prepend_the_all_life_summary_slot():
    """The viewer dimension offers "All life" first, then the nine groups."""
    assert TAXON_CATEGORIES[ALL_LIFE_SLOT] == "All life"
    assert TAXON_CATEGORIES[1:] == TAXON_GROUP_NAMES
    assert len(TAXON_CATEGORIES) == 10
    assert len(set(TAXON_CATEGORIES)) == 10


def test_taxon_slot_shifts_past_the_summary_slot():
    """A record's stored coordinate must never collide with ALL_LIFE_SLOT."""
    for gid, name in enumerate(TAXON_GROUP_NAMES):
        slot = taxon_slot(gid)
        assert slot != ALL_LIFE_SLOT
        # The shifted coordinate must name the same group in the category list.
        assert TAXON_CATEGORIES[slot] == name
    assert taxon_slot(0) == 1
    assert taxon_slot(len(TAXON_GROUP_NAMES) - 1) == len(TAXON_GROUP_NAMES)


def test_taxon_slot_rejects_out_of_range_group_ids():
    for bad in (-1, len(TAXON_GROUP_NAMES), 99):
        with pytest.raises(ValueError):
            taxon_slot(bad)


def test_period_categories_prepend_an_all_years_slot():
    assert PERIOD_CATEGORIES[PERIOD_ALL_SLOT] == "All years"
    assert len(PERIOD_CATEGORIES) == N_PERIODS + 1
    assert PERIOD_CATEGORIES[1] == "1900s"
    assert PERIOD_CATEGORIES[-1] == "2020s"
    assert len(set(PERIOD_CATEGORIES)) == len(PERIOD_CATEGORIES)


@pytest.mark.parametrize(
    "year,expected_label",
    [
        (1900, "1900s"),
        (1909, "1900s"),
        (1910, "1910s"),
        (1961, "1960s"),
        (1999, "1990s"),
        (2020, "2020s"),
        (2026, "2020s"),
    ],
)
def test_period_slot_maps_years_to_the_right_decade(year, expected_label):
    slot = int(period_slot(np.array([year]))[0])
    assert PERIOD_CATEGORIES[slot] == expected_label


def test_period_slot_never_collides_with_the_all_slot():
    slots = period_slot(np.arange(1900, 2027))
    assert slots.min() > PERIOD_ALL_SLOT
    assert slots.max() == N_PERIODS


def test_period_slot_clamps_out_of_range_years():
    assert int(period_slot(np.array([1850]))[0]) == 1
    assert int(period_slot(np.array([2400]))[0]) == N_PERIODS


# ---------------------------------------------------------------------------
# jitter_sigma_deg
# ---------------------------------------------------------------------------


def test_jitter_sigma_clamps_to_floor_and_ceiling():
    unc = np.array([0.0, 1.0, JITTER_FLOOR_M, JITTER_CEIL_M, 1e9])
    lat = np.zeros(5)
    s_lat, s_lon = jitter_sigma_deg(unc, lat)
    floor_deg = math.degrees(JITTER_FLOOR_M / (R_EARTH_KM * 1000.0))
    ceil_deg = math.degrees(JITTER_CEIL_M / (R_EARTH_KM * 1000.0))
    # Sub-floor uncertainties (including 0) take the floor, not zero.
    assert s_lat[0] == pytest.approx(floor_deg, rel=1e-5)
    assert s_lat[1] == pytest.approx(floor_deg, rel=1e-5)
    assert s_lat[2] == pytest.approx(floor_deg, rel=1e-5)
    assert s_lat[3] == pytest.approx(ceil_deg, rel=1e-5)
    assert s_lat[4] == pytest.approx(ceil_deg, rel=1e-5)
    # At the equator a degree of longitude equals a degree of latitude.
    np.testing.assert_allclose(s_lon, s_lat, rtol=1e-5)


def test_jitter_sigma_nan_uncertainty_takes_the_floor():
    s_lat, _ = jitter_sigma_deg(np.array([np.nan]), np.array([0.0]))
    expected = math.degrees(JITTER_FLOOR_M / (R_EARTH_KM * 1000.0))
    assert s_lat[0] == pytest.approx(expected, rel=1e-5)


def test_jitter_sigma_lon_widens_toward_the_poles():
    unc = np.full(3, 5000.0)
    lat = np.array([0.0, 60.0, 89.999])
    s_lat, s_lon = jitter_sigma_deg(unc, lat)
    # Latitude sigma is latitude-independent; longitude sigma grows as 1/cos.
    assert s_lat[0] == pytest.approx(s_lat[1], rel=1e-6)
    assert s_lon[1] == pytest.approx(s_lat[1] / math.cos(math.radians(60.0)), rel=1e-4)
    assert s_lon[2] > s_lon[1] > s_lon[0]
    # The cos floor keeps the pole finite rather than exploding.
    assert np.isfinite(s_lon[2])
    assert s_lon[2] <= s_lat[2] / 0.05 * (1 + 1e-6)


def test_jitter_sigma_is_a_physically_sane_magnitude():
    """A 1 km uncertainty must be ~0.009 deg, not degrees or microdegrees."""
    s_lat, _ = jitter_sigma_deg(np.array([1000.0]), np.array([0.0]))
    assert 0.008 < float(s_lat[0]) < 0.010


# ---------------------------------------------------------------------------
# great_circle_resample
# ---------------------------------------------------------------------------


def _on_sphere(lon, lat):
    return lonlat_to_xyz(lon, lat, np.zeros(len(lon)))


def test_great_circle_resample_preserves_endpoints():
    lon = np.array([0.0, 90.0])
    lat = np.array([0.0, 0.0])
    o_lon, o_lat, src = great_circle_resample(lon, lat, 10.0)
    assert o_lon[0] == pytest.approx(0.0, abs=1e-6)
    assert o_lat[0] == pytest.approx(0.0, abs=1e-6)
    assert o_lon[-1] == pytest.approx(90.0, abs=1e-6)
    assert o_lat[-1] == pytest.approx(0.0, abs=1e-6)
    assert src[0] == 0 and src[-1] == 1


def test_great_circle_resample_respects_max_step():
    lon = np.array([0.0, 170.0])
    lat = np.array([10.0, -40.0])
    o_lon, o_lat, _ = great_circle_resample(lon, lat, 2.0)
    p = _on_sphere(o_lon, o_lat).astype(np.float64) / RADIUS
    dots = np.clip(np.einsum("ij,ij->i", p[:-1], p[1:]), -1.0, 1.0)
    steps_deg = np.degrees(np.arccos(dots))
    assert steps_deg.max() <= 2.0 + 1e-6


def test_great_circle_resample_output_stays_on_the_sphere():
    """The whole point: interpolated vertices must not cut through the planet."""
    lon = np.array([-120.0, 150.0])  # a >150 deg leg — a chord would tunnel
    lat = np.array([35.0, -20.0])
    o_lon, o_lat, _ = great_circle_resample(lon, lat, 1.0)
    radii = np.linalg.norm(_on_sphere(o_lon, o_lat).astype(np.float64), axis=1)
    np.testing.assert_allclose(radii, RADIUS, rtol=1e-5)
    # A straight chord's midpoint would be far inside; the arc's is not.
    chord_mid = 0.5 * (
        _on_sphere(lon[:1], lat[:1])[0].astype(np.float64)
        + _on_sphere(lon[1:], lat[1:])[0].astype(np.float64)
    )
    assert np.linalg.norm(chord_mid) < RADIUS * 0.7


def test_great_circle_resample_no_duplicate_vertices_across_legs():
    lon = np.array([0.0, 10.0, 20.0])
    lat = np.array([0.0, 0.0, 0.0])
    o_lon, o_lat, _ = great_circle_resample(lon, lat, 5.0)
    d = np.hypot(np.diff(o_lon), np.diff(o_lat))
    assert d.min() > 1e-9


def test_great_circle_resample_source_index_is_monotone_and_in_range():
    lon = np.array([0.0, 40.0, 41.0, 200.0])
    lat = np.array([0.0, 10.0, 10.5, -30.0])
    _, _, src = great_circle_resample(lon, lat, 3.0)
    assert np.all(np.diff(src) >= 0)
    assert src.min() == 0 and src.max() == lon.size - 1


def test_great_circle_resample_coincident_fixes_do_not_divide_by_zero():
    lon = np.array([15.0, 15.0, 15.0])
    lat = np.array([-5.0, -5.0, -5.0])
    o_lon, o_lat, _ = great_circle_resample(lon, lat, 1.0)
    assert np.all(np.isfinite(o_lon)) and np.all(np.isfinite(o_lat))
    np.testing.assert_allclose(o_lon, 15.0, atol=1e-6)


def test_great_circle_resample_single_fix_is_passthrough():
    o_lon, o_lat, src = great_circle_resample(np.array([3.0]), np.array([4.0]), 1.0)
    assert o_lon.tolist() == [3.0] and o_lat.tolist() == [4.0]
    assert src.tolist() == [0]


def test_great_circle_resample_rejects_bad_input():
    with pytest.raises(ValueError):
        great_circle_resample(np.array([1.0, 2.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        great_circle_resample(np.array([]), np.array([]), 1.0)
    with pytest.raises(ValueError):
        great_circle_resample(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0.0)


# ---------------------------------------------------------------------------
# index builders
# ---------------------------------------------------------------------------


def test_polyline_segment_indices_never_joins_two_paths():
    idx = polyline_segment_indices(3, 4)
    pairs = idx.reshape(-1, 2)
    assert pairs.shape == (3 * 3, 2)
    # No segment may cross a path boundary (vertex 3->4, 7->8).
    assert not np.any((pairs[:, 0] // 4) != (pairs[:, 1] // 4))
    assert idx.dtype == np.uint32


def test_polyline_segment_indices_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        polyline_segment_indices(0, 4)
    with pytest.raises(ValueError):
        polyline_segment_indices(2, 1)


def test_chain_segment_indices_handles_ragged_lengths():
    idx = chain_segment_indices([3, 2, 4])
    pairs = idx.reshape(-1, 2)
    assert pairs.shape == (2 + 1 + 3, 2)
    assert pairs[:2].tolist() == [[0, 1], [1, 2]]  # chain 0 (offset 0, len 3)
    assert pairs[2].tolist() == [3, 4]  # chain 1 (offset 3, len 2)
    assert pairs[3:].tolist() == [[5, 6], [6, 7], [7, 8]]  # chain 2 (offset 5)


def test_chain_segment_indices_skips_short_chains_but_keeps_their_offsets():
    idx = chain_segment_indices([1, 3])
    pairs = idx.reshape(-1, 2)
    # The 1-vertex chain emits no segment but still occupies vertex 0.
    assert pairs.tolist() == [[1, 2], [2, 3]]


def test_chain_segment_indices_empty_is_empty():
    assert chain_segment_indices([]).size == 0
    assert chain_segment_indices([1, 1]).size == 0


# ---------------------------------------------------------------------------
# sampling / tiling arithmetic
# ---------------------------------------------------------------------------


def test_stratified_cap_caps_each_label_independently():
    labels = np.array([0] * 100 + [1] * 5 + [2] * 40)
    idx = stratified_cap(labels, 10, np.random.default_rng(0))
    kept = labels[idx]
    assert (kept == 0).sum() == 10
    assert (kept == 1).sum() == 5  # under the cap — kept whole
    assert (kept == 2).sum() == 10


def test_stratified_cap_returns_sorted_unique_indices():
    labels = np.repeat(np.arange(4), 50)
    idx = stratified_cap(labels, 7, np.random.default_rng(1))
    assert np.all(np.diff(idx) > 0)
    assert idx.size == 28


def test_stratified_cap_is_deterministic_for_a_given_seed():
    labels = np.repeat(np.arange(3), 30)
    a = stratified_cap(labels, 5, np.random.default_rng(42))
    b = stratified_cap(labels, 5, np.random.default_rng(42))
    c = stratified_cap(labels, 5, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_stratified_cap_zero_cap_and_empty_input():
    labels = np.array([0, 0, 1])
    assert stratified_cap(labels, 0, np.random.default_rng(0)).size == 0
    assert (
        stratified_cap(np.array([], dtype=np.int8), 5, np.random.default_rng(0)).size
        == 0
    )


def test_stratified_cap_rejects_negative_cap():
    with pytest.raises(ValueError):
        stratified_cap(np.array([0]), -1, np.random.default_rng(0))


@pytest.mark.parametrize(
    "n_points,target,expected",
    [
        (1, 2_000_000, 1),
        (2_000_000, 2_000_000, 1),
        (2_000_001, 2_000_000, 2),
        (15_000_000, 2_000_000, 8),
        (100_000_000, 2_000_000, 64),
    ],
)
def test_tile_count_for_is_a_power_of_two_matching_the_median_bsp(
    n_points, target, expected
):
    n = tile_count_for(n_points, target)
    assert n == expected
    assert n & (n - 1) == 0
    # Every tile must land under the silent element-texture clamp.
    assert math.ceil(n_points / n) <= 5_591_040


def test_tile_count_for_rejects_bad_input():
    with pytest.raises(ValueError):
        tile_count_for(0, 100)
    with pytest.raises(ValueError):
        tile_count_for(100, 0)


def test_parts_needed_for_oversamples_the_point_target():
    n = parts_needed_for(DEFAULT_N_POINTS)
    assert n == math.ceil(
        DEFAULT_N_POINTS * OVERSAMPLE_FACTOR / GBIF_USABLE_ROWS_PER_PART
    )
    # The default must read enough parts that per-part yield variance and
    # single-publisher parts cannot dominate the composition. Measured: 40 parts
    # gave 82.8% Aves, 250 gave 73.0%.
    assert n >= 200
    assert parts_needed_for(2 * DEFAULT_N_POINTS) > n
    assert parts_needed_for(DEFAULT_N_POINTS, 1.0) < n


def test_parts_needed_for_scales_with_the_oversample_factor():
    assert parts_needed_for(1_000_000, 4.0) == 2 * parts_needed_for(1_000_000, 2.0)


def test_parts_needed_for_is_at_least_one():
    assert parts_needed_for(1, 1.0) == 1


def test_parts_needed_for_rejects_bad_input():
    with pytest.raises(ValueError):
        parts_needed_for(1000, 0.5)
    with pytest.raises(ValueError):
        parts_needed_for(0, 3.0)


# ---------------------------------------------------------------------------
# BottomKSampler
# ---------------------------------------------------------------------------


def test_bottomk_keeps_exactly_k_and_tracks_what_it_saw():
    s = BottomKSampler(10, np.random.default_rng(0))
    for start in range(0, 100, 20):
        s.add([np.arange(start, start + 20)])
    assert s.n_seen == 100
    assert s.n_kept == 10
    (vals,) = s.result()
    assert vals.size == 10
    assert np.unique(vals).size == 10  # no duplicates


def test_bottomk_under_k_keeps_everything():
    s = BottomKSampler(50, np.random.default_rng(0))
    s.add([np.arange(7)])
    (vals,) = s.result()
    assert sorted(vals.tolist()) == list(range(7))


def test_bottomk_keeps_parallel_columns_aligned():
    s = BottomKSampler(25, np.random.default_rng(3))
    for start in range(0, 200, 50):
        v = np.arange(start, start + 50)
        s.add([v, (v * 10).astype(np.int64), (-v).astype(np.int64)])
    a, b, c = s.result()
    np.testing.assert_array_equal(b, a * 10)
    np.testing.assert_array_equal(c, -a)


def test_bottomk_is_order_independent_in_distribution():
    """The whole reason for bottom-k: chunking must not bias the sample.

    A greedy "fill from whoever finishes first" sampler would return only the
    first batch's values here. Bottom-k must spread across all batches.
    """
    k, n_batches, batch = 300, 10, 100
    s = BottomKSampler(k, np.random.default_rng(11))
    for b in range(n_batches):
        s.add([np.full(batch, b, dtype=np.int64)])
    (vals,) = s.result()
    counts = np.bincount(vals, minlength=n_batches)
    assert counts.size == n_batches
    assert np.all(counts > 0), f"some batches unrepresented: {counts}"
    # Every batch is equally likely, so ~30 each; allow generous slack.
    assert counts.min() > 10 and counts.max() < 60


def test_bottomk_sample_is_uniform_over_the_stream():
    """A uniform sample must not favour early or late items."""
    n, k = 20_000, 2_000
    s = BottomKSampler(k, np.random.default_rng(5))
    for start in range(0, n, 1_000):
        s.add([np.arange(start, start + 1_000)])
    (vals,) = s.result()
    # Mean of a uniform sample of 0..n-1 should sit near the midpoint.
    assert abs(float(vals.mean()) - (n - 1) / 2.0) < n * 0.03


def test_bottomk_zero_k_and_empty_batches():
    s = BottomKSampler(0, np.random.default_rng(0))
    s.add([np.arange(10)])
    assert s.n_seen == 10 and s.n_kept == 0 and s.result() == []
    e = BottomKSampler(5, np.random.default_rng(0))
    e.add([np.empty(0, dtype=np.int64)])
    assert e.n_seen == 0 and e.n_kept == 0


def test_bottomk_rejects_ragged_batches_and_negative_k():
    with pytest.raises(ValueError):
        BottomKSampler(-1, np.random.default_rng(0))
    s = BottomKSampler(5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        s.add([np.arange(3), np.arange(4)])


def test_bottomk_rejects_changing_column_count():
    s = BottomKSampler(5, np.random.default_rng(0))
    s.add([np.arange(3), np.arange(3)])
    with pytest.raises(ValueError):
        s.add([np.arange(3)])


def test_bottomk_is_deterministic_for_a_given_seed():
    def run(seed):
        s = BottomKSampler(20, np.random.default_rng(seed))
        for start in range(0, 300, 60):
            s.add([np.arange(start, start + 60)])
        return s.result()[0]

    np.testing.assert_array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))
