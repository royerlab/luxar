"""Tests for the pure helpers of ``demo_biodiversity_planetary_scale``.

Everything exercised here is network-free and IO-free: coordinate mapping,
taxonomy resolution, jitter scaling, great-circle densification, index builders
and the sampling/tiling arithmetic.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import consolidate, open_group
from luxar.conftest import read_ts_number_const, viewer_source
from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION
from luxar.core.group.lod.lines import indexed_components_are_chains
from luxar.core.group.partition import serialized_bsp_tree_separates
from luxar.demos import demo_biodiversity_planetary_scale as demo_module
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos.demo_biodiversity_planetary_scale import (
    ALL_LIFE_SLOT,
    ALLOWED_LICENSES,
    DEFAULT_N_POINTS,
    GBIF_USABLE_ROWS_PER_PART,
    JITTER_CEIL_M,
    JITTER_FLOOR_M,
    N_PERIODS,
    OCCURRENCE_COVERAGE,
    OCCURRENCE_LOD_LEVELS,
    OVERSAMPLE_FACTOR,
    PERIOD_ALL_SLOT,
    PERIOD_CATEGORIES,
    R_EARTH_KM,
    RADIUS,
    TAXON_CATEGORIES,
    TAXON_GROUP_COLORS,
    TAXON_GROUP_NAMES,
    BottomKSampler,
    DatasetRegistry,
    _dataset_ids,
    _dictionary_codes,
    _read_part,
    biodiversity_ladder,
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
from luxar.utils.lod_breakpoints import DEFAULT_MAX_ADDITIVE_COMMIT


def test_biodiversity_ladder_opens_at_one_quarter_and_conserves_rows() -> None:
    n_rows = 2_111_885
    stops = 139
    counts = biodiversity_ladder(n_rows, stops)["counts"]
    increments = [
        count - previous
        for previous, count in zip([0, *counts[:-1]], counts, strict=True)
    ]

    assert counts[0] == math.ceil(n_rows / 4)
    assert all(left < right for left, right in zip(counts, counts[1:], strict=False))
    assert counts[-1] == n_rows
    assert max(increments) <= DEFAULT_MAX_ADDITIVE_COMMIT * stops
    assert biodiversity_ladder(1, stops)["counts"] == [1]


def test_keep_stale_reuses_an_existing_scene_with_a_mismatched_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "biodiversity.luxar.zarr"
    group = open_group(output_path, mode="w")
    consolidate(group)
    monkeypatch.setattr(demo_module, "RECOMPUTE", False)
    monkeypatch.setattr(demo_module, "KEEP_STALE", True)
    monkeypatch.setattr(demo_module, "scene_marker_matches", lambda _path: False)

    assert demo_module.load_or_build_scene(output_path) == output_path


def test_keep_stale_rebuilds_an_unfinished_scene(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "biodiversity.luxar.zarr"
    open_group(output_path, mode="w")
    monkeypatch.setattr(demo_module, "RECOMPUTE", False)
    monkeypatch.setattr(demo_module, "KEEP_STALE", True)
    monkeypatch.setattr(demo_module, "scene_marker_matches", lambda _path: False)
    monkeypatch.setattr(
        demo_module,
        "load_gbif",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("rebuild reached")
        ),
    )

    with pytest.raises(RuntimeError, match="rebuild reached"):
        demo_module.load_or_build_scene(output_path)


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
    distance = np.linalg.norm(cam.position)
    old_fill = math.tan(math.asin(1.0 / 2.6)) / math.tan(math.radians(42.0) / 2.0)
    new_fill = math.tan(math.asin(RADIUS / distance)) / math.tan(
        math.radians(CINEMATIC_FOV_DEG) / 2.0
    )
    assert new_fill == pytest.approx(old_fill, rel=1e-6)
    assert cam.fov is None
    # The camera must sit over the requested surface point.
    surface = lonlat_to_xyz(np.array([-40.0]), np.array([25.0]), np.zeros(1))[0]
    cos = float(np.dot(cam.position / distance, surface / RADIUS))
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
# licence allowlist  (review finding: a blocklist retained nulls/unknowns)
# ---------------------------------------------------------------------------


def _license_mask(values):
    """Run the loader's OWN licence gate over a column of raw GBIF values.

    Calls the real ``_dictionary_codes`` against the real ``ALLOWED_LICENSES``,
    exactly as ``_read_part`` does, rather than restating the rule here — a
    test that reimplements the gate would keep passing if the gate itself were
    inverted again.
    """
    pa = pytest.importorskip("pyarrow")
    codes = _dictionary_codes(
        pa.array(values, type=pa.string()), {name: 1 for name in ALLOWED_LICENSES}
    )
    return codes == 1


def test_license_allowlist_keeps_only_cc_by_and_cc0():
    assert set(ALLOWED_LICENSES) == {"CC0_1_0", "CC_BY_4_0"}
    values = ["CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"]
    assert _license_mask(values).tolist() == [True, True, False]


def test_license_allowlist_drops_null_and_unknown_licenses():
    """The bug this replaces: a blocklist rejecting only CC_BY_NC_4_0 retained
    every record GBIF left null or spelled differently, while the demo claimed a
    CC BY / CC0-only sample."""
    values = [None, "", "CC_BY_NC_ND_4_0", "CC_BY_SA_4_0", "unspecified", "CC0"]
    assert not _license_mask(values).any()


def test_license_allowlist_is_exact_not_prefix():
    # 'CC_BY_4_0_DERIV' must not pass by sharing a prefix with an allowed value.
    assert not _license_mask(["CC_BY_4_0_DERIV", "XCC0_1_0"]).any()


def test_read_part_keeps_only_licensed_records_and_labels_their_publisher(tmp_path):
    """The whole reader over a synthetic part: filters, row ids, publisher ids.

    Guards the licensing claim where it is actually enforced -- a unit test of
    the gate alone would not notice `_read_part` forgetting to apply it -- and
    checks that a kept row's dataset id still names its own publisher after the
    dictionary round-trip.
    """
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        # (lat, lon, class, year, license, uncertainty, datasetkey)
        (10.0, 20.0, "Aves", 2001, "CC0_1_0", 100.0, "ds-a"),
        (11.0, 21.0, "Insecta", 1995, "CC_BY_4_0", None, "ds-b"),
        (12.0, 22.0, "Aves", 2010, "CC_BY_NC_4_0", 100.0, "ds-c"),  # NC
        (13.0, 23.0, "Aves", 2010, None, 100.0, "ds-c"),  # null license
        (14.0, 24.0, "Aves", 2010, "UNSPECIFIED", 100.0, "ds-c"),  # unknown
        (0.0, 0.0, "Aves", 2010, "CC0_1_0", 100.0, "ds-a"),  # null island
        (15.0, 25.0, "Aves", 1850, "CC0_1_0", 100.0, "ds-a"),  # pre-1900
        (16.0, 26.0, "Nonesuch", 2010, "CC0_1_0", 100.0, "ds-a"),  # unmappable
        (17.0, 27.0, "Aves", 2010, "CC0_1_0", 500_000.0, "ds-a"),  # centroid
        (18.0, 28.0, "Mammalia", 2020, "CC_BY_4_0", 50.0, "ds-a"),
    ]
    table = pa.table(
        {
            "decimallatitude": pa.array([r[0] for r in rows], pa.float64()),
            "decimallongitude": pa.array([r[1] for r in rows], pa.float64()),
            "kingdom": pa.array([None] * len(rows), pa.string()),
            "phylum": pa.array([None] * len(rows), pa.string()),
            "class": pa.array([r[2] for r in rows], pa.string()),
            "year": pa.array([r[3] for r in rows], pa.int32()),
            "license": pa.array([r[4] for r in rows], pa.string()),
            "coordinateuncertaintyinmeters": pa.array(
                [r[5] for r in rows], pa.float64()
            ),
            "datasetkey": pa.array([r[6] for r in rows], pa.string()),
        }
    )
    path = tmp_path / "000000"
    pq.write_table(table, path)

    registry = DatasetRegistry()
    res = _read_part(None, str(path), 3, registry)

    # Rows 0, 1 and 9 are the only ones that clear every filter.
    assert res.n_rows == len(rows)
    assert res.n_kept == 3
    np.testing.assert_allclose(res.lat, [10.0, 11.0, 18.0], atol=1e-4)
    # The part ordinal rides in the high bits so uids are unique across parts.
    assert res.uid.tolist() == [(3 << 32) | 0, (3 << 32) | 1, (3 << 32) | 9]
    assert [registry.key(int(i)) for i in res.ds_id] == ["ds-a", "ds-b", "ds-a"]
    # The scanned-candidate counts agree with the kept rows.
    assert res.datasets == {"ds-a": 2, "ds-b": 1}


# ---------------------------------------------------------------------------
# DatasetRegistry  (review finding: provenance counted scanned candidates)
# ---------------------------------------------------------------------------


def test_dataset_registry_assigns_stable_ids():
    r = DatasetRegistry()
    a = r.ids_for(["ds-a", "ds-b", "ds-a"])
    assert a.tolist() == [0, 1, 0]
    assert a.dtype == np.int32
    # Ids stay stable across calls, so codes gathered from different parts agree.
    b = r.ids_for(["ds-b", "ds-c"])
    assert b.tolist() == [1, 2]
    assert [r.key(i) for i in range(len(r))] == ["ds-a", "ds-b", "ds-c"]


def test_dataset_registry_maps_none_to_a_sentinel_key():
    r = DatasetRegistry()
    ids = r.ids_for([None, "ds-a", None])
    assert ids[0] == ids[2]
    assert r.key(int(ids[0])) == ""  # filtered out of the sidecar by the caller


def test_dataset_ids_map_kept_rows_through_the_column_dictionary():
    """The ids the samplers carry must name the right publisher.

    ``_dataset_ids`` goes through the column's dictionary rather than looping
    over rows, so this checks the gather (including the null slot) lands where
    a per-row mapping would have.
    """
    pa = pytest.importorskip("pyarrow")
    column = pa.array(["ds-a", "ds-b", None, "ds-a", "ds-c", "ds-b"])
    registry = DatasetRegistry()
    keep = np.array([0, 2, 3, 5])  # ds-a, null, ds-a, ds-b
    ids = _dataset_ids(column, keep, registry)
    assert [registry.key(int(i)) for i in ids] == ["ds-a", "", "ds-a", "ds-b"]
    assert ids.dtype == np.int32
    # A second part reuses the ids already assigned.
    more = _dataset_ids(column, np.array([4, 1]), registry)
    assert [registry.key(int(i)) for i in more] == ["ds-c", "ds-b"]
    assert int(more[1]) == int(ids[3])


def test_dataset_ids_on_an_empty_selection():
    pa = pytest.importorskip("pyarrow")
    ids = _dataset_ids(
        pa.array(["ds-a"]), np.array([], dtype=np.intp), DatasetRegistry()
    )
    assert ids.size == 0


def test_dataset_registry_is_safe_under_concurrent_readers():
    """The registry is shared by 48 reader threads.

    "Look up, else take the next id and append" is a read-modify-write that the
    GIL does not make atomic: two threads arriving with different new keys can
    claim the same id, and one publisher's records end up credited to the other.
    Every key must map to itself no matter how the threads interleave.
    """
    import sys
    from concurrent.futures import ThreadPoolExecutor

    registry = DatasetRegistry()
    names = [f"ds-{i:05d}" for i in range(20_000)]
    # Each worker sees the keys in a different rotation, so they meet unseen
    # keys at different moments rather than queueing behind one another.
    batches = [names[i * 1_000 :] + names[: i * 1_000] for i in range(16)]
    # A short read is over long before the default 5 ms switch interval, so
    # without this the interpreter would never preempt a worker mid-loop and
    # the test would pass whether or not the registry locks.
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(registry.ids_for, batches))
    finally:
        sys.setswitchinterval(previous)
    for batch, ids in zip(batches, results):
        assert [registry.key(int(i)) for i in ids] == batch
    assert len(registry) == len(names)


def test_provenance_counts_dedupe_records_emitted_to_several_slots():
    """A record emitted into a taxon marginal, a period marginal and a joint cell
    must count ONCE, or a dataset's total would depend on how many slices its
    records happen to occupy."""
    registry = DatasetRegistry()
    ds = registry.ids_for(["ds-a", "ds-a", "ds-b"])
    # Simulate the union: record uid=10 emitted 3x, uid=11 once, uid=12 twice.
    all_uid = np.array([10, 10, 10, 11, 12, 12], dtype=np.int64)
    all_ds = np.array([ds[0], ds[0], ds[0], ds[1], ds[2], ds[2]], dtype=np.int32)
    _, first = np.unique(all_uid, return_index=True)
    counts = np.bincount(all_ds[first].astype(np.int64), minlength=len(registry))
    got = {registry.key(i): int(c) for i, c in enumerate(counts) if c}
    assert got == {"ds-a": 2, "ds-b": 1}
    assert sum(got.values()) == np.unique(all_uid).size


# ---------------------------------------------------------------------------
# bottom-k order independence  (review finding: as_completed decided the sample)
# ---------------------------------------------------------------------------


def test_bottomk_with_caller_keys_is_order_independent():
    """The reproducibility fix: with fixed per-row keys, the selected set is the
    same however the batches arrive — which a threaded `as_completed()` read
    cannot otherwise guarantee."""
    rng = np.random.default_rng(0)
    batches = [(np.arange(i * 50, i * 50 + 50), rng.random(50)) for i in range(6)]

    def run(order):
        s = BottomKSampler(37, np.random.default_rng(999))
        for i in order:
            vals, keys = batches[i]
            s.add([vals], keys=keys)
        return np.sort(s.result()[0])

    forward = run(range(6))
    reversed_ = run(list(reversed(range(6))))
    shuffled = run([3, 0, 5, 1, 4, 2])
    np.testing.assert_array_equal(forward, reversed_)
    np.testing.assert_array_equal(forward, shuffled)
    assert forward.size == 37


def test_bottomk_without_caller_keys_is_order_dependent():
    """Guards the reason the fix was needed: self-drawn keys depend on arrival
    order, so this must NOT be relied on in the threaded read."""
    batches = [np.arange(i * 50, i * 50 + 50) for i in range(4)]

    def run(order):
        s = BottomKSampler(20, np.random.default_rng(7))
        for i in order:
            s.add([batches[i]])
        return np.sort(s.result()[0])

    assert not np.array_equal(run(range(4)), run(list(reversed(range(4)))))


def test_bottomk_selection_does_not_depend_on_batch_granularity():
    """The sampler buffers batches and only trims when the buffer is worth it.

    Chunking must stay invisible: one 600-row batch, six 100-row batches and
    600 single-row batches all have to select the same rows, or the buffering
    would have changed the sample rather than just the copying.
    """
    keys = np.random.default_rng(4).random(600)
    values = np.arange(600)

    def run(batch_size):
        s = BottomKSampler(120, np.random.default_rng(0))
        for start in range(0, 600, batch_size):
            stop = start + batch_size
            s.add([values[start:stop]], keys=keys[start:stop])
        return np.sort(s.result()[0]), s.n_kept

    whole, n_whole = run(600)
    chunked, n_chunked = run(100)
    dribbled, n_dribbled = run(1)
    expected = np.sort(np.argsort(keys)[:120])
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(chunked, expected)
    np.testing.assert_array_equal(dribbled, expected)
    assert n_whole == n_chunked == n_dribbled == 120


def test_bottomk_n_kept_is_exact_before_a_trim():
    """`n_kept` is read between parts to log progress, so it must be right
    while batches are still buffered -- and must not force a trim to say so."""
    s = BottomKSampler(50, np.random.default_rng(0))
    s.add([np.arange(10)])
    assert s.n_kept == 10
    s.add([np.arange(10, 100)])
    assert s.n_kept == 50
    assert s.result()[0].size == 50


def test_bottomk_rejects_mismatched_keys_length():
    s = BottomKSampler(5, np.random.default_rng(0))
    with pytest.raises(ValueError):
        s.add([np.arange(4)], keys=np.zeros(3))


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


def test_chain_segment_indices_are_safe_for_the_tracks_ladder():
    lengths = [0, 1, 3, 2, 4]
    indices = chain_segment_indices(lengths).reshape(-1, 2)
    n_vertices = sum(lengths)

    assert indexed_components_are_chains(n_vertices, indices)
    forked = np.vstack([indices, [2, 6]])
    assert not indexed_components_are_chains(n_vertices, forked)


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


def test_add_lod_tiles_records_the_bsp_tree(monkeypatch):
    class RecordingWrapper:
        def __init__(self) -> None:
            self.parts: list[np.ndarray] = []

        def add_points(self, _name, positions, **_attrs):
            self.parts.append(positions)

    class RecordingScene:
        def __init__(self) -> None:
            self.attrs = None
            self.wrapper = RecordingWrapper()

        def add_partition_group(self, _name, **attrs):
            self.attrs = attrs
            return self.wrapper

    monkeypatch.setattr(
        demo_module, "substitutive_lod_or_flat", lambda _config: {"levels": 1}
    )
    positions = np.array(
        [
            [-4.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scene = RecordingScene()

    demo_module.add_lod_tiles(
        scene,
        "occurrences",
        positions,
        np.ones((positions.shape[0], 3), dtype=np.float32),
        radii=1.0,
        opacity=1.0,
        max_elements=2,
        levels=1,
        coverage=[1.0],
    )

    assert scene.attrs is not None
    tree = scene.attrs["bsp_tree"]
    boxes = [(part.min(axis=0), part.max(axis=0)) for part in scene.wrapper.parts]
    assert len(scene.wrapper.parts) == 4
    assert serialized_bsp_tree_separates(tree, boxes)
    np.testing.assert_array_equal(
        np.sort(np.concatenate(scene.wrapper.parts, axis=0)[:, 0]), positions[:, 0]
    )


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


# ────────────────────────────────────────────────────────────────────────
# OCCURRENCE_COVERAGE — the hand-measured per-tile LOD calibration
#
# The only above-1.0 coverage ladder in the repo, and ~40 lines of in-browser
# measurement sit behind it (see the constant's comment block). Nothing else
# tests it, so an anchor move would silently invalidate the calibration rather
# than fail. These assertions are written in the units the comment records.
#
# UNITS (post-#1410). The two MEASURED_* constants below were taken
# in-browser BEFORE #1410, as a fraction of the viewport's DIAGONAL
# (``hypot(viewport.width, viewport.height)``) — a quantity the viewer no
# longer computes at all; it now normalises by the FITTED SCREEN AXIS
# (``min(width, height)``). They are kept here as the historical raw-diagonal
# record (re-measuring in-browser is out of scope for this fix), and
# converted to fitted-axis raw units through one explicit, NAMED,
# computed-not-hardcoded factor anchored at the mainstream 16:9 aspect ratio
# the calibration was measured at: ``hypot(16, 9) / 9``. That anchor is the
# same one documented in the ``FILL_FACTOR`` doc block in
# ``lod-group-registry.ts``; a browser window at a very different aspect
# ratio shifts the true conversion (see that doc's aspect table), which is
# exactly why these assertions read the REAL, live ``FILL_FACTOR`` from the
# TypeScript source (via ``read_ts_number_const``) rather than re-deriving it
# from ``MAX_COVERAGE_FRACTION`` — a future anchor move (either constant)
# fails HERE again, instead of the ~40 lines of in-browser measurement behind
# this ladder silently going stale.
# ────────────────────────────────────────────────────────────────────────

#: The 16:9-anchored diagonal-to-fitted-axis conversion factor documented
#: above and in ``lod-group-registry.ts``'s ``FILL_FACTOR`` doc block.
#: Computed, not hard-coded, so it is visibly tied to the 16:9 reference
#: aspect ratio rather than a magic number: ``hypot(16, 9) / 9 ≈ 2.0397``.
_REFERENCE_ASPECT_DIAGONAL_TO_FITTED_AXIS = math.hypot(16, 9) / 9

#: Lower bound INFERRED (not measured) for the largest/nearest tile's projected
#: bbox diagonal as a fraction of the viewport DIAGONAL (pre-#1410 units — see
#: above) at whole-globe framing. The demo's calibration record is a set of
#: thresholds that FAILED: at 0.78 and 0.82 two of the four globe tiles still
#: crossed into their middle level, from which only "the largest tile exceeds
#: 0.78" strictly follows. 0.82 is used here as the tightest value consistent
#: with that record — treat it as a floor on the real figure, not an
#: instrument reading. (The 0.60 below IS a direct measurement of the mean.)
MEASURED_LARGEST_TILE_RAW_FRACTION = 0.82
#: …and the mean, at which the coarsest level must still be the one selected.
#: Also viewport-DIAGONAL units (pre-#1410).
MEASURED_MEAN_TILE_RAW_FRACTION = 0.60


def _viewer_fill_factor() -> float:
    """The REAL, current ``FILL_FACTOR`` parsed from the viewer's TypeScript
    source (not re-derived from ``MAX_COVERAGE_FRACTION``, whose relation to
    ``FILL_FACTOR`` is itself only checked elsewhere — see
    ``test_max_coverage_fraction_matches_the_viewer_fill_factor`` in
    ``test_lod_group.py``). Reading the live constant is what makes a future
    anchor move fail HERE, on this calibration, instead of only on that
    separate coupling test.
    """
    registry = viewer_source("src/scene/lod-group-registry.ts")
    source = registry.read_text(encoding="utf-8")
    return read_ts_number_const(source, "FILL_FACTOR")


def test_occurrence_coverage_has_one_threshold_per_level() -> None:
    assert len(OCCURRENCE_COVERAGE) == OCCURRENCE_LOD_LEVELS + 1


def test_occurrence_coverage_is_a_valid_ladder() -> None:
    """Strictly ascending from the always-eligible floor up to the ceiling."""
    covs = list(OCCURRENCE_COVERAGE)
    assert covs[0] == 0.0
    assert all(covs[i] > covs[i - 1] for i in range(1, len(covs)))
    assert max(covs) == pytest.approx(MAX_COVERAGE_FRACTION)
    assert all(0.0 <= c <= MAX_COVERAGE_FRACTION for c in covs)


def test_occurrence_coverage_clears_the_measured_largest_tile() -> None:
    """The calibration's core claim, restated in metric space.

    Selection compares ``threshold <= rawFraction / FILL_FACTOR``, where
    ``rawFraction`` is in FITTED-AXIS units (post-#1410). The measured largest
    tile fraction is in the older DIAGONAL units, so it is converted through
    the named 16:9-anchored factor above before dividing by the REAL
    ``FILL_FACTOR`` read from the viewer source. Every non-zero threshold must
    sit ABOVE the resulting metric, so all tiles hold their coarsest level at
    whole-globe framing; a threshold at or below it is what made the tiles
    flap (both levels resident, 1.07M instead of 186k — see the constant's
    comment).

    Measured (this test, current constants): 0.82 diagonal-fraction ->
    1.6726 fitted-axis-fraction (x2.0397) -> metric 3.3452 (/ FILL_FACTOR
    0.5). The first rung (3.52) clears it by only ~0.17 — down from the 0.24
    margin the OLD (anchor-blind) computation reported, because the honest
    conversion is closer to that rung than the pre-#1410 arithmetic was. Real
    margin, not a hair's-breadth one, but noticeably tighter than it looks
    from the ladder's face value.
    """
    fill_factor = _viewer_fill_factor()
    largest_tile_raw_fitted_axis = (
        MEASURED_LARGEST_TILE_RAW_FRACTION * _REFERENCE_ASPECT_DIAGONAL_TO_FITTED_AXIS
    )
    largest_tile_metric = largest_tile_raw_fitted_axis / fill_factor
    non_zero = [c for c in OCCURRENCE_COVERAGE if c > 0.0]
    assert non_zero, "ladder has no refinement thresholds"
    assert min(non_zero) > largest_tile_metric, (
        f"the first refinement threshold {min(non_zero)} must clear the largest "
        f"per-tile metric {largest_tile_metric} at whole-globe framing"
    )


def test_occurrence_coverage_keeps_the_coarsest_level_at_whole_globe_framing() -> None:
    """End-to-end intent: at the measured mean framing the selected level is the
    coarsest (index 0), i.e. the cheap overview the demo is built around."""
    fill_factor = _viewer_fill_factor()
    mean_tile_raw_fitted_axis = (
        MEASURED_MEAN_TILE_RAW_FRACTION * _REFERENCE_ASPECT_DIAGONAL_TO_FITTED_AXIS
    )
    metric = mean_tile_raw_fitted_axis / fill_factor
    # The viewer picks the finest child whose threshold <= metric.
    selected = max(i for i, c in enumerate(OCCURRENCE_COVERAGE) if c <= metric)
    assert selected == 0, (
        f"whole-globe framing (metric {metric}) selects level {selected}; the "
        "demo's calibration requires the coarsest level there"
    )


def test_occurrence_coverage_refines_once_a_tile_fills_the_viewport() -> None:
    """The ceiling must be reachable: a tile that fills the screen (its own
    projected diagonal reaching ``SCREEN_FILL_DIAGONAL_RATIO`` times the
    fitted axis — the viewer's own definition of "screen-filling", see
    ``lod-group-registry.ts``) selects the finest level, so the ladder is not
    dead weight. Unlike the two calibration tests above this isn't a
    diagonal-era MEASUREMENT to convert — it's the viewer's own screen-fill
    definition, read live from both TypeScript constants."""
    registry = viewer_source("src/scene/lod-group-registry.ts")
    source = registry.read_text(encoding="utf-8")
    fill_factor = read_ts_number_const(source, "FILL_FACTOR")
    screen_fill_diagonal_ratio = read_ts_number_const(
        source, "SCREEN_FILL_DIAGONAL_RATIO"
    )
    metric = screen_fill_diagonal_ratio / fill_factor
    selected = max(i for i, c in enumerate(OCCURRENCE_COVERAGE) if c <= metric)
    assert selected == len(OCCURRENCE_COVERAGE) - 1


def test_the_globe_is_built_at_the_scene_radius() -> None:
    """The globe must be at ``RADIUS``, not at a hardcoded unit sphere.

    This shipped wrong and was hard to recognise. Every occurrence goes through
    the demo's own ``lonlat_to_xyz``, which multiplies by ``RADIUS`` (100), so a
    globe built at radius 1.0 is a marble at the centre of a 100-unit point shell.
    It renders perfectly — committed, textured, in frustum — and occupies about 1%
    of the frame, which is why it read as "the terrain is missing" and why no
    amount of node intensity fixed it. The stored bounds are what gave it away:
    the occurrence layer spanned +/-100 while the globe spanned +/-1.

    A scale mismatch between a backdrop and the data drawn on it is invisible to
    every other check in this suite, so it gets its own.
    """
    source = Path(demo_module.__file__).read_text()
    globe_call = source.split("build_earth(")[1].split("\n            )")[0]
    assert "radius=RADIUS" in globe_call, (
        "the globe must use the demo's RADIUS; a literal would silently rescale it"
    )
    assert "radius=1.0" not in globe_call


def test_the_scene_does_not_open_on_an_empty_slice() -> None:
    """The opening ``current_step`` must name a cell that has occurrences in it.

    The viewer shows the INTERSECTION of the non-displayed slices, so an opening
    pin at a cell no layer occupies yields a valid scene whose first frame is just
    the context globe. That is what happened when the dense all-life summary layer
    was removed from under an opening pin still aimed at ``(All life, All years)``:
    the compile succeeded, the store validated, every array round-tripped, and the
    demo simply came up with no data on it.

    So the invariant is checked two ways, because the source-text half alone would
    not have caught it (the pin *was* explicit and deliberate -- it just pointed at
    a cell whose contents had been deleted elsewhere in the file):

    * the pin does not name ``ALL_LIFE_SLOT``;
    * and ``assert_opening_slice_is_populated`` actually rejects that cell, so the
      build-time guard is not vacuous.
    """
    source = Path(demo_module.__file__).read_text()
    step = source.split("current_step=[")[1].split("]")[0]
    assert "taxon_slot(OPENING_TAXON_GROUP)" in step, (
        "the opening taxon must be a real group, not the all-taxa corner"
    )
    assert "ALL_LIFE_SLOT" not in step

    opening_taxon = float(demo_module.taxon_slot(demo_module.OPENING_TAXON_GROUP))
    opening_period = float(demo_module.PERIOD_ALL_SLOT)
    assert opening_taxon != float(demo_module.ALL_LIFE_SLOT)

    # One occurrence at the opening cell, one at the (all taxa, decade) marginal.
    positions = np.array(
        [
            [0.0, 0.0, 0.0, opening_taxon, opening_period],
            [0.0, 0.0, 0.0, float(demo_module.ALL_LIFE_SLOT), 3.0],
        ],
        dtype=np.float32,
    )
    demo_module.assert_opening_slice_is_populated(
        positions, opening_taxon, opening_period
    )
    # ... and the guard must REFUSE the corner the summary layer used to fill,
    # which is what makes the assertion above worth anything.
    with pytest.raises(RuntimeError, match="would open on"):
        demo_module.assert_opening_slice_is_populated(
            positions, float(demo_module.ALL_LIFE_SLOT), opening_period
        )
