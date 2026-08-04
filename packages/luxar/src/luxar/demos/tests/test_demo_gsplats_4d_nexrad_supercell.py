"""Tests for the pure helpers in demo_gsplats_4d_nexrad_supercell.

No network, no MetPy, no GPU — only the radar geometry, the split-cut sweep
selection, the coverage mask and the Barnes gridder. The demo is loaded by file
path (see test_demo_gsplats_3d_tng_cosmic_web for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_4d_nexrad_supercell.py"


def _load_demo_module(name: str = "_luxar_demo_nexrad_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
beam_height_and_arc = _demo.beam_height_and_arc
inverse_beam = _demo.inverse_beam
select_reflectivity_sweeps = _demo.select_reflectivity_sweeps
coverage_mask = _demo.coverage_mask
grid_volume = _demo.grid_volume
dbz_to_intensity = _demo.dbz_to_intensity


def _fake_sweep(el_angle: float, num_gates: int, n_rays: int = 8) -> list:
    """Build a minimal stand-in for one MetPy sweep.

    A MetPy radial is a 5-tuple; only ``ray[0].el_angle`` and the ``b"REF"``
    entry of ``ray[4]`` matter to the sweep selector.
    """
    header = types.SimpleNamespace(
        num_gates=num_gates, gate_width=0.25, first_gate=2.125
    )
    rays = []
    for _ in range(n_rays):
        hdr = types.SimpleNamespace(el_angle=el_angle, az_angle=0.0, el_num=0)
        rays.append((hdr, None, None, None, {b"REF": (header, np.zeros(num_gates))}))
    return rays


class TestBeamGeometry:
    def test_zero_elevation_beam_rises_with_range(self) -> None:
        """A horizontal beam still climbs, because the earth curves away."""
        z, s = beam_height_and_arc(np.array([10.0, 50.0, 100.0]), np.array([0.0]))
        assert np.all(np.diff(z) > 0)
        # ~4/3-earth rule of thumb: about 0.6 km at 100 km range.
        assert 0.5 < z[-1] < 0.7

    def test_height_increases_with_elevation_angle(self) -> None:
        r = np.array([60.0])
        z_low, _ = beam_height_and_arc(r, np.array([0.5]))
        z_high, _ = beam_height_and_arc(r, np.array([19.5]))
        assert z_high > z_low
        # 60 km at 19.5 deg is roughly 20 km up.
        assert 19.0 < z_high.item() < 22.0

    def test_arc_distance_never_exceeds_slant_range(self) -> None:
        r = np.array([[5.0, 50.0, 200.0]])
        for el in (0.0, 5.0, 19.5):
            _, s = beam_height_and_arc(r, np.array([[el]]))
            assert np.all(s <= r + 1e-9)

    def test_inverse_round_trips(self) -> None:
        r = np.array([[5.0], [40.0], [120.0], [230.0]])
        el = np.array([[0.5, 3.0, 10.0, 19.5]])
        z, s = beam_height_and_arc(r, el)
        r_back, el_back = inverse_beam(s, z)
        assert np.allclose(r_back, np.broadcast_to(r, z.shape), atol=1e-6)
        assert np.allclose(el_back, np.broadcast_to(el, z.shape), atol=1e-6)


class TestSelectReflectivitySweeps:
    def test_split_cut_pair_collapses_to_the_longer_range_member(self) -> None:
        """VCP 212's 0.9 deg cut appears twice; keep the surveillance half.

        This is the regression that matters: rounding the elevation angle puts
        0.80 and 0.92 in different buckets, so a naive dedup keeps both and
        double-counts the tilt.
        """
        sweeps = [_fake_sweep(0.80, 1832), _fake_sweep(0.92, 1192)]
        assert select_reflectivity_sweeps(sweeps) == [0]

    def test_split_cut_dedup_ignores_elevation_number(self) -> None:
        """Deduping must not depend on el_num, which is a cut index.

        Both halves here report el_num values that differ, exactly as the ICD
        allows; the selector must still collapse them.
        """
        sweeps = [_fake_sweep(0.48, 1832), _fake_sweep(0.53, 1192)]
        for i, sweep in enumerate(sweeps):
            for ray in sweep:
                ray[0].el_num = i + 1
        assert select_reflectivity_sweeps(sweeps) == [0]

    def test_distinct_elevations_are_all_kept(self) -> None:
        sweeps = [_fake_sweep(el, 800) for el in (0.5, 1.5, 2.4, 3.4, 19.5)]
        assert select_reflectivity_sweeps(sweeps) == [0, 1, 2, 3, 4]

    def test_sweeps_without_reflectivity_are_skipped(self) -> None:
        sweeps = [
            _fake_sweep(0.5, 1832),
            [(types.SimpleNamespace(el_angle=1.5), None, None, None, {})],
        ]
        assert select_reflectivity_sweeps(sweeps) == [0]

    def test_result_is_ascending_in_elevation(self) -> None:
        sweeps = [_fake_sweep(el, 900) for el in (5.0, 0.5, 12.0, 2.0)]
        kept = select_reflectivity_sweeps(sweeps)
        angles = [sweeps[i][0][0].el_angle for i in kept]
        assert angles == sorted(angles)

    def test_full_vcp212_volume_loses_only_the_split_duplicates(self) -> None:
        """17 sweeps in, one per true elevation out — the measured behaviour."""
        split_pairs = [
            (0.49, 1832),
            (0.53, 1192),
            (0.80, 1832),
            (0.92, 1192),
            (1.22, 1720),
            (1.36, 1192),
        ]
        singles = [
            (1.98, 1536),
            (2.30, 1336),
            (3.08, 1160),
            (3.92, 984),
            (5.02, 820),
            (6.33, 680),
            (8.18, 540),
            (9.98, 456),
            (12.38, 368),
            (15.62, 296),
            (19.73, 240),
        ]
        sweeps = [_fake_sweep(el, n) for el, n in split_pairs + singles]
        kept = select_reflectivity_sweeps(sweeps)
        # The three split pairs collapse; every genuine tilt survives.
        assert len(kept) == 14
        assert all(sweeps[i][0][4][b"REF"][0].num_gates != 1192 for i in kept)


class TestCoverageMask:
    def test_masks_above_the_highest_tilt_and_below_the_lowest(self) -> None:
        mask = coverage_mask(el_min=0.5, el_max=19.5, r_max_km=460.0)
        z_ax, y_ax, x_ax = _demo.grid_axes()
        assert mask.shape == (z_ax.size, y_ax.size, x_ax.size)

        def covered(east_km: float, north_km: float, up_km: float) -> bool:
            iz = int(np.argmin(np.abs(z_ax - up_km)))
            iy = int(np.argmin(np.abs(y_ax - north_km)))
            ix = int(np.argmin(np.abs(x_ax - east_km)))
            return bool(mask[iz, iy, ix])

        # 60 km out, 6 km up is ~5.5 deg — comfortably inside the swept range.
        assert covered(-60.0, 0.0, 6.0)
        # 10 km out, 17 km up is ~59 deg — far above the 19.5 deg top tilt.
        assert not covered(-10.0, 0.0, 17.0)
        # 100 km out, 0.5 km up is below the lowest tilt (the near-ground gap).
        assert not covered(-100.0, 0.0, 0.5)

    def test_is_cached_between_calls(self) -> None:
        a = coverage_mask(0.5, 19.5, 460.0)
        b = coverage_mask(0.5, 19.5, 460.0)
        assert a is b


class TestDbzToIntensity:
    def test_clips_below_floor_to_zero_and_is_non_negative(self) -> None:
        grid = np.array([[[-30.0, 0.0, _demo.DBZ_FLOOR, 45.0, 90.0]]])
        out = dbz_to_intensity(grid)
        assert out[0, 0, 0] == 0.0
        assert out[0, 0, 2] == 0.0
        assert np.all(out >= 0.0)

    def test_saturates_at_the_ceiling(self) -> None:
        grid = np.array([[[_demo.DBZ_CEIL, _demo.DBZ_CEIL + 25.0]]])
        out = dbz_to_intensity(grid)
        assert out[0, 0, 0] == pytest.approx(1.0)
        assert out[0, 0, 1] == pytest.approx(1.0)

    def test_no_data_sentinels_become_zero(self) -> None:
        grid = np.array([[[-np.inf, np.nan, 40.0]]])
        out = dbz_to_intensity(grid)
        assert out[0, 0, 0] == 0.0
        assert out[0, 0, 1] == 0.0
        assert out[0, 0, 2] > 0.0

    def test_is_linear_in_dbz_not_in_power(self) -> None:
        """A 20 dBZ step must be a constant amplitude step, not a 100x one."""
        grid = np.array([[[30.0, 50.0, 70.0]]])
        out = dbz_to_intensity(grid)[0, 0]
        assert (out[1] - out[0]) == pytest.approx(out[2] - out[1], rel=1e-6)


class TestGridVolume:
    def test_empty_neighbourhoods_report_no_data(self) -> None:
        """Cells with no gate inside the ROI must come back -inf, not zero dBZ."""
        # One gate, parked far outside the box.
        xyz = np.array([[1000.0, 1000.0, 5.0]])
        dbz = np.array([40.0], dtype=np.float32)
        mask = np.ones(tuple(a.size for a in _demo.grid_axes()), dtype=bool)
        grid = grid_volume(xyz, dbz, mask)
        assert np.all(np.isneginf(grid))

    def test_masked_cells_are_marked_no_data(self) -> None:
        xyz = np.array([[-60.0, 20.0, 5.0]])
        dbz = np.array([50.0], dtype=np.float32)
        shape = tuple(a.size for a in _demo.grid_axes())
        mask = np.zeros(shape, dtype=bool)
        grid = grid_volume(xyz, dbz, mask)
        assert np.all(np.isneginf(grid))

    def test_barnes_returns_a_weighted_mean_bounded_by_its_inputs(self) -> None:
        """Two gates straddling a cell: the answer sits between their values."""
        z_ax, y_ax, x_ax = _demo.grid_axes()
        # Pick an interior cell and place gates just either side of it.
        cx, cy, cz = float(x_ax[140]), float(y_ax[120]), float(z_ax[10])
        xyz = np.array([[cx - 0.3, cy, cz], [cx + 0.3, cy, cz]])
        dbz = np.array([20.0, 60.0], dtype=np.float32)
        mask = np.zeros((z_ax.size, y_ax.size, x_ax.size), dtype=bool)
        mask[10, 120, 140] = True
        grid = grid_volume(xyz, dbz, mask)
        value = grid[10, 120, 140]
        assert 20.0 <= value <= 60.0
        # Equidistant gates -> the mean.
        assert value == pytest.approx(40.0, abs=1e-3)

    def test_nearer_gate_dominates(self) -> None:
        z_ax, y_ax, x_ax = _demo.grid_axes()
        cx, cy, cz = float(x_ax[140]), float(y_ax[120]), float(z_ax[10])
        xyz = np.array([[cx + 0.05, cy, cz], [cx + 1.2, cy, cz]])
        dbz = np.array([60.0, 20.0], dtype=np.float32)
        mask = np.zeros((z_ax.size, y_ax.size, x_ax.size), dtype=bool)
        mask[10, 120, 140] = True
        grid = grid_volume(xyz, dbz, mask)
        assert grid[10, 120, 140] > 50.0


class TestFrameSelection:
    def test_full_run_uses_every_pinned_scan(self) -> None:
        idx = _demo._select_frame_indices(len(_demo.VOLUME_SCANS))
        assert idx == list(range(len(_demo.VOLUME_SCANS)))

    def test_subsampling_spans_the_hour(self) -> None:
        idx = _demo._select_frame_indices(4)
        assert len(idx) == 4
        assert idx[0] == 0
        assert idx == sorted(set(idx))
        # A subsample that only covered the first few scans would miss the
        # tornado entirely.
        assert idx[-1] >= len(_demo.VOLUME_SCANS) // 2


class TestPinnedScanList:
    def test_scan_times_are_ascending_and_well_formed(self) -> None:
        assert len(_demo.VOLUME_SCANS) == 82
        assert list(_demo.VOLUME_SCANS) == sorted(_demo.VOLUME_SCANS)
        for date_prefix, t in _demo.VOLUME_SCANS:
            assert date_prefix in ("2013/05/31", "2013/06/01")
            assert len(t) == 6 and t.isdigit()

    def test_window_spans_21z_to_03z_across_midnight(self) -> None:
        """The window deliberately crosses radar-days; a bare time is ambiguous."""
        first_day = [t for d, t in _demo.VOLUME_SCANS if d == "2013/05/31"]
        second_day = [t for d, t in _demo.VOLUME_SCANS if d == "2013/06/01"]
        assert first_day and second_day
        assert all(21 <= int(t[:2]) <= 23 for t in first_day)
        assert all(0 <= int(t[:2]) <= 2 for t in second_day)

    def test_scan_label_disambiguates_the_two_days(self) -> None:
        labels = [_demo.scan_label(i) for i in range(len(_demo.VOLUME_SCANS))]
        assert labels[0].startswith("05-31 21:")
        assert labels[-1].startswith("06-01 02:")
        assert len(set(labels)) == len(labels)

    def test_bundle_names_match_the_per_frame_cache_names(self) -> None:
        """The LFS bundle's inner names must equal the recompute cache names.

        If these drift, a warm bundle silently misses and every user re-fits.
        """
        for i in range(len(_demo.VOLUME_SCANS)):
            name = _demo._frame_cache_file(i).name
            assert name.endswith(".gsplats.zarr.zip")
            assert f"{i:04d}" in name


class TestTornadoTrack:
    """The marker is a surveyed damage path, so it must land where radar sees it."""

    def test_track_falls_inside_the_storm_box(self) -> None:
        for lat, lon in (_demo.TORNADO_START_LATLON, _demo.TORNADO_END_LATLON):
            east, north = _demo.latlon_to_local_km(lat, lon)
            assert _demo.BOX_X_KM[0] < east < _demo.BOX_X_KM[1]
            assert _demo.BOX_Y_KM[0] < north < _demo.BOX_Y_KM[1]

    def test_track_agrees_with_the_radar_hook_echo(self) -> None:
        """Independent cross-check of the geolocation math.

        The hook echo is at roughly (-70, +20) km in the gridded reflectivity.
        The damage survey is a completely separate source, so agreement to a few
        km is strong evidence the beam model and azimuth convention are right.
        A sign error in either would throw this off by tens of km.
        """
        e0, n0 = _demo.latlon_to_local_km(*_demo.TORNADO_START_LATLON)
        e1, n1 = _demo.latlon_to_local_km(*_demo.TORNADO_END_LATLON)
        mid = ((e0 + e1) / 2.0, (n0 + n1) / 2.0)
        assert np.hypot(mid[0] - (-70.0), mid[1] - 20.0) < 10.0

    def test_path_length_matches_the_surveyed_16_2_miles(self) -> None:
        e0, n0 = _demo.latlon_to_local_km(*_demo.TORNADO_START_LATLON)
        e1, n1 = _demo.latlon_to_local_km(*_demo.TORNADO_END_LATLON)
        straight = np.hypot(e1 - e0, n1 - n0)
        # The survey's 16.2 mi (26.1 km) is measured ALONG a slightly curved path,
        # so a straight chord must be a little shorter but the same order.
        assert 18.0 < straight < 26.5

    def test_position_is_none_outside_the_tornado_window(self) -> None:
        t0 = _demo._utc_seconds(*_demo.TORNADO_START_UTC)
        t1 = _demo._utc_seconds(*_demo.TORNADO_END_UTC)
        assert _demo.tornado_position_km(t0 - 60) is None
        assert _demo.tornado_position_km(t1 + 60) is None
        assert _demo.tornado_position_km(t0) is not None
        assert _demo.tornado_position_km(t1) is not None

    def test_position_advances_monotonically_east(self) -> None:
        t0 = _demo._utc_seconds(*_demo.TORNADO_START_UTC)
        t1 = _demo._utc_seconds(*_demo.TORNADO_END_UTC)
        easts = [
            _demo.tornado_position_km(t0 + int(f * (t1 - t0)))[0]
            for f in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert easts == sorted(easts)

    def test_midnight_rollover_in_utc_seconds(self) -> None:
        """00Z on 06-01 must sort AFTER 23Z on 05-31, not before."""
        a = _demo._utc_seconds("2013/05/31", "235556")
        b = _demo._utc_seconds("2013/06/01", "000034")
        assert b > a
        assert b - a == 278

    def test_scan_seconds_are_strictly_increasing(self) -> None:
        secs = [_demo.scan_utc_seconds(i) for i in range(len(_demo.VOLUME_SCANS))]
        assert secs == sorted(secs)
        assert len(set(secs)) == len(secs)

    def test_marker_segments_are_vertex_pairs_on_covered_frames(self) -> None:
        indices = list(range(len(_demo.VOLUME_SCANS)))
        verts, frames = _demo.tornado_marker_segments(indices)
        assert len(frames) > 0
        assert verts.shape == (2 * len(frames), 4)
        # Each pair shares east/north/time and differs only in height.
        for k in range(len(frames)):
            bot, top = verts[2 * k], verts[2 * k + 1]
            assert bot[1] == top[1] and bot[2] == top[2] and bot[3] == top[3]
            assert top[0] > bot[0]
            assert top[0] == pytest.approx(_demo.TORNADO_MARKER_TOP_KM)

    def test_marker_only_covers_the_tornado_frames(self) -> None:
        indices = list(range(len(_demo.VOLUME_SCANS)))
        _, frames = _demo.tornado_marker_segments(indices)
        t0 = _demo._utc_seconds(*_demo.TORNADO_START_UTC)
        t1 = _demo._utc_seconds(*_demo.TORNADO_END_UTC)
        for t in frames:
            assert t0 <= _demo.scan_utc_seconds(indices[t]) <= t1
        # ~40 min at ~4.3 min cadence -> a handful of frames, not all 82.
        assert 5 <= len(frames) <= 12


class TestCacheIdentity:
    """Every parameter that changes the FITTED splats must be in the cache key."""

    def test_dbz_floor_participates_in_the_fit_identity(self, monkeypatch) -> None:
        """Regression: --dbz-floor silently reused splats fitted at the default.

        The floor is applied by dbz_to_intensity BEFORE the fit, so it changes
        the splats — but it is not part of the grid geometry, so leaving it out
        of the key made the flag a no-op against any warm cache or the shipped
        bundle.
        """
        before = _demo._frame_cache_file(0).name
        monkeypatch.setattr(_demo, "DBZ_FLOOR", 30.0)
        after = _demo._frame_cache_file(0).name
        assert before != after, "changing the dBZ floor must change the cache name"

    def test_geometry_and_budget_both_participate(self, monkeypatch) -> None:
        base = _demo._frame_cache_file(0).name
        monkeypatch.setattr(_demo, "GRID_Z_M", 3000)
        assert _demo._frame_cache_file(0).name != base
        monkeypatch.setattr(_demo, "GRID_Z_M", 1500)
        monkeypatch.setattr(_demo, "SPLATS_OVERRIDE", 12345)
        assert _demo._frame_cache_file(0).name != base


class TestFloatFlags:
    def test_fractional_values_survive(self, monkeypatch) -> None:
        """Regression: parse_int_arg truncated 22.5 -> 22 with no warning."""
        monkeypatch.setattr(_demo.sys, "argv", ["x", "--dbz-floor=22.5"])
        assert _demo._parse_float_arg("dbz-floor", 20.0) == pytest.approx(22.5)

    def test_space_separated_form(self, monkeypatch) -> None:
        monkeypatch.setattr(_demo.sys, "argv", ["x", "--vert-exag", "2.5"])
        assert _demo._parse_float_arg("vert-exag", 1.0) == pytest.approx(2.5)

    def test_absent_and_malformed_fall_back(self, monkeypatch) -> None:
        monkeypatch.setattr(_demo.sys, "argv", ["x"])
        assert _demo._parse_float_arg("dbz-floor", 20.0) == 20.0
        monkeypatch.setattr(_demo.sys, "argv", ["x", "--dbz-floor=abc"])
        assert _demo._parse_float_arg("dbz-floor", 20.0) == 20.0


class TestDbzFloorValidation:
    """An out-of-range floor must be rejected loudly, not produce garbage.

    At floor == DBZ_CEIL the intensity span is zero, so every frame becomes an
    empty-sky placeholder and the global rescale divides by zero; ABOVE the
    ceiling np.clip's bounds invert and every cell — no-data included — maps to
    full intensity; at or below NO_DATA_DBZ the cached sentinel reads as echo.
    """

    @pytest.mark.parametrize("bad", ["70", "75", "-999", "-1500"])
    def test_out_of_range_floor_is_rejected_at_startup(
        self, monkeypatch, bad: str
    ) -> None:
        name = "_luxar_demo_nexrad_floor_check"
        monkeypatch.setattr(sys, "argv", ["demo", f"--dbz-floor={bad}"])
        try:
            with pytest.raises(SystemExit, match="dbz-floor"):
                _load_demo_module(name)
        finally:
            sys.modules.pop(name, None)

    def test_valid_floors_are_accepted(self, monkeypatch) -> None:
        name = "_luxar_demo_nexrad_floor_check"
        for good in ("0", "35.5", "69.9"):
            monkeypatch.setattr(sys, "argv", ["demo", f"--dbz-floor={good}"])
            try:
                module = _load_demo_module(name)
                assert module.DBZ_FLOOR == pytest.approx(float(good))
            finally:
                sys.modules.pop(name, None)


class Test4DAssembly:
    """The all-placeholder edge of combine_timepoints_to_4d."""

    @staticmethod
    def _frame(peak_amplitude: float):
        from luxar.gsplats.gsplat_data import GSplatData

        ndim = 3
        return GSplatData(
            centers=np.zeros((1, ndim), dtype=np.float32),
            amplitudes=np.full(1, peak_amplitude, dtype=np.float32),
            cholesky_factors=np.eye(ndim, dtype=np.float32)[np.tril_indices(ndim)][
                None, :
            ],
            truncation_radius=_demo.TRUNCATE_SIGMAS,
        )

    def test_all_placeholder_selection_fails_loudly(self) -> None:
        """A valid but high floor can leave every frame empty; the global
        rescale must refuse rather than divide by zero."""
        with pytest.raises(SystemExit, match="empty-sky"):
            _demo.combine_timepoints_to_4d([self._frame(0.0), self._frame(0.0)])

    def test_mixed_selection_scales_to_the_global_peak(self) -> None:
        combined = _demo.combine_timepoints_to_4d([self._frame(0.0), self._frame(0.5)])
        assert combined.ndim == 4
        assert float(combined.amplitudes.max()) == pytest.approx(_demo.AMPLITUDE_PEAK)


class TestFrameSubsampling:
    def test_subsample_spans_the_whole_window(self) -> None:
        """Regression: range(0, 82, 82//4) gave [0,20,40,60], dropping 21 scans.

        A subsampled run must SAMPLE the event, not truncate it.
        """
        total = len(_demo.VOLUME_SCANS)
        for n in (2, 3, 4, 7, 13, 40):
            idx = _demo._select_frame_indices(n)
            assert idx[0] == 0
            assert idx[-1] == total - 1, f"n={n} stopped at {idx[-1]} of {total - 1}"
            assert idx == sorted(set(idx))
            assert len(idx) <= n

    def test_single_frame_and_full_run(self) -> None:
        assert _demo._select_frame_indices(1) == [0]
        full = _demo._select_frame_indices(len(_demo.VOLUME_SCANS))
        assert full == list(range(len(_demo.VOLUME_SCANS)))


class TestVerticalExaggeration:
    def test_wireframe_and_marker_scale_with_the_splats(self, monkeypatch) -> None:
        """Regression: only the splats were stretched, so the storm grew out
        through the top of the box that is meant to bound it."""
        corners_1x, _ = _demo.domain_box_wireframe()
        verts_1x, frames = _demo.tornado_marker_segments(
            list(range(len(_demo.VOLUME_SCANS)))
        )
        monkeypatch.setattr(_demo, "VERT_EXAG", 3.0)
        corners_3x, _ = _demo.domain_box_wireframe()
        verts_3x, _ = _demo.tornado_marker_segments(
            list(range(len(_demo.VOLUME_SCANS)))
        )
        # Column 0 is `up`; heights must triple, ground plan must not move.
        assert corners_3x[:, 0].max() == pytest.approx(corners_1x[:, 0].max() * 3.0)
        assert np.allclose(corners_3x[:, 1:], corners_1x[:, 1:])
        assert len(frames) > 0
        assert verts_3x[:, 0].max() == pytest.approx(verts_1x[:, 0].max() * 3.0)
        assert np.allclose(verts_3x[:, 1:], verts_1x[:, 1:])
