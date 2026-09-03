"""Smoke tests for the pure helpers in demo_desi_galaxies.

The bulk exercise the deterministic array helpers only (no network, no astropy
read). ``TestOrbitCentre`` additionally builds two small synthetic scenes to pin
the camera-framing behaviour; it is marked ``slow`` so the ``-m 'not slow'`` CI
job skips the compiler passes. The demo is loaded by file path (see
test_demo_ppi_flow_field).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from luxar._zarr_compat import consolidate, create_array, open_group
from luxar.typing_utils.constants import MAX_POINTS_PER_POINTS_NODE

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_desi_galaxies.py"


def _load_demo_module():
    name = "_luxar_demo_desi_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
radec_z_to_xyz = _demo.radec_z_to_xyz
tracer_colors = _demo.tracer_colors
redshift_colors = _demo.redshift_colors
quantize_positions = _demo.quantize_positions
dequantize_positions = _demo.dequantize_positions
save_derived = _demo.save_derived
load_derived = _demo.load_derived
sample_scene_catalog = _demo.sample_scene_catalog


def test_per_node_budget_stays_under_the_point_texture_bound() -> None:
    assert _demo.SCENE_MAX_POINTS_PER_NODE <= MAX_POINTS_PER_POINTS_NODE


class TestRadecToXyz:
    def test_origin_axis_directions(self) -> None:
        # (RA=0, Dec=0) at distance d → +x axis.
        p = radec_z_to_xyz(np.array([0.0]), np.array([0.0]), np.array([100.0]))
        np.testing.assert_allclose(p[0], [100.0, 0.0, 0.0], atol=1e-3)
        # (RA=90, Dec=0) → +y.
        p = radec_z_to_xyz(np.array([90.0]), np.array([0.0]), np.array([100.0]))
        np.testing.assert_allclose(p[0], [0.0, 100.0, 0.0], atol=1e-3)
        # (Dec=90) → +z (north pole), independent of RA.
        p = radec_z_to_xyz(np.array([37.0]), np.array([90.0]), np.array([100.0]))
        np.testing.assert_allclose(p[0], [0.0, 0.0, 100.0], atol=1e-3)

    def test_radius_preserved(self) -> None:
        rng = np.random.default_rng(0)
        ra = rng.uniform(0, 360, 500)
        dec = rng.uniform(-90, 90, 500)
        d = rng.uniform(10, 3000, 500)
        p = radec_z_to_xyz(ra, dec, d)
        np.testing.assert_allclose(np.linalg.norm(p, axis=1), d, rtol=1e-4)
        assert p.dtype == np.float32


class TestTracerColors:
    def test_maps_ids_to_palette(self) -> None:
        cols = tracer_colors(np.array([0, 1, 2, 3], dtype=np.uint8))
        assert cols.shape == (4, 3)
        assert cols.dtype == np.float32
        # Each row is a distinct, in-gamut color.
        assert cols.min() >= 0.0 and cols.max() <= 1.0
        assert len({tuple(row) for row in cols}) == 4


class TestRedshiftColors:
    def test_shape_dtype_gamut(self) -> None:
        z = np.linspace(0.01, 3.5, 100).astype(np.float32)
        cols = redshift_colors(z)
        assert cols.shape == (100, 3)
        assert cols.dtype == np.float32
        assert cols.min() >= 0.0 and cols.max() <= 1.0

    def test_low_vs_high_z_distinct(self) -> None:
        # turbo maps low→cold, high→hot; nearby and distant must differ.
        cols = redshift_colors(np.array([0.02, 0.05, 0.5, 1.0, 3.0], dtype=np.float32))
        assert not np.allclose(cols[0], cols[-1])
        assert len({tuple(np.round(c, 3)) for c in cols}) >= 4

    def test_degenerate_and_empty(self) -> None:
        # all-equal redshift → valid (no div-by-zero), single color.
        same = redshift_colors(np.full(5, 0.3, dtype=np.float32))
        assert same.shape == (5, 3) and np.isfinite(same).all()
        empty = redshift_colors(np.array([], dtype=np.float32))
        assert empty.shape == (0, 3)


class TestQuantizeRoundtrip:
    def test_positions_roundtrip_sub_mpc(self) -> None:
        rng = np.random.default_rng(1)
        pos = rng.uniform(-3000, 3000, size=(2000, 3)).astype(np.float32)
        q, offset, scale = quantize_positions(pos)
        assert q.dtype == np.int16
        back = dequantize_positions(q, offset, scale)
        # 6000 Mpc span / 65534 levels ≈ 0.09 Mpc/step → within ~0.1 Mpc.
        assert np.max(np.abs(back - pos)) < 0.15

    def test_derived_npz_roundtrip(self, tmp_path) -> None:
        rng = np.random.default_rng(2)
        pos = rng.uniform(-2000, 2000, size=(1000, 3)).astype(np.float32)
        z = rng.uniform(0.01, 3.9, size=1000).astype(np.float32)
        tid = rng.integers(0, 4, size=1000).astype(np.uint8)
        p = tmp_path / "d.npz"
        save_derived(p, pos, z, tid)
        assert p.stat().st_size > 0
        pos2, z2, tid2 = load_derived(p)
        assert pos2.dtype == np.float32 and z2.dtype == np.float32
        np.testing.assert_array_equal(tid2, tid)
        assert np.max(np.abs(pos2 - pos)) < 0.15
        # float16 redshift → ~3 significant digits.
        np.testing.assert_allclose(z2, z, atol=2e-3)


class TestSceneCatalogSampling:
    def test_caps_deterministically_and_keeps_rows_aligned(self) -> None:
        n = 100
        row_ids = np.arange(n, dtype=np.int64)
        positions = np.column_stack([row_ids, row_ids + 100, row_ids + 200])
        redshift = row_ids.astype(np.float32)
        tracer_ids = (row_ids % 4).astype(np.uint8)

        first = sample_scene_catalog(positions, redshift, tracer_ids, max_points=25)
        second = sample_scene_catalog(positions, redshift, tracer_ids, max_points=25)

        for first_array, second_array in zip(first, second):
            np.testing.assert_array_equal(first_array, second_array)
            assert len(first_array) == 25
        sampled_positions, sampled_redshift, sampled_tracers = first
        np.testing.assert_array_equal(sampled_positions[:, 0], sampled_redshift)
        np.testing.assert_array_equal(
            sampled_tracers, sampled_redshift.astype(np.uint8) % 4
        )
        assert len(np.unique(sampled_redshift)) == 25

    def test_does_not_copy_catalog_below_cap(self) -> None:
        positions = np.zeros((5, 3), dtype=np.float32)
        redshift = np.zeros(5, dtype=np.float32)
        tracer_ids = np.zeros(5, dtype=np.uint8)

        sampled = sample_scene_catalog(positions, redshift, tracer_ids, max_points=10)

        assert sampled[0] is positions
        assert sampled[1] is redshift
        assert sampled[2] is tracer_ids

    def test_rejects_non_positive_cap(self) -> None:
        with pytest.raises(ValueError, match="max_points must be >= 1"):
            sample_scene_catalog(
                np.zeros((1, 3), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.uint8),
                max_points=0,
            )

    @pytest.mark.parametrize(
        ("positions_n", "redshift_n", "tracer_n"),
        [(3, 2, 3), (3, 3, 2)],
    )
    def test_rejects_misaligned_catalog_columns(
        self, positions_n: int, redshift_n: int, tracer_n: int
    ) -> None:
        with pytest.raises(ValueError, match="same number of rows"):
            sample_scene_catalog(
                np.zeros((positions_n, 3), dtype=np.float32),
                np.zeros(redshift_n, dtype=np.float32),
                np.zeros(tracer_n, dtype=np.uint8),
                max_points=2,
            )


class TestCatalogDownloadErrors:
    def test_http_failure_becomes_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import requests

        requested: list[str] = []

        def _fail_download(url: str, output_path: Path, **kwargs: object) -> Path:
            requested.append(url)
            response = requests.Response()
            response.status_code = 404
            response.reason = "Not Found"
            response.url = url
            raise requests.HTTPError(
                f"404 Client Error: Not Found for url: {url}", response=response
            )

        monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
        monkeypatch.setattr("luxar.demos.robust_download", _fail_download)

        with pytest.raises(_demo.DESICatalogDownloadError) as exc_info:
            _demo.download_catalogs()

        expected_url = f"{_demo.BASE_URL}/BGS_BRIGHT_NGC_clustering.dat.fits"
        assert requested == [expected_url]
        message = str(exc_info.value)
        assert expected_url in message
        assert "HTTP 404 Not Found" in message
        assert "data.desi.lbl.gov" in message
        assert "dataset manifest" in message
        assert "published record" in message
        assert "without --recompute" in message
        assert isinstance(exc_info.value.__cause__, requests.HTTPError)

    def test_connection_failure_becomes_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import requests

        def _fail_download(url: str, output_path: Path, **kwargs: object) -> Path:
            raise requests.ConnectionError("network unreachable")

        monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
        monkeypatch.setattr("luxar.demos.robust_download", _fail_download)

        with pytest.raises(_demo.DESICatalogDownloadError) as exc_info:
            _demo.download_catalogs()

        message = str(exc_info.value)
        assert "ConnectionError: network unreachable" in message
        assert f"{_demo.BASE_URL}/BGS_BRIGHT_NGC_clustering.dat.fits" in message
        assert "dataset manifest" in message
        assert "separate problem from the DESI host outage" in message
        assert isinstance(exc_info.value.__cause__, requests.ConnectionError)

    def test_non_request_failure_is_not_hidden(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fail_download(url: str, output_path: Path, **kwargs: object) -> Path:
            raise ValueError("catalog size mismatch")

        monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
        monkeypatch.setattr("luxar.demos.robust_download", _fail_download)

        with pytest.raises(ValueError, match="catalog size mismatch"):
            _demo.download_catalogs()

    def test_main_prints_download_guidance_and_exits_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        message = "DESI host unavailable; use the published record"

        def _fail_build() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            raise _demo.DESICatalogDownloadError(message)

        monkeypatch.setattr(_demo, "SERVE_ONLY", False)
        monkeypatch.setattr(_demo, "RECOMPUTE", True)
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(_demo, "load_or_build", _fail_build)

        with pytest.raises(SystemExit) as exc_info:
            _demo.main()

        assert exc_info.value.code == 1
        assert exc_info.value.__suppress_context__ is True
        assert message in capsys.readouterr().out


class TestWarnIfSceneIsStale:
    """The stale-scene check must inspect BOTH laddered layers.

    Fast synthetic zarr stores (no compiler) — this pins that a missing ladder
    on either 'By tracer type' or 'By redshift' is reported, per layer.
    """

    @staticmethod
    def _write_scene(path: Path, sublods_by_layer: dict[str, int]) -> None:
        import zarr

        root = zarr.open(str(path), mode="w")
        for layer, n_sublods in sublods_by_layer.items():
            finest = root.create_group(layer).create_group("child_3")
            finest.attrs["n_additive_sublods"] = n_sublods

    def test_silent_when_both_layers_laddered(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        self._write_scene(scene, {"By tracer type": 5, "By redshift": 5})
        _demo.warn_if_scene_is_stale(scene)
        assert "⚠" not in capsys.readouterr().out

    def test_warns_only_for_the_unladdered_layer(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        self._write_scene(scene, {"By tracer type": 5, "By redshift": 1})
        _demo.warn_if_scene_is_stale(scene)
        out = capsys.readouterr().out
        assert "'By redshift' finest level has no streaming" in out
        assert "'By tracer type'" not in out

    def test_missing_layer_is_reported_and_others_still_checked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        self._write_scene(scene, {"By tracer type": 1})
        _demo.warn_if_scene_is_stale(scene)
        out = capsys.readouterr().out
        assert "'By tracer type' finest level has no streaming" in out
        assert "Could not inspect" in out and "[By redshift]" in out

    def test_flat_layer_is_read_directly_not_reported_uninspectable(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A scene built without torch/scipy has no LOD children.

        substitutive_lod_or_flat writes the layer as a flat Points leaf then, so
        the layer itself is the finest level and still carries a ladder. Reading
        child_N unconditionally reported that scene as uninspectable — a
        misleading diagnostic on a machine that is fine.
        """
        import zarr

        scene = tmp_path / "desi.luxar.zarr"
        root = zarr.open(str(scene), mode="w")
        root.create_group("By tracer type").attrs["n_additive_sublods"] = 5
        root.create_group("By redshift").attrs["n_additive_sublods"] = 1

        _demo.warn_if_scene_is_stale(scene)
        out = capsys.readouterr().out
        assert "Could not inspect" not in out
        assert "'By redshift' finest level has no streaming" in out
        assert "'By tracer type'" not in out

    def test_finest_level_found_regardless_of_level_count(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A ladder built with a different LOD["levels"] names its finest child
        # something other than child_3; the check must still find it (children
        # are stored coarsest→finest, so the finest is the highest-numbered).
        # child_9 vs child_10 pins the NUMERIC suffix order: a lexicographic
        # sort would pick child_9 (laddered) and miss the warning.
        import zarr

        scene = tmp_path / "desi.luxar.zarr"
        root = zarr.open(str(scene), mode="w")
        for layer in ("By tracer type", "By redshift"):
            group = root.create_group(layer)
            group.create_group("child_0").attrs["n_additive_sublods"] = 5
            group.create_group("child_9").attrs["n_additive_sublods"] = 5
            group.create_group("child_10").attrs["n_additive_sublods"] = 1
        _demo.warn_if_scene_is_stale(scene)
        out = capsys.readouterr().out
        assert "'By tracer type' finest level has no streaming" in out
        assert "'By redshift' finest level has no streaming" in out

    @staticmethod
    def _write_laddered(scene: Path, increments: list[int]) -> None:
        """A scene whose finest level commits `increments` per additive rung."""
        import zarr

        root = zarr.open(str(scene), mode="w")
        for layer_name in ("By tracer type", "By redshift"):
            finest = root.create_group(layer_name).create_group("child_3")
            finest.attrs["n_additive_sublods"] = len(increments)
            finest.attrs["n_points"] = sum(increments)
            for i, inc in enumerate(increments):
                finest.create_group(f"additive_{i}").attrs["n_points"] = inc

    def test_warns_when_one_rung_commits_too_much(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The old geometric ladder: five rungs, last one n/2.

        This is the shape the 1.25M row cap was papering over — the level total
        is not the problem, the final increment is.
        """
        scene = tmp_path / "desi.luxar.zarr"
        self._write_laddered(scene, [609_498, 609_498, 1_218_996, 2_437_992, 4_875_971])

        _demo.warn_if_scene_is_stale(scene)

        out = capsys.readouterr().out
        assert out.count("commits 4,875,971 points in one rung") == 2
        assert "rm -rf" in out

    def test_silent_when_every_rung_is_within_the_ceiling(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A bounded-tail ladder over the SAME 9.75M total must not warn.

        Pins that the check reads the increment and not the level total, which
        is the whole point of dropping the row cap.
        """
        scene = tmp_path / "desi.luxar.zarr"
        geometric = [2_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000]
        tail = [_demo.SCENE_MAX_COMMIT] * 10
        increments = geometric + tail + [9_751_955 - sum(geometric) - sum(tail)]
        assert sum(increments) == 9_751_955
        self._write_laddered(scene, increments)

        _demo.warn_if_scene_is_stale(scene)

        assert "in one rung" not in capsys.readouterr().out

    def test_silent_for_partitioned_finest_level(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import zarr

        scene = tmp_path / "desi.luxar.zarr"
        root = zarr.open(str(scene), mode="w")
        increments = [900_000, 900_000, 637_989]
        for layer_name in ("By tracer type", "By redshift"):
            finest = root.create_group(layer_name).create_group("child_2")
            finest.attrs["kind"] = "partition"
            for part_index in range(4):
                part = finest.create_group(f"part_{part_index}")
                part.attrs["n_additive_sublods"] = len(increments)
                part.attrs["n_points"] = sum(increments)
                for rung_index, count in enumerate(increments):
                    part.create_group(f"additive_{rung_index}").attrs["n_points"] = (
                        count
                    )

        _demo.warn_if_scene_is_stale(scene)

        assert "⚠" not in capsys.readouterr().out

    def test_warns_when_partitioned_rung_commits_too_much(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import zarr

        scene = tmp_path / "desi.luxar.zarr"
        root = zarr.open(str(scene), mode="w")
        for layer_name in ("By tracer type", "By redshift"):
            finest = root.create_group(layer_name).create_group("child_2")
            finest.attrs["kind"] = "partition"
            part = finest.create_group("part_0")
            increments = [1_000_000, 3_875_978]
            part.attrs["n_additive_sublods"] = len(increments)
            part.attrs["n_points"] = sum(increments)
            for rung_index, count in enumerate(increments):
                part.create_group(f"additive_{rung_index}").attrs["n_points"] = count

        _demo.warn_if_scene_is_stale(scene)

        out = capsys.readouterr().out
        assert out.count("commits 3,875,978 points in one rung") == 2
        assert "'By tracer type' finest level commits" in out
        assert "'By redshift' finest level commits" in out

    def test_warns_when_one_partition_leaf_exceeds_the_node_capacity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import zarr

        scene = tmp_path / "desi.luxar.zarr"
        root = zarr.open(str(scene), mode="w")
        for layer_name in ("By tracer type", "By redshift"):
            finest = root.create_group(layer_name).create_group("child_2")
            finest.attrs["kind"] = "partition"
            for part_index, n_points in enumerate((4_875_978, 4_875_977)):
                part = finest.create_group(f"part_{part_index}")
                increments = [900_000, 900_000, 900_000, 900_000, 900_000]
                increments.append(n_points - sum(increments))
                part.attrs["n_additive_sublods"] = len(increments)
                part.attrs["n_points"] = n_points
                for rung_index, count in enumerate(increments):
                    part.create_group(f"additive_{rung_index}").attrs["n_points"] = (
                        count
                    )

        _demo.warn_if_scene_is_stale(scene)

        out = capsys.readouterr().out
        assert out.count("single node contains 4,875,978 points") == 2
        assert "rm -rf" in out

    def test_warns_when_flat_finest_exceeds_the_node_capacity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        increments = [
            2_000,
            2_000,
            4_000,
            8_000,
            16_000,
            32_000,
            64_000,
            128_000,
            256_000,
            512_000,
            *([900_000] * 9),
            627_955,
        ]
        self._write_laddered(scene, increments)

        _demo.warn_if_scene_is_stale(scene)

        out = capsys.readouterr().out
        assert out.count("single node contains 9,751,955 points") == 2
        assert "commits" not in out

    def test_warns_when_a_bounded_ladder_contains_only_the_old_sample(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        self._write_laddered(
            scene,
            [
                2_000,
                2_000,
                4_000,
                8_000,
                16_000,
                32_000,
                64_000,
                128_000,
                900_000,
                94_000,
            ],
        )

        _demo.warn_if_scene_is_stale(scene)

        out = capsys.readouterr().out
        assert out.count("contains only 1,250,000 points") == 2
        assert "full ~9.75M-object DR1 catalog" in out


class TestStreamingBreakpoints:
    """The ladder shape is what makes the full catalog streamable.

    A pure geometric ladder doubles to `n`, so its last increment is always
    `n/2` — the property that made 9.75M look like it needed a row cap.
    """

    @staticmethod
    def _increments(cuts: list[int]) -> list[int]:
        return [cuts[0]] + [b - a for a, b in zip(cuts, cuts[1:])]

    def test_no_rung_exceeds_the_ceiling_at_full_catalog_size(self) -> None:
        inc = self._increments(_demo.streaming_breakpoints(9_751_955))
        assert max(inc) <= _demo.SCENE_MAX_COMMIT
        assert sum(inc) == 9_751_955

    def test_first_paint_stays_one_chunk(self) -> None:
        for n in (19_519, 156_249, 1_250_000, 9_751_955):
            cuts = _demo.streaming_breakpoints(n)
            assert cuts[0] == _demo.SCENE_FIRST_CHUNK, n

    def test_the_ceiling_binds_at_every_scale(self) -> None:
        for n in (300_000, 1_218_970, 1_250_000, 9_751_955, 40_000_000):
            inc = self._increments(_demo.streaming_breakpoints(n))
            assert max(inc) <= _demo.SCENE_MAX_COMMIT, (n, max(inc))
            assert sum(inc) == n
            assert _demo.streaming_breakpoints(n) == sorted(
                set(_demo.streaming_breakpoints(n))
            ), "cuts must be strictly increasing"

    def test_middle_level_does_not_end_in_a_majority_rung(self) -> None:
        increments = self._increments(_demo.streaming_breakpoints(1_218_970))
        assert max(increments) == 512_000
        assert max(increments) / sum(increments) < 0.5

    def test_a_level_smaller_than_the_first_chunk_is_one_rung(self) -> None:
        assert _demo.streaming_breakpoints(500) == [500]

    def test_reaches_its_coarser_sibling_early_in_the_payload(self) -> None:
        """The upgrade must not "wait until fully loaded".

        With compression_factor=8 and levels=2 the finest level's coarser
        sibling holds n/8, so the rung that first exceeds that is the point the
        swap becomes worthwhile. With the shipped row count and ladder settings,
        that crossing is 1,924,000 / 9,751,955 = 19.73% of the payload.
        """
        n = 9_751_955
        cuts = _demo.streaming_breakpoints(n)
        sibling = n // 8
        crossing = next(c for c in cuts if c > sibling)
        assert crossing / n == pytest.approx(0.1973, abs=0.0001)


class TestSceneRowBudget:
    def test_caps_both_layers_but_frames_the_full_catalog(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(7)
        positions = rng.normal(size=(500, 3)).astype(np.float32)
        positions[-1] = (0.0, 0.0, 1000.0)
        redshift = np.linspace(0.01, 3.0, len(positions), dtype=np.float32)
        tracer_ids = (np.arange(len(positions)) % 4).astype(np.uint8)
        monkeypatch.setattr(_demo, "SCENE_MAX_POINTS", 100)
        monkeypatch.setattr(_demo, "substitutive_lod_or_flat", lambda spec: None)

        out = tmp_path / "desi.luxar.zarr"
        _demo.create_scene(positions, redshift, tracer_ids, out)

        import zarr

        root = zarr.open(str(out), mode="r")
        assert root["By tracer type"].attrs["n_points"] == 100
        assert root["By redshift"].attrs["n_points"] == 100
        expected_intensity = _demo.SCENE_INTENSITY * len(positions) / 100
        assert root["By tracer type"].attrs["intensity"] == pytest.approx(
            expected_intensity
        )
        assert root["By redshift"].attrs["intensity"] == pytest.approx(
            expected_intensity
        )

        radial = np.linalg.norm(positions.astype(np.float64), axis=1)
        r95 = float(np.percentile(radial, 95))
        cam_dist = 0.75 * r95 / np.tan(np.radians(_demo.CINEMATIC_FOV_DEG) / 2.0)
        camera = root.attrs["viewer_config"]["camera"]
        assert "fov" not in camera
        assert float(camera["far"]) == pytest.approx(
            (cam_dist + float(radial.max())) * 1.5
        )

    def test_partitions_both_layers_and_deduplicates_positions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        rng = np.random.default_rng(8)
        positions = rng.normal(size=(100, 3)).astype(np.float32)
        redshift = np.linspace(0.01, 3.0, len(positions), dtype=np.float32)
        tracer_ids = (np.arange(len(positions)) % 4).astype(np.uint8)
        monkeypatch.setattr(_demo, "SCENE_MAX_POINTS_PER_NODE", 40, raising=False)
        monkeypatch.setattr(_demo, "substitutive_lod_or_flat", lambda spec: None)

        out = tmp_path / "desi_partitioned.luxar.zarr"
        _demo.create_scene(positions, redshift, tracer_ids, out)

        import zarr

        root = zarr.open(str(out), mode="r")

        def position_arrays(group):
            arrays = []
            if "positions" in group:
                arrays.append(group["positions"])
            for child_name in group.group_keys():
                arrays.extend(position_arrays(group[child_name]))
            return arrays

        expected_intensity = _demo.SCENE_INTENSITY
        tracer_layer = root["By tracer type"]
        redshift_layer = root["By redshift"]
        assert tracer_layer.attrs["kind"] == "partition"
        assert redshift_layer.attrs["kind"] == "partition"
        assert tracer_layer.attrs["intensity"] == pytest.approx(expected_intensity)
        assert redshift_layer.attrs["intensity"] == pytest.approx(expected_intensity)

        tracer_positions = position_arrays(tracer_layer)
        redshift_positions = position_arrays(redshift_layer)
        assert len(tracer_positions) == len(redshift_positions) > 1
        assert sum(array.shape[0] for array in tracer_positions) == len(positions)
        assert all(array.shape[0] <= 40 for array in tracer_positions)
        assert all(
            array.attrs["encoding"]["name"] != "array_ref" for array in tracer_positions
        )
        assert all(
            array.attrs["encoding"]["name"] == "array_ref"
            and array.attrs["encoding"]["target"].startswith("By tracer type/")
            for array in redshift_positions
        )

    def test_cached_record_scene_carries_the_full_catalog(self) -> None:
        import json
        import zipfile

        scene_zip = (
            Path.home() / ".cache" / "luxar" / _demo.DEMO_NAME / _demo.SCENE_ZIP_FILE
        )
        if not scene_zip.exists():
            pytest.skip("DESI record scene is not present in the local cache")

        with zipfile.ZipFile(scene_zip) as archive:
            names = set(archive.namelist())

            def read_attrs(path: str) -> dict:
                document = json.loads(archive.read(f"{path}/zarr.json").decode("utf-8"))
                return document.get("attributes", document)

            for layer_name in ("By tracer type", "By redshift"):
                layer_attrs = read_attrs(layer_name)
                child_names = sorted(
                    name.removeprefix(f"{layer_name}/").removesuffix("/zarr.json")
                    for name in names
                    if name.startswith(f"{layer_name}/child_")
                    and name.count("/") == 2
                    and name.endswith("/zarr.json")
                )
                assert child_names == ["child_0", "child_1", "child_2"]
                child_attrs = [
                    read_attrs(f"{layer_name}/{child_name}")
                    for child_name in child_names
                ]
                assert layer_attrs["selector"] == "screen-area"
                assert [attrs["coverage_fraction"] for attrs in child_attrs] == [
                    0.0,
                    0.5,
                    1.0,
                ]

                # The finest child partitions the WHOLE catalog under the
                # conservative per-node Points capacity: no row cap, and no
                # viewer-side tail clamp on a 4096-class GPU.
                finest_path = f"{layer_name}/{child_names[-1]}"
                finest = child_attrs[-1]
                assert finest["kind"] == "partition"
                assert finest["max_elements"] == _demo.SCENE_MAX_POINTS_PER_NODE
                part_names = sorted(
                    name.removeprefix(f"{finest_path}/").removesuffix("/zarr.json")
                    for name in names
                    if name.startswith(f"{finest_path}/part_")
                    and name.count("/") == 3
                    and name.endswith("/zarr.json")
                )
                assert part_names == ["part_0", "part_1", "part_2", "part_3"]
                part_attrs = [
                    read_attrs(f"{finest_path}/{part_name}") for part_name in part_names
                ]
                assert all(
                    attrs["n_points"] <= _demo.SCENE_MAX_POINTS_PER_NODE
                    for attrs in part_attrs
                )
                n_finest = sum(attrs["n_points"] for attrs in part_attrs)
                assert n_finest == 9_751_955

                # And no single additive rung may exceed the commit ceiling —
                # the invariant that makes the full catalog streamable at all.
                increments = []
                position_encodings = []
                for part_name, attrs in zip(part_names, part_attrs):
                    part_path = f"{finest_path}/{part_name}"
                    part_increments = []
                    for index in range(attrs["n_additive_sublods"]):
                        rung_path = f"{part_path}/additive_{index}"
                        part_increments.append(read_attrs(rung_path)["n_points"])
                        position_encodings.append(
                            read_attrs(f"{rung_path}/positions")["encoding"]
                        )
                    assert sum(part_increments) == attrs["n_points"]
                    increments.extend(part_increments)
                assert sum(increments) == n_finest
                assert max(increments) <= _demo.SCENE_MAX_COMMIT, (
                    f"largest rung {max(increments):,} exceeds the "
                    f"{_demo.SCENE_MAX_COMMIT:,} ceiling"
                )
                if layer_name == "By tracer type":
                    assert all(
                        encoding["name"] != "array_ref"
                        for encoding in position_encodings
                    )
                else:
                    assert all(
                        encoding["name"] == "array_ref"
                        and encoding["target"].startswith("By tracer type/child_2/")
                        for encoding in position_encodings
                    )


class TestEnsureOriginFraming:
    """A reused scene must open on the observer, whatever its build framed on.

    `main()` prefers an existing `datasets/demos/` copy over the shipped asset,
    so a scene built before the pivot moved to the origin would otherwise keep
    its bounding-box camera forever. The ladder checks cannot see this: such a
    scene's geometry is perfectly current, only its framing is stale.
    """

    @staticmethod
    def _write_scene(path: Path, camera: dict | None) -> None:
        import zarr

        root = zarr.open(str(path), mode="w")
        root.attrs["viewer_config"] = {"camera": camera} if camera is not None else {}

    @staticmethod
    def _read_camera(path: Path) -> dict:
        import zarr

        root = zarr.open(str(path), mode="r")
        return dict(dict(root.attrs.get("viewer_config") or {}).get("camera") or {})

    def test_origin_targeted_scene_is_left_alone(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        self._write_scene(scene, {"position": [0.0, 0.0, 4000.0], "target": [0, 0, 0]})

        assert _demo.ensure_origin_framing(scene) is True
        assert self._read_camera(scene)["target"] == [0, 0, 0]
        assert capsys.readouterr().out.strip() == ""

    def test_bounding_box_pivot_is_repinned_to_the_origin(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The exact shape of the pre-fix scene: a pivot ~1.2 Gpc down +z."""
        scene = tmp_path / "desi.luxar.zarr"
        self._write_scene(
            scene,
            {
                "position": [-34.97, -114.41, 7699.01],
                "target": [-34.97, -114.41, 1159.37],
                "fov": 50.0,
                "near": 32.7,
                "far": 217921.16,
            },
        )

        assert _demo.ensure_origin_framing(scene) is False

        camera = self._read_camera(scene)
        assert camera["target"] == [0.0, 0.0, 0.0]
        # Only the pivot moves; the rest of the authored camera is untouched.
        assert camera["position"] == [-34.97, -114.41, 7699.01]
        assert camera["fov"] == 50.0
        assert camera["near"] == 32.7
        assert camera["far"] == 217921.16
        assert "Re-pinned" in capsys.readouterr().out

    def test_repinning_preserves_the_consolidated_scene_index(
        self, tmp_path: Path
    ) -> None:
        scene = tmp_path / "desi.luxar.zarr"
        root = open_group(scene, mode="w")
        root.attrs["viewer_config"] = {
            "camera": {"position": [0.0, 0.0, 4000.0], "target": [0.0, 0.0, 1200.0]}
        }
        layer = root.create_group("By tracer type")
        create_array(layer, "positions", data=np.zeros((2, 3), dtype=np.float32))
        consolidate(root)
        before = set(open_group(scene, mode="r").group_keys())

        assert _demo.ensure_origin_framing(scene) is False

        reopened = open_group(scene, mode="r")
        assert set(reopened.group_keys()) == before == {"By tracer type"}
        assert reopened["By tracer type"]["positions"].shape == (2, 3)

    def test_a_scene_with_no_camera_is_reported_not_invented(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Without the catalog there is no distance to derive, so say so."""
        scene = tmp_path / "desi.luxar.zarr"
        self._write_scene(scene, None)

        assert _demo.ensure_origin_framing(scene) is False

        out = capsys.readouterr().out
        assert "auto-frame" in out and "--recompute" in out
        assert self._read_camera(scene) == {}

    def test_a_pivot_inside_the_tolerance_is_not_rewritten(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Float noise around the origin is not a bounding-box centre."""
        scene = tmp_path / "desi.luxar.zarr"
        target = [1e-7, -2e-7, 3e-7]
        self._write_scene(scene, {"position": [0.0, 0.0, 4000.0], "target": target})

        assert _demo.ensure_origin_framing(scene) is True
        assert self._read_camera(scene)["target"] == target
        assert capsys.readouterr().out.strip() == ""


class TestMainSceneReuse:
    def test_cold_scene_uses_the_manifest_resolved_archive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        resolved = tmp_path / "resolved-scene.zip"
        resolved.write_bytes(b"manifest-resolved")
        calls: list[tuple[str, object]] = []

        monkeypatch.setattr(_demo, "SERVE_ONLY", False)
        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        monkeypatch.setattr(_demo, "NO_SERVE", True)
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(
            _demo,
            "ensure_dataset",
            lambda name: calls.append(("ensure", name)) or [resolved],
        )
        monkeypatch.setattr(
            _demo,
            "extract_shipped_scene",
            lambda source, output: calls.append(("extract", (source, output))),
        )
        monkeypatch.setattr(
            _demo,
            "ensure_origin_framing",
            lambda path: calls.append(("frame", path)),
        )
        monkeypatch.setattr(
            _demo,
            "warn_if_scene_is_stale",
            lambda path: calls.append(("warn", path)),
        )
        monkeypatch.setattr(
            _demo,
            "_load_or_build_or_exit",
            lambda: pytest.fail("manifest archive should avoid the compute fallback"),
        )

        _demo.main()

        output = tmp_path / "desi_galaxies.luxar.zarr"
        assert calls == [
            ("ensure", _demo.DEMO_NAME),
            ("extract", (resolved, output)),
            ("frame", output),
            ("warn", output),
        ]

    def test_unavailable_manifest_scene_falls_back_to_catalog_build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from luxar.demos import DatasetUnavailable

        output = tmp_path / "desi_galaxies.luxar.zarr"
        arrays = (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.zeros(2, dtype=np.uint8),
        )
        calls: list[tuple[str, object]] = []

        monkeypatch.setattr(_demo, "SERVE_ONLY", False)
        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        monkeypatch.setattr(_demo, "NO_SERVE", True)
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(
            _demo,
            "ensure_dataset",
            lambda name: (_ for _ in ()).throw(
                DatasetUnavailable(f"{name} record unavailable")
            ),
        )
        monkeypatch.setattr(_demo, "_load_or_build_or_exit", lambda: arrays)
        monkeypatch.setattr(
            _demo,
            "create_scene",
            lambda positions, redshift, tracers, path: calls.append(
                ("build", (positions, redshift, tracers, path))
            ),
        )

        _demo.main()

        assert len(calls) == 1
        assert calls[0][0] == "build"
        positions, redshift, tracers, path = calls[0][1]
        assert positions is arrays[0]
        assert redshift is arrays[1]
        assert tracers is arrays[2]
        assert path == output
        assert "desi_galaxies record unavailable" in capsys.readouterr().out

    def test_manifest_integrity_fault_does_not_trigger_catalog_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_demo, "SERVE_ONLY", False)
        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(
            _demo,
            "ensure_dataset",
            lambda name: (_ for _ in ()).throw(RuntimeError(f"bad cache for {name}")),
        )
        monkeypatch.setattr(
            _demo,
            "_load_or_build_or_exit",
            lambda: pytest.fail("integrity faults must not route to catalog rebuild"),
        )

        with pytest.raises(RuntimeError, match="bad cache"):
            _demo.main()

    def test_missing_manifest_payload_fault_does_not_trigger_catalog_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(_demo, "SERVE_ONLY", False)
        monkeypatch.setattr(_demo, "RECOMPUTE", False)
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(
            _demo,
            "ensure_dataset",
            lambda name: (_ for _ in ()).throw(
                FileNotFoundError(f"bad in-repo payload for {name}")
            ),
        )
        monkeypatch.setattr(
            _demo,
            "_load_or_build_or_exit",
            lambda: pytest.fail("payload faults must not route to catalog rebuild"),
        )

        with pytest.raises(FileNotFoundError, match="bad in-repo payload"):
            _demo.main()

    def test_serve_only_checks_staleness_before_launch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        scene = tmp_path / "desi_galaxies.luxar.zarr"
        scene.mkdir()
        calls: list[tuple[str, Path]] = []
        monkeypatch.setattr(_demo, "SERVE_ONLY", True)
        monkeypatch.setattr(_demo, "get_demos_output_dir", lambda: tmp_path)
        monkeypatch.setattr(
            _demo, "ensure_origin_framing", lambda path: calls.append(("frame", path))
        )
        monkeypatch.setattr(
            _demo, "warn_if_scene_is_stale", lambda path: calls.append(("warn", path))
        )
        monkeypatch.setattr(
            _demo, "launch_viewer", lambda path: calls.append(("launch", path))
        )

        _demo.main()

        assert calls == [("frame", scene), ("warn", scene), ("launch", scene)]


@pytest.mark.slow
class TestOrbitCentre:
    """The camera must orbit the OBSERVER (the origin), not a bounding box.

    Every DESI sightline radiates from Earth, so the origin is both the natural
    pivot and the one point in this scene with physical meaning — it is where our
    solar system is. Framing the 2-98 percentile box instead put the pivot ~1.2
    Gpc down +z (the two caps are asymmetric in z), so dragging swung the whole
    local universe around a point out in the ELG shell.
    """

    @staticmethod
    def _two_caps(n: int = 4000, seed: int = 0) -> np.ndarray:
        """Positions with the DESI shape: radial, z-asymmetric, centred on us."""
        rng = np.random.default_rng(seed)
        d = rng.uniform(200.0, 6000.0, size=n)
        ra = rng.uniform(0.0, 2.0 * np.pi, size=n)
        # Two caps, deliberately lopsided in z so a bbox centre is NOT the origin.
        dec = np.where(
            rng.random(n) < 0.7, rng.uniform(0.4, 1.2, n), rng.uniform(-0.9, -0.3, n)
        )
        return np.column_stack(
            [
                d * np.cos(dec) * np.cos(ra),
                d * np.cos(dec) * np.sin(ra),
                d * np.sin(dec),
            ]
        ).astype(np.float32)

    def test_camera_targets_the_origin(self, tmp_path) -> None:
        positions = self._two_caps()
        # Guard the premise: a bbox centre really is offset from the observer, so
        # this test would fail against the old framing.
        lo, hi = np.percentile(positions, [2, 98], axis=0)
        assert abs(float(((lo + hi) / 2.0)[2])) > 100.0

        out = tmp_path / "desi.luxar.zarr"
        _demo.create_scene(
            positions,
            np.full(len(positions), 0.5, dtype=np.float32),
            np.zeros(len(positions), dtype=np.uint8),
            out,
        )

        import zarr

        cam = zarr.open(str(out), mode="r").attrs["viewer_config"]["camera"]
        assert tuple(cam["target"]) == (0.0, 0.0, 0.0)
        # Camera sits out along +z at the framing distance, looking back at us.
        assert cam["position"][0] == 0.0 and cam["position"][1] == 0.0
        assert cam["position"][2] > 0.0

    def test_camera_sits_outside_the_cloud_and_frames_it(self, tmp_path) -> None:
        positions = self._two_caps()
        radial = np.linalg.norm(positions.astype(np.float64), axis=1)
        out = tmp_path / "desi.luxar.zarr"
        _demo.create_scene(
            positions,
            np.full(len(positions), 0.5, dtype=np.float32),
            np.zeros(len(positions), dtype=np.uint8),
            out,
        )

        import zarr

        cam = zarr.open(str(out), mode="r").attrs["viewer_config"]["camera"]
        dist = float(cam["position"][2])
        # Outside the populated bulk, so orbiting does not start inside the cloud
        # (a camera inside the bbox makes the coverage selector degenerate).
        assert dist > float(np.percentile(radial, 95))
        # ...but still close enough to be immersive rather than a distant speck.
        assert dist < 2.0 * float(radial.max())
        # Far plane must clear the antipodal galaxy.
        assert float(cam["far"]) > dist + float(radial.max())
        assert 0.0 < float(cam["near"]) < dist * 0.05
