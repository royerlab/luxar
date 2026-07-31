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
        monkeypatch.setattr("luxar.utils.download.robust_download", _fail_download)

        with pytest.raises(_demo.DESICatalogDownloadError) as exc_info:
            _demo.download_catalogs()

        expected_url = f"{_demo.BASE_URL}/BGS_BRIGHT_NGC_clustering.dat.fits"
        assert requested == [expected_url]
        message = str(exc_info.value)
        assert expected_url in message
        assert "HTTP 404 Not Found" in message
        assert "data.desi.lbl.gov" in message
        assert "git lfs pull" in message
        assert "without --recompute" in message
        assert isinstance(exc_info.value.__cause__, requests.HTTPError)

    def test_connection_failure_becomes_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import requests

        def _fail_download(url: str, output_path: Path, **kwargs: object) -> Path:
            raise requests.ConnectionError("network unreachable")

        monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
        monkeypatch.setattr("luxar.utils.download.robust_download", _fail_download)

        with pytest.raises(_demo.DESICatalogDownloadError) as exc_info:
            _demo.download_catalogs()

        message = str(exc_info.value)
        assert "ConnectionError: network unreachable" in message
        assert f"{_demo.BASE_URL}/BGS_BRIGHT_NGC_clustering.dat.fits" in message
        assert "git lfs pull" in message
        assert isinstance(exc_info.value.__cause__, requests.ConnectionError)

    def test_non_request_failure_is_not_hidden(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fail_download(url: str, output_path: Path, **kwargs: object) -> Path:
            raise ValueError("catalog size mismatch")

        monkeypatch.setattr(_demo, "CACHE_DIR", tmp_path)
        monkeypatch.setattr("luxar.utils.download.robust_download", _fail_download)

        with pytest.raises(ValueError, match="catalog size mismatch"):
            _demo.download_catalogs()

    def test_main_prints_download_guidance_and_exits_cleanly(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        message = "DESI host unavailable; run git lfs pull"

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
        from luxar.io.compiler import LuxarZarrCompiler  # noqa: F401  (import cost)

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
