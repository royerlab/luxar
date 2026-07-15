"""Smoke tests for the pure helpers in demo_global_rivers_earth.

Exercises only the network-free / IO-free helpers (Fibonacci sphere, sphere
mapping, hypsometric palette, polyline decimation, LUT builder). The demo is
loaded by file path (see test_demo_ppi_flow_field for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_global_rivers_earth.py"
)


def _load_demo_module():
    name = "_luxar_demo_globe_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


class TestFibonacciSphere:
    def test_ranges_and_count(self) -> None:
        lon, lat = _demo.fibonacci_sphere(10000)
        assert lon.shape == (10000,) and lat.shape == (10000,)
        assert lon.min() >= -180.0 and lon.max() <= 180.0
        assert lat.min() >= -90.0 and lat.max() <= 90.0
        # roughly even hemisphere split (uniform sphere coverage)
        assert abs(float((lat > 0).mean()) - 0.5) < 0.05

    def test_single_point(self) -> None:
        lon, lat = _demo.fibonacci_sphere(1)
        assert lon.shape == (1,) and lat.shape == (1,)


class TestLonLatToXyz:
    def test_radius_and_inverse(self) -> None:
        lon = np.array([0, 90, -90, 180, 45.0])
        lat = np.array([0, 0, 45, -30, 89.0])
        p = _demo.lonlat_to_xyz(lon, lat, np.zeros(5))
        assert np.allclose(np.linalg.norm(p, axis=1), _demo.RADIUS, atol=1e-3)
        rlat = np.degrees(np.arcsin(np.clip(p[:, 1] / _demo.RADIUS, -1, 1)))
        assert np.allclose(rlat, lat, atol=1e-3)

    def test_relief_scales_radius(self) -> None:
        p = _demo.lonlat_to_xyz(np.array([0.0]), np.array([0.0]), np.array([0.1]))
        assert np.isclose(np.linalg.norm(p[0]), _demo.RADIUS * 1.1, atol=1e-3)

    def test_right_handed_not_mirrored(self) -> None:
        # East(+dlon) x North(+dlat) must point OUTWARD -> right-handed globe
        # (a mirror/left-handed mapping would point inward). Regression guard.
        eps = 1e-3
        base = _demo.lonlat_to_xyz(np.array([0.0]), np.array([0.0]), np.zeros(1))[0]
        east = _demo.lonlat_to_xyz(np.array([eps]), np.array([0.0]), np.zeros(1))[0] - base
        north = _demo.lonlat_to_xyz(np.array([0.0]), np.array([eps]), np.zeros(1))[0] - base
        assert float(np.dot(np.cross(east, north), base)) > 0



class TestHypsometricScalars:
    def test_break_monotonic_range(self) -> None:
        elev = np.array([-10000, -100, 0, 100, 8000.0], dtype=np.float32)
        s = _demo.hypsometric_scalars(elev)
        assert s.min() >= 0.0 and s.max() <= 1.0
        assert np.all(np.diff(s) >= 0)
        assert 0.20 < float(s[2]) < 0.24  # sea level at the ocean/land break

    def test_degenerate_inputs_no_crash(self) -> None:
        assert _demo.hypsometric_scalars(np.array([-9000, -10.0], dtype=np.float32)).max() <= 1.0
        assert _demo.hypsometric_scalars(np.array([10, 5000.0], dtype=np.float32)).min() >= 0.0
        z = _demo.hypsometric_scalars(np.zeros(3, dtype=np.float32))
        assert not np.isnan(z).any()


class TestDecimatePolyline:
    def test_reduces_and_keeps_endpoints(self) -> None:
        pts = np.column_stack([np.linspace(0, 1, 50), np.zeros(50)]).astype(np.float32)
        d = _demo.decimate_polyline(pts, 0.025)
        assert len(d) < len(pts)
        assert np.allclose(d[0], pts[0]) and np.allclose(d[-1], pts[-1])

    def test_short_polyline_passthrough(self) -> None:
        pts = np.array([[0, 0], [1, 1.0]], dtype=np.float32)
        assert len(_demo.decimate_polyline(pts, 0.025)) == 2

    def test_sub_epsilon_collapses_to_endpoints(self) -> None:
        pts = np.array([[0, 0], [1e-4, 0], [2e-4, 0.0]], dtype=np.float32)
        assert len(_demo.decimate_polyline(pts, 0.025)) == 2


class TestLutFrom:
    def test_shape_and_endpoints(self) -> None:
        lut = _demo._lut_from([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))])
        assert lut.shape == (256, 3) and lut.dtype == np.uint8
        assert lut[0].tolist() == [0, 0, 0]
        assert lut[-1].tolist() == [255, 255, 255]
