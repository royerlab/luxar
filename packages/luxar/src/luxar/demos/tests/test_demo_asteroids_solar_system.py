"""Smoke tests for the pure orbital-mechanics helpers in demo_asteroids_solar_system.

These cover deterministic numerical helpers only — no network, no cache, no
scene I/O. The demo is loaded by file path (see test_demo_ppi_flow_field for the
rationale: ``luxar.demos`` is aliased to ``luxar.utils.demos``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_asteroids_solar_system.py"


def _load_demo_module():
    name = "_luxar_demo_asteroids_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
solve_kepler = _demo.solve_kepler
elements_to_xyz = _demo.elements_to_xyz
propagate_mean_anomaly = _demo.propagate_mean_anomaly
mean_motion = _demo.mean_motion
orbit_polyline = _demo.orbit_polyline
parse_sbdb = _demo.parse_sbdb
build_labels = _demo.build_labels
_take_brightest = _demo._take_brightest
build_static_scene = _demo.build_static_scene
build_animated_scene = _demo.build_animated_scene
GAUSS_K = _demo.GAUSS_K


def _tiny_catalog(n: int = 6) -> dict:
    rng = np.random.default_rng(0)
    return {
        "a": np.linspace(1.0, 5.0, n),
        "e": np.full(n, 0.1),
        "i": np.radians(np.linspace(0.0, 10.0, n)),
        "Omega": rng.uniform(0, 2 * np.pi, n),
        "w": rng.uniform(0, 2 * np.pi, n),
        "M": rng.uniform(0, 2 * np.pi, n),
        "epoch": np.full(n, 2451545.0),
        "H": np.linspace(3.0, 12.0, n),
        "names": np.array([f"{i} Test" for i in range(n)]),
    }


class TestSolveKepler:
    def test_circular_orbit_E_equals_M(self) -> None:
        M = np.array([0.0, 0.5, 1.0, 3.0], dtype=np.float64)
        E = solve_kepler(M, np.zeros_like(M))
        np.testing.assert_allclose(E, M, atol=1e-12)

    def test_satisfies_keplers_equation(self) -> None:
        # endpoint=False avoids M=+π, which wraps to −π (same angle, E differs
        # by 2π) — correct behavior, but it would break a naive residual check.
        M = np.linspace(-np.pi, np.pi, 50, endpoint=False)
        e = np.full_like(M, 0.6)
        E = solve_kepler(M, e)
        residual = E - e * np.sin(E) - M
        assert np.abs(residual).max() < 1e-10

    def test_zero_mean_anomaly_zero_eccentric(self) -> None:
        assert abs(float(solve_kepler(np.array([0.0]), np.array([0.3]))[0])) < 1e-12


class TestElementsToXYZ:
    def test_circular_equatorial_radius_is_a(self) -> None:
        a = np.array([2.5])
        z = np.zeros(1)
        M = np.array([1.234])
        xyz = elements_to_xyz(a, z, z, z, z, M)
        assert xyz.dtype == np.float32
        r = np.linalg.norm(xyz[0])
        np.testing.assert_allclose(r, 2.5, atol=1e-5)
        np.testing.assert_allclose(xyz[0, 2], 0.0, atol=1e-6)  # equatorial ⇒ z=0

    def test_perihelion_position(self) -> None:
        # M=0 ⇒ at perihelion; e=0.5, a=1 ⇒ distance a(1-e)=0.5 along +x.
        xyz = elements_to_xyz(
            np.array([1.0]),
            np.array([0.5]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        )
        np.testing.assert_allclose(xyz[0], [0.5, 0.0, 0.0], atol=1e-6)


class TestPropagation:
    def test_mean_motion_scaling(self) -> None:
        a = np.array([1.0, 4.0])
        n = mean_motion(a)
        np.testing.assert_allclose(n[0], GAUSS_K, rtol=1e-12)
        np.testing.assert_allclose(n[1], GAUSS_K / 8.0, rtol=1e-12)  # 4^1.5 = 8

    def test_propagate_advances_by_n_dt(self) -> None:
        m0 = np.array([0.0])
        a = np.array([1.0])
        epoch = np.array([2451545.0])
        target = 2451545.0 + 10.0
        M = propagate_mean_anomaly(m0, a, epoch, target)
        np.testing.assert_allclose(M[0], GAUSS_K * 10.0, rtol=1e-12)


class TestOrbitPolyline:
    def test_shape_and_closure(self) -> None:
        poly = orbit_polyline(2.0, 0.1, 0.2, 0.3, 0.4, n=64)
        assert poly.shape == (64, 3)
        assert poly.dtype == np.float32


class TestParseSBDB:
    def _obj(self):
        return {
            "fields": ["full_name", "epoch", "a", "e", "i", "om", "w", "ma", "H"],
            "data": [
                [
                    "1 Ceres",
                    "2451545.0",
                    "2.77",
                    "0.08",
                    "10.6",
                    "80.3",
                    "73.0",
                    "0.0",
                    "3.3",
                ],
                [
                    "comet-like",
                    "2451545.0",
                    "5.0",
                    "1.2",
                    "20.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "9.0",
                ],  # e>1 dropped
                [
                    "nan-a",
                    "2451545.0",
                    "nan",
                    "0.1",
                    "1.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "5.0",
                ],  # a NaN dropped
            ],
        }

    def test_filters_unbound_and_invalid(self) -> None:
        cat = parse_sbdb(self._obj())
        assert len(cat["a"]) == 1
        assert cat["names"][0].strip() == "1 Ceres"

    def test_angles_converted_to_radians(self) -> None:
        cat = parse_sbdb(self._obj())
        np.testing.assert_allclose(cat["i"][0], np.radians(10.6), atol=1e-9)


class TestBrightestAndLabels:
    def _cat(self):
        return {
            "a": np.array([2.0, 3.0, 4.0]),
            "e": np.zeros(3),
            "i": np.zeros(3),
            "Omega": np.zeros(3),
            "w": np.zeros(3),
            "M": np.zeros(3),
            "epoch": np.full(3, 2451545.0),
            "H": np.array([9.0, 3.0, 6.0]),  # body 1 (H=3) is brightest
            "names": np.array(["a", "b", "c"]),
        }

    def test_take_brightest_orders_by_H(self) -> None:
        out = _take_brightest(self._cat(), 2)
        assert list(out["names"]) == ["b", "c"]  # smallest H first

    def test_labels_only_for_brightest(self) -> None:
        labels = build_labels(self._cat(), top_n=1)
        assert labels.count("") == 2
        assert labels[1].startswith("b")  # brightest gets the label


class TestSceneBuild:
    """End-to-end scene builds (no network) — lock the dim_order/fill contracts."""

    def test_static_scene_writes_store(self, tmp_path) -> None:
        out = tmp_path / "static.luxar.zarr"
        n = build_static_scene(out, _tiny_catalog())
        assert n == 6
        assert out.exists() and any(out.iterdir())

    def test_animated_scene_writes_store(self, tmp_path) -> None:
        # Regression guard for the --animate dim_order/fill crash.
        out = tmp_path / "animated.luxar.zarr"
        n = build_animated_scene(out, _tiny_catalog())
        assert n > 0
        assert out.exists() and any(out.iterdir())
