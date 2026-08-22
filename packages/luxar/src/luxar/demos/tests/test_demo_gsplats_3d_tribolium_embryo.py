"""Smoke test for the Tribolium embryo demo's authored appearance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import GSplatData

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_tribolium_embryo.py"


def _load_demo_module():
    name = "_luxar_demo_tribolium_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DEMO_MODULE = _load_demo_module()
create_luxar_scene = _DEMO_MODULE.create_luxar_scene


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats, sufficient to build the scene offline."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


def test_scene_bakes_volumetric_appearance(tmp_path) -> None:
    out = create_luxar_scene(_tiny_gsplat_data(), tmp_path / "tribolium.luxar.zarr")
    node = zarr.open_group(str(out), mode="r")["tribolium_embryo"]
    attrs = dict(node.attrs)

    assert attrs["blending_mode"] == "volumetric"
    assert attrs["absorption"] == pytest.approx(3.13)
    assert attrs["opacity"] == pytest.approx(0.06)
    assert attrs["intensity"] == pytest.approx(1.0 / 1.085)


def test_fit_normalises_counts_and_floor_without_mutating_input(
    monkeypatch, tmp_path
) -> None:
    volume = np.array([[[675.0, 1675.0, 3675.0]]], dtype=np.float32)
    original = volume.copy()
    captured: dict[str, object] = {}

    class FakeResult:
        amplitudes = np.ones(1, dtype=np.float32)

        def save(self, path, **kwargs) -> None:
            captured["save_path"] = path
            captured["save_kwargs"] = kwargs

    def fake_fit_gaussian_splats(image, **kwargs):
        captured["image"] = image
        captured["fit_kwargs"] = kwargs
        return FakeResult()

    monkeypatch.setattr("luxar.gsplats.fit_gaussian_splats", fake_fit_gaussian_splats)
    monkeypatch.setattr(_DEMO_MODULE, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_DEMO_MODULE, "DEVICE", "cpu")

    _DEMO_MODULE.fit_tribolium(volume)

    fitted_image = captured["image"]
    fit_kwargs = captured["fit_kwargs"]
    assert isinstance(fitted_image, np.ndarray)
    assert isinstance(fit_kwargs, dict)
    assert fitted_image.dtype == np.float32
    assert fitted_image.max() == pytest.approx(1.0)
    assert fit_kwargs["floor"] == pytest.approx(
        _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS / float(volume.max())
    )
    np.testing.assert_array_equal(volume, original)


def test_roundtrip_comparison_uses_the_fit_output_scale() -> None:
    volume = np.array([[[675.0, 1675.0, 3675.0]]], dtype=np.float32)
    vmax = float(volume.max())
    recon = (volume - _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS) / vmax

    reference, scaled_recon = _DEMO_MODULE._prepare_roundtrip_comparison(
        volume, recon.copy()
    )

    expected = np.clip(
        (volume - _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS)
        / (vmax - _DEMO_MODULE.SPECIMEN_BACKGROUND_COUNTS),
        0.0,
        1.0,
    )
    np.testing.assert_allclose(reference, expected)
    np.testing.assert_allclose(scaled_recon, expected)
