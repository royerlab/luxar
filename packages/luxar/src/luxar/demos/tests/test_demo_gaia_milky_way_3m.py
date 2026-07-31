"""Scene-attribute regression tests for the Gaia Milky Way demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.demos.demo_gaia_milky_way_3m import load_and_convert_gaia_data


def _write_tiny_gaia_table(path: Path, n_stars: int = 16) -> None:
    """Write the five raw columns consumed by the demo converter."""
    root = zarr.open(str(path), mode="w")
    values = {
        "x_kpc": np.linspace(-2.0, 2.0, n_stars, dtype=np.float32),
        "y_kpc": np.linspace(-1.0, 1.0, n_stars, dtype=np.float32),
        "z_kpc": np.linspace(-0.2, 0.2, n_stars, dtype=np.float32),
        "phot_g_mean_mag": np.linspace(2.0, 20.0, n_stars, dtype=np.float32),
        "bp_rp": np.linspace(-0.5, 3.0, n_stars, dtype=np.float32),
    }
    for name, data in values.items():
        root.create_dataset(name, data=data, shape=data.shape, dtype=data.dtype)


def test_authored_nodes_keep_gaia_volumetric_appearance(tmp_path: Path) -> None:
    """The built scene pins the appearance settings introduced with containment."""
    raw = tmp_path / "gaia.zarr"
    scene_path = tmp_path / "gaia.luxar.zarr"
    _write_tiny_gaia_table(raw)

    assert load_and_convert_gaia_data(raw, scene_path) == 16

    scene = zarr.open(str(scene_path), mode="r")
    stars = dict(scene["Stars"].attrs)
    assert stars["blending_mode"] == "volumetric"
    assert stars["opacity"] == pytest.approx(1.0)
    assert stars["absorption"] == pytest.approx(1.3)
    assert stars["intensity"] == pytest.approx(0.075)

    for name in ("Sun", "Betelgeuse", "Rigel"):
        marker = dict(scene[name].attrs)
        assert marker["blending_mode"] == "volumetric"
        assert marker["opacity"] == pytest.approx(1.0)
        assert marker["absorption"] == pytest.approx(1.3)
