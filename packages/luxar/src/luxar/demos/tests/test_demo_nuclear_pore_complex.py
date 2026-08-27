"""Guards for the nuclear-pore demo's burial-shading path."""

import numpy as np
import pytest

from luxar.demos import demo_nuclear_pore_complex as demo


def test_burial_shading_varies_without_changing_cpk_hue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(5)
    positions = rng.normal(size=(800, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= rng.random((800, 1)) ** (1.0 / 3.0)
    radii = np.full(len(positions), 0.12, dtype=np.float32)
    base = np.array([0.7, 0.35, 0.15], dtype=np.float32)
    colors = np.tile(base, (len(positions), 1))
    monkeypatch.setattr(demo, "AO_GRID_CELLS", 32)
    monkeypatch.setattr(demo, "AO_RADIUS_NM", 0.5)

    shaded = demo._apply_burial_shading(colors, positions, radii)
    scale = shaded / base[None, :]

    assert float(scale[:, 0].max()) == pytest.approx(1.0, abs=1e-6)
    assert float(scale[:, 0].min()) < 0.9
    np.testing.assert_allclose(scale[:, 1], scale[:, 0], atol=1e-6)
    np.testing.assert_allclose(scale[:, 2], scale[:, 0], atol=1e-6)
