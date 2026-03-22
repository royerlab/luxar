"""Tests for progressive Gaussian splat fitting."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD


def _make_synthetic_volume(shape=(32, 32), seed=42):
    """Create a synthetic volume with blobs at multiple scales."""
    V = np.zeros(shape, dtype=np.float32)

    # Large blob
    center = np.array(shape) // 2
    for idx in np.ndindex(shape):
        dist = np.sqrt(sum((i - c) ** 2 for i, c in zip(idx, center)))
        V[idx] += np.exp(-(dist**2) / (2 * (shape[0] / 6) ** 2))

    # Small blob offset
    center2 = np.array(shape) // 4
    for idx in np.ndindex(shape):
        dist = np.sqrt(sum((i - c) ** 2 for i, c in zip(idx, center2)))
        V[idx] += 0.5 * np.exp(-(dist**2) / (2 * (shape[0] / 12) ** 2))

    return V


class TestProgressiveFitting:
    """Tests for fit_progressive_gaussian_splats."""

    def test_basic_progressive_fit_2d(self):
        """Basic progressive fitting on a small 2D volume."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            splats_per_pass=100,
            iters_per_pass=50,
            psnr_patience=0.1,
            verbose=False,
        )
        assert isinstance(result, GSplatData)
        assert result.n_splats > 0
        assert result.n_lods >= 1
        assert result.ndim == 2

    def test_multi_pass(self):
        """Verify multiple passes are executed."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=300,
            splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,  # Very low patience to force multiple passes
            verbose=False,
        )
        # Should have at least 2 passes (may stop by PSNR patience)
        assert result.n_lods >= 2
        # Total splats should be approximately splats_per_pass * n_lods
        assert result.n_splats > 0

    def test_max_splats_respected(self):
        """Verify max_splats limit is respected."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=150,
            splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,
            verbose=False,
        )
        assert result.n_splats <= 150

    def test_psnr_patience_stops_early(self):
        """Verify PSNR patience causes early stopping."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=10000,  # Very high limit
            splats_per_pass=200,
            iters_per_pass=100,
            psnr_patience=5.0,  # Very high patience = stop early
            verbose=False,
        )
        # Should stop after 2 passes since ΔPSNR < 5 dB
        assert result.n_lods <= 5  # At most a few passes
        assert result.stats.get("stop_reason") in (
            "psnr_patience",
            "residual_negligible",
        )

    def test_lod_structure(self):
        """Verify each pass produces a valid LOD."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,
            verbose=False,
        )
        for i in range(result.n_lods):
            lod = result.at_lod(i)
            assert isinstance(lod, GSplatLOD)
            assert lod.n_splats > 0
            assert lod.ndim == 2
            assert "pass_index" in lod.stats
            assert lod.stats["pass_index"] == i

    def test_cumulative_psnr_increases(self):
        """Verify PSNR (approximately) increases with each pass."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=300,
            splats_per_pass=100,
            iters_per_pass=50,
            psnr_patience=0.01,
            verbose=False,
        )
        if result.n_lods >= 2:
            psnrs = result.lod_psnrs()
            # Last PSNR should be better than first
            assert psnrs[-1] > psnrs[0]

    def test_callback_invoked(self):
        """Verify on_pass_complete callback is called."""
        V = _make_synthetic_volume(shape=(32, 32))
        callback_log: list[tuple[int, int, float]] = []

        def my_callback(pass_idx: int, lod_data: GSplatLOD, psnr: float) -> None:
            callback_log.append((pass_idx, lod_data.n_splats, psnr))

        result = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,
            on_pass_complete=my_callback,
            verbose=False,
        )
        assert len(callback_log) == result.n_lods
        assert callback_log[0][0] == 0  # First pass index is 0

    def test_stats(self):
        """Verify overall stats are populated."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,
            verbose=False,
        )
        assert result.stats["fitter_name"] == "progressive"
        assert result.stats["n_passes"] == result.n_lods
        assert result.stats["n_splats"] == result.n_splats
        assert "time_seconds" in result.stats
        assert "psnr_db" in result.stats
        assert "stop_reason" in result.stats

    def test_save_load_roundtrip(self):
        """Verify multi-LOD result can be saved and loaded."""
        V = _make_synthetic_volume(shape=(32, 32))
        result = fit_progressive_gaussian_splats(
            V,
            max_splats=200,
            splats_per_pass=100,
            iters_per_pass=30,
            psnr_patience=0.01,
            verbose=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            result.save(str(path), ordering="none")
            loaded = GSplatData.load(str(path), include_stats=True)

            assert loaded.n_lods == result.n_lods
            assert loaded.n_splats == result.n_splats
            for i in range(result.n_lods):
                np.testing.assert_allclose(
                    loaded.at_lod(i).centers,
                    result.at_lod(i).centers,
                    atol=1e-3,
                )
