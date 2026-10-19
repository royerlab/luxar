"""A whole fit on the native CPU kernels reaches the PyTorch path's quality."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")


def _blobs(shape: tuple[int, int, int], n_blobs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    vol = np.zeros(shape, np.float32)
    for _ in range(n_blobs):
        center = rng.uniform(0, 1, 3) * np.array(shape)
        sigma = rng.uniform(1.5, 3.5)
        d2 = sum((g - c) ** 2 for g, c in zip(grid, center, strict=True))
        vol += rng.uniform(0.3, 1.0) * np.exp(-d2 / (2 * sigma * sigma))
    vol += 0.1 + 0.02 * rng.standard_normal(shape).astype(np.float32)
    return vol


@pytest.mark.slow
def test_native_fit_matches_pytorch_fit_psnr(monkeypatch: pytest.MonkeyPatch) -> None:
    from luxar.gsplats import fit_gaussian_splats
    from luxar.gsplats.models.gsplats.cpu import backend as backend_mod

    vol = _blobs((24, 48, 48), n_blobs=60, seed=0)
    psnr = {}
    for choice in ("native", "torch"):
        monkeypatch.setenv(backend_mod.BACKEND_ENV, choice)
        result = fit_gaussian_splats(
            vol,
            seeds=600,
            n_iters=120,
            device="cpu",
            verbose=False,
            early_stop_patience=None,
        )
        psnr[choice] = float(result.stats["psnr_db"])
    assert psnr["native"] == pytest.approx(psnr["torch"], abs=0.2), psnr
