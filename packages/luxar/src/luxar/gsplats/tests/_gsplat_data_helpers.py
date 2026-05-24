"""Shared helpers for GSplatData tests.

The GSplatData test suite is split across multiple themed files
(`test_gsplat_data.py`, `test_gsplat_data_lod.py`, …). These helpers
build minimal-but-valid `GSplatData` instances used as fixtures across
all of them.
"""

from __future__ import annotations

import numpy as np

from luxar.gsplats.gsplat_data import GSplatData


def _make_3d_gsplat(n: int = 5, seed: int = 42) -> GSplatData:
    """Helper: create a simple 3D GSplatData with n splats."""
    rng = np.random.RandomState(seed)
    return GSplatData(
        centers=rng.rand(n, 3).astype(np.float32) * 100,
        amplitudes=rng.rand(n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (n, 1)
        ),
    )


def _make_empty_gsplat(ndim: int = 3) -> GSplatData:
    """Helper: create an empty GSplatData."""
    tril = ndim * (ndim + 1) // 2
    return GSplatData(
        centers=np.zeros((0, ndim), dtype=np.float32),
        amplitudes=np.zeros(0, dtype=np.float32),
        cholesky_factors=np.zeros((0, tril), dtype=np.float32),
    )


def _make_2d_gsplat(n: int = 5, seed: int = 42) -> GSplatData:
    """Helper: create a 2D GSplatData."""
    rng = np.random.RandomState(seed)
    return GSplatData(
        centers=rng.rand(n, 2).astype(np.float32) * 100,
        amplitudes=rng.rand(n).astype(np.float32),
        cholesky_factors=np.tile(np.array([1, 0, 1], dtype=np.float32), (n, 1)),
    )
