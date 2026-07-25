"""Smoke tests for pure helpers in demo_zebrahub_velocity_streamlines.

These tests cover deterministic numerical helpers that do not touch the network,
the cache, or matplotlib. Network-fetching code paths (``resolve_h5ad`` Drive
download, ``load_zebrahub``) are intentionally not exercised. All heavy deps are
imported lazily via ``luxar.demos.require_module``, so the module imports with
no extras installed. That shared gate is tested in ``test_demos_dependencies.py``.
"""

from __future__ import annotations

import numpy as np

# The ``luxar.demos`` package is now importable directly (the sys.modules alias
# that used to shadow it was removed).
from luxar.demos.demo_zebrahub_velocity_streamlines import (
    ZebrahubData,
    _array_hash,
    _select_seeds,
    _stabilize_3d,
)


class TestArrayHash:
    def test_deterministic_same_input(self) -> None:
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert _array_hash(a) == _array_hash(b)

    def test_changes_with_content(self) -> None:
        a = np.arange(12, dtype=np.float32)
        b = a.copy()
        b[0] += 0.5
        assert _array_hash(a) != _array_hash(b)

    def test_combines_multiple_arrays(self) -> None:
        a = np.arange(4, dtype=np.float32)
        b = np.arange(4, 8, dtype=np.float32)
        # Hashing (a, b) is not the same as hashing each individually.
        assert _array_hash(a, b) != _array_hash(a)
        assert _array_hash(a, b) != _array_hash(b)


class TestStabilize3D:
    def test_returns_3x3_rotation_and_keepdims_mean(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((50, 3)).astype(np.float32)
        rotation, mean = _stabilize_3d(coords)
        assert rotation.shape == (3, 3)
        # mean is keepdims-style: shape (1, 3)
        assert mean.shape == (1, 3)
        assert rotation.dtype == np.float32
        assert mean.dtype == np.float32

    def test_handles_few_points_with_identity(self) -> None:
        # With <3 points, stabilize falls back to identity rotation.
        coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        rotation, mean = _stabilize_3d(coords)
        np.testing.assert_allclose(rotation, np.eye(3, dtype=np.float32))
        np.testing.assert_allclose(mean.ravel(), [1.0, 2.0, 3.0])

    def test_centers_at_mean(self) -> None:
        coords = np.array(
            [[10.0, 20.0, 30.0], [12.0, 22.0, 32.0], [14.0, 24.0, 34.0]],
            dtype=np.float32,
        )
        _, mean = _stabilize_3d(coords)
        np.testing.assert_allclose(mean.ravel(), coords.mean(axis=0), atol=1e-5)


class TestSelectSeeds:
    def _make_data(self, n_cells: int, anatomy_codes: np.ndarray) -> "ZebrahubData":
        return ZebrahubData(
            positions=np.zeros((n_cells, 3), dtype=np.float32),
            velocities=np.zeros((n_cells, 3), dtype=np.float32),
            anatomy_codes=anatomy_codes,
            anatomy_categories=[
                f"class{i}" for i in range(int(anatomy_codes.max()) + 1)
            ],
            stage_codes=np.zeros(n_cells, dtype=np.int32),
            stage_categories=["s0"],
        )

    def test_n_seeds_none_returns_every_cell(self) -> None:
        data = self._make_data(50, np.zeros(50, dtype=np.int32))
        result = _select_seeds(data, n_seeds=None)
        assert result.shape == (50,)
        np.testing.assert_array_equal(np.sort(result), np.arange(50))

    def test_n_seeds_geq_total_returns_every_cell(self) -> None:
        data = self._make_data(20, np.zeros(20, dtype=np.int32))
        result = _select_seeds(data, n_seeds=100)
        assert result.shape == (20,)

    def test_subsamples_to_requested_count_or_less(self) -> None:
        rng = np.random.default_rng(1)
        codes = rng.integers(0, 4, size=200, dtype=np.int32)
        data = self._make_data(200, codes)
        result = _select_seeds(data, n_seeds=50)
        # Stratified subsampling: result should be roughly n_seeds and never more.
        assert result.size <= 200
        assert result.size > 0
        # All indices should be valid and unique.
        assert len(set(result.tolist())) == result.size
        assert result.min() >= 0
        assert result.max() < 200
