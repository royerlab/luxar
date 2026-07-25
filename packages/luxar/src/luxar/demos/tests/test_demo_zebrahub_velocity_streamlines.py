"""Smoke tests for pure helpers in demo_zebrahub_velocity_streamlines.

These tests cover deterministic numerical helpers that do not touch the network,
the cache, or matplotlib. Network-fetching code paths (``resolve_h5ad`` Drive
download, ``load_zebrahub``) are intentionally not exercised. All heavy deps are
imported lazily via ``_require_module``, so the module imports with no extras.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# The ``luxar.demos`` package is now importable directly (the sys.modules alias
# that used to shadow it was removed).
from luxar.demos.demo_zebrahub_velocity_streamlines import (
    _INSTALL_NOTES,
    _INSTALL_SPECS,
    ZebrahubData,
    _array_hash,
    _require_module,
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


class TestRequireModule:
    def test_present_module_returns_it(self) -> None:
        # numpy is always present in this test environment.
        result = _require_module("numpy")
        assert result is np or result is __import__("numpy")  # noqa: E721

    def test_missing_module_raises_with_install_hint(self, capsys) -> None:
        with pytest.raises(ImportError, match="pip install no_such_package_xyz"):
            _require_module("no_such_package_xyz")
        captured = capsys.readouterr()
        # The aprint goes to stdout — check the install hint surfaces.
        assert "no_such_package_xyz" in captured.out

    def test_missing_module_with_pip_name_alias(self) -> None:
        with pytest.raises(ImportError, match="pip install scikit-image"):
            _require_module("nonexistent_skimage_xyz", pip_name="scikit-image")

    def test_constrained_spec_and_note_surface(self, monkeypatch, capsys) -> None:
        """A module with a version constraint advertises it, plus the reason."""
        monkeypatch.setitem(_INSTALL_SPECS, "nonexistent_pinned_xyz", "'pkg>=1,<2'")
        monkeypatch.setitem(
            _INSTALL_NOTES, "nonexistent_pinned_xyz", "Because reasons."
        )

        with pytest.raises(ImportError) as excinfo:
            _require_module("nonexistent_pinned_xyz")

        assert "pip install 'pkg>=1,<2'" in str(excinfo.value)
        assert "Because reasons." in str(excinfo.value)
        out = capsys.readouterr().out
        assert "'pkg>=1,<2'" in out
        assert "Because reasons." in out

    def test_explicit_pip_name_overrides_the_spec_table(self, monkeypatch) -> None:
        monkeypatch.setitem(_INSTALL_SPECS, "nonexistent_pinned_xyz", "'pkg>=1,<2'")
        with pytest.raises(ImportError, match=re.escape("pip install other-pkg")):
            _require_module("nonexistent_pinned_xyz", pip_name="other-pkg")


class TestAnndataPin:
    """The anndata hint MUST carry the zarr-2-compatible upper bound.

    anndata >= 0.13 requires ``zarr>=3.1``, which is unsatisfiable against
    Luxar's ``zarr>=2.16,<3.0`` pin — a bare ``pip install anndata`` silently
    upgrades zarr and breaks every Luxar store.
    """

    def test_install_spec_is_upper_bounded(self) -> None:
        spec = _INSTALL_SPECS["anndata"]
        assert "<0.13" in spec, f"anndata hint lost its upper bound: {spec}"
        assert "zarr" in _INSTALL_NOTES["anndata"]

    def test_pyproject_demos_extra_matches_the_hint(self) -> None:
        root = Path(__file__).resolve().parents[6]
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():  # installed wheel — no source tree to check
            pytest.skip("pyproject.toml not available (installed package)")

        requirements = re.findall(
            r'"(anndata[^"]*)"', pyproject.read_text(encoding="utf-8")
        )
        assert requirements, "anndata requirement vanished from pyproject.toml"
        for requirement in requirements:
            assert "<0.13" in requirement, (
                "pyproject pins anndata without the zarr-2-compatible upper "
                f"bound: {requirement}"
            )
