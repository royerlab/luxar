"""Tests for extend_to_all functionality in Scene.add_gsplats()."""

import warnings
from typing import Any, cast

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def create_test_cholesky(n_splats: int, ndim: int) -> np.ndarray:
    """Create valid Cholesky factors for testing.

    Args:
        n_splats: Number of splats
        ndim: Dimensionality

    Returns:
        Array of shape (n_splats, k) where k = ndim*(ndim+1)/2
    """
    k = ndim * (ndim + 1) // 2
    return np.random.rand(n_splats, k).astype(np.float32) * 0.1


class TestGSplatsExtendToAll:
    """Test extend_to_all parameter in Scene.add_gsplats()."""

    def test_no_extension_with_empty_list(self, tmp_path) -> None:
        """Test that extend_to_all=[] has no extension and no warning."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 10)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # All splats at Time=0 - would normally trigger warning
            centers = np.array([[0, 0, 0, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 4)

            # Using empty list explicitly silences the warning
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                scene.add_gsplats(
                    "gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                    extend_to_all=[],
                )

        # Check that extend_to_all is not in attributes
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" not in store["gsplats"].attrs

    def test_warning_when_none_and_candidates_detected(self, tmp_path) -> None:
        """Test that a warning is issued when extend_to_all=None and candidates exist."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 10)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # All splats at Time=0, but Time range is [0, 10]
            centers = np.array(
                [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 2, 0]], dtype=np.float32
            )
            cholesky = create_test_cholesky(3, 4)

            # Should warn about Time being a candidate
            with pytest.warns(UserWarning, match="Time"):
                scene.add_gsplats(
                    "gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                )  # extend_to_all=None (default)

        # Should still write without extension
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" not in store["gsplats"].attrs

    def test_no_warning_when_multiple_values_in_dimension(self, tmp_path) -> None:
        """Test no warning when dimension has multiple values (not a candidate)."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 10)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Splats at different Time values - not a candidate for extension
            centers = np.array(
                [[0, 0, 0, 0], [1, 1, 1, 5], [2, 2, 2, 10]], dtype=np.float32
            )
            cholesky = create_test_cholesky(3, 4)

            # Should NOT warn because Time has multiple values
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                scene.add_gsplats(
                    "gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                )  # extend_to_all=None

    def test_explicit_extend_to_all_list(self, tmp_path) -> None:
        """Test explicit extend_to_all list."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Splats only at Time=0, Channel=0 but extend to all times/channels
            centers = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 5)

            scene.add_gsplats(
                "extended_gsplats",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all=["Time", "Channel"],
            )

        # Check that extend_to_all is saved in attributes
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["extended_gsplats"].attrs
        assert store["extended_gsplats"].attrs["extend_to_all"] == ["Time", "Channel"]

    def test_extend_to_all_non_displayed(self, tmp_path) -> None:
        """Test extend_to_all='all' extends all non-displayed dimensions."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            centers = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 5)

            scene.add_gsplats(
                "extend_all",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all="all",
            )

        # Check that all non-displayed dimensions are extended
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["extend_all"].attrs
        # Should include both Time and Channel
        extend_dims = store["extend_all"].attrs["extend_to_all"]
        assert "Time" in extend_dims
        assert "Channel" in extend_dims

    def test_invalid_extend_to_all_value(self, tmp_path) -> None:
        """Test that invalid extend_to_all value raises error."""
        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())

            centers = np.array([[0, 0, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 3)

            with pytest.raises(ValueError, match="Invalid extend_to_all value"):
                scene.add_gsplats(
                    "gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                    extend_to_all=cast(Any, 123),  # Invalid type
                )

    def test_invalid_dimension_name(self, tmp_path) -> None:
        """Test that invalid dimension name raises error."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            centers = np.array([[1, 2, 3, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 4)

            with pytest.raises(ValueError, match="Unknown dimension.*InvalidDim"):
                scene.add_gsplats(
                    "gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                    extend_to_all=["InvalidDim"],
                )

    def test_extend_with_colors(self, tmp_path) -> None:
        """Test extension works with colors."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            centers = np.array([[1, 2, 3, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 4)
            colors = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)  # Orange

            scene.add_gsplats(
                "colored_extended",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                colors=colors,
                extend_to_all=["Time"],
            )

        # Check the data was written correctly
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["colored_extended"].attrs
        assert store["colored_extended"].attrs["extend_to_all"] == ["Time"]

        # Check data arrays (v3.1 splits Cholesky: 4D → diag=4, offdiag=10-4=6)
        assert store["colored_extended/centers"].shape == (1, 4)
        assert store["colored_extended/cholesky_factors_diag"].shape == (1, 4)
        assert store["colored_extended/cholesky_factors_offdiag"].shape == (1, 6)
        assert store["colored_extended/amplitudes"].shape == (1,)
        assert store["colored_extended/colors"].shape == (1, 3)

    def test_center_array_must_include_all_dimensions(self, tmp_path) -> None:
        """Test that center arrays must include ALL dimensions even when extending."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # This is correct - includes Time dimension even though extending
            centers_correct = np.array([[1, 2, 3, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 4)

            scene.add_gsplats(
                "correct",
                centers_correct,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all=["Time"],
            )

            # Center array should have 4 dimensions (X, Y, Z, Time)
            store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
            assert store["correct/centers"].shape == (1, 4)

    def test_no_warning_without_range(self, tmp_path) -> None:
        """Test no warning when dimension has no defined range."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False),  # No range defined
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # All at Time=0, but no range defined so no warning
            centers = np.array([[0, 0, 0, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 4)

            with warnings.catch_warnings():
                warnings.simplefilter("error")  # Turn warnings into errors
                scene.add_gsplats(
                    "gsplats",
                    centers,
                    amplitudes=1.0,
                    cholesky_factors=cholesky,
                )  # extend_to_all=None

    def test_add_gsplats_from_data_with_extend_to_all(self, tmp_path) -> None:
        """Test add_gsplats_from_data supports extend_to_all parameter."""
        from luxar.gsplats.gsplat_data import GSplatData

        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Create GSplatData
            centers = np.array([[1, 2, 3, 0]], dtype=np.float32)
            amplitudes = np.array([1.0], dtype=np.float32)
            cholesky = create_test_cholesky(1, 4)

            result = GSplatData(
                centers=centers,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=None,
                stats={},
            )

            scene.add_gsplats_from_data("from_data", result, extend_to_all=["Time"])

        # Check that extend_to_all was applied
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")
        assert "extend_to_all" in store["from_data"].attrs
        assert store["from_data"].attrs["extend_to_all"] == ["Time"]

    def test_multiple_gsplats_different_extensions(self, tmp_path) -> None:
        """Test multiple gsplats nodes with different extend_to_all settings."""
        dims = Dimensions(
            [
                Dimension("X", display=True),
                Dimension("Y", display=True),
                Dimension("Z", display=True),
                Dimension("Time", display=False, range=(0, 2)),
                Dimension("Channel", display=False, range=(0, 1)),
            ]
        )

        with LuxarZarrCompiler(tmp_path / "test.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=dims)

            centers = np.array([[1, 2, 3, 0, 0]], dtype=np.float32)
            cholesky = create_test_cholesky(1, 5)

            # Splats 1: Extend across Time only
            scene.add_gsplats(
                "time_extended",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all=["Time"],
            )

            # Splats 2: Extend across Channel only
            scene.add_gsplats(
                "channel_extended",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all=["Channel"],
            )

            # Splats 3: Extend across both
            scene.add_gsplats(
                "both_extended",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all=["Time", "Channel"],
            )

            # Splats 4: No extension
            scene.add_gsplats(
                "not_extended",
                centers,
                amplitudes=1.0,
                cholesky_factors=cholesky,
                extend_to_all=[],
            )

        # Verify each has correct extension settings
        store = zarr.open_group(tmp_path / "test.luxar.zarr", mode="r")

        assert store["time_extended"].attrs["extend_to_all"] == ["Time"]
        assert store["channel_extended"].attrs["extend_to_all"] == ["Channel"]
        assert store["both_extended"].attrs["extend_to_all"] == ["Time", "Channel"]
        assert "extend_to_all" not in store["not_extended"].attrs
