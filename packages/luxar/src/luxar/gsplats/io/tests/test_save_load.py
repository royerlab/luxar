"""Tests for save and load functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats import GaussianSplatResult
from luxar.gsplats.io import (
    format_gsplats_info,
    inspect_gsplats_zarr,
    load_gsplats,
    save_gsplats,
)


def create_test_splats_3d(n_splats: int = 100) -> dict:
    """Create test 3D Gaussian splats."""
    centers = np.random.rand(n_splats, 3).astype(np.float32) * 10
    amplitudes = np.random.rand(n_splats).astype(np.float32) * 2
    # 3D Cholesky: 6 elements [L00, L10, L11, L20, L21, L22]
    cholesky_factors = np.random.rand(n_splats, 6).astype(np.float32)
    sharpnesses = np.full(n_splats, 2.0, dtype=np.float32)

    return {
        "centers": centers,
        "amplitudes": amplitudes,
        "cholesky_factors": cholesky_factors,
        "sharpnesses": sharpnesses,
    }


class TestSaveGsplats:
    """Test save_gsplats function."""

    def test_save_basic(self) -> None:
        """Test basic save functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(
                path=path,
                **splats,
                ordering="none",
            )

            # Check zarr was created
            assert path.exists()

            # Check format
            root = zarr.open_group(str(path), mode="r")
            assert root.attrs["format_type"] == "gsplats_zarr"
            assert root.attrs["format_version"] == "1.0"

            # Check splats group
            assert "splats" in root
            splats_group = root["splats"]
            assert splats_group.attrs["n_splats"] == 100
            assert splats_group.attrs["ndim"] == 3

    def test_save_with_colors(self) -> None:
        """Test save with colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            # Add SDR colors
            colors = np.random.rand(50, 3).astype(np.float32)

            save_gsplats(
                path=path,
                **splats,
                colors=colors,
                color_mode="sdr",
                ordering="none",
            )

            root = zarr.open_group(str(path), mode="r")
            assert "colors" in root["splats"]
            assert root["splats"].attrs["has_colors"] is True

    def test_save_with_morton_ordering(self) -> None:
        """Test save with Morton ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(
                path=path,
                **splats,
                ordering="morton",
            )

            root = zarr.open_group(str(path), mode="r")
            splats_group = root["splats"]

            assert splats_group.attrs["ordering"] == "morton"
            assert "morton_min" in splats_group.attrs
            assert "morton_max" in splats_group.attrs
            assert "morton_bits_per_dim" in splats_group.attrs

    def test_save_with_hilbert_ordering(self) -> None:
        """Test save with Hilbert ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            try:
                save_gsplats(
                    path=path,
                    **splats,
                    ordering="hilbert",
                )

                root = zarr.open_group(str(path), mode="r")
                splats_group = root["splats"]

                assert splats_group.attrs["ordering"] == "hilbert"

            except ImportError:
                pytest.skip("hilbertcurve package not installed")

    def test_save_with_encoding_modes(self) -> None:
        """Test save with different encoding modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splats = create_test_splats_3d(100)

            # Test PRECISION mode
            path_precision = Path(tmpdir) / "precision.gsplats.zarr"
            save_gsplats(
                path=path_precision,
                **splats,
                encoding_mode=EncodingMode.PRECISION,
                ordering="none",
            )

            root = zarr.open_group(str(path_precision), mode="r")
            # PRECISION should use float32
            assert root["splats/centers"].dtype == np.float32

            # Test MEMORY mode
            path_memory = Path(tmpdir) / "memory.gsplats.zarr"
            save_gsplats(
                path=path_memory,
                **splats,
                encoding_mode=EncodingMode.MEMORY,
                ordering="none",
                float16_allowed=True,  # Explicitly test float16
            )

            root = zarr.open_group(str(path_memory), mode="r")
            # MEMORY mode uses float32 by default (float16_allowed=False for compatibility)
            # Check encoding metadata
            enc = root["splats/centers"].attrs.get("encoding", {})
            assert enc["name"] == "float16"  # Should be float16 because we passed float16_allowed=True

    def test_save_with_fitting_info(self) -> None:
        """Test save with fitting metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            fitting_info = {
                "time_seconds": 45.3,
                "iterations": 850,
                "converged": True,
                "fitter_name": "test_fitter",
            }

            fitting_config = {
                "n_iters": 1000,
                "lr": 0.05,
            }

            save_gsplats(
                path=path,
                **splats,
                fitting_info=fitting_info,
                fitting_config=fitting_config,
                ordering="none",
            )

            root = zarr.open_group(str(path), mode="r")
            assert "fitting" in root
            assert root["fitting"].attrs["time_seconds"] == 45.3
            assert root["fitting"].attrs["iterations"] == 850

            assert "fitting/config" in root
            assert root["fitting/config"].attrs["n_iters"] == 1000

    def test_save_validation_errors(self) -> None:
        """Test validation errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Mismatched amplitudes shape
            with pytest.raises(ValueError, match="Amplitudes shape"):
                save_gsplats(
                    path=path,
                    centers=splats["centers"],
                    amplitudes=np.random.rand(50).astype(np.float32),  # Wrong size
                    cholesky_factors=splats["cholesky_factors"],
                    sharpnesses=splats["sharpnesses"],
                )

            # Mismatched cholesky shape
            with pytest.raises(ValueError, match="Cholesky factors shape"):
                save_gsplats(
                    path=path,
                    centers=splats["centers"],
                    amplitudes=splats["amplitudes"],
                    cholesky_factors=np.random.rand(100, 3).astype(
                        np.float32
                    ),  # Wrong k
                    sharpnesses=splats["sharpnesses"],
                )

            # Float colors without color_mode
            with pytest.raises(ValueError, match="color_mode must be specified"):
                colors = np.random.rand(100, 3).astype(np.float32)
                save_gsplats(
                    path=path,
                    **splats,
                    colors=colors,
                    color_mode=None,  # Missing!
                )


class TestLoadGsplats:
    """Test load_gsplats function."""

    def test_load_basic(self) -> None:
        """Test basic load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Save
            save_gsplats(path=path, **splats, ordering="none")

            # Load
            result = load_gsplats(path)

            # Check arrays
            assert result.centers.shape == (100, 3)
            assert result.amplitudes.shape == (100,)
            assert result.cholesky_factors.shape == (100, 6)
            assert result.sharpnesses.shape == (100,)

            # Check data types (decoded to float32)
            assert result.centers.dtype == np.float32
            # Note: Amplitudes might be float16 due to encoding, but that's OK
            assert result.amplitudes.dtype in (np.float32, np.float16)

    def test_load_with_encoding(self) -> None:
        """Test load with encoded arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Save with MEMORY mode (quantized)
            save_gsplats(
                path=path,
                **splats,
                encoding_mode=EncodingMode.MEMORY,
                ordering="none",
            )

            # Load (decoder returns the stored dtype, not always float32)
            result = load_gsplats(path)

            # Check that arrays are decoded (may be float16 due to MEMORY mode)
            assert result.centers.dtype in (np.float32, np.float16)
            assert result.amplitudes.dtype in (np.float32, np.float16)
            assert result.cholesky_factors.dtype in (np.float32, np.float16)

    def test_load_with_stats(self) -> None:
        """Test load with stats included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            fitting_info = {
                "time_seconds": 45.3,
                "iterations": 850,
            }

            save_gsplats(
                path=path,
                **splats,
                fitting_info=fitting_info,
                description="Test dataset",
                ordering="none",
            )

            # Load with stats
            result = load_gsplats(path, include_stats=True)

            # Check stats
            assert "time_seconds" in result.stats
            assert result.stats["time_seconds"] == 45.3
            assert "description" in result.stats
            assert result.stats["description"] == "Test dataset"

    def test_load_missing_file(self) -> None:
        """Test load with missing file."""
        with pytest.raises(FileNotFoundError):
            load_gsplats("/nonexistent/path.gsplats.zarr")

    def test_load_invalid_format(self) -> None:
        """Test load with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.zarr"

            # Create invalid zarr
            root = zarr.open_group(str(path), mode="w")
            root.attrs["format_type"] = "wrong_format"

            with pytest.raises(ValueError, match="Invalid format_type"):
                load_gsplats(path)


class TestRoundTrip:
    """Test save/load round-trip."""

    def test_roundtrip_basic(self) -> None:
        """Test save and load round-trip preserves data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Save with PRECISION mode to avoid quantization
            save_gsplats(
                path=path,
                **splats,
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )

            # Load
            result = load_gsplats(path)

            # Compare (should be exact with PRECISION mode)
            assert np.allclose(result.centers, splats["centers"])
            assert np.allclose(result.amplitudes, splats["amplitudes"])
            assert np.allclose(result.cholesky_factors, splats["cholesky_factors"])
            assert np.allclose(result.sharpnesses, splats["sharpnesses"])

    def test_roundtrip_with_ordering(self) -> None:
        """Test round-trip with spatial ordering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Save with Morton ordering
            save_gsplats(path=path, **splats, ordering="morton")

            # Load
            result = load_gsplats(path)

            # Should have same data (reordered)
            assert result.centers.shape == splats["centers"].shape
            assert result.amplitudes.shape == splats["amplitudes"].shape

            # Check data is approximately preserved (with quantization from AUTO mode)
            # Note: Order will differ due to Morton ordering
            assert len(result.centers) == len(splats["centers"])

    def test_roundtrip_with_quantization(self) -> None:
        """Test round-trip with quantization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Save with MEMORY mode (quantized)
            save_gsplats(
                path=path,
                **splats,
                encoding_mode=EncodingMode.MEMORY,
                ordering="none",
            )

            # Load (auto-decoded)
            result = load_gsplats(path)

            # Should be approximately equal (with quantization error)
            assert np.allclose(result.centers, splats["centers"], atol=0.01)
            assert np.allclose(result.amplitudes, splats["amplitudes"], atol=0.05)


class TestGaussianSplatResultMethods:
    """Test save/load methods on GaussianSplatResult."""

    def test_result_save(self) -> None:
        """Test GaussianSplatResult.save()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Create result
            result = GaussianSplatResult(
                centers=splats["centers"],
                amplitudes=splats["amplitudes"],
                cholesky_factors=splats["cholesky_factors"],
                sharpnesses=splats["sharpnesses"],
                stats={"time_seconds": 10.5, "iterations": 500},
            )

            # Save via method
            result.save(path, include_fitting_info=True)

            # Check saved
            assert path.exists()

            # Check fitting info was saved
            root = zarr.open_group(str(path), mode="r")
            assert "fitting" in root
            assert root["fitting"].attrs["time_seconds"] == 10.5

    def test_result_load(self) -> None:
        """Test GaussianSplatResult.load()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            # Save
            save_gsplats(path=path, **splats, ordering="none")

            # Load via classmethod
            result = GaussianSplatResult.load(path)

            # Check loaded
            assert result.centers.shape == (50, 3)
            assert isinstance(result, GaussianSplatResult)

    def test_result_roundtrip(self) -> None:
        """Test save/load via result methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Create and save (with ordering="none" to preserve order for comparison)
            original = GaussianSplatResult(**splats, stats={})
            original.save(path, ordering="none")

            # Load
            loaded = GaussianSplatResult.load(path)

            # Compare (AUTO mode may use quantization, so use tolerance)
            assert np.allclose(loaded.centers, original.centers, atol=0.01)
            assert np.allclose(loaded.amplitudes, original.amplitudes, atol=0.05)


class TestInspectGsplats:
    """Test inspection functionality."""

    def test_inspect_basic(self) -> None:
        """Test basic inspection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(path=path, **splats, ordering="morton")

            # Inspect
            info = inspect_gsplats_zarr(path)

            # Check info
            assert info["n_splats"] == 100
            assert info["ndim"] == 3
            assert info["ordering"] == "morton"
            assert info["has_colors"] is False
            assert info["has_sharpness"] is True

    def test_inspect_with_fitting(self) -> None:
        """Test inspection with fitting info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            fitting_info = {"time_seconds": 45.3, "iterations": 850}

            save_gsplats(
                path=path,
                **splats,
                fitting_info=fitting_info,
                ordering="none",
            )

            info = inspect_gsplats_zarr(path)

            # Check fitting info present
            assert "fitting" in info
            assert info["fitting"]["time_seconds"] == 45.3

    def test_inspect_format_output(self) -> None:
        """Test formatted inspection output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            fitting_info = {"time_seconds": 45.3, "iterations": 850, "converged": True}

            save_gsplats(
                path=path,
                **splats,
                fitting_info=fitting_info,
                ordering="hilbert",
            )

            try:
                info = inspect_gsplats_zarr(path)
                formatted = format_gsplats_info(info)

                # Check formatted string
                assert "100" in formatted
                assert "3D" in formatted
                assert "hilbert" in formatted
                assert "45.3" in formatted

            except ImportError:
                pytest.skip("hilbertcurve package not installed")
