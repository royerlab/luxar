"""Tests for save and load functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats import GSplatData
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

    return {
        "centers": centers,
        "amplitudes": amplitudes,
        "cholesky_factors": cholesky_factors,
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
            assert root.attrs["format_version"] == "2.0"

            # Check splats group
            assert "splats" in root
            splats_group = root["splats"]
            assert splats_group.attrs["type"] == "gsplats"  # Required for SceneLoader
            assert root["splats/substitutive_0/additive_0"].attrs["n_splats"] == 100
            assert root["splats/substitutive_0/additive_0"].attrs["ndim"] == 3

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
            assert "colors" in root["splats/substitutive_0/additive_0"]
            assert root["splats/substitutive_0/additive_0"].attrs["has_colors"] is True

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
            root["splats"]

            assert (
                root["splats/substitutive_0/additive_0"].attrs["ordering"] == "morton"
            )
            assert "ordering_min" in root["splats/substitutive_0/additive_0"].attrs
            assert "ordering_max" in root["splats/substitutive_0/additive_0"].attrs
            assert (
                "ordering_bits_per_dim"
                in root["splats/substitutive_0/additive_0"].attrs
            )

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
                root["splats"]

                assert (
                    root["splats/substitutive_0/additive_0"].attrs["ordering"]
                    == "hilbert"
                )

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
            assert root["splats/substitutive_0/additive_0/centers"].dtype == np.float32

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
            enc = root["splats/substitutive_0/additive_0/centers"].attrs.get(
                "encoding", {}
            )
            assert (
                enc["name"] == "float16"
            )  # Should be float16 because we passed float16_allowed=True

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


class TestGSplatDataMethods:
    """Test save/load methods on GSplatData."""

    def test_result_save(self) -> None:
        """Test GSplatData.save()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Create result
            result = GSplatData(
                centers=splats["centers"],
                amplitudes=splats["amplitudes"],
                cholesky_factors=splats["cholesky_factors"],
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
        """Test GSplatData.load()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            # Save
            save_gsplats(path=path, **splats, ordering="none")

            # Load via classmethod
            result = GSplatData.load(path)

            # Check loaded
            assert result.centers.shape == (50, 3)
            assert isinstance(result, GSplatData)

    def test_result_roundtrip(self) -> None:
        """Test save/load via result methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Create and save (with ordering="none" to preserve order for comparison)
            original = GSplatData(**splats, stats={})
            original.save(path, ordering="none")

            # Load
            loaded = GSplatData.load(path)

            # Compare (AUTO mode may use quantization, so use tolerance)
            assert np.allclose(loaded.centers, original.centers, atol=0.01)
            assert np.allclose(loaded.amplitudes, original.amplitudes, atol=0.05)


class TestColorRoundtrip:
    """Test color save/load round-trip functionality."""

    def test_roundtrip_with_sdr_colors(self) -> None:
        """Test round-trip with SDR float32 colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Add SDR colors (float32)
            colors = np.random.rand(100, 3).astype(np.float32)

            # Save
            save_gsplats(
                path=path,
                **splats,
                colors=colors,
                color_mode="sdr",
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )

            # Load
            result = load_gsplats(path)

            # Verify colors are present and correct
            assert result.colors is not None
            assert result.colors.shape == (100, 3)
            assert np.allclose(result.colors, colors, atol=0.01)

    def test_roundtrip_with_uint8_colors(self) -> None:
        """Test round-trip with uint8 colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            # Add uint8 colors
            colors = np.random.randint(0, 256, size=(50, 3), dtype=np.uint8)

            # Save
            save_gsplats(
                path=path,
                **splats,
                colors=colors,
                ordering="none",
            )

            # Load
            result = load_gsplats(path)

            # Verify colors are present and correct
            assert result.colors is not None
            assert result.colors.shape == (50, 3)
            assert result.colors.dtype == np.uint8
            assert np.array_equal(result.colors, colors)

    def test_roundtrip_with_hdr_colors(self) -> None:
        """Test round-trip with HDR float32 colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(75)

            # Add HDR colors (values can exceed 1.0)
            colors = np.random.rand(75, 3).astype(np.float32) * 5.0

            # Save
            save_gsplats(
                path=path,
                **splats,
                colors=colors,
                color_mode="hdr",
                ordering="none",
                encoding_mode=EncodingMode.PRECISION,
            )

            # Load
            result = load_gsplats(path)

            # Verify colors are present and correct
            assert result.colors is not None
            assert result.colors.shape == (75, 3)
            assert np.allclose(result.colors, colors, atol=0.05)

    def test_roundtrip_without_colors(self) -> None:
        """Test round-trip without colors (colors should be None)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Save without colors
            save_gsplats(path=path, **splats, ordering="none")

            # Load
            result = load_gsplats(path)

            # Verify colors are None
            assert result.colors is None

    def test_result_method_roundtrip_with_colors(self) -> None:
        """Test GSplatData.save()/load() with colors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            # Add colors
            colors = np.random.rand(100, 3).astype(np.float32)

            # Create result with colors
            original = GSplatData(
                **splats,
                colors=colors,
                stats={"test": "value"},
            )

            # Save via method
            original.save(path, ordering="none", color_mode="sdr")

            # Load via classmethod
            loaded = GSplatData.load(path)

            # Verify colors roundtripped correctly
            assert loaded.colors is not None
            assert loaded.colors.shape == colors.shape
            assert np.allclose(loaded.colors, colors, atol=0.01)

            # Verify other data is intact
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


class TestCompression:
    """Test blosc compression and chunk sizing."""

    def test_default_compression_applied(self):
        """save_gsplats uses Blosc(zstd) compression by default."""
        from numcodecs import Blosc

        splats = create_test_splats_3d(100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, **splats)

            root = zarr.open(str(path), "r")
            centers = root["splats/substitutive_0/additive_0/centers"]
            assert isinstance(centers.compressor, Blosc)
            assert centers.compressor.cname == "zstd"

    def test_compression_disabled(self):
        """compressor=None disables compression."""
        splats = create_test_splats_3d(100)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, compressor=None, **splats)

            root = zarr.open(str(path), "r")
            assert root["splats/substitutive_0/additive_0/centers"].compressor is None

    def test_chunk_capping_small_array(self):
        """Chunk rows should not exceed n_splats."""
        splats = create_test_splats_3d(50)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            save_gsplats(path=path, **splats)

            root = zarr.open(str(path), "r")
            assert root["splats/substitutive_0/additive_0/centers"].chunks[0] <= 50
            assert root["splats/substitutive_0/additive_0/amplitudes"].chunks[0] <= 50
            assert (
                root["splats/substitutive_0/additive_0/cholesky_factors"].chunks[0]
                <= 50
            )

    def test_gsplatdata_save_default_compression(self):
        """GSplatData.save() uses Blosc compression by default."""
        from numcodecs import Blosc

        splats = create_test_splats_3d(100)
        g = GSplatData(**splats)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path)

            root = zarr.open(str(path), "r")
            assert isinstance(
                root["splats/substitutive_0/additive_0/centers"].compressor, Blosc
            )

    def test_multi_lod_compression(self):
        """Multi-LOD save applies compression to each LOD."""
        from numcodecs import Blosc

        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lods = [
            AdditiveSubLOD(
                centers=np.random.randn(n, 3).astype(np.float32),
                amplitudes=np.random.rand(n).astype(np.float32),
                cholesky_factors=np.random.randn(n, 6).astype(np.float32),
            )
            for n in [50, 80]
        ]
        g = GSplatData(additive_sublods=lods)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path)

            root = zarr.open(str(path), "r")
            # Both LODs should have compression
            assert isinstance(
                root["splats/substitutive_0/additive_0/centers"].compressor, Blosc
            )
            assert isinstance(
                root["splats/substitutive_0/additive_1/centers"].compressor, Blosc
            )
            # Chunks capped at LOD size
            assert root["splats/substitutive_0/additive_0/centers"].chunks[0] <= 50
            assert root["splats/substitutive_0/additive_1/centers"].chunks[0] <= 80

    def test_roundtrip_with_compression(self):
        """Compressed files load correctly."""
        rng = np.random.RandomState(42)
        splats = {
            "centers": rng.rand(200, 3).astype(np.float32) * 10,
            "amplitudes": rng.rand(200).astype(np.float32) * 2,
            "cholesky_factors": rng.rand(200, 6).astype(np.float32),
        }
        g = GSplatData(**splats)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)

            g2 = GSplatData.load(path)
            np.testing.assert_allclose(g.centers, g2.centers, atol=1e-6)
            np.testing.assert_allclose(g.amplitudes, g2.amplitudes, atol=1e-6)
            np.testing.assert_allclose(
                g.cholesky_factors, g2.cholesky_factors, atol=1e-6
            )


class TestTruncationRadiusRoundtrip:
    """Tests for truncation_radius save/load round-trip."""

    def test_default_truncation_radius(self):
        """Default truncation_radius (3.0) survives save/load."""
        splats = create_test_splats_3d(50)
        g = GSplatData(**splats)
        assert g.truncation_radius == 3.0

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.truncation_radius == 3.0

    def test_custom_truncation_radius_single_lod(self):
        """Non-default truncation_radius survives single-LOD save/load."""
        splats = create_test_splats_3d(50)
        g = GSplatData(**splats, truncation_radius=2.75)
        assert g.truncation_radius == 2.75

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.truncation_radius == 2.75

    def test_custom_truncation_radius_multi_lod(self):
        """Non-default truncation_radius survives multi-LOD save/load."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lods = [
            AdditiveSubLOD(
                centers=np.random.randn(n, 3).astype(np.float32),
                amplitudes=np.random.rand(n).astype(np.float32),
                cholesky_factors=np.random.randn(n, 6).astype(np.float32),
                truncation_radius=2.5,
            )
            for n in [30, 50]
        ]
        g = GSplatData(additive_sublods=lods)
        assert g.truncation_radius == 2.5

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.truncation_radius == 2.5
            assert g2.additive_sublods[0].truncation_radius == 2.5
            assert g2.additive_sublods[1].truncation_radius == 2.5

    def test_per_level_truncation_radius_roundtrip(self):
        """M4: differing per-level truncation_radius must not collapse to the
        finest level's value on load."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD

        lods = [
            AdditiveSubLOD(
                centers=np.random.randn(n, 3).astype(np.float32),
                amplitudes=np.random.rand(n).astype(np.float32),
                cholesky_factors=np.random.randn(n, 6).astype(np.float32),
                truncation_radius=tr,
            )
            for n, tr in [(30, 2.5), (50, 4.0)]
        ]
        g = GSplatData(additive_sublods=lods)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)
            assert g2.additive_sublods[0].truncation_radius == 2.5
            assert g2.additive_sublods[1].truncation_radius == 4.0

    def test_level_stats_read_without_include_stats(self):
        """L10: per-level stats (n_splats_total) are available on a default
        load, not only when include_stats=True."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel

        def _lod(n: int) -> AdditiveSubLOD:
            return AdditiveSubLOD(
                centers=np.random.randn(n, 3).astype(np.float32),
                amplitudes=np.random.rand(n).astype(np.float32),
                cholesky_factors=np.random.randn(n, 6).astype(np.float32),
            )

        levels = [
            SubstitutiveLevel(
                additive_sublods=[_lod(8)],
                compression_factor=1,
                stats={"n_splats_total": 8},
            ),
            SubstitutiveLevel(
                additive_sublods=[_lod(2)],
                compression_factor=4,
                level_index=1,
                stats={"n_splats_total": 2},
            ),
        ]
        g = GSplatData.from_substitutive_levels(levels)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            g2 = GSplatData.load(path)  # default: include_stats not set
            assert g2.substitutive_levels[1].stats.get("n_splats_total") == 2

    def test_truncation_radius_in_zarr_metadata(self):
        """truncation_radius is written to zarr splats group attrs."""
        splats = create_test_splats_3d(50)
        g = GSplatData(**splats, truncation_radius=2.0)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)
            root = zarr.open(str(path), "r")
            assert root["splats"].attrs["truncation_radius"] == 2.0

    def test_backward_compat_missing_truncation_radius(self):
        """Files without truncation_radius default to 3.0 on load."""
        splats = create_test_splats_3d(50)
        g = GSplatData(**splats)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gsplats.zarr"
            g.save(path, ordering="none", encoding_mode=EncodingMode.PRECISION)

            # Remove truncation_radius from zarr attrs to simulate old file
            root = zarr.open(str(path), "r+")
            attrs = dict(root["splats"].attrs)
            del attrs["truncation_radius"]
            root["splats"].attrs.put(attrs)

            g2 = GSplatData.load(path)
            assert g2.truncation_radius == 3.0
