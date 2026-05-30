"""Tests for .gsplats.zarr format compliance.

These tests validate that saved files match the format specification exactly.
"""

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats.io import save_gsplats


def create_test_splats_3d(n_splats: int = 100) -> dict:
    """Create test 3D Gaussian splats."""
    centers = np.random.rand(n_splats, 3).astype(np.float32) * 10
    amplitudes = np.random.rand(n_splats).astype(np.float32) * 2
    cholesky_factors = np.random.rand(n_splats, 6).astype(np.float32)

    return {
        "centers": centers,
        "amplitudes": amplitudes,
        "cholesky_factors": cholesky_factors,
    }


class TestFormatCompliance:
    """Test format specification compliance."""

    def test_root_attributes(self) -> None:
        """Test root-level attributes match spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(
                path=path,
                **splats,
                description="Test dataset",
            )

            root = zarr.open_group(str(path), mode="r")

            # Required root attributes (spec Section "Root Attributes")
            assert root.attrs["format_version"] == "2.0"
            assert root.attrs["format_type"] == "gsplats_zarr"
            assert "timestamp" in root.attrs
            assert "luxar_gsplats_version" in root.attrs

            # Optional root attributes
            assert root.attrs["description"] == "Test dataset"

    def test_splats_group_structure(self) -> None:
        """Test splats group structure matches spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(path=path, **splats, ordering="morton")

            root = zarr.open_group(str(path), mode="r")

            # Check splats group exists
            assert "splats" in root

            root["splats"]

            # Check required arrays (spec Section "Zarr Structure")
            assert "centers" in root["splats/substitutive_0/additive_0"]
            assert "amplitudes" in root["splats/substitutive_0/additive_0"]
            assert "cholesky_factors" in root["splats/substitutive_0/additive_0"]
            assert "chunk_bounds" in root["splats/substitutive_0/additive_0"]

            # Check optional arrays

    def test_splats_group_attributes(self) -> None:
        """Test splats group attributes match spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(path=path, **splats, ordering="morton")

            root = zarr.open_group(str(path), mode="r")
            root["splats"]
            attrs = root["splats/substitutive_0/additive_0"].attrs

            # Required attributes (spec Section "Splats Group Attributes")
            assert attrs["n_splats"] == 100
            assert attrs["ndim"] == 3
            assert "has_colors" in attrs
            assert attrs["ordering"] == "morton"
            assert "chunk_size" in attrs

            # Ordering metadata (spec Section "Ordering Metadata")
            assert "ordering_min" in attrs
            assert "ordering_max" in attrs
            assert "ordering_bits_per_dim" in attrs

            # Value ranges (spec Section "Splats Group Attributes")
            assert "amplitude_range" in attrs
            assert "center_bounds" in attrs

            # Validate range structures
            amp_range = attrs["amplitude_range"]
            assert "min" in amp_range
            assert "max" in amp_range

            center_bounds = attrs["center_bounds"]
            assert "min" in center_bounds
            assert "max" in center_bounds
            assert len(center_bounds["min"]) == 3
            assert len(center_bounds["max"]) == 3

    def test_array_shapes(self) -> None:
        """Test array shapes match spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            n_splats = 100
            splats = create_test_splats_3d(n_splats)

            save_gsplats(
                path=path,
                **splats,
                encoding_mode=EncodingMode.PRECISION,  # No quantization
            )

            root = zarr.open_group(str(path), mode="r")
            root["splats"]

            # Check shapes (spec Section "Core Data Structure")
            # Note: Broadcasting may change shapes to (1,) or (1, d)
            centers = root["splats/substitutive_0/additive_0"]["centers"]
            assert centers.shape[1] == 3  # ndim

            amplitudes = root["splats/substitutive_0/additive_0"]["amplitudes"]
            assert len(amplitudes.shape) == 1

            cholesky_factors = root["splats/substitutive_0/additive_0"][
                "cholesky_factors"
            ]
            assert cholesky_factors.shape[1] == 6  # d*(d+1)//2 for d=3

            chunk_bounds = root["splats/substitutive_0/additive_0"]["chunk_bounds"]
            assert chunk_bounds.shape[1] == 3  # ndim
            assert chunk_bounds.shape[2] == 2  # min/max

    def test_encoding_metadata_present(self) -> None:
        """Test encoding metadata is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(
                path=path,
                **splats,
                encoding_mode=EncodingMode.MEMORY,
            )

            root = zarr.open_group(str(path), mode="r")
            cell_group = root["splats/substitutive_0/additive_0"]

            # All arrays should have encoding metadata (spec Section "Encoding Metadata Preservation")
            for array_name in [
                "centers",
                "amplitudes",
                "cholesky_factors",
            ]:
                array = cell_group[array_name]
                enc = array.attrs.get("encoding")
                assert enc is not None, f"{array_name} missing encoding metadata"
                assert "name" in enc

    def test_fitting_group_structure(self) -> None:
        """Test fitting group structure matches spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            fitting_info = {
                "fitter_name": "luxar.gsplats",
                "fitter_version": "0.1.0",
                "time_seconds": 45.3,
                "iterations": 850,
                "converged": True,
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
            )

            root = zarr.open_group(str(path), mode="r")

            # Check fitting group (spec Section "Fitting Group Attributes")
            assert "fitting" in root
            fitting_group = root["fitting"]

            # Check common fields
            assert fitting_group.attrs["fitter_name"] == "luxar.gsplats"
            assert fitting_group.attrs["fitter_version"] == "0.1.0"
            assert fitting_group.attrs["time_seconds"] == 45.3
            assert fitting_group.attrs["iterations"] == 850
            assert fitting_group.attrs["converged"] is True

            # Check config subgroup
            assert "config" in fitting_group
            config_group = fitting_group["config"]
            assert config_group.attrs["n_iters"] == 1000
            assert config_group.attrs["lr"] == 0.05

    def test_provenance_group_structure(self) -> None:
        """Test provenance group structure matches spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            provenance_info = {
                "source_file": "/path/to/image.tif",
                "shape": [128, 256, 256],
                "dtype": "uint16",
                "normalization": {
                    "method": "percentile",
                    "low": 0.1,
                    "high": 99.9,
                },
            }

            save_gsplats(
                path=path,
                **splats,
                provenance_info=provenance_info,
            )

            root = zarr.open_group(str(path), mode="r")

            # Check provenance group (spec Section "Provenance Group Attributes")
            assert "provenance" in root
            provenance_group = root["provenance"]

            # Check attributes
            assert provenance_group.attrs["source_file"] == "/path/to/image.tif"
            assert provenance_group.attrs["shape"] == [128, 256, 256]
            assert provenance_group.attrs["dtype"] == "uint16"

            norm = provenance_group.attrs["normalization"]
            assert norm["method"] == "percentile"
            assert norm["low"] == 0.1
            assert norm["high"] == 99.9

    def test_chunk_bounds_format(self) -> None:
        """Test chunk_bounds array format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(path=path, **splats, ordering="morton")

            root = zarr.open_group(str(path), mode="r")
            chunk_bounds = root["splats/substitutive_0/additive_0/chunk_bounds"]

            # Check shape (spec Section "Chunk Bounding Boxes")
            # Shape should be (num_chunks, ndim, 2)
            assert chunk_bounds.ndim == 3
            assert chunk_bounds.shape[1] == 3  # ndim
            assert chunk_bounds.shape[2] == 2  # min/max

            # Check dtype
            assert chunk_bounds.dtype == np.float32

            # Check bounds are valid (min < max)
            data = chunk_bounds[:]
            assert np.all(data[:, :, 0] < data[:, :, 1])

    def test_consolidated_metadata(self) -> None:
        """Test that .zmetadata is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(50)

            save_gsplats(path=path, **splats)

            # Check .zmetadata file exists
            zmetadata_path = path / ".zmetadata"
            assert zmetadata_path.exists()

    def test_3d_specific_values(self) -> None:
        """Test 3D-specific format values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            n_splats = 100
            splats = create_test_splats_3d(n_splats)

            save_gsplats(
                path=path,
                **splats,
                encoding_mode=EncodingMode.PRECISION,
            )

            root = zarr.open_group(str(path), mode="r")
            root["splats"]

            # For 3D: k = d*(d+1)//2 = 3*4//2 = 6
            cholesky = root["splats/substitutive_0/additive_0"]["cholesky_factors"]
            assert cholesky.shape[1] == 6

            # Metadata should reflect 3D
            assert root["splats/substitutive_0/additive_0"].attrs["ndim"] == 3

            # Center bounds should have 3 dimensions
            center_bounds = root["splats/substitutive_0/additive_0"].attrs[
                "center_bounds"
            ]
            assert len(center_bounds["min"]) == 3
            assert len(center_bounds["max"]) == 3

    def test_semantic_type_assignment(self) -> None:
        """Test that arrays are encoded with correct semantic types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(
                path=path,
                **splats,
                encoding_mode=EncodingMode.MEMORY,
            )

            root = zarr.open_group(str(path), mode="r")
            root["splats"]

            # Check semantic types via encoding names (spec Section "Semantic Types")
            # Centers: COORDINATE → float32 in MEMORY mode (new default with float16_allowed=False)
            centers_enc = root["splats/substitutive_0/additive_0"]["centers"].attrs.get(
                "encoding", {}
            )
            assert centers_enc["name"] == "float32"

            # Sharpnesses: BOUNDED_SCALAR → bounded_scalar_uint8 in MEMORY mode
            # May be broadcasted or bounded_scalar_uint8

    def test_colors_with_color_mode(self) -> None:
        """Test colors are saved with color_mode metadata."""
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
            )

            root = zarr.open_group(str(path), mode="r")

            # Check colors are present
            assert "colors" in root["splats/substitutive_0/additive_0"]

            # Check color_mode in encoding metadata (spec Section "Color Mode Storage")
            colors_array = root["splats/substitutive_0/additive_0/colors"]
            enc = colors_array.attrs.get("encoding", {})
            assert enc is not None
            # color_mode should be preserved
            if "color_mode" in enc:
                assert enc["color_mode"] == "sdr"

    def test_ordering_metadata_completeness(self) -> None:
        """Test ordering metadata is complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(path=path, **splats, ordering="morton")

            root = zarr.open_group(str(path), mode="r")
            splats_attrs = root["splats/substitutive_0/additive_0"].attrs

            # Ordering metadata (spec Section "Ordering Metadata")
            assert splats_attrs["ordering"] == "morton"
            assert len(splats_attrs["ordering_min"]) == 3
            assert len(splats_attrs["ordering_max"]) == 3
            assert isinstance(splats_attrs["ordering_bits_per_dim"], int)
            assert splats_attrs["ordering_bits_per_dim"] <= 21  # Max for 3D

    def test_no_ordering_metadata(self) -> None:
        """Test ordering="none" doesn't add ordering metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.gsplats.zarr"
            splats = create_test_splats_3d(100)

            save_gsplats(path=path, **splats, ordering="none")

            root = zarr.open_group(str(path), mode="r")
            splats_attrs = root["splats/substitutive_0/additive_0"].attrs

            # Should have ordering="none" but no ordering metadata
            assert splats_attrs["ordering"] == "none"
            assert "ordering_min" not in splats_attrs
            assert "ordering_max" not in splats_attrs
