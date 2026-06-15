"""Tests for scalar input support in ArrayEncoder.

Tests the v0.6.0 feature that allows passing scalars directly to encoder
without creating intermediate arrays.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.encoding import ArrayEncoder
from luxar.encoding.modes import EncodingMode
from luxar.encoding.semantic_types import SemanticType


class TestScalarInputBasics:
    """Test basic scalar input functionality."""

    def test_float_scalar_with_n_elements(self) -> None:
        """Test encoding a float scalar with n_elements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            # Encode scalar
            encoder.encode(
                data=0.5,
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                n_elements=1000,
            )

            # Verify storage
            radii = store["radii"]
            assert radii.shape == (1,)
            assert radii[:][0] == pytest.approx(0.5)

            # Verify metadata
            enc = radii.attrs["encoding"]
            assert enc["name"] == "broadcasted"
            assert enc["n_elements"] == 1000

    def test_int_scalar_with_n_elements(self) -> None:
        """Test encoding an int scalar (converted to float)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=5,  # Int
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                n_elements=500,
            )

            radii = store["radii"]
            assert radii.shape == (1,)
            assert radii[:][0] == pytest.approx(5.0)
            assert radii.attrs["encoding"]["n_elements"] == 500

    def test_color_tuple_with_n_elements(self) -> None:
        """Test encoding a color tuple (R, G, B)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=(1.0, 0.5, 0.0),  # RGB tuple
                zarr_group=store,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
                n_elements=2000,
            )

            colors = store["colors"]
            assert colors.shape == (1, 3)
            assert np.allclose(colors[:], [[1.0, 0.5, 0.0]])
            assert colors.attrs["encoding"]["n_elements"] == 2000

    def test_color_list_with_n_elements(self) -> None:
        """Test encoding a color list [R, G, B]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=[0.0, 1.0, 0.0],  # RGB list
                zarr_group=store,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
                n_elements=3000,
            )

            colors = store["colors"]
            assert colors.shape == (1, 3)
            assert np.allclose(colors[:], [[0.0, 1.0, 0.0]])


class TestScalarInputErrors:
    """Test error handling for scalar inputs."""

    def test_scalar_without_n_elements_raises(self) -> None:
        """Test that scalar input without n_elements raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="Scalar input requires n_elements"):
                encoder.encode(
                    data=0.5,  # Scalar without n_elements
                    zarr_group=store,
                    name="radii",
                    semantic_type=SemanticType.POSITIVE_SCALAR,
                )

    def test_coordinate_scalar_raises(self) -> None:
        """Test that COORDINATE type blocks scalar input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(
                ValueError, match="COORDINATE.*does not support scalar/broadcasting"
            ):
                encoder.encode(
                    data=0.0,
                    zarr_group=store,
                    name="positions",
                    semantic_type=SemanticType.COORDINATE,
                    n_elements=100,
                )

    def test_color_tuple_wrong_length_raises(self) -> None:
        """Test that color tuple with wrong length raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="must have 3 or 4 elements"):
                encoder.encode(
                    data=(1.0, 0.0),  # Only 2 elements
                    zarr_group=store,
                    name="colors",
                    semantic_type=SemanticType.COLOR,
                    color_mode="sdr",
                    n_elements=100,
                )

    def test_tuple_for_non_color_raises(self) -> None:
        """Test that tuple input only works for COLOR semantic type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="only supported for COLOR"):
                encoder.encode(
                    data=(1.0, 2.0, 3.0),
                    zarr_group=store,
                    name="radii",
                    semantic_type=SemanticType.POSITIVE_SCALAR,
                    n_elements=100,
                )

    def test_array_with_wrong_n_elements_raises(self) -> None:
        """Test that providing mismatched n_elements raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            with pytest.raises(ValueError, match="n_elements=100 but data has shape"):
                encoder.encode(
                    data=data,  # Shape (3,)
                    zarr_group=store,
                    name="radii",
                    semantic_type=SemanticType.POSITIVE_SCALAR,
                    n_elements=100,  # Mismatched!
                )

    def test_array_with_n_elements_non_uniform_raises(self) -> None:
        """Test that full array with n_elements must be uniform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            # Non-uniform array
            data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            # Providing n_elements=3 implies "I expect this to be broadcast"
            # But data is not uniform, so should raise error
            with pytest.raises(ValueError, match="varying values"):
                encoder.encode(
                    data=data,  # Varying values!
                    zarr_group=store,
                    name="radii",
                    semantic_type=SemanticType.POSITIVE_SCALAR,
                    n_elements=3,  # Implies broadcasting expectation
                )

    def test_uniform_array_with_n_elements_works(self) -> None:
        """Test that uniform array with n_elements is accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            # Uniform array - all same value
            data = np.full(100, 0.5, dtype=np.float32)

            # Should work - array is uniform
            encoder.encode(
                data=data,
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                n_elements=100,
            )

            # Verify broadcasted storage
            radii = store["radii"]
            assert radii.shape == (1,)
            assert radii.attrs["encoding"]["name"] == "broadcasted"


class TestScalarInputSemanticTypes:
    """Test scalar input with different semantic types."""

    def test_positive_scalar(self) -> None:
        """Test POSITIVE_SCALAR with scalar input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=1.5,
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                n_elements=500,
            )

            radii = store["radii"]
            assert radii.shape == (1,)
            assert radii[:][0] == pytest.approx(1.5)

    def test_bounded_scalar(self) -> None:
        """Test BOUNDED_SCALAR with scalar input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=2.0,
                zarr_group=store,
                name="sharpness",
                semantic_type=SemanticType.BOUNDED_SCALAR,
                bounds=(0.0, 31.0),
                n_elements=1000,
            )

            sharp = store["sharpness"]
            assert sharp.shape == (1,)
            assert sharp[:][0] == pytest.approx(2.0)

    def test_color_sdr_tuple(self) -> None:
        """Test COLOR SDR with tuple input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=(0.8, 0.3, 0.1),
                zarr_group=store,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
                n_elements=750,
            )

            colors = store["colors"]
            assert colors.shape == (1, 3)
            assert np.allclose(colors[:], [[0.8, 0.3, 0.1]])

    def test_color_rgba_tuple(self) -> None:
        """Test COLOR with RGBA tuple (4 elements)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=(1.0, 0.0, 0.0, 0.5),  # RGBA
                zarr_group=store,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
                n_elements=100,
            )

            colors = store["colors"]
            assert colors.shape == (1, 4)
            assert np.allclose(colors[:], [[1.0, 0.0, 0.0, 0.5]])


class TestScalarInputEncodingModes:
    """Test scalar input with different encoding modes."""

    def test_auto_mode(self) -> None:
        """Test scalar with AUTO mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=0.5,
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                mode=EncodingMode.AUTO,
                n_elements=1000,
            )

            assert store["radii"].shape == (1,)
            assert store["radii"].attrs["encoding"]["name"] == "broadcasted"

    def test_precision_mode(self) -> None:
        """Test scalar with PRECISION mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=0.5,
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                mode=EncodingMode.PRECISION,
                n_elements=1000,
            )

            # Should still broadcast (it's a lossless optimization)
            assert store["radii"].shape == (1,)
            assert store["radii"].attrs["encoding"]["name"] == "broadcasted"

    def test_memory_mode(self) -> None:
        """Test scalar with MEMORY mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=2.0,
                zarr_group=store,
                name="sharpness",
                semantic_type=SemanticType.BOUNDED_SCALAR,
                bounds=(0.0, 31.0),
                mode=EncodingMode.MEMORY,
                n_elements=500,
            )

            assert store["sharpness"].shape == (1,)
            assert store["sharpness"].attrs["encoding"]["name"] == "broadcasted"


class TestScalarInputValidation:
    """Test validation for scalar inputs."""

    def test_negative_positive_scalar_raises(self) -> None:
        """Test that negative value for POSITIVE_SCALAR raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            # This should be caught by encoder's validation
            # (after converting to array)
            with pytest.raises(ValueError, match="non-negative"):
                encoder.encode(
                    data=-0.5,
                    zarr_group=store,
                    name="radii",
                    semantic_type=SemanticType.POSITIVE_SCALAR,
                    n_elements=100,
                )

    def test_color_without_mode_raises(self) -> None:
        """Test that float color tuple without color_mode raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            with pytest.raises(ValueError, match="color_mode"):
                encoder.encode(
                    data=(1.0, 0.0, 0.0),
                    zarr_group=store,
                    name="colors",
                    semantic_type=SemanticType.COLOR,
                    # Missing color_mode!
                    n_elements=100,
                )


class TestScalarVsArrayConsistency:
    """Test that scalar and uniform array produce identical results."""

    def test_scalar_matches_uniform_array_radii(self) -> None:
        """Test that radii=0.5 produces same result as uniform array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Scalar path
            store1 = zarr.open_group(Path(tmpdir) / "scalar.luxar.zarr", mode="w")
            encoder1 = ArrayEncoder()
            encoder1.encode(
                data=0.5,
                zarr_group=store1,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                n_elements=100,
            )

            # Array path
            store2 = zarr.open_group(Path(tmpdir) / "array.luxar.zarr", mode="w")
            encoder2 = ArrayEncoder()
            encoder2.encode(
                data=np.full(100, 0.5, dtype=np.float32),
                zarr_group=store2,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
            )

            # Both should produce identical storage
            radii1 = store1["radii"]
            radii2 = store2["radii"]

            assert radii1.shape == radii2.shape == (1,)
            assert np.allclose(radii1[:], radii2[:])
            assert radii1.attrs["encoding"]["name"] == "broadcasted"
            assert radii2.attrs["encoding"]["name"] == "broadcasted"
            assert radii1.attrs["encoding"]["n_elements"] == 100
            assert radii2.attrs["encoding"]["n_elements"] == 100

    def test_scalar_matches_uniform_array_colors(self) -> None:
        """Test that color tuple produces same result as uniform color array."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Scalar path
            store1 = zarr.open_group(Path(tmpdir) / "scalar.luxar.zarr", mode="w")
            encoder1 = ArrayEncoder()
            encoder1.encode(
                data=(1.0, 0.0, 0.0),
                zarr_group=store1,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
                n_elements=200,
            )

            # Array path
            store2 = zarr.open_group(Path(tmpdir) / "array.luxar.zarr", mode="w")
            encoder2 = ArrayEncoder()
            encoder2.encode(
                data=np.full((200, 3), [1.0, 0.0, 0.0], dtype=np.float32),
                zarr_group=store2,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
            )

            # Both should produce identical storage
            colors1 = store1["colors"]
            colors2 = store2["colors"]

            assert colors1.shape == colors2.shape == (1, 3)
            assert np.allclose(colors1[:], colors2[:])
            assert colors1.attrs["encoding"]["name"] == "broadcasted"
            assert colors2.attrs["encoding"]["name"] == "broadcasted"


class TestScalarInputIntegration:
    """Integration tests with compiler."""

    def test_points_with_all_scalars(self) -> None:
        """Test writing points with all optional attributes as scalars."""
        from luxar import Dimensions, LuxarZarrCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=False) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(500, 3).astype(np.float32)

                scene.add_points(
                    "test",
                    positions,
                    radii=0.8,  # Scalar
                    colors=(0.5, 0.5, 1.0),  # Tuple
                    sharpness=0.8,  # Scalar
                )

            # Verify all were stored as broadcasted
            store = zarr.open_group(zarr_path, mode="r")
            assert store["test/radii"].shape == (1,)
            assert store["test/colors"].shape == (1, 3)
            assert store["test/sharpnesses"].shape == (1,)

            # Check values
            assert store["test/radii"][:][0] == pytest.approx(0.8)
            assert np.allclose(store["test/colors"][:], [[0.5, 0.5, 1.0]])
            assert store["test/sharpnesses"][:][0] == pytest.approx(0.8)

    def test_mixed_scalar_and_array(self) -> None:
        """Test mixing scalar and array attributes."""
        from luxar import Dimensions, LuxarZarrCompiler

        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = Path(tmpdir) / "test.luxar.zarr"

            with LuxarZarrCompiler(zarr_path, enable_spatial_index=False) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                positions = np.random.randn(100, 3).astype(np.float32)
                varied_radii = np.random.rand(100).astype(np.float32)

                scene.add_points(
                    "test",
                    positions,
                    radii=varied_radii,  # Array (varying)
                    colors=(1.0, 1.0, 0.0),  # Scalar tuple
                    sharpness=0.5,  # Scalar
                )

            # Verify storage
            store = zarr.open_group(zarr_path, mode="r")
            assert store["test/radii"].shape == (100,)  # Full array
            assert store["test/colors"].shape == (1, 3)  # Broadcasted
            assert store["test/sharpnesses"].shape == (1,)  # Broadcasted


class TestScalarNoIntermediateArrays:
    """Test that scalars don't create intermediate arrays (performance)."""

    def test_no_memory_allocation(self) -> None:
        """Verify that scalar path doesn't create intermediate (N,) arrays.

        This is a conceptual test - we verify the storage is minimal.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            # Large n_elements - if intermediate array was created, would use 4MB
            # With scalar passthrough, only 4 bytes stored
            encoder.encode(
                data=0.5,
                zarr_group=store,
                name="radii",
                semantic_type=SemanticType.POSITIVE_SCALAR,
                n_elements=1_000_000,  # 1 million elements
            )

            # Verify only (1,) array stored
            radii = store["radii"]
            assert radii.shape == (1,)
            # Actual bytes stored: just the single float32 value
            assert radii.nbytes == 4  # Not 4,000,000!

    def test_color_tuple_no_intermediate(self) -> None:
        """Verify color tuples don't create intermediate arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = zarr.open_group(Path(tmpdir) / "test.luxar.zarr", mode="w")
            encoder = ArrayEncoder()

            encoder.encode(
                data=(1.0, 0.5, 0.25),
                zarr_group=store,
                name="colors",
                semantic_type=SemanticType.COLOR,
                color_mode="sdr",
                n_elements=500_000,  # 500K elements = 6MB if expanded
            )

            # Verify only (1, 3) array stored
            colors = store["colors"]
            assert colors.shape == (1, 3)
            assert colors.nbytes == 12  # Just 3 float32 values, not 6MB!
