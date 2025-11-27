"""Comprehensive tests for streaming module."""

import numpy as np
import pytest
import zarr

from luxar import LuxarZarrCompiler, StreamingPoints


class TestStreamingPointsComprehensive:
    """Comprehensive tests for StreamingPoints class."""

    def test_streaming_basic_workflow(self, tmp_path) -> None:
        """Test basic streaming workflow."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # Initial state
            assert streaming.name == "stream"
            assert streaming.expected_dims == 3
            assert streaming.position == 0
            assert streaming.total_points == 0

            # Append batch
            positions = np.random.randn(100, 3).astype(np.float32)
            streaming.append_batch(positions)

            assert streaming.position == 100
            assert streaming.total_points == 100

            # Append another batch
            positions2 = np.random.randn(50, 3).astype(np.float32)
            streaming.append_batch(positions2)

            assert streaming.position == 150
            assert streaming.total_points == 150

            # Finalize
            metadata = streaming.finalize()
            assert metadata["n_points"] == 150
            assert metadata["dims"] == 3
            assert metadata["streaming"] is True

    def test_streaming_with_colors_late_addition(self, tmp_path) -> None:
        """Test adding colors to later batches."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # First batch without colors
            positions1 = np.random.randn(100, 3).astype(np.float32)
            streaming.append_batch(positions1)
            assert streaming.colors_dataset is None

            # Second batch with colors
            positions2 = np.random.randn(50, 3).astype(np.float32)
            colors2 = np.random.rand(50, 3).astype(np.float32)
            streaming.append_batch(positions2, colors=colors2)
            assert streaming.colors_dataset is not None
            assert streaming.has_colors is True

            metadata = streaming.finalize()
            assert metadata["has_colors"] is True

        # Verify in zarr
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        colors = store["stream/colors"][:]
        # First 100 points have no colors (zeros), next 50 have colors
        assert np.all(colors[:100] == 0)
        assert np.any(colors[100:150] != 0)

    def test_streaming_with_radii_late_addition(self, tmp_path) -> None:
        """Test adding radii to later batches."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # First batch without radii
            positions1 = np.random.randn(100, 3).astype(np.float32)
            streaming.append_batch(positions1)
            assert streaming.radii_dataset is None

            # Second batch with radii
            positions2 = np.random.randn(50, 3).astype(np.float32)
            radii2 = np.random.uniform(0.1, 2.0, 50).astype(np.float32)
            streaming.append_batch(positions2, radii=radii2)
            assert streaming.radii_dataset is not None
            assert streaming.has_radii is True

            metadata = streaming.finalize()
            assert metadata["has_radii"] is True

    def test_streaming_with_sharpness_late_addition(self, tmp_path) -> None:
        """Test adding sharpness to later batches."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # First batch without sharpness
            positions1 = np.random.randn(100, 3).astype(np.float32)
            streaming.append_batch(positions1)
            assert streaming.sharpness_dataset is None

            # Second batch with sharpness
            positions2 = np.random.randn(50, 3).astype(np.float32)
            sharpness2 = np.random.uniform(0.5, 10.0, 50).astype(np.float32)
            streaming.append_batch(positions2, sharpness=sharpness2)
            assert streaming.sharpness_dataset is not None
            assert streaming.has_sharpness is True

            metadata = streaming.finalize()
            assert metadata["has_sharpness"] is True

    def test_streaming_dimension_validation(self, tmp_path) -> None:
        """Test dimension validation in streaming."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # Try to append wrong dimensions
            positions_2d = np.random.randn(100, 2).astype(np.float32)
            with pytest.raises(ValueError, match="Expected 3D points, got 2D"):
                streaming.append_batch(positions_2d)

            positions_5d = np.random.randn(100, 5).astype(np.float32)
            with pytest.raises(ValueError, match="Expected 3D points, got 5D"):
                streaming.append_batch(positions_5d)

    def test_streaming_invalid_positions(self, tmp_path) -> None:
        """Test invalid positions in streaming."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # 1D positions
            positions_1d = np.array([1, 2, 3])
            with pytest.raises(ValueError):
                streaming.append_batch(positions_1d)

            # 3D positions
            positions_3d = np.zeros((10, 10, 3))
            with pytest.raises(ValueError):
                streaming.append_batch(positions_3d)

    def test_streaming_from_generator_tuple(self, tmp_path) -> None:
        """Test streaming from generator with tuples."""

        def data_generator():  # type: ignore[no-untyped-def]
            for i in range(3):
                positions = np.random.randn(50, 3).astype(np.float32)
                colors = np.random.rand(50, 3).astype(np.float32)
                radii = np.random.uniform(0.1, 2.0, 50).astype(np.float32)
                sharpness = np.random.uniform(0.5, 10.0, 50).astype(np.float32)
                yield (positions, colors, radii, sharpness)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            total = streaming.append_from_generator(data_generator())
            assert total == 150  # 3 batches * 50 points

            metadata = streaming.finalize()
            assert metadata["n_points"] == 150
            assert metadata["has_colors"] is True
            assert metadata["has_radii"] is True
            assert metadata["has_sharpness"] is True

    def test_streaming_from_generator_positions_only(self, tmp_path) -> None:
        """Test streaming from generator with positions only."""

        def data_generator():  # type: ignore[no-untyped-def]
            for i in range(5):
                yield np.random.randn(20, 3).astype(np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            total = streaming.append_from_generator(data_generator())
            assert total == 100  # 5 batches * 20 points

            metadata = streaming.finalize()
            assert metadata["n_points"] == 100
            assert metadata["has_colors"] is False
            assert metadata["has_radii"] is False
            assert metadata["has_sharpness"] is False

    def test_streaming_from_generator_max_batches(self, tmp_path) -> None:
        """Test streaming with max_batches limit."""

        def infinite_generator():  # type: ignore[no-untyped-def]
            while True:
                yield np.random.randn(10, 3).astype(np.float32)

        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            total = streaming.append_from_generator(infinite_generator(), max_batches=5)
            assert total == 50  # 5 batches * 10 points

            metadata = streaming.finalize()
            assert metadata["n_points"] == 50

    def test_streaming_finalize_with_attrs(self, tmp_path) -> None:
        """Test finalize with additional attributes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            positions = np.random.randn(100, 3).astype(np.float32)
            streaming.append_batch(positions)

            # Finalize with custom attributes
            metadata = streaming.finalize(
                opacity=0.8,
                gamma=1.5,
                blending_mode="additive",
                custom_attr="test_value",
            )

            assert metadata["n_points"] == 100

        # Verify attributes in zarr
        store = zarr.open_group(tmp_path / "test.zarr", mode="r")
        attrs = store["stream"].attrs
        assert attrs["opacity"] == 0.8
        assert attrs["gamma"] == 1.5
        assert attrs["blending_mode"] == "additive"
        assert attrs["custom_attr"] == "test_value"

    def test_streaming_empty_finalize(self, tmp_path) -> None:
        """Test finalizing without any data."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # Finalize without appending any data
            metadata = streaming.finalize()
            assert metadata["n_points"] == 0
            assert metadata["has_colors"] is False
            assert metadata["has_radii"] is False
            assert metadata["has_sharpness"] is False

    def test_streaming_mixed_attributes(self, tmp_path) -> None:
        """Test streaming with mixed presence of attributes."""
        with LuxarZarrCompiler(tmp_path / "test.zarr") as compiler:
            streaming = StreamingPoints("stream", compiler, expected_dims=3)

            # Batch 1: positions only
            positions1 = np.random.randn(30, 3).astype(np.float32)
            streaming.append_batch(positions1)

            # Batch 2: positions + colors
            positions2 = np.random.randn(30, 3).astype(np.float32)
            colors2 = np.random.rand(30, 3).astype(np.float32)
            streaming.append_batch(positions2, colors=colors2)

            # Batch 3: positions + colors + radii
            positions3 = np.random.randn(30, 3).astype(np.float32)
            colors3 = np.random.rand(30, 3).astype(np.float32)
            radii3 = np.ones(30, dtype=np.float32)
            streaming.append_batch(positions3, colors=colors3, radii=radii3)

            # Batch 4: positions + all attributes
            positions4 = np.random.randn(30, 3).astype(np.float32)
            colors4 = np.random.rand(30, 3).astype(np.float32)
            radii4 = np.ones(30, dtype=np.float32) * 0.5
            sharpness4 = np.ones(30, dtype=np.float32) * 2.0
            streaming.append_batch(
                positions4, colors=colors4, radii=radii4, sharpness=sharpness4
            )

            metadata = streaming.finalize()
            assert metadata["n_points"] == 120
            assert metadata["has_colors"] is True
            assert metadata["has_radii"] is True
            assert metadata["has_sharpness"] is True
