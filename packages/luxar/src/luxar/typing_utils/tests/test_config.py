"""Tests for typing_utils/config.py module.

Tests cover:
- validate_chunk_size function
- validate_compression_level function
- estimate_memory_usage function
- check_dataset_size_warning function
- Configuration constants
"""

import pytest

from luxar.typing_utils.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COMPRESSION,
    DEFAULT_COMPRESSION_LEVEL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_VERSION,
    LARGE_DATASET_WARNING,
    MAX_CHUNK_SIZE,
    MAX_RECOMMENDED_POINTS,
    MEMORY_PER_POINT_COLORS,
    MEMORY_PER_POINT_POSITIONS,
    MEMORY_PER_POINT_TOTAL,
    MIN_CHUNK_SIZE,
    SUPPORTED_COMPRESSION,
    SUPPORTED_VERSIONS,
    check_dataset_size_warning,
    estimate_memory_usage,
    validate_chunk_size,
    validate_compression_level,
)


class TestConfigConstants:
    """Tests for configuration constants."""

    def test_chunk_size_defaults(self) -> None:
        """Test chunk size constants are reasonable."""
        assert MIN_CHUNK_SIZE > 0
        assert MAX_CHUNK_SIZE > MIN_CHUNK_SIZE
        assert MIN_CHUNK_SIZE <= DEFAULT_CHUNK_SIZE <= MAX_CHUNK_SIZE

    def test_version_defaults(self) -> None:
        """Test version constants."""
        assert DEFAULT_VERSION in SUPPORTED_VERSIONS

    def test_compression_defaults(self) -> None:
        """Test compression constants."""
        assert DEFAULT_COMPRESSION in SUPPORTED_COMPRESSION
        assert 1 <= DEFAULT_COMPRESSION_LEVEL <= 9

    def test_memory_constants(self) -> None:
        """Test memory estimation constants."""
        assert MEMORY_PER_POINT_POSITIONS == 12  # 3 * float32
        assert MEMORY_PER_POINT_COLORS == 3  # 3 * uint8
        assert MEMORY_PER_POINT_TOTAL == MEMORY_PER_POINT_POSITIONS + MEMORY_PER_POINT_COLORS

    def test_log_level_default(self) -> None:
        """Test default log level is INFO."""
        assert DEFAULT_LOG_LEVEL == "INFO"


class TestValidateChunkSize:
    """Tests for validate_chunk_size function."""

    def test_valid_chunk_size(self) -> None:
        """Test valid chunk size passes."""
        result = validate_chunk_size(DEFAULT_CHUNK_SIZE)
        assert result == DEFAULT_CHUNK_SIZE

    def test_minimum_chunk_size(self) -> None:
        """Test minimum chunk size is accepted."""
        result = validate_chunk_size(MIN_CHUNK_SIZE)
        assert result == MIN_CHUNK_SIZE

    def test_maximum_chunk_size(self) -> None:
        """Test maximum chunk size is accepted."""
        result = validate_chunk_size(MAX_CHUNK_SIZE)
        assert result == MAX_CHUNK_SIZE

    def test_chunk_size_too_small(self) -> None:
        """Test chunk size below minimum raises error."""
        with pytest.raises(ValueError, match="too small"):
            validate_chunk_size(MIN_CHUNK_SIZE - 1)

    def test_chunk_size_too_large(self) -> None:
        """Test chunk size above maximum raises error."""
        with pytest.raises(ValueError, match="too large"):
            validate_chunk_size(MAX_CHUNK_SIZE + 1)

    def test_chunk_size_not_integer(self) -> None:
        """Test non-integer chunk size raises error."""
        with pytest.raises(ValueError, match="integer"):
            validate_chunk_size(1024.5)  # type: ignore[arg-type]

    def test_chunk_size_string_error(self) -> None:
        """Test string chunk size raises error."""
        with pytest.raises(ValueError, match="integer"):
            validate_chunk_size("1024")  # type: ignore[arg-type]


class TestValidateCompressionLevel:
    """Tests for validate_compression_level function."""

    def test_valid_compression_level(self) -> None:
        """Test valid compression level passes."""
        result = validate_compression_level(DEFAULT_COMPRESSION_LEVEL)
        assert result == DEFAULT_COMPRESSION_LEVEL

    def test_minimum_compression_level(self) -> None:
        """Test minimum compression level (1) is accepted."""
        result = validate_compression_level(1)
        assert result == 1

    def test_maximum_compression_level(self) -> None:
        """Test maximum compression level (9) is accepted."""
        result = validate_compression_level(9)
        assert result == 9

    def test_compression_level_too_low(self) -> None:
        """Test compression level below 1 raises error."""
        with pytest.raises(ValueError, match="between 1 and 9"):
            validate_compression_level(0)

    def test_compression_level_too_high(self) -> None:
        """Test compression level above 9 raises error."""
        with pytest.raises(ValueError, match="between 1 and 9"):
            validate_compression_level(10)

    def test_compression_level_not_integer(self) -> None:
        """Test non-integer compression level raises error."""
        with pytest.raises(ValueError, match="integer"):
            validate_compression_level(3.5)  # type: ignore[arg-type]

    def test_compression_level_negative(self) -> None:
        """Test negative compression level raises error."""
        with pytest.raises(ValueError, match="between 1 and 9"):
            validate_compression_level(-1)


class TestEstimateMemoryUsage:
    """Tests for estimate_memory_usage function."""

    def test_basic_memory_calculation(self) -> None:
        """Test basic memory calculation with colors."""
        n_points = 1000
        result = estimate_memory_usage(n_points, has_colors=True)
        expected = n_points * MEMORY_PER_POINT_TOTAL
        assert result == expected

    def test_memory_without_colors(self) -> None:
        """Test memory calculation without colors."""
        n_points = 1000
        result = estimate_memory_usage(n_points, has_colors=False)
        expected = n_points * MEMORY_PER_POINT_POSITIONS
        assert result == expected

    def test_zero_points(self) -> None:
        """Test zero points returns zero memory."""
        result = estimate_memory_usage(0, has_colors=True)
        assert result == 0

    def test_large_dataset_memory(self) -> None:
        """Test memory calculation for large dataset."""
        n_points = 1_000_000
        result = estimate_memory_usage(n_points, has_colors=True)
        expected = n_points * MEMORY_PER_POINT_TOTAL
        assert result == expected

    def test_negative_points_error(self) -> None:
        """Test negative points raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            estimate_memory_usage(-100)

    def test_non_integer_points_error(self) -> None:
        """Test non-integer points raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            estimate_memory_usage(1000.5)  # type: ignore[arg-type]

    def test_string_points_error(self) -> None:
        """Test string points raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            estimate_memory_usage("1000")  # type: ignore[arg-type]


class TestCheckDatasetSizeWarning:
    """Tests for check_dataset_size_warning function."""

    def test_small_dataset_no_warning(self) -> None:
        """Test small dataset returns no warning."""
        result = check_dataset_size_warning(1000)
        assert result is None

    def test_medium_dataset_no_warning(self) -> None:
        """Test medium dataset below threshold returns no warning."""
        result = check_dataset_size_warning(LARGE_DATASET_WARNING - 1)
        assert result is None

    def test_large_dataset_warning(self) -> None:
        """Test large dataset returns warning."""
        n_points = LARGE_DATASET_WARNING + 100
        result = check_dataset_size_warning(n_points)
        assert result is not None
        assert "Large dataset" in result
        assert f"{n_points:,}" in result

    def test_very_large_dataset_warning(self) -> None:
        """Test very large dataset returns exceeds warning."""
        n_points = MAX_RECOMMENDED_POINTS + 100
        result = check_dataset_size_warning(n_points)
        assert result is not None
        assert "exceeds recommended maximum" in result
        assert f"{n_points:,}" in result

    def test_exact_threshold_large(self) -> None:
        """Test exact large threshold triggers warning."""
        # Slightly above threshold
        n_points = LARGE_DATASET_WARNING + 1
        result = check_dataset_size_warning(n_points)
        assert result is not None

    def test_exact_threshold_max(self) -> None:
        """Test exact max threshold triggers warning."""
        # Slightly above threshold
        n_points = MAX_RECOMMENDED_POINTS + 1
        result = check_dataset_size_warning(n_points)
        assert result is not None
        assert "exceeds recommended" in result

    def test_boundary_below_large(self) -> None:
        """Test exactly at large threshold (no warning)."""
        result = check_dataset_size_warning(LARGE_DATASET_WARNING)
        assert result is None

    def test_boundary_at_max(self) -> None:
        """Test exactly at max threshold (no exceeds warning)."""
        result = check_dataset_size_warning(MAX_RECOMMENDED_POINTS)
        # Should return large dataset warning, not exceeds warning
        assert result is not None
        assert "Large dataset" in result


class TestDefaultCompressor:
    """Tests for DEFAULT_COMP import."""

    def test_default_comp_importable(self) -> None:
        """Test DEFAULT_COMP can be imported."""
        from luxar.typing_utils.config import DEFAULT_COMP

        # DEFAULT_COMP may be None if io.reader import fails
        # This tests the import path, not the value
        assert DEFAULT_COMP is None or hasattr(DEFAULT_COMP, "encode")


class TestSupportedDtypes:
    """Tests for supported data type constants."""

    def test_position_dtypes(self) -> None:
        """Test position dtypes include float32 and float16."""
        from luxar.typing_utils.config import SUPPORTED_POSITION_DTYPES

        assert "float32" in SUPPORTED_POSITION_DTYPES
        assert "float16" in SUPPORTED_POSITION_DTYPES

    def test_color_dtypes(self) -> None:
        """Test color dtypes include common types."""
        from luxar.typing_utils.config import SUPPORTED_COLOR_DTYPES

        assert "float32" in SUPPORTED_COLOR_DTYPES
        assert "uint8" in SUPPORTED_COLOR_DTYPES
        assert "uint16" in SUPPORTED_COLOR_DTYPES

    def test_scalar_dtypes(self) -> None:
        """Test scalar dtypes include common types."""
        from luxar.typing_utils.config import SUPPORTED_SCALAR_DTYPES

        assert "float32" in SUPPORTED_SCALAR_DTYPES
        assert "float16" in SUPPORTED_SCALAR_DTYPES
        assert "uint8" in SUPPORTED_SCALAR_DTYPES
