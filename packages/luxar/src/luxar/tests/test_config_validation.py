"""Test validation functions in config.py module."""

import pytest

from luxar.config import (
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    validate_chunk_size,
    validate_compression_level,
)


class TestChunkSizeValidation:
    """Test validate_chunk_size function."""

    def test_valid_chunk_size(self):
        """Test that valid chunk sizes are accepted."""
        # Test minimum valid size
        assert validate_chunk_size(MIN_CHUNK_SIZE) == MIN_CHUNK_SIZE
        
        # Test maximum valid size
        assert validate_chunk_size(MAX_CHUNK_SIZE) == MAX_CHUNK_SIZE
        
        # Test common sizes
        assert validate_chunk_size(32768) == 32768
        assert validate_chunk_size(65536) == 65536

    def test_chunk_size_not_integer(self):
        """Test that non-integer chunk sizes raise ValueError."""
        with pytest.raises(ValueError, match="Chunk size must be an integer"):
            validate_chunk_size(32768.5)
        
        with pytest.raises(ValueError, match="Chunk size must be an integer"):
            validate_chunk_size("32768")
        
        with pytest.raises(ValueError, match="Chunk size must be an integer"):
            validate_chunk_size(None)

    def test_chunk_size_too_small(self):
        """Test that too small chunk sizes raise ValueError."""
        with pytest.raises(ValueError, match=f"Chunk size .* is too small \\(min: {MIN_CHUNK_SIZE}\\)"):
            validate_chunk_size(MIN_CHUNK_SIZE - 1)
        
        with pytest.raises(ValueError, match=f"Chunk size .* is too small \\(min: {MIN_CHUNK_SIZE}\\)"):
            validate_chunk_size(0)
        
        with pytest.raises(ValueError, match=f"Chunk size .* is too small \\(min: {MIN_CHUNK_SIZE}\\)"):
            validate_chunk_size(-1000)

    def test_chunk_size_too_large(self):
        """Test that too large chunk sizes raise ValueError."""
        with pytest.raises(ValueError, match=f"Chunk size .* is too large \\(max: {MAX_CHUNK_SIZE}\\)"):
            validate_chunk_size(MAX_CHUNK_SIZE + 1)
        
        with pytest.raises(ValueError, match=f"Chunk size .* is too large \\(max: {MAX_CHUNK_SIZE}\\)"):
            validate_chunk_size(10_000_000)


class TestCompressionLevelValidation:
    """Test validate_compression_level function."""

    def test_valid_compression_levels(self):
        """Test that valid compression levels are accepted."""
        for level in range(1, 10):
            assert validate_compression_level(level) == level

    def test_compression_level_not_integer(self):
        """Test that non-integer compression levels raise ValueError."""
        with pytest.raises(ValueError, match="Compression level must be an integer"):
            validate_compression_level(5.5)
        
        with pytest.raises(ValueError, match="Compression level must be an integer"):
            validate_compression_level("5")
        
        with pytest.raises(ValueError, match="Compression level must be an integer"):
            validate_compression_level(None)

    def test_compression_level_out_of_range(self):
        """Test that out-of-range compression levels raise ValueError."""
        with pytest.raises(ValueError, match="Compression level .* must be between 1 and 9"):
            validate_compression_level(0)
        
        with pytest.raises(ValueError, match="Compression level .* must be between 1 and 9"):
            validate_compression_level(10)
        
        with pytest.raises(ValueError, match="Compression level .* must be between 1 and 9"):
            validate_compression_level(-1)
