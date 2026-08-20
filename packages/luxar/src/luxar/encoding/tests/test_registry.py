"""Tests for ArrayRefRegistry class.

Tests cover duplicate detection, hashing, and registry lifecycle.
"""

import numpy as np

from luxar.encoding.registry import ArrayRefRegistry


class TestDuplicateDetection:
    """Test duplicate array detection."""

    def test_detect_duplicate_exact_match(self):
        """Test detecting exact duplicate arrays."""
        registry = ArrayRefRegistry()

        data1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data2 = data1.copy()

        # First array is new
        match1 = registry.check(data1, "path1")
        assert not match1.is_duplicate
        assert match1.target_path is None
        assert match1.hash.startswith("xxh64:")

        # Second array is duplicate
        match2 = registry.check(data2, "path2")
        assert match2.is_duplicate
        assert match2.target_path == "path1"
        assert match2.hash == match1.hash  # Same hash

    def test_different_arrays_not_duplicates(self):
        """Test different arrays not detected as duplicates."""
        registry = ArrayRefRegistry()

        data1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)

        match1 = registry.check(data1, "path1")
        match2 = registry.check(data2, "path2")

        assert not match1.is_duplicate
        assert not match2.is_duplicate
        assert match1.hash != match2.hash

    def test_different_dtype_not_duplicate(self):
        """Test same values but different dtype not detected as duplicate."""
        registry = ArrayRefRegistry()

        data1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        data2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        match1 = registry.check(data1, "path1")
        match2 = registry.check(data2, "path2")

        assert not match1.is_duplicate
        assert not match2.is_duplicate

    def test_same_bytes_different_shape_is_duplicate(self):
        """Test that arrays with same bytes but different shape ARE detected as duplicates.

        This is correct behavior because:
        1. Content-based deduplication should detect identical byte content
        2. The encoder stores original_shape in array_ref metadata
        3. The decoder uses original_shape to reconstruct properly
        """
        registry = ArrayRefRegistry()

        data1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)  # shape (4,)
        data2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # shape (2, 2)

        # Same bytes, different shape → should still be detected as duplicate
        assert data1.tobytes() == data2.tobytes()  # Verify they have same bytes

        match1 = registry.check(data1, "path1")
        match2 = registry.check(data2, "path2")

        assert not match1.is_duplicate  # First is new
        assert match2.is_duplicate  # Second has same bytes → duplicate
        assert match2.target_path == "path1"
        assert match1.hash == match2.hash  # Same hash since same bytes


class TestRegistryLifecycle:
    """Test registry lifecycle and management."""

    def test_registry_clear(self):
        """Test clearing registry."""
        registry = ArrayRefRegistry()

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Register an array
        match1 = registry.check(data, "path1")
        assert not match1.is_duplicate

        # Clear registry
        registry.clear()

        # Same array should now be new
        match2 = registry.check(data, "path2")
        assert not match2.is_duplicate
        assert match2.hash == match1.hash  # Same hash, but registry was cleared

    def test_snapshot_restore_keeps_prior_entries_and_discards_later_ones(self):
        registry = ArrayRefRegistry()
        first = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        second = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        registry.check(first, "before")
        snapshot = registry.snapshot()
        registry.check(second, "rolled_back")

        registry.restore(snapshot)

        first_match = registry.check(first.copy(), "first_retry")
        second_match = registry.check(second.copy(), "second_retry")
        assert first_match.is_duplicate
        assert first_match.target_path == "before"
        assert not second_match.is_duplicate

    def test_large_array_hashing(self):
        """Test hashing of large arrays (triggers two-stage check)."""
        registry = ArrayRefRegistry()

        # Create array > 32KB (triggers two-stage hashing)
        # float32 = 4 bytes, so need > 8192 elements
        large_data = np.random.rand(10000).astype(np.float32)

        match1 = registry.check(large_data, "path1")
        assert not match1.is_duplicate

        # Duplicate should be detected
        match2 = registry.check(large_data.copy(), "path2")
        assert match2.is_duplicate
        assert match2.target_path == "path1"

    def test_small_array_hashing(self):
        """Test hashing of small arrays (skips quick check)."""
        registry = ArrayRefRegistry()

        # Small array (< 32KB)
        small_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        match1 = registry.check(small_data, "path1")
        assert not match1.is_duplicate
        assert match1.hash.startswith("xxh64:")

        # Duplicate should be detected
        match2 = registry.check(small_data.copy(), "path2")
        assert match2.is_duplicate
        assert match2.target_path == "path1"


class TestHashConsistency:
    """Test hash consistency and collision handling."""

    def test_hash_consistency(self):
        """Test same data produces same hash."""
        registry = ArrayRefRegistry()

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        match1 = registry.check(data, "path1")
        match2 = registry.check(data, "path2")  # Same data, different path

        assert match2.is_duplicate
        assert match1.hash == match2.hash

    def test_multiple_duplicates(self):
        """Test multiple duplicates all point to first occurrence."""
        registry = ArrayRefRegistry()

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        match1 = registry.check(data, "path1")
        match2 = registry.check(data.copy(), "path2")
        match3 = registry.check(data.copy(), "path3")

        assert not match1.is_duplicate
        assert match2.is_duplicate
        assert match3.is_duplicate
        assert match2.target_path == "path1"
        assert match3.target_path == "path1"  # All point to first
