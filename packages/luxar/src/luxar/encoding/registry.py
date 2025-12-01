"""Array reference registry for deduplication.

This module provides efficient duplicate detection for array references,
using a two-stage hashing approach with xxhash64.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import xxhash


@dataclass
class ArrayRefMatch:
    """Result of checking an array against the registry.

    Attributes:
        is_duplicate: True if array matches an existing registered array
        target_path: Path to the original array (if duplicate), else None
        hash: Full content hash (xxh64 hex digest), always provided
    """

    is_duplicate: bool
    target_path: Optional[str]
    hash: str


class ArrayRefRegistry:
    """Track arrays for deduplication via references.

    Uses a two-stage detection algorithm:
    1. Quick check: (dtype, shape, first_32KB_hash)
    2. Full hash: xxhash64 of entire array data (if quick check matches)

    The registry maintains mappings from quick keys and full hashes to paths,
    enabling efficient duplicate detection with minimal false positives.
    """

    # Size threshold for small arrays (skip quick check)
    _SMALL_ARRAY_BYTES = 32 * 1024  # 32KB

    def __init__(self) -> None:
        """Initialize an empty registry."""
        # Quick key → (full_hash, path) mapping
        self._quick_map: dict[tuple, tuple[str, str]] = {}

        # Full hash → path mapping (for verification)
        self._full_map: dict[str, str] = {}

    def check(self, data: np.ndarray, path: str) -> ArrayRefMatch:
        """Check if array matches an existing one.

        If the array is new (no duplicate), it is automatically registered
        for future checks.

        Args:
            data: Array to check for duplicates
            path: Path where this array will be stored

        Returns:
            ArrayRefMatch indicating whether array is duplicate, with hash
        """
        dtype_str = str(data.dtype)
        shape_tuple = data.shape
        data_bytes = data.tobytes()
        array_size = len(data_bytes)

        # Always check dtype and shape first (per spec: not hashed, compared separately)
        # Stage 1: Quick check (unless array is small)
        if array_size <= self._SMALL_ARRAY_BYTES:
            # Small array: skip quick check, compute full hash directly
            # But still need to check dtype/shape, so create a simple key
            full_hash = self._compute_full_hash(data_bytes)
            simple_key = (dtype_str, shape_tuple)  # For small arrays
        else:
            # Large array: compute quick key from first 32KB
            first_chunk = data_bytes[: self._SMALL_ARRAY_BYTES]
            quick_hash = self._compute_hash(first_chunk)
            quick_key = (dtype_str, shape_tuple, quick_hash)

            # Check if quick key exists
            if quick_key not in self._quick_map:
                # No match: compute full hash and register
                full_hash = self._compute_full_hash(data_bytes)
                self._register(quick_key, full_hash, path)
                return ArrayRefMatch(
                    is_duplicate=False, target_path=None, hash=full_hash
                )

            # Quick key matched: verify with full hash
            stored_full_hash, stored_path = self._quick_map[quick_key]
            full_hash = self._compute_full_hash(data_bytes)

            if full_hash == stored_full_hash:
                # True duplicate found
                return ArrayRefMatch(
                    is_duplicate=True, target_path=stored_path, hash=full_hash
                )
            else:
                # Hash collision in quick key (rare): register as new
                self._register(quick_key, full_hash, path)
                return ArrayRefMatch(
                    is_duplicate=False, target_path=None, hash=full_hash
                )

        # Small array path: check dtype/shape first, then full hash
        # Check if we have this dtype/shape combination
        if simple_key in self._quick_map:
            # Dtype/shape match exists, verify with full hash
            stored_full_hash, stored_path = self._quick_map[simple_key]
            if full_hash == stored_full_hash:
                # True duplicate (same simple_key, same hash)
                return ArrayRefMatch(
                    is_duplicate=True, target_path=stored_path, hash=full_hash
                )
            else:
                # Different data with same dtype/shape
                # BUT: check if this hash was registered under a DIFFERENT simple_key
                # (e.g., colors and positions may have same dtype/shape but different content)
                if full_hash in self._full_map:
                    # This content was already registered elsewhere!
                    return ArrayRefMatch(
                        is_duplicate=True,
                        target_path=self._full_map[full_hash],
                        hash=full_hash,
                    )
                # Truly new data - update quick_map (most recent wins for simple_key)
                self._quick_map[simple_key] = (full_hash, path)
                self._full_map[full_hash] = path
                return ArrayRefMatch(
                    is_duplicate=False, target_path=None, hash=full_hash
                )
        else:
            # New dtype/shape combination
            # Still check full_map in case same content was registered with different dtype/shape
            if full_hash in self._full_map:
                return ArrayRefMatch(
                    is_duplicate=True,
                    target_path=self._full_map[full_hash],
                    hash=full_hash,
                )
            # Truly new: register both
            self._quick_map[simple_key] = (full_hash, path)
            self._full_map[full_hash] = path
            return ArrayRefMatch(is_duplicate=False, target_path=None, hash=full_hash)

    def clear(self) -> None:
        """Reset registry (e.g., between independent scenes)."""
        self._quick_map.clear()
        self._full_map.clear()

    def _compute_hash(self, data: bytes) -> str:
        """Compute xxhash64 hex digest of data.

        Args:
            data: Byte data to hash

        Returns:
            Hex digest string (16 hex characters)
        """
        return xxhash.xxh64(data).hexdigest()

    def _compute_full_hash(self, data: bytes) -> str:
        """Compute full hash with xxh64 prefix.

        Args:
            data: Complete array byte data

        Returns:
            Hash string in format "xxh64:<hex_digest>"
        """
        return f"xxh64:{self._compute_hash(data)}"

    def _register(self, quick_key: tuple, full_hash: str, path: str) -> None:
        """Register a new array in the registry.

        Args:
            quick_key: (dtype, shape, quick_hash) tuple
            full_hash: Full content hash with xxh64 prefix
            path: Path where array is stored
        """
        self._quick_map[quick_key] = (full_hash, path)
        self._full_map[full_hash] = path
