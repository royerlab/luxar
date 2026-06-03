"""Finalize-time content hashing (post-order xxhash64 over the zarr tree)."""

from __future__ import annotations

import json

import xxhash
import zarr
from arbol import aprint


def compute_content_hashes(store: zarr.Group) -> str:
    """Compute content hashes for all nodes using post-order traversal.

    Called during finalize() after all nodes have been written.
    Uses xxhash64 for speed.

    Args:
        store: Root zarr group

    Returns:
        Root content hash
    """

    def compute_hash_recursive(group_path: str) -> str:
        """Recursively compute hash for a group and its children."""
        group = store[group_path] if group_path else store

        hasher = xxhash.xxh64()

        # 1. Hash this node's own datasets (positions, colors, etc.)
        for dataset_name in sorted(group.array_keys()):
            dataset = group[dataset_name]
            hasher.update(dataset[:].tobytes())

        # 2. Hash metadata (excluding content_hash to avoid recursion)
        attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())

        # 3. Hash child groups (recursively, sorted for determinism)
        for child_name in sorted(group.group_keys()):
            child_path = f"{group_path}/{child_name}" if group_path else child_name
            child_hash = compute_hash_recursive(child_path)
            hasher.update(child_hash.encode())

        # Store hash in this node's attrs
        content_hash = hasher.hexdigest()
        group.attrs["content_hash"] = content_hash

        return content_hash

    # Start from root (empty path)
    root_hash = compute_hash_recursive("")
    aprint(f"Scene content hash: {root_hash[:16]}...")
    return root_hash
