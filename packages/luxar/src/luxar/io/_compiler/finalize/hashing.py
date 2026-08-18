"""Finalize-time content hashing (post-order xxhash64 over the zarr tree)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import xxhash
import zarr
from arbol import aprint

from luxar._zarr_compat import NODE_ATTR_DOCS, NODE_GROUP_DOCS, read_raw_bytes

# Attr keys whose value is the filename of a plain (non-zarr) payload file stored
# *inside* the group's own directory. Such files have no chunk grid and no zarr
# metadata, so `array_keys()` and `group_keys()` are blind to them and their bytes
# reach the digest only through this list. A new payload kind registers its attr
# key here. Full rationale: `finalize/README.md`.
PAYLOAD_FILE_ATTRS: tuple[str, ...] = ("image_file",)

#: zarr's own metadata documents, both on-disk formats (`_zarr_compat` already
#: names the attr/group ones; `.zarray` and `.zmetadata` complete the set).
_ZARR_METADATA_DOCS: frozenset[str] = frozenset(
    (*NODE_ATTR_DOCS, *NODE_GROUP_DOCS, ".zarray", ".zmetadata")
)


def _is_safe_payload_name(filename: str) -> bool:
    """True when ``filename`` is a name this walk is willing to READ at all.

    Refuses on SEMANTICS only: a name that is not a single path component (zarr's
    ``normalize_path`` rewrites ``\\`` to ``/`` and raises on ``.`` / ``..``, so
    the name addresses a key outside the group's own directory, or nothing at
    all), or one of zarr's metadata documents (they carry the ``content_hash``
    this very walk stamps, so reading one would make the digest non-convergent).
    Whether some store could open a name is the STORE's verdict, not this
    predicate's — see the ``unreadable:`` fold in :func:`_payload_terms`. A
    refused name is folded in BY NAME and never read, so the walk stays total
    over whatever attrs a store on disk actually carries.
    """
    return (
        "/" not in filename
        and "\\" not in filename
        and filename not in (".", "..")
        and filename not in _ZARR_METADATA_DOCS
    )


def _payload_terms(group: zarr.Group, attrs: dict[str, Any]) -> Iterator[bytes]:
    """Yield the hash terms for whatever payload files ``attrs`` names.

    Every variable-length term carries its own byte length, which is what makes
    the block injective by itself: without the key and name lengths, a group
    naming ``a`` with bytes ``5:hello`` and one naming ``a7:`` with bytes
    ``hello`` both emit ``image_filea7:5:hello``. That ambiguity is unreachable
    through :func:`compute_content_hashes` today (step 2 folds in the attrs JSON,
    which already carries the filename), so the prefixes are what stop this block
    leaning on that coincidence. A group naming no payload file yields nothing at
    all — what keeps a payload-free tree's digest identical to earlier versions'.

    Args:
        group: Group whose directory holds the payload files.
        attrs: The group's attrs, already stripped of ``content_hash``.

    Yields:
        Byte terms to fold into the group's hasher, in order.
    """
    for attr_key in sorted(PAYLOAD_FILE_ATTRS):
        filename = attrs.get(attr_key)
        if not isinstance(filename, str) or not filename:
            continue
        # Key + name go in unconditionally — a rejected name included.
        key_bytes = attr_key.encode()
        # `surrogatepass` so ANY `str` maps to bytes: a lone surrogate survives a
        # round-trip through zarr attrs, and a plain `.encode()` would raise here.
        name_bytes = filename.encode(errors="surrogatepass")
        yield f"payload:{len(key_bytes)}:".encode()
        yield key_bytes
        yield f"{len(name_bytes)}:".encode()
        yield name_bytes
        if not _is_safe_payload_name(filename):
            yield b"unsafe:"  # rejected by name, never read
            continue
        try:
            payload = read_raw_bytes(group, filename)
        except (OSError, ValueError):
            # The store owns the readability verdict — a length/charset heuristic
            # here is wrong in both directions (see `finalize/README.md`). A
            # payload we cannot read degrades to a deterministic term rather than
            # aborting the compile: a raise reaches `finalize()`, which stamps the
            # store `incomplete`. Honestly, this also folds a GENUINE transient
            # I/O error into a different-but-deterministic digest: one re-download.
            yield b"unreadable:"
            continue
        if payload is None:
            # Distinct from a zero-byte file ("0:"), which would otherwise hash
            # the same as a missing one.
            yield b"absent:"
        else:
            yield f"{len(payload)}:".encode()
            yield payload


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

        # 3. Hash plain payload files named by attrs (overlay images): neither
        # arrays nor groups, so steps 1-2 fold in the FILENAME but never the
        # bytes. Attrs-driven, not by directory listing, and COMPILE time only —
        # nothing re-hashes a finished store. Why: `finalize/README.md`.
        for term in _payload_terms(group, attrs):
            hasher.update(term)

        # 4. Hash child groups (recursively, sorted for determinism)
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
