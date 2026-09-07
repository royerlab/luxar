"""Finalize-time content hashing (post-order xxhash64 over the zarr tree)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import xxhash
import zarr
from arbol import aprint

from luxar._zarr_compat import (
    NODE_ATTR_DOCS,
    NODE_GROUP_DOCS,
    list_raw_keys,
    read_raw_bytes,
)

# Attr keys whose value is the filename of a plain (non-zarr) payload file stored
# *inside* the group's own directory. Such files have no chunk grid and no zarr
# metadata, so `array_keys()` and `group_keys()` are blind to them and their bytes
# reach the digest only through this list. A new payload kind registers its attr
# key here. Full rationale: `finalize/README.md`.
PAYLOAD_FILE_ATTRS: tuple[str, ...] = ("image_file", "poster_file", "video_file")

#: zarr's own metadata documents, both on-disk formats (`_zarr_compat` already
#: names the attr/group ones; `.zarray` and `.zmetadata` complete the set).
_ZARR_METADATA_DOCS: frozenset[str] = frozenset(
    (*NODE_ATTR_DOCS, *NODE_GROUP_DOCS, ".zarray", ".zmetadata")
)
#: Lowercased for the narrow case-insensitive metadata-name collision test.
#:
#: ``str.lower()`` rather than ``str.casefold()``: every metadata name is
#: lowercase ASCII, so both catch the real hazard (``Zarr.json``, ``.ZAttrs``),
#: while casefolding would also map ``ſ`` (U+017F) to ``s`` and wrongly refuse
#: an ordinary distinct payload such as ``.zattrſ``.
_ZARR_METADATA_DOCS_LOWERCASED: frozenset[str] = frozenset(
    name.lower() for name in _ZARR_METADATA_DOCS
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
        if filename.lower() in _ZARR_METADATA_DOCS_LOWERCASED:
            if not group.store_path.store.supports_listing:
                yield b"unreadable:"
                continue
            try:
                present_exactly = filename in list_raw_keys(group)
            except (NotImplementedError, OSError, ValueError):
                # Never fall back to the read here: on a case-insensitive
                # filesystem that is exactly the operation that can resolve the
                # name onto the group's own metadata document and make the hash
                # depend on its previous stamp.
                yield b"unreadable:"
                continue
            if not present_exactly:
                yield b"absent:"
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


def _storage_identity(name: str, dataset: zarr.Array) -> dict[str, Any]:
    """The array's STORAGE identity — everything that changes what a chunk key means.

    Why layout counts as identity, stated here once and referenced elsewhere:
    ``content_hash`` is the token the viewer's ``MultiLevelCachingStore``
    compares against the remote to validate a cached dataset, and that cache holds
    ENCODED CHUNKS KEYED BY CHUNK INDEX. So a re-chunked store with byte-identical
    values is NOT interchangeable with its input — serving one's cached chunks for
    the other hands the viewer wrong bytes at every key — and must not share its
    hash. Every term is metadata already in hand, so folding it in costs no I/O.

    ``name``/``shape``/``dtype``
        A renamed array, or the same values reshaped, is different content, and
        ``tobytes()`` alone tells neither apart (a ``(2, 3)`` and a ``(3, 2)``
        array holding the same values serialize identically).
    ``chunks``/``shards``
        What a chunk key addresses.
    ``attrs``
        The array's OWN attrs, where Luxar keeps its DEQUANTIZATION parameters
        (``encoding``: a scalar's ``min``/``max``, a per-channel
        ``col_lo``/``col_hi``, a ``broadcasted`` array's ``n_elements``). They
        decide what decoded value the stored ints stand for — and for
        ``broadcasted``, how many.
    ``codec_ids``
        The encode pipeline as a flat id list, at either on-disk format (see
        :func:`codec_ids`). A raw↔compressed or codec-family (blosc↔gzip) switch
        changes what the bytes at an unchanged chunk key MEAN.
    ``codecs`` — SHARDED arrays only
        The full pipeline, alongside the ids: a sharded array's top-level pipeline
        is one ``ShardingCodec``, so the inner ``chunk_shape``/``codecs``,
        ``index_codecs`` and ``index_location`` — everything deciding what a
        shard's bytes mean and where each inner chunk starts — sit nested in its
        ``configuration``, behind an id list reading only ``["sharding_indexed"]``.

    Codec SETTINGS (a blosc ``clevel``/``shuffle``/``blocksize``, a gzip
    ``level``) are left out of the ids, which only churns a digest without adding
    safety: blosc/gzip/zstd streams are SELF-DESCRIBING, so cached bytes decode
    correctly against metadata announcing different settings (measured — chunks
    encoded at ``clevel 9`` read back exactly under metadata saying 5); and the one
    Luxar codec whose settings ARE load-bearing, ``luxar_delta_v1``, derives its
    ``cols``/``bits`` from the array's column count and dtype itemsize, both
    already hashed above.
    """
    # `shards` is a plain attribute on every zarr>=3.2 Array (`ArrayV2Metadata`
    # defines it too, returning None unconditionally), so no defensive access:
    # an AttributeError here would be a real bug, not a version skew. A stable
    # `None` for an unsharded array is what makes folding it in unconditional.
    shards = dataset.shards
    identity: dict[str, Any] = {
        "name": name,
        "shape": list(dataset.shape),
        "chunks": list(dataset.chunks),
        "dtype": str(dataset.dtype),
        "shards": list(shards) if shards is not None else None,
        # One term, serialized exactly the way the group attrs are. No
        # `content_hash` to exclude here (only groups carry one), and nothing
        # writes an array attr after the stamp — finalize() back-fills GROUP
        # attrs only, all of it before the hash — so this input is stable.
        "attrs": json.dumps(dict(dataset.attrs), sort_keys=True, default=str),
        "codec_ids": codec_ids(dataset.metadata),
    }
    if shards is not None:
        # `to_dict()` output IS the JSON document zarr writes into `zarr.json`, so
        # it is JSON-ready by construction: the call site's single
        # `json.dumps(..., default=str)` needs no pre-normalization.
        identity["codecs"] = [codec.to_dict() for codec in dataset.metadata.codecs]
    return identity


def codec_ids(metadata: Any) -> list[str]:
    """The array's encode pipeline as a flat list of codec ids, at EITHER format.

    Format 3 lists its whole pipeline under one ``codecs`` member; format 2 holds
    the same information under other names, ``filters`` plus ``compressor``. The
    branch is keyed on which member exists rather than on a format number, because
    that member is exactly what each side reads.

    Order is ENCODE order, preserved and never sorted: zarr applies a format-2
    array's ``filters`` in listed order and its ``compressor`` last, which is also
    the order a format-3 ``codecs`` member stores. The two formats' lists are not
    comparable with each other (only format 3 names an array→bytes codec) and never
    need to be — a store is one format or the other.
    """
    codecs = getattr(metadata, "codecs", None)
    if codecs is not None:
        return [str(codec.to_dict()["name"]) for codec in codecs]
    ids = [str(codec.get_config()["id"]) for codec in metadata.filters or ()]
    compressor = metadata.compressor
    if compressor is not None:
        ids.append(str(compressor.get_config()["id"]))
    return ids


def compute_content_hashes(store: zarr.Group) -> str:
    """Compute content hashes for all nodes using post-order traversal.

    Called during finalize() after all nodes have been written.
    Uses xxhash64 for speed.

    Each node's hash covers, in order: every array's STORAGE IDENTITY followed by
    its decoded VALUES, then the group's own attrs, then the BYTES of any plain
    payload file those attrs name, then each child group's NAME and hash. See
    :func:`_storage_identity` for what that identity is and why layout is part of
    it, and :func:`_payload_terms` for the payload step.

    Args:
        store: Root zarr group

    Returns:
        Root content hash
    """

    def compute_hash_recursive(group_path: str) -> str:
        """Recursively compute hash for a group and its children."""
        group = store[group_path] if group_path else store

        hasher = xxhash.xxh64()

        # 1. Hash this node's own datasets (positions, colors, etc.) — storage
        #    identity first, then the decoded values.
        for dataset_name in sorted(group.array_keys()):
            dataset = group[dataset_name]
            identity = _storage_identity(dataset_name, dataset)
            hasher.update(json.dumps(identity, sort_keys=True, default=str).encode())
            hasher.update(dataset[:].tobytes())

        # 2. Hash metadata (excluding content_hash to avoid recursion)
        attrs = {k: v for k, v in dict(group.attrs).items() if k != "content_hash"}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())

        # 3. Hash plain payload files named by attrs (overlay images): neither
        #    arrays nor groups, so steps 1-2 fold in the FILENAME but never the
        #    bytes. Attrs-driven, not by directory listing. Two walks hash a
        #    store this way — this compile-time one, and `luxar optimise`'s
        #    slab-wise re-chunk walk, which reuses these very helpers over a
        #    FINISHED store. Why: `finalize/README.md`.
        for term in _payload_terms(group, attrs):
            hasher.update(term)

        # 4. Hash child groups (recursively, sorted for determinism), each one
        #    keyed by its NAME. A node's own digest does not carry its name, so
        #    hashing the digests alone left a renamed child invisible to every
        #    ancestor — the same hole `_storage_identity` closes for arrays, and
        #    a group name is a path segment, so it decides which keys the
        #    viewer's cache is holding.
        for child_name in sorted(group.group_keys()):
            child_path = f"{group_path}/{child_name}" if group_path else child_name
            child_hash = compute_hash_recursive(child_path)
            hasher.update(f"{child_name}:{child_hash}".encode())

        # Store hash in this node's attrs
        content_hash = hasher.hexdigest()
        group.attrs["content_hash"] = content_hash

        return content_hash

    # Start from root (empty path)
    root_hash = compute_hash_recursive("")
    aprint(f"Scene content hash: {root_hash[:16]}...")
    return root_hash
