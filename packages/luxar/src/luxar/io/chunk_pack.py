"""Chunk packs: one request for a small geometry node instead of one per chunk.

A laddered timelapse holds thousands of tiny nodes (a render-gate ``tp50`` part
is 4 rungs x 4 arrays, ~1 KB in total). Their cost is round trips, not bytes:
first-pass playback of ``tp50`` measured ~58 chunk requests per tick, holding it
to ~5.5 fps against a 10 fps target, and serving each part from one object
lifted the same workload to 9.2-9.5 fps.

A pack is a COPY. Every chunk object stays exactly where it is, so the store is
still an ordinary zarr store that zarr-python, napari and a viewer predating
packs read unchanged. Per packable node, the copies go into one plain file
``chunk_packs/<n>.pack`` under the root-level
:data:`~luxar.typing_utils.constants.CHUNK_PACKS_GROUP` sidecar group, laid out
as a 4-byte little-endian header length, a UTF-8 JSON header
``{"members": {chunk key relative to the node: [offset, length]}}`` (offsets
into the data after the header) and the chunk bytes back to back. The group's
attrs carry the index, kept small because it rides in the root's consolidated
metadata, which every load fetches first:

``scene_content_hash``
    The root ``content_hash`` the packs were built for. The sidecar is excluded
    from that hash (like the baked ``environment`` group), so packing leaves the
    hash alone and a reader can tell a stale pack — any later edit that restamps
    the hash — from a current one without reading a single chunk.
``packs``
    One entry per pack: ``key`` (its store key), ``sha256`` (of the whole file)
    and ``prefix`` (the packed node's path plus ``/``). Every chunk object the
    node stores is a member, so a reader knows which keys a pack can answer
    without the member list.

A node is packed whole, rungs and all, when its stored chunk bytes are at most
:data:`PACK_MAX_BYTES` — the 64 KB transfer unit the chunk policy targets, so a
pack is never a bigger object than an ordinary target chunk — and it has at
least two chunk objects (one would save nothing). Too big, it is not packed, but
its typed children (an additive ladder's rungs) may be. Containers (``type:
group``: partitions, lod groups) are never packed whole, so lod levels that are
never shown together are never fetched together.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

import numpy as np
import zarr

from .._zarr_compat import array_keys, group_keys, read_raw_bytes, write_raw_bytes
from ..typing_utils._format_contract import GEOMETRY_TYPES
from ..typing_utils.constants import CHUNK_PACKS_GROUP, TARGET_CHUNK_BYTES

#: Largest node (stored chunk bytes) packed into one object.
PACK_MAX_BYTES = TARGET_CHUNK_BYTES


def _chunk_objects(group: zarr.Group, path: str = "") -> Iterator[tuple[str, bytes]]:
    """``(key relative to group, bytes)`` for every chunk object stored below it."""
    for name in sorted(array_keys(group)):
        array = group[name]
        for coords in np.ndindex(*array.cdata_shape):
            chunk = f"{name}/{array.metadata.encode_chunk_key(coords)}"
            data = read_raw_bytes(group, chunk)
            if data is not None:  # an all-fill chunk is never written
                yield path + chunk, data
    for name in sorted(group_keys(group)):
        yield from _chunk_objects(group[name], f"{path}{name}/")


def _small_node(group: zarr.Group, max_bytes: int) -> list[tuple[str, bytes]] | None:
    """The node's chunk objects when it fits in one pack, else ``None``."""
    objects: list[tuple[str, bytes]] = []
    total = 0
    for key, data in _chunk_objects(group):
        total += len(data)
        if total > max_bytes:
            return None
        objects.append((key, data))
    return objects if len(objects) > 1 else None


def _packable(
    group: zarr.Group, path: str, max_bytes: int
) -> Iterator[tuple[str, list[tuple[str, bytes]]]]:
    """``(node path, chunk objects)`` for every node to pack, outermost first."""
    for name in sorted(group_keys(group)):
        if not path and name == CHUNK_PACKS_GROUP:
            continue
        child = group[name]
        typed = child.attrs.get("type") in GEOMETRY_TYPES
        objects = _small_node(child, max_bytes) if typed else None
        if objects is not None:
            yield f"{path}{name}", objects
        else:
            yield from _packable(child, f"{path}{name}/", max_bytes)


def pack_bytes(objects: list[tuple[str, bytes]]) -> bytes:
    """One pack file: header length, JSON member index, then the chunk bytes."""
    members: dict[str, list[int]] = {}
    offset = 0
    for key, data in objects:
        members[key] = [offset, len(data)]
        offset += len(data)
    header = json.dumps({"members": members}, separators=(",", ":")).encode()
    return (
        len(header).to_bytes(4, "little")
        + header
        + b"".join(data for _, data in objects)
    )


def write_chunk_packs(
    root: zarr.Group, scene_content_hash: str, max_bytes: int = PACK_MAX_BYTES
) -> int:
    """Write ``root``'s chunk packs for ``scene_content_hash``; return the count."""
    packs: list[dict[str, Any]] = []
    sidecar: zarr.Group | None = None
    for node_path, objects in _packable(root, "", max_bytes):
        if sidecar is None:
            sidecar = root.create_group(CHUNK_PACKS_GROUP)
        name = f"{len(packs)}.pack"
        blob = pack_bytes(objects)
        write_raw_bytes(sidecar, name, blob)
        packs.append(
            {
                "key": f"{CHUNK_PACKS_GROUP}/{name}",
                "sha256": hashlib.sha256(blob).hexdigest(),
                "prefix": f"{node_path}/",
            }
        )
    if sidecar is not None:
        sidecar.attrs.update(
            {
                "scene_content_hash": scene_content_hash,
                "max_bytes": max_bytes,
                "packs": packs,
            }
        )
    return len(packs)
