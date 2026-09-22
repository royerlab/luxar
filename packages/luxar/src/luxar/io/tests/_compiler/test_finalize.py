"""Direct unit tests for the finalize-time passes in luxar.io._compiler.finalize."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import pytest
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import MemoryStore

from luxar._zarr_compat import create_array
from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION
from luxar.io._compiler.finalize import hashing
from luxar.io._compiler.finalize.hashing import (
    _payload_terms,
    _storage_identity,
    codec_ids,
    compute_content_hashes,
)
from luxar.io._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
    warn_one_part_partition_anchors,
)
from luxar.io._compiler.finalize.validation import (
    prune_childless_wrappers,
    validate_discrete_dimension_ranges,
)
from luxar.typing_utils._format_contract import GEOMETRY_TYPES
from luxar.typing_utils.geometry_capabilities import lod_capable_types


@pytest.mark.parametrize("kind", ["partition", "lod"])
def test_prune_childless_wrappers_removes_childless_wrapper(kind: str) -> None:
    root = zarr.group()
    wrapper = root.create_group("broken")
    wrapper.attrs["kind"] = kind

    with pytest.warns(UserWarning, match=rf"childless kind={kind} wrapper at 'broken'"):
        prune_childless_wrappers(root)

    assert "broken" not in root


def test_prune_childless_wrappers_accepts_populated_wrapper() -> None:
    root = zarr.group()
    wrapper = root.create_group("valid")
    wrapper.attrs["kind"] = "partition"
    wrapper.create_group("part_0")

    prune_childless_wrappers(root)

    assert "valid" in root


def test_prune_childless_wrappers_removes_empty_wrapper_chain_post_order() -> None:
    root = zarr.group()
    lod = root.create_group("lod")
    lod.attrs["kind"] = "lod"
    partition = lod.create_group("partition")
    partition.attrs["kind"] = "partition"

    with pytest.warns(UserWarning) as warnings_seen:
        prune_childless_wrappers(root)

    assert "lod" not in root
    assert [str(item.message) for item in warnings_seen] == [
        "Pruning childless kind=partition wrapper at 'lod/partition'; "
        "it was created but never populated.",
        "Pruning childless kind=lod wrapper at 'lod'; it was created but never populated.",
    ]


def _lod_tree() -> zarr.Group:
    """Root with a kind=lod wrapper holding two gsplat leaves (no parent bounds)."""
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    coarse = lod.create_group("coarse")
    coarse.attrs["type"] = "gsplats"
    coarse.attrs["position_bounds"] = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    fine = lod.create_group("fine")
    fine.attrs["type"] = "gsplats"
    fine.attrs["position_bounds"] = {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 1.0]}
    return root


def test_finalize_lod_position_bounds_backfills_union() -> None:
    root = _lod_tree()
    assert "position_bounds" not in dict(root["lodgrp"].attrs)
    finalize_lod_position_bounds(root)
    bounds = dict(root["lodgrp"].attrs)["position_bounds"]
    assert bounds["min"] == [-1.0, 0.0, 0.0]
    assert bounds["max"] == [1.0, 2.0, 1.0]


def test_finalize_lod_position_bounds_never_overwrites() -> None:
    root = _lod_tree()
    authored = {"min": [-9.0, -9.0, -9.0], "max": [9.0, 9.0, 9.0]}
    root["lodgrp"].attrs["position_bounds"] = authored
    finalize_lod_position_bounds(root)
    assert dict(root["lodgrp"].attrs)["position_bounds"] == authored


def _partition_tree() -> zarr.Group:
    """Root with a kind=partition wrapper holding two gsplat-leaf parts (no parent
    bounds) — the shape a grafted partition (add_gsplats_from_file) produces, since
    add_partition_group does not compute the children-union the standalone writer
    stamps at write time."""
    root = zarr.group()
    part = root.create_group("partgrp")
    part.attrs["kind"] = "partition"
    p0 = part.create_group("part_0")
    p0.attrs["type"] = "gsplats"
    p0.attrs["position_bounds"] = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    p1 = part.create_group("part_1")
    p1.attrs["type"] = "gsplats"
    p1.attrs["position_bounds"] = {"min": [5.0, -2.0, 0.0], "max": [6.0, 1.0, 3.0]}
    return root


def test_finalize_position_bounds_backfills_partition_wrapper() -> None:
    """A grafted kind=partition wrapper missing position_bounds must be back-filled
    with the union of its parts (regression: graft dropped wrapper bounds, losing
    partition-unit culling / graft-vs-standalone parity)."""
    root = _partition_tree()
    assert "position_bounds" not in dict(root["partgrp"].attrs)
    finalize_lod_position_bounds(root)
    bounds = dict(root["partgrp"].attrs)["position_bounds"]
    assert bounds["min"] == [0.0, -2.0, 0.0]
    assert bounds["max"] == [6.0, 1.0, 3.0]


def test_finalize_position_bounds_partition_never_overwrites() -> None:
    root = _partition_tree()
    authored = {"min": [-9.0, -9.0, -9.0], "max": [9.0, 9.0, 9.0]}
    root["partgrp"].attrs["position_bounds"] = authored
    finalize_lod_position_bounds(root)
    assert dict(root["partgrp"].attrs)["position_bounds"] == authored


def test_finalize_lod_display_types_resolves_from_finest() -> None:
    root = _lod_tree()
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "gsplats"


def test_finalize_lod_display_types_uses_child_index_not_name_order() -> None:
    """The finest child is resolved by ``child_index`` (coarsest→finest), not by
    alphabetical name order. With >=10 children, name order puts ``child_10``
    before ``child_2`` — so a name-sorted "last" child is NOT the finest. Mixing
    leaf types across levels makes the mis-resolution observable: pre-fix the
    name-sorted last child (child_9, a ``points`` leaf) wins; post-fix the
    highest-child_index child (child_11, the actual finest ``gsplats`` leaf) wins.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    # 12 children, child_index 0..11 (coarsest→finest). All but the true finest
    # are 'points'; the finest (child_index 11) is 'gsplats'. Name-sorting yields
    # child_9 last (a 'points' leaf) — the wrong answer.
    for i in range(12):
        c = lod.create_group(f"child_{i}")
        c.attrs["type"] = "gsplats" if i == 11 else "points"
        c.attrs["child_index"] = i
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "gsplats"


def test_finalize_lod_display_types_falls_back_to_name_order_without_child_index() -> (
    None
):
    """Legacy trees with no ``child_index`` keep the historical name-order
    behaviour (finest = last by name) so nothing regresses for old data."""
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    for i in range(3):  # child_0..child_2, no child_index
        c = lod.create_group(f"child_{i}")
        c.attrs["type"] = "lines" if i == 2 else "points"
    finalize_lod_display_types(root)
    assert dict(root["lodgrp"].attrs)["display_type"] == "lines"


def test_compute_content_hashes_is_deterministic_and_stamps_attrs() -> None:
    root = _lod_tree()
    h1 = compute_content_hashes(root)
    assert isinstance(h1, str) and len(h1) > 0
    assert dict(root.attrs)["content_hash"] == h1
    # Recompute on an identical fresh tree → same root hash (determinism).
    h2 = compute_content_hashes(_lod_tree())
    assert h1 == h2


def test_compute_content_hashes_changes_with_data() -> None:
    root = _lod_tree()
    create_array(
        root["lodgrp"]["fine"], "centers", data=np.ones((3, 3), dtype=np.float32)
    )
    h_with = compute_content_hashes(root)
    assert h_with != compute_content_hashes(_lod_tree())


# A minimal valid PNG (1x1 red pixel) — no PIL needed.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _overlay_image_store(
    root_path: Path,
    payload: bytes | None,
    filename: str = "image.png",
    zarr_format: int = 3,
) -> zarr.Group:
    """Real on-disk store with one `overlays/logo` image-overlay group.

    ``payload`` is written as a plain file inside the group's directory (exactly
    what ``write_overlay`` does); ``None`` writes no file at all.

    ``zarr_format`` is pinned explicitly because which metadata DOCUMENT exists
    on disk depends on it — `.zattrs` at format 2, `zarr.json` at format 3 — and
    a case naming a document the store does not have tests nothing. Explicit also
    keeps the cases below independent of `LUXAR_ZARR_FORMAT` (which raw
    `zarr.open_group` never consulted anyway).
    """
    root = zarr.open_group(str(root_path), mode="w", zarr_format=zarr_format)
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["type"] = "overlay_image"
    logo.attrs["image_file"] = filename
    if payload is not None:
        image_dir = root_path / "overlays" / "logo"
        image_dir.mkdir(parents=True, exist_ok=True)
        (image_dir / filename).write_bytes(payload)
    return root


def test_compute_content_hashes_covers_overlay_payload_bytes(tmp_path: Path) -> None:
    """Issue #1720: an overlay PNG is neither an array nor a group, so its bytes
    used to sit outside the hash entirely — two stores differing only in the
    image stamped one digest, and `content_hash` is supposed to fingerprint the
    content it covers."""
    store_path = tmp_path / "scene.luxar.zarr"
    root = _overlay_image_store(store_path, _TINY_PNG)
    h_before = compute_content_hashes(root)

    # Recomputing over unchanged bytes reproduces the digest (the previous stamp
    # is excluded from the walk by design).
    assert compute_content_hashes(root) == h_before

    # Overwrite the payload at the SAME filename, touching nothing else. The
    # edit is SAME-LENGTH so this pins the bytes: a longer payload would still
    # fail on the length prefix alone, with the bytes never folded in.
    (store_path / "overlays" / "logo" / "image.png").write_bytes(
        _TINY_PNG[:-1] + b"\x83"
    )
    assert compute_content_hashes(root) != h_before


def test_compute_content_hashes_absent_payload_differs_from_empty(
    tmp_path: Path,
) -> None:
    """A missing named file must not hash the same as a zero-byte one."""
    h_absent = compute_content_hashes(_overlay_image_store(tmp_path / "a", None))
    h_empty = compute_content_hashes(_overlay_image_store(tmp_path / "b", b""))
    assert h_absent != h_empty


def test_case_shifted_metadata_payload_uses_case_exact_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dangling ``Zarr.json`` must stay absent even when the store lookup
    case-folds it onto the group's own ``zarr.json`` metadata document."""
    root = _overlay_image_store(
        tmp_path / "scene", None, filename="Zarr.json", zarr_format=3
    )
    expected = compute_content_hashes(root)
    logo = root["overlays/logo"]

    original_read = hashing.read_raw_bytes
    folded_reads = 0

    def _case_folding_read(group: zarr.Group, key: str) -> bytes | None:
        nonlocal folded_reads
        if group.path == logo.path and key == "Zarr.json":
            folded_reads += 1
            return original_read(group, "zarr.json")
        return original_read(group, key)

    monkeypatch.setattr(hashing, "read_raw_bytes", _case_folding_read)

    assert compute_content_hashes(root) == expected
    assert compute_content_hashes(root) == expected
    assert folded_reads == 0


@pytest.mark.filterwarnings(
    "ignore:Object at Zarr.json is not recognized.*:zarr.errors.ZarrUserWarning"
)
def test_case_shifted_metadata_payload_is_hashed_when_present_exactly() -> None:
    """A real case-shifted payload remains valid on stores that can hold it."""
    root = zarr.group(zarr_format=3)  # MemoryStore is case-sensitive everywhere.
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["image_file"] = "Zarr.json"

    def _write(payload: bytes) -> None:
        buffer = default_buffer_prototype().buffer.from_bytes(payload)
        sync((logo.store_path / "Zarr.json").set(buffer))

    _write(_TINY_PNG)
    h_before = compute_content_hashes(root)
    assert compute_content_hashes(root) == h_before
    _write(_TINY_PNG[:-1] + b"\x83")
    assert compute_content_hashes(root) != h_before


def test_case_shifted_metadata_payload_does_not_fall_back_when_listing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store without listing support must not fall back to the unsafe read."""

    class NoListingStore(MemoryStore):
        @property
        def supports_listing(self) -> bool:
            return False

        async def list_dir(self, prefix: str) -> AsyncIterator[str]:
            if False:
                yield prefix

    root = zarr.group(store=NoListingStore(), zarr_format=3)
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["image_file"] = "Zarr.json"
    payload = default_buffer_prototype().buffer.from_bytes(_TINY_PNG)
    sync((logo.store_path / "Zarr.json").set(payload))
    assert hashing.read_raw_bytes(logo, "Zarr.json") == _TINY_PNG

    def _must_not_read(_group: zarr.Group, _key: str) -> bytes | None:
        raise AssertionError("case-shifted metadata name reached raw lookup")

    monkeypatch.setattr(hashing, "read_raw_bytes", _must_not_read)

    terms = b"".join(_payload_terms(logo, dict(logo.attrs)))
    assert b"unreadable:" in terms


def test_case_shifted_metadata_payload_handles_listing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing supported listing must not abort or fall back to a raw read."""

    class FailingListingStore(MemoryStore):
        async def list_dir(self, prefix: str) -> AsyncIterator[str]:
            if False:
                yield prefix
            raise OSError("listing failed")

    root = zarr.group(store=FailingListingStore(), zarr_format=3)
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["image_file"] = "Zarr.json"

    def _must_not_read(_group: zarr.Group, _key: str) -> bytes | None:
        raise AssertionError("case-shifted metadata name reached raw lookup")

    monkeypatch.setattr(hashing, "read_raw_bytes", _must_not_read)

    terms = b"".join(_payload_terms(logo, dict(logo.attrs)))
    assert b"unreadable:" in terms


def test_compute_content_hashes_payload_free_digest_is_unchanged() -> None:
    """A tree with no payload attrs must hash byte-identically to what the
    pre-payload implementation produced — the payload step only ever adds terms
    for groups that actually name a payload file.

    The digest below was captured by running the implementation with the payload
    step commented out, over the tree built right here. The tree is local on
    purpose: pinning a digest over a helper shared with other tests would turn
    any legitimate edit to that helper into a failure of this tripwire. It must
    only ever change deliberately — a moving digest means the hash inputs
    changed, and every already-cached dataset re-downloads once.

    Two things are pinned so the digest is a function of the tree alone.
    ``zarr_format=3`` because ``_storage_identity`` folds in ``codec_ids``, which
    is derived from ``codecs`` at format 3 and from ``filters``/``compressor`` at
    format 2 — an ambient ``LUXAR_ZARR_FORMAT=2`` would otherwise move the digest
    and read as a regression. And ``compressor=None`` (RAW) rather than the
    ``"auto"`` default, since ``"auto"`` is whatever codec the installed zarr
    happens to pick, which would make a zarr upgrade look like a Luxar-side
    change here.
    """
    root = zarr.group(zarr_format=3)
    root.attrs["type"] = "scene"
    child = root.create_group("pts")
    child.attrs["type"] = "points"
    child.attrs["visible"] = True
    create_array(
        child,
        "positions",
        compressor=None,
        data=np.arange(6, dtype=np.float32).reshape(3, 2),
    )
    assert "image_file" not in dict(child.attrs)  # really the payload-free path
    assert compute_content_hashes(root) == "8f350b91c3848005"


def test_payload_terms_are_prefix_free(tmp_path: Path) -> None:
    """The payload block must be injective on its own.

    This is the collision `_payload_terms` documents: without the key and name
    length prefixes, both stores below fold the identical ``image_filea7:5:hello``.
    Both names are readable, so no sentinel is involved either way.
    """
    a = _overlay_image_store(tmp_path / "a", b"5:hello", filename="a")
    b = _overlay_image_store(tmp_path / "b", b"hello", filename="a7:")

    def terms(root: zarr.Group) -> bytes:
        group = root["overlays/logo"]
        return b"".join(_payload_terms(group, dict(group.attrs)))

    assert terms(a) != terms(b)
    # The framing itself, spelled out: length, key, length, name, length, bytes.
    assert terms(a) == b"payload:10:image_file1:a7:5:hello"
    assert terms(b) == b"payload:10:image_file3:a7:5:hello"


@pytest.mark.parametrize(
    ("filename", "zarr_format", "sentinel"),
    [
        # Rejected by NAME (semantic): not one path component, or a document
        # this very walk stamps.
        ("../../../evil.png", 3, b"unsafe:"),
        ("sub/evil.png", 3, b"unsafe:"),
        ("sub\\evil.png", 3, b"unsafe:"),  # normalize_path turns `\` into `/`
        (".", 3, b"unsafe:"),
        ("..", 3, b"unsafe:"),
        ("zarr.json", 3, b"unsafe:"),  # the v3 doc — exists only at format 3
        (".zattrs", 2, b"unsafe:"),  # the v2 doc — exists only at format 2
        # Rejected by the STORE (mechanical): `LocalStore.get` catches only the
        # not-found family, so these reach the payload step as exceptions.
        ("x" * 300 + ".png", 3, b"unreadable:"),  # OSError(ENAMETOOLONG)
        ("a\x00b.png", 3, b"unreadable:"),  # ValueError: embedded null byte
        # A lone surrogate round-trips zarr attrs (`json.dumps` escapes it), so
        # the walk really does meet such a name; encoding it must not raise, and
        # the store's own UnicodeEncodeError is a ValueError.
        ("im\ud800age.png", 3, b"unreadable:"),
        # Absent by the case-exact listing: these names would otherwise resolve
        # onto zarr's own metadata document on a case-insensitive filesystem.
        ("Zarr.json", 3, b"absent:"),
        (".ZAttrs", 2, b"absent:"),
    ],
)
def test_compute_content_hashes_tolerates_unreadable_payload_names(
    tmp_path: Path, filename: str, zarr_format: int, sentinel: bytes
) -> None:
    """A payload name the walk cannot read must fold in deterministically.

    Pins only what the payload step owns: no raise (one escaping would reach
    ``finalize()``, which stamps the store ``incomplete``), a convergent digest,
    and which of the three sentinels was folded — ``unsafe:`` for a name refused
    semantically, ``unreadable:`` for the store's own verdict, or ``absent:``
    when the case-exact listing does not contain a metadata-colliding name. That
    a change to the *name* moves the hash comes from step 2, which hashes the
    attrs JSON, not from here — do not read this test as covering it.
    """
    root = _overlay_image_store(
        tmp_path / "scene", None, filename=filename, zarr_format=zarr_format
    )

    first = compute_content_hashes(root)  # must not raise
    assert compute_content_hashes(root) == first

    group = root["overlays/logo"]
    assert sentinel in b"".join(_payload_terms(group, dict(group.attrs)))


def test_compute_content_hashes_reads_the_payload_through_the_store() -> None:
    """The read goes through the store, not the filesystem — so it works on a
    store with no filesystem at all."""
    root = zarr.group()  # MemoryStore
    logo = root.create_group("overlays").create_group("logo")
    logo.attrs["image_file"] = "image.png"

    def _write(payload: bytes) -> None:
        buffer = default_buffer_prototype().buffer.from_bytes(payload)
        sync((logo.store_path / "image.png").set(buffer))

    _write(_TINY_PNG)
    h_before = compute_content_hashes(root)
    # A same-length edit, so this pins the BYTES and not just the length prefix.
    _write(_TINY_PNG[:-1] + b"\x83")
    assert compute_content_hashes(root) != h_before


def test_compiled_scene_hash_tracks_overlay_image_bytes(tmp_path: Path) -> None:
    """End-to-end: two scenes compiled from the same script and differing only
    in their overlay PNG get different root content hashes."""
    from luxar import Dimensions, LuxarZarrCompiler

    def _compile(path: Path, payload: bytes) -> str:
        with LuxarZarrCompiler(path) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", positions=np.zeros((2, 3), dtype=np.float32))
            scene.add_image(payload, position=(0.9, 0.05))
        return str(zarr.open_group(str(path), mode="r").attrs["content_hash"])

    # Same-length payloads, so this pins the bytes end to end: a length-changing
    # edit would separate the two digests on the length prefix alone.
    h_a = _compile(tmp_path / "a.luxar.zarr", _TINY_PNG)
    h_b = _compile(tmp_path / "b.luxar.zarr", _TINY_PNG[:-1] + b"\x83")
    assert h_a != h_b


# ────────────────────────────────────────────────────────────────────────
# content_hash covers STORAGE IDENTITY, not just values (#1718)
#
# The hash is the token the viewer's MultiLevelCachingStore compares against the
# remote, and that cache holds ENCODED CHUNKS KEYED BY CHUNK INDEX. So what a
# chunk index MEANS — chunk/shard shape, codec ids, array name and shape — plus
# what the bytes it returns DECODE to (the array's own `encoding` attrs) is part
# of what makes two stores interchangeable. See `_storage_identity`.
# ────────────────────────────────────────────────────────────────────────


def _leaf_store(
    *,
    name: str = "positions",
    shape: tuple[int, ...] = (12, 2),
    chunks: tuple[int, ...] = (12, 2),
    shards: tuple[int, ...] | None = None,
    compressor: Any = None,
    filters: list[Any] | None = None,
    fill: float = 1.0,
    array_attrs: dict[str, Any] | None = None,
    zarr_format: int = 3,
    group_name: str = "leaf",
) -> zarr.Group:
    """Root holding one points leaf whose single array is fully caller-specified."""
    root = zarr.group(zarr_format=zarr_format)
    leaf = root.create_group(group_name)
    leaf.attrs["type"] = "points"
    data = np.full(shape, fill, dtype=np.float32)
    extra: dict[str, Any] = {} if shards is None else {"shards": shards}
    if filters is not None:
        extra["filters"] = filters
    array = create_array(
        leaf, name, data=data, chunks=chunks, compressor=compressor, **extra
    )
    if array_attrs is not None:
        array.attrs.update(array_attrs)
    return root


# Luxar reads both on-disk formats and writes 3 by default, so the layout fix has
# to hold for a store of either vintage — the two carry chunk shape in entirely
# different metadata documents (`.zarray` vs `zarr.json`).
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_compute_content_hashes_changes_with_chunk_layout(zarr_format: int) -> None:
    """Same VALUES, different ``chunks`` ⇒ different hash.

    The bug this pins: two stores holding byte-identical values under different
    chunk shapes hashed EQUAL, so a viewer holding a cached copy of one would
    serve its chunks for the other — and a chunk index addresses different data
    in each. A change to the chunk shapes the compiler emits, or any re-chunking
    pass over an existing store, made every output a hash collision with its input.
    """
    whole = compute_content_hashes(_leaf_store(chunks=(12, 2), zarr_format=zarr_format))
    split = compute_content_hashes(_leaf_store(chunks=(3, 2), zarr_format=zarr_format))
    assert whole != split


def test_compute_content_hashes_changes_with_shard_shape() -> None:
    """Same values and same inner ``chunks``, different SHARD shape ⇒ different hash.

    A shard is what a format-3 chunk KEY actually addresses when sharding is on,
    so two stores whose shards group the same inner chunks differently are not
    interchangeable in the viewer's cache.
    """
    small = compute_content_hashes(_leaf_store(chunks=(3, 2), shards=(6, 2)))
    large = compute_content_hashes(_leaf_store(chunks=(3, 2), shards=(12, 2)))
    assert small != large


def test_compute_content_hashes_changes_with_sharded_inner_codec() -> None:
    """A SHARDED array's inner codec pipeline is part of its identity.

    For a sharded array the top-level pipeline is always exactly one
    ``ShardingCodec``, and the inner ``codecs`` / ``index_codecs`` /
    ``chunk_shape`` / ``index_location`` — everything that decides what a shard's
    bytes mean, including where each inner chunk starts — live NESTED inside its
    ``configuration``. Hashing only the top-level codec NAMES therefore recorded
    the constant ``"sharding_indexed"`` and nothing else. Nothing in Luxar emits
    a sharded store today, so this test is the only thing holding that fold
    honest.
    """
    raw = _leaf_store(chunks=(3, 2), shards=(6, 2), compressor=None)
    zstd = _leaf_store(
        chunks=(3, 2),
        shards=(6, 2),
        compressor=numcodecs.Blosc(cname="zstd", clevel=9),
    )
    # Same shape/chunks/shards — the inner compressor is the only difference.
    raw_arr, zstd_arr = raw["leaf/positions"], zstd["leaf/positions"]
    assert (raw_arr.shape, raw_arr.chunks, raw_arr.shards) == (
        zstd_arr.shape,
        zstd_arr.chunks,
        zstd_arr.shards,
    )
    assert _storage_identity("positions", raw_arr) != _storage_identity(
        "positions", zstd_arr
    )
    assert compute_content_hashes(raw) != compute_content_hashes(zstd)


# Both formats, because the SAME hole exists in both and the two carry the
# pipeline under entirely different metadata members: format 3 lists it under
# `codecs`, format 2 splits it across `filters` + `compressor`. Luxar writes
# format 2 on demand (`LUXAR_ZARR_FORMAT=2`) and re-finalizes legacy v2 stores.
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_compute_content_hashes_changes_with_unsharded_codec_family(
    zarr_format: int,
) -> None:
    """An UNSHARDED array's codec IDS are part of its identity, at either format.

    Everything else is held fixed (values, shape, chunks, dtype, name), so the
    only difference is what the stored bytes at a chunk key MEAN: raw vs
    blosc-compressed vs gzip. All three used to hash EQUAL — raw↔compressed and
    blosc↔gzip were invisible even though the encoded bytes a cache holds are
    completely different.
    """
    hashes = {
        label: compute_content_hashes(
            _leaf_store(compressor=compressor, zarr_format=zarr_format)
        )
        for label, compressor in (
            ("raw", None),
            ("blosc", numcodecs.Blosc(cname="zstd", clevel=9)),
            ("gzip", numcodecs.GZip(level=5)),
        )
    }
    assert len(set(hashes.values())) == 3, hashes


def test_compute_content_hashes_changes_with_format_2_filters() -> None:
    """A format-2 array's ``filters`` are part of its codec ids, in encode order.

    Format 2 keeps its pipeline in two members rather than one, and a filter is
    every bit as decisive as the compressor about what the stored bytes mean — a
    delta-encoded chunk read as plain codes is garbage.
    """
    plain = _leaf_store(compressor=numcodecs.GZip(level=5), zarr_format=2)
    delta = _leaf_store(
        compressor=numcodecs.GZip(level=5),
        filters=[numcodecs.Delta(dtype="float32")],
        zarr_format=2,
    )
    # Encode order: filters in listed order, compressor last — the order zarr
    # itself applies them, and the order format 3 stores in its `codecs` member.
    assert codec_ids(plain["leaf/positions"].metadata) == ["gzip"]
    assert codec_ids(delta["leaf/positions"].metadata) == ["delta", "gzip"]
    assert compute_content_hashes(plain) != compute_content_hashes(delta)


def test_compute_content_hashes_ignores_unsharded_codec_settings() -> None:
    """An UNSHARDED array's codec SETTINGS are deliberately NOT hashed.

    This pins a trade-off, not an ideal: only the codec ids are folded in, so a
    pure tuning change (here a blosc ``clevel``) leaves the hash where it was
    rather than churning every store's digest. The residual exposure is documented
    on :func:`_storage_identity`. If a future change starts hashing settings, this
    test is what says so out loud instead of letting every dataset silently
    re-download.
    """
    nine = _leaf_store(compressor=numcodecs.Blosc(cname="zstd", clevel=9))
    five = _leaf_store(compressor=numcodecs.Blosc(cname="zstd", clevel=5))
    nine_arr, five_arr = nine["leaf/positions"], five["leaf/positions"]
    # The stores really do differ in their stored codec configuration...
    assert [c.to_dict() for c in nine_arr.metadata.codecs] != [
        c.to_dict() for c in five_arr.metadata.codecs
    ]
    # ...and the identity really is reduced to the ids, so nothing differs there.
    assert _storage_identity("positions", nine_arr) == _storage_identity(
        "positions", five_arr
    )
    assert compute_content_hashes(nine) == compute_content_hashes(five)


def test_compute_content_hashes_changes_with_array_name() -> None:
    """A renamed array is different content, at identical values and layout."""
    a = compute_content_hashes(_leaf_store(name="positions"))
    b = compute_content_hashes(_leaf_store(name="radii"))
    assert a != b


def test_compute_content_hashes_changes_with_group_name() -> None:
    """A renamed CHILD GROUP is different content, at identical contents.

    A node's own digest does not carry its name, and the parent folded in only
    the digests of its children — so renaming a node while leaving everything
    under it alone left the root hash EQUAL to the original's. A group name is a
    path segment, so the rename also moves every key the viewer's cache holds for
    that subtree, and the stale root document it keeps serving still enumerates
    the old names.
    """
    cells = compute_content_hashes(_leaf_store(group_name="cells"))
    nuclei = compute_content_hashes(_leaf_store(group_name="nuclei"))
    assert cells != nuclei


def test_compute_content_hashes_changes_with_per_array_attrs() -> None:
    """An array's OWN attrs are identity — that is where DEQUANTIZATION lives.

    Luxar writes each array's `encoding` document into its own attrs, so those
    attrs decide what decoded value the stored ints stand for. A changed
    quantization `min` shifts every decoded float in the array while its bytes,
    shape, chunks, dtype, codecs and the group's attrs all stay put — and used to
    move neither digest, so a viewer would keep serving the old scaling forever.
    """
    lo = _leaf_store(
        array_attrs={
            "encoding": {"name": "bounded_scalar_uint8", "min": 0.0, "bits": 8}
        }
    )
    hi = _leaf_store(
        array_attrs={
            "encoding": {"name": "bounded_scalar_uint8", "min": 0.25, "bits": 8}
        }
    )
    assert compute_content_hashes(lo) != compute_content_hashes(hi)


def test_compute_content_hashes_canonicalizes_ulp_perturbed_float_attrs() -> None:
    """Platform-last-bit drift in hashed metadata must not change store identity."""
    camera_x = 1.23456789012345
    certificate = 8.146910239026359e-08

    def stamped(direction: float) -> tuple[str, dict[str, Any], dict[str, Any]]:
        root = _leaf_store(
            array_attrs={
                "encoding": {
                    "name": "perchannel_log_uint8",
                    "certificate": {
                        "metric": "rel_frobenius_p95",
                        "value": np.nextafter(certificate, direction),
                    },
                }
            }
        )
        root.attrs["viewer_config"] = {
            "camera": {"position": [np.nextafter(camera_x, direction), 2.0, 3.0]}
        }
        content_hash = compute_content_hashes(root)
        return (
            content_hash,
            dict(root.attrs)["viewer_config"],
            dict(root["leaf/positions"].attrs)["encoding"],
        )

    lower = stamped(-np.inf)
    upper = stamped(np.inf)

    assert lower == upper
    assert lower[1]["camera"]["position"][0] == float(f"{camera_x:.12g}")
    assert lower[2]["certificate"]["value"] == float(f"{certificate:.12g}")


def test_compute_content_hashes_changes_with_shape_at_identical_bytes() -> None:
    """A reshape is visible even though ``tobytes()`` is identical.

    ``np.full((2, 3))`` and ``np.full((3, 2))`` serialize to the same bytes, so
    hashing values alone cannot tell a transposed/reshaped store from its input.
    Chunks are pinned to a shape both arrays accept so SHAPE is the only variable.
    """
    tall = compute_content_hashes(_leaf_store(shape=(2, 3), chunks=(1, 1)))
    wide = compute_content_hashes(_leaf_store(shape=(3, 2), chunks=(1, 1)))
    assert tall != wide


def test_compute_content_hashes_still_changes_with_values_at_fixed_layout() -> None:
    """Guard against over-fitting the layout fix: identical layout, different
    values must still hash differently."""
    ones = compute_content_hashes(_leaf_store(fill=1.0))
    twos = compute_content_hashes(_leaf_store(fill=2.0))
    assert ones != twos


def test_compute_content_hashes_is_stable_for_identical_stores_with_arrays() -> None:
    """Determinism with arrays present: hashing the same store twice, and hashing
    an independently-built identical store, both give the same hash."""
    root = _leaf_store()
    first = compute_content_hashes(root)
    assert compute_content_hashes(root) == first
    assert compute_content_hashes(_leaf_store()) == first


def test_validate_discrete_dimension_ranges_noop_without_bounds() -> None:
    # No scene_bounds → returns immediately without raising.
    validate_discrete_dimension_ranges(zarr.group(), None)


def _store_with_discrete_time(range_: tuple[float, float]) -> zarr.Group:
    """Root with a 4D scene: [time (discrete, hidden, step=1), z, y, x]."""
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(
                name="time", range=range_, step=1.0, display=False, discrete=True
            ),
            Dimension(name="z", range=(0, 10)),
            Dimension(name="y", range=(0, 10)),
            Dimension(name="x", range=(0, 10)),
        ]
    )
    root = zarr.group()
    root.attrs["scene_dimensions"] = dims.to_dict()
    return root


def test_validate_discrete_ranges_tolerance_matches_viewer_quarter_step() -> None:
    """The misalignment warning threshold mirrors the viewer's discrete query
    tolerance (0.25 × step — see DISCRETE_TOLERANCE_FRACTION in
    tolerance-computer.ts), NOT the legacy 0.5 × step. Data starting 0.3 below
    the declared range min is inside the legacy half-step tolerance (no warning
    pre-fix) but outside the quarter-step one, and a viewer initialized at the
    range min would show no data — so it MUST warn."""
    import warnings as _warnings

    bounds = {"min": [1.3, 0.0, 0.0, 0.0], "max": [5.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.0, 5.0))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    assert any("range starting at" in str(w.message) for w in caught), (
        f"Expected misalignment warning for 0.3-step offset, got: {[str(w.message) for w in caught]}"
    )


def test_validate_discrete_ranges_warns_on_off_grid_data() -> None:
    """Discrete data more than a quarter-step off the min+k*step grid can
    pass the viewer's half-step visibility gate while its (epsilon-padded)
    chunks are never fetched by the quarter-step query — it would silently not
    display. The declared range matches the data exactly here, so the
    range-edge checks stay silent; only the on-grid check must fire."""
    import warnings as _warnings

    bounds = {"min": [1.3, 0.0, 0.0, 0.0], "max": [5.6, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.3, 5.6))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    messages = [str(w.message) for w in caught]
    assert not any("range starting at" in m for m in messages), messages
    assert any("off the step grid" in m for m in messages), (
        f"Expected off-grid warning for 0.3 grid offset, got: {messages}"
    )


def test_validate_discrete_ranges_accepts_grid_anchored_at_declared_min() -> None:
    """Offset coordinates are on-grid when measured from the range minimum."""
    import warnings as _warnings

    bounds = {"min": [3.0, 0.0, 0.0, 0.0], "max": [13.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(3.0, 13.0))
    scene_dims = store.attrs["scene_dimensions"]
    scene_dims["dimensions"][0]["step"] = 5.0
    store.attrs["scene_dimensions"] = scene_dims

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)

    assert not caught, [str(w.message) for w in caught]


def test_validate_discrete_ranges_warns_on_off_grid_data_without_step() -> None:
    """Regression (deep-double-check): a discrete dimension WITHOUT a declared
    step is not exempt from the on-grid check — the viewer defaults a missing
    step to 1.0 and anchors it at the declared minimum. Pre-fix the check was gated on
    `if dim.step:` and this compiled warning-free."""
    import warnings as _warnings

    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(name="time", range=(0.5, 4.8), display=False, discrete=True),
            Dimension(name="z", range=(0, 10)),
            Dimension(name="y", range=(0, 10)),
            Dimension(name="x", range=(0, 10)),
        ]
    )
    root = zarr.group()
    root.attrs["scene_dimensions"] = dims.to_dict()

    bounds = {"min": [0.5, 0.0, 0.0, 0.0], "max": [4.8, 10.0, 10.0, 10.0]}
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(root, bounds)
    messages = [str(w.message) for w in caught]
    assert any("off the step grid" in m for m in messages), (
        f"Expected off-grid warning for step-less discrete data, got: {messages}"
    )
    assert any("no step is declared" in m for m in messages), messages


def test_validate_discrete_ranges_on_grid_data_is_silent() -> None:
    """Exactly on-grid data (and data within a quarter-step of the grid)
    triggers no off-grid warning."""
    import warnings as _warnings

    bounds = {"min": [1.0, 0.0, 0.0, 0.0], "max": [5.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.0, 5.0))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    assert not caught, [str(w.message) for w in caught]


def test_validate_discrete_ranges_within_quarter_step_is_silent() -> None:
    """A range min within the quarter-step tolerance of the data start is
    reachable by the viewer's on-grid query — no warning."""
    import warnings as _warnings

    bounds = {"min": [1.2, 0.0, 0.0, 0.0], "max": [5.0, 10.0, 10.0, 10.0]}
    store = _store_with_discrete_time(range_=(1.0, 5.0))
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        validate_discrete_dimension_ranges(store, bounds)
    assert not any("range starting at" in str(w.message) for w in caught), (
        f"No warning expected within quarter-step tolerance, got: {[str(w.message) for w in caught]}"
    )


def _lod_tree_with_unrecognized_leaf() -> zarr.Group:
    """kind=lod wrapper (no display_type) over a leaf whose ``type`` is unknown.

    ``resolve()`` early-returns only for the geometry types it hardcodes. Any
    other leaf falls through to the "recurse into the finest child" branch — and
    a LEAF group's ``keys()`` lists its ARRAYS, not sub-groups.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    leaf = lod.create_group("leaf")
    leaf.attrs["type"] = "some_future_type"
    leaf.attrs["child_index"] = 0
    create_array(leaf, "vertices", data=np.zeros((3, 3), dtype=np.float32))
    create_array(leaf, "faces", data=np.zeros((1, 3), dtype=np.uint32))
    return root


def test_display_type_backfill_survives_unrecognized_leaf_type() -> None:
    """An unknown leaf ``type`` must not crash the finalize pass.

    Regression guard: ``resolve()`` used to pick the alphabetically-last ARRAY
    ("vertices") as the "finest child" and recurse into it, then call ``.keys()``
    on a ``zarr.Array`` — which does not have it — raising a bare AttributeError
    from deep inside finalize. Resolution should simply yield nothing.
    """
    root = _lod_tree_with_unrecognized_leaf()

    finalize_lod_display_types(root)  # must not raise

    # Nothing resolvable ⇒ no display_type is invented for the wrapper.
    assert "display_type" not in dict(root["lodgrp"].attrs)


def test_display_type_backfill_ignores_arrays_when_picking_finest_child() -> None:
    """Array siblings must never shadow a real child GROUP.

    A plain wrapper holding both arrays and a genuine geometry sub-group must
    resolve through the sub-group. Sorting by name alone would pick "zz_data".
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    wrapper = lod.create_group("wrapper")
    wrapper.attrs["type"] = "group"
    create_array(wrapper, "zz_data", data=np.zeros((2, 2), dtype=np.float32))
    leaf = wrapper.create_group("aa_leaf")
    leaf.attrs["type"] = "lines"

    finalize_lod_display_types(root)

    assert dict(root["lodgrp"].attrs).get("display_type") == "lines"


def test_display_type_backfill_resolves_every_lod_capable_geometry_type() -> None:
    """Leaf resolution is driven by the contract, not a local literal.

    Pins the single-sourcing: an LOD-capable geometry type added to
    ``contract.yaml`` is resolvable here without editing this module.

    Parametrised over ``lod_capable_types()`` rather than ``GEOMETRY_TYPES``,
    because the two are no longer the same set. Being a geometry leaf (the
    vocabulary) and being usable as a ``kind=lod`` group's ``display_type`` (a
    capability) are different questions, and a type can answer yes to the first
    and no to the second — see the rejection test below for the other half.
    """
    for geometry_type in lod_capable_types():
        root = zarr.group()
        lod = root.create_group("lodgrp")
        lod.attrs["kind"] = "lod"
        leaf = lod.create_group("leaf")
        leaf.attrs["type"] = geometry_type

        finalize_lod_display_types(root)

        assert dict(root["lodgrp"].attrs).get("display_type") == geometry_type


def test_display_type_backfill_rejects_every_lod_incapable_geometry_type() -> None:
    """A geometry type with no LOD ladder must not be back-filled onto a lod group.

    The other half of the test above, and the reason the guard exists: without it
    the back-fill happily stamps ``display_type='mesh'``, producing a ``kind=lod``
    group that no viewer path can load — written with no error at all. Failing
    here instead means the broken store is never produced.

    Parametrised over the complement so it covers a fifth type automatically, and
    skips cleanly (rather than passing vacuously) if every geometry type ever
    becomes LOD-capable.
    """
    incapable = [t for t in GEOMETRY_TYPES if t not in lod_capable_types()]
    if not incapable:
        pytest.skip("every contract geometry type is LOD-capable")

    for geometry_type in incapable:
        root = zarr.group()
        lod = root.create_group("lodgrp")
        lod.attrs["kind"] = "lod"
        leaf = lod.create_group("leaf")
        leaf.attrs["type"] = geometry_type

        with pytest.raises(ValueError, match="display_type for a kind=lod group"):
            finalize_lod_display_types(root)

        # And nothing was stamped on the way out.
        assert "display_type" not in dict(root["lodgrp"].attrs)


def test_display_type_backfill_ignores_a_typed_array_sibling() -> None:
    """A zarr ARRAY carrying a ``type`` attr must never win finest-child.

    The nastier sibling of the crash case: arrays can hold attrs, so an array
    named after the real child and stamped with a recognised geometry ``type``
    used to be picked as the finest child and resolve EARLY — no exception,
    just the wrong ``display_type`` written to the wrapper.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    leaf = lod.create_group("aaa_real_child")
    leaf.attrs["type"] = "lines"
    stray = create_array(lod, "zzz_array", data=np.zeros((2, 2), dtype=np.float32))
    stray.attrs["type"] = "gsplats"

    finalize_lod_display_types(root)

    assert dict(root["lodgrp"].attrs).get("display_type") == "lines"


def test_position_bounds_backfill_ignores_array_siblings() -> None:
    """The bounds pass must also walk/resolve child GROUPS only.

    Sibling of the display-type guard: both passes recurse through the tree, and
    a stray array beside real children would be descended into — ``resolve``
    looking for its ``position_bounds`` and ``walk`` looking for its ``kind`` —
    dying on ``Array.keys()`` either way.
    """
    root = zarr.group()
    lod = root.create_group("lodgrp")
    lod.attrs["kind"] = "lod"
    create_array(lod, "stray", data=np.zeros((2, 2), dtype=np.float32))
    for name, bounds in (
        ("a", {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}),
        ("b", {"min": [-1.0, 0.0, 0.0], "max": [1.0, 2.0, 1.0]}),
    ):
        child = lod.create_group(name)
        child.attrs["type"] = "points"
        child.attrs["position_bounds"] = bounds

    finalize_lod_position_bounds(root)

    assert dict(root["lodgrp"].attrs)["position_bounds"] == {
        "min": [-1.0, 0.0, 0.0],
        "max": [1.0, 2.0, 1.0],
    }


# ────────────────────────────────────────────────────────────────────────
# warn_one_part_partition_anchors — the check no PRODUCER can make
# ────────────────────────────────────────────────────────────────────────


def _anchored_ladder(parent: zarr.Group, name: str, finest: float) -> None:
    """A kind=lod group of two mesh children whose finest threshold is ``finest``."""
    lod = parent.create_group(name)
    lod.attrs["kind"] = "lod"
    lod.attrs["display_type"] = "mesh"
    for i, coverage in enumerate((0.0, finest)):
        child = lod.create_group(f"child_{i}")
        child.attrs["type"] = "mesh"
        child.attrs["coverage_fraction"] = coverage


def _partition_of_ladders(n_parts: int, finest: float) -> zarr.Group:
    root = zarr.group()
    part = root.create_group("tiled")
    part.attrs["kind"] = "partition"
    part.attrs["display_type"] = "mesh"
    part.attrs["max_elements"] = 100
    for i in range(n_parts):
        _anchored_ladder(part, f"part_{i}", finest)
    return root


def test_warns_for_a_tile_anchored_ladder_under_a_one_part_partition(capsys) -> None:
    # The undetectable-at-derive-time case: the scene adder anchored part 0 at
    # 4.0 before it could know part 1 would never arrive.
    warn_one_part_partition_anchors(_partition_of_ladders(1, MAX_COVERAGE_FRACTION))
    out = capsys.readouterr().out
    assert "tiled/part_0" in out
    assert "ONE part" in out
    assert "coverage_fractions" in out, "the message must name the way out"


def test_does_not_warn_for_a_genuine_multi_part_partition(capsys) -> None:
    # Two parts IS a tiling, so 4.0 is the correct anchor there.
    warn_one_part_partition_anchors(_partition_of_ladders(2, MAX_COVERAGE_FRACTION))
    assert capsys.readouterr().out == ""


def test_does_not_warn_for_a_whole_object_ladder(capsys) -> None:
    # A one-part partition holding a 1.0-anchored ladder is exactly right.
    warn_one_part_partition_anchors(_partition_of_ladders(1, 1.0))
    assert capsys.readouterr().out == ""


def test_does_not_warn_without_a_partition_ancestor(capsys) -> None:
    # A hand-authored 4.0 ladder outside any partition is the author's business.
    root = zarr.group()
    _anchored_ladder(root, "ladder", MAX_COVERAGE_FRACTION)
    warn_one_part_partition_anchors(root)
    assert capsys.readouterr().out == ""


def test_blames_the_nearest_partition_not_an_outer_one_part_wrapper(capsys) -> None:
    """A real tiling nested inside a one-part wrapper must not be blamed.

    Its ladders switch on a genuine tile, so their 4.0 anchor is correct; only
    the ladder bound DIRECTLY to the lone part is wrong.
    """
    root = zarr.group()
    outer = root.create_group("outer")
    outer.attrs["kind"] = "partition"
    inner = outer.create_group("only_part")
    inner.attrs["kind"] = "partition"
    for i in range(2):
        _anchored_ladder(inner, f"tile_{i}", MAX_COVERAGE_FRACTION)

    warn_one_part_partition_anchors(root)

    assert capsys.readouterr().out == ""


def test_does_not_warn_for_a_one_part_partition_inside_a_real_tiling(capsys) -> None:
    """The inverse nesting: a lone part sitting INSIDE a genuine tiling.

    Both gsplat writers OR an incoming binding in rather than overwriting it
    (``under_partition or len(children) > 1``), so the ladder here really is
    inside one of the outer partition's two tiles and 4.0 is the anchor they
    derive for it. Blaming the nearest wrapper alone would report that as a
    mistake.
    """
    root = zarr.group()
    outer = root.create_group("tiled")
    outer.attrs["kind"] = "partition"
    _anchored_ladder(outer, "part_0", MAX_COVERAGE_FRACTION)
    inner = outer.create_group("part_1")
    inner.attrs["kind"] = "partition"
    _anchored_ladder(inner, "only_part", MAX_COVERAGE_FRACTION)

    warn_one_part_partition_anchors(root)

    assert capsys.readouterr().out == ""


def test_warns_once_per_offending_group(capsys) -> None:
    root = zarr.group()
    part = root.create_group("tiled")
    part.attrs["kind"] = "partition"
    holder = part.create_group("only_part")  # a plain group between the two
    _anchored_ladder(holder, "ladder", MAX_COVERAGE_FRACTION)

    warn_one_part_partition_anchors(root)

    assert capsys.readouterr().out.count("tiled/only_part/ladder") == 1
