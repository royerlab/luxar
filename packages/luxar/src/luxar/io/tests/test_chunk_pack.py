"""``luxar optimize --pack`` — one request per small node, the store unchanged.

A pack is a COPY of a small node's chunk objects in one plain ``.pack`` file
under the root ``chunk_packs/`` sidecar, so a viewer can fetch the node in
one request instead of one per chunk. Every assertion here guards a property
that would otherwise fail silently: a member slice that is not byte-identical to
its chunk renders wrong data, a pack that is folded into ``content_hash`` can
never certify the hash it was built for, and a sidecar that a plain zarr reader
trips over breaks the "still a valid zarr store" promise.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import open_group, read_raw_bytes, set_zarr_format, zarr_format
from luxar.core.dimensions import Dimensions
from luxar.io import LuxarZarrCompiler
from luxar.io.optimize import optimize_store
from luxar.io.reader import LuxarScene
from luxar.typing_utils.constants import CHUNK_PACKS_GROUP, TARGET_CHUNK_BYTES


def _build(path: Path) -> Path:
    """Many small laddered parts (packable) plus one large node (not)."""
    rng = np.random.default_rng(11)
    with LuxarZarrCompiler(str(path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "small",
            rng.random((2000, 3)).astype(np.float32),
            colors=rng.random((2000, 3)).astype(np.float32),
            partition={"max_elements": 250},
            additive_lod={"n_lods": 3},
        )
        scene.add_points("big", rng.random((40_000, 3)).astype(np.float32))
    return path


@pytest.fixture(scope="module", params=[2, 3])
def stores(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, Path]:
    """``(plain, packed)``: the same source optimized without and with packs.

    Both zarr formats: their chunk keys are spelled differently (``0.0`` versus
    ``c/0/0``), and the index hands the viewer keys it uses verbatim.
    """
    base = tmp_path_factory.mktemp(f"pack_v{request.param}")
    original = zarr_format()
    set_zarr_format(request.param)
    try:
        source = _build(base / "src.luxar.zarr")
    finally:
        set_zarr_format(original)
    plain = base / "plain.luxar.zarr"
    packed = base / "packed.luxar.zarr"
    optimize_store(source, plain)
    optimize_store(source, packed, pack=True)
    return plain, packed


def _index(root: zarr.Group) -> dict[str, Any]:
    return dict(root[CHUNK_PACKS_GROUP].attrs)


def _members(root: zarr.Group, pack: dict[str, Any]) -> tuple[dict, bytes]:
    """The pack file's member table and data section."""
    blob = read_raw_bytes(root, pack["key"])
    assert blob is not None
    assert hashlib.sha256(blob).hexdigest() == pack["sha256"]
    n = int.from_bytes(blob[:4], "little")
    return json.loads(blob[4 : 4 + n])["members"], blob[4 + n :]


def _walk_arrays(group: zarr.Group, path: str = ""):
    for name in sorted(group.array_keys()):
        yield (f"{path}/{name}" if path else name), group[name]
    for name in sorted(group.group_keys()):
        yield from _walk_arrays(group[name], f"{path}/{name}" if path else name)


def test_every_member_is_its_chunk_byte_for_byte(stores: tuple[Path, Path]) -> None:
    _, packed = stores
    root = open_group(packed, mode="r")
    packs = _index(root)["packs"]
    assert packs, "no node was packed"
    for pack in packs:
        members, data = _members(root, pack)
        for key, (offset, length) in members.items():
            chunk = read_raw_bytes(root, pack["prefix"] + key)
            assert data[offset : offset + length] == chunk, key
        assert sum(n for _, n in members.values()) == len(data)


def test_packs_cover_small_nodes_only(stores: tuple[Path, Path]) -> None:
    _, packed = stores
    root = open_group(packed, mode="r")
    packs = _index(root)["packs"]
    prefixes = {p["prefix"] for p in packs}
    assert all(p.startswith("small/") for p in prefixes)
    assert len(prefixes) == len(packs) > 1
    assert all(len(_members(root, p)[1]) <= TARGET_CHUNK_BYTES for p in packs)
    # Every chunk a packed node stores is a member: a viewer that holds the pack
    # never needs a second request for that node.
    for pack in packs:
        node = root[pack["prefix"].rstrip("/")]
        stored = set()
        for rel, array in _walk_arrays(node):
            for coords in np.ndindex(*array.cdata_shape):
                key = f"{rel}/{array.metadata.encode_chunk_key(coords)}"
                if read_raw_bytes(node, key) is not None:
                    stored.add(key)
        assert stored == set(_members(root, pack)[0]), pack["prefix"]


def test_the_index_stays_small(stores: tuple[Path, Path]) -> None:
    """It rides in the consolidated root every load fetches first, so it holds
    no member list — the pack's own header does."""
    _, packed = stores
    index = _index(open_group(packed, mode="r"))
    assert all(set(p) == {"key", "sha256", "prefix"} for p in index["packs"])
    assert len(json.dumps(index)) < 200 * len(index["packs"]) + 200


def test_packing_leaves_the_content_hash_and_records_it(
    stores: tuple[Path, Path],
) -> None:
    plain, packed = stores
    plain_root = open_group(plain, mode="r")
    packed_root = open_group(packed, mode="r")
    # The packs copy chunks; they change no chunk, so a warm cache keyed on the
    # hash stays right — and the index certifies the hash it was built for.
    assert packed_root.attrs["content_hash"] == plain_root.attrs["content_hash"]
    assert (
        _index(packed_root)["scene_content_hash"] == packed_root.attrs["content_hash"]
    )
    assert CHUNK_PACKS_GROUP not in plain_root


def test_packed_store_reads_as_the_plain_one(stores: tuple[Path, Path]) -> None:
    plain, packed = stores
    plain_root = zarr.open_group(str(plain), mode="r")
    packed_root = zarr.open_group(str(packed), mode="r")
    plain_arrays = dict(_walk_arrays(plain_root))
    packed_arrays = {
        path: array
        for path, array in _walk_arrays(packed_root)
        if not path.startswith(f"{CHUNK_PACKS_GROUP}/")
    }
    assert packed_arrays.keys() == plain_arrays.keys()
    for path, array in plain_arrays.items():
        np.testing.assert_array_equal(
            packed_arrays[path][...], array[...], err_msg=path
        )
    # Luxar's own reader skips the sidecar like the environment group.
    names = [sorted(n["name"] for n in LuxarScene.load(p).nodes) for p in stores]
    assert names[0] == names[1]
    # Without the consolidated index too: the pack files are plain keys inside
    # the sidecar group, which a member walk skips (with zarr's own warning).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unconsolidated = zarr.open_group(str(packed), mode="r", use_consolidated=False)
        walked = {
            path
            for path, _ in _walk_arrays(unconsolidated)
            if not path.startswith(CHUNK_PACKS_GROUP)
        }
    assert walked == plain_arrays.keys()


def test_repacking_a_packed_store_replaces_its_packs(
    stores: tuple[Path, Path], tmp_path: Path
) -> None:
    _, packed = stores
    again = tmp_path / "again.luxar.zarr"
    optimize_store(packed, again, pack=True)
    first = _index(open_group(packed, mode="r"))
    second = _index(open_group(again, mode="r"))
    assert [p["sha256"] for p in second["packs"]] == [
        p["sha256"] for p in first["packs"]
    ]


def test_optimizing_without_pack_drops_the_sidecar(
    stores: tuple[Path, Path], tmp_path: Path
) -> None:
    """Derived data is rebuilt or dropped, never copied stale."""
    _, packed = stores
    out = tmp_path / "unpacked.luxar.zarr"
    optimize_store(packed, out, verify=True)
    assert CHUNK_PACKS_GROUP not in open_group(out, mode="r")


def test_pack_refuses_a_store_it_cannot_certify(tmp_path: Path) -> None:
    """Only a compiled scene's hash excludes the sidecar, so only it is packed."""
    from luxar.gsplats.io.save_gsplats import save_gsplats

    n = 64
    source = tmp_path / "s.gsplats.zarr"
    save_gsplats(
        source,
        centers=np.random.default_rng(0).random((n, 3)).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=np.tile(np.array([1, 0, 1, 0, 0, 1], np.float32), (n, 1)),
    )
    with pytest.raises(ValueError, match="--pack"):
        optimize_store(source, tmp_path / "out.gsplats.zarr", pack=True)
