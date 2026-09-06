"""``luxar env attach`` and the store rules a baked environment rests on.

What is pinned is the CONTRACT (``MESH_PHYSICAL_MATERIALS_SPEC.md`` §3.3), not
the picture: the environment group is invisible to every node walker, the scene
``content_hash`` does not move when a map is attached (so the header's
``scene_content_hash`` guard is exact and warm caches survive), the faces array
is named by its own digest (so a re-bake is a new path), and attaching the same
map twice writes nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarScene, LuxarZarrCompiler
from luxar._zarr_compat import open_group
from luxar.cli.info_command import _dfs
from luxar.environment import (
    ENVIRONMENT_FORMAT,
    FACE_ORDER,
    attach_environment,
    pack,
    unpack,
)
from luxar.io._compiler.finalize.hashing import compute_content_hashes
from luxar.io.optimise import _restamp_content_hash, optimise_store
from luxar.typing_utils.constants import ENVIRONMENT_GROUP, RESERVED_ROOT_GROUPS

RES = 16


def _scene(path: Path) -> str:
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        rng = np.random.default_rng(3)
        scene.add_points("cloud", rng.random((500, 3)).astype(np.float32))
    return str(dict(open_group(path, mode="r").attrs)["content_hash"])


def _faces(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.random((6, RES, RES, 4), dtype=np.float32).astype(np.float16)
    values[..., 3] = 1.0
    return values.view(np.uint16)


def _header(scene_hash: str, **extra) -> dict:
    header = {
        "format": ENVIRONMENT_FORMAT,
        "face_order": list(FACE_ORDER),
        "coordinate_system": "webgl",
        "probe": {"spec": "auto", "position": [0.5, 0.5, 0.5]},
        "resolution": RES,
        "scene_content_hash": scene_hash,
        "appearance": {},
        "baked_at": "2026-09-06T00:00:00Z",
        "viewer_version": "test",
    }
    header.update(extra)
    return header


# =============================================================================
# The container
# =============================================================================


def test_container_round_trips_and_refuses_junk() -> None:
    faces = _faces()
    header = _header("abc")
    blob = pack(header, faces)
    back_header, back_faces = unpack(blob)
    assert back_header == header
    assert back_faces.dtype == np.uint16
    assert np.array_equal(back_faces, faces)
    with pytest.raises(ValueError, match="magic"):
        unpack(b"PNG\x00" + blob[4:])
    with pytest.raises(ValueError, match="sample bytes"):
        unpack(blob[:-2])
    with pytest.raises(ValueError, match="missing"):
        pack({"format": ENVIRONMENT_FORMAT}, faces)
    with pytest.raises(ValueError, match="shape"):
        pack(header, faces[:5])
    with pytest.raises(ValueError, match="uint16"):
        pack(header, faces.view(np.float16))
    # A `type`/`kind` key would make the viewer read the sidecar as a node.
    with pytest.raises(ValueError, match="'type'"):
        pack(_header("abc", type="group"), faces)


# =============================================================================
# Attaching
# =============================================================================


def test_attach_writes_a_digest_named_array_and_leaves_the_scene_hash_alone(
    tmp_path,
) -> None:
    store = tmp_path / "scene.luxar.zarr"
    before = _scene(store)
    report = attach_environment(store, pack(_header(before), _faces()))

    assert report.status == "attached"
    assert report.array_name.startswith("faces-")
    assert report.array_name == f"faces-{report.digest[:8]}"

    root = zarr.open_group(str(store), mode="r")
    env = root[ENVIRONMENT_GROUP]
    attrs = dict(env.attrs)
    assert attrs["faces"] == report.array_name
    assert attrs["scene_content_hash"] == before
    assert attrs["sample_format"] == "half-float-bits"
    assert attrs["shape"] == [6, RES, RES, 4]
    assert attrs["face_order"] == list(FACE_ORDER)
    assert "type" not in attrs and "kind" not in attrs
    array = env[report.array_name]
    assert array.shape == (6, RES, RES, 4)
    assert array.dtype == np.uint16
    assert array.chunks == (1, RES, RES, 4)
    assert np.array_equal(array[:], _faces())

    # THE load-bearing property: the root digest is unchanged, both as stored
    # and when recomputed by either hashing walk.
    assert dict(root.attrs)["content_hash"] == before
    rw = open_group(store, mode="r+")
    assert compute_content_hashes(rw) == before
    assert _restamp_content_hash(rw) == before


def test_attach_is_idempotent_and_a_rebake_replaces_the_old_array(tmp_path) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    first = attach_environment(store, pack(_header(scene_hash), _faces(0)))
    again = attach_environment(store, pack(_header(scene_hash), _faces(0)))
    assert again.status == "unchanged"
    assert again.array_name == first.array_name

    rebake = attach_environment(store, pack(_header(scene_hash), _faces(1)))
    assert rebake.status == "replaced"
    assert rebake.array_name != first.array_name
    assert rebake.removed == [first.array_name]
    env = zarr.open_group(str(store), mode="r")[ENVIRONMENT_GROUP]
    assert list(env.array_keys()) == [rebake.array_name]
    assert dict(env.attrs)["faces"] == rebake.array_name
    assert (
        dict(zarr.open_group(str(store), mode="r").attrs)["content_hash"] == scene_hash
    )


def test_attach_refuses_a_stale_bake_unless_forced(tmp_path) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    blob = pack(_header("0123456789abcdef0123"), _faces())
    with pytest.raises(ValueError, match="scene changed since the bake"):
        attach_environment(store, blob)
    assert ENVIRONMENT_GROUP not in zarr.open_group(str(store), mode="r")
    report = attach_environment(store, blob, force=True)
    assert report.status == "attached"
    env = zarr.open_group(str(store), mode="r")[ENVIRONMENT_GROUP]
    # Forced: the header is rewritten to the CURRENT digest, so the viewer's
    # guard passes for this scene as it is now.
    assert dict(env.attrs)["scene_content_hash"] == scene_hash


def test_attach_refuses_a_non_scene_store(tmp_path) -> None:
    store = tmp_path / "plain.zarr"
    root = zarr.open_group(str(store), mode="w")
    root.attrs["content_hash"] = "x"
    with pytest.raises(ValueError, match="not a Luxar scene store"):
        attach_environment(store, pack(_header("x"), _faces()))


# =============================================================================
# The group is a sidecar, not a node
# =============================================================================


def test_environment_group_is_invisible_to_every_python_node_walker(tmp_path) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    attach_environment(store, pack(_header(scene_hash), _faces()))

    assert ENVIRONMENT_GROUP in RESERVED_ROOT_GROUPS
    names = [n["name"] for n in LuxarScene.load(store).nodes]
    assert names == ["cloud"]
    walked = [g.basename for _, g in _dfs(open_group(store, mode="r"))]
    assert ENVIRONMENT_GROUP not in walked
    assert "cloud" in walked


def test_compiler_refuses_a_user_node_named_environment(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "scene.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="reserved for non-node metadata"):
            scene.add_group(ENVIRONMENT_GROUP)


def test_optimise_copies_the_environment_verbatim(tmp_path) -> None:
    store = tmp_path / "scene.luxar.zarr"
    scene_hash = _scene(store)
    report = attach_environment(store, pack(_header(scene_hash), _faces()))
    out = tmp_path / "out.luxar.zarr"
    plan = optimise_store(store, out)
    env_plans = [p for p in plan.arrays if p.path.startswith(f"{ENVIRONMENT_GROUP}/")]
    assert len(env_plans) == 1
    assert env_plans[0].skip_reason == "baked environment map"
    copied = zarr.open_group(str(out), mode="r")[ENVIRONMENT_GROUP]
    assert copied[report.array_name].chunks == (1, RES, RES, 4)
    assert np.array_equal(copied[report.array_name][:], _faces())
    assert dict(copied.attrs)["faces"] == report.array_name
