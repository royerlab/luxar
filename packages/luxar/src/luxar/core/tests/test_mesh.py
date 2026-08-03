"""Tests for the Mesh node: add_mesh, round-trip, and the §9 exclusions.

Covers the three things that can only be checked through the full stack — the
store a write actually produces, the values a read gives back, and the paths mesh
must be refused from. The pure validators live in
``validation/tests/test_mesh_validation.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.mesh import Mesh
from luxar.io import LuxarScene, MeshData

# A welded tetrahedron: 4 vertices, 4 faces, every vertex shared by 3 faces.
_V = np.array(
    [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
    dtype=np.float32,
)
_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
_N = np.array(
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)


def _write(tmp_path, name="m", **kwargs):
    """Write one mesh into a fresh scene and return the store path."""
    store = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(name, _V, _F, **kwargs)
    return store


# =============================================================================
# add_mesh + node object
# =============================================================================


def test_add_mesh_returns_a_mesh_node(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "a.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_mesh("surface", _V, _F)
        assert isinstance(node, Mesh)
        assert node.n_elements == 4
        assert node.n_faces == 4


def test_n_elements_counts_vertices_not_faces(tmp_path) -> None:
    """The primary element is the vertex, matching ``Lines.n_elements``.

    Pinned with a mesh whose vertex and face counts DIFFER, so the assertion
    cannot pass by coincidence — the tetrahedron's 4-and-4 would not distinguish
    them.
    """
    # A quad as two triangles: 4 vertices, 2 faces.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    with LuxarZarrCompiler(tmp_path / "q.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_mesh("quad", vertices, faces)
        assert node.n_elements == 4
        assert node.n_faces == 2


@pytest.mark.parametrize(
    "kwargs,expected_shading,test_id",
    [
        ({}, "flat", "no_normals_defaults_flat"),
        ({"normals": _N, "normal_dims": (0, 1, 2)}, "smooth", "normals_default_smooth"),
        (
            {"normals": _N, "normal_dims": (0, 1, 2), "shading": "flat"},
            "flat",
            "explicit_flat_overrides_normals",
        ),
        ({"shading": "smooth"}, "smooth", "explicit_smooth_without_normals_kept"),
    ],
)
def test_shading_resolution(tmp_path, kwargs, expected_shading, test_id) -> None:
    """``shading`` defaults by normal presence and is never rewritten when explicit.

    The last case is the load-bearing one: an explicit "smooth" with no normals is
    stored AS GIVEN rather than corrected to "flat", because the viewer resolves it
    at render time via the derivative fallback. A write-time rewrite would discard
    the author's intent for a mesh that later gains normals.
    """
    store = _write(tmp_path, name=test_id, **kwargs)
    node = zarr.open_group(store, mode="r")[test_id]
    assert dict(node.attrs)["shading"] == expected_shading


def test_double_sided_defaults_true_and_round_trips_false(tmp_path) -> None:
    default_store = _write(tmp_path, name="ds_default")
    assert (
        dict(zarr.open_group(default_store, mode="r")["ds_default"].attrs)[
            "double_sided"
        ]
        is True
    )

    off_store = _write(tmp_path, name="ds_off", double_sided=False)
    assert (
        dict(zarr.open_group(off_store, mode="r")["ds_off"].attrs)["double_sided"]
        is False
    )


def test_ordering_is_none(tmp_path) -> None:
    """v1 has no spatial index, and the attr is stamped rather than left absent.

    A reader should never have to distinguish "no ordering" from "attr missing",
    and ``Mesh.ordering`` reads the stamp rather than returning a literal — a
    property that lies is worse than one that is missing.
    """
    store = _write(tmp_path, name="ord")
    assert dict(zarr.open_group(store, mode="r")["ord"].attrs)["ordering"] == "none"


# =============================================================================
# Round-trip
# =============================================================================


def test_round_trip_preserves_topology_exactly(tmp_path) -> None:
    """Vertices and faces must survive byte-exact.

    Topology cannot tolerate lossy encoding: a face index off by one is a
    different surface, which is why ``faces`` is written with dedup and LUT
    encoding disabled.
    """
    store = _write(tmp_path, name="rt")
    mesh = LuxarScene.load(store).get_mesh("rt")

    assert isinstance(mesh, MeshData)
    assert np.array_equal(mesh.vertices, _V)
    assert np.array_equal(mesh.faces, _F)
    assert mesh.faces.dtype == np.uint32 or np.issubdtype(mesh.faces.dtype, np.integer)


def test_round_trip_normals_and_dims(tmp_path) -> None:
    store = _write(tmp_path, name="rtn", normals=_N, normal_dims=(0, 1, 2))
    mesh = LuxarScene.load(store).get_mesh("rtn")

    assert mesh.normal_dims == [0, 1, 2]
    # Normals go through the COORDINATE encoder (per-axis uint16 over [-1, 1]),
    # so compare within quantization rather than exactly.
    assert np.allclose(mesh.normals, _N, atol=1e-3)


def test_normal_dims_absent_without_normals(tmp_path) -> None:
    """No normals means no ``normal_dims``, not a leftover value."""
    store = _write(tmp_path, name="nd")
    mesh = LuxarScene.load(store).get_mesh("nd")
    assert mesh.normals is None
    assert mesh.normal_dims is None


def test_get_mesh_normalizes_flat_faces(tmp_path) -> None:
    """A flat ``(3F,)`` faces array reads back as ``(F, 3)``.

    The writer always emits pairs, but the write-side validator accepts the flat
    form and an external producer may use it, so the reader normalizes rather than
    making every consumer branch.
    """
    flat = _F.reshape(-1)
    store = tmp_path / "flat.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("flat", _V, flat)

    mesh = LuxarScene.load(store).get_mesh("flat")
    assert mesh.faces.shape == (4, 3)
    assert np.array_equal(mesh.faces, _F)


def test_list_meshes_and_node_inventory(tmp_path) -> None:
    store = _write(tmp_path, name="inv")
    scene = LuxarScene.load(store)

    assert scene.list_meshes() == ["inv"]
    assert scene.get_node_type("inv") == "mesh"
    entry = next(n for n in scene.nodes if n["type"] == "mesh")
    assert entry["n_vertices"] == 4
    assert entry["n_faces"] == 4


def test_get_mesh_rejects_a_non_mesh_node(tmp_path) -> None:
    store = tmp_path / "wrong.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("pts", _V)

    scene_read = LuxarScene.load(store)
    with pytest.raises(ValueError, match="is not a mesh node"):
        scene_read.get_mesh("pts")


def test_mesh_contributes_local_scene_bounds(tmp_path) -> None:
    """A mesh's extent reaches the scene's local bounds accumulator.

    This exercises the writer's own ``update_scene_bounds`` call, NOT the
    finalize-time world-bounds walk — see the transform test below for that. The
    two are separate paths and only the second consults the contract vocabulary.
    """
    store = _write(tmp_path, name="bnd")
    root_attrs = dict(zarr.open_group(store, mode="r").attrs)
    bounds = root_attrs.get("position_bounds")
    assert bounds is not None, f"no local scene bounds: {sorted(root_attrs)}"
    assert bounds["max"][:3] == [10.0, 10.0, 10.0]


def test_mesh_reaches_the_world_bounds_walk_under_a_transform(tmp_path) -> None:
    """A TRANSFORMED mesh must widen world bounds — the contract-vocabulary path.

    The finalize walk (``expand_bounds_with_transforms``) is what selects leaves by
    testing ``type in GEOMETRY_TYPES``, and it is the only path that applies
    ancestor transforms. Without a transform this test would be vacuous: the
    writer's local accumulator already records the untransformed extent, so world
    bounds would look right even if the walk skipped the mesh entirely (verified —
    an earlier version of this test passed with mesh excluded from the walk).

    Translating by +100 makes the two paths give different answers, so only the
    walk having seen the mesh can produce the expected maximum. The viewer derives
    near/far clipping from world bounds, so a mesh skipped here frames the camera
    wrongly with no error — the defect class #1203 fixed for the other three types.
    """
    store = tmp_path / "wbnd.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        moved = scene.add_group(
            "moved",
            transform=[
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        moved.add_mesh("surface", _V, _F)

    # The root's ``position_bounds`` holds the WORLD-expanded value: the finalize
    # walk re-stamps the key after applying transforms, so this reads world bounds
    # despite the local-sounding name.
    world = dict(zarr.open_group(store, mode="r").attrs).get("position_bounds")
    assert world is not None, "no scene bounds written"
    # The mesh spans x in [0, 10] locally, so translated it must reach x = 110.
    # A skipped mesh leaves the walk with nothing to transform, and the untouched
    # local extent (max x = 10) shows through instead.
    assert world["max"][0] == pytest.approx(110.0), (
        f"world bounds {world} do not include the translated mesh — the "
        "finalize walk did not treat 'mesh' as a geometry leaf"
    )
    assert world["min"][0] == pytest.approx(100.0)


# =============================================================================
# §9 exclusions — each must raise, not silently degrade
# =============================================================================


def test_mesh_under_a_lod_group_is_rejected(tmp_path) -> None:
    """Refused at ADD time, before any array lands on disk.

    The finalize-time LOD guard would also catch this, but only after the mesh's
    vertices and faces are already written — so the store would be left with a
    partial node. Failing here keeps the write fail-fast.
    """
    with LuxarZarrCompiler(tmp_path / "lod.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("ladder")
        with pytest.raises(ValueError, match="kind=lod"):
            lod.add_mesh("child_0", _V, _F)


def test_mesh_under_a_partition_group_is_rejected(tmp_path) -> None:
    """A partition's declared ``display_type`` must not be made a lie.

    ``add_partition_group`` rejects ``display_type='mesh'`` directly, but nothing
    stopped a caller from declaring a ``points`` partition and then adding a mesh
    child into it.
    """
    with LuxarZarrCompiler(tmp_path / "part.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        part = scene.add_partition_group(
            "parts", display_type="points", max_elements=100
        )
        with pytest.raises(ValueError, match="kind=partition"):
            part.add_mesh("part_0", _V, _F)


def test_partition_group_rejects_mesh_display_type(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "pd.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="display_type for a partition group"):
            scene.add_partition_group("p", display_type="mesh", max_elements=100)


def test_lod_group_rejects_explicit_mesh_display_type(tmp_path) -> None:
    """The explicit-kwarg route is gated too, not only the finalize back-fill.

    ``display_type`` is an accepted node attr, so it rides in through ``**attrs``
    and reaches zarr without passing the back-fill at all — the back-fill only
    ever sees groups that supplied nothing.
    """
    with LuxarZarrCompiler(tmp_path / "ld.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="display_type for a kind=lod group"):
            scene.add_lod_group("l", display_type="mesh")


# =============================================================================
# Authoring lint
# =============================================================================


def test_unwelded_mesh_warns(tmp_path, capsys) -> None:
    """An independent-triangle soup gets the warn-only unwelded lint.

    Uses 10 faces so it clears the minimum-face gate; each triangle owns its three
    vertices (V == 3F, no shared index), which is what flattening a soup produces.
    """
    n_faces = 10
    vertices = np.arange(n_faces * 3 * 3, dtype=np.float32).reshape(-1, 3)
    faces = np.arange(n_faces * 3, dtype=np.uint32).reshape(-1, 3)
    with LuxarZarrCompiler(tmp_path / "soup.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("soup", vertices, faces)
    assert "independent triangles" in capsys.readouterr().out


def test_welded_mesh_does_not_warn(tmp_path, capsys) -> None:
    """The lint must not fire on a properly welded surface.

    The other half of the test above — a lint that always fires is as useless as
    one that never does. Uses a shared-vertex fan with enough faces to clear the
    gate, so silence here is the heuristic discriminating rather than the
    face-count floor suppressing it.
    """
    n_faces = 12
    # A fan: one hub vertex shared by every triangle.
    rim = np.stack(
        [
            np.cos(np.linspace(0, 2 * np.pi, n_faces + 1)[:-1]),
            np.sin(np.linspace(0, 2 * np.pi, n_faces + 1)[:-1]),
            np.zeros(n_faces),
        ],
        axis=1,
    ).astype(np.float32)
    vertices = np.vstack([np.zeros((1, 3), dtype=np.float32), rim])
    faces = np.array(
        [[0, i + 1, (i + 1) % n_faces + 1] for i in range(n_faces)], dtype=np.uint32
    )
    with LuxarZarrCompiler(tmp_path / "fan.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("fan", vertices, faces)
    assert "independent triangles" not in capsys.readouterr().out
