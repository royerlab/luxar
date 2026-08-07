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


def test_mesh_with_volumetric_blending_is_rejected(tmp_path) -> None:
    """The one §9 exclusion that was documented but never enforced.

    Both the spec and CLAUDE.md say a mesh refuses ``volumetric`` "with an
    explanation rather than silently degraded", and it did not: the mode wrote and
    loaded cleanly because ``validate_blending_mode`` is geometry-agnostic by design
    and nothing downstream asked the question. Volumetric integrates emission and
    absorption along the view ray, so a zero-thickness surface has no path length to
    integrate over and ``absorption`` has nothing to attenuate.
    """
    with LuxarZarrCompiler(tmp_path / "vol.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="blending_mode='volumetric'"):
            scene.add_mesh("surface", _V, _F, blending_mode="volumetric")


def test_mesh_accepts_the_blending_modes_it_does_support(tmp_path) -> None:
    """The acceptance half — otherwise the check above could be "reject every mode".

    These have to be WRITABLE independently of what any given viewer phase draws —
    the mesh material honours them today, and the scenes authored before it did must
    not need rewriting.
    """
    with LuxarZarrCompiler(tmp_path / "modes.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        for i, mode in enumerate(("normal", "additive")):
            mesh = scene.add_mesh(f"surface_{i}", _V, _F, blending_mode=mode)
            assert mesh is not None


def test_mesh_blending_mode_setter_also_rejects_volumetric(tmp_path) -> None:
    """The post-add mutation path, which the adder check alone cannot cover.

    ``Node.blending_mode`` (and the chaining ``set_blending_mode``) is a public
    setter that persists straight to zarr, so without the :class:`Mesh` override a
    caller could add a mesh with a supported mode and flip it to ``volumetric``
    one line later — re-opening exactly the door the adder closes.
    """
    with LuxarZarrCompiler(tmp_path / "setter.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        mesh = scene.add_mesh("surface", _V, _F, blending_mode="normal")
        with pytest.raises(ValueError, match="volumetric"):
            mesh.blending_mode = "volumetric"
        with pytest.raises(ValueError, match="volumetric"):
            mesh.set_blending_mode("volumetric")
        # The supported modes still flow through the base validation unchanged.
        mesh.blending_mode = "additive"
        assert mesh.blending_mode == "additive"


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


def test_lod_refusal_distinguishes_the_two_ladder_flavours(tmp_path) -> None:
    """The REASON is pinned, not just the fact of the refusal.

    Every other assertion in this file matches a short structural substring
    (``"kind=lod"``), which is why the justification was free to rot: the message
    spent months telling users that substitutive LOD reduces independent elements
    and is therefore impossible for a surface. Only the additive flavour works that
    way. Substitutive levels are independently-authored ``(vertices, faces)`` pairs
    chosen by ``coverage_fraction`` — the machinery makes no independence assumption
    and the only missing piece is a decimator (spec §9).

    "Impossible" and "not written yet" are different answers to a user asking
    whether to wait for it, so the distinction is worth a test.
    """
    with LuxarZarrCompiler(tmp_path / "why.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("ladder")
        with pytest.raises(ValueError) as excinfo:
            lod.add_mesh("child_0", _V, _F)

    message = str(excinfo.value)
    assert "ADDITIVE" in message and "SUBSTITUTIVE" in message
    # The additive arm is excluded on principle: a prefix of an index buffer is a
    # holed surface, not a coarse one.
    assert "holes" in message
    # The substitutive arm is excluded only for want of a producer.
    assert "no producer" in message
    # ...and must NOT be blamed on the independence assumption that only the
    # additive ladder makes. This is the exact sentence that was wrong. Scoped to the
    # substitutive clause on purpose: "the ADDITIVE ladder reduces independent
    # elements" is accurate, and the sibling copy of this rationale in
    # ``typing_utils/geometry_capabilities.py`` says exactly that.
    _additive_arm, substitutive_arm = message.split("SUBSTITUTIVE", 1)
    assert "independent elements" not in substitutive_arm


def test_mesh_rejects_hand_supplied_energy_stamps(tmp_path) -> None:
    """Spec §9.1: a mesh must never carry an additive ladder's energy stamps.

    ``level_stats`` / ``lod_stats`` are in the geometry-blind allow-list in
    ``io/_compiler/node_common.py``, so before this guard they wrote clean on a mesh.
    The viewer's ``energyCompensation`` scales brightness by ``1/energy_fraction_cum``
    and is gated on the BLENDING MODE (``additive``/``luminous``/``volumetric``), not
    on geometry type — and mesh supports two of those. Brightening a dimmer splat
    prefix is right; brightening a holed surface is not.

    Prophylactic rather than a live-bug fix (mesh cannot be in a ``kind=lod`` group,
    and the mesh commit never stamps ``committedEnergyFraction``), so this test is
    what keeps the rule true once substitutive LOD or a reveal ladder removes one of
    those latches.
    """
    with LuxarZarrCompiler(tmp_path / "stamps.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        for key in ("lod_stats", "level_stats"):
            with pytest.raises(ValueError, match="energy"):
                scene.add_mesh(f"m_{key}", _V, _F, **{key: {"reference_energy": 1.0}})

        # Refused on KEY PRESENCE, not on what the dict happens to hold: a
        # ``quality``-only ``level_stats`` is not an energy stamp, but it has no
        # meaning on a mesh either, so the container goes too. Deliberate breadth —
        # the day substitutive mesh levels land, this narrows to the energy keys and
        # this assertion is what makes that an explicit decision.
        with pytest.raises(ValueError, match="energy"):
            scene.add_mesh("m_quality", _V, _F, level_stats={"quality": 0.9})

        # The keys are refused, not the whole attrs surface: a mesh with ordinary
        # render attrs still writes.
        assert scene.add_mesh("plain", _V, _F, opacity=0.5) is not None


def test_mesh_names_the_reason_for_the_lod_and_partition_parameters(tmp_path) -> None:
    """``add_mesh(additive_lod=…)`` must not answer like a typo.

    ``partition`` / ``additive_lod`` / ``substitutive_lod`` are real parameters on the
    other three adders, so a caller reaching for one on a mesh spelled a real feature
    correctly. Without a refusal of its own they fall into ``**attrs`` and come back as
    "Unknown node attribute … The viewer would silently ignore it. Remove it or use a
    supported attribute" — and ``partition`` even draws a "Did you mean 'absorption'?"
    hint. Right outcome, misleading reason: the same defect the ``kind=lod`` parent
    message carried, in the arm a user is far more likely to hit.
    """
    reasons = {
        "additive_lod": "holes",
        "substitutive_lod": "no producer",
        "partition": "boundary vertices",
    }
    with LuxarZarrCompiler(tmp_path / "params.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        for key, reason in reasons.items():
            with pytest.raises(ValueError) as excinfo:
                scene.add_mesh(f"m_{key}", _V, _F, **{key: True})
            message = str(excinfo.value)
            assert key in message
            assert reason in message
            assert "Unknown node attribute" not in message
            assert "Did you mean" not in message


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


# =============================================================================
# nD meshes — the spec's actual target shape
# =============================================================================


def test_4d_mesh_round_trips_with_non_leading_normal_dims(tmp_path) -> None:
    """A 4D mesh whose normals describe dims (1,2,3), not the leading three.

    This is the case `normal_dims` exists for and the shape the spec targets
    (3D surfaces whose hidden dimensions are discrete — time, channel), yet every
    other test here is 3D where `normal_dims == (0,1,2)` and an implicit
    "first three dimensions" would work by accident. For a `(t, x, y, z)` mesh the
    first three are `(t, x, y)`, so a normal against them is meaningless — only an
    explicit triple distinguishes right from wrong here.

    Asserts the full chain: `ndim` 4 on disk, `(V, 4)` vertices, `(V, 3)` normals,
    4-component `position_bounds`, and the triple surviving the round trip.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            # `t` range matches the data exactly (two timepoints, 0 and 1) so the
            # discrete-range validator stays quiet — its warning is about the test
            # fixture, not the mesh.
            Dimension(name="t", unit="px", range=(0, 1), step=1.0, display=False),
            Dimension(name="x", unit="um", range=(0, 20), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 20), step=1.0, display=True),
            Dimension(name="z", unit="um", range=(0, 20), step=1.0, display=True),
        ]
    )
    # One tetrahedron at each of two timepoints.
    tetra = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32)
    vertices = np.vstack(
        [np.hstack([np.full((4, 1), t, np.float32), tetra]) for t in (0.0, 1.0)]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [4, 5, 7],
            [4, 6, 7],
            [5, 6, 7],
        ],
        dtype=np.uint32,
    )
    normals = np.tile(_N, (2, 1))

    store = tmp_path / "nd.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        node = scene.add_mesh(
            "nd_surface", vertices, faces, normals=normals, normal_dims=(1, 2, 3)
        )
        assert node.normal_dims == [1, 2, 3]

    mesh = LuxarScene.load(store).get_mesh("nd_surface")
    assert mesh.vertices.shape == (8, 4)
    assert mesh.faces.shape == (8, 3)
    assert mesh.normals is not None and mesh.normals.shape == (8, 3)
    assert mesh.normal_dims == [1, 2, 3]
    assert mesh.metadata["ndim"] == 4
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)
    # Bounds span every dimension, not just the displayed three.
    assert len(mesh.metadata["position_bounds"]["min"]) == 4


def test_2d_mesh_is_accepted(tmp_path) -> None:
    """A planar 2D mesh writes and reads.

    2D geometry is a first-class authoring path in this codebase (the 2D gsplat
    demos), and a mesh needs only two axes to be a valid surface. Its normals can
    only name two dimensions plus... nothing — so normals are omitted here, which
    is exactly why `normal_dims` validation rejects a triple containing an
    out-of-range index for a 2D mesh.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(name="x", unit="um", range=(0, 20), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 20), step=1.0, display=True),
        ]
    )
    vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    store = tmp_path / "flat2d.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_mesh("plane", vertices, faces)

    mesh = LuxarScene.load(store).get_mesh("plane")
    assert mesh.vertices.shape == (4, 2)
    assert mesh.metadata["ndim"] == 2
    assert np.array_equal(mesh.faces, faces)


def test_normal_dims_out_of_range_for_a_2d_mesh(tmp_path) -> None:
    """A 2D mesh cannot carry normals: no valid triple of dimension indices exists.

    Confirms the ndim-relative bound is applied against the MESH's own
    dimensionality rather than a fixed 3, which a `(0,1,2)` default would hide.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(name="x", unit="um", range=(0, 20), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 20), step=1.0, display=True),
        ]
    )
    vertices = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    normals = np.zeros((4, 3), dtype=np.float32)
    normals[:, 2] = 1.0

    with LuxarZarrCompiler(tmp_path / "bad2d.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=dims)
        with pytest.raises(ValueError, match="out of range"):
            scene.add_mesh(
                "plane", vertices, faces, normals=normals, normal_dims=(0, 1, 2)
            )


def test_dim_order_permutes_vertex_columns_without_breaking_faces(tmp_path) -> None:
    """`dim_order` reorders vertex COLUMNS, so face indices stay valid untouched.

    The writer deliberately does not reorder `faces`, and this is the invariant that
    makes that correct: `dim_order` permutes dimensions (columns), never vertex rows,
    so a row index still names the same physical vertex afterwards. If it ever
    permuted rows instead, every face would silently point at the wrong vertices —
    a corruption with no error, so it is asserted on the physical points rather than
    on the arrays alone.

    Authored in (z, y, x) with distinct magnitudes per axis so a permutation is
    visible in the values rather than only in the shape.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(name="x", unit="um", range=(0, 30), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 30), step=1.0, display=True),
            Dimension(name="z", unit="um", range=(0, 30), step=1.0, display=True),
        ]
    )
    authored = np.array(
        [[1, 2, 3], [1, 2, 13], [1, 12, 3], [11, 2, 3]], dtype=np.float32
    )

    store = tmp_path / "do.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_mesh("m", authored, _F, dim_order=["z", "y", "x"])

    mesh = LuxarScene.load(store).get_mesh("m")

    assert np.array_equal(mesh.vertices, authored[:, ::-1])
    assert mesh.vertices.shape[0] == authored.shape[0]
    assert np.array_equal(mesh.faces, _F)
    # The invariant that matters: each face still names the same three points.
    for face_index, triangle in enumerate(_F):
        assert np.array_equal(
            authored[triangle][:, ::-1], mesh.vertices[mesh.faces[face_index]]
        ), f"face {face_index} no longer names its authored vertices"


@pytest.mark.filterwarnings("ignore:Dimension 't' has range")
def test_extend_to_all_on_a_mesh(tmp_path) -> None:
    """A mesh stays visible across a named non-displayed dimension.

    The discrete-range warning is filtered deliberately: a mesh authored at one
    `t` and extended across the whole axis is exactly the intended use, so the
    "data ends before the range" notice is correct but not what this test is about.

    `extend_to_all` is resolved by the shared scene helper, which takes the
    geometry type only for its warning text — so this checks the attr actually
    reaches the mesh node's zarr attrs, which is the part specific to this writer.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(name="t", unit="px", range=(0, 3), step=1.0, display=False),
            Dimension(name="x", unit="um", range=(0, 20), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 20), step=1.0, display=True),
            Dimension(name="z", unit="um", range=(0, 20), step=1.0, display=True),
        ]
    )
    vertices = np.hstack([np.zeros((4, 1), np.float32), _V])

    store = tmp_path / "ex.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_mesh("m", vertices, _F, extend_to_all=["t"])

    attrs = dict(zarr.open_group(store, mode="r")["m"].attrs)
    assert attrs["extend_to_all"] == ["t"]


@pytest.mark.parametrize("encoding_mode_name", ["AUTO", "PRECISION", "MEMORY"])
def test_topology_is_exact_under_every_encoding_mode(
    tmp_path, encoding_mode_name
) -> None:
    """`faces` must round-trip byte-exact in every encoding mode, MEMORY included.

    The encoder narrows integer arrays by observed value range, so a 400-vertex
    mesh's `faces` is stored as uint16 rather than uint32. That is lossless and
    reversible (the original dtype is recorded), but it is exactly the kind of
    space optimisation that would be catastrophic if it ever became lossy: a face
    index off by one is a different surface, and nothing downstream would report
    it. MEMORY mode is the one that quantizes most aggressively, so it is the case
    worth pinning.

    400 vertices deliberately — above the uint8 range, so any narrowing to a byte
    would be fatal and visible rather than coincidentally survivable.
    """
    from luxar.encoding import EncodingMode

    n_vertices = 400
    rng = np.random.default_rng(0)
    vertices = (rng.random((n_vertices, 3), dtype=np.float32) * 100).astype(np.float32)
    faces = np.stack(
        [
            np.arange(0, n_vertices - 2),
            np.arange(1, n_vertices - 1),
            np.arange(2, n_vertices),
        ],
        axis=1,
    ).astype(np.uint32)

    store = tmp_path / f"enc_{encoding_mode_name}.luxar.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=getattr(EncodingMode, encoding_mode_name)
    ) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("m", vertices, faces)

    mesh = LuxarScene.load(store).get_mesh("m")
    assert np.array_equal(mesh.faces, faces), (
        f"{encoding_mode_name} mode altered the topology"
    )
    # And the encoder must not have LUT-encoded it: the loader reads `faces` as raw
    # chunked zarr without resolving indirection, so a LUT would decode as garbage.
    encoding = dict(zarr.open_group(store, mode="r")["m"]["faces"].attrs).get(
        "encoding", {}
    )
    assert "lut" not in str(encoding.get("name", "")), (
        f"faces were LUT-encoded ({encoding}) despite allow_lut=False"
    )


def test_face_index_width_escalates_past_uint16(tmp_path) -> None:
    """A mesh with more than 65535 vertices must not have its indices stuck at uint16.

    The encoder picks integer width from the observed value range. Below the
    boundary `faces` legitimately stores as uint16; above it, staying there would
    wrap every index past 65535 and silently rewire the surface. Tested at both
    sides of the boundary so the assertion cannot pass by the width simply never
    narrowing at all.
    """
    results = {}
    # Tight bracket around the 65535 boundary — enough to prove escalation
    # without paying for a 70k-vertex write.
    for n_vertices in (65_000, 66_000):
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        vertices[:, 0] = np.arange(n_vertices, dtype=np.float32)
        faces = np.stack(
            [
                np.arange(0, n_vertices - 2),
                np.arange(1, n_vertices - 1),
                np.arange(2, n_vertices),
            ],
            axis=1,
        ).astype(np.uint32)

        store = tmp_path / f"w{n_vertices}.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("m", vertices, faces)

        mesh = LuxarScene.load(store).get_mesh("m")
        assert np.array_equal(mesh.faces, faces), f"topology broke at {n_vertices}"
        results[n_vertices] = zarr.open_group(store, mode="r")["m"]["faces"].dtype

    assert results[65_000] == np.uint16, "expected narrowing below the boundary"
    assert results[66_000].itemsize >= 4, (
        f"index width did not escalate above 65535 (got {results[66_000]})"
    )


def test_mesh_nested_under_a_plain_group_inside_a_lod_group_is_refused(
    tmp_path,
) -> None:
    """The indirection case that ONLY the finalize back-fill catches.

    `add_mesh`'s add-time guard inspects the IMMEDIATE parent, so a mesh whose
    parent is a plain group that itself sits inside a `kind=lod` group sails past
    it — and the resolved display type still walks down to `mesh`. This is the case
    that makes the multi-route guard necessary rather than redundant: with only the
    add-time check, this store would be written with `display_type="mesh"` and load
    nowhere.

    Unlike the direct case the failure surfaces at finalize, so it is raised from
    the compiler's context-manager exit rather than from `add_mesh`.
    """
    with pytest.raises(ValueError, match="display_type"):
        with LuxarZarrCompiler(tmp_path / "nested.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            inner = scene.add_lod_group("ladder").add_group("inner")
            inner.add_mesh("c0", _V, _F)


def test_mesh_under_nested_lod_groups_is_refused(tmp_path) -> None:
    """A lod-inside-lod ladder is refused at the innermost add, not silently nested."""
    with LuxarZarrCompiler(tmp_path / "ll.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        inner = scene.add_lod_group("l1").add_lod_group("l2")
        with pytest.raises(ValueError, match="kind=lod"):
            inner.add_mesh("c0", _V, _F)


# =============================================================================
# Reader robustness on a corrupt / externally produced store
# =============================================================================


def _handwritten_mesh_store(tmp_path, mutate):
    """Build a mesh store directly in zarr, bypassing the writer's validation."""
    path = tmp_path / "hand.luxar.zarr"
    root = zarr.open_group(path, mode="w")
    root.attrs.update(
        {
            "type": "scene",
            "luxar_version": "0.3",
            "scene_dimensions": {
                "dimensions": [
                    {
                        "name": n,
                        "unit": "um",
                        "range": [0.0, 10.0],
                        "step": 1.0,
                        "display": True,
                    }
                    for n in ("x", "y", "z")
                ]
            },
        }
    )
    node = root.create_group("m")
    node.attrs.update(
        {
            "type": "mesh",
            "n_vertices": 4,
            "n_faces": 4,
            "ndim": 3,
            "has_normals": False,
            "has_colors": False,
            "has_scalars": False,
            "shading": "flat",
            "double_sided": True,
            "ordering": "none",
            "position_bounds": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        }
    )
    node.create_dataset("vertices", data=_V)
    node.create_dataset("faces", data=_F)
    mutate(node)
    return path


def test_reader_rejects_a_malformed_normal_dims_triple(tmp_path) -> None:
    """`MeshData.normal_dims` is a TRIPLE; a 2-entry attr must not be handed back.

    A short list satisfies the `Optional[List[int]]` annotation while breaking the
    first consumer that indexes `[2]`, so the corrupt store would surface as an
    IndexError far from its cause. Built by writing zarr directly, since the
    writer's own validator makes this unreachable through `add_mesh`.
    """
    path = _handwritten_mesh_store(
        tmp_path,
        lambda node: (
            node.create_dataset("normals", data=np.zeros((4, 3), dtype=np.float32)),
            node.attrs.update({"has_normals": True, "normal_dims": [0, 1]}),
        ),
    )
    with pytest.raises(ValueError, match="malformed 'normal_dims'"):
        LuxarScene.load(path).get_mesh("m")


def test_reader_rejects_undersized_normals(tmp_path) -> None:
    """Normals are per-vertex, so a short array is a corrupt store, not a partial one."""
    path = _handwritten_mesh_store(
        tmp_path,
        lambda node: (
            node.create_dataset("normals", data=np.zeros((2, 3), dtype=np.float32)),
            node.attrs.update({"has_normals": True, "normal_dims": [0, 1, 2]}),
        ),
    )
    with pytest.raises(ValueError, match="2 normals for 4 vertices"):
        LuxarScene.load(path).get_mesh("m")


def test_reader_ignores_normal_dims_without_a_normals_array(tmp_path) -> None:
    """A stray `normal_dims` attr with no normals orients nothing, so it is dropped.

    Not an error: the attr alone is inert, and refusing the whole node over it would
    make an otherwise-readable mesh unreadable.
    """
    path = _handwritten_mesh_store(
        tmp_path,
        lambda node: node.attrs.update({"has_normals": True, "normal_dims": [0, 1, 2]}),
    )
    mesh = LuxarScene.load(path).get_mesh("m")
    assert mesh.normals is None
    assert mesh.normal_dims is None


def test_identical_vertices_are_not_deduplicated(tmp_path) -> None:
    """Coincident vertex rows must survive as distinct rows.

    This is the concrete hazard `deduplicate=False` exists for. The encoder can
    store a duplicate array as an `array_ref` with physical shape `(0, D)`; if it
    ever did that to `vertices`, every face index would address a row that is not
    there. An all-coincident mesh is the worst case — every row is a duplicate of
    every other — so it is the input that would trigger collapsing if it were
    enabled.
    """
    vertices = np.zeros((4, 3), dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)

    store = tmp_path / "dup.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("m", vertices, faces)

    mesh = LuxarScene.load(store).get_mesh("m")
    assert mesh.vertices.shape == (4, 3), (
        f"vertices collapsed to {mesh.vertices.shape} — face indices now dangle"
    )
    assert np.array_equal(mesh.faces, faces)
    # And the stored array is materialised, not an indirection the loader would
    # have to resolve (it reads `vertices` as raw chunked zarr).
    stored = zarr.open_group(store, mode="r")["m"]["vertices"]
    assert stored.shape == (4, 3), f"physical shape {stored.shape} is not (4, 3)"


@pytest.mark.parametrize(
    "label,vertices,faces",
    [
        (
            "single_triangle",
            np.eye(3, dtype=np.float32),
            np.array([[0, 1, 2]], np.uint32),
        ),
        (
            "all_degenerate_faces",
            np.eye(3, dtype=np.float32),
            np.array([[0, 0, 0], [1, 1, 1]], np.uint32),
        ),
        (
            "orphan_vertex",
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]], np.float32),
            np.array([[0, 1, 2]], np.uint32),
        ),
        (
            "huge_coordinates",
            (np.eye(3, dtype=np.float32) * 1e30),
            np.array([[0, 1, 2]], np.uint32),
        ),
    ],
)
def test_degenerate_meshes_round_trip(tmp_path, label, vertices, faces) -> None:
    """Nasty-but-valid meshes write and read with topology intact.

    None of these is an error: a zero-area triangle, a vertex no face references,
    and extreme coordinates are all things real decimation and isosurface output
    produce. Rejecting them would refuse legitimate data, so the contract is that
    they round-trip rather than that they are caught.
    """
    store = tmp_path / f"{label}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("m", vertices, faces)

    mesh = LuxarScene.load(store).get_mesh("m")
    assert np.array_equal(mesh.faces, faces)
    assert mesh.vertices.shape == vertices.shape


# =============================================================================
# Per-vertex labels and image labels (CSR)
# =============================================================================


def test_labels_round_trip(tmp_path) -> None:
    """Per-vertex hover labels are CSR-written and flagged.

    Labels are per-VERTEX for a mesh, matching Lines (which is also per-vertex
    rather than per-segment) — the count must line up with the vertex array, not
    the face array.
    """
    store = _write(tmp_path, name="lab", labels=["a", "b", "c", "d"])
    node = zarr.open_group(store, mode="r")["lab"]

    assert dict(node.attrs)["has_labels"] is True
    assert "label_offsets" in node
    assert "label_bytes" in node

    mesh = LuxarScene.load(store).get_mesh("lab")
    assert mesh.metadata["has_labels"] is True


def test_labels_length_must_match_vertex_count(tmp_path) -> None:
    """A label per FACE rather than per vertex is the natural mistake, and fails."""
    with LuxarZarrCompiler(tmp_path / "badlab.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="[Ll]abels"):
            scene.add_mesh("m", _V, _F, labels=["only", "two"])


def test_image_labels_round_trip(tmp_path) -> None:
    """Per-vertex hover thumbnails write through the image-label CSR path."""
    thumbnails = [
        np.full((2, 2, 3), fill_value=i * 60, dtype=np.uint8) for i in range(4)
    ]
    store = _write(tmp_path, name="img", image_labels=thumbnails)
    node = zarr.open_group(store, mode="r")["img"]

    assert dict(node.attrs)["has_image_labels"] is True
    assert "image_label_offsets" in node
    assert "image_label_bytes" in node


@pytest.mark.parametrize(
    "kwargs,pattern,test_id",
    [
        (
            {"colors": np.ones((4, 3), np.float32), "colormap": "viridis"},
            "both 'colors' and 'colormap'",
            "colors_plus_colormap",
        ),
        (
            {"scalars": np.zeros(4, np.float32)},
            "requires a 'colormap'",
            "scalars_without_colormap",
        ),
    ],
)
def test_appearance_mutual_exclusivity(tmp_path, kwargs, pattern, test_id) -> None:
    """Colour sources are mutually exclusive, matching the sibling adders.

    `colors` and `colormap` both decide the surface's colour, so supplying both
    leaves the winner to writer ordering rather than to the author; `scalars`
    without a `colormap` has no LUT to map through and would render as nothing.
    """
    with LuxarZarrCompiler(tmp_path / f"{test_id}.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match=pattern):
            scene.add_mesh("m", _V, _F, **kwargs)


def test_vertices_must_be_2d_at_the_adder(tmp_path) -> None:
    """A 1D vertices array is rejected by the adder before the writer sees it."""
    with LuxarZarrCompiler(tmp_path / "v1d.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match=r"shape \(V, D\)"):
            scene.add_mesh("m", np.zeros(9, dtype=np.float32), _F)


def test_double_sided_must_be_a_bool(tmp_path) -> None:
    """A truthy non-bool would be persisted as-is and read as a lie by the viewer."""
    with LuxarZarrCompiler(tmp_path / "dsb.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="double_sided must be a bool"):
            scene.add_mesh("m", _V, _F, double_sided="yes")


def test_per_vertex_color_array_round_trips(tmp_path) -> None:
    """An ndarray `colors` (not a broadcast tuple) takes the per-vertex write path.

    The broadcast-tuple and per-vertex-array paths differ inside the writer — only
    the array one runs the shape validator and stores one row per vertex — so both
    need exercising. RGBA here so the per-vertex alpha column is covered too.
    """
    colors = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.25],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    store = _write(tmp_path, name="pvc", colors=colors)
    mesh = LuxarScene.load(store).get_mesh("pvc")

    assert mesh.colors is not None
    assert mesh.colors.shape == (4, 4), (
        f"expected per-vertex RGBA, got {mesh.colors.shape}"
    )
    assert mesh.metadata["has_colors"] is True


def test_mesh_node_properties_reflect_what_was_written(tmp_path) -> None:
    """Every `Mesh` property must read back what the writer actually stamped.

    Exercised through a fully-populated node rather than per-property, because the
    failure mode these guard against is a property reading the WRONG metadata key —
    which returns a plausible default instead of raising, so only comparing against
    known-written values catches it.
    """
    thumbnails = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    with LuxarZarrCompiler(tmp_path / "props.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_mesh(
            "full",
            _V,
            _F,
            normals=_N,
            normal_dims=(0, 1, 2),
            scalars=np.linspace(0, 1, 4).astype(np.float32),
            colormap="viridis",
            shading="smooth",
            double_sided=False,
            labels=["a", "b", "c", "d"],
            image_labels=thumbnails,
        )

        assert node.n_elements == 4
        assert node.n_faces == 4
        assert node.has_normals is True
        assert node.normal_dims == [0, 1, 2]
        assert node.has_colors is False  # scalars+colormap, not explicit colors
        assert node.has_scalars is True
        assert node.has_labels is True
        assert node.has_image_labels is True
        assert node.shading == "smooth"
        assert node.double_sided is False
        assert node.ordering == "none"


def test_add_mesh_refuses_one_dimensional_vertices(tmp_path):
    """A triangle needs two dimensions to enclose any area.

    Deliberately stricter than `add_points` / `add_lines`, which take whatever width
    they are given: a 1D scatter and 1D segments are both meaningful, a 1D triangle is
    not. Without this the mesh writes and loads cleanly and then renders NOTHING, with no
    diagnostic anywhere — every face is collinear. The viewer's Stage-1 preflight mirrors
    this floor, so the two sides agree on what is admissible.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [Dimension(name="x", unit="um", range=(0, 20), step=1.0, display=True)]
    )
    vertices = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    with LuxarZarrCompiler(tmp_path / "collinear.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=dims)
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            scene.add_mesh("line-ish", vertices, faces)
