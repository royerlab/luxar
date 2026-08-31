"""Tests for the Mesh node: add_mesh, round-trip, and the §9 exclusions.

Covers the three things that can only be checked through the full stack — the
store a write actually produces, the values a read gives back, and the paths mesh
must be refused from. The pure validators live in
``validation/tests/test_mesh_validation.py``.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import create_array
from luxar.core.mesh import Mesh, ShadingMode
from luxar.io import LuxarScene, MeshData
from luxar.io._compiler.geometry_writers.mesh import VALID_SHADING_MODES

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

_KTX2_IDENTIFIER = b"\xabKTX 20\xbb\r\n\x1a\n"


def _ktx2_bytes(supercompression_scheme: int) -> bytes:
    payload = bytearray(64)
    payload[: len(_KTX2_IDENTIFIER)] = _KTX2_IDENTIFIER
    payload[44:48] = supercompression_scheme.to_bytes(4, "little")
    return bytes(payload)


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
        ({"shading": "none"}, "none", "explicit_unlit_kept"),
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


def test_shading_mode_type_matches_the_writer_contract() -> None:
    """The public read type must cover every shading value the writer accepts."""
    assert set(get_args(ShadingMode)) == set(VALID_SHADING_MODES)


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


def test_mesh_reports_viewer_default_blending_mode(tmp_path) -> None:
    with LuxarZarrCompiler(tmp_path / "default-mode.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        mesh = scene.add_mesh("surface", _V, _F)

        assert mesh.blending_mode == "opaque"
        assert "blending_mode" not in mesh.attrs


def test_mesh_under_a_lod_group_is_accepted(tmp_path) -> None:
    """A mesh IS a valid substitutive level, now that a producer exists.

    This was a refusal until `luxar.mesh.decimate` landed, and the refusal was
    correct at the time: a `kind=lod` group's levels must each be an independently
    renderable stand-in for the finer one, and nothing could produce a coarser
    surface. Hand-assembling the ladder is the same thing `substitutive_lod=` does
    internally, so it has to work.

    The ADDITIVE flavour is untouched by this and remains impossible — see
    `test_additive_ladder_is_still_refused_with_its_own_reason`.
    """
    with LuxarZarrCompiler(tmp_path / "lod.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        lod = scene.add_lod_group("ladder")
        lod.add_mesh("child_0", _V, _F, coverage_fraction=0.0)


def test_mesh_under_a_mesh_partition_group_is_allowed(tmp_path) -> None:
    """Mesh is partition-capable, so a mesh child of a mesh partition is legal.

    This used to raise. It must not any more, because it is precisely the shape
    ``add_mesh(partition=...)`` writes — a guard here would refuse the adder's
    own output.
    """
    with LuxarZarrCompiler(tmp_path / "part.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        part = scene.add_partition_group("parts", display_type="mesh", max_elements=100)
        assert part.add_mesh("part_0", _V, _F) is not None


@pytest.mark.parametrize("declared", ["points", "lines", "gsplats"])
def test_mesh_under_a_partition_of_another_type_is_rejected(tmp_path, declared) -> None:
    """A partition's declared ``display_type`` must not be made a lie.

    Lifting the blanket ``kind=partition`` refusal must not lift THIS one: a
    partition is homogeneous, and nothing re-checks that before the store is
    finalized, so a mesh dropped into a ``points`` partition would write clean
    and load as a layer claiming to be points.
    """
    with LuxarZarrCompiler(tmp_path / f"part_{declared}.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        part = scene.add_partition_group(
            "parts", display_type=declared, max_elements=100
        )
        with pytest.raises(ValueError, match="kind=partition group declared"):
            part.add_mesh("part_0", _V, _F)


def test_non_mesh_under_a_mesh_partition_group_is_rejected(tmp_path) -> None:
    """The homogeneity rule is symmetric — the INVERSE pairing raises too.

    Refusing a mesh under a ``points`` partition while accepting a points leaf
    under a ``mesh`` partition would enforce homogeneity in one direction only,
    and the direction left open is the one this feature newly makes reachable: a
    ``display_type='mesh'`` partition could not even be built before mesh became
    partition-capable.
    """
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    with LuxarZarrCompiler(tmp_path / "mesh_part.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        part = scene.add_partition_group("parts", display_type="mesh", max_elements=100)
        with pytest.raises(ValueError, match="kind=partition group declared"):
            part.add_points("part_0", pos)
        with pytest.raises(ValueError, match="kind=partition group declared"):
            part.add_lines("part_1", pos, np.ones(2, dtype=np.float32))


def test_only_the_arbitrary_order_half_of_additive_is_refused(tmp_path) -> None:
    """The refusal narrowed to what it always meant, and kept naming ITS reason.

    The message spent months telling users that SUBSTITUTIVE LOD reduces
    independent elements and is therefore impossible for a surface; then that an
    ADDITIVE ladder is impossible at all. Both were too broad, and the true
    statement is narrower than either: a prefix of an *arbitrarily ordered* index
    buffer is a holed surface. A spatially coherent REVEAL has no such problem —
    every prefix is a contiguous partial surface — so ``additive_lod=True`` writes a
    ladder now, and only the non-reveal methods are refused.

    Pinned as a PAIR so neither half can rot alone: the accepted spelling must
    write, and the refused one must still say "holes" rather than something
    vaguer.
    """
    with LuxarZarrCompiler(tmp_path / "why.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        # The reveal is accepted (the tetrahedron is too small to ladder into 4
        # non-empty face groups plus stay a ladder, so this only pins acceptance —
        # the ladder's shape is pinned by the reveal tests at the end of the file).
        assert scene.add_mesh("m", _V, _F, additive_lod=True) is not None

        with pytest.raises(ValueError) as excinfo:
            scene.add_mesh("m_random", _V, _F, additive_lod={"method": "random"})

    message = str(excinfo.value)
    assert "additive_lod" in message
    # Excluded on principle: a prefix of an arbitrary order is a holed surface.
    assert "HOLES" in message or "holes" in message
    # And NOT by appeal to a missing producer — that was the substitutive arm's
    # reason, and it no longer applies to anything.
    assert "no producer" not in message
    # It must point at the flavour that DOES make a surface coarser.
    assert "substitutive_lod" in message


def test_mesh_rejects_hand_supplied_energy_stamps(tmp_path) -> None:
    """Spec §9.1: a mesh must never carry an additive ladder's energy stamps.

    ``level_stats`` / ``lod_stats`` are in the geometry-blind allow-list in
    ``io/_compiler/node_common.py``, so before this guard they wrote clean on a mesh.
    The viewer's ``energyCompensation`` scales brightness by ``1/energy_fraction_cum``
    and is gated on the BLENDING MODE (``additive``/``luminous``/``volumetric``), not
    on geometry type — and mesh supports two of those. Brightening a dimmer splat
    prefix is right; brightening a holed surface is not.

    Prophylactic rather than a live-bug fix: the mesh commit never stamps
    ``committedEnergyFraction``, so the compensation factor is 1 today. Substitutive
    LOD has already removed the other latch (a mesh CAN be in a ``kind=lod`` group
    now), so this test is what keeps the rule true.
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


def test_every_sibling_structural_parameter_is_bound_by_name() -> None:
    """No structural knob may reach ``**attrs`` and answer like a typo.

    The hazard this pins: a parameter ``add_mesh`` does not BIND lands in ``**attrs``
    and comes back as "Unknown node attribute … The viewer would silently ignore it.
    Remove it or use a supported attribute" — a typo diagnostic for a caller who
    spelled a real sibling feature correctly.

    All three are bound now (``substitutive_lod`` when the decimator landed,
    ``partition`` when the splitter did, ``additive_lod`` with the reveal ladder), so
    ``_UNSUPPORTED_STRUCTURE_PARAMS`` is empty. The table survives as the extension
    point, and this test is what makes adding a FOURTH unbound knob a visible
    decision rather than a silent typo diagnostic: either bind it, or put it in the
    table with its reason.
    """
    import inspect

    from luxar.core.group.adders.mesh import _UNSUPPORTED_STRUCTURE_PARAMS
    from luxar.core.group.group import Group

    parameters = inspect.signature(Group.add_mesh).parameters
    for key in ("additive_lod", "substitutive_lod", "partition"):
        assert key in parameters, f"{key} must be bound by name, not ride **attrs"
        assert key not in _UNSUPPORTED_STRUCTURE_PARAMS

    # Anything still in the table must carry a reason worth printing — the whole
    # point of the table over the generic unknown-attr message.
    for key, reason in _UNSUPPORTED_STRUCTURE_PARAMS.items():
        assert key not in parameters, f"{key} is bound, so its table row is dead"
        assert len(reason) > 20


def test_partition_group_accepts_mesh_display_type(tmp_path) -> None:
    """The capability table now says mesh partitions, so this route opens with it.

    Kept as the inverse of the LOD test below: the two guards read identically
    and are driven by the same table, so pinning both is what shows the table is
    actually consulted rather than the answer hardcoded.
    """
    with LuxarZarrCompiler(tmp_path / "pd.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        group = scene.add_partition_group("p", display_type="mesh", max_elements=100)
        assert group.attrs["display_type"] == "mesh"


def test_lod_group_accepts_explicit_mesh_display_type(tmp_path) -> None:
    """``display_type='mesh'`` is now a valid kind=lod group, by every route.

    ``display_type`` is an accepted node attr, so it rides in through ``**attrs``
    and reaches zarr without passing the finalize back-fill at all. It used to be
    refused there; the gate still exists and still refuses a LOD-less type, but no
    contract type is LOD-less any more (see
    ``typing_utils/tests/test_geometry_capabilities.py``).
    """
    with LuxarZarrCompiler(tmp_path / "ld.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        group = scene.add_lod_group("l", display_type="mesh")
        group.add_mesh("child_0", _V, _F, coverage_fraction=1.0)


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


# --- dim_order and face winding (#2141) -------------------------------------
#
# `dim_order` renumbers the vertex COLUMNS. `normals` needs no companion transform
# — `normal_dims` names SCENE dimension indices, so the components are already
# expressed in the destination frame (the docstring's own example, "for a
# (t, x, y, z) mesh those are (t, x, y)", is about the stored layout, and
# `demo_lsystem_forest` says so inline: "The three scene dims the normals
# describe"). This differs from gsplats, whose Cholesky factors ARE authored in
# the source frame and so must be carried through the map.
#
# Face winding is the part `dim_order` can invalidate, and the part Luxar
# deliberately does NOT repair: `cross(Ra, Rb) = det(R)·R·cross(a, b)`, so an
# orientation-reversing permutation negates a triangle's geometric normal while
# its corner order is untouched. Whether that is WRONG depends on which frame the
# caller wound in, which only they know — so the writer reports and leaves the
# data alone. These tests pin the report and, just as importantly, pin that
# nothing is silently rewritten.

_XYZ = ("x", "y", "z")


def _xyz_dims():
    """A plain 3D (x, y, z) scene — the frame every case below maps ONTO."""
    from luxar.core.dimensions import Dimension, Dimensions

    return Dimensions(
        [
            Dimension(name=n, unit="um", range=(0, 30), step=1.0, display=True)
            for n in _XYZ
        ]
    )


_TRI_V = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
_TRI_F = np.array([[0, 1, 2]], dtype=np.uint32)
_TRI_N = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (3, 1))


def _write_tri(tmp_path, name, **kwargs):
    store = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=_xyz_dims())
        scene.add_mesh(name, _TRI_V, _TRI_F, **kwargs)
    return LuxarScene.load(store).get_mesh(name)


@pytest.mark.parametrize(
    "dim_order,test_id",
    [
        (["x", "y", "z"], "identity"),
        (["y", "z", "x"], "even_cycle"),
        (["z", "y", "x"], "odd_reversal"),
        (["y", "x", "z"], "odd_swap"),
    ],
)
def test_dim_order_never_rewrites_faces_or_normals(
    tmp_path, dim_order, test_id
) -> None:
    """`dim_order` touches vertex columns and NOTHING else.

    The guard against a well-meant "fix". Repairing winding automatically looks
    right until you notice `normal_dims` names SCENE dimensions: a caller who
    followed that contract wound against the scene frame and is already correct,
    so flipping their faces would corrupt working data. Same for permuting normal
    components — they are already in the destination frame.

    So the invariant is the strong one: faces and normals come out byte-identical
    to what went in, for every permutation including the reversing ones.
    """
    mesh = _write_tri(
        tmp_path, "m", normals=_TRI_N, normal_dims=[0, 1, 2], dim_order=dim_order
    )
    assert np.array_equal(mesh.faces, _TRI_F), f"{test_id}: faces were rewritten"
    assert np.array_equal(mesh.normals, _TRI_N), f"{test_id}: normals were rewritten"
    assert mesh.normal_dims == [0, 1, 2], f"{test_id}: normal_dims was rewritten"


@pytest.mark.parametrize(
    "dim_order,expect_warning,test_id",
    [
        (["x", "y", "z"], False, "identity_is_orientation_preserving"),
        (["y", "z", "x"], False, "even_cycle_is_orientation_preserving"),
        (["z", "y", "x"], True, "reversal_flips_handedness"),
        (["y", "x", "z"], True, "single_swap_flips_handedness"),
    ],
)
def test_dim_order_warns_exactly_when_handedness_reverses(
    tmp_path, capsys, dim_order, expect_warning, test_id
) -> None:
    """The lint fires on the reversing permutations and only those.

    The even cases are not padding: a lint that fired on any non-identity
    `dim_order` would pass every reversing case and fail these, and it would cry
    wolf on the shipped `demo_lsystem_forest`, whose `dim_order` is
    orientation-PRESERVING.
    """
    _write_tri(
        tmp_path,
        f"m_{test_id}",
        normals=_TRI_N,
        normal_dims=[0, 1, 2],
        dim_order=dim_order,
        double_sided=False,
    )
    warned = "reverses handedness" in capsys.readouterr().out
    assert warned is expect_warning, test_id


def test_dim_order_winding_warning_includes_a_double_sided_mesh(
    tmp_path, capsys
) -> None:
    """Stored-normal shading keeps winding observable when both sides draw.

    ``gl_FrontFacing`` still chooses the stored normal's sign on a double-sided
    material, so reversed winding flips the shading gradient even though coverage
    is unchanged. ``double_sided`` defaults true, making this the common case.
    """
    _write_tri(
        tmp_path,
        "m",
        normals=_TRI_N,
        normal_dims=[0, 1, 2],
        dim_order=["z", "y", "x"],
        double_sided=True,
    )
    assert "reverses handedness" in capsys.readouterr().out


def test_dim_order_winding_warning_is_silent_without_a_winding_frame(
    tmp_path, capsys
) -> None:
    """No normals means no declared frame, so handedness is undecidable.

    Spec §3.2: `sorted(normal_dims)` is the ONLY signal for which three axes the
    author wound against. Warning on a guess would be noise, and the viewer
    already renders such a mesh `DoubleSide` regardless of `double_sided`.
    """
    _write_tri(tmp_path, "m", dim_order=["z", "y", "x"], double_sided=False)
    assert "reverses handedness" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "normal_dims,expected",
    [
        (
            ["a", "b", "c"],
            "normal_dims: Entry 0 must be an integer dimension index, got 'a' (str)",
        ),
        (
            3,
            "normal_dims: Expected a sequence of 3 dimension indices, got int",
        ),
    ],
)
def test_dim_order_winding_lint_defers_malformed_normal_dims_to_validator(
    tmp_path, normal_dims, expected
) -> None:
    """The advisory lint must not replace the writer's actionable refusal."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        _write_tri(
            tmp_path,
            "m",
            normals=_TRI_N,
            normal_dims=normal_dims,
            dim_order=["z", "y", "x"],
        )


def test_reveal_ladder_budget_is_charged_as_a_sum(tmp_path, monkeypatch) -> None:
    """A ladder is charged on its levels' SUM, not on the flat surface.

    The one place the flat write-time check is *multiplicatively* short rather
    than merely incomplete: the viewer concatenates `additive_<i>` levels into one
    node's buffers and keeps all of them resident, so it charges the total. A
    shell ladder duplicates every boundary vertex, so the total exceeds the flat
    mesh by a factor that grows with the level count — meaning an under-budget
    surface could still write a ladder the viewer refuses.

    The ceiling is shrunk rather than the fixture grown, and to a value the FLAT
    surface fits inside so the assertion can only be satisfied by the sum: a build
    that checked levels individually, or only the authored mesh, admits this.
    """
    from luxar.validation.base import mesh_decoded_value_count

    flat_values = mesh_decoded_value_count(
        _V.shape[0], _V.shape[1], int(_F.size // 3), normals=_N
    )
    # Between one flat surface and the ladder's duplicated total.
    monkeypatch.setattr(
        "luxar.typing_utils.constants.MESH_DECODE_BUDGET_BYTES",
        int(flat_values * 4 * 1.5),
        raising=True,
    )
    with pytest.raises(ValueError, match="levels decode to"):
        store = tmp_path / "ladder.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "m",
                _V,
                _F,
                normals=_N,
                normal_dims=[0, 1, 2],
                additive_lod={"n_lods": 2},
            )
    root = zarr.open_group(str(store), mode="r")
    assert list(root.groups()) == [], "the refused ladder was partly written"


def test_dim_order_winding_warning_is_silent_when_frame_has_no_preimage(
    tmp_path, capsys
) -> None:
    """A constant-filled frame axis is undecidable, not an authoring failure."""
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(
                name="t",
                unit="s",
                range=(0, 0),
                step=1.0,
                display=False,
                discrete=True,
            ),
            Dimension(name="x", unit="um", range=(0, 30), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 30), step=1.0, display=True),
            Dimension(name="z", unit="um", range=(0, 30), step=1.0, display=True),
        ]
    )
    store = tmp_path / "filled_frame.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_mesh(
            "m",
            _TRI_V,
            _TRI_F,
            normals=_TRI_N,
            normal_dims=[0, 1, 2],
            dim_order=["x", "y", "z"],
            fill={"t": 0.0},
            double_sided=False,
        )
    assert LuxarScene.load(store).get_mesh("m").normal_dims == [0, 1, 2]
    assert "reverses handedness" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "route_kwargs,test_id",
    [
        ({}, "flat_leaf"),
        ({"partition": {"max_elements": 2}}, "partition"),
        ({"substitutive_lod": {"levels": 2}}, "substitutive_lod"),
        ({"additive_lod": {"n_lods": 2}}, "additive_lod"),
    ],
)
def test_dim_order_winding_lint_reaches_every_structural_route(
    tmp_path, capsys, route_kwargs, test_id
) -> None:
    """One call site serves all four routes — this is what proves it.

    `add_mesh_impl` runs the lint once, above the structural branches, and relies
    on every recursive re-entry passing `dim_order=None` so it fires exactly once.
    That is an argument, not a guarantee: a regression that moved the call below a
    branch would leave `partition=`, `substitutive_lod=` and `additive_lod=`
    authoring in silence while the flat leaf stayed green — a lint that is absent
    on three quarters of the API and looks fine in every other test.

    Uses the closed tetrahedron rather than the single triangle: `partition=`
    needs enough faces to split and the LOD routes need something to coarsen.
    """
    store = tmp_path / f"{test_id}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "m",
            _V,
            _F,
            normals=_N,
            normal_dims=[0, 1, 2],
            dim_order=["z", "y", "x"],
            **route_kwargs,
        )
    assert "reverses handedness" in capsys.readouterr().out, (
        f"{test_id}: the winding lint did not fire on this route"
    )


def test_dim_order_scene_semantics_hold_for_normal_dims(tmp_path) -> None:
    """`normal_dims` indexes the SCENE dimensions, so it may exceed the authored width.

    Pins the contract my own first reading of this code got backwards, and that
    `demo_lsystem_forest` depends on: it authors 4 columns
    (`dim_order=["season", "x", "y", "z"]`) and passes `normal_dims=[2, 3, 4]` —
    scene indices, one of which is larger than any authored column index. A build
    that treated `normal_dims` as authored columns rejects that demo outright.
    """
    from luxar.core.dimensions import Dimension, Dimensions

    dims = Dimensions(
        [
            Dimension(name="t", unit="s", range=(0, 3), step=1.0, display=False),
            Dimension(name="x", unit="um", range=(0, 30), step=1.0, display=True),
            Dimension(name="y", unit="um", range=(0, 30), step=1.0, display=True),
            Dimension(name="z", unit="um", range=(0, 30), step=1.0, display=True),
        ]
    )
    store = tmp_path / "scene_dims.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        # Three authored columns; normal_dims names scene 1..3, above that width.
        scene.add_mesh(
            "m",
            _TRI_V,
            _TRI_F,
            normals=_TRI_N,
            normal_dims=[1, 2, 3],
            dim_order=["x", "y", "z"],
            fill={"t": 0.0},
        )
    assert LuxarScene.load(store).get_mesh("m").normal_dims == [1, 2, 3]


def test_add_mesh_refuses_a_mesh_over_the_viewers_decode_budget(
    tmp_path, monkeypatch
) -> None:
    """The budget gate is WIRED into `add_mesh`, not merely defined.

    The arithmetic is unit-tested in `test_mesh_validation.py`; what cannot be
    checked there is whether anything calls it. Authoring a genuinely over-budget
    mesh would mean allocating half a gigabyte of test fixture, so the ceiling is
    shrunk instead — the validator reads the constant at call time, so a tiny
    budget makes a four-vertex tetrahedron over-budget and proves the path runs.

    Asserts on the store as well as the exception: a fail-fast gate that raises
    AFTER writing arrays would leave a half-built node behind, which is the
    failure mode `validate_mesh_arrays` exists to prevent. Checked on disk rather
    than through `LuxarScene.load`, which refuses the whole store as incomplete
    (the writer exited before finalizing) and so cannot tell a node that was
    never created from one that was half-written.
    """
    monkeypatch.setattr(
        "luxar.typing_utils.constants.MESH_DECODE_BUDGET_BYTES", 16, raising=True
    )
    store = tmp_path / "over.luxar.zarr"
    with pytest.raises(ValueError, match="over the viewer"):
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("m", _V, _F)
    root = zarr.open_group(str(store), mode="r")
    assert "m" not in dict(root.groups()), "the refused node was partly written"


def test_slab_tolerance_round_trips_under_exactly_that_key(tmp_path) -> None:
    """`slab_tolerance` reaches zarr spelled the way the viewer reads it.

    The viewer's `MeshMetadata` is a closed TypeScript interface, so its read
    site is compile-checked — but nothing checks that the key Python WRITES is
    the key TypeScript declares. `check-contract` does not model node appearance
    attrs at all, and no fixture authors one, so the two spellings agree by
    review alone. This pins the Python half against a rename, and names its twin
    so a future rename has somewhere to look.

    TWIN: `slab_tolerance` in `packages/luxar-viewer/src/types/mesh.ts`
    (`MeshMetadata`), consumed at
    `data/scene-loader/process/data-processor-mesh.ts` as the
    `meshSlabTolerance` tolerance option.
    """
    store = _write(tmp_path, slab_tolerance=2.5)
    node = zarr.open_group(str(store), mode="r")["m"]
    assert node.attrs["slab_tolerance"] == 2.5


@pytest.mark.parametrize(
    "value,test_id",
    [(0.0, "zero_would_render_nothing"), (-1.0, "negative"), (float("nan"), "nan")],
)
def test_slab_tolerance_rejects_a_non_positive_value(tmp_path, value, test_id) -> None:
    """Zero is the one that matters, and it is refused for a concrete reason.

    A zero slab reduces mesh's whole-triangle membership test to exact float
    equality with the slice plane, so the node renders NOTHING — the same trap
    that stops mesh reusing the Lines tolerance arm (spec §5.2.1). Accepting it
    would hand the user a silent blank node.
    """
    with pytest.raises(ValueError, match="Slab tolerance"):
        _write(tmp_path, name=f"m_{test_id}", slab_tolerance=value)


def test_slab_tolerance_is_refused_on_the_other_geometry_types(tmp_path) -> None:
    """Mesh-only, and enforced by the same guard as the shading controls.

    It is a loading knob rather than an appearance one, so it rides in the set
    named `MESH_ONLY_APPEARANCE_ATTRS` on a technicality. This pins the
    behaviour that name is a technicality ABOUT: a points node must still refuse
    it, or a user would silently author a no-op.
    """
    store = tmp_path / "pts.luxar.zarr"
    with pytest.raises(ValueError, match="slab_tolerance"):
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("p", _V, slab_tolerance=2.0)


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


def test_add_mesh_rejects_geometry_and_texture_over_the_combined_budget(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two individually admissible terms must not author an unloadable node."""
    monkeypatch.setattr(
        "luxar.typing_utils.constants.MESH_DECODE_BUDGET_BYTES", 1_000, raising=True
    )
    store = tmp_path / "combined-budget.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="over the viewer"):
            scene.add_mesh(
                "m",
                _V,
                _F,
                uvs=np.zeros((len(_V), 2), dtype=np.float32),
                texture=np.zeros(69, dtype=np.uint8),
                texture_encoding="png",
                texture_width=16,
                texture_height=16,
                texture_channels=3,
            )
        assert "m" not in zarr.open_group(store, mode="r")


def test_mesh_nested_under_a_plain_group_inside_a_lod_group_is_accepted(
    tmp_path,
) -> None:
    """The indirection case, which the finalize back-fill resolves to `mesh`.

    A mesh whose parent is a plain group that itself sits inside a `kind=lod` group
    is invisible to `add_mesh`'s immediate-parent guard, and the back-fill still
    walks down to `display_type="mesh"`. That combination used to be the argument
    for the multi-route guard; now it simply resolves to a display type that is
    legal, and the store finalizes.
    """
    with LuxarZarrCompiler(tmp_path / "nested.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        inner = scene.add_lod_group("ladder").add_group("inner")
        inner.add_mesh("c0", _V, _F)


def test_mesh_under_nested_lod_groups_is_accepted(tmp_path) -> None:
    """A lod-inside-lod ladder of meshes writes, the same as for the other types."""
    with LuxarZarrCompiler(tmp_path / "ll.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        inner = scene.add_lod_group("l1").add_lod_group("l2")
        inner.add_mesh("c0", _V, _F, coverage_fraction=1.0)


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
    create_array(node, "vertices", data=_V)
    create_array(node, "faces", data=_F)
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
            create_array(node, "normals", data=np.zeros((4, 3), dtype=np.float32)),
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
            create_array(node, "normals", data=np.zeros((2, 3), dtype=np.float32)),
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


# =============================================================================
# partition= (spec §9's lifted exclusion)
# =============================================================================

# A 6x6 vertex grid triangulated into 50 faces: fully connected, so ANY interior
# cut is forced to duplicate boundary vertices. The tetrahedron above is too
# small to split meaningfully.
_GX, _GY = np.meshgrid(np.arange(6, dtype=np.float32), np.arange(6), indexing="ij")
_GRID_V = np.stack([_GX.ravel(), _GY.ravel(), np.zeros(36, np.float32)], axis=1)
_GRID_F = np.asarray(
    [
        tri
        for i in range(5)
        for j in range(5)
        for tri in (
            [i * 6 + j, i * 6 + j + 1, (i + 1) * 6 + j],
            [i * 6 + j + 1, (i + 1) * 6 + j + 1, (i + 1) * 6 + j],
        )
    ],
    dtype=np.uint32,
)


def _write_partitioned(tmp_path, name="pm", **kwargs):
    store = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(name, _GRID_V, _GRID_F, **kwargs)
    return store


def test_partitioned_mesh_writes_a_partition_group(tmp_path) -> None:
    store = _write_partitioned(tmp_path, partition={"max_elements": 10})
    root = zarr.open_group(str(store), mode="r")
    node = root["pm"]
    assert node.attrs["kind"] == "partition"
    assert node.attrs["display_type"] == "mesh"
    assert node.attrs["max_elements"] == 10
    assert "position_bounds" in node.attrs
    parts = [k for k in node.keys() if k.startswith("part_")]
    assert len(parts) > 1


def test_partition_conserves_every_triangle(tmp_path) -> None:
    """The parts' faces sum to the input's — no triangle dropped or duplicated.

    Read back through the public reader, so this covers the write and the read,
    not just the in-memory split.
    """
    store = _write_partitioned(tmp_path, partition={"max_elements": 10})
    scene = LuxarScene.load(str(store))
    total = 0
    for path in scene.list_meshes():
        total += int(np.asarray(scene.get_mesh(path).faces).reshape(-1, 3).shape[0])
    assert total == _GRID_F.shape[0]


def test_partitioned_mesh_round_trips_the_same_surface(tmp_path) -> None:
    """The union of the parts is the SAME set of triangles, in the same places.

    The invariant that matters end-to-end: compare the multiset of corner-position
    triples, which is invariant to how the parts renumbered their vertices and to
    what order the BSP emitted them in.
    """
    store = _write_partitioned(tmp_path, partition={"max_elements": 10})
    scene = LuxarScene.load(str(store))

    got = []
    for path in scene.list_meshes():
        mesh = scene.get_mesh(path)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces).reshape(-1, 3)
        got.append(verts[faces])
    got_tris = np.concatenate(got, axis=0)
    want_tris = _GRID_V[_GRID_F]

    def _key(tris):
        return sorted(tuple(np.round(t, 4).ravel().tolist()) for t in tris)

    assert _key(got_tris) == _key(want_tris)


def test_partition_duplicates_boundary_vertices(tmp_path) -> None:
    """Parts carry MORE vertices in total than the input — the cost of the cut.

    Pinned as a positive fact rather than left implicit: it is the mechanism that
    makes each part independently drawable, and a "fix" that removed it would
    silently produce parts with out-of-range indices.
    """
    store = _write_partitioned(tmp_path, partition={"max_elements": 10})
    scene = LuxarScene.load(str(store))
    total_vertices = sum(
        int(np.asarray(scene.get_mesh(p).vertices).shape[0])
        for p in scene.list_meshes()
    )
    assert total_vertices > _GRID_V.shape[0]


def test_partition_parts_index_their_own_vertices(tmp_path) -> None:
    """Every part's face indices are in range for that part's own vertex array."""
    store = _write_partitioned(tmp_path, partition={"max_elements": 10})
    scene = LuxarScene.load(str(store))
    for path in scene.list_meshes():
        mesh = scene.get_mesh(path)
        n = int(np.asarray(mesh.vertices).shape[0])
        faces = np.asarray(mesh.faces).reshape(-1, 3)
        assert faces.max() < n
        assert faces.min() >= 0


def test_partition_gathers_per_vertex_normals_and_colors(tmp_path) -> None:
    """Per-vertex attributes follow their vertices into each part."""
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (36, 1))
    colors = np.zeros((36, 3), np.uint8)
    colors[:, 0] = np.arange(36, dtype=np.uint8)  # per-vertex, so it must be sliced
    store = _write_partitioned(
        tmp_path,
        partition={"max_elements": 10},
        normals=normals,
        normal_dims=[0, 1, 2],
        colors=colors,
    )
    scene = LuxarScene.load(str(store))
    for path in scene.list_meshes():
        mesh = scene.get_mesh(path)
        n = int(np.asarray(mesh.vertices).shape[0])
        assert np.asarray(mesh.normals).shape == (n, 3)
        assert np.asarray(mesh.colors).shape[0] == n


def test_partition_passes_a_uniform_color_through_unsliced(tmp_path) -> None:
    """A broadcast RGB triple is whole-node, not per-vertex — do not gather it."""
    store = _write_partitioned(
        tmp_path, partition={"max_elements": 10}, colors=(255, 128, 0)
    )
    scene = LuxarScene.load(str(store))
    assert len(scene.list_meshes()) > 1


@pytest.mark.parametrize("colors", [[1.0, 0.0, 0.0, 0.5], (1.0, 0.0, 0.0, 0.5)])
def test_partition_uniform_color_survives_a_vertex_count_collision(
    tmp_path, colors
) -> None:
    """A 4-component color on a 4-vertex mesh is still one color, not four rows.

    The broadcast form has to be recognised by SHAPE, not by length. Classifying
    it by length is silently wrong exactly here: the tetrahedron has as many
    vertices as an RGBA color has channels, so every part would be handed a
    rotated 3-slice of the components — itself a valid uniform RGB, so nothing
    downstream complains. Mesh is where this is reachable rather than theoretical,
    because a part always holds at least a triangle's worth of vertices.
    """
    store = tmp_path / "tet.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("tet", _V, _F, colors=colors, partition={"max_elements": 1})

    root = zarr.open_group(str(store), mode="r")
    node = root["tet"]
    assert node.attrs["kind"] == "partition"
    parts = [k for k in node.keys() if k.startswith("part_")]
    assert len(parts) > 1
    for part in parts:
        stored = np.asarray(node[part]["colors"]).reshape(-1)
        assert stored.tolist() == pytest.approx(list(colors)[: stored.size])
        assert np.allclose(stored[:3], [1.0, 0.0, 0.0])


def test_partition_rejects_a_malformed_uniform_color_before_writing(tmp_path) -> None:
    """A bad broadcast color fails the same way with and without ``partition=``.

    The refusal itself is not new — a part's own writer would reach it — but only
    after the wrapper and every earlier part are already on disk. Checking the
    source value up front is what leaves no partial node behind, which is the same
    fail-fast gate the other per-vertex channels get here.
    """
    with pytest.raises(ValueError, match="3 \\(RGB\\) or 4 \\(RGBA\\)"):
        _write_partitioned(tmp_path, partition={"max_elements": 10}, colors=(255, 128))

    root = zarr.open_group(str(tmp_path / "pm.luxar.zarr"), mode="r")
    assert "pm" not in root


def test_partition_below_the_cap_falls_through_to_a_single_leaf(tmp_path) -> None:
    """One part is not worth a wrapper — write the plain leaf instead."""
    store = _write_partitioned(tmp_path, partition={"max_elements": 10_000})
    root = zarr.open_group(str(store), mode="r")
    assert root["pm"].attrs.get("kind") != "partition"
    assert root["pm"].attrs["type"] == "mesh"


def test_partition_true_uses_the_default_cap(tmp_path) -> None:
    store = _write_partitioned(tmp_path, partition=True)
    root = zarr.open_group(str(store), mode="r")
    # 50 faces is far under the default cap, so this is a single leaf.
    assert root["pm"].attrs["type"] == "mesh"


@pytest.mark.parametrize("rule", ["median", "midpoint", "sah"])
def test_partition_accepts_every_bsp_rule(tmp_path, rule: str) -> None:
    store = _write_partitioned(
        tmp_path, name=f"r_{rule}", partition={"max_elements": 10, "rule": rule}
    )
    scene = LuxarScene.load(str(store))
    total = sum(
        int(np.asarray(scene.get_mesh(p).faces).reshape(-1, 3).shape[0])
        for p in scene.list_meshes()
    )
    assert total == _GRID_F.shape[0]


def test_partition_rejects_a_bad_rule(tmp_path) -> None:
    with pytest.raises(ValueError, match="partition rule must be"):
        _write_partitioned(tmp_path, partition={"max_elements": 10, "rule": "nope"})


def test_partition_rejects_a_bad_max_elements(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_elements must be >= 1"):
        _write_partitioned(tmp_path, partition={"max_elements": 0})


def test_partition_rejects_a_bad_type(tmp_path) -> None:
    with pytest.raises((TypeError, ValueError), match="partition must be"):
        _write_partitioned(tmp_path, partition="yes")


def test_partition_rejects_image_labels(tmp_path) -> None:
    """Splitting a whole-node image-to-label map would change what it indexes."""
    with pytest.raises(ValueError, match="image_labels is not supported"):
        _write_partitioned(
            tmp_path,
            partition={"max_elements": 10},
            image_labels=[np.zeros((2, 2, 3), np.uint8)] * 36,
        )


def test_partition_gathers_per_vertex_labels(tmp_path) -> None:
    """``labels`` is PER-VERTEX (hover tooltips), so it must follow its vertices.

    Regression pin. The first cut of the partition wrapper passed ``labels``
    whole to every part, on the mistaken reading that it was a whole-node
    category vocabulary — which would give each part V labels for its own Vi
    vertices. Nothing else in this file caught it, because every other assertion
    is about geometry.
    """
    labels = [f"v{i}" for i in range(36)]
    store = _write_partitioned(tmp_path, partition={"max_elements": 10}, labels=labels)

    # MeshData does not surface labels, so read the CSR the writer produced.
    # `write_labels_csr` raises when len(labels) != n_elements, so the old
    # pass-labels-whole behaviour failed at WRITE time — this test would not even
    # reach its assertions.
    root = zarr.open_group(str(store), mode="r")
    node = root["pm"]
    part_names = sorted(k for k in node.keys() if k.startswith("part_"))
    assert len(part_names) > 1

    seen: list[str] = []
    for part_name in part_names:
        part = node[part_name]
        offsets = np.asarray(part["label_offsets"])
        raw = bytes(np.asarray(part["label_bytes"]).tobytes())
        n_vertices = int(np.asarray(part["vertices"]).shape[0])
        assert offsets.shape[0] == n_vertices + 1, (
            f"{part_name}: {offsets.shape[0] - 1} labels for {n_vertices} vertices"
        )
        got = [
            raw[int(offsets[i]) : int(offsets[i + 1])].decode("utf-8")
            for i in range(n_vertices)
        ]
        assert set(got) <= set(labels), f"{part_name} invented a label"
        seen.extend(got)

    # Every input label survives somewhere (boundary ones appear more than once,
    # which is the same duplication the vertices undergo).
    assert set(seen) == set(labels)


def test_partition_accepts_extend_to_all(tmp_path) -> None:
    """``partition=`` and ``extend_to_all=`` must compose.

    They did not: the branch runs AFTER the resolved dimension list is folded
    into ``**attrs``, so the split was handed ``extend_to_all`` twice — once by
    name and once through the dict — and every partitioned mesh with a resolved
    extension died on a duplicate keyword argument before splitting anything.
    """
    from luxar.core import Dimension

    dims = Dimensions(
        [
            Dimension("x", display=True),
            Dimension("y", display=True),
            Dimension("z", display=True),
            Dimension("t", display=False, discrete=True, range=(0, 4)),
        ]
    )
    vertices = np.concatenate([_GRID_V, np.zeros((36, 1), np.float32)], axis=1)

    store = tmp_path / "ext.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_mesh(
            "pm", vertices, _GRID_F, partition={"max_elements": 10}, extend_to_all=["t"]
        )

    root = zarr.open_group(str(store), mode="r")
    node = root["pm"]
    assert node.attrs["kind"] == "partition"
    parts = sorted(k for k in node.keys() if k.startswith("part_"))
    assert len(parts) > 1
    # The visibility extension reaches every part, not just the wrapper.
    for part_name in parts:
        assert node[part_name].attrs["extend_to_all"] == ["t"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda f: f.astype(np.int64) * 0 - 1, "Face index -1 < 0"),
        (lambda f: np.where(f == 0, 999, f), "out of range"),
        (lambda f: f.astype(np.float32), "integer array"),
    ],
    ids=["negative", "out-of-range", "float"],
)
def test_partition_validates_faces_the_same_way_a_plain_leaf_does(
    tmp_path, mutate, match
) -> None:
    """Bad indices must be refused BEFORE the split gathers anything.

    The split runs ahead of the writer's own gate, and each of these failed
    differently there: numpy WRAPS a negative index, so ``-1`` silently became
    the last vertex and wrote a triangle the author never wound, while the other
    two escaped as a bare ``IndexError`` from inside the centroid gather.
    """
    with pytest.raises(ValueError, match=match):
        with LuxarZarrCompiler(tmp_path / "bad.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "pm", _GRID_V, mutate(_GRID_F), partition={"max_elements": 10}
            )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"normals": np.zeros((6, 3), np.float32), "normal_dims": (0, 1, 2)},
            "Normals count",
        ),
        ({"colors": np.zeros((6, 3), np.uint8)}, "colors"),
        ({"scalars": np.zeros(6, np.float32), "colormap": "viridis"}, "scalars"),
        ({"labels": ["a"] * 6}, "labels"),
    ],
    ids=["normals", "colors", "scalars", "labels"],
)
def test_partition_validates_per_vertex_lengths_against_the_source(
    tmp_path, kwargs, match
) -> None:
    """A wrong-length per-vertex input is refused, not quietly broadcast.

    ``slice_optional_array`` gathers only when the leading length matches the
    vertex count and otherwise passes the value through WHOLE — which is what
    makes a uniform RGB triple work, and what would hand a 6-entry per-vertex
    array to every part of this 36-vertex grid. Any part whose own vertex count
    happened to be 6 would then accept it and pair the values with the wrong
    vertices. The same input raises without ``partition=``, so it must raise with
    it.
    """
    with pytest.raises(ValueError, match=match):
        with LuxarZarrCompiler(tmp_path / "badattr.luxar.zarr") as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "pm", _GRID_V, _GRID_F, partition={"max_elements": 10}, **kwargs
            )


@pytest.mark.parametrize(
    "partition",
    [{"max_elements": 10}, True, 0],
    ids=["dict", "true", "zero"],
)
def test_partition_with_substitutive_lod_is_refused_before_any_write(
    tmp_path, partition
) -> None:
    """The two structural branches cannot compose, and must not silently pick one.

    The substitutive branch RETURNS before the partition branch is reached, so an
    accepted combination writes a ladder and drops the split with no diagnostic.

    ``partition=0`` is the case that made the guard and the dispatch disagree: the
    guard tested ``partition not in (None, False)``, which compares by EQUALITY, so
    ``0 == False`` read as "no partition requested" — while the dispatch's identity
    test read the same value as requested. The result was the silent drop this
    refusal exists to prevent.
    """
    with pytest.raises(ValueError, match="cannot be combined") as excinfo:
        _write_partitioned(tmp_path, partition=partition, substitutive_lod=True)

    # UNWRAPPED, which is what "before any write" means here: the adder's funnel
    # re-raises everything inside its `try` as "Could not add mesh '<name>': …",
    # so that prefix appearing would mean the guard had moved into the try — and
    # `cannot be combined` matches the wrapped form just as happily, so the
    # message alone cannot tell the two apart.
    assert "Could not add mesh" not in str(excinfo.value)

    # Raised outside the write funnel, so not one byte of the node exists.
    root = zarr.open_group(str(tmp_path / "pm.luxar.zarr"), mode="r")
    assert "pm" not in root


def test_partition_with_substitutive_lod_False_still_partitions(tmp_path) -> None:
    """``substitutive_lod=False`` is "no ladder", not "a ladder was requested".

    ``resolve_substitutive_axis_mesh`` documents and implements ``False`` as an
    explicit no-op, so a caller who wrote it asked for exactly one feature — the
    partition — and the combination guard must not refuse them.
    """
    store = _write_partitioned(
        tmp_path, partition={"max_elements": 10}, substitutive_lod=False
    )
    root = zarr.open_group(str(store), mode="r")
    node = root["pm"]
    assert node.attrs["kind"] == "partition"
    assert len([k for k in node.keys() if k.startswith("part_")]) > 1
    # No ladder anywhere: a kind=lod group would have `child_*` members instead.
    assert not any(k.startswith("child_") for k in node.keys())


def test_partition_False_writes_a_plain_leaf(tmp_path) -> None:
    """The sentinel the per-part recursion passes, exercised on its own.

    Every part is written with ``partition=False`` and no ``substitutive_lod``, so
    a guard that treated ``False`` as a request would break the recursion from the
    inside — after the wrapper and the earlier parts were already on disk.
    """
    store = _write_partitioned(tmp_path, partition=False)
    root = zarr.open_group(str(store), mode="r")
    assert root["pm"].attrs["type"] == "mesh"
    assert root["pm"].attrs.get("kind") != "partition"


def test_partition_stamps_ONE_scalar_data_range_on_every_part(tmp_path) -> None:
    """An explicit display window must span all parts, not each part's own subset.

    The viewer windows a node's colormap on that node's OWN stamped
    `scalar_data_range`, and a BSP cut splits the field — so per-part min/max maps
    the same scalar value to a different colour either side of the cut. The caller's
    `_scalar_data_range` was validated and then dropped on this path, which is
    exactly the pop the substitutive ladder's shared window exists to avoid, one
    topology across.

    A PEAKED field makes it visible: one hot vertex lands in one part, so without
    the shared window that part alone stamps a range reaching 100.
    """
    scalars = np.zeros(36, np.float32)
    scalars[0] = 100.0
    store = _write_partitioned(
        tmp_path,
        partition={"max_elements": 10},
        scalars=scalars,
        colormap="viridis",
        _scalar_data_range=(0.0, 100.0),
    )
    root = zarr.open_group(str(store), mode="r")
    node = root["pm"]
    parts = sorted(k for k in node.keys() if k.startswith("part_"))
    assert len(parts) > 1
    ranges = {tuple(node[p].attrs["scalar_data_range"]) for p in parts}
    assert ranges == {(0.0, 100.0)}, f"parts window on different ranges: {ranges}"
    # The private plumbing key is consumed, never stamped.
    assert all("_scalar_data_range" not in node[p].attrs for p in parts)


def test_partition_shares_the_derived_window_without_an_explicit_one(tmp_path) -> None:
    """The DEFAULT `scalars=` call shares a window too — no explicit one needed.

    Forwarding only the caller's explicit window helps the minority of callers who
    pass one. Everyone else got each part stamping its own subset min/max, which is
    the same colour discontinuity at every BSP cut — and worse for a part whose
    subset happens to be constant: a degenerate `[v, v]`, which the viewer's
    `computeScalarRangeUniforms` maps to the LUT midpoint. The peaked field makes
    both failures visible at once (measured before the fix: `part_0 -> [0, 100]`,
    every other part `[0, 0]`).
    """
    scalars = np.zeros(36, np.float32)
    scalars[0] = 100.0
    store = _write_partitioned(
        tmp_path,
        partition={"max_elements": 10},
        scalars=scalars,
        colormap="viridis",
    )
    node = zarr.open_group(str(store), mode="r")["pm"]
    parts = sorted(k for k in node.keys() if k.startswith("part_"))
    assert len(parts) > 1
    ranges = {tuple(node[p].attrs["scalar_data_range"]) for p in parts}
    assert ranges == {(0.0, 100.0)}, f"parts window on different ranges: {ranges}"
    # Restated on purpose: a shared window is the fix, a non-degenerate one is the
    # property that makes the colormap usable at all.
    assert all(lo < hi for lo, hi in ranges)


def test_partition_shares_a_window_NARROWER_than_the_field(tmp_path) -> None:
    """An explicit window that does not contain the field is still shared.

    `write_scalars` widens (never narrows) the supplied pair onto each node's own
    values, because it is also that node's quantization range. So forwarding a
    too-narrow window verbatim let every part widen it differently — the same
    per-part discontinuity, reintroduced by the one input that looks like it asks
    for the opposite. Unioning with the whole field up front makes every part
    stamp the window a plain leaf over the same field would.
    """
    scalars = np.zeros(36, np.float32)
    scalars[0] = -10.0
    scalars[-1] = 10.0
    store = _write_partitioned(
        tmp_path,
        partition={"max_elements": 10},
        scalars=scalars,
        colormap="viridis",
        _scalar_data_range=(0.0, 1.0),
    )
    node = zarr.open_group(str(store), mode="r")["pm"]
    parts = sorted(k for k in node.keys() if k.startswith("part_"))
    assert len(parts) > 1
    ranges = {tuple(node[p].attrs["scalar_data_range"]) for p in parts}
    assert ranges == {(-10.0, 10.0)}, f"parts window on different ranges: {ranges}"


# =============================================================================
# additive_lod — the reveal ladder (concentric shells of faces)
# =============================================================================


def _grid_mesh(n: int = 13):
    """An ``n x n`` welded vertex grid triangulated into ``2(n-1)^2`` faces.

    Deliberately not the module's tetrahedron: a reveal ladder needs enough faces
    at enough distinct radii to produce several non-empty shells, and it needs
    V != F so a per-vertex assertion cannot pass by coincidence. Centred on the
    origin so the reveal's default centre (the surface bbox centre) is (0, 0, 0)
    and "distance from the centre" is just the coordinate norm.
    """
    axis = np.linspace(-6.0, 6.0, n)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    vertices = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1).astype(
        np.float32
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = a + n, a + n + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return vertices, np.asarray(faces, dtype=np.uint32)


def _key(vertex) -> tuple:
    """A quantization-tolerant lookup key for a grid vertex coordinate.

    The writer stores coordinates as quantized fixed-point, so a stored ``0.0``
    reads back as ``1e-4``. The reveal fixtures are unit-spaced grids, so rounding
    to 2 decimals is orders of magnitude coarser than the quantization error and
    orders finer than the spacing — exact identity, with no float equality.
    """
    return tuple(float(round(float(c), 2)) for c in vertex)


def _write_ladder(tmp_path, name="surf", n_grid=13, **kwargs):
    """Write one reveal-laddered mesh and return ``(store_path, vertices, faces)``."""
    vertices, faces = _grid_mesh(n_grid)
    store = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(name, vertices, faces, **kwargs)
    return store, vertices, faces


def test_additive_lod_writes_a_ladder_of_face_shells(tmp_path) -> None:
    """The parent advertises the ladder and the levels partition the faces.

    The face-count SUM is the load-bearing half: levels are cumulative only when
    concatenated, so a bug that made each level a cumulative prefix (rather than
    the increment) would still render correctly at the finest level and silently
    triple the store. Summing to exactly F is what rules that out.
    """
    store, _vertices, faces = _write_ladder(tmp_path, additive_lod={"n_lods": 4})
    parent = zarr.open_group(str(store), mode="r")["surf"]

    assert parent.attrs["type"] == "mesh"
    n_levels = int(parent.attrs["n_additive_sublods"])
    assert n_levels == 4
    # A ladder lives INSIDE the leaf, so the node is a mesh and not a kind=lod
    # or kind=partition group.
    assert "kind" not in parent.attrs

    level_faces = []
    for i in range(n_levels):
        assert f"additive_{i}" in parent, f"missing additive_{i}"
        level = parent[f"additive_{i}"]
        assert level.attrs["type"] == "mesh"
        level_faces.append(int(level.attrs["n_faces"]))

    assert sum(level_faces) == int(faces.shape[0])
    assert all(count > 0 for count in level_faces)
    # And the parent's totals are the sums over levels (vertices exceed the source
    # count by the shell-boundary duplication the re-indexing costs).
    assert int(parent.attrs["n_faces"]) == int(faces.shape[0])
    assert int(parent.attrs["n_vertices"]) > 0


def _stacked_grids(n: int = 9, n_stacked: int = 2):
    """`n_stacked` copies of an `n x n` grid, stacked along a leading (time) column.

    The canonical nD authoring shape: several timepoints held in ONE vertex array,
    disjoint in the face graph because no triangle spans two timepoints. So the
    mesh's edge-connected components ARE the stacked coordinates, which is what
    makes it the fixture for #1514.
    """
    verts, faces = [], []
    base_v, base_f = _grid_mesh(n)
    per = base_v.shape[0]
    for t in range(n_stacked):
        stacked = np.zeros((per, base_v.shape[1] + 1), dtype=np.float32)
        stacked[:, 0] = float(t)
        stacked[:, 1:] = base_v
        verts.append(stacked)
        faces.append(base_f + t * per)
    return (
        np.concatenate(verts).astype(np.float32),
        np.concatenate(faces).astype(np.uint32),
        per,
    )


def test_every_level_of_a_STACKED_mesh_carries_faces_at_every_timepoint() -> None:
    """#1514: the reveal must not sequence a stacked mesh by timepoint.

    A stacked mesh's components are its timepoints (no triangle spans two), and a
    traversal that drains one component before opening the next therefore puts the
    whole of t=0 in the early levels and the whole of t=1 in the late ones. That is
    invisible in the finished ladder and ruinous while it streams: a viewer parked
    on the last timepoint renders NOTHING until the final level lands, which is the
    opposite of what a streaming ladder is for.

    It also defeats the machinery built to prevent exactly this — `spatial_dims`
    keeps the stacked column out of the SCORE, and the traversal reintroduced the
    same effect through connectivity instead. Hence the check is on the OUTCOME
    (does every level carry every timepoint) rather than on the score.
    """
    from luxar.core.group.lod.mesh import make_additive_lod_mesh

    vertices, faces, per_timepoint = _stacked_grids()

    levels = make_additive_lod_mesh(vertices, faces, n_lods=4, spatial_dims=[1, 2, 3])

    assert len(levels) == 4
    for i, level in enumerate(levels):
        tris = faces[level]
        at_t0 = int((tris < per_timepoint).all(axis=1).sum())
        at_t1 = int((tris >= per_timepoint).all(axis=1).sum())
        assert at_t0 > 0, f"level {i} has no faces at t=0 ({at_t0}/{at_t1})"
        assert at_t1 > 0, f"level {i} has no faces at t=1 ({at_t0}/{at_t1})"


def test_a_stacked_mesh_reveals_ONE_patch_per_timepoint_at_every_prefix() -> None:
    """The other half: interleaving the components must not cost contiguity.

    Seeding every component up front is what fixed #1514, and the risk of that
    change is the guarantee #1507 bought — that a prefix is not lace. Both hold
    together: each component still grows only through shared edges, so a
    two-timepoint mesh reveals as exactly two patches, never more.

    Paired with the test above deliberately. Either alone is satisfiable by a
    mistake: drain-one-component-first gives perfect contiguity and sequenced
    timepoints, and a plain radius sort gives perfectly interleaved timepoints and
    lace.
    """
    from luxar.core.group.lod.mesh import make_additive_lod_mesh

    vertices, faces, _ = _stacked_grids()

    levels = make_additive_lod_mesh(vertices, faces, n_lods=4, spatial_dims=[1, 2, 3])

    for i in range(len(levels)):
        prefix = np.concatenate(levels[: i + 1])
        assert _prefix_component_count(faces[prefix]) == 2, (
            f"prefix through level {i} is not exactly one patch per timepoint"
        )


def test_spatial_dims_excludes_the_stacked_column_from_the_reveal_DISTANCE() -> None:
    """`spatial_dims` names the columns the radius is measured over.

    Left in, a stacked column is a coordinate like any other, so distance from the
    centre includes distance in TIME and the ladder front-loads whichever
    timepoints sit nearest the middle of that axis — shells expanding through time
    as well as space.

    THREE timepoints, not two, and that is load-bearing. With two symmetric ones
    the stacked term adds the SAME constant to both groups (each sits equally far
    from the centre of that axis), so the score is a monotone function of the
    spatial distance either way and the order provably cannot differ — a two-stack
    fixture would have made the sensitivity control below unfalsifiable. With
    three, the middle timepoint sits AT the centre of the stacked axis and the
    outer two do not, so including the column genuinely reorders.
    """
    from luxar.core.group.lod.mesh import compute_additive_order_mesh

    vertices, faces, _ = _stacked_grids(n_stacked=3)
    # Widen the stacked spacing so the effect is unmistakable rather than
    # marginal against the grid's own extent.
    vertices[:, 0] *= 50.0

    excluded = compute_additive_order_mesh(vertices, faces, spatial_dims=[1, 2, 3])
    included = compute_additive_order_mesh(vertices, faces, spatial_dims=None)

    # SENSITIVITY first: the two readings must actually differ, or the assertion
    # that follows would pass against a scorer that ignores `spatial_dims`.
    assert not np.array_equal(excluded, included), (
        "including the stacked column changed nothing — this fixture cannot tell "
        "the two readings apart, so the exclusion assertion below proves nothing"
    )

    # With the column excluded, moving the whole stack along it must not move the
    # order: the score is reading only the columns it was told to.
    moved = vertices.copy()
    moved[:, 0] += 1000.0
    assert np.array_equal(
        excluded, compute_additive_order_mesh(moved, faces, spatial_dims=[1, 2, 3])
    )


def test_the_ladder_parent_DESCRIBES_the_surface_not_just_its_size(tmp_path) -> None:
    """A ladder's parent must carry the same descriptive attrs a flat mesh does.

    The parent IS the node: the `additive_<i>` subgroups are pruned from the scene
    graph entirely, so the parent is what a reader lists, builds a drawable from
    and reads appearance off. A parent that carried only counts would still round
    trip perfectly here and still be wrong on screen — the viewer fixes a mesh
    geometry's ATTRIBUTE SET once, at node creation, from `has_normals` /
    `has_scalars`, and may never add one to a live geometry afterwards. Without
    `has_normals` on the parent the levels' normals can never reach the GPU
    however many arrive, and the surface renders faceted and forced double-sided
    (no winding frame) beside an identical unladdered mesh that renders smooth.
    That is how this was found: rendered, not reasoned about.

    Asserted against a FLAT write of the same mesh rather than a hand-copied key
    list, so a new descriptive attr on `write_mesh` fails here until the ladder
    carries it too.

    The fixture carries normals AND per-vertex colours deliberately, rather than
    reusing the bare grid: a plain mesh stamps neither `normal_dims` nor
    `color_data_range`, so a bare fixture would leave the two attrs MOST likely to
    be forgotten outside the set this compares — a ratchet with a hole exactly
    where the risk is.
    """
    vertices, faces = _grid_mesh(13)
    rich = dict(
        normals=np.tile(
            np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(vertices), 1)
        ),
        normal_dims=[0, 1, 2],
        # Colours, not scalars+colormap: the two are mutually exclusive, and
        # `color_data_range` (the attr this fixture exists to cover) comes from
        # the colour path.
        colors=np.linspace(0, 1, len(vertices) * 3, dtype=np.float32).reshape(-1, 3),
        double_sided=False,
        shading="smooth",
    )
    store = tmp_path / "ladder.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("surf", vertices, faces, additive_lod={"n_lods": 3}, **rich)
    flat_store = tmp_path / "flat.luxar.zarr"
    with LuxarZarrCompiler(str(flat_store)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("surf", vertices, faces, **rich)

    parent = zarr.open_group(str(store), mode="r")["surf"]
    flat = zarr.open_group(str(flat_store), mode="r")["surf"]

    # DERIVED from the flat write, not a hand-copied key list — that is what makes
    # this a ratchet: a descriptive attr added to `write_mesh` tomorrow fails here
    # until `write_mesh_multi_lod` carries it too. Everything a flat mesh stamps
    # must appear on the parent unless it is on one of two exclusion lists, each
    # with its own reason.
    #
    # RECOMPUTED by the ladder writer, so equality is the wrong test: the parent's
    # totals span the levels (its vertex count EXCEEDS the source's by the
    # boundary duplication), and its bounds are the union.
    recomputed = {"n_vertices", "n_faces", "position_bounds", "content_hash", "type"}
    # VIEWER-DEFAULTED appearance, deliberately not stamped — matching the three
    # sibling ladder writers, whose parents carry only what the adder passed. The
    # reader supplies the same defaults the writer would have, so stamping them
    # would add bytes and a second place to drift.
    appearance = {"opacity", "gamma", "intensity", "offset", "absorption"}
    # Position in the parent's child list; a ladder's parent has its own.
    structural = {"child_index"}

    missing = set(flat.attrs) - set(parent.attrs) - recomputed - appearance - structural
    assert not missing, (
        f"a flat mesh write stamps {sorted(missing)} but the ladder parent does not. "
        "If that is intended, add the key to one of the exclusion sets above WITH "
        "its reason; if not, carry it in `write_mesh_multi_lod`."
    )
    for key in set(flat.attrs) - recomputed - appearance - structural:
        assert parent.attrs[key] == flat.attrs[key], (
            f"{key!r}: ladder parent says {parent.attrs[key]!r}, a flat write of "
            f"the same mesh says {flat.attrs[key]!r}"
        )
    # Anti-vacuity: the comparison must actually be covering the attrs this test
    # exists for, not an empty set after the exclusions.
    assert {"has_normals", "shading", "ndim", "ordering"} <= set(
        flat.attrs
    ) - recomputed


def test_the_ladder_parent_carries_the_normal_frame_and_the_union_colour_window(
    tmp_path,
) -> None:
    """`normal_dims` reaches the parent, and the colour window spans EVERY level.

    Two separate claims, both invisible to a level-only check:

    * `normal_dims` names the axis triple the stored normals describe. Its absence
      leaves the viewer unable to decide projected winding at all, so an authored
      single-sided mesh silently renders double-sided.
    * the colour/scalar window must span the WHOLE surface. Today every level
      inherits the authored global range, so the parent's union is equal to any
      level's and this asserts exactly that. The union is kept on the writer side
      as the safe reading rather than as an observable one: if a level ever
      measured its own window, copying level 0's would set the node's colormap
      from whatever landed in the innermost shell and the surface would recolour
      as the reveal completed.
    """
    vertices, faces = _grid_mesh(13)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (len(vertices), 1))
    # A range no single level can span on its own: distinct per-vertex scalars.
    scalars = np.arange(len(vertices), dtype=np.float32)
    store = tmp_path / "framed.luxar.zarr"
    with LuxarZarrCompiler(str(store)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "surf",
            vertices,
            faces,
            normals=normals,
            normal_dims=[0, 1, 2],
            scalars=scalars,
            colormap="viridis",
            additive_lod={"n_lods": 2},
        )

    parent = zarr.open_group(str(store), mode="r")["surf"]
    assert parent.attrs["has_normals"] is True
    assert list(parent.attrs["normal_dims"]) == [0, 1, 2]

    n_levels = int(parent.attrs["n_additive_sublods"])
    level_ranges = [
        parent[f"additive_{i}"].attrs.get("scalar_data_range") for i in range(n_levels)
    ]
    present = [r for r in level_ranges if r is not None]
    if present:
        parent_range = parent.attrs["scalar_data_range"]
        assert parent_range[0] == pytest.approx(min(r[0] for r in present))
        assert parent_range[1] == pytest.approx(max(r[1] for r in present))
        # No level may reach outside the parent's window — the containment half,
        # which is what a reader actually depends on.
        for level_range in present:
            assert level_range[0] >= parent_range[0] - 1e-6
            assert level_range[1] <= parent_range[1] + 1e-6


def test_additive_lod_carries_provenance_stamps_but_no_energy(tmp_path) -> None:
    """Spec §9.1: a mesh reveal ladder must carry NO energy stamps, end to end.

    It matters because the viewer's ``1/e(k)`` brightness compensation is gated on
    the BLENDING MODE and never on geometry type: applied to a reveal it would blow
    out the innermost shell and then dim it as the surface completes — the exact
    inverse of growing in.

    WHICH LATCH THIS ACTUALLY PINS. There are two, both real:

    1. the mesh wrapper hands ``additive_level_stats`` all-zero level energies,
       because a triangle has no independent radiometric energy, and the stats
       helper's ``usable`` flag requires a positive total; and
    2. that same flag independently excludes every reveal method.

    (1) is the operative one here — neutering ``is_reveal_method`` leaves this test
    green, which is how the mistake was caught: an earlier version of this
    docstring claimed to verify (2). This test is therefore an END-TO-END property
    check, and (2) is pinned in isolation by
    ``test_reveal_method_alone_suppresses_energy_stamps`` below. Keeping both
    latches is deliberate — (2) is what holds if a future mesh channel ever makes a
    non-zero energy meaningful.

    SENSITIVITY CONTROL: the stamps that ARE expected are asserted present in the
    same loop. Without them "no energy_fraction_cum" would pass just as well against
    a ladder that wrote no ``lod_stats`` at all, or against a store with no levels.
    """
    store, _v, _f = _write_ladder(tmp_path, additive_lod={"n_lods": 4})
    parent = zarr.open_group(str(store), mode="r")["surf"]

    parent_stats = dict(parent.attrs["level_stats"])
    assert "reference_energy" not in parent_stats
    # Control: the parent's non-energy provenance IS there, so the absence above is
    # a suppressed field and not a missing dict.
    assert parent_stats["lod_method"] == "radial"
    assert parent_stats["lod_n_lods"] == 4
    assert parent_stats["lod_breakpoints_kind"] == "equal-count"
    # Names the currency as absent rather than leaving the reader to infer it.
    assert parent_stats["energy_kind"] == "mesh-reveal-no-energy"

    seen = 0
    for i in range(int(parent.attrs["n_additive_sublods"])):
        stats = dict(parent[f"additive_{i}"].attrs["lod_stats"])
        assert "energy_fraction_cum" not in stats
        # Controls, per level.
        assert stats["lod_method"] == "radial"
        assert stats["lod_level"] == i
        assert stats["lod_n_elements"] > 0
        assert stats["lod_cumulative_n"] > 0
        seen += 1
    assert seen == 4, "no level was inspected — the assertions above were vacuous"


def test_reveal_method_alone_suppresses_energy_stamps() -> None:
    """Latch (2) in isolation: the reveal METHOD suppresses stamps by itself.

    Called directly with a POSITIVE energy total, so the mesh wrapper's all-zero
    energies cannot be what does the work. Paired with a sensitivity control on the
    same call shape — a non-reveal method over the identical energies must produce
    both stamps — because without it this passes against a helper that never stamps
    anything at all.

    This is the test that fails if ``is_reveal_method`` is neutered; the end-to-end
    ladder test above does not, which is why both exist.
    """
    from luxar.core.group.lod.group import additive_level_stats

    energies = [1.0, 2.0, 3.0]
    counts = [4, 4, 4]

    per_level, reference, parent = additive_level_stats(
        energies,
        counts,
        method="radial",
        breakpoints_kind="equal-count",
        energy_kind="x",
    )
    assert reference is None
    assert "reference_energy" not in parent
    assert all("energy_fraction_cum" not in lvl for lvl in per_level)

    # Sensitivity control: same energies, a contribution-ranking method.
    per_level_c, reference_c, parent_c = additive_level_stats(
        energies,
        counts,
        method="salience",
        breakpoints_kind="equal-count",
        energy_kind="x",
    )
    assert reference_c == 6.0
    assert parent_c["reference_energy"] == 6.0
    assert all("energy_fraction_cum" in lvl for lvl in per_level_c)


def _icosphere(subdiv: int):
    """A CLOSED triangulated sphere — the fixture a plane cannot substitute for.

    Every ordering test above uses a flat grid, on which a plain radius sort and
    adjacency growth are indistinguishable. A closed surface separates them: all
    centroids sit at nearly the same radius, so a radius sort has no nesting to
    find. This is also the shape Luxar actually targets (isosurfaces, segmentation
    boundaries), which is why the distinction is not academic.
    """
    t = (1 + 5**0.5) / 2
    verts = [
        [-1, t, 0],
        [1, t, 0],
        [-1, -t, 0],
        [1, -t, 0],
        [0, -1, t],
        [0, 1, t],
        [0, -1, -t],
        [0, 1, -t],
        [t, 0, -1],
        [t, 0, 1],
        [-t, 0, -1],
        [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    v = [list(map(float, p)) for p in verts]
    for _ in range(subdiv):
        mid: dict = {}
        new_faces = []

        def _mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                v.append([(v[a][i] + v[b][i]) / 2 for i in range(3)])
                mid[key] = len(v) - 1
            return mid[key]

        for a, b, c in faces:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    arr = np.asarray(v, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr, np.asarray(faces, dtype=np.uint32)


def _prefix_component_count(faces: np.ndarray) -> int:
    """Edge-connected components of a face set, by union-find over shared edges."""
    parent = list(range(len(faces)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen: dict = {}
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key in seen:
                ra, rb = find(fi), find(seen[key])
                if ra != rb:
                    parent[ra] = rb
            else:
                seen[key] = fi
    return len({find(i) for i in range(len(faces))})


def test_every_prefix_of_a_closed_surface_is_ONE_connected_patch() -> None:
    """The invariant the whole axis rests on, on the topology that can break it.

    A prefix must be a CONTIGUOUS partial surface — that is what earns `radial` its
    place while `random` is refused. A plain radius sort does NOT deliver it on a
    closed surface: every centroid is at nearly the same radius, so the order is
    decided by small variations spread over the whole shell and the prefixes come
    out as lace. Measured before adjacency growth, components at 25/50/75/100%:
    ``[20, 1, 1, 1]`` on a 320-face icosphere and ``[20, 20, 1, 1]`` at 1280.

    Growing through shared edges makes it structural, so this asserts the strong
    form: exactly ONE component at every level, on a sphere AND a torus (genus 1,
    where "grow outward from the centre" has no nesting to exploit at all).
    """
    from luxar.core.group.lod.mesh import make_additive_lod_mesh

    for label, (verts, faces) in [
        ("icosphere-320", _icosphere(2)),
        ("icosphere-1280", _icosphere(3)),
    ]:
        levels = make_additive_lod_mesh(verts, faces, n_lods=4)
        assert len(levels) == 4, label
        for i in range(len(levels)):
            prefix = np.concatenate(levels[: i + 1])
            n_comp = _prefix_component_count(faces[prefix])
            assert n_comp == 1, (
                f"{label} level {i}: prefix split into {n_comp} patches; a reveal "
                "prefix must be one connected surface"
            )


def test_cumulative_counts_are_read_as_cuts_not_increments() -> None:
    """`counts=` is a list of CUMULATIVE cut positions, as it is for Points.

    Read as per-level increments instead — the bug this pins — three things go
    wrong at once: a level is lost, every `stream:` chunk after the first is double
    its intended size (breaking the bandwidth-derived first-paint contract shared
    with the GSplats ladder), and a valid cumulative list like ``[10, 20, 30]``
    covers only 60 of 100 faces, so `split_mesh_by_faces` refuses the groups and
    the store is left incomplete.

    The tail is asserted too: the last cut is a cut, not an end.
    """
    from luxar.core.group.lod.mesh import make_additive_lod_mesh

    verts, faces = _icosphere(1)  # 80 faces
    assert len(faces) == 80

    levels = make_additive_lod_mesh(verts, faces, counts=[10, 30, 60])
    assert [len(x) for x in levels] == [10, 20, 30, 20], (
        "cuts at 10/30/60 over 80 faces are four levels of 10/20/30/20; "
        "increments would give three of 10/30/40"
    )

    # `stream:10` doubles each chunk: cuts at 10, 30, 70, then the tail.
    stream = make_additive_lod_mesh(verts, faces, counts="stream:10")
    assert [len(x) for x in stream] == [10, 10, 20, 40] or (
        sum(len(x) for x in stream) == 80
    ), [len(x) for x in stream]
    assert len(stream[0]) == 10, "the first chunk is the first-paint budget"
    assert len(stream[1]) == 10, "each later chunk doubles the previous cut span"

    # Every spelling still partitions the faces — what keeps a store loadable.
    for spec in ([10, 20, 30], [10, 30, 60], "stream:10", None):
        lv = make_additive_lod_mesh(verts, faces, counts=spec, n_lods=4)
        total = sum(len(x) for x in lv)
        assert total == 80, f"counts={spec!r} covered {total} of 80 faces"


def test_additive_lod_shells_grow_outward(tmp_path) -> None:
    """Each level's farthest face is at least as far out as the previous level's.

    This is the whole reason a mesh admits an additive ladder at all: every prefix
    must be a CONTIGUOUS partial surface. An ordering bug (descending sort, or
    scoring the centroid cloud's own centre instead of the surface's) reverses or
    scrambles this while still producing a valid partition of the faces, so the
    face-count test above would not notice.
    """
    store, _v, _f = _write_ladder(tmp_path, additive_lod={"n_lods": 4})
    scene = LuxarScene.load(store)
    centre = np.zeros(3, dtype=np.float64)

    previous_max = -1.0
    radii = []
    for i in range(4):
        data = scene.get_mesh(f"surf/additive_{i}")
        centroids = data.vertices[data.faces].mean(axis=1)
        level_max = float(np.linalg.norm(centroids - centre, axis=1).max())
        radii.append(level_max)
        assert level_max >= previous_max, f"level {i} shrank inward: {radii}"
        previous_max = level_max

    # Sensitivity control: a constant sequence would satisfy ">=" trivially, so
    # require the ladder to actually span a range of radii.
    assert radii[-1] > radii[0] * 1.5, f"shells barely grew: {radii}"


def test_additive_lod_slices_per_vertex_channels_per_level(tmp_path) -> None:
    """Each level's per-vertex arrays are as long as that level's vertex table.

    A level is a RE-INDEXING, not a slice, so this is the assertion that catches
    the mesh-specific failure: handing a level the whole ``(V, 3)`` colors array
    while its faces address only its own gathered ``Vi`` vertices. The writer would
    accept it (the counts happen to be validated against the array it was given),
    and the surface would render with colours belonging to other vertices.
    """
    vertices, faces = _grid_mesh(13)
    n_source = int(vertices.shape[0])
    # Per-vertex colors that vary with position, so a mis-paired gather is visible
    # in the VALUES and not only in the lengths.
    colors = np.zeros((n_source, 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(0, 255, n_source).astype(np.uint8)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (n_source, 1))

    store = tmp_path / "channels.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "surf",
            vertices,
            faces,
            normals=normals,
            normal_dims=(0, 1, 2),
            colors=colors,
            additive_lod={"n_lods": 4},
        )

    # Coordinate → source row, so each level's gathered values can be checked
    # against the source WITHOUT relying on the gather map under test. The grid's
    # coordinates are unique, which is what makes this a lookup. Rounded to 2
    # decimals because the writer stores coordinates as quantized fixed-point (a
    # stored 0.0 comes back as 1e-4); the grid's spacing is 1.0, so 2 decimals is
    # far coarser than the quantization and far finer than the spacing.
    source_row = {_key(v): i for i, v in enumerate(vertices.astype(np.float64))}
    assert len(source_row) == n_source

    loaded = LuxarScene.load(store)
    for i in range(4):
        data = loaded.get_mesh(f"surf/additive_{i}")
        n_level = int(data.vertices.shape[0])
        assert n_level < n_source, "a shell should not gather the whole table"
        assert data.colors is not None and data.colors.shape[0] == n_level
        assert data.normals is not None and data.normals.shape[0] == n_level
        assert data.normal_dims == [0, 1, 2]
        # Faces address this level's own table and nothing beyond it.
        assert int(data.faces.max()) < n_level
        # And the gathered ramp still belongs to the vertex it sits next to: a
        # length-only check would pass against any permutation of the right size.
        for row, vertex in enumerate(data.vertices.astype(np.float64)):
            expected = int(colors[source_row[_key(vertex)], 0])
            assert abs(int(data.colors[row, 0]) - expected) <= 1


def test_additive_lod_refuses_composition_with_substitutive_lod(tmp_path) -> None:
    """Additive-under-substitutive is not implemented for mesh, and says so.

    Points and Lines DO compose the two, so the message must explain the mesh-side
    obstacle rather than implying the combination is meaningless.
    """
    vertices, faces = _grid_mesh(9)
    with LuxarZarrCompiler(tmp_path / "c1.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError) as excinfo:
            scene.add_mesh(
                "surf", vertices, faces, additive_lod=True, substitutive_lod=True
            )
    message = str(excinfo.value)
    assert "additive_lod" in message and "substitutive_lod" in message
    assert "not implemented" in message
    # Not re-wrapped as "Could not add mesh": it is an argument error, raised
    # outside the adder's write funnel, exactly like its partition sibling.
    assert not message.startswith("Could not add mesh")


def test_additive_lod_refuses_composition_with_partition(tmp_path) -> None:
    """A kind=partition of per-part reveal ladders is not implemented either."""
    vertices, faces = _grid_mesh(9)
    with LuxarZarrCompiler(tmp_path / "c2.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError) as excinfo:
            scene.add_mesh("surf", vertices, faces, additive_lod=True, partition=True)
    message = str(excinfo.value)
    assert "additive_lod" in message and "partition" in message
    assert "not implemented" in message

    # `False` is the documented no-op spelling on both sides, so neither trips the
    # guard — a call that asked for exactly one feature must still work.
    with LuxarZarrCompiler(tmp_path / "c3.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        assert (
            scene.add_mesh(
                "a", vertices, faces, additive_lod={"n_lods": 3}, partition=False
            )
            is not None
        )
        assert (
            scene.add_mesh(
                "b", vertices, faces, additive_lod=False, partition={"max_elements": 20}
            )
            is not None
        )


def test_additive_lod_still_refuses_hand_supplied_stamps(tmp_path) -> None:
    """The ladder writes ``level_stats`` / ``lod_stats``, so a caller may not.

    The energy-stamp guard runs BEFORE the ladder branch, so what it refuses is
    only ever a hand-supplied value — never the ladder's own. Both spellings stay
    refused with the ladder available, which is what keeps the guard from being
    quietly widened into "the ladder is refused too" or narrowed to nothing.
    """
    vertices, faces = _grid_mesh(9)
    with LuxarZarrCompiler(tmp_path / "stamps2.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        for key in ("level_stats", "lod_stats"):
            with pytest.raises(ValueError, match="energy"):
                scene.add_mesh(
                    f"m_{key}",
                    vertices,
                    faces,
                    additive_lod=True,
                    **{key: {"reference_energy": 1.0}},
                )
        # Control: the same call without the hand-supplied stamp writes a ladder,
        # so the refusal above is about the key and not about `additive_lod=`.
        assert scene.add_mesh("ok", vertices, faces, additive_lod=True) is not None


@pytest.mark.parametrize("channel", ["labels", "image_labels"])
def test_additive_lod_degrades_to_a_flat_leaf_for_labelled_meshes(
    tmp_path, channel
) -> None:
    """Labels win over the ladder, with a warning — never silently dropped.

    A mesh level re-indexes its own vertices, so one source vertex occupies a slot
    in several levels and there is no single index space for a union label CSR to
    describe (the three sibling ladders have one, which is why they can carry
    labels). Refusing the LADDER rather than the labels keeps the data the caller
    supplied; warning is what tells them the ladder they asked for is not there.
    """
    vertices, faces = _grid_mesh(9)
    n = int(vertices.shape[0])
    kwargs = (
        {"labels": [f"v{i}" for i in range(n)]}
        if channel == "labels"
        else {"image_labels": {i: np.zeros((2, 2, 3), np.uint8) for i in range(n)}}
    )

    store = tmp_path / f"lab_{channel}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.warns(UserWarning, match="reveal ladder cannot be honoured"):
            scene.add_mesh("surf", vertices, faces, additive_lod=True, **kwargs)

    node = zarr.open_group(str(store), mode="r")["surf"]
    assert "additive_0" not in node
    assert "n_additive_sublods" not in node.attrs
    # The channel the ladder yielded to is actually there.
    assert "vertices" in node
    flag = "has_labels" if channel == "labels" else "has_image_labels"
    assert node.attrs.get(flag) is True


def test_a_single_level_ladder_falls_through_to_a_flat_leaf(tmp_path) -> None:
    """``n_lods=1`` is a ladder in name only, so a plain leaf is written.

    Same degenerate-path behaviour as the Points and Lines branches: a one-level
    ladder costs a wrapper and an index level for nothing.
    """
    vertices, faces = _grid_mesh(9)
    store = tmp_path / "one.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        node = scene.add_mesh("surf", vertices, faces, additive_lod={"n_lods": 1})

    assert isinstance(node, Mesh)
    group = zarr.open_group(str(store), mode="r")["surf"]
    assert "additive_0" not in group
    assert "n_additive_sublods" not in group.attrs
    # A real leaf, with its own arrays and the full face count.
    assert "vertices" in group and "faces" in group
    assert int(group.attrs["n_faces"]) == int(faces.shape[0])


def test_additive_ladder_round_trips_through_the_reader(tmp_path) -> None:
    """The reader agrees with the writer about the ladder and its levels.

    The parent is a mesh node with no arrays of its own, so a reader that treated
    it as a leaf would raise; the levels are ordinary mesh nodes. Reassembling the
    levels must reproduce the source triangle SET, which is the end-to-end
    statement that the re-indexing is lossless.
    """
    store, vertices, faces = _write_ladder(tmp_path, additive_lod={"n_lods": 4})
    scene = LuxarScene.load(store)

    assert scene.list_meshes() == ["surf"]
    metadata = scene.get_node_metadata("surf")
    assert metadata["n_additive_sublods"] == 4
    assert metadata["n_faces"] == int(faces.shape[0])
    # No union label CSR on the parent — deliberately absent for a mesh ladder.
    assert metadata.get("has_labels") is not True

    reassembled = set()
    for i in range(4):
        data = scene.get_mesh(f"surf/additive_{i}")
        # Map the level's renumbered faces back to source vertex COORDINATES; the
        # source indices are not recoverable (that is the point of re-indexing).
        level_vertices = data.vertices.astype(np.float64)
        for triangle in data.faces:
            reassembled.add(tuple(sorted(_key(level_vertices[c]) for c in triangle)))

    source_vertices = vertices.astype(np.float64)
    expected = {
        tuple(sorted(_key(source_vertices[c]) for c in triangle)) for triangle in faces
    }
    assert reassembled == expected


def test_additive_lod_stamps_ONE_scalar_window_on_every_level(tmp_path) -> None:
    """Every shell shares one ``scalar_data_range`` — the whole field's.

    The same rule the substitutive and partition wrappers follow, and it matters
    here for the same reason: the viewer windows a node's colormap on that node's
    OWN stamped range, so a shell that stamped its own subset min/max would render
    the same scalar value as a different colour from its neighbour, and a shell
    whose subset is constant would stamp a degenerate ``[v, v]`` the viewer maps to
    the LUT midpoint. A radial reveal makes this near-certain rather than a corner
    case: shells are spatially contiguous, so any scalar field with spatial
    structure has a different range in each one.
    """
    vertices, faces = _grid_mesh(13)
    scalars = np.linspace(-5.0, 25.0, vertices.shape[0]).astype(np.float32)

    store = tmp_path / "win.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "surf",
            vertices,
            faces,
            scalars=scalars,
            colormap="viridis",
            additive_lod={"n_lods": 4},
        )

    parent = zarr.open_group(str(store), mode="r")["surf"]
    ranges = {
        tuple(parent[f"additive_{i}"].attrs["scalar_data_range"]) for i in range(4)
    }
    assert ranges == {(-5.0, 25.0)}, f"levels window on different ranges: {ranges}"
    # Restated on purpose: a shared window is the fix, a non-degenerate one is what
    # makes the colormap usable at all.
    assert all(lo < hi for lo, hi in ranges)


def test_additive_lod_propagates_a_resolved_extend_to_all(tmp_path) -> None:
    """``extend_to_all="all"`` reaches the parent AND every level, resolved.

    Two failure modes in one assertion. The writer stamps the value VERBATIM, so an
    unresolved ``"all"`` sentinel would reach disk where the viewer expects
    dimension names. And mesh resolves ``extend_to_all`` into ``attrs`` BEFORE it
    dispatches its structural branches, while the wrapper also takes it by name —
    so a ladder that forwarded the whole attrs dict would die with "multiple values
    for keyword argument", which is exactly how this broke for the partition branch.
    """
    from luxar import Dimension

    vertices, faces = _grid_mesh(11)
    dims = Dimensions(
        [
            Dimension("x", unit="um", range=(-10, 10), display=True),
            Dimension("y", unit="um", range=(-10, 10), display=True),
            Dimension("z", unit="um", range=(-10, 10), display=True),
            Dimension("t", unit="s", range=(0, 0), display=False),
        ]
    )
    store = tmp_path / "ext.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_mesh(
            "surf",
            np.hstack([vertices, np.zeros((vertices.shape[0], 1), np.float32)]),
            faces,
            additive_lod={"n_lods": 3},
            extend_to_all="all",
        )

    parent = zarr.open_group(str(store), mode="r")["surf"]
    assert parent.attrs["extend_to_all"] == ["t"]
    for i in range(3):
        assert parent[f"additive_{i}"].attrs["extend_to_all"] == ["t"]


def test_no_sub_LOD_carries_the_private_skip_scene_bounds_flag(tmp_path) -> None:
    """`_skip_scene_bounds` is plumbing between writers, not part of the format.

    The ladder writers pass it to the per-level write to say "the parent will
    aggregate the bbox, do not do it per level". It was popped BELOW
    `group.attrs.update(attrs)`, so every sub-LOD of every ladder carried it on
    disk — an internal flag in the on-disk format, and one that round-trips: a tool
    that reads a level's attrs and re-writes them hands it straight back as a
    caller attribute.

    Shared with the Points and Lines ladder writers, which had the identical
    ordering; this asserts the mesh one and their own suites cover theirs.
    """
    store, _vertices, _faces = _write_ladder(tmp_path, additive_lod={"n_lods": 3})
    parent = zarr.open_group(str(store), mode="r")["surf"]

    assert "_skip_scene_bounds" not in parent.attrs
    for i in range(int(parent.attrs["n_additive_sublods"])):
        level = parent[f"additive_{i}"]
        assert "_skip_scene_bounds" not in level.attrs, (
            f"additive_{i} carries the private flag; it is popped after "
            "`group.attrs.update(attrs)` again"
        )
    # The flag must still DO its job: the parent describes the whole ladder.
    assert "position_bounds" in parent.attrs


# --- textures end to end (#2175) -------------------------------------------

_TEX_UV = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
_TEX_RGB = np.arange(8 * 4 * 3, dtype=np.uint8).reshape(8, 4, 3)


def _write_textured(tmp_path, name="m", **kwargs):
    """Write a textured tetrahedron. `uvs` defaults but stays overridable."""
    kwargs.setdefault("uvs", _TEX_UV)
    store = tmp_path / f"{name}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(name, _V, _F, **kwargs)
    return LuxarScene.load(store).get_mesh(name)


def test_raw_texture_round_trips_byte_exact_with_its_declared_dimensions(
    tmp_path,
) -> None:
    """A raw texture survives the round trip, and declares what it decodes to.

    The declared dimensions are asserted for a RAW texture even though they are
    redundant with the array's own shape, because that redundancy is the point: a
    reader must never have to open the payload to learn how large it decodes to.
    The viewer's admission gate budgets a node from these numbers before it
    fetches a single chunk.
    """
    mesh = _write_textured(tmp_path, texture=_TEX_RGB)
    assert np.array_equal(mesh.texture, _TEX_RGB)
    assert mesh.texture.dtype == np.uint8
    assert np.array_equal(mesh.uvs, _TEX_UV)
    assert mesh.metadata["has_texture"] is True
    assert mesh.metadata["has_uvs"] is True
    assert mesh.metadata["texture_encoding"] == "raw"
    assert mesh.metadata["texture_height"] == 8
    assert mesh.metadata["texture_width"] == 4
    assert mesh.metadata["texture_channels"] == 3
    # sRGB is the default because an ordinary PNG/JPEG is sRGB-encoded, and
    # getting this wrong gives a subtly over-dark surface rather than an obvious
    # failure.
    assert mesh.metadata["texture_color_space"] == "srgb"


def test_hdr_texture_keeps_its_range_and_stamps_a_window(tmp_path) -> None:
    """A float texture above 1.0 is HDR, and says so the way colours do.

    Follows the element-colour contract exactly rather than inventing a second
    one: float dtype plus any value > 1.0 is HDR, and the RGB min/max is stamped
    (alpha excluded) so the viewer derives a window from one shape whatever the
    source. PRECISION keeps it float32 — the encoder deliberately refuses float16
    for colours as measurably worse than its quantized alternative.
    """
    from luxar.encoding import EncodingMode

    hdr = np.zeros((4, 4, 3), dtype=np.float32)
    hdr[..., 0] = 6.5
    store = tmp_path / "hdr.luxar.zarr"
    with LuxarZarrCompiler(
        store, encoding_mode=EncodingMode.PRECISION, compressor=None
    ) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh(
            "m", _V, _F, uvs=_TEX_UV, texture=hdr, texture_color_space="linear"
        )
    mesh = LuxarScene.load(store).get_mesh("m")
    assert mesh.texture.dtype == np.float32
    assert float(mesh.texture.max()) == pytest.approx(6.5)
    assert mesh.metadata["texture_data_range"] == [0.0, 6.5]
    assert mesh.metadata["texture_color_space"] == "linear"


def test_hdr_texture_refuses_srgb_transfer(tmp_path) -> None:
    hdr = np.full((4, 4, 3), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="HDR values above 1.0 cannot use"):
        _write_textured(tmp_path, texture=hdr)


@pytest.mark.parametrize("color_space", ["sRGB", "banana", "rec2020"])
def test_bad_texture_color_space_is_refused(tmp_path, color_space) -> None:
    with pytest.raises(ValueError, match="texture_color_space must be one of"):
        _write_textured(tmp_path, texture=_TEX_RGB, texture_color_space=color_space)


def test_sdr_float_texture_stamps_no_hdr_window(tmp_path) -> None:
    """A float texture entirely within [0, 1] is SDR, and gets no range attr.

    The negative twin of the test above, and the one that pins the rule is on
    VALUES rather than on dtype — a build keyed on "is it float?" would stamp a
    window here.
    """
    sdr = np.full((4, 4, 3), 0.5, dtype=np.float32)
    mesh = _write_textured(tmp_path, texture=sdr)
    assert "texture_data_range" not in mesh.metadata


def test_encoded_texture_round_trips_as_opaque_bytes(tmp_path) -> None:
    """Encoded payloads are stored and returned byte-identical, undecoded.

    The reader deliberately does not decode: `luxar.io` has no image-codec
    dependency and should not acquire one. `texture_encoding` plus the declared
    dimensions tell a consumer exactly what it is holding.

    Uses a synthetic byte string rather than a real PNG on purpose — the writer
    treats the payload as opaque, so requiring Pillow here would only test
    Pillow.
    """
    blob = np.frombuffer(b"\x89PNG\r\n\x1a\n" + bytes(range(64)), dtype=np.uint8)
    mesh = _write_textured(
        tmp_path,
        texture=blob,
        texture_encoding="png",
        texture_width=4,
        texture_height=8,
        texture_channels=3,
    )
    assert np.array_equal(mesh.texture, blob)
    assert mesh.metadata["texture_encoding"] == "png"
    # Declared, not derived — they cannot be read from encoded bytes without
    # decoding them, which is exactly why they are mandatory.
    assert mesh.metadata["texture_width"] == 4
    assert mesh.metadata["texture_height"] == 8


def test_ktx2_texture_encodes_raw_rgba_and_stores_opaque_bytes(
    tmp_path, monkeypatch
) -> None:
    """The public mesh API invokes the optional encoder and stores its result."""
    encoded = np.frombuffer(b"\xabKTX 20\xbb\r\n\x1a\nfixture", dtype=np.uint8)
    calls = []

    def fake_encode(texture, mode, quality, color_space):
        calls.append((texture.copy(), mode, quality, color_space))
        return encoded.copy()

    monkeypatch.setattr(
        "luxar.io._compiler.dataset_writers.texture._encode_ktx2", fake_encode
    )
    source = np.zeros((4, 8, 4), dtype=np.uint8)
    source[..., 3] = np.arange(8, dtype=np.uint8)
    mesh = _write_textured(
        tmp_path,
        texture=source,
        texture_encoding="ktx2",
        texture_ktx2_mode="etc1s",
        texture_ktx2_quality=200,
    )
    assert np.array_equal(mesh.texture, encoded)
    assert mesh.metadata["texture_encoding"] == "ktx2"
    assert mesh.metadata["texture_width"] == 8
    assert mesh.metadata["texture_height"] == 4
    assert mesh.metadata["texture_channels"] == 4
    assert len(calls) == 1
    assert calls[0][1:] == ("etc1s", 200, "srgb")
    assert np.array_equal(calls[0][0][..., 3], source[..., 3])


def test_ktx2_encoder_defaults_to_uastc_and_preserves_rgba_input(
    tmp_path, monkeypatch
) -> None:
    """The subprocess contract keeps alpha and selects the agreed default codec."""
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    seen = {}

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "", "toktx v4.4.2\n")
        seen["command"] = command
        seen["source"] = Path(command[-1]).read_bytes()
        Path(command[-2]).write_bytes(_ktx2_bytes(2))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: "/usr/bin/toktx")
    monkeypatch.setattr(texture_writer.subprocess, "run", fake_run)
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = [[0, 64, 255], [255, 64, 0]]
    encoded = texture_writer._encode_ktx2(rgba, "uastc", None, "srgb")
    assert bytes(encoded) == _ktx2_bytes(2)
    assert seen["command"][1:8] == [
        "--t2",
        "--genmipmap",
        "--encode",
        "uastc",
        "--uastc_quality",
        "2",
        "--zcmp",
    ]
    assert seen["command"][8:11] == ["3", "--assign_oetf", "srgb"]
    header, pixels = seen["source"].split(b"ENDHDR\n", 1)
    assert b"TUPLTYPE RGB_ALPHA" in header
    assert np.array_equal(
        np.frombuffer(pixels, dtype=np.uint8).reshape(rgba.shape), rgba
    )


def test_ktx2_encoder_writes_rgb_as_binary_ppm(monkeypatch) -> None:
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    seen = {}

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "toktx 4.4.2\n", "")
        seen["source"] = Path(command[-1]).read_bytes()
        Path(command[-2]).write_bytes(_ktx2_bytes(2))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: "/usr/bin/toktx")
    monkeypatch.setattr(texture_writer.subprocess, "run", fake_run)
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    texture_writer._encode_ktx2(rgb, "uastc", None, "srgb")
    header, pixels = seen["source"].split(b"255\n", 1)
    assert header == b"P6\n3 2\n"
    assert np.array_equal(np.frombuffer(pixels, dtype=np.uint8).reshape(rgb.shape), rgb)


def test_ktx2_encoder_uses_supported_etc1s_spelling(monkeypatch) -> None:
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    seen = {}

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "", "toktx v4.0.0\n")
        seen["command"] = command
        Path(command[-2]).write_bytes(_ktx2_bytes(1))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: "/usr/bin/toktx")
    monkeypatch.setattr(texture_writer.subprocess, "run", fake_run)
    texture_writer._encode_ktx2(
        np.zeros((2, 2, 3), dtype=np.uint8), "etc1s", 128, "srgb"
    )
    assert seen["command"][3:5] == ["--encode", "etc1s"]


def test_ktx2_encoder_rejects_old_toktx_with_actionable_version(monkeypatch) -> None:
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "toktx v3.2.1\n")

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: "/usr/bin/toktx")
    monkeypatch.setattr(texture_writer.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"toktx 4\.0\.0 or newer.*3\.2\.1"):
        texture_writer._encode_ktx2(
            np.zeros((2, 2, 3), dtype=np.uint8), "uastc", None, "srgb"
        )


@pytest.mark.parametrize(
    "mode,scheme",
    [("etc1s", 0), ("uastc", 1)],
    ids=["etc1s_not_basis_lz", "uastc_not_zstandard"],
)
def test_ktx2_encoder_rejects_wrong_supercompression_scheme(
    monkeypatch, mode, scheme
) -> None:
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "toktx v4.4.2\n", "")
        Path(command[-2]).write_bytes(_ktx2_bytes(scheme))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: "/usr/bin/toktx")
    monkeypatch.setattr(texture_writer.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=rf"toktx.*{mode}.*supercompression"):
        texture_writer._encode_ktx2(
            np.zeros((2, 2, 3), dtype=np.uint8), mode, None, "srgb"
        )


def test_ktx2_encoder_rejects_malformed_output(monkeypatch) -> None:
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    def fake_run(command, **kwargs):
        if command[-1] == "--version":
            return subprocess.CompletedProcess(command, 0, "toktx v4.4.2\n", "")
        Path(command[-2]).write_bytes(b"ktx2")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: "/usr/bin/toktx")
    monkeypatch.setattr(texture_writer.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match=r"toktx.*valid KTX2 header"):
        texture_writer._encode_ktx2(
            np.zeros((2, 2, 3), dtype=np.uint8), "uastc", None, "srgb"
        )


def test_ktx2_encoder_missing_binary_has_actionable_fallback(monkeypatch) -> None:
    from luxar.io._compiler.dataset_writers import texture as texture_writer

    monkeypatch.setattr(texture_writer.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"toktx.*raw.*jpeg"):
        texture_writer._encode_ktx2(
            np.zeros((2, 2, 3), dtype=np.uint8), "uastc", None, "srgb"
        )


@pytest.mark.parametrize(
    "kwargs,error_pattern,test_id",
    [
        (
            dict(texture=_TEX_RGB, colors=np.zeros((4, 3), np.float32)),
            "one base colour source",
            "texture_and_colors",
        ),
        (
            dict(texture=_TEX_RGB, scalars=np.zeros(4, np.float32), colormap="viridis"),
            "one base colour source",
            "texture_and_colormap",
        ),
        (
            dict(texture=_TEX_RGB, partition={"max_elements": 2}),
            "cannot be combined with partition",
            "texture_and_partition",
        ),
        (
            dict(texture=_TEX_RGB, substitutive_lod={"levels": 2}),
            "cannot be combined with substitutive_lod",
            "texture_and_substitutive_lod",
        ),
        (
            dict(texture=_TEX_RGB, additive_lod={"n_lods": 2}),
            "cannot be combined with additive_lod",
            "texture_and_additive_lod",
        ),
    ],
)
def test_texture_composition_refusals(tmp_path, kwargs, error_pattern, test_id) -> None:
    """Each unsupported texture combination is refused by name, with a reason.

    The three structural routes are each coherent to want and each needs work
    nothing does yet, so they are named as *not implemented* rather than as
    meaningless — the distinction the sibling refusals in this adder are careful
    to draw. A UV sphere needs none of them.
    """
    with pytest.raises(ValueError, match=error_pattern):
        _write_textured(tmp_path, name=f"m_{test_id}", **kwargs)


@pytest.mark.parametrize(
    "kwargs,error_pattern,test_id",
    [
        (dict(texture=_TEX_RGB), "'texture' requires 'uvs'", "texture_alone"),
        (dict(uvs=_TEX_UV), "'uvs' requires 'texture'", "uvs_alone"),
    ],
)
def test_uvs_and_texture_are_a_pair(tmp_path, kwargs, error_pattern, test_id) -> None:
    """Neither half is accepted alone — the mesh peer of normals/normal_dims.

    Both render *something* rather than failing, which is why both are refused
    rather than warned: a texture with no UVs samples one arbitrary texel across
    every triangle, and UVs with no texture cost a per-vertex array to affect
    nothing. Silent-but-wrong is the case the normals pairing already argues must
    be made loud.
    """
    store = tmp_path / f"pair_{test_id}.luxar.zarr"
    with pytest.raises(ValueError, match=error_pattern):
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("m", _V, _F, **kwargs)


def test_uvs_outside_the_unit_square_survive_the_round_trip(tmp_path) -> None:
    """Tiling UVs are not clamped on the way to disk.

    A UV outside [0, 1] is how you tile a detail texture under
    `texture_wrap="repeat"`. Clamping at write time would silently destroy that,
    and the loss would be invisible in every count-based assertion.
    """
    tiling = np.array(
        [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]], dtype=np.float32
    )
    mesh = _write_textured(tmp_path, uvs=tiling, texture=_TEX_RGB)
    assert float(mesh.uvs.max()) == pytest.approx(4.0)


def test_texture_sampling_attrs_round_trip(tmp_path) -> None:
    """`texture_filter` / `texture_wrap` reach the store as authored.

    Authorable because the right answer is data-dependent and the viewer cannot
    infer it: `nearest` is correct for a categorical or index-like texture, where
    interpolating two class ids invents a third that means nothing, and `clamp`
    is correct for a texture that is not meant to tile.
    """
    mesh = _write_textured(
        tmp_path, texture=_TEX_RGB, texture_filter="nearest", texture_wrap="clamp"
    )
    assert mesh.metadata["texture_filter"] == "nearest"
    assert mesh.metadata["texture_wrap"] == "clamp"


def test_texture_sampling_attrs_are_absent_by_default(tmp_path) -> None:
    """Unset means unset — the viewer owns the default, not the writer.

    The negative twin of the round trip above. Stamping a default here would make
    a store indistinguishable from one that authored the same value on purpose,
    which is the distinction `shading` deliberately preserves by only ever
    stamping what it resolved.
    """
    mesh = _write_textured(tmp_path, texture=_TEX_RGB)
    assert "texture_filter" not in mesh.metadata
    assert "texture_wrap" not in mesh.metadata


@pytest.mark.parametrize(
    "attrs,error_pattern,test_id",
    [
        (
            dict(texture_filter="nearest"),
            "'texture_filter' requires 'texture'",
            "filter_without_texture",
        ),
        (
            dict(texture_wrap="clamp"),
            "'texture_wrap' requires 'texture'",
            "wrap_without_texture",
        ),
        (
            dict(texture_filter="nearest", texture_wrap="clamp"),
            "require 'texture'",
            "both_without_texture",
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_sampling_attrs_are_refused_without_a_texture(
    tmp_path, attrs, error_pattern, test_id
) -> None:
    """A sampling attr on an untextured mesh is a silent no-op, so it is refused.

    It would validate, persist, and read back exactly as authored while changing
    no pixel — the same failure `reject_mesh_only_appearance` prevents when these
    are set on a points node, reached from the other direction.
    """
    store = tmp_path / f"{test_id}.luxar.zarr"
    with pytest.raises(ValueError, match=error_pattern):
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("m", _V, _F, **attrs)


@pytest.mark.parametrize("filt", ["bilinear", "LINEAR", 3, None])
def test_bad_texture_filter_is_refused(tmp_path, filt) -> None:
    """The sampling vocabularies are closed, so a typo fails loudly."""
    store = tmp_path / "bad_filter.luxar.zarr"
    with pytest.raises(ValueError, match="Texture filter must be one of"):
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "m", _V, _F, uvs=_TEX_UV, texture=_TEX_RGB, texture_filter=filt
            )


def test_shading_none_is_stamped_as_authored(tmp_path) -> None:
    """`shading='none'` is the unlit arm, and only ever explicit.

    An unlit mesh is what a data basemap needs — a textured globe whose colours
    carry meaning must not be reshaded by a view-anchored key — and it is what the
    other three geometry types always do, being purely emissive.
    """
    mesh = _write_textured(tmp_path, texture=_TEX_RGB, shading="none")
    assert mesh.metadata["shading"] == "none"


def test_shading_none_is_never_a_default(tmp_path) -> None:
    """Nothing resolves TO 'none'; defaulting to it would un-light every mesh.

    Both default arms are checked, because "never a default" is a claim about the
    whole resolution rather than about one branch of it.
    """
    with_normals = _write_textured(
        tmp_path, name="lit", texture=_TEX_RGB, normals=_N, normal_dims=[0, 1, 2]
    )
    assert with_normals.metadata["shading"] == "smooth"
    without = _write_textured(tmp_path, name="unlit", texture=_TEX_RGB)
    assert without.metadata["shading"] == "flat"


def test_unknown_shading_still_names_every_arm(tmp_path) -> None:
    """A typo'd `shading` must advertise the new third arm, not just the old two."""
    store = tmp_path / "bad_shading.luxar.zarr"
    expected = f"shading must be one of {VALID_SHADING_MODES}, got 'phong'"
    with pytest.raises(ValueError, match=re.escape(expected)):
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh("m", _V, _F, shading="phong")
