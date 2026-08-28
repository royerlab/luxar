"""Authoring-time warnings for overlapping blend-state hazards."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.conftest import viewer_source
from luxar.io._compiler.bounds import WorldBoundsLeaf
from luxar.io._compiler.finalize.blending_warnings import (
    _MESH_SUPPORTED_BLENDING_MODES,
    _BlendLeaf,
    _candidate_pairs,
    warn_overlapping_blending,
)
from luxar.typing_utils._format_contract import GEOMETRY_TYPES
from luxar.typing_utils.constants import DEFAULT_BLENDING_MODE_BY_GEOMETRY


def _root(n_dims: int = 3) -> zarr.Group:
    root = zarr.group()
    root.attrs["type"] = "scene"
    root.attrs["scene_dimensions"] = {
        "dimensions": [
            {
                "name": f"d{index}",
                "unit": "um",
                "range": [0.0, 100.0],
                "step": 1.0,
                "display": index < 3,
                "discrete": index >= 3,
            }
            for index in range(n_dims)
        ]
    }
    return root


def _leaf(
    parent: zarr.Group,
    name: str,
    geometry_type: str,
    minimum: list[float] | None = None,
    maximum: list[float] | None = None,
    **attrs: object,
) -> zarr.Group:
    group = parent.create_group(name)
    group.attrs.update(
        {
            "type": geometry_type,
            "position_bounds": {
                "min": minimum or [0.0, 0.0, 0.0],
                "max": maximum or [1.0, 1.0, 1.0],
            },
            **attrs,
        }
    )
    return group


def _warning_output(output: str) -> str:
    return "\n".join(line for line in output.splitlines() if "⚠️" in line)


def test_viewer_default_modes_match_python_contract() -> None:
    assert set(DEFAULT_BLENDING_MODE_BY_GEOMETRY) == set(GEOMETRY_TYPES)
    files = {
        "points": (
            viewer_source("src/rendering/node-factory/create-points-node.ts"),
            r"blendingMode:.*\?\? '([^']+)'",
        ),
        "lines": (
            viewer_source("src/rendering/node-factory/create-lines-node.ts"),
            r"blendingMode:.*\?\? '([^']+)'",
        ),
        "gsplats": (
            viewer_source("src/rendering/node-factory/create-gsplats-node.ts"),
            r"blendingMode:.*\?\? '([^']+)'",
        ),
        "mesh": (
            viewer_source("src/rendering/node-factory/create-mesh-node.ts"),
            r"return .*\?\? '([^']+)'",
        ),
    }
    for geometry_type, (source_path, pattern) in files.items():
        filename = source_path.name
        match = re.search(pattern, source_path.read_text(encoding="utf-8"))
        assert match is not None, f"cannot locate the default mode in {filename}"
        assert match.group(1) == DEFAULT_BLENDING_MODE_BY_GEOMETRY[geometry_type]


def test_viewer_normal_depth_write_contract_matches_warning_logic() -> None:
    source_path = viewer_source("src/rendering/blending-state.ts")
    source = source_path.read_text(encoding="utf-8")

    threshold = re.search(
        r"function normalModeDepthWrite\([^)]*\).*?return opacity >= ([0-9.]+)",
        source,
        re.DOTALL,
    )
    assert threshold is not None
    assert float(threshold.group(1)) == 0.99
    assert re.search(
        r"function getGSplatNormalBlendingState\(\).*?depthWrite: false",
        source,
        re.DOTALL,
    )
    assert re.search(
        r"function getPointBlendingState\([^}]+mode === 'normal'"
        r" \? \{ \.\.\.state, depthWrite: false \}",
        source,
        re.DOTALL,
    )
    assert re.search(
        r"function needsDepthSort\([^)]*\).*?return isNormalMode\(mode\)"
        r" \|\| isVolumetricMode\(mode\)",
        source,
        re.DOTALL,
    )
    assert re.search(
        r"function normalizeBlendingMode\([^)]*\).*?return 'normal'",
        source,
        re.DOTALL,
    )


def test_viewer_scene_root_and_mesh_mode_contracts_match_warning_logic() -> None:
    attrs_source = viewer_source("src/data/attrs-composer.ts")
    mesh_source = viewer_source("src/rendering/materials/mesh/appearance.ts")
    assert "if (root.type !== 'scene') chain.push(root);" in attrs_source.read_text(
        encoding="utf-8"
    )

    source = mesh_source.read_text(encoding="utf-8")
    match = re.search(
        r"MESH_SUPPORTED_BLENDING_MODES = \[(.*?)\] as const",
        source,
        re.DOTALL,
    )
    assert match is not None
    assert (
        set(re.findall(r"'([^']+)'", match.group(1)))
        == set(_MESH_SUPPORTED_BLENDING_MODES)
        == {
            "opaque",
            "normal",
            "additive",
            "luminous",
            "max",
        }
    )


def test_default_additive_points_over_opaque_mesh_warns(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points")
    _leaf(root, "nuclei", "mesh")

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("⚠️") == 1
    assert "'points' (additive)" in output
    assert "'nuclei' (opaque)" in output
    assert "blending_mode='luminous'" in output


def test_explicit_additive_is_treated_as_intentional(capsys) -> None:
    root = _root()
    holder = root.create_group("holder")
    holder.attrs["blending_mode"] = "additive"
    _leaf(holder, "points", "points")
    _leaf(root, "mesh", "mesh")

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_scene_root_blending_mode_does_not_compose(capsys) -> None:
    root = _root()
    root.attrs["blending_mode"] = "normal"
    _leaf(root, "points", "points")
    _leaf(root, "gsplats", "gsplats")

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_scene_root_opacity_does_not_hide_depth_writer(capsys) -> None:
    root = _root()
    root.attrs["opacity"] = 0.5
    _leaf(root, "points", "points")
    _leaf(root, "line", "lines", blending_mode="normal", opacity=1.0)

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("⚠️") == 1
    assert "'points' (additive)" in output
    assert "'line' (normal)" in output


def test_compiler_finalize_reports_default_points_mesh_hazard(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "points",
            positions=np.array([[0.1, 0.1, -0.1], [0.2, 0.2, 0.1]], dtype=np.float32),
        )
        scene.add_mesh("mesh", vertices, faces)

    output = capsys.readouterr().out
    assert output.count("mix depth-ignoring and depth-writing geometry") == 1
    store = zarr.open_group(str(output_path), mode="r")
    assert "blending_mode" not in store["points"].attrs
    assert "blending_mode" not in store["mesh"].attrs


def test_additive_lod_chunks_do_not_warn_against_their_parent(
    tmp_path: Path, capsys
) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    positions = np.random.default_rng(0).uniform(0.0, 1.0, (12, 3)).astype(np.float32)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            positions,
            blending_mode="normal",
            additive_lod={"n_lods": 3},
        )

    assert "overlapping order-dependent nodes" not in capsys.readouterr().out


def test_additive_lod_warning_names_parent_once(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    positions = np.random.default_rng(1).uniform(0.0, 1.0, (12, 3)).astype(np.float32)
    vertices = np.array(
        [[-1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("cloud", positions, additive_lod={"n_lods": 3})
        scene.add_mesh("surface", vertices, faces)

    output = _warning_output(capsys.readouterr().out)
    assert output.count("mix depth-ignoring and depth-writing geometry") == 1
    assert "'cloud' (additive)" in output
    assert "cloud/additive_" not in output


def test_partitioned_node_warning_names_parent_once(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    positions = np.random.default_rng(2).uniform(0.0, 1.0, (12, 3)).astype(np.float32)
    vertices = np.array(
        [[-1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("cloud", positions, partition={"max_elements": 3})
        scene.add_mesh("surface", vertices, faces)

    output = _warning_output(capsys.readouterr().out)
    assert output.count("mix depth-ignoring and depth-writing geometry") == 1
    assert "'cloud' (additive)" in output
    assert "cloud/part_" not in output


def test_separate_partitioned_nodes_warn_separately(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "scene.luxar.zarr"
    positions = np.random.default_rng(3).uniform(0.0, 1.0, (12, 3)).astype(np.float32)
    vertices = np.array(
        [[-1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points("cloud_a", positions, partition={"max_elements": 3})
        scene.add_points("cloud_b", positions + 0.1, partition={"max_elements": 3})
        scene.add_mesh("surface", vertices, faces)

    output = _warning_output(capsys.readouterr().out)
    assert output.count("mix depth-ignoring and depth-writing geometry") == 2
    assert "'cloud_a' (additive)" in output
    assert "'cloud_b' (additive)" in output
    assert "/part_" not in output


def test_effective_opacity_and_nearest_blend_setter_drive_depth_writes(capsys) -> None:
    root = _root()
    holder = root.create_group("holder")
    holder.attrs["blending_mode"] = "normal"
    holder.attrs["opacity"] = 0.9
    nested = holder.create_group("nested")
    nested.attrs["opacity"] = 0.5
    _leaf(nested, "dim_line", "lines")
    _leaf(holder, "opaque_line", "lines", opacity=1.0)

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("⚠️") == 1
    assert "overlapping order-dependent nodes" in output
    assert "holder/nested/dim_line" in output
    assert "holder/opaque_line" in output


def test_depth_writing_sorted_nodes_do_not_warn(capsys) -> None:
    root = _root()
    _leaf(
        root,
        "outer",
        "lines",
        maximum=[10.0, 10.0, 10.0],
        blending_mode="normal",
        opacity=1.0,
    )
    _leaf(
        root,
        "inner",
        "lines",
        minimum=[1.0, 1.0, 1.0],
        maximum=[2.0, 2.0, 2.0],
        blending_mode="normal",
        opacity=1.0,
    )

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("geometry_type", ["points", "gsplats"])
def test_normal_points_and_gsplats_never_count_as_depth_writers(
    geometry_type: str, capsys
) -> None:
    root = _root()
    _leaf(root, "additive", "points")
    _leaf(root, "normal", geometry_type, blending_mode="normal", opacity=1.0)

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_set_but_unknown_mode_matches_viewer_normal_fallback(capsys) -> None:
    root = _root()
    _leaf(root, "additive", "points")
    _leaf(root, "malformed", "lines", blending_mode="", opacity=1.0)

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "'malformed' (normal)" in output
    assert "mix depth-ignoring and depth-writing geometry" in output


def test_inherited_unsupported_mesh_mode_falls_back_to_opaque(capsys) -> None:
    root = _root()
    holder = root.create_group("holder")
    holder.attrs["blending_mode"] = "volumetric"
    _leaf(holder, "mesh", "mesh")
    _leaf(root, "points", "points")

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "'holder/mesh' (opaque)" in output
    assert "'points' (additive)" in output


def test_overlapping_sorted_nodes_warn_but_lod_alternatives_do_not(capsys) -> None:
    root = _root()
    _leaf(root, "surface", "points", blending_mode="normal")
    _leaf(root, "volume", "gsplats", blending_mode="volumetric")

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "'surface' (normal)" in output
    assert "'volume' (volumetric)" in output

    root = _root()
    lod = root.create_group("lod")
    lod.attrs["kind"] = "lod"
    _leaf(lod, "coarse", "points", blending_mode="normal")
    _leaf(lod, "fine", "points", blending_mode="normal")

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_partially_overlapping_sorted_nodes_do_not_warn(capsys) -> None:
    root = _root()
    _leaf(root, "left", "points", blending_mode="normal")
    _leaf(
        root,
        "right",
        "gsplats",
        minimum=[0.5, 0.0, 0.0],
        maximum=[1.5, 1.0, 1.0],
        blending_mode="volumetric",
    )

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_slider_separation_suppresses_warning(capsys) -> None:
    root = _root(4)
    _leaf(
        root,
        "time_0",
        "points",
        minimum=[0.0, 0.0, 0.0, 0.0],
        maximum=[1.0, 1.0, 1.0, 0.0],
    )
    _leaf(
        root,
        "time_1",
        "mesh",
        minimum=[0.0, 0.0, 0.0, 1.0],
        maximum=[1.0, 1.0, 1.0, 1.0],
    )

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_equal_slider_coordinate_remains_co_visible(capsys) -> None:
    root = _root(4)
    _leaf(
        root,
        "points",
        "points",
        minimum=[0.0, 0.0, 0.0, 2.0],
        maximum=[1.0, 1.0, 1.0, 2.0],
    )
    _leaf(
        root,
        "mesh",
        "mesh",
        minimum=[0.0, 0.0, 0.0, 2.0],
        maximum=[1.0, 1.0, 1.0, 2.0],
    )

    warn_overlapping_blending(root)

    assert capsys.readouterr().out.count("⚠️") == 1


def test_spatial_boundary_contact_is_not_reported_as_overlap(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points", maximum=[1.0, 1.0, 1.0])
    _leaf(
        root,
        "mesh",
        "mesh",
        minimum=[1.0, 0.0, 0.0],
        maximum=[2.0, 1.0, 1.0],
    )

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_non_sweep_axis_boundary_contact_is_not_reported_as_overlap(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points", maximum=[1.0, 1.0, 1.0])
    _leaf(
        root,
        "mesh",
        "mesh",
        minimum=[0.0, 1.0, 0.0],
        maximum=[1.0, 2.0, 1.0],
    )

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_spatial_sweep_prunes_disjoint_leaf_pairs() -> None:
    leaves = [
        _BlendLeaf(
            WorldBoundsLeaf(
                path=f"leaf_{index}",
                geometry_type="points",
                bounds={
                    "min": [float(index * 2), 0.0, 0.0],
                    "max": [float(index * 2 + 1), 1.0, 1.0],
                },
            ),
            mode="normal",
            opacity=1.0,
        )
        for index in range(1000)
    ]

    assert list(_candidate_pairs(leaves, {0, 1, 2})) == []


def test_each_additive_offender_warns_only_once(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points")
    _leaf(root, "mesh_a", "mesh")
    _leaf(root, "mesh_b", "mesh")

    warn_overlapping_blending(root)

    assert capsys.readouterr().out.count("⚠️") == 1


def test_sorted_overlap_cluster_warns_once_with_every_remedy(capsys) -> None:
    root = _root()
    _leaf(
        root,
        "envelope",
        "points",
        minimum=[0.0, 0.0, 0.0],
        maximum=[20.0, 10.0, 10.0],
        blending_mode="normal",
    )
    for index in range(10):
        _leaf(
            root,
            f"contained_{index}",
            "points",
            minimum=[float(index + 1), 1.0, 1.0],
            maximum=[float(index + 2), 2.0, 2.0],
            blending_mode="normal",
        )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("⚠️") == 1
    assert "'envelope' (normal) [contains another node]" in output
    for index in range(4):
        assert f"'contained_{index}' (normal)" in output
    assert "contained_4" not in output
    assert "and 6 more" in output
    assert 'partition={"max_elements": N}' in output
    assert "Put the displayed dimensions first" not in output
    assert "Points, Lines, Mesh, and Gaussian Splats" in output
    assert "emissive medium" in output
    assert "changes surface appearance" in output


def test_disconnected_sorted_overlap_clusters_warn_separately(capsys) -> None:
    root = _root()
    for prefix, offset in (("left", 0.0), ("right", 20.0)):
        _leaf(
            root,
            f"{prefix}_outer",
            "points",
            minimum=[offset, 0.0, 0.0],
            maximum=[offset + 10.0, 10.0, 10.0],
            blending_mode="normal",
        )
        _leaf(
            root,
            f"{prefix}_inner",
            "points",
            minimum=[offset + 1.0, 1.0, 1.0],
            maximum=[offset + 2.0, 2.0, 2.0],
            blending_mode="normal",
        )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    warnings = [line for line in output.splitlines() if "⚠️" in line]
    assert len(warnings) == 2
    assert "left_inner" in warnings[0]
    assert "left_outer" in warnings[0]
    assert "right_" not in warnings[0]
    assert "right_inner" in warnings[1]
    assert "right_outer" in warnings[1]
    assert "left_" not in warnings[1]
    assert output.count('partition={"max_elements": N}') == 1
    assert output.count("For an emissive medium") == 1


def test_heterogeneous_sorted_overlap_does_not_suggest_merging(capsys) -> None:
    root = _root()
    _leaf(
        root,
        "cloud",
        "points",
        maximum=[10.0, 10.0, 10.0],
        blending_mode="normal",
        opacity=0.9,
    )
    _leaf(
        root,
        "surface",
        "mesh",
        minimum=[1.0, 1.0, 1.0],
        maximum=[2.0, 2.0, 2.0],
        blending_mode="normal",
        opacity=0.9,
    )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "Merging cannot preserve this cluster" in output
    assert "partition=" not in output
    assert "otherwise separate their bounds" in output


def test_heterogeneous_overlap_omits_partition_displayed_dimension_advice(
    capsys,
) -> None:
    root = _root(n_dims=4)
    scene_dimensions = root.attrs["scene_dimensions"]
    for index, dimension in enumerate(scene_dimensions["dimensions"]):
        dimension["display"] = index in {1, 2, 3}
        dimension["discrete"] = index == 0
    root.attrs["scene_dimensions"] = scene_dimensions
    _leaf(
        root,
        "cloud",
        "points",
        maximum=[10.0, 10.0, 10.0, 10.0],
        blending_mode="normal",
        opacity=0.9,
    )
    _leaf(
        root,
        "surface",
        "mesh",
        minimum=[1.0, 1.0, 1.0, 1.0],
        maximum=[2.0, 2.0, 2.0, 2.0],
        blending_mode="normal",
        opacity=0.9,
    )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "Merging cannot preserve this cluster" in output
    assert "Displayed dimensions are position columns" not in output
    assert "Put the displayed dimensions first" not in output


def test_mixed_mode_sorted_overlap_does_not_suggest_merging(capsys) -> None:
    root = _root()
    _leaf(
        root,
        "outer",
        "points",
        maximum=[10.0, 10.0, 10.0],
        blending_mode="normal",
    )
    _leaf(
        root,
        "inner",
        "points",
        minimum=[1.0, 1.0, 1.0],
        maximum=[2.0, 2.0, 2.0],
        blending_mode="volumetric",
    )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "Merging cannot preserve this cluster" in output
    assert "partition=" not in output


def test_mixed_opacity_sorted_overlap_does_not_suggest_merging(capsys) -> None:
    root = _root()
    _leaf(
        root,
        "outer",
        "points",
        maximum=[10.0, 10.0, 10.0],
        blending_mode="normal",
        opacity=0.2,
    )
    _leaf(
        root,
        "inner",
        "points",
        minimum=[1.0, 1.0, 1.0],
        maximum=[2.0, 2.0, 2.0],
        blending_mode="normal",
        opacity=0.95,
    )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "Merging cannot preserve this cluster" in output
    assert "opacities differ" in output
    assert "partition=" not in output


def test_shared_advice_covers_mergeable_and_unmergeable_clusters(capsys) -> None:
    root = _root(n_dims=4)
    scene_dimensions = root.attrs["scene_dimensions"]
    for index, dimension in enumerate(scene_dimensions["dimensions"]):
        dimension["display"] = index in {1, 2, 3}
        dimension["discrete"] = index == 0
    root.attrs["scene_dimensions"] = scene_dimensions
    for name, geometry_type, minimum, maximum in (
        ("merge_outer", "points", [0.0] * 4, [10.0] * 4),
        ("merge_inner", "points", [1.0] * 4, [2.0] * 4),
        ("mixed_outer", "points", [20.0, 0.0, 0.0, 0.0], [30.0, 10.0, 10.0, 10.0]),
        ("mixed_inner", "mesh", [21.0, 1.0, 1.0, 1.0], [22.0, 2.0, 2.0, 2.0]),
    ):
        _leaf(
            root,
            name,
            geometry_type,
            minimum=minimum,
            maximum=maximum,
            blending_mode="normal",
            opacity=0.9,
        )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("overlapping order-dependent nodes") == 2
    assert output.count('partition={"max_elements": N}') == 1
    assert "Put the displayed dimensions first" not in output
    assert output.count("Merging cannot preserve this cluster") == 1
    assert output.count("For an emissive medium") == 1


def test_additive_warning_precedes_sorted_overlap_cluster(capsys) -> None:
    root = _root()
    _leaf(root, "additive", "points", maximum=[1.0, 1.0, 1.0])
    _leaf(root, "writer", "mesh", maximum=[1.0, 1.0, 1.0])
    _leaf(
        root,
        "outer",
        "points",
        minimum=[20.0, 0.0, 0.0],
        maximum=[30.0, 10.0, 10.0],
        blending_mode="normal",
    )
    _leaf(
        root,
        "inner",
        "points",
        minimum=[21.0, 1.0, 1.0],
        maximum=[22.0, 2.0, 2.0],
        blending_mode="normal",
    )

    warn_overlapping_blending(root)

    warning_lines = [
        line for line in capsys.readouterr().out.splitlines() if "⚠️" in line
    ]
    assert len(warning_lines) == 2
    assert "mix depth-ignoring and depth-writing geometry" in warning_lines[0]
    assert "overlapping order-dependent nodes" in warning_lines[1]


def test_nonstandard_displayed_dimensions_do_not_add_obsolete_advice(capsys) -> None:
    root = _root(n_dims=4)
    scene_dimensions = root.attrs["scene_dimensions"]
    for index, dimension in enumerate(scene_dimensions["dimensions"]):
        dimension["display"] = index in {1, 2, 3}
        dimension["discrete"] = index == 0
    root.attrs["scene_dimensions"] = scene_dimensions
    _leaf(
        root,
        "outer",
        "points",
        maximum=[10.0, 10.0, 10.0, 10.0],
        blending_mode="normal",
    )
    _leaf(
        root,
        "inner",
        "points",
        minimum=[1.0, 1.0, 1.0, 1.0],
        maximum=[2.0, 2.0, 2.0, 2.0],
        blending_mode="normal",
    )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert 'partition={"max_elements": N}' in output
    assert "Displayed dimensions are position columns" not in output
    assert "Put the displayed dimensions first" not in output


def test_two_dimensional_scene_warns_for_overlap_cluster(capsys) -> None:
    root = _root(n_dims=2)
    _leaf(
        root,
        "outer",
        "points",
        maximum=[10.0, 10.0],
        blending_mode="normal",
    )
    _leaf(
        root,
        "inner",
        "points",
        minimum=[1.0, 1.0],
        maximum=[2.0, 2.0],
        blending_mode="normal",
    )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "overlapping order-dependent nodes" in output


def test_overlap_union_joins_existing_multi_member_clusters(capsys) -> None:
    root = _root()
    for name, minimum, maximum in (
        ("a_outer", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
        ("b_inner", [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]),
        ("c_outer", [20.0, 0.0, 0.0], [30.0, 10.0, 10.0]),
        ("d_inner", [21.0, 1.0, 1.0], [22.0, 2.0, 2.0]),
        ("e_bridge", [0.5, 0.5, 0.5], [22.5, 2.5, 2.5]),
    ):
        _leaf(
            root,
            name,
            "points",
            minimum=minimum,
            maximum=maximum,
            blending_mode="normal",
        )

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("⚠️") == 1
    for name in ("a_outer", "b_inner", "c_outer", "d_inner", "e_bridge"):
        assert f"'{name}' (normal)" in output


def test_multiple_modes_for_one_owner_use_clear_separator(capsys) -> None:
    root = _root()
    world_leaves = [
        WorldBoundsLeaf(
            path="multi/normal",
            owner_path="multi",
            geometry_type="points",
            bounds={"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
            blending_mode="normal",
        ),
        WorldBoundsLeaf(
            path="multi/volumetric",
            owner_path="multi",
            geometry_type="points",
            bounds={"min": [0.0, 0.0, 0.0], "max": [10.0, 10.0, 10.0]},
            blending_mode="volumetric",
        ),
        WorldBoundsLeaf(
            path="inner",
            geometry_type="points",
            bounds={"min": [1.0, 1.0, 1.0], "max": [2.0, 2.0, 2.0]},
            blending_mode="normal",
        ),
    ]

    warn_overlapping_blending(root, world_leaves)

    output = capsys.readouterr().out
    assert "'multi' (normal + volumetric)" in output
    assert "normal/volumetric" not in output


def test_world_transforms_decide_overlap(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points")
    mesh = _leaf(root, "mesh", "mesh")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 10.0
    mesh.attrs["transform"] = transform.T.reshape(-1).tolist()

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""
