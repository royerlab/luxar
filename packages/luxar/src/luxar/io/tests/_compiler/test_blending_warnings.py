"""Authoring-time warnings for overlapping blend-state hazards."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.conftest import find_repo_relative_file
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
        "points": ("create-points-node.ts", r"blendingMode:.*\?\? '([^']+)'"),
        "lines": ("create-lines-node.ts", r"blendingMode:.*\?\? '([^']+)'"),
        "gsplats": ("create-gsplats-node.ts", r"blendingMode:.*\?\? '([^']+)'"),
        "mesh": ("create-mesh-node.ts", r"return .*\?\? '([^']+)'"),
    }
    for geometry_type, (filename, pattern) in files.items():
        source_path = find_repo_relative_file(
            Path(f"packages/luxar-viewer/src/rendering/node-factory/{filename}"),
            Path(__file__).resolve(),
        )
        assert source_path is not None, f"cannot locate viewer factory {filename}"
        match = re.search(pattern, source_path.read_text(encoding="utf-8"))
        assert match is not None, f"cannot locate the default mode in {filename}"
        assert match.group(1) == DEFAULT_BLENDING_MODE_BY_GEOMETRY[geometry_type]


def test_viewer_normal_depth_write_contract_matches_warning_logic() -> None:
    source_path = find_repo_relative_file(
        Path("packages/luxar-viewer/src/rendering/blending-state.ts"),
        Path(__file__).resolve(),
    )
    assert source_path is not None, "cannot locate viewer blending-state.ts"
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
    attrs_source = find_repo_relative_file(
        Path("packages/luxar-viewer/src/data/attrs-composer.ts"),
        Path(__file__).resolve(),
    )
    mesh_source = find_repo_relative_file(
        Path("packages/luxar-viewer/src/rendering/materials/mesh/appearance.ts"),
        Path(__file__).resolve(),
    )
    assert attrs_source is not None, "cannot locate viewer attrs-composer.ts"
    assert mesh_source is not None, "cannot locate viewer mesh appearance.ts"
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
            lod_branches=(),
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


def test_each_sorted_offender_warns_only_once(capsys) -> None:
    root = _root()
    _leaf(
        root,
        "envelope",
        "points",
        minimum=[0.0, 0.0, 0.0],
        maximum=[10.0, 10.0, 10.0],
        blending_mode="normal",
    )
    for index in range(3):
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
    assert "'envelope'" in output


def test_world_transforms_decide_overlap(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points")
    mesh = _leaf(root, "mesh", "mesh")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 10.0
    mesh.attrs["transform"] = transform.T.reshape(-1).tolist()

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""
