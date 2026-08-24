"""Authoring-time warnings for overlapping blend-state hazards."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.conftest import find_repo_relative_file
from luxar.io._compiler.finalize.blending_warnings import warn_overlapping_blending
from luxar.typing_utils.constants import DEFAULT_BLENDING_MODE_BY_GEOMETRY


def _root(n_dims: int = 3) -> zarr.Group:
    root = zarr.group()
    root.attrs["scene_dimensions"] = {
        "dimensions": [
            {
                "name": f"d{index}",
                "unit": "um",
                "range": [0.0, 100.0],
                "step": 1.0,
                "display": index < 3,
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


def test_viewer_default_modes_match_python_contract() -> None:
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


def test_effective_opacity_and_nearest_blend_setter_drive_depth_writes(capsys) -> None:
    root = _root()
    root.attrs["blending_mode"] = "normal"
    root.attrs["opacity"] = 0.9
    holder = root.create_group("holder")
    holder.attrs["opacity"] = 0.5
    _leaf(holder, "dim_line", "lines")
    _leaf(root, "opaque_line", "lines", opacity=1.0)

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert output.count("⚠️") == 1
    assert "overlapping depth-sorted nodes" in output
    assert "holder/dim_line" in output
    assert "opaque_line" in output


@pytest.mark.parametrize("geometry_type", ["points", "gsplats"])
def test_normal_points_and_gsplats_never_count_as_depth_writers(
    geometry_type: str, capsys
) -> None:
    root = _root()
    _leaf(root, "additive", "points")
    _leaf(root, "normal", geometry_type, blending_mode="normal", opacity=1.0)

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""


def test_overlapping_sorted_nodes_warn_but_lod_alternatives_do_not(capsys) -> None:
    root = _root()
    _leaf(root, "surface", "points", blending_mode="normal")
    _leaf(root, "volume", "gsplats", blending_mode="volumetric")
    lod = root.create_group("lod")
    lod.attrs["kind"] = "lod"
    _leaf(lod, "coarse", "points", blending_mode="normal")
    _leaf(lod, "fine", "points", blending_mode="normal")

    warn_overlapping_blending(root)

    output = capsys.readouterr().out
    assert "'surface' (normal) and 'volume' (volumetric)" in output
    assert "'lod/coarse' (normal) and 'lod/fine' (normal)" not in output


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


def test_world_transforms_decide_overlap(capsys) -> None:
    root = _root()
    _leaf(root, "points", "points")
    mesh = _leaf(root, "mesh", "mesh")
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = 10.0
    mesh.attrs["transform"] = transform.T.reshape(-1).tolist()

    warn_overlapping_blending(root)

    assert capsys.readouterr().out == ""
