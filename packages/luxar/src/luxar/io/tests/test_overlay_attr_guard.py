"""Regression tests for the internal overlay-attribute namespace."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar import Dimensions, LuxarZarrCompiler


def test_bare_overlays_group_is_reserved_by_raw_compiler_api() -> None:
    """Only real ``overlays/<name>`` entries use the internal schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "test.luxar.zarr"
        with LuxarZarrCompiler(zarr_path) as compiler:
            compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="reserved for screen-space overlay"):
                compiler.write_group("overlays", blending="max")

        root = zarr.open_group(str(zarr_path), mode="r")
        assert "overlays" not in root


@pytest.mark.parametrize("node_type", ["group", "points", "lines", "gsplats"])
def test_top_level_overlays_name_is_reserved_for_user_nodes(
    tmp_path: Path, node_type: str
) -> None:
    """User nodes cannot occupy the scene-internal overlay namespace."""
    zarr_path = tmp_path / "test.luxar.zarr"
    with LuxarZarrCompiler(zarr_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        with pytest.raises(ValueError, match="reserved for screen-space overlay"):
            if node_type == "group":
                scene.add_group("overlays")
            elif node_type == "points":
                scene.add_points("overlays", np.zeros((1, 3), dtype=np.float32))
            elif node_type == "lines":
                scene.add_lines(
                    "overlays",
                    np.zeros((2, 3), dtype=np.float32),
                    widths=1.0,
                )
            else:
                scene.add_gsplats(
                    "overlays",
                    centers=np.zeros((1, 3), dtype=np.float32),
                    amplitudes=np.ones(1, dtype=np.float32),
                    cholesky_factors=np.ones((1, 6), dtype=np.float32),
                )

    root = zarr.open_group(str(zarr_path), mode="r")
    assert "overlays" not in root


def test_overlays_name_remains_available_below_scene_root(tmp_path: Path) -> None:
    """Only the root namespace collides with screen-space overlay storage."""
    zarr_path = tmp_path / "test.luxar.zarr"
    with LuxarZarrCompiler(zarr_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        group = scene.add_group("geometry")
        group.add_group("overlays", blending_mode="max")

    root = zarr.open_group(str(zarr_path), mode="r")
    assert root["geometry/overlays"].attrs["blending_mode"] == "max"


def test_reserved_user_name_does_not_block_real_overlays(tmp_path: Path) -> None:
    """Internal overlay writers still own and populate ``overlays/``."""
    zarr_path = tmp_path / "test.luxar.zarr"
    with LuxarZarrCompiler(zarr_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="reserved for screen-space overlay"):
            scene.add_group("overlays")
        scene.add_text("Allowed", position=(0.5, 0.5))

    root = zarr.open_group(str(zarr_path), mode="r")
    assert root["overlays/overlay_0"].attrs["text"] == "Allowed"
