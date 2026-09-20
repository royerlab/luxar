"""Tests for the LOD visual A/B fixture metadata reader."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_generator() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[5]
        / "packages/luxar-viewer/scripts/generate-lod-visual-ab-fixture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_lod_visual_ab_fixture", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_reads_level_counts_from_both_zarr_formats(
    tmp_path: Path, zarr_format: int
) -> None:
    generator = _load_generator()
    for level, count in enumerate((4096, 16384)):
        positions = tmp_path / "points_additive" / f"child_{level}" / "positions"
        positions.mkdir(parents=True)
        if zarr_format == 2:
            (positions / ".zarray").write_text(json.dumps({"shape": [count, 3]}))
        else:
            (positions / "zarr.json").write_text(
                json.dumps({"node_type": "array", "shape": [count, 3]})
            )

    assert generator.read_level_element_counts(tmp_path) == [4096, 16384]


def test_reads_counts_for_a_named_gsplat_node(tmp_path: Path) -> None:
    generator = _load_generator()
    for level, count in enumerate((1024, 4096)):
        centers = tmp_path / "gsplats_chromatic" / f"child_{level}" / "centers"
        centers.mkdir(parents=True)
        (centers / "zarr.json").write_text(
            json.dumps({"node_type": "array", "shape": [count, 3]})
        )

    assert generator.read_level_element_counts(
        tmp_path, "gsplats_chromatic", "centers"
    ) == [1024, 4096]


def test_reads_count_from_array_reference_metadata(tmp_path: Path) -> None:
    generator = _load_generator()
    centers = tmp_path / "gsplats_spatial" / "child_0" / "centers"
    centers.mkdir(parents=True)
    (centers / "zarr.json").write_text(
        json.dumps(
            {
                "node_type": "array",
                "shape": [0, 3],
                "attributes": {
                    "encoding": {
                        "name": "array_ref",
                        "original_shape": [4096, 3],
                    }
                },
            }
        )
    )

    assert generator.read_level_element_counts(
        tmp_path, "gsplats_spatial", "centers"
    ) == [4096]


def test_rejects_fixture_without_readable_levels(tmp_path: Path) -> None:
    generator = _load_generator()

    with pytest.raises(RuntimeError, match="contains no readable LOD levels"):
        generator.read_level_element_counts(tmp_path)
