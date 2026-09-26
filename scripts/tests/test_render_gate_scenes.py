"""Tests for the render gate's scene generator.

The generator lives with the gate in ``packages/luxar-viewer/scripts/render-gate``
and writes the stores used by the viewer harness. These tests prove that every
declared scene has a writer and that generated stores open.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from luxar._zarr_compat import open_group

_GATE_DIR = (
    Path(__file__).resolve().parents[2] / "packages/luxar-viewer/scripts/render-gate"
)
_spec = importlib.util.spec_from_file_location(
    "generate_gate_scenes", _GATE_DIR / "generate_gate_scenes.py"
)
assert _spec is not None and _spec.loader is not None
gate_scenes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_scenes)


def test_every_scene_name_has_a_writer() -> None:
    assert set(gate_scenes.WRITERS) == set(gate_scenes.SCENE_NAMES)


def test_tiny_scene_writes_an_openable_store(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--only", "tiny_units_ortho"])
    root = open_group(tmp_path / "tiny_units_ortho.luxar.zarr", mode="r")
    assert "tiny_lines" in set(root.group_keys())


@pytest.mark.slow
def test_every_scene_writes_an_openable_store(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path)])
    for name in gate_scenes.SCENE_NAMES:
        root = open_group(tmp_path / f"{name}.luxar.zarr", mode="r")
        assert list(root.group_keys()), name
