"""Tests for the render gate's scene generator and its agreement with the manifest.

The generator lives with the gate in ``packages/luxar-viewer/scripts/render-gate``
and the harness reads ``gate-scenes.json`` there; a store the manifest names but
the generator never writes would make every gate run error on that case, and a
store written but never named is dead weight. These tests keep the two in step
and prove a generated store opens.
"""

from __future__ import annotations

import importlib.util
import json
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

_MANIFEST = json.loads((_GATE_DIR / "gate-scenes.json").read_text())


def _manifest_store_names() -> set[str]:
    stores = [
        c["store"] for c in _MANIFEST["exact"] + _MANIFEST["perf"] if "store" in c
    ]
    names = set()
    for store in stores:
        assert store.startswith("gate/") and store.endswith(".luxar.zarr"), store
        names.add(store[len("gate/") : -len(".luxar.zarr")])
    return names


def test_every_scene_name_has_a_writer() -> None:
    assert set(gate_scenes.WRITERS) == set(gate_scenes.SCENE_NAMES)


def test_manifest_names_exactly_the_generated_stores() -> None:
    assert _manifest_store_names() == set(gate_scenes.SCENE_NAMES)


def test_manifest_case_ids_are_unique() -> None:
    ids = [c["id"] for c in _MANIFEST["exact"]] + [c["id"] for c in _MANIFEST["perf"]]
    assert len(ids) == len(set(ids))


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
