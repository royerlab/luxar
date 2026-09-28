"""Tests for the render gate's scene generator.

The generator lives with the gate in ``packages/luxar-viewer/scripts/render-gate``
and writes the stores used by the viewer harness. These tests prove that every
declared scene has a writer and that generated stores open.
"""

from __future__ import annotations

import importlib.util
import json
import zipfile
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


#: The perf-audit writers in the default set (the original six predate ``--small``).
_NEW_DEFAULT = (
    "tp50",
    "sp64",
    "lod_timelapse",
    "pl_timelapse",
    "arrayref_4d",
    "mesh_gsplat_lod",
)


def _manifest_store_names() -> set[str]:
    stores = [
        c["store"] for c in _MANIFEST["exact"] + _MANIFEST["perf"] if "store" in c
    ]
    by_file = {gate_scenes.store_filename(n): n for n in gate_scenes.ALL_SCENE_NAMES}
    names = set()
    for store in stores:
        assert store.startswith("gate/"), store
        assert store[len("gate/") :] in by_file, f"{store} is written by no writer"
        names.add(by_file[store[len("gate/") :]])
    return names


def test_every_scene_name_has_a_writer() -> None:
    assert set(gate_scenes.WRITERS) == set(gate_scenes.ALL_SCENE_NAMES)
    assert len(gate_scenes.ALL_SCENE_NAMES) == len(set(gate_scenes.ALL_SCENE_NAMES))


def test_manifest_names_only_generated_stores() -> None:
    # A subset, not equality: new default stores land in the generator before
    # the manifest cases that use them, and heavy stores are opt-in.
    assert _manifest_store_names() <= set(gate_scenes.ALL_SCENE_NAMES)


def test_heavy_scenes_are_not_in_the_default_set() -> None:
    heavy = set(gate_scenes.HEAVY_SCENE_NAMES)
    assert heavy == {
        "tp2000",
        "sp500",
        "sp2000",
        "adaptive512",
        "normal_gsplats_10m",
        "timelapse51",
    }
    assert not heavy & set(gate_scenes.SCENE_NAMES)


def test_default_run_writes_exactly_the_default_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[str] = []

    def record(name: str):
        def writer(path: Path, *, small: bool = False) -> None:
            written.append(name)

        return writer

    stubs = {name: record(name) for name in gate_scenes.ALL_SCENE_NAMES}
    monkeypatch.setattr(gate_scenes, "WRITERS", stubs)
    gate_scenes.main(["--out", str(tmp_path)])
    assert written == list(gate_scenes.SCENE_NAMES)
    written.clear()
    gate_scenes.main(["--out", str(tmp_path), "--heavy"])
    assert written == list(gate_scenes.HEAVY_SCENE_NAMES)


def test_zip_scenes_are_named_as_zipped_stores() -> None:
    assert gate_scenes.store_filename("zip_mixed") == "zip_mixed.luxar.zarr.zip"
    assert gate_scenes.store_filename("zip_lod_ladder") == (
        "zip_lod_ladder.luxar.zarr.zip"
    )
    assert gate_scenes.store_filename("sp64") == "sp64.luxar.zarr"


def _assert_zip_stored_and_openable(archive: Path, node: str) -> None:
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
    assert infos
    assert {i.compress_type for i in infos} == {zipfile.ZIP_STORED}
    assert node in set(open_group(archive, mode="r").group_keys())


def test_zip_store_is_zip_stored_and_opens(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--only", "tiny_units_ortho"])
    archive = tmp_path / "tiny.luxar.zarr.zip"
    gate_scenes._zip_store(tmp_path / "tiny_units_ortho.luxar.zarr", archive)
    _assert_zip_stored_and_openable(archive, "tiny_lines")


def test_time_partition_writes_bounded_parts_and_poses(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--only", "tp50", "--small"])
    root = open_group(tmp_path / "tp50.luxar.zarr", mode="r")
    partition = root["splats"]
    assert partition.attrs["kind"] == "partition"
    parts = list(partition.group_keys())
    assert len(parts) == 4 * 4  # --small: 4 timepoints x 4 tiles
    for key in parts:
        bounds = partition[key].attrs["position_bounds"]
        assert bounds["min"][3] == bounds["max"][3], key  # one timepoint per part
    poses = json.loads((tmp_path / "POSES.json").read_text())
    entry = poses["scenes"]["tp50"]
    assert entry["store"] == "gate/tp50.luxar.zarr"
    assert entry["time"]["dimIndex"] == 3
    assert entry["time"]["range"] == [0.0, 3.0]
    assert set(entry["poses"]["full"]) == {"position", "target", "up"}


@pytest.mark.slow
@pytest.mark.parametrize("name", _NEW_DEFAULT)
def test_new_default_writers_write_openable_stores(tmp_path: Path, name: str) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--only", name, "--small"])
    root = open_group(tmp_path / f"{name}.luxar.zarr", mode="r")
    assert list(root.group_keys()), name
    poses = json.loads((tmp_path / "POSES.json").read_text())["scenes"]
    assert poses[name]["store"] == f"gate/{name}.luxar.zarr"
    if name == "sp64":
        assert set(poses[name]["poses"]) == {"full", "closeup"}
        assert len(list(root["splats"].group_keys())) == 64


@pytest.mark.slow
def test_zip_mixed_copies_mixed(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--only", "zip_mixed"])
    assert (tmp_path / "mixed.luxar.zarr").is_dir()
    _assert_zip_stored_and_openable(tmp_path / "zip_mixed.luxar.zarr.zip", "points")


def test_manifest_case_ids_are_unique() -> None:
    ids = [c["id"] for c in _MANIFEST["exact"]] + [c["id"] for c in _MANIFEST["perf"]]
    assert len(ids) == len(set(ids))


def test_tiny_scene_writes_an_openable_store(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--only", "tiny_units_ortho"])
    root = open_group(tmp_path / "tiny_units_ortho.luxar.zarr", mode="r")
    assert "tiny_lines" in set(root.group_keys())


@pytest.mark.slow
def test_every_scene_writes_an_openable_store(tmp_path: Path) -> None:
    gate_scenes.main(["--out", str(tmp_path), "--small"])
    for name in gate_scenes.SCENE_NAMES:
        root = open_group(tmp_path / gate_scenes.store_filename(name), mode="r")
        assert list(root.group_keys()), name
