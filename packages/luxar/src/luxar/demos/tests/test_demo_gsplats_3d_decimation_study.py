"""Regression tests for the decimation study rebuild route."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_decimation_study.py"


def _load_demo_module(name: str = "_luxar_demo_decimation_study_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_recompute_rebuilds_every_level_when_parent_changes(
    tmp_path: Path, monkeypatch
) -> None:
    demo = _load_demo_module("_luxar_demo_decimation_recompute_for_tests")
    parent_a = tmp_path / "parent_a.gsplats.zarr"
    parent_b = tmp_path / "parent_b.gsplats.zarr"
    parent_a.mkdir()
    parent_b.mkdir()
    work_dir = tmp_path / "recompute"
    commands: list[tuple[str, ...]] = []

    def fake_luxar(*args: str) -> None:
        commands.append(args)
        Path(args[3]).mkdir(parents=True)

    monkeypatch.setattr(demo, "run_luxar_cli", fake_luxar)
    counts = {
        level["file"].replace(".zip", ""): level["splats"] for level in demo.LEVELS
    }
    monkeypatch.setattr(
        demo,
        "load_gsplat_node",
        lambda path: (
            type("FakeNode", (), {"n_splats": counts[Path(path).name]})(),
            {},
        ),
    )
    monkeypatch.setattr(demo, "PARENT_ARG", parent_a)

    demo.recompute_levels(work_dir)
    first_commands = list(commands)
    commands.clear()
    monkeypatch.setattr(demo, "PARENT_ARG", parent_b)

    demo.recompute_levels(work_dir)

    assert len(first_commands) == 4
    assert len(commands) == 4
    assert commands[0][2] == str(parent_b)
    assert all("--seed" not in command for command in commands)


def test_recompute_rejects_parent_whose_counts_make_labels_false(
    tmp_path: Path, monkeypatch
) -> None:
    demo = _load_demo_module("_luxar_demo_decimation_wrong_parent_for_tests")
    parent = tmp_path / "wrong_parent.gsplats.zarr"
    parent.mkdir()

    def fake_luxar(*args: str) -> None:
        Path(args[3]).mkdir(parents=True)

    class ForeignNode:
        n_splats = 4_000

    monkeypatch.setattr(demo, "run_luxar_cli", fake_luxar)
    monkeypatch.setattr(demo, "load_gsplat_node", lambda _path: (ForeignNode(), {}))
    monkeypatch.setattr(demo, "PARENT_ARG", parent)

    with pytest.raises(ValueError, match=r"--parent.*shipped labels"):
        demo.recompute_levels(tmp_path / "recompute")
