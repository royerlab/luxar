"""Focused tests for the galaxy simulation's frame-grid contract."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_galaxy_simulation.py"


def _load_demo_module():
    name = "_luxar_demo_galaxy_simulation_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


@pytest.mark.parametrize("n_frames", [3, 25, 101])
def test_validate_frame_count_accepts_odd_grids(n_frames: int) -> None:
    assert _demo.validate_frame_count(n_frames) == n_frames


@pytest.mark.parametrize("n_frames", [-1, 0, 1, 2])
def test_validate_frame_count_rejects_grids_without_a_time_step(
    n_frames: int,
) -> None:
    with pytest.raises(ValueError, match="at least 3.*time step"):
        _demo.validate_frame_count(n_frames)


@pytest.mark.parametrize("n_frames", [4, 24, 100])
def test_validate_frame_count_rejects_midpoints_between_frames(
    n_frames: int,
) -> None:
    with pytest.raises(ValueError, match="odd.*opening midpoint.*frame"):
        _demo.validate_frame_count(n_frames)


def test_main_validates_frames_before_building(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(_DEMO_PATH), "--frames=24", "--no-serve"])

    with pytest.raises(ValueError, match="odd"):
        _demo.main()


def test_opening_camera_is_solved_once_at_the_cinematic_lens(tmp_path: Path) -> None:
    from luxar.io.reader import LuxarScene

    position = _demo.galaxy_camera_position()
    distance = math.dist((0.0, 0.0, 0.0), position)
    assert distance == pytest.approx(35.05, rel=1e-3)

    output = tmp_path / "galaxy.luxar.zarr"
    _demo.generate_galaxy(output, n_disc=50, n_frames=3)
    viewer_config = LuxarScene.load(output).viewer_config
    assert viewer_config is not None
    assert viewer_config.camera is not None
    assert viewer_config.camera.position == pytest.approx(position)
    assert viewer_config.camera.target == (0.0, 0.0, 0.0)
    assert viewer_config.camera.fov is None
