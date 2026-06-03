"""Direct unit tests for luxar.io._compiler.bounds."""

from __future__ import annotations

import numpy as np

from luxar.io._compiler.bounds import compute_position_bounds, update_scene_bounds


def test_compute_position_bounds_basic() -> None:
    pos = np.array([[0.0, 1.0, 2.0], [4.0, -1.0, 2.0]], dtype=np.float32)
    b = compute_position_bounds(pos)
    assert b["min"] == [0.0, -1.0, 2.0]
    assert b["max"] == [4.0, 1.0, 2.0]


def test_compute_position_bounds_empty() -> None:
    pos = np.zeros((0, 3), dtype=np.float32)
    b = compute_position_bounds(pos)
    assert b == {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}


def test_update_scene_bounds_first_node_initializes() -> None:
    out = update_scene_bounds(None, {"min": [0.0, 0.0], "max": [1.0, 1.0]})
    assert out == {"min": [0.0, 0.0], "max": [1.0, 1.0]}


def test_update_scene_bounds_takes_union() -> None:
    cur = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    out = update_scene_bounds(cur, {"min": [-1.0, 0.5], "max": [0.5, 3.0]})
    assert out["min"] == [-1.0, 0.0]
    assert out["max"] == [1.0, 3.0]


def test_update_scene_bounds_extends_dimensionality() -> None:
    cur = {"min": [0.0, 0.0], "max": [1.0, 1.0]}
    out = update_scene_bounds(cur, {"min": [0.0, 0.0, 5.0], "max": [1.0, 1.0, 9.0]})
    assert out["min"] == [0.0, 0.0, 5.0]
    assert out["max"] == [1.0, 1.0, 9.0]
