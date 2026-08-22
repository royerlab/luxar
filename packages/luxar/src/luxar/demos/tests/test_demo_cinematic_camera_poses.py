"""Regression tests for demo poses that cannot use target-relative pull-in."""

from __future__ import annotations

import math

import numpy as np
import pytest

from luxar.demos import demo_gsplats_lod_embryo_line as embryo_line
from luxar.demos import demo_lsystem_forest as forest
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG


def test_forest_camera_stays_outside_the_tree_square() -> None:
    camera = forest._viewer_config().camera
    assert camera is not None

    position = np.asarray(camera.position)
    target = np.asarray(camera.target)
    half_size = forest.FOREST_SIZE / 2.0
    distance = float(np.linalg.norm(position - target))

    assert position[0] < -half_size or position[1] < -half_size
    assert distance * math.tan(math.radians(CINEMATIC_FOV_DEG / 2.0)) == pytest.approx(
        half_size
    )


def test_embryo_camera_preserves_the_near_end_standoff() -> None:
    diameter = 1.0
    spacing = diameter * embryo_line.SPACING_FACTOR
    total_len = (embryo_line.COUNT - 1) * spacing
    camera = embryo_line.camera_for_line(embryo_line.COUNT, spacing, diameter)

    old_standoff = total_len * 0.12
    new_standoff = -camera.position[0]

    assert new_standoff > 0.0
    assert new_standoff * math.tan(
        math.radians(CINEMATIC_FOV_DEG / 2.0)
    ) == pytest.approx(
        old_standoff * math.tan(math.radians(embryo_line.AUTHORED_CAMERA_FOV_DEG / 2.0))
    )
