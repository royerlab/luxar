"""Cross-implementation parity of the LOD metrics (Python mirror <-> viewer).

:mod:`luxar.io.lod_screening` ports ``lod-selector-math.ts`` and claims the two
agree to ``1e-9``. A hand value pinned on each side does not prove that, so this
module GENERATES a set of ``(box, camera, viewport)`` cases with the Python
metrics into a committed JSON fixture, and the viewer's
``lod-metric-parity.test.ts`` recomputes every case with its own metric. This
test fails when the committed fixture no longer matches what Python computes;
regenerate it with::

    LUXAR_UPDATE_LOD_PARITY=1 hatch run pytest \\
        packages/luxar/src/luxar/io/tests/test_lod_metric_parity.py

The cases cover the regimes the two ports could drift apart in: face-on,
off-axis and rotated groups, a box the eye plane cuts with the eye outside it
(near-clipped), the camera inside a box, behind/off-screen boxes, a degenerate
(line-thin) box, and orthographic cameras. The fixture also records the
module's ``HYSTERESIS_RATIO`` / ``FILL_FACTOR`` and a set of hysteresis picks,
so a change to the viewer's ``config.lod.hysteresisRatio`` default fails there.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from luxar.io.lod_screening import (
    FILL_FACTOR,
    HYSTERESIS_RATIO,
    Box3,
    legacy_coverage_metric,
    perspective_matrix,
    pick_child_with_hysteresis,
    project_box_area_fraction,
    project_box_diagonal_px,
    view_matrix,
)

#: The committed fixture the viewer's parity test reads.
FIXTURE = (
    Path(__file__).resolve().parents[5]
    / "luxar-viewer/src/tests/unit/scene/fixtures/lod-metric-parity.json"
)

Vec3 = Tuple[float, float, float]


def _orthographic_matrix(
    left: float, right: float, top: float, bottom: float, near: float, far: float
) -> np.ndarray:
    """THREE's ``Matrix4.makeOrthographic`` (WebGL clip space), row-major."""
    w, h, p = right - left, top - bottom, far - near
    return np.array(
        [
            [2.0 / w, 0.0, 0.0, -(right + left) / w],
            [0.0, 2.0 / h, 0.0, -(top + bottom) / h],
            [0.0, 0.0, -2.0 / p, -(far + near) / p],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation(axis: str, degrees: float) -> np.ndarray:
    """A 4x4 rotation about ``x`` or ``y``, row-major."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    m = np.eye(4, dtype=np.float64)
    if axis == "x":
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    else:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    return m


def _translation(t: Vec3) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = t
    return m


def _persp(
    fov: float, aspect: float, near: float, far: float, position: Vec3
) -> Dict[str, Any]:
    return {
        "type": "perspective",
        "fov": fov,
        "aspect": aspect,
        "near": near,
        "far": far,
        "position": list(position),
    }


def _ortho(
    extent: Tuple[float, float, float, float], near: float, far: float, position: Vec3
) -> Dict[str, Any]:
    left, right, top, bottom = extent
    return {
        "type": "orthographic",
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "near": near,
        "far": far,
        "position": list(position),
    }


def _projection(camera: Dict[str, Any]) -> np.ndarray:
    if camera["type"] == "perspective":
        return perspective_matrix(
            camera["fov"], camera["aspect"], camera["near"], camera["far"]
        )
    return _orthographic_matrix(
        camera["left"],
        camera["right"],
        camera["top"],
        camera["bottom"],
        camera["near"],
        camera["far"],
    )


ORTHO = _ortho((-4.0, 4.0, 2.25, -2.25), 0.1, 100.0, (0.0, 0.0, 10.0))

#: ``(name, box min, box max, camera, world matrix, viewport)``.
CASES: Sequence[
    Tuple[str, Vec3, Vec3, Dict[str, Any], Optional[np.ndarray], Tuple[int, int]]
] = (
    (
        "face-on cube, 16:9",
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
        _persp(47.0, 16 / 9, 0.01, 100.0, (0.0, 0.0, 3.0)),
        None,
        (1920, 1080),
    ),
    (
        "off-axis box, off-centre camera",
        (1.0, -0.2, -1.0),
        (2.0, 0.6, 0.5),
        _persp(60.0, 1.0, 0.01, 100.0, (0.3, -0.2, 5.0)),
        None,
        (1000, 1000),
    ),
    (
        "rotated, translated group, 21:9",
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        _persp(47.0, 21 / 9, 0.01, 100.0, (0.0, 0.0, 6.0)),
        _translation((0.5, 0.0, 0.0)) @ _rotation("y", 37.0),
        (2520, 1080),
    ),
    (
        "beside the camera, eye plane cuts the box (near-clipped)",
        (2.0, -0.5, -4.0),
        (3.0, 0.5, 1.0),
        _persp(90.0, 1.0, 0.001, 10_000.0, (0.0, 0.0, 0.0)),
        None,
        (1000, 1000),
    ),
    (
        "small box clipped at a real near distance",
        (0.02, 0.0, -0.2),
        (0.04, 0.02, 0.2),
        _persp(90.0, 1.0, 0.1, 100.0, (0.0, 0.0, 0.0)),
        None,
        (1000, 1000),
    ),
    (
        "corners behind the eye, eye outside the box",
        (0.05, -0.3, -3.0),
        (0.6, 0.3, 0.2),
        _persp(60.0, 16 / 9, 0.01, 100.0, (0.0, 0.0, 0.0)),
        None,
        (1920, 1080),
    ),
    (
        "camera inside the box",
        (-2.0, -2.0, -2.0),
        (2.0, 2.0, 2.0),
        _persp(60.0, 1.0, 0.01, 100.0, (0.0, 0.0, 0.0)),
        None,
        (1000, 1000),
    ),
    (
        "wholly behind the camera",
        (-1.0, -1.0, 2.0),
        (1.0, 1.0, 4.0),
        _persp(60.0, 1.0, 0.01, 100.0, (0.0, 0.0, 0.0)),
        None,
        (1000, 1000),
    ),
    (
        "off-screen to the side",
        (50.0, -1.0, -11.0),
        (52.0, 1.0, -9.0),
        _persp(60.0, 1.0, 0.01, 100.0, (0.0, 0.0, 0.0)),
        None,
        (1000, 1000),
    ),
    (
        "line-thin box (degenerate ramp)",
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        _persp(47.0, 16 / 9, 0.01, 100.0, (0.0, 0.0, 4.0)),
        None,
        (1920, 1080),
    ),
    (
        "orthographic, face-on",
        (-1.0, -0.5, -1.0),
        (1.0, 0.5, 1.0),
        ORTHO,
        None,
        (1600, 900),
    ),
    (
        "orthographic, rotated group",
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        ORTHO,
        _rotation("x", 30.0) @ _rotation("y", 20.0),
        (1600, 900),
    ),
    (
        "orthographic, partly outside the view",
        (3.0, 1.0, -1.0),
        (6.0, 4.0, 1.0),
        ORTHO,
        None,
        (1600, 900),
    ),
)

#: ``(thresholds, current index, metric)`` picks around the hysteresis band.
PICKS: Sequence[Tuple[List[float], int, float]] = (
    ([0.0, 0.25, 0.5], 0, 0.3),
    ([0.0, 0.25, 0.5], 2, 0.476),
    ([0.0, 0.25, 0.5], 2, 0.474),
    ([0.0, 0.25, 0.5], 1, 0.226),
    ([0.0, 0.25, 0.5], 1, 0.224),
    ([0.0, 0.0625, 0.125, 0.25, 0.5], 4, 0.2),
    ([0.0, 0.0625, 0.125, 0.25, 0.5], 3, 0.24),
    ([0.0, 0.5, 1.0], 2, math.inf),
)


def _number(value: float) -> Any:
    """JSON has no infinity: ``+inf`` travels as the string ``"inf"``."""
    return "inf" if math.isinf(value) else value


def _case(
    name: str,
    lo: Vec3,
    hi: Vec3,
    camera: Dict[str, Any],
    world: Optional[np.ndarray],
    viewport: Tuple[int, int],
) -> Dict[str, Any]:
    world = np.eye(4) if world is None else world
    proj_view = _projection(camera) @ view_matrix(tuple(camera["position"]))
    box_to_clip = proj_view @ world
    box = Box3(lo, hi)
    near = camera["near"]
    width, height = viewport
    return {
        "name": name,
        "box": {"min": list(lo), "max": list(hi)},
        "camera": camera,
        # Column-major, as THREE's ``Matrix4.elements`` / ``fromArray``.
        "world": world.T.ravel().tolist(),
        "box_to_clip": box_to_clip.T.ravel().tolist(),
        "viewport": [width, height],
        "area_fraction": _number(project_box_area_fraction(box, box_to_clip, near)),
        "diagonal_px": _number(
            project_box_diagonal_px(box, box_to_clip, width, height, near)
        ),
        "legacy_coverage": _number(
            legacy_coverage_metric(box, box_to_clip, width, height, near)
        ),
    }


def build_fixture() -> Dict[str, Any]:
    """The parity fixture, computed by the Python metrics."""
    return {
        "_comment": (
            "Generated by packages/luxar/src/luxar/io/tests/test_lod_metric_parity.py"
            " (LUXAR_UPDATE_LOD_PARITY=1); read by lod-metric-parity.test.ts."
        ),
        "hysteresis_ratio": HYSTERESIS_RATIO,
        "fill_factor": FILL_FACTOR,
        "cases": [_case(*case) for case in CASES],
        "picks": [
            {
                "thresholds": thresholds,
                "current": current,
                "metric": _number(metric),
                "expected": pick_child_with_hysteresis(thresholds, current, metric),
            }
            for thresholds, current, metric in PICKS
        ],
    }


def _dumps(fixture: Dict[str, Any]) -> str:
    """The fixture as JSON with one case / pick per line, so a diff reads per case."""
    parts = []
    for key, value in fixture.items():
        if isinstance(value, list):
            rows = ",\n".join(f"    {json.dumps(row)}" for row in value)
            parts.append(f"  {json.dumps(key)}: [\n{rows}\n  ]")
        else:
            parts.append(f"  {json.dumps(key)}: {json.dumps(value)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def _assert_close(got: Any, want: Any, where: str) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys(), where
        for key in want:
            _assert_close(got[key], want[key], f"{where}.{key}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want, strict=True)):
            _assert_close(g, w, f"{where}[{i}]")
    elif isinstance(want, float):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15), where
    else:
        assert got == want, where


def test_committed_parity_fixture_matches_the_python_metrics() -> None:
    """The fixture the viewer checks against is what Python computes today."""
    generated = json.loads(json.dumps(build_fixture()))
    if os.environ.get("LUXAR_UPDATE_LOD_PARITY") == "1":
        FIXTURE.write_text(_dumps(generated))
    committed = json.loads(FIXTURE.read_text())
    _assert_close(committed, generated, "fixture")


def test_the_cases_exercise_every_regime() -> None:
    """Guard the fixture's coverage: saturated, zero, near-clipped, ortho, ramp."""
    fixture = build_fixture()
    areas = {c["name"]: c["area_fraction"] for c in fixture["cases"]}
    assert areas["camera inside the box"] == "inf"
    assert areas["wholly behind the camera"] == 0.0
    assert areas["off-screen to the side"] == 0.0
    clipped = areas["beside the camera, eye plane cuts the box (near-clipped)"]
    assert clipped == pytest.approx(0.25, rel=1e-9)
    assert 0.0 < areas["corners behind the eye, eye outside the box"] < 1.0
    assert areas["line-thin box (degenerate ramp)"] > 0.0
    assert all(
        0.0 < areas[n] < 1.0 for n in areas if n.startswith("orthographic, face")
    )
