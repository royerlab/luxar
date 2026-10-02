"""Waypoint trajectories (``luxar.core.trajectories``): validation, the JSON the
viewer reads, and round trips through ``Waypoint``.

The JSON shape is a cross-language contract: the viewer's waypoint driver
(``core/app/camera/waypoint-driver.ts``) reads exactly these snake_case keys, so
a renamed parameter here would silently fall back to a default there.
"""

from __future__ import annotations

import json
import math

import pytest

from luxar import CameraConfig, Waypoint
from luxar import trajectories as T


def _waypoint(trajectory: object) -> Waypoint:
    return Waypoint(
        when={"story": 1},
        camera=CameraConfig(position=(1, 0, 0), target=(0, 0, 0)),
        trajectory=trajectory,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("trajectory", "expected"),
    [
        ("orbit", "orbit"),
        ("fly-through", "fly-through"),
        (T.Orbit(), {"kind": "orbit"}),
        (T.ZoomPan(), {"kind": "zoom-pan", "rho": math.sqrt(2)}),
        (T.ZoomPan(rho=2.0), {"kind": "zoom-pan", "rho": 2.0}),
        (T.Arc(lift=0.8), {"kind": "arc", "lift": 0.8}),
        (T.Straight(), {"kind": "straight"}),
        (T.Swing(), {"kind": "swing"}),
        (T.Swing(pivot=(1, 2, 3)), {"kind": "swing", "pivot": [1.0, 2.0, 3.0]}),
        (
            T.FlyThrough(look_ahead=0.3, turn=0.25),
            {"kind": "fly-through", "look_ahead": 0.3, "turn": 0.25},
        ),
        (
            T.Via(camera=CameraConfig(position=(0, 0, 50)), leg="arc"),
            {"kind": "via", "camera": {"position": [0, 0, 50]}, "leg": "arc"},
        ),
    ],
)
def test_serialises_to_the_json_the_viewer_reads_and_round_trips(
    trajectory: object, expected: object
) -> None:
    d = json.loads(json.dumps(_waypoint(trajectory).to_dict()))
    assert d["trajectory"] == expected
    assert Waypoint.from_dict(d).trajectory == trajectory


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: T.ZoomPan(rho=0), "rho"),
        (lambda: T.ZoomPan(rho=float("nan")), "rho"),
        (lambda: T.Arc(lift=-0.1), "lift"),
        (lambda: T.FlyThrough(look_ahead=0), "look_ahead"),
        (lambda: T.FlyThrough(turn=0.6), "turn"),
        (lambda: T.Swing(pivot=(0, 0)), "pivot"),  # type: ignore[arg-type]
        (lambda: T.Swing(pivot=(0, float("inf"), 0)), "pivot"),
        (lambda: T.Via(camera=CameraConfig()), "at least one field"),
        (lambda: T.Via(camera=CameraConfig(position=(0, 0, 9)), leg="via"), "leg"),
    ],
)
def test_bad_parameters_are_refused_where_they_are_written(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_a_waypoint_refuses_an_unknown_trajectory() -> None:
    with pytest.raises(ValueError, match="trajectory"):
        _waypoint("spline")
    with pytest.raises(ValueError, match="trajectory"):
        _waypoint({"kind": "zoom-pan"})  # a dict is the wire form, not the API


def test_a_bare_via_name_is_not_a_trajectory() -> None:
    # A Via has to say where it goes through; the name alone names no pose.
    with pytest.raises(ValueError, match="trajectory"):
        _waypoint("via")


def test_every_name_has_a_class_and_every_class_a_name() -> None:
    kinds = {cls.kind for cls in T.TRAJECTORY_TYPES}
    assert set(T.TRAJECTORY_NAMES) | {"via"} == kinds


def test_reading_an_unknown_kind_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown trajectory kind"):
        T.trajectory_from_json({"kind": "teleport"})
