"""Camera trajectories for waypoint flights.

A :class:`~luxar.core.viewer_config.Waypoint` flies the camera from wherever it
is to the waypoint's pose. HOW it gets there is the trajectory: pass one of the
classes below as ``Waypoint(trajectory=...)``, or its name as a string for its
defaults (``trajectory="zoom-pan"``).

Every trajectory is measured by the same PERCEIVED length: panning in view
heights, zooming in log scale and turning in radians. That is what a waypoint's
``speed`` divides to time the flight, so with ``speed`` set, a long jump takes
longer than a short hop at the same cruising pace whichever trajectory is
chosen, and zooming counts as travel. ``easing`` (``"cruise"``, ``"smooth"``,
``"ease-in-out"``, ``"linear"``) shapes the speed along any of them.

Choosing one:

================  ==============================================  ==========================
Trajectory        What the camera does                            Reach for it when
================  ==============================================  ==========================
``Orbit``         The target slides in a straight line; the       A short re-aim. The
                  camera's direction turns and its distance       default.
                  changes geometrically.
``ZoomPan``       Pulls back while it travels and dives in at     Long jumps across a
                  the end, at a constant perceived speed.         dataset: the smoothest.
``Arc``           ``Orbit`` plus a pull-back you set: the         A predictable hop of a
                  distance rises by ``lift`` times the travel.    chosen height.
``Straight``      Camera and target move in straight lines; it    A dolly, or a deliberate
                  may pass through the data.                      pass through the data.
``Swing``         Camera and target travel on great circles       Touring a cloud from the
                  about a pivot (default: the scene centre).      outside, round it.
``FlyThrough``    First person: flies in a straight line          A journey INTO the data.
                  looking ahead, then turns to the stop.
``Via``           Two legs through an intermediate pose, at       Stepping back out (to an
                  rest there.                                     overview) between stops.
================  ==============================================  ==========================

``Straight``, ``Swing`` and ``FlyThrough`` decide where the camera LOOKS along the
way, so under auto-rotate they take the view direction from the turntable for
the flight (the spin resumes from where they land); the others keep the
turntable's live direction and move only the target and distance. Every
trajectory but ``Orbit`` needs a perspective camera; an orthographic flight
follows ``Orbit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

if TYPE_CHECKING:  # viewer_config imports this module; import it lazily at runtime
    from .viewer_config import CameraConfig

#: Names a waypoint may give instead of a trajectory object (its defaults).
TRAJECTORY_NAMES = ("orbit", "zoom-pan", "arc", "straight", "swing", "fly-through")
#: What a ``Via`` leg may be (anything but another ``Via``).
LEG_NAMES = TRAJECTORY_NAMES


def _positive(name: str, value: float) -> None:
    """Raise unless ``value`` is finite and strictly positive."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be finite and > 0, got {value}")


def _fraction(name: str, value: float, *, upper: float) -> None:
    """Raise unless ``value`` is finite and within ``[0, upper]``."""
    if not (math.isfinite(value) and 0 <= value <= upper):
        raise ValueError(f"{name} must be within [0, {upper}], got {value}")


@dataclass(frozen=True)
class Orbit:
    """The default: target in a straight line, direction slerped, distance geometric.

    Predictable and never passes through the target. On a long jump at close
    range it stays close, so the data crosses the screen quickly mid-flight;
    :class:`ZoomPan` is the remedy.
    """

    kind = "orbit"

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        return {"kind": self.kind}


@dataclass(frozen=True)
class ZoomPan:
    """Van Wijk & Nuij's smooth zooming and panning (InfoVis 2003).

    The path through (target, view height) space that minimises perceived
    motion: it pulls back while it travels and dives in at the end, so content
    crosses the screen at an even, moderate speed. A short hop barely zooms; a
    long jump rises high enough to see both ends.

    Attributes:
        rho: The zoom/pan trade-off. ``√2`` (default) is the value their user
            study preferred; larger zooms out further on the same jump, smaller
            pans closer to the ground.
    """

    rho: float = math.sqrt(2)
    kind = "zoom-pan"

    def __post_init__(self) -> None:
        """Validate the parameters where the trajectory is written."""
        _positive("rho", self.rho)

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        return {"kind": self.kind, "rho": self.rho}


@dataclass(frozen=True)
class Arc:
    """:class:`Orbit` with an explicit pull-back.

    The camera's distance rises by ``lift`` times the distance the target
    travels, at mid-flight (a sine bump), then settles onto the destination.
    Unlike :class:`ZoomPan`, the height does not depend on how far in or out the
    two ends are, which makes it the predictable choice when a tour should hop
    by the same proportion every time.

    Attributes:
        lift: Mid-flight pull-back as a fraction of the target travel
            (``0.5`` default; ``0`` is :class:`Orbit`).
    """

    lift: float = 0.5
    kind = "arc"

    def __post_init__(self) -> None:
        """Validate the parameters where the trajectory is written."""
        _fraction("lift", self.lift, upper=100.0)

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        return {"kind": self.kind, "lift": self.lift}


@dataclass(frozen=True)
class Straight:
    """A dolly: the camera and the target each move in a straight line.

    The simplest path in space. It may carry the camera through the data on the
    way, which is the reason to choose it; for a path that stays outside, use
    :class:`Swing` or :class:`ZoomPan`.
    """

    kind = "straight"

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        return {"kind": self.kind}


@dataclass(frozen=True)
class Swing:
    """Travel AROUND a pivot: camera and target follow great circles about it.

    The camera goes round the cloud rather than through or over it, keeping its
    distance from the pivot changing smoothly (geometrically) between the two
    ends. The natural choice for a tour of an object seen from outside, a UMAP
    included.

    Attributes:
        pivot: World-space centre to swing about. ``None`` (default) uses the
            centre of the scene's data.
    """

    pivot: Optional[Tuple[float, float, float]] = None
    kind = "swing"

    def __post_init__(self) -> None:
        """Validate the parameters where the trajectory is written."""
        if self.pivot is not None and (
            len(self.pivot) != 3 or not all(math.isfinite(float(c)) for c in self.pivot)
        ):
            raise ValueError(
                f"pivot must be three finite coordinates, got {self.pivot}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        out: Dict[str, Any] = {"kind": self.kind}
        if self.pivot is not None:
            out["pivot"] = [float(c) for c in self.pivot]
        return out


@dataclass(frozen=True)
class FlyThrough:
    """First-person travel: fly in a straight line, looking where you are going.

    The camera turns from the start target toward the direction of travel over
    the first ``turn`` of the flight, flies looking ahead, and turns onto the
    destination over the last ``turn``. Feels like a flight into the data rather
    than a view of it being moved.

    Attributes:
        look_ahead: How far ahead the camera looks, as a fraction of the trip
            (``0.2`` default).
        turn: Fraction of the flight spent turning at each end (``0.3`` default,
            at most ``0.5``).
    """

    look_ahead: float = 0.2
    turn: float = 0.3
    kind = "fly-through"

    def __post_init__(self) -> None:
        """Validate the parameters where the trajectory is written."""
        _positive("look_ahead", self.look_ahead)
        _fraction("turn", self.turn, upper=0.5)

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        return {"kind": self.kind, "look_ahead": self.look_ahead, "turn": self.turn}


@dataclass(frozen=True)
class Via:
    """Two legs through an intermediate pose, coming to rest there.

    Each leg follows ``leg`` and is eased on its own, so the camera arrives at
    ``camera``, pauses for an instant, and sets off again — stepping back out to
    an overview between two stories, for example. The time is split between the
    legs in proportion to their perceived lengths.

    Attributes:
        camera: The intermediate pose (``CameraConfig``; fields left ``None`` keep
            the live camera's value, as for a waypoint's own pose).
        leg: Trajectory of each leg, by name (``"zoom-pan"`` default).
    """

    camera: CameraConfig
    leg: str = "zoom-pan"
    kind = "via"

    def __post_init__(self) -> None:
        """Validate the parameters where the trajectory is written."""
        from .viewer_config import CameraConfig

        if not isinstance(self.camera, CameraConfig) or not self.camera.to_dict():
            raise ValueError(
                "Via.camera must be a CameraConfig with at least one field set"
            )
        if self.leg not in LEG_NAMES:
            raise ValueError(f"Via.leg must be one of {LEG_NAMES}, got '{self.leg}'")

    def to_dict(self) -> Dict[str, Any]:
        """The JSON the viewer reads: ``{"kind": ..., <parameters>}``."""
        return {"kind": self.kind, "camera": self.camera.to_dict(), "leg": self.leg}


Trajectory = Union[Orbit, ZoomPan, Arc, Straight, Swing, FlyThrough, Via]
TRAJECTORY_TYPES = (Orbit, ZoomPan, Arc, Straight, Swing, FlyThrough, Via)


def validate_trajectory(value: Any) -> None:
    """Raise unless ``value`` is a trajectory object or one of the names."""
    if isinstance(value, TRAJECTORY_TYPES):
        return
    if isinstance(value, str) and value in TRAJECTORY_NAMES:
        return
    raise ValueError(
        f"trajectory must be one of {TRAJECTORY_NAMES} or a trajectory object "
        f"(Orbit, ZoomPan, Arc, Straight, Swing, FlyThrough, Via), got {value!r}"
    )


def trajectory_to_json(value: Union[str, Trajectory]) -> Union[str, Dict[str, Any]]:
    """A name stays a name; an object becomes ``{"kind": ..., <parameters>}``."""
    return value if isinstance(value, str) else value.to_dict()


def trajectory_from_json(
    value: Union[str, Dict[str, Any], None],
) -> Union[str, Trajectory, None]:
    """Inverse of :func:`trajectory_to_json`."""
    if value is None or isinstance(value, str):
        return value
    kind = value.get("kind")
    params = {k: v for k, v in value.items() if k != "kind"}
    if kind == "orbit":
        return Orbit()
    if kind == "zoom-pan":
        return ZoomPan(**params)
    if kind == "arc":
        return Arc(**params)
    if kind == "straight":
        return Straight()
    if kind == "swing":
        pivot = params.get("pivot")
        return Swing(pivot=tuple(pivot) if pivot is not None else None)
    if kind == "fly-through":
        return FlyThrough(**params)
    if kind == "via":
        from .viewer_config import CameraConfig

        return Via(
            camera=CameraConfig.from_dict(params["camera"]),
            leg=params.get("leg", "zoom-pan"),
        )
    raise ValueError(f"unknown trajectory kind {kind!r}")
