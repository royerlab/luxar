"""Viewer configuration hints for the Luxar viewer.

Provides dataclasses that are stored in the zarr file and read by the
viewer at load time, becoming the scene-specific defaults.

Design philosophy: expose everything the user can see and change in the
viewer so that Python -> zarr -> viewer round-trips preserve all settings.
``Scene.to_zarr()`` finalizes and copies the backing store, providing
basic Python-side export; full round-trip read-back into Python is
planned but not currently available.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Valid enum values (must match TypeScript RenderingSettings union types)
VALID_TONE_MAPPINGS = ("None", "Linear", "Reinhard", "Cineon", "ACES", "AgX", "Neutral")
VALID_CONTROL_TYPES = ("orbit", "fly", "ortho")
# Turntable axis. Two families. CAMERA frame, named for what the viewer sees:
# "vertical" = screen-up (the historical and default behavior), "horizontal" =
# screen-right (the scene tumbles over the top), "view" = the view direction (a
# pure roll — the camera never moves). WORLD frame, a fixed scene axis:
# "world-x" / "world-y" / "world-z" — the classic turntable, where the subject
# spins about its own axis at any camera elevation (a camera-frame "vertical"
# turntable makes that axis precess instead). Only the world family takes
# letters, matching what a letter means everywhere else in the repo; a bare
# letter for the camera frame would read as a DATA axis here.
VALID_AUTO_ROTATE_AXES = (
    "vertical",
    "horizontal",
    "view",
    "world-x",
    "world-y",
    "world-z",
)
VALID_FOV_PRESETS = (
    "28mm Wide",
    "35mm",
    "50mm Normal",
    "85mm Portrait",
    "135mm Tele",
    "Custom",
)
# Ambient occlusion is unsupported: SSAO needs surface normals, which
# point/gsplat/line geometry does not provide. The `ao_*` keys are not
# accepted.
VALID_THEMES = ("dark", "light", "frosted-glass", "liquid-glass")
VALID_LOOP_MODES = ("once", "loop", "bounce")
VALID_DIRECTIONS = ("forward", "backward")


@dataclass
class CameraConfig:
    """Initial camera configuration for the viewer.

    All fields are optional — unset fields use the viewer's built-in defaults.
    Camera config is applied on every scene load (not persisted in localStorage).
    An authored position is restored with its resolved scene FOV even when the
    visitor has stored rendering settings.

    Attributes:
        position: Camera position in world coordinates (x, y, z).
        target: Look-at target point in world coordinates (x, y, z).
        up: Camera up vector (x, y, z). Defaults to (0, 1, 0).
        fov: Field of view in degrees (1-180).
        fov_preset: Named FOV preset (e.g. '50mm Normal').
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        target_node: Name of a scene graph node whose bounding box center
            becomes the camera target. Resolved at viewer load time.
            If both target and target_node are set, target_node takes precedence.
    """

    position: Optional[Tuple[float, float, float]] = None
    target: Optional[Tuple[float, float, float]] = None
    up: Optional[Tuple[float, float, float]] = None
    fov: Optional[float] = None
    fov_preset: Optional[str] = None
    near: Optional[float] = None
    far: Optional[float] = None
    target_node: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate camera configuration values."""
        if self.position is not None:
            if len(self.position) != 3:
                raise ValueError(
                    f"position must have 3 elements, got {len(self.position)}"
                )

        if self.target is not None:
            if len(self.target) != 3:
                raise ValueError(f"target must have 3 elements, got {len(self.target)}")

        if self.up is not None:
            if len(self.up) != 3:
                raise ValueError(f"up must have 3 elements, got {len(self.up)}")

        if self.fov is not None:
            if not (1 <= self.fov <= 180):
                raise ValueError(f"fov must be between 1 and 180, got {self.fov}")

        if self.fov_preset is not None and self.fov_preset not in VALID_FOV_PRESETS:
            raise ValueError(
                f"fov_preset must be one of {VALID_FOV_PRESETS}, got '{self.fov_preset}'"
            )

        if self.near is not None and self.near <= 0:
            raise ValueError(f"near must be > 0, got {self.near}")

        if self.far is not None and self.far <= 0:
            raise ValueError(f"far must be > 0, got {self.far}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields."""
        result: Dict[str, Any] = {}
        if self.position is not None:
            result["position"] = list(self.position)
        if self.target is not None:
            result["target"] = list(self.target)
        if self.up is not None:
            result["up"] = list(self.up)
        if self.fov is not None:
            result["fov"] = self.fov
        if self.fov_preset is not None:
            result["fov_preset"] = self.fov_preset
        if self.near is not None:
            result["near"] = self.near
        if self.far is not None:
            result["far"] = self.far
        if self.target_node is not None:
            result["target_node"] = self.target_node
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraConfig:
        """Create from dictionary."""
        return cls(
            position=tuple(data["position"]) if "position" in data else None,
            target=tuple(data["target"]) if "target" in data else None,
            up=tuple(data["up"]) if "up" in data else None,
            fov=data.get("fov"),
            fov_preset=data.get("fov_preset"),
            near=data.get("near"),
            far=data.get("far"),
            target_node=data.get("target_node"),
        )


@dataclass
class UIConfig:
    """UI panel visibility configuration.

    All fields are optional — unset fields use the viewer's built-in defaults.
    """

    show_help: Optional[bool] = None
    show_rendering_controls: Optional[bool] = None
    show_performance_monitor: Optional[bool] = None
    show_dimensions: Optional[bool] = None
    show_scale_bar: Optional[bool] = None
    show_layers: Optional[bool] = None
    show_overlays: Optional[bool] = None

    # All field names for sparse serialization (alphabetical after show_)
    _FIELDS = (
        "show_dimensions",
        "show_help",
        "show_layers",
        "show_overlays",
        "show_performance_monitor",
        "show_rendering_controls",
        "show_scale_bar",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields."""
        result: Dict[str, Any] = {}
        for field_name in self._FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UIConfig:
        """Create from dictionary."""
        return cls(**{f: data.get(f) for f in cls._FIELDS})


@dataclass
class DimensionsConfig:
    """nD dimension navigation state.

    The two fields are indexed DIFFERENTLY, which is the single easiest thing
    to get wrong here — both demos that set ``selected_dimension`` set it
    wrongly before this was written down.

    Attributes:
        current_step: Slice position per dimension, indexed by ABSOLUTE
            dimension index — one entry for every dimension the scene declares,
            displayed ones included.
        selected_dimension: Which dimension the keyboard navigates, indexed by
            NAVIGABLE POSITION among the non-displayed dimensions only. For
            ``[x, y, z, time]`` the only navigable axis is ``time``, so this is
            ``0`` and not ``3``; the number keys follow the same numbering,
            which is why ``1`` selects the first hidden axis. A value at or past
            the navigable count resolves to "nothing selected"
            (``getSelectedDimensionIndex`` in the viewer).
    """

    current_step: Optional[List[float]] = None
    selected_dimension: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields."""
        result: Dict[str, Any] = {}
        if self.current_step is not None:
            result["current_step"] = list(self.current_step)
        if self.selected_dimension is not None:
            result["selected_dimension"] = self.selected_dimension
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DimensionsConfig:
        """Create from dictionary."""
        return cls(
            current_step=list(data["current_step"]) if "current_step" in data else None,
            selected_dimension=data.get("selected_dimension"),
        )


@dataclass
class AnimationConfig:
    """Per-dimension animation state.

    Attributes:
        playing: Whether animation is playing.
        target_fps: Target frames per second.
        loop: Loop mode ('once', 'loop', 'bounce').
        direction: Playback direction ('forward', 'backward').
        step_size: How far one animation tick advances the dimension, in the
            dimension's own units. ``None`` means Auto — the viewer derives a
            step from the dimension's declared ``step`` (discrete) or its range
            (continuous), which is what almost every scene wants.
    """

    playing: Optional[bool] = None
    target_fps: Optional[float] = None
    loop: Optional[str] = None
    direction: Optional[str] = None
    # Present because the VIEWER already writes it. Ctrl+Shift+S captures a
    # per-dimension step override into the scene's `animation` block, and
    # without a field here the round trip silently dropped it: capture a scene
    # with a custom step, read it into Python, write it back, and the override
    # was gone with nothing said. It is the only member of that block Python
    # could not express.
    step_size: Optional[float] = None

    def __post_init__(self) -> None:
        if self.loop is not None and self.loop not in VALID_LOOP_MODES:
            raise ValueError(
                f"loop must be one of {VALID_LOOP_MODES}, got '{self.loop}'"
            )
        if self.direction is not None and self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got '{self.direction}'"
            )
        if self.target_fps is not None and (
            not math.isfinite(self.target_fps) or self.target_fps <= 0
        ):
            raise ValueError(
                f"target_fps must be finite and > 0, got {self.target_fps}"
            )
        # Matches the viewer's own guard in `setStepSize`.
        if self.step_size is not None and (
            not math.isfinite(self.step_size) or self.step_size <= 0
        ):
            raise ValueError(f"step_size must be > 0, got {self.step_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields."""
        result: Dict[str, Any] = {}
        for field_name in ("playing", "target_fps", "loop", "direction", "step_size"):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnimationConfig:
        """Create from dictionary."""
        return cls(
            playing=data.get("playing"),
            target_fps=data.get("target_fps"),
            loop=data.get("loop"),
            direction=data.get("direction"),
            step_size=data.get("step_size"),
        )


VALID_WAYPOINT_EASINGS = ("linear", "ease-in-out")

# A `when` clause: dimension NAME -> exact value, or an inclusive (min, max)
# range. Deliberately the same syntax as an overlay's `visible_range`, so one
# vocabulary describes both "show this caption here" and "look from here".
WaypointCondition = Dict[str, Union[float, Tuple[float, float]]]


@dataclass
class Waypoint:
    """A camera pose bound to a position along the scene's hidden dimensions.

    Waypoints are how a scene tells a story: give the scene a hidden discrete
    "story" dimension, author one waypoint per value, and stepping that
    dimension (keyboard ``[`` / ``]``, the slider, or an external controller
    calling ``setDimensionValue``) flies the camera to the matching pose. The
    matching rule is the overlay rule (``visible_range``): every named
    dimension must match, an exact value matches within ±0.5 of the current
    step, a ``(min, max)`` range matches inclusively. The FIRST matching
    waypoint in list order wins, so put specific clauses before broad ones.

    At load the viewer snaps (no flight) to whichever waypoint matches the
    opening dimension state, ahead of the plain ``camera`` block — the more
    specific pose wins. Afterwards a change of matched waypoint flies; slider
    moves that stay inside the same waypoint's ranges do nothing, and leaving
    every waypoint leaves the camera where it is. A visitor moving the camera
    mid-flight cancels the flight (their hand always wins).

    Attributes:
        when: Condition on dimension positions, keyed by dimension name.
        camera: Pose to reach. Fields left ``None`` keep the live camera's
            value at flight time, so a waypoint may name only a ``target`` (or
            ``target_node``) to re-aim without moving.
        duration_ms: Flight duration in milliseconds; ``None`` uses the viewer
            default (1500). ``0`` snaps.
        easing: ``"ease-in-out"`` (default) or ``"linear"``.
        rendering: Optional rendering overrides applied on arrival, using the
            same snake_case keys as ``ViewerConfig`` itself (``exposure``,
            ``bloom_strength``, ``tone_mapping``, ...). Validated against that
            key list; values are validated by the viewer exactly as authored
            defaults are.
    """

    when: WaypointCondition
    camera: CameraConfig
    duration_ms: Optional[float] = None
    easing: Optional[str] = None
    rendering: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.when, dict) or not self.when:
            raise ValueError("when must be a non-empty dict of dimension name -> value")
        for name, constraint in self.when.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"when keys must be dimension names, got {name!r}")
            if isinstance(constraint, (int, float)) and not isinstance(
                constraint, bool
            ):
                if not math.isfinite(constraint):
                    raise ValueError(f"when[{name!r}] must be finite, got {constraint}")
                continue
            if (
                isinstance(constraint, (tuple, list))
                and len(constraint) == 2
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                    for v in constraint
                )
            ):
                lo, hi = constraint
                if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
                    raise ValueError(
                        f"when[{name!r}] range must be finite with min <= max, got {constraint}"
                    )
                continue
            raise ValueError(
                f"when[{name!r}] must be a number or a (min, max) pair, got {constraint!r}"
            )

        if not isinstance(self.camera, CameraConfig) or not self.camera.to_dict():
            raise ValueError(
                "camera must be a CameraConfig with at least one field set"
            )

        if self.duration_ms is not None and (
            not math.isfinite(self.duration_ms) or self.duration_ms < 0
        ):
            raise ValueError(
                f"duration_ms must be finite and >= 0, got {self.duration_ms}"
            )

        if self.easing is not None and self.easing not in VALID_WAYPOINT_EASINGS:
            raise ValueError(
                f"easing must be one of {VALID_WAYPOINT_EASINGS}, got '{self.easing}'"
            )

        if self.rendering is not None:
            if not isinstance(self.rendering, dict):
                raise ValueError("rendering must be a dict of viewer_config keys")
            unknown = sorted(
                k for k in self.rendering if k not in ViewerConfig._RENDERING_FIELDS
            )
            if unknown:
                raise ValueError(
                    f"rendering has unknown keys {unknown}; use ViewerConfig field "
                    "names such as 'exposure' or 'bloom_strength'"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields."""
        result: Dict[str, Any] = {
            "when": {
                name: (list(c) if isinstance(c, (tuple, list)) else c)
                for name, c in self.when.items()
            },
            "camera": self.camera.to_dict(),
        }
        if self.duration_ms is not None:
            result["duration_ms"] = self.duration_ms
        if self.easing is not None:
            result["easing"] = self.easing
        if self.rendering:
            result["rendering"] = dict(self.rendering)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Waypoint:
        """Create from dictionary."""
        when_raw = data.get("when", {})
        when: WaypointCondition = {
            name: (tuple(c) if isinstance(c, list) else c)
            for name, c in when_raw.items()
        }
        return cls(
            when=when,
            camera=CameraConfig.from_dict(data.get("camera", {})),
            duration_ms=data.get("duration_ms"),
            easing=data.get("easing"),
            rendering=data.get("rendering"),
        )


def _validate_hex_color(color: str) -> None:
    """Validate a hex color string like '#rrggbb'."""
    if not re.match(r"^#[0-9a-fA-F]{6}$", color):
        raise ValueError(
            f"Invalid hex color '{color}'. Expected format: '#rrggbb' (e.g., '#1a1a2e')"
        )


def _validate_range(
    value: Optional[float], name: str, min_val: float, max_val: float
) -> None:
    """Validate that value is within [min_val, max_val] if set."""
    if value is not None and not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def _validate_min(value: Optional[float], name: str, min_val: float) -> None:
    """Validate that value >= min_val if set."""
    if value is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")


@dataclass
class ViewerConfig:
    """Viewer configuration hints stored in the zarr file.

    All fields are optional — only set fields are written to the zarr file.
    Unset fields use the viewer's built-in defaults.

    Priority chain (highest to lowest):
        1. localStorage per-scene user overrides, except an authored camera
           position is restored with its resolved scene FOV
        2. zarr viewer_config (this object)
        3. Viewer application built-in defaults

    Three modes of creation:

        # 1. Programmatic — typed fields with autocomplete
        vc = luxar.ViewerConfig(
            camera=CameraConfig(position=(0, 5, 20), target_node="embryo"),
            bloom_strength=0.5,
        )

        # 2. From snapshot — exported JSON from viewer (Ctrl+Shift+S)
        vc = luxar.ViewerConfig.from_file("my_view.json")

        # 3. Hybrid — load snapshot, tweak specific fields
        vc = luxar.ViewerConfig.from_file("base_view.json")
        vc.bloom_strength = 0.8
        vc.camera.position = (10, 5, 30)
    """

    # Camera
    camera: Optional[CameraConfig] = None

    # Scene identity — shown as the browser tab title (document.title) so
    # several open viewer tabs are tellable apart. Falls back to the ?title=
    # URL parameter `luxar serve --open` derives from the dataset file name.
    title: Optional[str] = None

    # Scene appearance
    background_color: Optional[str] = None

    # Rendering pipeline
    # "ACES" is the recommended choice for almost every scene: its filmic
    # highlight rolloff is what keeps dense, bright structure from clipping
    # flat, and it is also the viewer's default when this is left None. It
    # does intentionally shift hues, so in the narrower case where a colormap
    # LUT carries an exact color encoding that must survive to the screen,
    # prefer "None" — an exact passthrough, valid while the scene stays
    # inside [0, 1]. "Neutral" is not a passthrough: even below its knee it
    # subtracts an offset taken from the channel minimum, so anything but a
    # fully saturated colour moves, and over range it keeps the hue angle but
    # sheds chroma (a `None` clamp distorts both and flattens everything above
    # 1.0) — a gentle rolloff rather than a fidelity choice.
    # Setting this explicitly — to "ACES" as much
    # as to anything else — silences the compiler's LUT tone-mapping notice,
    # which only fires when no choice was made at all.
    tone_mapping: Optional[str] = None
    exposure: Optional[float] = None  # Log2 stops, default 0.0
    global_offset: Optional[float] = None  # Additive shift, default 0.0
    global_gamma: Optional[float] = None  # Midtone curve, default 1.0

    # Bloom
    bloom_enabled: Optional[bool] = None
    bloom_strength: Optional[float] = None
    bloom_radius: Optional[float] = None
    bloom_threshold: Optional[float] = None
    bloom_levels: Optional[int] = None

    # Navigation
    control_type: Optional[str] = None
    auto_rotate: Optional[bool] = None
    # Turntable rate in REVOLUTIONS PER MINUTE (a three.js OrbitControls
    # inheritance): a full turn takes 60 / auto_rotate_speed seconds, so the
    # 0.25 the demos mostly use is one turn every four minutes. Frame-rate
    # independent. The viewer's Navigation popover shows the equivalent PERIOD
    # in seconds — same number, friendlier question — but the stored unit stays
    # a rate so every already-published scene keeps meaning what it meant.
    auto_rotate_speed: Optional[float] = None
    # Axis the turntable revolves around — a camera-frame axis ("vertical",
    # "horizontal", "view") or a fixed scene axis ("world-x"/"-y"/"-z"); see
    # VALID_AUTO_ROTATE_AXES. Left unset the viewer spins about screen-up,
    # which is what `auto_rotate` has always done. For a scene with a natural
    # up whose opening camera looks down at it, the world axis matching that up
    # is usually the one you want: it spins the subject about its own axis,
    # where the camera-frame default makes that axis precess.
    auto_rotate_axis: Optional[str] = None
    # Auto-dolly: oscillate the viewing distance on a sine — the turntable's
    # radial sibling, equivalent to turning the mousewheel back and forth. The
    # amplitude is a PERCENT of the viewing distance (hence the field name), so
    # it means the same thing at any scene scale, and the viewer keeps honouring
    # the user's own zoom while it runs. Unlike the turntable it is also alive
    # in ortho mode, where it breathes the orthographic zoom instead.
    auto_dolly: Optional[bool] = None
    # Peak swing as a percent of the viewing distance (15 -> +/-15%), up to 95.
    # A big swing is a legitimate choice, not a hazard — but it is not free.
    # Screen area goes as 1/d^2, so the swing moves projected area by (1 + a)^4
    # and the LOD ladder answers by loading finer levels at the near extreme:
    # measured on a 100-group demo, 15% keeps 118k elements resident while 95%
    # pulls in 2.29M. A local warm cache absorbs that; a hosted scene pays for
    # it in requests. Prefer a modest amplitude unless the motion is the point.
    auto_dolly_amplitude_percent: Optional[float] = None
    # Seconds per full in-and-out oscillation.
    auto_dolly_period: Optional[float] = None
    # Touchpad-friendly orbit drag mapping (LEFT=rotate, RIGHT=pan). When
    # unset, the viewer derives a default from `navigator.platform` (true on
    # macOS, false elsewhere) and persists the user's choice per-scene.
    natural_drag: Optional[bool] = None

    # Cinematic effects. `cinematic_mode=True` is not just a checkbox: the
    # viewer expands the full preset (ACES tone mapping, subtle wide bloom,
    # detector noise, vignette, 35 mm chromatic lens + FOV) for every field
    # this config does NOT set explicitly, so a scene can enable the look and
    # still override individual fields (e.g. `bloom_strength`) on top of it.
    # The 35 mm field of view (63°) is wider than the viewer's default framing,
    # and `camera.fov` / `camera.fov_preset` count as ONE unit: pin either of
    # them and the preset leaves both alone, so a composed framing survives.
    cinematic_mode: Optional[bool] = None
    vignette_enabled: Optional[bool] = None
    vignette_darkness: Optional[float] = None
    vignette_offset: Optional[float] = None

    # Detector noise (physics-based: Poisson + Gaussian + FPN)
    detector_noise_enabled: Optional[bool] = None
    detector_noise_readout_sigma: Optional[float] = None
    detector_noise_photon_gain: Optional[float] = None
    detector_noise_fpn_sigma: Optional[float] = None

    # Anti-aliasing. SMAA is unsupported because its 3-pass blend does
    # not fit Luxar's single-pass post-processing model. FXAA / MSAA /
    # SSAA remain.
    fxaa_enabled: Optional[bool] = None
    msaa_enabled: Optional[bool] = None
    msaa_samples: Optional[int] = None
    ssaa_enabled: Optional[bool] = None
    ssaa_multiplier: Optional[float] = None

    # Depth-of-Field and Ambient Occlusion are unsupported: DoF needs
    # depth-aware multi-pass blur, and SSAO needs surface normals which
    # point/gsplat/line geometry does not provide. The corresponding
    # `dof_*` / `ao_*` keys are not accepted.

    # Chromatic lens distortion
    chromatic_lens_distortion_enabled: Optional[bool] = None
    chromatic_lens_distortion_x: Optional[float] = None
    chromatic_lens_distortion_y: Optional[float] = None
    chromatic_lens_dispersion: Optional[float] = None
    chromatic_lens_principal_point_x: Optional[float] = None
    chromatic_lens_principal_point_y: Optional[float] = None
    chromatic_lens_focal_length_x: Optional[float] = None
    chromatic_lens_focal_length_y: Optional[float] = None
    chromatic_lens_skew: Optional[float] = None

    # Fly controls
    fly_movement_speed: Optional[float] = None
    fly_rotation_speed: Optional[float] = None
    fly_inertial_mode: Optional[bool] = None
    fly_damping: Optional[float] = None
    fly_rotation_damping: Optional[float] = None

    # Dynamic clipping
    dynamic_clipping_enabled: Optional[bool] = None

    # Adaptive resolution
    adaptive_dpr_enabled: Optional[bool] = None

    # Whether the viewer may render above CSS resolution (device pixel
    # ratio > 1) on a HiDPI display. Off by default: a 2x display costs
    # 4x the fragment work, and for soft-edged emissive geometry the
    # extra pixels buy little.
    #
    # TURN IT ON FOR THIN-LINE SCENES. That is the one case measured to
    # be worth it, and it is worth it twice over:
    #
    # - Dense line work is what the cap actually costs. Measured on a
    #   Retina panel against DPR 2, brightness and coverage are preserved
    #   to within 2.5% on every geometry type — the whole visible effect
    #   is a 15-35% loss of high-frequency detail. On points and splats
    #   that reads as slightly softer. On a river network, a tractogram
    #   or a wiring diagram it reads as MUSH: the individual lines stop
    #   being separable, which is an information loss, not a cosmetic one.
    # - Line scenes are the cheapest place to pay for it. They are not
    #   fill-bound, so they gain least from the cap in the first place:
    #   1.06-1.17x on a trajectory scene, against 2.6-2.7x on a point
    #   cloud. You buy back the detail for almost no frame time.
    #
    # Leave it off for points, gsplats and mesh unless a specific scene
    # proves otherwise; those are where the cap earns its keep.
    allow_high_dpr: Optional[bool] = None

    # UI panel visibility
    ui: Optional[UIConfig] = None

    # Theme
    theme: Optional[str] = None

    # Dimension navigation state
    dimensions: Optional[DimensionsConfig] = None

    # Animation state (per-dimension, list indexed by dimension)
    animation: Optional[List[AnimationConfig]] = None

    # Story waypoints: camera poses bound to hidden-dimension positions. See
    # `Waypoint`. First match in list order wins.
    waypoints: Optional[List[Waypoint]] = None

    def __post_init__(self) -> None:
        """Validate all configuration values."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values. Raises ValueError on invalid values."""
        # Camera validation is handled by CameraConfig.__post_init__

        if self.waypoints is not None:
            if not isinstance(self.waypoints, list) or not all(
                isinstance(w, Waypoint) for w in self.waypoints
            ):
                raise ValueError("waypoints must be a list of Waypoint")

        if self.title is not None:
            if not isinstance(self.title, str) or not self.title.strip():
                raise ValueError(
                    f"title must be a non-empty string, got {self.title!r}"
                )

        if self.background_color is not None:
            _validate_hex_color(self.background_color)

        if (
            self.tone_mapping is not None
            and self.tone_mapping not in VALID_TONE_MAPPINGS
        ):
            raise ValueError(
                f"tone_mapping must be one of {VALID_TONE_MAPPINGS}, got '{self.tone_mapping}'"
            )

        if (
            self.control_type is not None
            and self.control_type not in VALID_CONTROL_TYPES
        ):
            raise ValueError(
                f"control_type must be one of {VALID_CONTROL_TYPES}, got '{self.control_type}'"
            )

        if (
            self.auto_rotate_axis is not None
            and self.auto_rotate_axis not in VALID_AUTO_ROTATE_AXES
        ):
            raise ValueError(
                f"auto_rotate_axis must be one of {VALID_AUTO_ROTATE_AXES}, "
                f"got '{self.auto_rotate_axis}'"
            )

        if self.theme is not None and self.theme not in VALID_THEMES:
            raise ValueError(f"theme must be one of {VALID_THEMES}, got '{self.theme}'")

        _validate_range(self.exposure, "exposure", -10.0, 10.0)
        _validate_range(self.global_offset, "global_offset", -1.0, 1.0)
        _validate_range(self.global_gamma, "global_gamma", 0.1, 10.0)
        _validate_min(self.bloom_strength, "bloom_strength", 0)
        _validate_min(self.bloom_radius, "bloom_radius", 0)
        _validate_range(self.bloom_threshold, "bloom_threshold", 0, 1)
        if self.bloom_levels is not None and self.bloom_levels < 1:
            raise ValueError(f"bloom_levels must be >= 1, got {self.bloom_levels}")
        _validate_range(self.vignette_darkness, "vignette_darkness", 0, 1)
        _validate_range(
            self.detector_noise_readout_sigma, "detector_noise_readout_sigma", 0, 0.1
        )
        _validate_range(
            self.detector_noise_photon_gain, "detector_noise_photon_gain", 0.0001, 0.1
        )
        _validate_range(
            self.detector_noise_fpn_sigma, "detector_noise_fpn_sigma", 0, 0.05
        )

        # Control speeds and damping
        _validate_min(self.auto_rotate_speed, "auto_rotate_speed", 0)
        _validate_range(
            self.auto_dolly_amplitude_percent, "auto_dolly_amplitude_percent", 1, 95
        )
        if self.auto_dolly_period is not None and self.auto_dolly_period <= 0:
            raise ValueError(
                f"auto_dolly_period must be > 0, got {self.auto_dolly_period}"
            )
        _validate_min(self.fly_movement_speed, "fly_movement_speed", 0)
        _validate_min(self.fly_rotation_speed, "fly_rotation_speed", 0)
        _validate_range(self.fly_damping, "fly_damping", 0, 1)
        _validate_range(self.fly_rotation_damping, "fly_rotation_damping", 0, 1)
        _validate_min(self.vignette_offset, "vignette_offset", 0)

    # -- Simple field names for sparse serialization --
    _SIMPLE_FIELDS = [
        "title",
        "background_color",
        "tone_mapping",
        "exposure",
        "global_offset",
        "global_gamma",
        "bloom_enabled",
        "bloom_strength",
        "bloom_radius",
        "bloom_threshold",
        "bloom_levels",
        "control_type",
        "auto_rotate",
        "auto_rotate_speed",
        "auto_rotate_axis",
        "auto_dolly",
        "auto_dolly_amplitude_percent",
        "auto_dolly_period",
        "natural_drag",
        "cinematic_mode",
        "vignette_enabled",
        "vignette_darkness",
        "vignette_offset",
        "detector_noise_enabled",
        "detector_noise_readout_sigma",
        "detector_noise_photon_gain",
        "detector_noise_fpn_sigma",
        "fxaa_enabled",
        "msaa_enabled",
        "msaa_samples",
        "ssaa_enabled",
        "ssaa_multiplier",
        "chromatic_lens_distortion_enabled",
        "chromatic_lens_distortion_x",
        "chromatic_lens_distortion_y",
        "chromatic_lens_dispersion",
        "chromatic_lens_principal_point_x",
        "chromatic_lens_principal_point_y",
        "chromatic_lens_focal_length_x",
        "chromatic_lens_focal_length_y",
        "chromatic_lens_skew",
        "fly_movement_speed",
        "fly_rotation_speed",
        "fly_inertial_mode",
        "fly_damping",
        "fly_rotation_damping",
        "dynamic_clipping_enabled",
        "adaptive_dpr_enabled",
        "allow_high_dpr",
        "theme",
    ]

    # The subset of _SIMPLE_FIELDS a `Waypoint.rendering` patch may carry: the
    # rendering pipeline + navigation feel, i.e. what the viewer's
    # `extractRenderingOverrides` reads. Scene identity and theme are not
    # per-waypoint state.
    _RENDERING_FIELDS = frozenset(
        f for f in _SIMPLE_FIELDS if f not in ("title", "background_color", "theme")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields.

        Returns:
            Dictionary with only non-None fields set.
        """
        result: Dict[str, Any] = {}

        if self.camera is not None:
            cam = self.camera.to_dict()
            if cam:  # only include if camera has any fields set
                result["camera"] = cam

        # Simple fields — emit only if not None
        for field_name in self._SIMPLE_FIELDS:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        # Nested configs
        if self.ui is not None:
            ui_dict = self.ui.to_dict()
            if ui_dict:
                result["ui"] = ui_dict

        if self.dimensions is not None:
            dims_dict = self.dimensions.to_dict()
            if dims_dict:
                result["dimensions"] = dims_dict

        if self.animation is not None:
            anim_list = [a.to_dict() for a in self.animation]
            # Only include if at least one entry has data
            if any(a for a in anim_list):
                result["animation"] = anim_list

        if self.waypoints:
            result["waypoints"] = [w.to_dict() for w in self.waypoints]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ViewerConfig:
        """Create from dictionary. Unknown keys are ignored for forward compatibility.

        Args:
            data: Dictionary from zarr attributes or exported JSON.

        Returns:
            ViewerConfig instance.
        """
        camera = None
        if "camera" in data and isinstance(data["camera"], dict):
            camera = CameraConfig.from_dict(data["camera"])

        ui = None
        if "ui" in data and isinstance(data["ui"], dict):
            ui = UIConfig.from_dict(data["ui"])

        dimensions = None
        if "dimensions" in data and isinstance(data["dimensions"], dict):
            dimensions = DimensionsConfig.from_dict(data["dimensions"])

        animation = None
        if "animation" in data and isinstance(data["animation"], list):
            animation = [AnimationConfig.from_dict(a) for a in data["animation"]]

        waypoints = None
        if "waypoints" in data and isinstance(data["waypoints"], list):
            waypoints = [
                Waypoint.from_dict(w) for w in data["waypoints"] if isinstance(w, dict)
            ]

        kwargs: Dict[str, Any] = {
            "camera": camera,
            "ui": ui,
            "dimensions": dimensions,
            "animation": animation,
            "waypoints": waypoints,
        }

        # Populate simple fields from data
        for field_name in cls._SIMPLE_FIELDS:
            if field_name in data:
                kwargs[field_name] = data[field_name]

        return cls(**kwargs)

    # -- JSON I/O methods --

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> ViewerConfig:
        """Load from JSON file exported by viewer (Ctrl+Shift+S).

        Args:
            path: Path to JSON file.

        Returns:
            ViewerConfig instance.
        """
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_json(cls, json_string: str) -> ViewerConfig:
        """Load from JSON string.

        Args:
            json_string: JSON string (e.g. from clipboard).

        Returns:
            ViewerConfig instance.
        """
        return cls.from_dict(json.loads(json_string))

    def to_file(self, path: Union[str, Path]) -> None:
        """Save to JSON file.

        Args:
            path: Path to write JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON string with indentation.
        """
        return json.dumps(self.to_dict(), indent=2)
