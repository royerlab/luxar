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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Valid enum values (must match TypeScript RenderingSettings union types)
VALID_TONE_MAPPINGS = ("None", "Linear", "Reinhard", "Cineon", "ACES", "AgX", "Neutral")
VALID_CONTROL_TYPES = ("orbit", "fly", "ortho")
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

    Attributes:
        current_step: The current slice position for each dimension.
        selected_dimension: Index of the currently selected dimension for keyboard navigation.
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
    """

    playing: Optional[bool] = None
    target_fps: Optional[float] = None
    loop: Optional[str] = None
    direction: Optional[str] = None

    def __post_init__(self) -> None:
        if self.loop is not None and self.loop not in VALID_LOOP_MODES:
            raise ValueError(
                f"loop must be one of {VALID_LOOP_MODES}, got '{self.loop}'"
            )
        if self.direction is not None and self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got '{self.direction}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary, omitting None fields."""
        result: Dict[str, Any] = {}
        for field_name in ("playing", "target_fps", "loop", "direction"):
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
        1. localStorage per-scene user overrides
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

    # Scene appearance
    background_color: Optional[str] = None

    # Rendering pipeline
    # None lets the viewer use its default ("ACES"), which gives the most
    # consistent, pleasing HDR look but intentionally shifts hues. Set
    # "Neutral" when exact color fidelity matters (e.g. scientific colormap
    # LUTs); the compiler warns when a LUT is used under the ACES default.
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
    auto_rotate_speed: Optional[float] = None
    # Touchpad-friendly orbit drag mapping (LEFT=rotate, RIGHT=pan). When
    # unset, the viewer derives a default from `navigator.platform` (true on
    # macOS, false elsewhere) and persists the user's choice per-scene.
    natural_drag: Optional[bool] = None

    # Cinematic effects
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

    # UI panel visibility
    ui: Optional[UIConfig] = None

    # Theme
    theme: Optional[str] = None

    # Dimension navigation state
    dimensions: Optional[DimensionsConfig] = None

    # Animation state (per-dimension, list indexed by dimension)
    animation: Optional[List[AnimationConfig]] = None

    def __post_init__(self) -> None:
        """Validate all configuration values."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values. Raises ValueError on invalid values."""
        # Camera validation is handled by CameraConfig.__post_init__

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

        if self.theme is not None and self.theme not in VALID_THEMES:
            raise ValueError(f"theme must be one of {VALID_THEMES}, got '{self.theme}'")

        _validate_range(self.exposure, "exposure", -5.0, 5.0)
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
        _validate_min(self.fly_movement_speed, "fly_movement_speed", 0)
        _validate_min(self.fly_rotation_speed, "fly_rotation_speed", 0)
        _validate_range(self.fly_damping, "fly_damping", 0, 1)
        _validate_range(self.fly_rotation_damping, "fly_rotation_damping", 0, 1)
        _validate_min(self.vignette_offset, "vignette_offset", 0)

    # -- Simple field names for sparse serialization --
    _SIMPLE_FIELDS = [
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
        "theme",
    ]

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

        kwargs: Dict[str, Any] = {
            "camera": camera,
            "ui": ui,
            "dimensions": dimensions,
            "animation": animation,
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
