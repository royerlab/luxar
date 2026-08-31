"""Tests for ViewerConfig, CameraConfig, and related dataclasses."""

import json
import re
from pathlib import Path

import pytest

from luxar.conftest import viewer_source
from luxar.core.viewer_config import (
    VALID_FOV_PRESETS,
    AnimationConfig,
    CameraConfig,
    DimensionsConfig,
    UIConfig,
    ViewerConfig,
)


class TestCameraConfig:
    """Tests for CameraConfig validation and serialization."""

    def test_default_camera(self) -> None:
        cam = CameraConfig()
        assert cam.position is None
        assert cam.target is None
        assert cam.up is None
        assert cam.fov is None

    def test_valid_camera(self) -> None:
        cam = CameraConfig(
            position=(1.0, 2.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=60.0,
        )
        assert cam.position == (1.0, 2.0, 3.0)
        assert cam.fov == 60.0

    def test_invalid_fov_too_low(self) -> None:
        with pytest.raises(ValueError, match="fov must be between 1 and 180"):
            CameraConfig(fov=0)

    def test_invalid_fov_too_high(self) -> None:
        with pytest.raises(ValueError, match="fov must be between 1 and 180"):
            CameraConfig(fov=200)

    def test_fov_boundary_values(self) -> None:
        assert CameraConfig(fov=1).fov == 1
        assert CameraConfig(fov=180).fov == 180

    def test_invalid_position_length(self) -> None:
        with pytest.raises(ValueError, match="position must have 3 elements"):
            CameraConfig(position=(1.0, 2.0))  # type: ignore

    def test_to_dict_sparse(self) -> None:
        cam = CameraConfig(position=(1.0, 2.0, 3.0))
        d = cam.to_dict()
        assert d == {"position": [1.0, 2.0, 3.0]}
        assert "target" not in d
        assert "fov" not in d

    def test_to_dict_empty(self) -> None:
        cam = CameraConfig()
        assert cam.to_dict() == {}

    def test_round_trip(self) -> None:
        cam = CameraConfig(
            position=(1.0, 2.0, 3.0),
            target=(0.0, 0.0, 0.0),
            fov=45.0,
        )
        d = cam.to_dict()
        cam2 = CameraConfig.from_dict(d)
        assert cam2.position == (1.0, 2.0, 3.0)
        assert cam2.target == (0.0, 0.0, 0.0)
        assert cam2.fov == 45.0
        assert cam2.up is None

    # -- New fields --

    def test_fov_preset(self) -> None:
        cam = CameraConfig(fov_preset="50mm Normal")
        assert cam.fov_preset == "50mm Normal"
        d = cam.to_dict()
        assert d["fov_preset"] == "50mm Normal"

    def test_fov_preset_names_match_viewer_contract(self) -> None:
        camera_source = viewer_source("src/config/sections/camera/data.ts")

        source = camera_source.read_text(encoding="utf-8")
        presets_match = re.search(
            r"fovPresets:\s*\{(?P<body>.*?)^\s*\},",
            source,
            re.DOTALL | re.MULTILINE,
        )
        assert presets_match is not None, (
            "cannot find cameraConfig.fovPresets in viewer data.ts"
        )
        preset_names = tuple(
            quoted or bare
            for quoted, bare in re.findall(
                r"^\s*(?:['\"]([^'\"]+)['\"]|([A-Za-z_$][\w$]*))\s*:",
                presets_match.group("body"),
                re.MULTILINE,
            )
        )

        assert preset_names == VALID_FOV_PRESETS

    def test_invalid_fov_preset(self) -> None:
        with pytest.raises(ValueError, match="fov_preset must be one of"):
            CameraConfig(fov_preset="wrong")

    def test_near_far(self) -> None:
        cam = CameraConfig(near=0.1, far=1000.0)
        assert cam.near == 0.1
        assert cam.far == 1000.0

    def test_invalid_near(self) -> None:
        with pytest.raises(ValueError, match="near must be > 0"):
            CameraConfig(near=-1.0)

    def test_target_node(self) -> None:
        cam = CameraConfig(target_node="embryo")
        d = cam.to_dict()
        assert d["target_node"] == "embryo"
        cam2 = CameraConfig.from_dict(d)
        assert cam2.target_node == "embryo"

    def test_full_round_trip(self) -> None:
        cam = CameraConfig(
            position=(1.0, 2.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=60.0,
            fov_preset="Custom",
            near=0.1,
            far=500.0,
            target_node="my_node",
        )
        d = cam.to_dict()
        cam2 = CameraConfig.from_dict(d)
        assert cam2.position == (1.0, 2.0, 3.0)
        assert cam2.fov_preset == "Custom"
        assert cam2.near == 0.1
        assert cam2.far == 500.0
        assert cam2.target_node == "my_node"


class TestUIConfig:
    """Tests for UIConfig."""

    def test_default(self) -> None:
        ui = UIConfig()
        assert ui.to_dict() == {}

    def test_round_trip(self) -> None:
        ui = UIConfig(show_help=False, show_rendering_controls=True)
        d = ui.to_dict()
        assert d == {"show_help": False, "show_rendering_controls": True}
        ui2 = UIConfig.from_dict(d)
        assert ui2.show_help is False
        assert ui2.show_rendering_controls is True
        assert ui2.show_performance_monitor is None

    def test_show_overlays_round_trip(self) -> None:
        ui = UIConfig(show_overlays=False)
        d = ui.to_dict()
        assert d == {"show_overlays": False}
        ui2 = UIConfig.from_dict(d)
        assert ui2.show_overlays is False
        assert ui2.show_help is None


class TestDimensionsConfig:
    """Tests for DimensionsConfig."""

    def test_round_trip(self) -> None:
        dims = DimensionsConfig(current_step=[5.0, 0.0, 0.0, 0.0], selected_dimension=0)
        d = dims.to_dict()
        dims2 = DimensionsConfig.from_dict(d)
        assert dims2.current_step == [5.0, 0.0, 0.0, 0.0]
        assert dims2.selected_dimension == 0


class TestAnimationConfig:
    """Tests for AnimationConfig."""

    def test_round_trip(self) -> None:
        anim = AnimationConfig(
            playing=True, target_fps=10.0, loop="bounce", direction="forward"
        )
        d = anim.to_dict()
        anim2 = AnimationConfig.from_dict(d)
        assert anim2.playing is True
        assert anim2.loop == "bounce"

    def test_step_size_round_trips(self) -> None:
        """The viewer writes this field; Python has to be able to carry it.

        Ctrl+Shift+S captures a per-dimension step override into the scene's
        `animation` block. Before this field existed the round trip dropped it
        silently: read a captured scene into Python, write it back, and the
        override was gone with nothing said.
        """
        anim = AnimationConfig(playing=True, step_size=2.5)
        restored = AnimationConfig.from_dict(anim.to_dict())
        assert restored.step_size == 2.5

        # Auto (the usual case) must stay absent rather than serialize as null,
        # so it cannot be mistaken for an explicit override downstream.
        assert "step_size" not in AnimationConfig(playing=True).to_dict()

    def test_invalid_step_size(self) -> None:
        # Matches the viewer's own `setStepSize` guard.
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="step_size must be > 0"):
                AnimationConfig(step_size=bad)

    def test_invalid_target_fps(self) -> None:
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with pytest.raises(ValueError, match="target_fps must be finite and > 0"):
                AnimationConfig(target_fps=bad)

    def test_invalid_loop(self) -> None:
        with pytest.raises(ValueError, match="loop must be one of"):
            AnimationConfig(loop="invalid")

    def test_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="direction must be one of"):
            AnimationConfig(direction="sideways")


class TestViewerConfig:
    """Tests for ViewerConfig validation and serialization."""

    def test_default_config(self) -> None:
        vc = ViewerConfig()
        assert vc.camera is None
        assert vc.background_color is None
        assert vc.bloom_enabled is None

    def test_valid_full_config(self) -> None:
        vc = ViewerConfig(
            camera=CameraConfig(position=(0, 5, 20)),
            background_color="#1a1a2e",
            tone_mapping="ACES",
            exposure=1.0,
            global_offset=0.0,
            global_gamma=1.0,
            bloom_enabled=True,
            bloom_strength=0.5,
            bloom_threshold=0.01,
            bloom_levels=8,
            control_type="orbit",
            auto_rotate=True,
            auto_rotate_speed=0.25,
            cinematic_mode=True,
            vignette_enabled=True,
            vignette_darkness=0.6,
            detector_noise_enabled=True,
            detector_noise_readout_sigma=0.005,
            detector_noise_photon_gain=0.003,
            detector_noise_fpn_sigma=0.001,
            fxaa_enabled=True,
        )
        assert vc.bloom_strength == 0.5
        assert vc.detector_noise_enabled is True
        assert vc.bloom_levels == 8

    # -- Validation tests --

    def test_invalid_background_color(self) -> None:
        with pytest.raises(ValueError, match="Invalid hex color"):
            ViewerConfig(background_color="red")

    def test_invalid_background_color_short(self) -> None:
        with pytest.raises(ValueError, match="Invalid hex color"):
            ViewerConfig(background_color="#fff")

    def test_valid_background_color(self) -> None:
        for hexcol in ("#000000", "#FFFFFF", "#1a2B3c"):
            assert ViewerConfig(background_color=hexcol).background_color.lower() == (
                hexcol.lower()
            )

    def test_invalid_tone_mapping(self) -> None:
        with pytest.raises(ValueError, match="tone_mapping must be one of"):
            ViewerConfig(tone_mapping="HDR10")

    def test_valid_tone_mappings(self) -> None:
        for tm in ("None", "Linear", "Reinhard", "Cineon", "ACES", "AgX", "Neutral"):
            assert ViewerConfig(tone_mapping=tm).tone_mapping == tm

    def test_invalid_control_type(self) -> None:
        with pytest.raises(ValueError, match="control_type must be one of"):
            ViewerConfig(control_type="trackball")

    def test_valid_control_types(self) -> None:
        for ct in ("orbit", "fly", "ortho"):
            assert ViewerConfig(control_type=ct).control_type == ct

    def test_invalid_auto_rotate_axis(self) -> None:
        with pytest.raises(ValueError, match="auto_rotate_axis must be one of"):
            ViewerConfig(auto_rotate_axis="z")

    def test_valid_auto_rotate_axes(self) -> None:
        # The camera-frame three are words and the world three are letters —
        # the viewer's dropdown and this accepted set have to agree or an
        # authored scene silently falls back to the screen-vertical turntable.
        for axis in (
            "vertical",
            "horizontal",
            "view",
            "world-x",
            "world-y",
            "world-z",
        ):
            assert ViewerConfig(auto_rotate_axis=axis).auto_rotate_axis == axis

    def test_world_axis_letters_are_not_bare(self) -> None:
        # A bare letter is rejected: it would be ambiguous between a world axis
        # and a data axis, which is why the world tokens carry the prefix.
        for axis in ("x", "y", "z", "world_y", "World-Y"):
            with pytest.raises(ValueError, match="auto_rotate_axis must be one of"):
                ViewerConfig(auto_rotate_axis=axis)

    def test_auto_rotate_axis_round_trips_through_dict(self) -> None:
        vc = ViewerConfig(auto_rotate=True, auto_rotate_axis="view")
        restored = ViewerConfig.from_dict(vc.to_dict())
        assert restored.auto_rotate_axis == "view"

    def test_auto_rotate_axis_absent_when_unset(self) -> None:
        # Sparse serialization: an unset axis must not write a key, so every
        # existing scene keeps its exact (screen-vertical) behavior.
        assert "auto_rotate_axis" not in ViewerConfig(auto_rotate=True).to_dict()

    def test_invalid_exposure_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="exposure"):
            ViewerConfig(exposure=-11.0)
        with pytest.raises(ValueError, match="exposure"):
            ViewerConfig(exposure=11.0)

    def test_exposure_zero(self) -> None:
        assert ViewerConfig(exposure=0.0).exposure == 0.0

    def test_exposure_range_bounds(self) -> None:
        assert ViewerConfig(exposure=10.0).exposure == 10.0
        assert ViewerConfig(exposure=-10.0).exposure == -10.0

    def test_invalid_global_gamma(self) -> None:
        with pytest.raises(ValueError, match="global_gamma"):
            ViewerConfig(global_gamma=0.05)

    def test_invalid_bloom_threshold(self) -> None:
        with pytest.raises(ValueError, match="bloom_threshold must be between 0 and 1"):
            ViewerConfig(bloom_threshold=1.5)

    def test_invalid_vignette_darkness(self) -> None:
        with pytest.raises(
            ValueError, match="vignette_darkness must be between 0 and 1"
        ):
            ViewerConfig(vignette_darkness=-0.1)

    def test_invalid_detector_noise_readout_sigma(self) -> None:
        with pytest.raises(ValueError, match="detector_noise_readout_sigma"):
            ViewerConfig(detector_noise_readout_sigma=0.5)

    def test_invalid_detector_noise_photon_gain_too_low(self) -> None:
        with pytest.raises(ValueError, match="detector_noise_photon_gain"):
            ViewerConfig(detector_noise_photon_gain=0.00001)

    def test_invalid_theme(self) -> None:
        with pytest.raises(ValueError, match="theme must be one of"):
            ViewerConfig(theme="neon")

    def test_valid_themes(self) -> None:
        for t in ("dark", "light", "frosted-glass", "liquid-glass"):
            assert ViewerConfig(theme=t).theme == t

    # -- New fields --

    def test_new_aa_fields(self) -> None:
        vc = ViewerConfig(
            msaa_enabled=True,
            msaa_samples=4,
            ssaa_enabled=False,
            ssaa_multiplier=2.0,
        )
        assert vc.msaa_samples == 4

    def test_chromatic_lens_fields(self) -> None:
        vc = ViewerConfig(
            chromatic_lens_distortion_enabled=True,
            chromatic_lens_distortion_x=0.1,
            chromatic_lens_dispersion=0.5,
        )
        assert vc.chromatic_lens_distortion_enabled is True

    def test_fly_controls_fields(self) -> None:
        vc = ViewerConfig(
            fly_movement_speed=2.0,
            fly_rotation_speed=0.5,
            fly_inertial_mode=True,
        )
        assert vc.fly_movement_speed == 2.0

    def test_clipping_fields(self) -> None:
        vc = ViewerConfig(dynamic_clipping_enabled=True)
        assert vc.dynamic_clipping_enabled is True

    def test_adaptive_dpr(self) -> None:
        vc = ViewerConfig(adaptive_dpr_enabled=True)
        assert vc.adaptive_dpr_enabled is True

    def test_ui_config(self) -> None:
        vc = ViewerConfig(ui=UIConfig(show_help=False, show_dimensions=True))
        d = vc.to_dict()
        assert d["ui"] == {"show_help": False, "show_dimensions": True}

    def test_dimensions_config(self) -> None:
        vc = ViewerConfig(
            dimensions=DimensionsConfig(current_step=[5, 0, 0, 0], selected_dimension=0)
        )
        d = vc.to_dict()
        assert d["dimensions"]["current_step"] == [5, 0, 0, 0]

    def test_animation_config(self) -> None:
        vc = ViewerConfig(
            animation=[
                AnimationConfig(playing=True, target_fps=10),
                AnimationConfig(),
            ]
        )
        d = vc.to_dict()
        assert len(d["animation"]) == 2
        assert d["animation"][0]["playing"] is True

    # -- Serialization tests --

    def test_to_dict_sparse(self) -> None:
        vc = ViewerConfig(bloom_enabled=True, exposure=1.0)
        d = vc.to_dict()
        assert d == {"bloom_enabled": True, "exposure": 1.0}
        assert "camera" not in d
        assert "background_color" not in d

    def test_to_dict_empty(self) -> None:
        vc = ViewerConfig()
        assert vc.to_dict() == {}

    def test_to_dict_with_camera(self) -> None:
        vc = ViewerConfig(
            camera=CameraConfig(position=(1, 2, 3)),
            bloom_enabled=True,
        )
        d = vc.to_dict()
        assert d["camera"] == {"position": [1, 2, 3]}
        assert d["bloom_enabled"] is True

    def test_to_dict_empty_camera_omitted(self) -> None:
        """Camera with no fields set should not appear in dict."""
        vc = ViewerConfig(camera=CameraConfig())
        d = vc.to_dict()
        assert "camera" not in d

    def test_round_trip(self) -> None:
        vc = ViewerConfig(
            camera=CameraConfig(position=(1, 2, 3), fov=60),
            background_color="#112233",
            bloom_strength=0.3,
            tone_mapping="ACES",
            detector_noise_enabled=True,
            detector_noise_readout_sigma=0.005,
            cinematic_mode=True,
        )
        d = vc.to_dict()
        vc2 = ViewerConfig.from_dict(d)
        assert vc2.camera is not None
        assert vc2.camera.position == (1, 2, 3)
        assert vc2.camera.fov == 60
        assert vc2.background_color == "#112233"
        assert vc2.bloom_strength == 0.3
        assert vc2.tone_mapping == "ACES"
        assert vc2.detector_noise_enabled is True
        assert vc2.detector_noise_readout_sigma == 0.005
        assert vc2.cinematic_mode is True
        # Unset fields remain None
        assert vc2.bloom_enabled is None
        assert vc2.fxaa_enabled is None

    def test_round_trip_all_new_fields(self) -> None:
        """Round-trip with all new fields."""
        vc = ViewerConfig(
            camera=CameraConfig(
                position=(1, 2, 3),
                target_node="my_node",
                fov_preset="35mm",
                near=0.1,
                far=500,
            ),
            bloom_levels=8,
            vignette_offset=0.3,
            msaa_enabled=True,
            msaa_samples=4,
            ssaa_enabled=False,
            ssaa_multiplier=2.0,
            chromatic_lens_distortion_enabled=True,
            chromatic_lens_distortion_x=0.1,
            fly_movement_speed=2.0,
            fly_inertial_mode=True,
            dynamic_clipping_enabled=True,
            adaptive_dpr_enabled=True,
            theme="dark",
            ui=UIConfig(show_help=False),
            dimensions=DimensionsConfig(current_step=[5, 0, 0]),
            animation=[AnimationConfig(playing=True, target_fps=10)],
        )
        d = vc.to_dict()
        vc2 = ViewerConfig.from_dict(d)
        assert vc2.camera is not None
        assert vc2.camera.target_node == "my_node"
        assert vc2.bloom_levels == 8
        assert vc2.msaa_samples == 4
        assert vc2.chromatic_lens_distortion_enabled is True
        assert vc2.fly_movement_speed == 2.0
        assert vc2.dynamic_clipping_enabled is True
        assert vc2.adaptive_dpr_enabled is True
        assert vc2.theme == "dark"
        assert vc2.ui is not None
        assert vc2.ui.show_help is False
        assert vc2.dimensions is not None
        assert vc2.dimensions.current_step == [5, 0, 0]
        assert vc2.animation is not None
        assert vc2.animation[0].playing is True

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Forward compatibility — unknown keys are silently ignored."""
        d = {"bloom_enabled": True, "future_feature": 42}
        vc = ViewerConfig.from_dict(d)
        assert vc.bloom_enabled is True

    def test_from_dict_empty(self) -> None:
        vc = ViewerConfig.from_dict({})
        assert vc.camera is None
        assert vc.bloom_enabled is None

    # -- JSON I/O tests --

    def test_to_json_and_from_json(self) -> None:
        vc = ViewerConfig(
            camera=CameraConfig(position=(1, 2, 3)),
            bloom_enabled=True,
            theme="dark",
        )
        json_str = vc.to_json()
        parsed = json.loads(json_str)
        assert parsed["bloom_enabled"] is True
        assert parsed["theme"] == "dark"

        vc2 = ViewerConfig.from_json(json_str)
        assert vc2.bloom_enabled is True
        assert vc2.theme == "dark"

    def test_to_file_and_from_file(self, tmp_path: Path) -> None:
        vc = ViewerConfig(
            camera=CameraConfig(position=(0, 5, 20), fov=47),
            background_color="#000000",
            bloom_strength=0.5,
        )
        path = tmp_path / "viewer_state.json"
        vc.to_file(path)

        vc2 = ViewerConfig.from_file(path)
        assert vc2.camera is not None
        assert vc2.camera.position == (0, 5, 20)
        assert vc2.camera.fov == 47
        assert vc2.background_color == "#000000"
        assert vc2.bloom_strength == 0.5

    def test_snapshot_round_trip(self) -> None:
        """Simulate viewer export → Python load → tweak → re-export."""
        # This is what a viewer Ctrl+Shift+S export would look like
        snapshot_json = json.dumps(
            {
                "camera": {
                    "position": [0.5, 5.2, 19.8],
                    "target": [0, 0, 0],
                    "up": [0, 1, 0],
                    "fov": 47,
                    "fov_preset": "50mm Normal",
                    "near": 0.1,
                    "far": 1000,
                },
                "background_color": "#000000",
                "bloom_enabled": True,
                "bloom_strength": 0.5,
                "bloom_radius": 1.0,
                "bloom_threshold": 0.01,
                "bloom_levels": 8,
                "exposure": 1.0,
                "global_offset": 0.0,
                "global_gamma": 1.0,
                "tone_mapping": "ACES",
                "control_type": "orbit",
                "auto_rotate": True,
                "theme": "dark",
                "dynamic_clipping_enabled": True,
                "adaptive_dpr_enabled": True,
                "ui": {
                    "show_help": False,
                    "show_rendering_controls": False,
                    "show_performance_monitor": False,
                    "show_dimensions": True,
                },
                "dimensions": {"current_step": [5, 0, 0, 0], "selected_dimension": 0},
            }
        )

        # Load as ViewerConfig
        vc = ViewerConfig.from_json(snapshot_json)
        assert vc.camera is not None
        assert vc.camera.position == (0.5, 5.2, 19.8)
        assert vc.bloom_strength == 0.5
        assert vc.theme == "dark"
        assert vc.ui is not None
        assert vc.ui.show_help is False
        assert vc.dimensions is not None
        assert vc.dimensions.current_step == [5, 0, 0, 0]

        # Tweak
        vc.bloom_strength = 0.8
        vc.camera.position = (10, 5, 30)

        # Re-export
        d = vc.to_dict()
        assert d["bloom_strength"] == 0.8
        assert d["camera"]["position"] == [10, 5, 30]
        assert d["camera"]["fov"] == 47


# ───────────────────────── scene title (browser tab) ──────────────────────────
class TestSceneTitle:
    """`title` names the browser tab; sparse like every other field."""

    def test_title_round_trips(self) -> None:
        vc = ViewerConfig(title="Rivers of Earth")
        d = vc.to_dict()
        assert d["title"] == "Rivers of Earth"
        assert ViewerConfig.from_dict(d).title == "Rivers of Earth"

    def test_unset_title_is_omitted(self) -> None:
        assert "title" not in ViewerConfig().to_dict()

    def test_blank_or_non_string_title_rejected(self) -> None:
        with pytest.raises(ValueError, match="title"):
            ViewerConfig(title="   ")
        with pytest.raises(ValueError, match="title"):
            ViewerConfig(title=123)  # type: ignore[arg-type]
