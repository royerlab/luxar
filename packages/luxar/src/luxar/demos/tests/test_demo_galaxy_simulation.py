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
def test_validate_frame_count_accepts_even_grids(n_frames: int) -> None:
    """Even counts are fine now: a DISCRETE T only ever lands on frames.

    The old rule ("must be odd") existed solely so the continuous slider's
    opening midpoint fell on a frame; it never fixed a dragged position.
    """
    assert _demo.validate_frame_count(n_frames) == n_frames


def test_main_validates_frames_before_building(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(_DEMO_PATH), "--frames=2", "--no-serve"])

    with pytest.raises(ValueError, match="at least 3"):
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


# =============================================================================
# The T axis is the thing that broke: a continuous, spatial T gave the viewer a
# 1000-position slider whose off-frame stops matched no points at all, so every
# time-resolved layer emptied on the first mouse drag. These pin the contract
# that makes that impossible.
# =============================================================================


@pytest.mark.parametrize("n_frames", [3, 24, 25, 101])
def test_time_axis_is_discrete_so_every_slider_stop_is_a_frame(n_frames: int) -> None:
    dim = _demo.time_dimension(n_frames)

    assert dim.discrete is True
    # Discrete + non-displayed implies non-spatial; `spatial=True` here is what
    # made the axis continuous and the slider free-running.
    assert dim.spatial is False
    assert dim.display is False
    assert dim.step == pytest.approx(_demo.T_SPAN_MYR / (n_frames - 1))


@pytest.mark.parametrize("n_frames", [3, 24, 25, 101])
def test_frame_times_sit_exactly_on_the_snap_grid(n_frames: int) -> None:
    """`k * step` in binary, not merely close to it.

    The viewer snaps to `round(v / step) * step` and only fetches within a
    quarter step of the snapped value, so an off-grid plane is unreachable.
    """
    dim = _demo.time_dimension(n_frames)
    times = _demo.frame_times(n_frames)

    assert len(times) == n_frames
    for k, t in enumerate(times):
        assert t == k * dim.step  # exact, abs=0.0


@pytest.mark.parametrize("n_frames", [3, 24, 25, 101])
def test_time_range_ends_on_the_last_frame(n_frames: int) -> None:
    """So the final stop is reachable, whatever `(n - 1) * step` rounds to."""
    dim = _demo.time_dimension(n_frames)

    assert dim.range == (0.0, float(_demo.frame_times(n_frames)[-1]))


def test_default_frame_count_keeps_a_round_binary_step() -> None:
    """241 frames over 480 Myr is exactly 2 Myr — no float drift to reason about.

    The default count is chosen so the step is a round binary value: an axis
    built from a step like 480/23 puts the LAST stop a hair above `max` in the
    browser's own arithmetic, and it becomes unreachable.
    """
    assert _demo.time_dimension(_demo.T_FRAMES).step == 2.0
    assert _demo.frame_times(_demo.T_FRAMES)[-1] == _demo.T_SPAN_MYR


# =============================================================================
# Density scaling: `--stars` must change the WEIGHT of the scene, not its look.
# =============================================================================


def test_density_scale_is_unity_at_the_reference_count() -> None:
    assert _demo.density_scale(_demo.GAIN_REFERENCE_STARS) == pytest.approx(1.0)


@pytest.mark.parametrize("factor", [2, 4, 8])
def test_density_scale_follows_the_cube_root_of_the_thinning(factor: int) -> None:
    """Spacing grows as f^(1/3), so radii and gain do too."""
    n = _demo.GAIN_REFERENCE_STARS // factor

    assert _demo.density_scale(n) == pytest.approx(factor ** (1 / 3))


@pytest.mark.parametrize("factor", [1, 4, 16])
def test_summed_additive_light_is_held_across_star_counts(factor: int) -> None:
    """`N * gain * radius^2` is the additive-sum invariant; keep it flat.

    Without this a 4x lighter build is a 4x dimmer galaxy, which is not what
    "fewer points" is supposed to mean.
    """
    n = _demo.GAIN_REFERENCE_STARS // factor
    scale = _demo.density_scale(n)
    light = n * (_demo.STAR_GAIN * scale) * scale**2
    reference = _demo.GAIN_REFERENCE_STARS * _demo.STAR_GAIN

    assert light == pytest.approx(reference, rel=1e-9)


@pytest.mark.parametrize("factor", [1, 4, 16])
def test_globular_light_is_held_across_star_counts(factor: int) -> None:
    """The static cluster population must thin with the dynamic populations."""
    n = _demo.GAIN_REFERENCE_STARS // factor
    scale = _demo.density_scale(n)
    per_cluster = _demo.globular_stars_per_cluster(n)
    light = per_cluster * (_demo.STAR_GAIN * scale) * scale**2
    reference = _demo.GLOBULAR_STARS_PER_CLUSTER * _demo.STAR_GAIN

    # Integer points per cluster quantise the requested density. The residual
    # may be at most half a point's contribution at the requested scale.
    tolerance = 0.5 * (_demo.STAR_GAIN * scale) * scale**2
    assert abs(light - reference) <= tolerance + 1e-12


def test_density_scale_survives_a_degenerate_star_count() -> None:
    """A zero would divide by zero before the count validator ever sees it."""
    assert _demo.density_scale(0) > 0.0
