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


@pytest.mark.parametrize("n_frames", [4, 24, 100, 960])
def test_validate_frame_count_accepts_even_grids(n_frames: int) -> None:
    """Even counts are fine now: a DISCRETE T only ever lands on frames.

    The old rule ("must be odd") existed solely so the continuous slider's
    opening midpoint fell on a frame; it never fixed a dragged position.
    """
    assert _demo.validate_frame_count(n_frames) == n_frames


@pytest.mark.parametrize("n_frames", [961, 1_000])
def test_validate_frame_count_rejects_overlapping_discrete_frames(
    n_frames: int,
) -> None:
    with pytest.raises(ValueError, match="at most 960.*discrete tolerance"):
        _demo.validate_frame_count(n_frames)


def test_max_frame_count_tracks_span_and_discrete_membership_tolerance() -> None:
    assert _demo.MAX_FRAME_COUNT == math.ceil(
        _demo.T_SPAN_MYR / _demo.DISCRETE_MEMBERSHIP_TOLERANCE
    )
    assert _demo.time_step(_demo.MAX_FRAME_COUNT) > (
        _demo.DISCRETE_MEMBERSHIP_TOLERANCE
    )
    assert _demo.time_step(_demo.MAX_FRAME_COUNT + 1) <= (
        _demo.DISCRETE_MEMBERSHIP_TOLERANCE
    )


@pytest.mark.parametrize("n_stars", [2, 100_000])
def test_validate_star_count_accepts_buildable_counts(n_stars: int) -> None:
    assert _demo.validate_star_count(n_stars) == n_stars


@pytest.mark.parametrize("n_stars", [-1, 0, 1])
def test_validate_star_count_rejects_counts_without_hii_regions(n_stars: int) -> None:
    with pytest.raises(ValueError, match="at least 2.*below the 3rd age percentile"):
        _demo.validate_star_count(n_stars)


def test_main_validates_frames_before_building(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(_DEMO_PATH), "--frames=2", "--no-serve"])

    with pytest.raises(ValueError, match="at least 3"):
        _demo.main()


def test_main_validates_stars_before_building(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(_DEMO_PATH), "--stars=1", "--no-serve"])

    with pytest.raises(ValueError, match="at least 2.*below the 3rd age percentile"):
        _demo.main()


def test_scene_description_preserves_fractional_frame_cadence() -> None:
    description = _demo.scene_description(
        {"ILR": 1.7, "corotation": 11.4, "OLR": 19.4}, n_frames=24
    )

    assert "24 frames of 20.9 Myr" in description


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


@pytest.mark.parametrize("n_frames", [3, 23, 24, 25, 101])
def test_frame_times_sit_exactly_on_the_snap_grid(n_frames: int) -> None:
    """`k * step` in binary, not merely close to it.

    23 is a mutation-sensitive case: `linspace` ends at 480.0 while the snap
    grid's final `k * step` is 479.99999999999994.
    """
    dim = _demo.time_dimension(n_frames)
    times = _demo.frame_times(n_frames)

    assert len(times) == n_frames
    for k, t in enumerate(times):
        assert t == k * dim.step  # exact, abs=0.0


@pytest.mark.parametrize("n_frames", [3, 23, 24, 25, 101])
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


# =============================================================================
# The played-layer ladder.
#
# `played_layer_ladder` is the highest-risk arithmetic in the #2657 ladder pass:
# T is PLAYED, so rung 0 has to clear a PER-FRAME floor that `stream_ladder`
# knows nothing about, and the floor then has to be clamped so it cannot author a
# ladder the gate rejects. Every number here is measured, either off the built
# store or off the flag combinations named in LADDER_GATE_AUDITED_ROWS.
# =============================================================================


def _increments(counts: list[int]) -> list[int]:
    """Per-rung commit sizes from cumulative cuts."""
    return [
        cut - previous for previous, cut in zip([0, *counts[:-1]], counts, strict=True)
    ]


def test_the_thin_age_bin_gets_the_per_frame_floor() -> None:
    """`Disc 50-300 Myr` at the authored build: 966 stars x 241 frames.

    The one node in the corpus where the floor BINDS. Without it rung 0 is the
    unsliced 39,062-element download budget, whose sparsest frame is a measured
    144 rows — under the gate's 250-row absolute first-paint floor.
    """
    rows, stops = 232_806, 241

    counts = _demo.played_layer_ladder(rows, stops)["counts"]

    assert counts[0] == stops * _demo.FIRST_RUNG_ROWS_PER_FRAME == 72_300
    assert counts == [72_300, 144_600, rows]
    assert max(_increments(counts)) / rows < 0.379 + 1e-3


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (3_001_655, [375_207, 750_414, 1_500_828, 3_001_655]),  # Disc 0.3-2 Gyr
        (10_741_129, [1_342_642, 2_685_284, 5_370_568, 10_741_129]),  # Disc 2-6 Gyr
        (10_097_900, [1_262_238, 2_524_476, 5_048_952, 10_097_900]),  # Disc > 6 Gyr
        (723_000, [90_375, 180_750, 361_500, 723_000]),  # HII regions
        (4_820_000, [602_500, 1_205_000, 2_410_000, 4_820_000]),  # Bulge
    ],
)
def test_a_fat_played_layer_keeps_the_policy_share_rung(
    rows: int, expected: list[int]
) -> None:
    """The three fat age bins, the HII regions and the bulge.

    The floor is a ``max()``, so a layer whose ``n/8`` share rung already exceeds
    it keeps the policy's own ladder byte-for-byte — which is why passing the
    floor on every played layer costs the shipped store nothing. Asserted as the
    whole cut LIST, read off the built store, and not just rung 0: the commit
    ceiling also has to stay scaled by the slice count, or the same rung 0
    resolves into six rungs instead of four.
    """
    counts = _demo.played_layer_ladder(rows, 241)["counts"]

    assert counts == expected
    assert counts[0] == -(-rows // 8) > 241 * _demo.FIRST_RUNG_ROWS_PER_FRAME


def test_the_floor_is_clamped_to_the_gates_degeneracy_bound() -> None:
    """``--stars 40000 --frames 600``: 405 stars in ``50-300 Myr``.

    243,000 rows against an unclamped 180,000-row floor is a rung 0 holding 74%
    of the node, past the gate's 0.6 bound on any single level.
    """
    rows, stops = 243_000, 600

    counts = _demo.played_layer_ladder(rows, stops)["counts"]

    assert counts[0] < stops * _demo.FIRST_RUNG_ROWS_PER_FRAME
    assert max(_increments(counts)) / rows <= _demo.LADDER_GATE_MAX_RUNG_SHARE


def test_the_clamp_never_leaves_an_audited_layer_unladdered() -> None:
    """``--stars 250000 --frames 900``: 242 stars in ``< 50 Myr``.

    217,800 rows against an unclamped 270,000-row floor resolves to a single cut
    — a FLAT leaf above the gate's 200,000-row threshold, which is the very
    thing the ladder pass exists to remove.
    """
    counts = _demo.played_layer_ladder(217_800, 900)["counts"]

    assert len(counts) > 1
    assert max(_increments(counts)) / 217_800 <= _demo.LADDER_GATE_MAX_RUNG_SHARE


@pytest.mark.parametrize("rows", [26_510, 60_000])
def test_a_layer_below_the_gate_threshold_keeps_its_flat_leaf(rows: int) -> None:
    """26,510 rows is `Disc < 50 Myr` at the authored build.

    The clamp must NOT reach down here. Below 200,000 rows the gate audits
    neither ladder arm, so a flat leaf passes — while a ladder whose rung 0
    cannot put 250 rows in a frame turns a SKIPPED node into an audited, failing
    one (measured: 28 rows at the sparsest frame of a 26,510-row layer laddered
    at a third of the node).
    """
    assert _demo.played_layer_ladder(rows, 241)["counts"] == [rows]


def test_the_floored_rung_fills_every_frame_in_a_built_store(tmp_path: Path) -> None:
    """The end-to-end arm: measure the sparsest frame off the written rung.

    Built at the shipped `Disc 50-300 Myr` shape rather than through
    `generate_galaxy`, because the floor only engages above 39,062 rows and the
    thin bin is ~1% of the disc — so a real build that reaches it costs ~5M
    points, while this reproduces the shipped store's rung-0 histogram exactly
    (p05 = 275, min = 264) in under a second.

    This is the assertion the gate makes: rung 0's SPARSEST frames, not its
    busiest, decide whether playback renders anything.
    """
    from collections import Counter

    import numpy as np
    import zarr

    from luxar import Dimension, Dimensions, LuxarZarrCompiler
    from luxar.encoding.decoder import decode_coordinate_columns

    stars, frames = 966, 241
    rng = np.random.default_rng(0)
    xyz = rng.normal(scale=5.0, size=(stars, 3)).astype(np.float32)
    positions = np.empty((stars * frames, 4), dtype=np.float32)
    for frame, t_myr in enumerate(_demo.frame_times(frames)):
        rows = slice(frame * stars, (frame + 1) * stars)
        positions[rows, :3] = xyz
        positions[rows, 3] = t_myr

    dimensions = Dimensions(
        [
            Dimension("X", unit="kpc", range=(-36, 36), display=True),
            Dimension("Y", unit="kpc", range=(-36, 36), display=True),
            Dimension("Z", unit="kpc", range=(-36, 36), display=True),
            _demo.time_dimension(frames),
        ]
    )
    output = tmp_path / "played_layer.luxar.zarr"
    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)
        scene.add_points(
            "Disc probe",
            positions=positions,
            additive_lod=_demo.played_layer_ladder(len(positions), frames),
        )

    root = zarr.open(str(output), mode="r")
    rung = root["Disc probe"]["additive_0"]
    histogram = Counter(
        tuple(row) for row in decode_coordinate_columns(rung["positions"], [3], root)
    )
    per_frame = sorted(histogram.values())

    assert int(dict(rung.attrs)["n_points"]) == 72_300
    assert len(per_frame) == frames, "rung 0 must reach every frame"
    # The gate's reduction: the lower 5th-percentile ORDER STATISTIC.
    assert per_frame[int(0.05 * (frames - 1))] >= 250
