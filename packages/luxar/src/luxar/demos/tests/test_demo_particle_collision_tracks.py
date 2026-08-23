from collections.abc import Callable

import numpy as np
import pytest

from luxar.demos._particle_collision_tracks import (
    MAX_AZIMUTH_SAMPLE_STEP,
    generate_helix_points,
)
from luxar.demos.demo_particle_collision import (
    B_FIELD,
    PARTICLE_LEGEND_HTML,
    PARTICLE_TYPES,
    Particle,
    generate_helix_track,
)
from luxar.demos.demo_particle_collision_animated import (
    generate_helix_track_with_times,
)

TrackGenerator = Callable[[Particle, np.random.Generator, int], tuple[np.ndarray, ...]]


def _particle(*, charge: int, px: float, py: float = 0.0, pz: float = 0.0) -> Particle:
    particle_type = "muon_plus" if charge > 0 else "muon_minus"
    return Particle(
        particle_type=particle_type,
        energy=float(np.linalg.norm([px, py, pz])),
        px=px,
        py=py,
        pz=pz,
        origin=np.zeros(3),
    )


def _track_points(
    generator: TrackGenerator, particle: Particle, n_points: int = 100
) -> np.ndarray:
    result = generator(particle, np.random.default_rng(0), n_points)
    vertices = result[0]
    if generator is generate_helix_track:
        return vertices
    return np.concatenate((vertices[0::2], vertices[-1:]), axis=0)


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
@pytest.mark.parametrize("charge", [-1, 1])
@pytest.mark.parametrize(
    "momentum", [(3.0, 0.0, 0.0), (3.0, 4.0, -2.0), (0.0, 0.0, 3.0)]
)
def test_charged_track_starts_along_momentum(
    generator: TrackGenerator, charge: int, momentum: tuple[float, float, float]
) -> None:
    particle = _particle(charge=charge, px=momentum[0], py=momentum[1], pz=momentum[2])
    points = _track_points(generator, particle)

    first_step = points[1] - points[0]
    measured_direction = first_step / np.linalg.norm(first_step)
    momentum_direction = np.array(momentum) / np.linalg.norm(momentum)

    assert np.dot(measured_direction, momentum_direction) > 0.999


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
@pytest.mark.parametrize("charge", [-1, 1])
def test_charge_controls_transverse_turn_sense(
    generator: TrackGenerator, charge: int
) -> None:
    points = _track_points(generator, _particle(charge=charge, px=3.0))
    first_step = points[1, :2] - points[0, :2]
    second_step = points[2, :2] - points[1, :2]

    signed_turn = first_step[0] * second_step[1] - first_step[1] * second_step[0]

    assert np.sign(signed_turn) == -charge


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
@pytest.mark.parametrize("pt", [1.0, 3.0, 10.0])
def test_helix_pitch_follows_longitudinal_to_transverse_momentum_ratio(
    generator: TrackGenerator, pt: float
) -> None:
    pz = 5.0
    points = _track_points(generator, _particle(charge=1, px=pt, pz=pz))
    first_step = points[1] - points[0]

    measured_ratio = first_step[2] / np.linalg.norm(first_step[:2])

    assert measured_ratio == pytest.approx(pz / pt, rel=1e-3)


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
@pytest.mark.parametrize("pt", [0.05, 3.0, 10.0])
def test_rendered_curvature_keeps_the_documented_radius_scale(
    generator: TrackGenerator, pt: float
) -> None:
    points = _track_points(generator, _particle(charge=1, px=pt))
    first_step = points[1, :2] - points[0, :2]
    second_step = points[2, :2] - points[1, :2]
    turn_angle = np.arctan2(
        first_step[0] * second_step[1] - first_step[1] * second_step[0],
        np.dot(first_step, second_step),
    )
    chord_length = np.linalg.norm(first_step)
    measured_radius = chord_length / (2 * np.sin(abs(turn_angle) / 2))

    physical_radius = pt / B_FIELD
    assert measured_radius / physical_radius == pytest.approx(2.0, rel=1e-3)


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
def test_neutral_track_remains_straight(generator: TrackGenerator) -> None:
    particle = Particle(
        particle_type="photon",
        energy=5.0,
        px=3.0,
        py=4.0,
        pz=0.0,
        origin=np.zeros(3),
    )
    points = _track_points(generator, particle)
    steps = np.diff(points, axis=0)

    assert np.linalg.matrix_rank(steps) == 1
    assert steps[0] / np.linalg.norm(steps[0]) == pytest.approx([0.6, 0.8, 0.0])


def _sample_points(
    *,
    momentum: np.ndarray | None = None,
    max_radius: float = 15.0,
    n_points: int = 200,
    shower_radius: float | None = None,
) -> np.ndarray:
    return generate_helix_points(
        origin=np.zeros(3),
        momentum=np.array([3.0, 0.0, 0.0]) if momentum is None else momentum,
        charge=1,
        magnetic_field=B_FIELD,
        max_radius=max_radius,
        max_z=25.0,
        n_points=n_points,
        transverse_step=0.25,
        shower_radius=shower_radius,
    )


def test_track_stops_at_maximum_detector_radius() -> None:
    points = _sample_points(max_radius=0.5)

    assert len(points) < 200
    assert np.all(np.linalg.norm(points[:, :2], axis=1) <= 0.5)


def test_track_stops_after_first_shower_radius_sample() -> None:
    points = _sample_points(shower_radius=0.5)
    radii = np.linalg.norm(points[:, :2], axis=1)

    assert len(points) < 200
    assert radii[-2] <= 0.5 < radii[-1]


def test_soft_track_azimuth_sampling_stays_below_cap() -> None:
    points = _sample_points(
        momentum=np.array([0.05, 0.0, 0.0]), max_radius=1.0, n_points=5000
    )
    steps = np.diff(points[:, :2], axis=0)
    step_azimuths = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))

    azimuth_steps = np.abs(np.diff(step_azimuths))
    assert np.median(azimuth_steps) == pytest.approx(MAX_AZIMUTH_SAMPLE_STEP, abs=5e-6)
    assert np.max(azimuth_steps) <= MAX_AZIMUTH_SAMPLE_STEP + 3e-5


@pytest.mark.parametrize("pz, expected_z", [(-3.0, -25.0), (3.0, 25.0)])
def test_pure_longitudinal_track_reaches_detector_endcap(
    pz: float, expected_z: float
) -> None:
    points = _sample_points(momentum=np.array([0.0, 0.0, pz]), n_points=11)

    assert points[:, :2] == pytest.approx(np.zeros((11, 2)))
    assert points[-1, 2] == pytest.approx(expected_z)


@pytest.mark.parametrize("transverse_momentum", [0.12, 0.3, 0.6, 1.0, 1.5, 2.0])
def test_static_and_animated_tracks_cover_comparable_transverse_arc(
    transverse_momentum: float,
) -> None:
    particle = _particle(charge=1, px=transverse_momentum)
    static_points = _track_points(generate_helix_track, particle, n_points=100)
    animated_points = _track_points(
        generate_helix_track_with_times, particle, n_points=1000
    )

    static_arc = np.linalg.norm(np.diff(static_points[:, :2], axis=0), axis=1).sum()
    animated_arc = np.linalg.norm(np.diff(animated_points[:, :2], axis=0), axis=1).sum()

    assert static_arc == pytest.approx(animated_arc, rel=0.04)


@pytest.mark.parametrize(
    "particle_type",
    [
        "electron",
        "positron",
        "muon_minus",
        "muon_plus",
        "pion_plus",
        "pion_minus",
        "kaon",
        "proton",
        "photon",
    ],
)
def test_particle_legend_uses_rendered_track_colors(particle_type: str) -> None:
    red, green, blue = PARTICLE_TYPES[particle_type]["color"]
    css_color = f"rgb({round(red * 255)} {round(green * 255)} {round(blue * 255)})"

    assert f"color:{css_color}" in PARTICLE_LEGEND_HTML


def test_particle_legend_names_only_generated_charges() -> None:
    assert "π±/K⁺" in PARTICLE_LEGEND_HTML
    assert "p (protons)" in PARTICLE_LEGEND_HTML
    assert "K±" not in PARTICLE_LEGEND_HTML
    assert "p/p̄" not in PARTICLE_LEGEND_HTML


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
def test_empty_charged_track_arrays_remain_float32(generator: TrackGenerator) -> None:
    result = generator(
        _particle(charge=1, px=0.0, py=0.0, pz=0.0),
        np.random.default_rng(0),
        8,
    )

    assert all(array.dtype == np.float32 for array in result)
