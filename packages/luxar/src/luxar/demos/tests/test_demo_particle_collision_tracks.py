from collections.abc import Callable

import numpy as np
import pytest

from luxar.demos.demo_particle_collision import Particle, generate_helix_track
from luxar.demos.demo_particle_collision_animated import (
    generate_helix_track_with_times,
)

TrackGenerator = Callable[[Particle, np.random.Generator, int], tuple[np.ndarray, ...]]


def _particle(*, charge: int, pt: float, pz: float = 0.0) -> Particle:
    particle_type = "muon_plus" if charge > 0 else "muon_minus"
    return Particle(
        particle_type=particle_type,
        energy=float(np.hypot(pt, pz)),
        px=pt,
        py=0.0,
        pz=pz,
        origin=np.zeros(3),
    )


def _track_points(
    generator: TrackGenerator, particle: Particle, n_points: int = 8
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
def test_charged_track_starts_along_momentum(
    generator: TrackGenerator, charge: int
) -> None:
    points = _track_points(generator, _particle(charge=charge, pt=3.0))

    first_step = points[1] - points[0]

    assert first_step[0] / np.linalg.norm(first_step) > 0.999


@pytest.mark.parametrize(
    "generator", [generate_helix_track, generate_helix_track_with_times]
)
@pytest.mark.parametrize("charge", [-1, 1])
def test_charge_controls_transverse_turn_sense(
    generator: TrackGenerator, charge: int
) -> None:
    points = _track_points(generator, _particle(charge=charge, pt=3.0))
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
    points = _track_points(generator, _particle(charge=1, pt=pt, pz=pz))
    first_step = points[1] - points[0]

    measured_ratio = first_step[2] / np.linalg.norm(first_step[:2])

    assert measured_ratio == pytest.approx(pz / pt, rel=1e-3)


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
