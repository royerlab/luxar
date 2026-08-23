from __future__ import annotations

import numpy as np

HELIX_RADIUS_VISUAL_SCALE = 2.0
MAX_AZIMUTH_SAMPLE_STEP = 0.05


def generate_helix_points(
    *,
    origin: np.ndarray,
    momentum: np.ndarray,
    charge: int,
    magnetic_field: float,
    max_radius: float,
    max_z: float,
    n_points: int,
    transverse_step: float,
    shower_radius: float | None = None,
) -> np.ndarray:
    """Generate a charged-particle helix through a uniform axial field.

    ``transverse_step`` advances along the projected path in the x-y plane;
    z advances by the matching ``p_z / p_T`` ratio. The requested step is
    reduced for tight helices so soft tracks remain smooth.

    ``HELIX_RADIUS_VISUAL_SCALE`` is the only deliberate geometric distortion:
    it opens the transverse curvature for readability without changing the
    track tangent or pitch.
    """
    if charge == 0:
        raise ValueError("helix generation requires a charged particle")
    if magnetic_field == 0:
        raise ValueError("helix generation requires a non-zero magnetic field")
    if n_points < 1:
        return np.empty((0, 3), dtype=np.float32)
    if transverse_step <= 0:
        raise ValueError("transverse_step must be positive")

    origin = np.asarray(origin, dtype=np.float64)
    momentum = np.asarray(momentum, dtype=np.float64)
    momentum_magnitude = float(np.linalg.norm(momentum))
    if momentum_magnitude == 0:
        return np.empty((0, 3), dtype=np.float32)

    px, py, pz = momentum
    transverse_momentum = float(np.hypot(px, py))
    if transverse_momentum > 0:
        initial_azimuth = float(np.arctan2(py, px))
        radius = (
            HELIX_RADIUS_VISUAL_SCALE
            * transverse_momentum
            / abs(charge * magnetic_field)
        )
        turn_sign = -float(np.sign(charge * magnetic_field))
        sample_step = min(transverse_step, radius * MAX_AZIMUTH_SAMPLE_STEP)
        longitudinal_rate = pz / transverse_momentum
    else:
        sample_step = max_z / max(n_points - 1, 1)
        longitudinal_rate = float(np.sign(pz))

    points: list[np.ndarray] = []
    for point_index in range(n_points):
        sampled_distance = point_index * sample_step
        z = origin[2] + longitudinal_rate * sampled_distance

        if transverse_momentum == 0:
            x, y = origin[:2]
        else:
            azimuth = initial_azimuth + turn_sign * sampled_distance / radius
            x = origin[0] + radius / turn_sign * (
                np.sin(azimuth) - np.sin(initial_azimuth)
            )
            y = origin[1] - radius / turn_sign * (
                np.cos(azimuth) - np.cos(initial_azimuth)
            )

        radial_distance = float(np.hypot(x, y))
        if radial_distance > max_radius or abs(z) > max_z:
            break

        points.append(np.array([x, y, z]))
        if shower_radius is not None and radial_distance > shower_radius:
            break

    return np.asarray(points, dtype=np.float32).reshape(-1, 3)
