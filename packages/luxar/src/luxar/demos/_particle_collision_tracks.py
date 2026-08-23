"""Shared charged-track sampling for the particle-collision demos.

The demos deliberately render helices at twice their physical radius for visual
clarity. This module owns that distortion and the sampling contract so the static
and animated adapters cannot drift apart. It returns only centerline points;
colors, widths, segment expansion, and animation timing remain adapter concerns.
"""

from __future__ import annotations

import numpy as np

HELIX_RADIUS_VISUAL_SCALE = 2.0
MAX_AZIMUTH_SAMPLE_STEP = 0.05
TRACK_TRANSVERSE_ARC_BUDGET = 16.0


def _transverse_sampling(
    *,
    transverse_momentum: float,
    charge: int,
    magnetic_field: float,
    requested_step: float,
    n_points: int,
) -> tuple[float, float, float, float]:
    radius = (
        HELIX_RADIUS_VISUAL_SCALE * transverse_momentum / abs(charge * magnetic_field)
    )
    turn_sign = -float(np.sign(charge * magnetic_field))
    smooth_step = min(requested_step, radius * MAX_AZIMUTH_SAMPLE_STEP)
    max_sampled_distance = min(TRACK_TRANSVERSE_ARC_BUDGET, 2 * np.pi * radius)
    coverage_step = max_sampled_distance / max(n_points - 1, 1)
    sample_step = max(smooth_step, coverage_step)
    return radius, turn_sign, sample_step, max_sampled_distance


def _sample_track_points(
    *,
    origin: np.ndarray,
    initial_azimuth: float,
    transverse_momentum: float,
    longitudinal_rate: float,
    radius: float,
    turn_sign: float,
    sample_step: float,
    max_radius: float,
    max_z: float,
    n_points: int,
    shower_radius: float | None,
    max_sampled_distance: float | None,
) -> np.ndarray:
    points: list[np.ndarray] = []
    for point_index in range(n_points):
        sampled_distance = point_index * sample_step
        if max_sampled_distance is not None and sampled_distance > max_sampled_distance:
            break
        z = origin[2] + longitudinal_rate * sampled_distance

        if transverse_momentum == 0:
            x, y = origin[:2]
        else:
            azimuth = initial_azimuth + turn_sign * sampled_distance / radius
            x = origin[0] + radius * turn_sign * (
                np.sin(azimuth) - np.sin(initial_azimuth)
            )
            y = origin[1] - radius * turn_sign * (
                np.cos(azimuth) - np.cos(initial_azimuth)
            )

        radial_distance = float(np.hypot(x, y))
        if radial_distance > max_radius or abs(z) > max_z:
            break

        points.append(np.array([x, y, z]))
        if shower_radius is not None and radial_distance > shower_radius:
            break

    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


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
    reduced for tight helices. Sampling covers at most one transverse revolution
    or the shared ``TRACK_TRANSVERSE_ARC_BUDGET``, whichever is shorter. If the
    point ceiling cannot cover that useful arc at the angular smoothing limit,
    the samples become only as coarse as needed to preserve coverage.

    The transverse path follows the parametric equations
    ``x = x0 + sign*r*(sin(phi) - sin(phi0))`` and
    ``y = y0 - sign*r*(cos(phi) - cos(phi0))``. In real detector units,
    ``r[m] = p_T[GeV/c] / (0.3 * |q| * B[T])``; the demos omit the unit-conversion
    factor and apply the named visual scale below.

    ``HELIX_RADIUS_VISUAL_SCALE`` is the only deliberate geometric distortion:
    it opens the transverse curvature for readability without changing the
    track tangent or pitch.

    Args:
        origin: Track origin as an ``(x, y, z)`` vector.
        momentum: Momentum as a ``(p_x, p_y, p_z)`` vector.
        charge: Non-zero electric charge in elementary-charge units.
        magnetic_field: Non-zero axial magnetic field strength.
        max_radius: Maximum cylindrical detector radius.
        max_z: Maximum absolute longitudinal detector coordinate.
        n_points: Maximum number of returned samples. If this ceiling cannot
            cover the useful transverse arc at the angular smoothing limit,
            coverage wins and the samples become coarser.
        transverse_step: Requested arc-length step in the transverse plane.
        shower_radius: Optional radius at which to include one final shower sample,
            used for electrons once bremsstrahlung starts an EM cascade.

    Returns:
        An ``(N, 3)`` float32 array of sampled centerline points.

    Raises:
        ValueError: If charge or magnetic field is zero, or the requested step
            is not positive.
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
        radius, turn_sign, sample_step, max_sampled_distance = _transverse_sampling(
            transverse_momentum=transverse_momentum,
            charge=charge,
            magnetic_field=magnetic_field,
            requested_step=transverse_step,
            n_points=n_points,
        )
        longitudinal_rate = pz / transverse_momentum
    else:
        # With no transverse motion, use the point budget along z: a transverse
        # step is undefined and neither radial detector boundary can be reached.
        sample_step = max_z / max(n_points - 1, 1)
        longitudinal_rate = float(np.sign(pz))
        initial_azimuth = radius = turn_sign = 0.0
        max_sampled_distance = None

    return _sample_track_points(
        origin=origin,
        initial_azimuth=initial_azimuth,
        transverse_momentum=transverse_momentum,
        longitudinal_rate=longitudinal_rate,
        radius=radius,
        turn_sign=turn_sign,
        sample_step=sample_step,
        max_radius=max_radius,
        max_z=max_z,
        n_points=n_points,
        shower_radius=shower_radius,
        max_sampled_distance=max_sampled_distance,
    )
