"""Tests for the Lorenz integrator in demo_lorenz.

``lorenz_trajectory`` is the demo's distinctive scientific content, and it also
backs the ``create_lorenz_attractor`` fixture builder in ``luxar.utils.demos``,
so its determinism and framing are worth pinning down here.
"""

from __future__ import annotations

import numpy as np

from luxar.demos.demo_lorenz import (
    LORENZ_BETA,
    LORENZ_DT,
    LORENZ_RHO,
    LORENZ_SIGMA,
    lorenz_trajectory,
)

# Captured from the integrator itself; the point of pinning them is that a
# change to the constants, the Euler step, the 0.1 scaling or the centering
# moves these numbers well outside the tolerance below.
REFERENCE_LAST_POINT = (-0.05608694, -0.16396192, -1.45546365)
REFERENCE_EXTENT = (3.72236300, 4.87627125, 5.43200636)
# Seeded run, so the ±0.01 start perturbation is pinned too.
REFERENCE_LAST_POINT_SEED_7 = (-0.05835141, -0.16847360, -1.46010971)


class TestLorenzTrajectory:
    """Tests for the lorenz_trajectory helper."""

    def test_shape_and_dtype(self) -> None:
        """Trajectory is an (n_points, 3) float32 array."""
        positions = lorenz_trajectory(200)
        assert positions.shape == (200, 3)
        assert positions.dtype == np.float32

    def test_seed_is_deterministic(self) -> None:
        """The same seed reproduces the same trajectory exactly."""
        first = lorenz_trajectory(150, seed=42)
        second = lorenz_trajectory(150, seed=42)
        assert np.array_equal(first, second)

    def test_different_seeds_differ(self) -> None:
        """Different seeds perturb the start point, diverging chaotically."""
        first = lorenz_trajectory(150, seed=1)
        second = lorenz_trajectory(150, seed=2)
        assert not np.array_equal(first, second)

    def test_no_seed_is_reproducible(self) -> None:
        """seed=None means no perturbation at all, so runs are identical."""
        first = lorenz_trajectory(150)
        second = lorenz_trajectory(150)
        assert np.array_equal(first, second)

    def test_mean_centered(self) -> None:
        """The trajectory is centered on its center of mass."""
        positions = lorenz_trajectory(500, seed=7)
        np.testing.assert_allclose(positions.mean(axis=0), np.zeros(3), atol=1e-4)

    def test_is_not_degenerate(self) -> None:
        """The attractor spreads over a few units along all three axes.

        A plain ``> 0`` check would pass on a system that decays to the origin
        (ρ=0.5 gives extents of 9e-3 and smaller), and an upper bound is what
        catches a dropped ``*= 0.1`` framing scale (which gives ~40).
        """
        extent = np.ptp(lorenz_trajectory(500, seed=7), axis=0)
        assert np.all(extent > 1.0), extent
        assert np.all(extent < 10.0), extent

    def test_uses_the_classic_lorenz_parameters(self) -> None:
        """The published σ/ρ/β/dt — the demo's banner prints these values."""
        assert LORENZ_SIGMA == 10.0
        assert LORENZ_RHO == 28.0
        assert LORENZ_BETA == 8.0 / 3.0
        assert LORENZ_DT == 0.01

    def test_matches_reference_trajectory(self) -> None:
        """Pin the actual numbers, not just the shape of the output.

        Determinism and centering hold for any integrator; this is what fails
        if the constants, the Euler step order or the framing scale change.
        The seeded run pins the size of the start perturbation as well.
        """
        positions = lorenz_trajectory(500)
        np.testing.assert_allclose(
            positions[-1], REFERENCE_LAST_POINT, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.ptp(positions, axis=0), REFERENCE_EXTENT, rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            lorenz_trajectory(500, seed=7)[-1],
            REFERENCE_LAST_POINT_SEED_7,
            rtol=1e-4,
            atol=1e-4,
        )
