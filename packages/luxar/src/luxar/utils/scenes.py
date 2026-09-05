"""Reusable demo scene generators."""

from __future__ import annotations

from typing import Optional

import numpy as np
from arbol import aprint

from ..core.dimensions import Dimension, Dimensions
from ..typing_utils.aliases import PathLike
from .colors import hsv_to_rgb

# Point counts above which a demo scene prints a heads-up, and the per-point
# byte cost used to size the estimate (float32 xyz + uint8 rgb). These used to
# live in ``typing_utils.config`` alongside ~25 constants nothing read; this is
# the only caller, so they live with it now.
_MAX_RECOMMENDED_POINTS = 10_000_000
_LARGE_DATASET_POINTS = 1_000_000
_BYTES_PER_POINT = 3 * 4 + 3  # positions (3 x float32) + colors (3 x uint8)


def _dataset_size_warning(n_points: int) -> Optional[str]:
    """Return a performance heads-up for a large demo scene, or ``None``.

    Args:
        n_points: Number of points the demo is about to generate.

    Returns:
        A message when ``n_points`` is large enough to be worth mentioning,
        ``None`` otherwise.
    """
    if n_points > _MAX_RECOMMENDED_POINTS:
        return (
            f"Dataset with {n_points:,} points exceeds recommended maximum "
            f"of {_MAX_RECOMMENDED_POINTS:,} points. Performance may be degraded."
        )
    if n_points > _LARGE_DATASET_POINTS:
        memory_mb = n_points * _BYTES_PER_POINT / (1024 * 1024)
        return (
            f"Large dataset with {n_points:,} points (~{memory_mb:.1f}MB). "
            f"Consider using compression and chunking for better performance."
        )
    return None


def create_lorenz_attractor(
    store_path: PathLike,
    n_points: int = 10_000,
    seed: Optional[int] = None,
) -> None:
    """Create a demo scene with a Lorenz attractor visualization.

    This creates a beautiful butterfly-shaped 3D structure with colors
    that transition smoothly over time, demonstrating the Luxar scene format
    with an aesthetically pleasing mathematical visualization.

    Args:
        store_path: Path to the Zarr store to create
        n_points: Number of points to generate along the attractor
        seed: Random seed for reproducible results

    Example:
        >>> from luxar.utils import create_lorenz_attractor
        >>> create_lorenz_attractor('demo.zarr', n_points=50000)
    """
    aprint(f"Creating Lorenz attractor demo scene with {n_points:,} points.")

    # Check for performance warnings
    warning = _dataset_size_warning(n_points)
    if warning:
        aprint(f"Performance warning: {warning}")

    # Reuse the demo's own integrator so this fixture and `luxar demo run
    # lorenz` trace the very same trajectory (the two scenes still differ in
    # radii, brightness and overlays).
    # Imported here (not at module top) to break the import cycle
    # demos.demo_lorenz -> luxar.demos -> utils.scenes.
    from ..demos.demo_lorenz import lorenz_trajectory

    positions = lorenz_trajectory(n_points, seed=seed)
    # Cycle through hues twice for smooth time-based colour transitions.
    colors = hsv_to_rgb((np.linspace(0, 1, n_points) * 2.0) % 1.0)

    # Create scene.
    # Imported here (not at module top) to break the import cycle
    # io.compiler -> luxar.utils -> utils.scenes -> io.compiler.
    from ..io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(store_path) as compiler:
        # Define 3D dimensions
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        compiler.create_scene(dimensions=dims)

        # Write the attractor data
        # NOTE: radii must be scaled by same factor as positions!
        compiler.write_points(
            "LorenzAttractor",
            positions * 100.0 - 50.0,  # type: ignore[arg-type]  # Scale up for better visibility
            colors=colors,
            radii=2.0,  # Uniform radius for visibility
            opacity=0.9,
            blending_mode="additive",
        )

    aprint(f"✓ Lorenz attractor demo scene created at {store_path}")


def create_random_spheres(
    store_path: PathLike,
    n_spheres: int = 100,
    points_per_sphere: int = 1000,
    seed: Optional[int] = None,
) -> None:
    """Create a demo scene with random colored spheres.

    Args:
        store_path: Path to the Zarr store to create
        n_spheres: Number of spheres to generate
        points_per_sphere: Points per sphere
        seed: Random seed for reproducible results
    """
    rng = np.random.default_rng(seed)

    aprint(f"Creating random spheres demo with {n_spheres} spheres")

    # Imported here (not at module top) to break the import cycle
    # io.compiler -> luxar.utils -> utils.scenes -> io.compiler.
    from ..io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(store_path) as compiler:
        from luxar import Dimensions

        compiler.create_scene(dimensions=Dimensions.default_3d())

        for i in range(n_spheres):
            # Random center position
            center = rng.uniform(-10, 10, 3)

            # Generate points on sphere surface
            phi = rng.uniform(0, 2 * np.pi, points_per_sphere)
            costheta = rng.uniform(-1, 1, points_per_sphere)
            u = rng.uniform(0, 1, points_per_sphere)

            theta = np.arccos(costheta)

            radius = rng.uniform(0.5, 2.0)
            r = radius * u ** (1 / 3)

            x = r * np.sin(theta) * np.cos(phi) + center[0]
            y = r * np.sin(theta) * np.sin(phi) + center[1]
            z = r * np.cos(theta) + center[2]

            positions = np.column_stack([x, y, z]).astype(np.float32)

            # Random HDR color for each sphere - use tuple for scalar passthrough
            color_arr = rng.uniform(0.5, 2.0, 3)
            color_tuple = tuple(float(c) for c in color_arr)  # Convert to tuple

            compiler.write_points(
                f"sphere_{i:03d}",
                positions,
                colors=color_tuple,  # Single color for whole sphere (tuple for broadcasting)
                radii=0.05,  # Scalar passthrough
                opacity=0.8,
            )

    aprint(f"✓ Random spheres demo created at {store_path}")


def create_time_series_demo(
    store_path: PathLike,
    n_timepoints: int = 10,
    n_points_per_time: int = 1000,
    seed: Optional[int] = None,
) -> None:
    """Create a 4D time series demo scene.

    Args:
        store_path: Path to the Zarr store to create
        n_timepoints: Number of time points
        n_points_per_time: Points per time step
        seed: Random seed
    """
    rng = np.random.default_rng(seed)

    aprint(f"Creating 4D time series demo with {n_timepoints} time points")

    # Ensure at least 2 timepoints for valid dimension range
    if n_timepoints < 2:
        n_timepoints = 2
        aprint("  Note: Minimum 2 time points required, using 2")

    # Create 4D dimensions
    dims = Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
            Dimension(
                "time",
                unit="s",
                display=False,
                discrete=True,
                range=(0, n_timepoints - 1),
            ),
        ]
    )

    # Imported here (not at module top) to break the import cycle
    # io.compiler -> luxar.utils -> utils.scenes -> io.compiler.
    from ..io.compiler import LuxarZarrCompiler

    with LuxarZarrCompiler(store_path) as compiler:
        compiler.create_scene(dimensions=dims)

        # Generate all time points
        all_positions = []
        all_colors = []

        for t in range(n_timepoints):
            # Generate expanding sphere over time
            radius = 1.0 + t * 0.5

            # Random points in sphere
            phi = rng.uniform(0, 2 * np.pi, n_points_per_time)
            costheta = rng.uniform(-1, 1, n_points_per_time)
            u = rng.uniform(0, 1, n_points_per_time)

            theta = np.arccos(costheta)
            r = radius * u ** (1 / 3)

            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Add time dimension
            time_coord = np.full(n_points_per_time, t, dtype=np.float32)

            positions_4d = np.column_stack([x, y, z, time_coord]).astype(np.float32)
            all_positions.append(positions_4d)

            # Color changes over time
            color = np.array(
                [1.0 - t / n_timepoints, 0.5, t / n_timepoints], dtype=np.float32
            )
            colors = np.tile(color, (n_points_per_time, 1))
            all_colors.append(colors)

        # Concatenate all time points
        positions_array = np.vstack(all_positions)
        colors_array = np.vstack(all_colors)

        # Write as single 4D dataset
        compiler.write_points(
            "time_series",
            positions_array,
            colors=colors_array,
            radii=0.1,  # Scalar passthrough
        )

    aprint(f"✓ Time series demo created at {store_path}")
