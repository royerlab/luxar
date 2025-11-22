"""luxar.demos – Demo scene generators for Luxar.

This module provides functions to generate various demo scenes for testing
and demonstration purposes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from arbol import aprint

from ..core.dimensions import Dimension, Dimensions
from ..io.compiler import LuxarZarrCompiler
from ..typing_utils.config import check_dataset_size_warning
from ..typing_utils.protocols import PathLike


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
    warning = check_dataset_size_warning(n_points)
    if warning:
        aprint(f"Performance warning: {warning}")

    # Lorenz attractor parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dt = 0.01

    # Initialize arrays
    positions = np.zeros((n_points, 3), dtype=np.float32)

    # Starting point (with small random perturbation if seed is provided)
    rng = np.random.default_rng(seed)
    x, y, z = 0.1, 0.0, 0.0
    if seed is not None:
        x += rng.uniform(-0.01, 0.01)

    # Generate Lorenz attractor points
    for i in range(n_points):
        # Lorenz equations
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        x += dx
        y += dy
        z += dz

        positions[i] = [x, y, z]

    # Scale positions to fit nicely in view
    positions *= 0.1

    # Center the attractor at its center of mass
    center_of_mass = np.mean(positions, axis=0)
    positions -= center_of_mass

    # Create time-based colors with smooth transitions
    # Using HSV color space for smooth color transitions
    t = np.linspace(0, 1, n_points)
    hue = (t * 2) % 1.0  # Cycle through hues twice

    # Convert HSV to RGB using vectorized operations
    # Full saturation and value for vibrant colors
    s, v = 1.0, 1.0

    # HSV to RGB vectorized conversion
    # Based on standard HSV→RGB algorithm, vectorized for performance
    c = v * s  # Chroma
    h_prime = hue * 6.0  # Hue in [0, 6) range
    x = c * (1 - np.abs(h_prime % 2 - 1))  # Intermediate value
    m = v - c  # Match value

    # Initialize RGB arrays
    r = np.zeros(n_points, dtype=np.float32)
    g = np.zeros(n_points, dtype=np.float32)
    b = np.zeros(n_points, dtype=np.float32)

    # Apply RGB values based on hue sector (0-5)
    # Each sector represents 60° of the color wheel
    sector = np.floor(h_prime).astype(int)

    # Sector 0: Red to Yellow (R=max, G=rising, B=0)
    mask = (sector == 0)
    r[mask], g[mask], b[mask] = c, x[mask], 0.0

    # Sector 1: Yellow to Green (R=falling, G=max, B=0)
    mask = (sector == 1)
    r[mask], g[mask], b[mask] = x[mask], c, 0.0

    # Sector 2: Green to Cyan (R=0, G=max, B=rising)
    mask = (sector == 2)
    r[mask], g[mask], b[mask] = 0.0, c, x[mask]

    # Sector 3: Cyan to Blue (R=0, G=falling, B=max)
    mask = (sector == 3)
    r[mask], g[mask], b[mask] = 0.0, x[mask], c

    # Sector 4: Blue to Magenta (R=rising, G=0, B=max)
    mask = (sector == 4)
    r[mask], g[mask], b[mask] = x[mask], 0.0, c

    # Sector 5: Magenta to Red (R=max, G=0, B=falling)
    mask = (sector == 5)
    r[mask], g[mask], b[mask] = c, 0.0, x[mask]

    # Add match value to get final RGB (adjust for brightness)
    colors = np.column_stack([r + m, g + m, b + m]).astype(np.float32)

    # Generate radii based on position in the trajectory (growing over time)
    # This creates a visual effect of the attractor "growing" as it evolves
    radii = np.linspace(0.01, 0.02, n_points).astype(np.float32)

    # Create scene with new API
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
        compiler.write_points(
            "LorenzAttractor",
            positions * 100.0 - 50.0,  # Scale up for better visibility
            colors=colors,
            radii=radii,
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

    with LuxarZarrCompiler(store_path) as compiler:
        compiler.create_scene()

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

            # Random HDR color for each sphere
            color = rng.uniform(0.5, 2.0, 3).astype(np.float32)  # HDR colors

            compiler.write_points(
                f"sphere_{i:03d}",
                positions,
                colors=color,  # Single color for whole sphere
                radii=np.float32(0.05),
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
            "time_series", positions_array, colors=colors_array, radii=np.float32(0.1)
        )

    aprint(f"✓ Time series demo created at {store_path}")
