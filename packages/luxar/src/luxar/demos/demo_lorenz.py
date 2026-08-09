#!/usr/bin/env python3
"""Self-Contained Demo: Lorenz Attractor Visualization

This demo demonstrates:
- Generating a beautiful Lorenz attractor trajectory
- Time-based color gradient cycling through the color wheel
- Progressive writing for efficient memory usage
- Complete workflow: generate → serve → view → cleanup

The demo is self-contained - the generation code is in this file. The only
shared helper it borrows for generation is the vectorized HSV→RGB colouring
(alongside the usual `launch_viewer` / `parse_int_arg` plumbing).

Usage:
    python demo_lorenz.py [--points N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "lorenz",
    "title": "Lorenz Attractor",
    "description": "The chaotic Lorenz attractor trajectory with a time-based color gradient.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["lorenz"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import hsv_to_rgb, launch_viewer, parse_int_arg
from luxar.utils.paths import get_demos_output_dir

# Lorenz system parameters (classic values)
LORENZ_SIGMA = 10.0  # Prandtl number
LORENZ_RHO = 28.0  # Rayleigh number
LORENZ_BETA = 8.0 / 3.0  # Geometric factor
LORENZ_DT = 0.01  # Time step for Euler integration


def lorenz_trajectory(n_points: int, seed: int | None = None) -> np.ndarray:
    """Integrate the classic Lorenz attractor and return its trajectory.

    The Lorenz system

    .. code-block:: text

        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y
        dz/dt = xy - βz

    is integrated with forward Euler using the classic parameters σ=10.0
    (Prandtl number), ρ=28.0 (Rayleigh number), β=8/3 (geometric factor) and a
    time step dt=0.01, starting from (0.1, 0.0, 0.0). The result is scaled by
    0.1 and centered on its center of mass so it frames nicely in a viewer.

    Pure and silent: the narrated progress lives in the caller.

    Args:
        n_points: Number of trajectory samples to integrate.
        seed: If given, the start point is perturbed by a small seeded random
            offset (``x += U(-0.01, 0.01)``) for variety. ``None`` (the
            default) means no perturbation at all — a fully deterministic run.

    Returns:
        ``(n_points, 3)`` float32 positions, scaled and mean-centered.
    """
    positions = np.zeros((n_points, 3), dtype=np.float32)

    # Starting point (with small random perturbation if seed is provided)
    x, y, z = 0.1, 0.0, 0.0
    if seed is not None:
        rng = np.random.default_rng(seed)
        x += rng.uniform(-0.01, 0.01)

    for i in range(n_points):
        dx = LORENZ_SIGMA * (y - x) * LORENZ_DT
        dy = (x * (LORENZ_RHO - z) - y) * LORENZ_DT
        dz = (x * y - LORENZ_BETA * z) * LORENZ_DT

        x += dx
        y += dy
        z += dz

        positions[i] = [x, y, z]

    # Scale positions to fit nicely in view, then center at the center of mass.
    positions *= 0.1
    positions -= np.mean(positions, axis=0)
    return positions


def generate_lorenz_attractor(
    output_path: Path, n_points: int = 500000, seed: int | None = None
) -> None:
    """Generate Lorenz attractor with time-based colors.

    The generation logic is in this file (:func:`lorenz_trajectory`); the only
    thing borrowed is the shared ``hsv_to_rgb`` colour helper.

    The Lorenz system is a set of differential equations that produces
    chaotic behavior, creating a beautiful butterfly-shaped attractor.

    Args:
        output_path: Where to write the zarr store
        n_points: Number of points along the trajectory
        seed: Random seed for reproducibility (adds small perturbation)
    """
    with asection(f"Generating Lorenz Attractor ({n_points:,} points)"):
        aprint(
            f"Parameters: σ={LORENZ_SIGMA}, ρ={LORENZ_RHO}, "
            f"β={LORENZ_BETA:.3f}, dt={LORENZ_DT}"
        )
        if seed is not None:
            aprint(f"Using seed {seed}: start point perturbed by up to ±0.01 in x")

        # Integrate Lorenz equations using Euler method, then scale and center
        # the attractor for nice viewing.
        aprint("Integrating Lorenz equations...")
        positions = lorenz_trajectory(n_points, seed=seed)
        aprint(
            f"✓ Generated trajectory (bounds: {positions.min():.2f} to {positions.max():.2f})"
        )

        # Create time-based color gradient (cycles through color wheel twice)
        aprint("Generating time-based color gradient...")
        colors = hsv_to_rgb((np.linspace(0, 1, n_points) * 2.0) % 1.0)
        aprint("✓ Generated rainbow color gradient")

    # Write to zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Scale up for better visibility and center in view
            # NOTE: radii must be scaled by same factor as positions!
            scene.add_points(
                "LorenzAttractor",
                positions * 100.0 - 50.0,  # Scale and offset for good framing
                colors=colors * 0.5,  # Dim colors for better visibility
                radii=1.0,  # Uniform radius for visibility
                opacity=0.9,
                blending_mode="additive",
                intensity=0.0625,
                layer=True,
            )

            # Overlay annotations
            scene.add_text(
                "Lorenz Attractor",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "\u03c3=10, \u03c1=28, \u03b2=8/3 \u2022 500K points",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written to {output_path}")


def main() -> None:
    """Main demo entry point."""
    # Parse command line args (optional). parse_int_arg accepts both
    # `--points=N` and `--points N`, so forwarded args from `luxar demo run`
    # are not position-sensitive.
    n_points = parse_int_arg("points", 500000)

    aprint("=" * 70)
    aprint("LORENZ ATTRACTOR DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a beautiful chaotic attractor with rainbow colors")
    aprint(f"Points: {n_points:,}")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "lorenz.luxar.zarr"
        generate_lorenz_attractor(output_path, n_points=n_points)
        aprint(f"✓ Dataset generated at {output_path}")
        return

    # Use temporary directory for demo data (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_lorenz_") as tmpdir:
        output_path = Path(tmpdir) / "lorenz.luxar.zarr"

        # Generate the dataset (the integrator lives in this file)
        generate_lorenz_attractor(output_path, n_points=n_points)

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically when tempfile context exits
    aprint("✓ Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
