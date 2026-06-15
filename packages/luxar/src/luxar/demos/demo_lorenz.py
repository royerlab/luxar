#!/usr/bin/env python3
"""Self-Contained Demo: Lorenz Attractor Visualization

This demo demonstrates:
- Generating a beautiful Lorenz attractor trajectory
- Time-based color gradient cycling through the color wheel
- Progressive writing for efficient memory usage
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all generation code is in this file.

Usage:
    python demo_lorenz.py [--points N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def generate_lorenz_attractor(
    output_path: Path, n_points: int = 500000, seed: int | None = None
) -> None:
    """Generate Lorenz attractor with time-based colors.

    This function contains ALL the generation logic - completely self-contained.

    The Lorenz system is a set of differential equations that produces
    chaotic behavior, creating a beautiful butterfly-shaped attractor.

    Args:
        output_path: Where to write the zarr store
        n_points: Number of points along the trajectory
        seed: Random seed for reproducibility (adds small perturbation)
    """
    with asection(f"Generating Lorenz Attractor ({n_points:,} points)"):
        # Lorenz system parameters (classic values)
        sigma = 10.0  # Prandtl number
        rho = 28.0  # Rayleigh number
        beta = 8.0 / 3.0  # Geometric factor
        dt = 0.01  # Time step for Euler integration

        aprint(f"Parameters: σ={sigma}, ρ={rho}, β={beta:.3f}, dt={dt}")

        # Initialize trajectory
        positions = np.zeros((n_points, 3), dtype=np.float32)

        # Starting point (with optional random perturbation for variety)
        x, y, z = 0.1, 0.0, 0.0
        if seed is not None:
            rng = np.random.default_rng(seed)
            x += rng.uniform(-0.01, 0.01)
            aprint(f"Using seed {seed} with perturbed start: ({x:.3f}, {y}, {z})")

        # Integrate Lorenz equations using Euler method
        aprint("Integrating Lorenz equations...")
        for i in range(n_points):
            # Lorenz differential equations:
            # dx/dt = σ(y - x)
            # dy/dt = x(ρ - z) - y
            # dz/dt = xy - βz
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt

            x += dx
            y += dy
            z += dz

            positions[i] = [x, y, z]

        # Scale and center the attractor for nice viewing
        positions *= 0.1  # Scale down
        center = np.mean(positions, axis=0)
        positions -= center  # Center at origin
        aprint(
            f"✓ Generated trajectory (bounds: {positions.min():.2f} to {positions.max():.2f})"
        )

        # Create time-based color gradient (cycles through color wheel twice)
        aprint("Generating time-based color gradient...")
        t = np.linspace(0, 1, n_points)
        hue = (t * 2) % 1.0  # Cycle through hues twice for visual interest

        # Vectorized HSV to RGB conversion (full saturation and value)
        s, v = 1.0, 1.0
        c = v * s  # Chroma
        h_prime = hue * 6.0  # Hue in [0, 6) range
        x_val = c * (1 - np.abs(h_prime % 2 - 1))  # Intermediate value
        m = v - c  # Match value

        # Initialize RGB arrays
        r = np.zeros(n_points, dtype=np.float32)
        g = np.zeros(n_points, dtype=np.float32)
        b = np.zeros(n_points, dtype=np.float32)

        # Apply RGB values based on hue sector (vectorized)
        sector = np.floor(h_prime).astype(int)
        mask = sector == 0
        r[mask], g[mask], b[mask] = c, x_val[mask], 0.0
        mask = sector == 1
        r[mask], g[mask], b[mask] = x_val[mask], c, 0.0
        mask = sector == 2
        r[mask], g[mask], b[mask] = 0.0, c, x_val[mask]
        mask = sector == 3
        r[mask], g[mask], b[mask] = 0.0, x_val[mask], c
        mask = sector == 4
        r[mask], g[mask], b[mask] = x_val[mask], 0.0, c
        mask = sector == 5
        r[mask], g[mask], b[mask] = c, 0.0, x_val[mask]

        colors = np.column_stack([r + m, g + m, b + m]).astype(np.float32)
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
    # Parse simple command line args (optional)
    n_points = 500000
    if len(sys.argv) > 1 and sys.argv[1].startswith("--points="):
        n_points = int(sys.argv[1].split("=")[1])

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

        # Generate the dataset (all code in this file!)
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
