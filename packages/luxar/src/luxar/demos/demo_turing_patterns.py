#!/usr/bin/env python3
"""Self-Contained Demo: 2D Turing Patterns (Reaction-Diffusion)

This demo demonstrates:
- Gray-Scott reaction-diffusion system creating organic patterns
- Multiple pattern types: spots, stripes, spirals, labyrinths
- Temporal evolution showing pattern formation
- Categorical navigation between different parameter sets
- Beautiful emergent complexity from simple rules
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all simulation code is here!

Mathematical Background:
    The Gray-Scott model simulates two chemicals U and V reacting and diffusing:

    ∂U/∂t = D_u∇²U - UV² + F(1-U)
    ∂V/∂t = D_v∇²V + UV² - (F+k)V

    Where:
    - D_u, D_v: Diffusion rates
    - F: Feed rate (adds U)
    - k: Kill rate (removes V)
    - UV²: Reaction term

    Different (F, k) parameters create different patterns:
    - Spots: F=0.055, k=0.062 (leopard/cheetah)
    - Stripes: F=0.035, k=0.060 (zebra)
    - Spirals: F=0.018, k=0.051
    - Mitosis: F=0.029, k=0.057

Usage:
    python demo_turing_patterns.py [--size=N] [--steps=N]

Controls:
    - Press '1' to select PATTERN TYPE
    - Press '2' to navigate through TIME
    - Watch patterns emerge and evolve!
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "turing_patterns",
    "title": "Turing Patterns",
    "description": "Gray-Scott reaction-diffusion Turing patterns (spots, stripes, spirals) evolving in time.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["turing_patterns"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection
from scipy.ndimage import laplace

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def gray_scott_step(
    U: np.ndarray,
    V: np.ndarray,
    D_u: float,
    D_v: float,
    F: float,
    k: float,
    dt: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Single step of Gray-Scott reaction-diffusion.

    Args:
        U, V: Concentration fields
        D_u, D_v: Diffusion coefficients
        F: Feed rate
        k: Kill rate
        dt: Time step

    Returns:
        Updated (U, V) fields
    """
    # Compute Laplacian (diffusion)
    lap_U = laplace(U)
    lap_V = laplace(V)

    # Reaction term
    UV2 = U * V * V

    # Update equations
    dU = D_u * lap_U - UV2 + F * (1 - U)
    dV = D_v * lap_V + UV2 - (F + k) * V

    U_new = U + dt * dU
    V_new = V + dt * dV

    # Clamp to valid range
    U_new = np.clip(U_new, 0, 1)
    V_new = np.clip(V_new, 0, 1)

    return U_new, V_new


def simulate_pattern(
    size: int,
    F: float,
    k: float,
    D_u: float = 0.16,
    D_v: float = 0.08,
    n_steps: int = 10000,
    save_every: int = 500,
) -> list[np.ndarray]:
    """Simulate Gray-Scott to generate Turing pattern.

    Args:
        size: Grid size (NxN)
        F: Feed rate (controls pattern type)
        k: Kill rate (controls pattern type)
        D_u: Diffusion rate for U
        D_v: Diffusion rate for V
        n_steps: Total simulation steps
        save_every: Save snapshot every N steps

    Returns:
        List of V concentration fields at different times
    """
    # Initialize with U=1 everywhere, V=0
    U = np.ones((size, size))
    V = np.zeros((size, size))

    # Add small random perturbation in center to seed pattern
    center = size // 2
    radius = size // 10
    y, x = np.ogrid[:size, :size]
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
    V[mask] = 1.0
    U[mask] = 0.5

    snapshots = []

    # Simulate
    for step in range(n_steps):
        U, V = gray_scott_step(U, V, D_u, D_v, F, k, dt=1.0)

        # Save snapshots
        if step % save_every == 0:
            snapshots.append(V.copy())

    return snapshots


def generate_turing_patterns(
    output_path: Path,
    grid_size: int = 256,
    n_steps: int = 10000,
) -> int:
    """Generate multiple Turing patterns with different parameters.

    Args:
        output_path: Where to write zarr
        grid_size: Resolution (NxN)
        n_steps: Simulation steps

    Returns:
        Total points
    """
    # Pattern configurations (F, k, name)
    patterns = [
        (0.055, 0.062, "Spots"),  # Leopard spots
        (0.035, 0.060, "Stripes"),  # Zebra stripes
        (0.018, 0.051, "Spirals"),  # Spiral waves
        (0.029, 0.057, "Mitosis"),  # Cell division
        (0.026, 0.051, "Coral"),  # Coral growth
        (0.014, 0.054, "Labyrinth"),  # Maze-like
    ]

    all_positions = []
    all_colors = []

    with asection(f"Generating {len(patterns)} Turing Patterns"):
        for pattern_id, (F, k, name) in enumerate(patterns):
            with asection(f"Pattern {pattern_id}: {name} (F={F}, k={k})"):
                aprint(f"  Simulating {n_steps:,} steps...")

                # Simulate with high temporal resolution
                snapshots = simulate_pattern(
                    size=grid_size,
                    F=F,
                    k=k,
                    n_steps=n_steps,
                    save_every=n_steps // 200,  # 200 snapshots (10× more detail)
                )

                aprint(f"  ✓ Generated {len(snapshots)} time snapshots")

                # Convert to points
                for time_id, V_field in enumerate(snapshots):
                    # Sample points from field (keep all grid points)
                    y_coords, x_coords = np.mgrid[:grid_size, :grid_size]

                    # Flatten
                    x_flat = x_coords.ravel().astype(np.float32) - grid_size / 2
                    y_flat = y_coords.ravel().astype(np.float32) - grid_size / 2
                    v_flat = V_field.ravel()

                    # Create 4D positions: [pattern, time, x, y]
                    pattern_ids = np.full(len(x_flat), pattern_id, dtype=np.float32)
                    time_ids = np.full(len(x_flat), time_id, dtype=np.float32)

                    positions = np.column_stack([pattern_ids, time_ids, x_flat, y_flat])

                    # Color by concentration (cool to warm)
                    colors = np.zeros((len(v_flat), 3), dtype=np.float32)
                    colors[:, 0] = v_flat  # Red increases with V
                    colors[:, 1] = v_flat * 0.5  # Some orange
                    colors[:, 2] = 1 - v_flat  # Blue for low V

                    all_positions.append(positions)
                    all_colors.append(colors)

                aprint(f"  ✓ {len(snapshots) * grid_size**2:,} points total")

    # Combine all patterns
    with asection("Combining patterns"):
        positions = np.vstack(all_positions)
        colors = np.vstack(all_colors)

        aprint(f"✓ Total: {len(positions):,} points")
        aprint(f"  {len(patterns)} patterns × 200 timepoints × {grid_size**2:,} pixels")

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension(
                    "pattern",
                    unit="",
                    categories=[
                        "Spots",
                        "Stripes",
                        "Spirals",
                        "Mitosis",
                        "Coral",
                        "Labyrinth",
                    ],
                    display=False,
                    description="Reaction-diffusion pattern type (different F,k parameters)",
                ),
                Dimension(
                    "time",
                    unit="step",
                    range=(0, len(snapshots) - 1),
                    step=1,
                    display=False,
                    discrete=True,
                    description="Time evolution (watch patterns emerge!)",
                ),
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Small flat points for 2D appearance
            radii = np.full(len(positions), 0.6, dtype=np.float32)
            sharpnesses = np.full(len(positions), 0.6, dtype=np.float32)

            scene.add_points(
                "TuringPatterns",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.9,
                intensity=0.1,
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "Gray-Scott Reaction-Diffusion",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Dimension-aware pattern labels with parameters
            pattern_info = [
                ("Spots", "F=0.055, k=0.062", "Leopard/cheetah pattern"),
                ("Stripes", "F=0.035, k=0.060", "Zebra-like pattern"),
                ("Spirals", "F=0.018, k=0.051", "Rotating spiral waves"),
                ("Mitosis", "F=0.029, k=0.057", "Cell division pattern"),
                ("Coral", "F=0.026, k=0.051", "Branching coral growth"),
                ("Labyrinth", "F=0.014, k=0.054", "Maze-like structure"),
            ]

            for pat_id, (name, params, desc) in enumerate(pattern_info):
                scene.add_html(
                    f'<div style="font-size:1.5vh;font-weight:bold;color:#ffcc44">{name}</div>'
                    f'<div style="font-size:1.3vh;color:#aaa;font-family:monospace">{params}</div>'
                    f'<div style="font-size:1.3vh;color:#888;margin-top:0.3vh">{desc}</div>',
                    position=(0.02, 0.97),
                    anchor="bottom-left",
                    visible_range={"pattern": pat_id},
                    transition="fade",
                    transition_duration=0.2,
                )

            # Equation (always visible)
            scene.add_html(
                '<div style="font-size:1.3vh;color:rgba(200,200,200,0.5);font-family:monospace">'
                "\u2202V/\u2202t = D\u2207\u00b2V + UV\u00b2 \u2212 (F+k)V"
                "</div>",
                position=(0.98, 0.97),
                anchor="bottom-right",
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{len(positions) * 32 / 1024 / 1024:.0f} MB")

    return len(positions)


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    grid_size = 256  # 256x256 grid
    n_steps = 10000  # 10k simulation steps

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--size="):
                grid_size = int(arg.split("=")[1])
            elif arg.startswith("--steps="):
                n_steps = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("TURING PATTERNS DEMO - REACTION-DIFFUSION")
    aprint("=" * 70)
    aprint("")
    aprint("Watch organic patterns emerge from simple chemical rules!")
    aprint(f"Grid: {grid_size}×{grid_size}")
    aprint(f"Simulation: {n_steps:,} steps")
    aprint("")
    aprint("Pattern types:")
    aprint("  • Spots: Leopard/cheetah-like (F=0.055, k=0.062)")
    aprint("  • Stripes: Zebra-like (F=0.035, k=0.060)")
    aprint("  • Spirals: Rotating waves (F=0.018, k=0.051)")
    aprint("  • Mitosis: Cell division (F=0.029, k=0.057)")
    aprint("  • Coral: Branching growth (F=0.026, k=0.051)")
    aprint("  • Labyrinth: Maze patterns (F=0.014, k=0.054)")
    aprint("")
    aprint("⏱️  Generation: ~2-5 minutes (running 6 simulations)")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "turing_patterns.luxar.zarr"
        generate_turing_patterns(
            output_path,
            grid_size=grid_size,
            n_steps=n_steps,
        )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_turing_") as tmpdir:
        output_path = Path(tmpdir) / "turing_patterns.luxar.zarr"

        # Generate patterns
        generate_turing_patterns(
            output_path,
            grid_size=grid_size,
            n_steps=n_steps,
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION - WATCH PATTERNS EMERGE!")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  1. Press '1' → Select PATTERN TYPE")
        aprint("     Switch between spots, stripes, spirals, etc.")
        aprint("")
        aprint("  2. Press '2' → Navigate through TIME")
        aprint("     Watch the pattern form from random noise!")
        aprint("     Use '[' to go backward, ']' to go forward")
        aprint("")
        aprint("What you're seeing:")
        aprint("  • Chemical concentration V (red=high, blue=low)")
        aprint("  • Self-organizing patterns from local interactions")
        aprint("  • Same equations explain animal markings in nature!")
        aprint("")
        aprint("Try this:")
        aprint("  1. Start with Spots (pattern=0)")
        aprint("  2. Navigate time from 0 → 20 to see spots form")
        aprint("  3. Switch to Stripes (pattern=1)")
        aprint("  4. Compare how different F,k create different patterns")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
