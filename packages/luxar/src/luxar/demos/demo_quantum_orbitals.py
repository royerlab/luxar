#!/usr/bin/env python3
"""Self-Contained Demo: Quantum Atomic Orbitals Visualization

This demo demonstrates:
- Hydrogen atom electron probability density (quantum orbitals)
- Beautiful 3D shapes: spheres, dumbbells, cloverleafs
- Categorical navigation between different quantum states
- Color-coded by probability density or phase
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all quantum mechanics code is here!

Mathematical Background:
    Hydrogen atom wavefunctions are products of:
    - Radial function R_nl(r) (depends on n, l)
    - Spherical harmonic Y_lm(θ, φ) (depends on l, m)

    ψ_nlm(r,θ,φ) = R_nl(r) × Y_lm(θ,φ)

    Probability density: ρ = |ψ|²

    Quantum numbers:
    - n: Principal (1, 2, 3, ...) - energy level
    - l: Angular momentum (0 to n-1) - shape (s, p, d, f)
    - m: Magnetic (-l to +l) - orientation

    Orbital naming:
    - l=0: s (spherical)
    - l=1: p (dumbbell)
    - l=2: d (cloverleaf)
    - l=3: f (complex)

Usage:
    python demo_quantum_orbitals.py [--grid=N]

Controls:
    - Press '1' to select ORBITAL TYPE
    - Press '['/']' to cycle through orbitals (1s → 2s → 2pz → ... → 3dxy → 1s)
    - Use dropdown to jump directly to any orbital
    - See different quantum states with clear labels!
    - Ctrl+C to stop
"""

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection
from scipy.special import genlaguerre, sph_harm_y

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def hydrogen_radial_wavefunction(
    r: np.ndarray, n: int, l_quantum: int, a0: float = 1.0
) -> np.ndarray:
    """Compute radial part of hydrogen wavefunction.

    Uses associated Laguerre polynomials.

    Args:
        r: Radial coordinates
        n: Principal quantum number (1, 2, 3, ...)
        l_quantum: Angular momentum (0 to n-1)
        a0: Bohr radius (scaling factor)

    Returns:
        Radial wavefunction R_nl(r)
    """
    # Dimensionless radius
    rho = 2 * r / (n * a0)

    # Normalization constant
    norm = np.sqrt(
        (2 / (n * a0)) ** 3
        * math.factorial(n - l_quantum - 1)
        / (2 * n * math.factorial(n + l_quantum))
    )

    # Associated Laguerre polynomial
    laguerre = genlaguerre(n - l_quantum - 1, 2 * l_quantum + 1)

    # Radial wavefunction
    R = norm * np.exp(-rho / 2) * (rho**l_quantum) * laguerre(rho)

    return R  # type: ignore[no-any-return]


def compute_orbital_density(
    n: int,
    l_quantum: int,
    m: int,
    grid_size: int = 80,
    box_size: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute probability density for hydrogen orbital.

    Args:
        n: Principal quantum number
        l_quantum: Angular momentum quantum number
        m: Magnetic quantum number
        grid_size: Resolution (N^3 points)
        box_size: Physical extent in Bohr radii

    Returns:
        Tuple of (positions, densities) where density threshold is applied
    """
    # Create 3D grid in Cartesian coordinates
    coords = np.linspace(-box_size / 2, box_size / 2, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    # Convert to spherical coordinates
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(np.clip(Z / (r + 1e-10), -1, 1))  # Polar angle
    phi = np.arctan2(Y, X)  # Azimuthal angle

    # Compute wavefunction components
    R = hydrogen_radial_wavefunction(r, n, l_quantum, a0=1.0)

    # Spherical harmonic (complex-valued)
    Y_lm = sph_harm_y(l_quantum, m, theta, phi)

    # Probability density: |ψ|² = |R|² × |Y|²
    density = (R * np.conj(R)).real * (Y_lm * np.conj(Y_lm)).real

    # Threshold to keep only significant density (reduce points)
    # Keep top regions that sum to 98% of total probability
    density_flat = density.ravel()
    sorted_indices = np.argsort(density_flat)[::-1]
    cumsum = np.cumsum(density_flat[sorted_indices])
    threshold_idx = np.searchsorted(cumsum, cumsum[-1] * 0.98)  # Keep 98%

    # Create mask for significant density
    threshold = density_flat[sorted_indices[threshold_idx]]
    keep_mask = density >= threshold

    # Get positions of significant density points
    positions = np.column_stack(
        [
            X[keep_mask],
            Y[keep_mask],
            Z[keep_mask],
        ]
    )

    density_values = density[keep_mask]

    return positions.astype(np.float32), density_values.astype(np.float32)


def generate_quantum_orbitals(
    output_path: Path,
    grid_size: int = 80,
) -> int:
    """Generate multiple hydrogen orbitals for visualization.

    Args:
        output_path: Where to write zarr
        grid_size: Resolution per orbital

    Returns:
        Total points
    """
    # Define orbitals to visualize
    orbitals = [
        (1, 0, 0, "1s"),  # Spherical ground state
        (2, 0, 0, "2s"),  # Sphere with radial node
        (2, 1, 0, "2p_z"),  # Dumbbell along z
        (2, 1, 1, "2p_x"),  # Dumbbell along x (complex combination)
        (3, 1, 0, "3p_z"),  # Larger dumbbell
        (3, 2, 0, "3d_z²"),  # Dumbbell with torus
        (3, 2, 1, "3d_xz"),  # Cloverleaf
        (3, 2, 2, "3d_xy"),  # Cloverleaf rotated
    ]

    all_positions = []
    all_colors = []
    all_orbital_ids = []

    with asection(f"Generating {len(orbitals)} Quantum Orbitals"):
        for orb_id, (n, l_quantum, m, name) in enumerate(orbitals):
            with asection(f"Orbital: {name} (n={n}, l={l_quantum}, m={m})"):
                aprint(f"  Grid: {grid_size}³ = {grid_size**3:,} samples")

                # Compute orbital density
                # Box size scales with principal quantum number n (larger n = bigger orbital)
                box_size = 20.0 + n * 15.0  # n=1: 35, n=2: 50, n=3: 65 Bohr radii
                positions, densities = compute_orbital_density(
                    n,
                    l_quantum,
                    m,
                    grid_size=grid_size,
                    box_size=box_size,
                )

                aprint(f"  ✓ {len(positions):,} points above threshold")
                aprint(f"    Density: {len(positions) / grid_size**3 * 100:.2f}%")

                # Normalize densities for coloring
                d_norm = densities / densities.max()

                # Create colors from density (hot colormap)
                colors = np.zeros((len(positions), 3), dtype=np.float32)
                colors[:, 0] = d_norm  # Red increases with density
                colors[:, 1] = d_norm * 0.5  # Some orange
                colors[:, 2] = np.maximum(0, 1 - d_norm * 2)  # Blue for low density

                # Add orbital ID as first coordinate
                orbital_ids = np.full(len(positions), orb_id, dtype=np.float32)

                all_positions.append(positions)
                all_colors.append(colors)
                all_orbital_ids.append(orbital_ids)

    # Combine all orbitals
    with asection("Combining orbitals"):
        positions = np.vstack(all_positions)
        colors = np.vstack(all_colors)
        orbital_ids = np.concatenate(all_orbital_ids)

        # Create 4D positions: [orbital_type, x, y, z]
        positions_4d = np.column_stack(
            [
                orbital_ids,
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
            ]
        )

        aprint(f"✓ Total: {len(positions_4d):,} points")
        aprint(f"  Per orbital: {len(positions_4d) // len(orbitals):,} avg")

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension(
                    "orbital",
                    unit="",
                    categories=[
                        "1s",
                        "2s",
                        "2pz",
                        "2px",
                        "3pz",
                        "3dz²",
                        "3dxz",
                        "3dxy",
                    ],
                    cyclic=True,  # Enable wrap-around from 3dxy back to 1s
                    display=False,
                    description="Quantum orbital state - hydrogen atom wavefunctions",
                ),
                Dimension("x", unit="a₀", display=True),
                Dimension("y", unit="a₀", display=True),
                Dimension("z", unit="a₀", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Larger radii to hide grid pattern and create smooth volumetric appearance
            # Grid spacing: ~0.25 units, use 0.35 for ~140% overlap
            radii = np.full(len(positions_4d), 0.35, dtype=np.float32)
            sharpnesses = np.full(
                len(positions_4d), 0.5, dtype=np.float32
            )  # Soft edges (normalized [0, 1] knob; 0.5 = Gaussian)

            scene.add_points(
                "QuantumOrbitals",
                positions_4d,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.6,  # Semi-transparent for volumetric effect
                intensity=0.016,
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "Hydrogen Atom Orbitals",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Dimension-aware orbital labels — one per orbital state
            orbital_descriptions = [
                ("1s", "n=1, l=0, m=0", "Spherical ground state"),
                ("2s", "n=2, l=0, m=0", "Sphere with radial node"),
                ("2pz", "n=2, l=1, m=0", "Dumbbell along z-axis"),
                ("2px", "n=2, l=1, m=1", "Dumbbell along x-axis"),
                ("3pz", "n=3, l=1, m=0", "Larger dumbbell"),
                ("3dz\u00b2", "n=3, l=2, m=0", "Dumbbell with torus"),
                ("3dxz", "n=3, l=2, m=1", "Cloverleaf pattern"),
                ("3dxy", "n=3, l=2, m=2", "Rotated cloverleaf"),
            ]

            for orb_id, (name, quantum, desc) in enumerate(orbital_descriptions):
                scene.add_html(
                    f'<div style="font-size:1.5vh;font-weight:bold;color:#ffcc44">{name}</div>'
                    f'<div style="font-size:1.3vh;color:#aaa">{quantum}</div>'
                    f'<div style="font-size:1.3vh;color:#888;margin-top:0.3vh">{desc}</div>',
                    position=(0.02, 0.97),
                    anchor="bottom-left",
                    visible_range={"orbital": orb_id},
                    transition="fade",
                    transition_duration=0.2,
                )

            # Color scale hint (always visible)
            scene.add_text(
                "\u2588 High probability  \u2588 Low probability",
                position=(0.5, 0.97),
                font_size=0.015,
                anchor="bottom-center",
                color="rgba(200,200,200,0.6)",
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{len(positions_4d) * 40 / 1024 / 1024:.0f} MB")

    return len(positions_4d)


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    grid_size = 80  # 80³ = 512k samples per orbital

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--grid="):
                grid_size = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("QUANTUM ATOMIC ORBITALS DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Visualize electron probability density in hydrogen atom!")
    aprint(f"Grid: {grid_size}³ per orbital")
    aprint("Orbitals: 1s, 2s, 2p, 3p, 3d")
    aprint("")
    aprint("Quantum mechanics made visible:")
    aprint("  • See where electrons are likely to be found")
    aprint("  • Different shapes for different energy levels")
    aprint("  • Beautiful mathematical structures")
    aprint("")
    aprint("Orbital shapes:")
    aprint("  • 1s: Spherical cloud (ground state)")
    aprint("  • 2s: Larger sphere with radial node")
    aprint("  • 2p: Dumbbell shapes (3 orientations)")
    aprint("  • 3d: Cloverleaf and complex shapes")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "quantum_orbitals.luxar.zarr"
        generate_quantum_orbitals(output_path, grid_size=grid_size)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_quantum_") as tmpdir:
        output_path = Path(tmpdir) / "quantum_orbitals.luxar.zarr"

        # Generate orbitals
        generate_quantum_orbitals(output_path, grid_size=grid_size)

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION - EXPLORE QUANTUM STATES!")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  1. Press '1' → Select ORBITAL TYPE")
        aprint("  2. Press '[' or ']' → Switch between orbitals:")
        aprint("     • 1s: Perfect sphere (simplest)")
        aprint("     • 2s: Sphere with shell structure")
        aprint("     • 2p: Three dumbbell orientations")
        aprint("     • 3p: Larger dumbbells")
        aprint("     • 3d: Beautiful cloverleaf patterns")
        aprint("")
        aprint("  Colors: Red (high probability) → Blue (low probability)")
        aprint("")
        aprint("What you're seeing:")
        aprint("  • Probability density |ψ|² (where electron likely is)")
        aprint("  • NOT electron trajectories (quantum mechanics has no orbits!)")
        aprint("  • Cloudy appearance shows probability distribution")
        aprint("  • Shapes determined by quantum numbers n, l, m")
        aprint("")
        aprint("Fun fact: These exact shapes explain chemistry!")
        aprint("  • s orbitals → spherical bonds")
        aprint("  • p orbitals → directional bonds (sp³ hybridization)")
        aprint("  • d orbitals → transition metal chemistry")
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
