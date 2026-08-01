#!/usr/bin/env python3
"""Self-Contained Demo: Mandelbulb Fractal Visualization

This demo demonstrates:
- Volumetric representation of the famous Mandelbulb 3D fractal
- Distance estimation for surface detection
- Iteration-based coloring for visual depth
- Adaptive point sizing based on detail level
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all Mandelbulb computation code is in this file.

Mathematical Background:
    The Mandelbulb is a 3D extension of the Mandelbrot set. For each point in 3D space,
    we iterate the formula z → z^n + c (where n=8 is standard) in spherical coordinates.
    Points that don't escape to infinity are part of the fractal. We use distance
    estimation to find points near the surface for volumetric rendering.

    Formula: Convert (x,y,z) to spherical (r,θ,φ), then:
        r_new = r^n
        θ_new = n × θ
        φ_new = n × φ

    After each iteration, add the original point c and check for escape.

Usage:
    python demo_mandelbulb.py [--resolution=N] [--power=8]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
    - Try different powers (6, 8, 9) for different shapes!
"""

DEMO_META = {
    "key": "mandelbulb",
    "title": "Mandelbulb Fractal",
    "description": "Volumetric Mandelbulb 3D fractal via distance estimation, iteration-colored surface points.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["mandelbulb"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def mandelbulb_distance_estimate(
    points: np.ndarray,
    power: int = 8,
    max_iterations: int = 128,
    bailout: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Mandelbulb distance estimate and iteration count.

    This function contains the complete Mandelbulb algorithm - self-contained.

    Uses the triplex algebra formulation for stable computation.

    Args:
        points: Array of shape (N, 3) with (x, y, z) coordinates
        power: Mandelbulb power (8 is classic, 6 and 9 also interesting)
        max_iterations: Maximum iterations before considering point bounded
        bailout: Escape radius (points with r > bailout have escaped)

    Returns:
        Tuple of (distances, iterations) where:
        - distances: Estimated distance to fractal surface (lower = closer)
        - iterations: Number of iterations before escape (higher = deeper in set)
    """
    n_points = points.shape[0]

    # Initialize
    z = points.copy()  # Start at the point itself
    dr = np.ones(n_points)  # Derivative for distance estimation
    r = np.sqrt(np.sum(z**2, axis=1))

    iterations = np.zeros(n_points, dtype=np.int32)
    escaped = np.zeros(n_points, dtype=bool)

    for i in range(max_iterations):
        # Only iterate points that haven't escaped
        active = ~escaped
        if not np.any(active):
            break

        r_active = r[active]
        z_active = z[active]
        dr_active = dr[active]

        # Convert to spherical coordinates
        # r is already calculated
        theta = np.arctan2(
            np.sqrt(z_active[:, 0] ** 2 + z_active[:, 1] ** 2), z_active[:, 2]
        )
        phi = np.arctan2(z_active[:, 1], z_active[:, 0])

        # Calculate derivative (for distance estimation)
        # dr = power * r^(power-1) * dr + 1
        dr_active = power * (r_active ** (power - 1)) * dr_active + 1.0

        # Mandelbulb formula: z^power
        r_new = r_active**power
        theta_new = theta * power
        phi_new = phi * power

        # Convert back to Cartesian and add original point c
        z_active_new = np.zeros_like(z_active)
        z_active_new[:, 0] = (
            r_new * np.sin(theta_new) * np.cos(phi_new) + points[active, 0]
        )
        z_active_new[:, 1] = (
            r_new * np.sin(theta_new) * np.sin(phi_new) + points[active, 1]
        )
        z_active_new[:, 2] = r_new * np.cos(theta_new) + points[active, 2]

        # Update active points
        z[active] = z_active_new
        dr[active] = dr_active
        r[active] = np.sqrt(np.sum(z_active_new**2, axis=1))

        # Check for escape
        newly_escaped = (r > bailout) & active
        escaped[newly_escaped] = True
        iterations[active] += 1

    # Distance estimate: 0.5 * r * log(r) / |dr|
    # Clamp dr to avoid division by very small numbers
    dr_clamped = np.maximum(dr, 1e-6)
    distances = 0.5 * r * np.log(np.maximum(r, 1e-6)) / dr_clamped

    return distances, iterations


def generate_mandelbulb_volumetric(
    output_path: Path,
    resolution: int = 100,
    power: int = 8,
    max_distance: float = 0.01,
    seed: int = 0,
) -> int:
    """Generate volumetric Mandelbulb fractal.

    This function contains ALL the generation logic - completely self-contained.

    Args:
        output_path: Where to write the zarr store
        resolution: Grid resolution per axis (100 = 100^3 = 1M samples)
        power: Mandelbulb power (6, 8, or 9 recommended)
        max_distance: Maximum distance to include (controls surface thickness)
        seed: Seed for the anti-aliasing jitter RNG (reproducible output)

    Returns:
        Number of points in final dataset
    """
    with asection(f"Computing Mandelbulb (power={power}, resolution={resolution}³)"):
        aprint(f"Sampling {resolution**3:,} points in 3D grid...")

        # Create 3D sampling grid centered at origin
        # Mandelbulb is typically bounded in [-1.2, 1.2]³
        coord_range = 1.3
        coords = np.linspace(-coord_range, coord_range, resolution)

        # Create meshgrid
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

        # Flatten to (N, 3) array
        sample_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Add small jitter to avoid aliasing artifacts from regular grid
        grid_spacing = coords[1] - coords[0]
        jitter_amount = grid_spacing * 0.3  # 30% of grid spacing
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-jitter_amount, jitter_amount, sample_points.shape).astype(
            np.float32
        )
        sample_points += jitter

        aprint(f"✓ Created sampling grid: {resolution}×{resolution}×{resolution}")
        aprint(f"  Applied jitter: ±{jitter_amount:.4f} units (30% of grid spacing)")

        # Compute Mandelbulb distance for all points
        aprint(f"Computing Mandelbulb iterations (power={power})...")
        distances, iterations = mandelbulb_distance_estimate(
            sample_points,
            power=power,
        )

        aprint(f"✓ Computed distance estimates for {len(sample_points):,} points")

        # Filter to points near the surface
        # Keep points with distance estimate below threshold
        near_surface = (distances < max_distance) & (distances > 0)

        positions = sample_points[near_surface].astype(np.float32)
        surface_distances = distances[near_surface]
        surface_iterations = iterations[near_surface]

        aprint(f"✓ Found {len(positions):,} points near fractal surface")
        aprint(f"  Surface density: {len(positions) / resolution**3 * 100:.2f}%")

        if len(positions) == 0:
            aprint("⚠️  No points found! Try increasing max_distance or resolution")
            return 0

        # Generate colors based on iteration count (rainbow gradient)
        aprint("Generating colors from iteration count...")

        # Normalize iterations to [0, 1]
        iter_normalized = surface_iterations / surface_iterations.max()

        # Create rainbow using HSV-like approach
        hue = iter_normalized  # Hue varies with iteration

        # Convert hue to RGB (simplified HSV to RGB)
        colors = np.zeros((len(positions), 3), dtype=np.float32)
        colors[:, 0] = np.abs(np.sin(2 * np.pi * hue))  # Red
        colors[:, 1] = np.abs(np.sin(2 * np.pi * hue + 2 * np.pi / 3))  # Green
        colors[:, 2] = np.abs(np.sin(2 * np.pi * hue + 4 * np.pi / 3))  # Blue

        aprint("✓ Generated rainbow gradient based on fractal depth")

        # Generate radii based on distance (closer to surface = smaller points)
        aprint("Calculating adaptive point sizes...")

        # Points closer to surface get smaller radii for detail
        # Map distance [0, max_distance] to radii [0.005, 0.02]
        distance_normalized = np.clip(surface_distances / max_distance, 0, 1)
        radii = (0.005 + distance_normalized * 0.015).astype(np.float32)

        # Use moderate sharpness for fractal edges (normalized [0, 1] knob; 0.5 = Gaussian)
        sharpnesses = np.full(len(positions), 0.5, dtype=np.float32)

        aprint(f"✓ Radii range: [{radii.min():.4f}, {radii.max():.4f}]")

    # Write to Zarr
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

            # Volumetric emission–absorption (VOLUMETRIC_BLENDING_SPEC.md
            # phase 3): the dense fractal shell self-occludes instead of
            # blowing out, so the historical additive anti-blowout
            # workarounds (colors ×= 0.1 AND intensity = 0.0625) are gone
            # — real depth cueing rather than a flat-clipped glow.
            #
            # κ = 8 (inside the panel slider range) is what leaves the
            # surface glowing: higher κ over-self-screens toward a dim
            # solid, lower κ loses the depth cue.
            #
            # intensity = 0.025 is the EXPOSURE-NEUTRAL authoring: the
            # scene reads correctly at the viewer's default exposure of
            # 0 EV, so no one has to dial the HDR exposure down to see
            # it. The earlier 0.5 needed −4.32 EV in the panel (the
            # comment's "just under white" claim did not survive
            # measurement — a deep ray saturates ≈ 20× over white at
            # this density), and 0.5 × 2**−4.32 = 0.025. The mapping is
            # exact, not a guess: emitted radiance is linear in
            # intensity (the point shader multiplies the colour by
            # uIntensity; optical depth τ depends only on κ, opacity and
            # radius) while exposure multiplies the composited frame by
            # 2**EV, so this is the SAME image the −4.32 EV panel gave.
            # Authoring it here instead of leaving it to the exposure
            # slider also puts the bloom threshold and the detector-noise
            # sigmas back in their intended range relative to white.
            scene.add_points(
                "Mandelbulb",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.9,
                blending_mode="volumetric",
                absorption=8.0,
                intensity=0.025,
            )

            # Overlay annotations
            scene.add_text(
                "Mandelbulb Fractal",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Power 8 \u2022 Distance estimation",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{len(positions) * 40 / 1024 / 1024:.1f} MB")

    return len(positions)


def main() -> None:
    """Main demo entry point."""
    # Parse command line arguments
    resolution = 256  # Default: 256^3 = 16.8M samples → ~1.2M surface points
    power = 8  # Classic Mandelbulb

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--resolution="):
                resolution = int(arg.split("=")[1])
            elif arg.startswith("--power="):
                power = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("MANDELBULB FRACTAL DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a volumetric Mandelbulb fractal (3D Mandelbrot set)")
    aprint(f"Resolution: {resolution}³ = {resolution**3:,} samples")
    aprint(f"Power: {power} (classic=8, try 6 or 9 for variations!)")
    aprint("")
    aprint("The algorithm:")
    aprint("  1. Sample 3D space on a grid")
    aprint("  2. For each point, iterate the Mandelbulb formula")
    aprint("  3. Estimate distance to fractal surface")
    aprint("  4. Keep points near the surface")
    aprint("  5. Color by iteration depth (rainbow gradient)")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "mandelbulb.luxar.zarr"
        n_points = generate_mandelbulb_volumetric(
            output_path, resolution=resolution, power=power
        )
        if n_points == 0:
            aprint("\n❌ No points generated - try different parameters")
            return
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_mandelbulb_") as tmpdir:
        output_path = Path(tmpdir) / "mandelbulb.luxar.zarr"

        # Generate the fractal
        n_points = generate_mandelbulb_volumetric(
            output_path, resolution=resolution, power=power
        )

        if n_points == 0:
            aprint("\n❌ No points generated - try different parameters")
            return

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("Once the viewer opens:")
        aprint("  • Use mouse to rotate and explore the fractal")
        aprint("  • Zoom in to see fine details and tendrils")
        aprint("  • Colors show iteration depth (structure complexity)")
        aprint("  • The fractal has infinite detail at all scales!")
        aprint("")
        aprint("Try different powers:")
        aprint("  --power=6  → Rounder, more bulbous")
        aprint("  --power=8  → Classic Mandelbulb (default)")
        aprint("  --power=9  → Sharper, more spiky")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically when TemporaryDirectory context exits
    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
