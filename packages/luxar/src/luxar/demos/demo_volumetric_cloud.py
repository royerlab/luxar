#!/usr/bin/env python3
"""Self-Contained Demo: Volumetric Cloud with Fractal Density

This demo demonstrates:
- Volumetric cloud-like structures using fractal noise
- Multi-scale density for realistic wisps and puffs
- Varying point sizes based on local density
- Soft, cloud-like appearance (low sharpness)
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all generation code is in this file,
including a simple Perlin-like noise implementation.

Mathematical Background:
    Clouds have fractal structure - large puffs contain smaller puffs, which
    contain even smaller wisps. This is created using multi-octave noise where
    each octave adds detail at a different scale. Point sizes vary with density
    to create soft, natural boundaries.

Usage:
    python demo_volumetric_cloud.py [--points=N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "cloud",
    "title": "Cloud",
    "description": "A volumetric cloud from multi-octave fractal noise, rendered as soft points.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["cloud"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def simple_noise_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Simple 3D noise function (Perlin-like) - completely self-contained.

    This is a simplified noise implementation that creates smooth, organic-looking
    variations suitable for cloud density. Not cryptographically secure!

    Args:
        x, y, z: Coordinate arrays (same shape)
        seed: Random seed for reproducibility

    Returns:
        Noise values in approximate range [-1, 1]
    """
    # This creates smooth interpolated noise from grid coordinates.
    # Note: this is a deterministic positional hash (see hash_coords below) - it
    # does not touch the global RNG, so no np.random.seed() call is needed here.

    # Get integer grid coordinates
    xi = np.floor(x).astype(int)
    yi = np.floor(y).astype(int)
    zi = np.floor(z).astype(int)

    # Fractional parts for interpolation
    xf = x - xi
    yf = y - yi
    zf = z - zi

    # Smooth interpolation (fade function: 6t^5 - 15t^4 + 10t^3)
    u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
    v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
    w = zf * zf * zf * (zf * (zf * 6 - 15) + 10)

    # Generate pseudo-random gradients for cube corners
    # Using a simple hash function based on position
    def hash_coords(xi, yi, zi):  # type: ignore[no-untyped-def]
        # Simple position-based hash (not secure, but good for graphics)
        h = (xi * 374761393 + yi * 668265263 + zi * 1274126177 + seed) & 0x7FFFFFFF
        return (h % 1000000) / 500000.0 - 1.0

    # Get gradient values at cube corners (8 corners)
    # This is simplified - real Perlin uses gradient vectors
    n000 = hash_coords(xi, yi, zi)
    n001 = hash_coords(xi, yi, zi + 1)
    n010 = hash_coords(xi, yi + 1, zi)
    n011 = hash_coords(xi, yi + 1, zi + 1)
    n100 = hash_coords(xi + 1, yi, zi)
    n101 = hash_coords(xi + 1, yi, zi + 1)
    n110 = hash_coords(xi + 1, yi + 1, zi)
    n111 = hash_coords(xi + 1, yi + 1, zi + 1)

    # Trilinear interpolation
    # Interpolate along x
    nx00 = n000 * (1 - u) + n100 * u
    nx01 = n001 * (1 - u) + n101 * u
    nx10 = n010 * (1 - u) + n110 * u
    nx11 = n011 * (1 - u) + n111 * u

    # Interpolate along y
    nxy0 = nx00 * (1 - v) + nx10 * v
    nxy1 = nx01 * (1 - v) + nx11 * v

    # Interpolate along z
    nxyz = nxy0 * (1 - w) + nxy1 * w

    return nxyz  # type: ignore[no-any-return]


def fractal_noise_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    octaves: int = 4,
    persistence: float = 0.5,
) -> np.ndarray:
    """Multi-octave fractal noise - completely self-contained.

    Combines multiple scales of noise for natural, fractal-like detail.

    Args:
        x, y, z: Coordinate arrays
        octaves: Number of noise octaves (scales)
        persistence: How much each octave contributes (< 1.0 for fading detail)

    Returns:
        Fractal noise values
    """
    noise = np.zeros_like(x)
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for octave in range(octaves):
        # Add noise at this frequency
        noise += (
            simple_noise_3d(x * frequency, y * frequency, z * frequency, seed=octave)
            * amplitude
        )

        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= 2.0

    # Normalize to approximately [-1, 1]
    return noise / max_amplitude  # type: ignore[no-any-return]


def generate_volumetric_cloud(
    output_path: Path,
    n_candidate_points: int = 1000000,
    cloud_size: float = 20.0,
    density_threshold: float = 0.25,
    min_radius: float = 0.03,
    max_radius: float = 0.5,
) -> None:
    """Generate a volumetric cloud with fractal density.

    This function contains ALL the generation logic - completely self-contained.

    The cloud is created by:
    1. Sampling random points in a volume
    2. Computing fractal noise density at each point
    3. Applying 3D Gaussian falloff (spherical cloud shape)
    4. Filtering out low-density regions (creates gaps and wisps)
    5. Sizing points based on local density (large in dense areas)
    6. Using low sharpness for soft, cloud-like appearance

    Args:
        output_path: Where to write the zarr store
        n_candidate_points: Initial point candidates (many will be filtered out)
        cloud_size: Approximate cloud diameter
        density_threshold: Minimum density to keep points (0-1, higher = wispier)
        min_radius: Minimum point radius
        max_radius: Maximum point radius (in dense regions)
    """
    with asection(f"Generating Volumetric Cloud ({n_candidate_points:,} candidates)"):
        aprint(f"Cloud size: {cloud_size} units")
        aprint(f"Density threshold: {density_threshold} (higher = wispier)")
        aprint(f"Point radius range: [{min_radius}, {max_radius}]")

        # === STEP 1: Generate candidate points in volume ===
        aprint("\nGenerating candidate points in 3D volume...")

        # Random points in a cube (will filter to create cloud shape)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        positions_raw = rng.uniform(
            -cloud_size / 2, cloud_size / 2, (n_candidate_points, 3)
        ).astype(np.float32)

        aprint(f"✓ Generated {n_candidate_points:,} candidate positions")

        # === STEP 2: Calculate fractal density with turbulence ===
        aprint("Calculating fractal noise density (7 octaves + turbulence)...")

        x, y, z = positions_raw[:, 0], positions_raw[:, 1], positions_raw[:, 2]

        # Add turbulence (distortion) for wispy structure
        # This creates the characteristic cloud wisps and tendrils
        turbulence_x = fractal_noise_3d(
            x / cloud_size * 2,
            y / cloud_size * 2,
            z / cloud_size * 2,
            octaves=3,
            persistence=0.5,
        )
        turbulence_y = fractal_noise_3d(
            x / cloud_size * 2 + 100,
            y / cloud_size * 2 + 100,
            z / cloud_size * 2,
            octaves=3,
            persistence=0.5,
        )

        # Apply turbulence to create wispy distortions
        noise_density = fractal_noise_3d(
            (x + turbulence_x * cloud_size * 0.3) / cloud_size,
            (y + turbulence_y * cloud_size * 0.3) / cloud_size,
            z / cloud_size,
            octaves=7,
            persistence=0.6,
        )

        # Normalize noise to [0, 1]
        noise_density = (noise_density + 1.0) / 2.0

        # Apply power function to increase contrast (more dramatic density variation)
        noise_density = np.power(noise_density, 1.5)  # Emphasize dense areas

        aprint("✓ Generated 7-octave fractal noise with turbulence")

        # === STEP 3: Apply 3D Gaussian falloff (spherical cloud shape) ===
        aprint("Applying Gaussian falloff for puff shape...")

        # Distance from center
        r = np.sqrt(x**2 + y**2 + z**2)

        # Gaussian falloff: density decreases from center
        # Using exp(-r²/σ²) where σ controls puff size
        sigma = cloud_size / 3.0  # Cloud extends ~3σ
        gaussian_falloff = np.exp(-(r**2) / (2 * sigma**2))

        # Combine noise and falloff
        # This creates a puff shape with fractal internal structure
        combined_density = noise_density * gaussian_falloff

        aprint(f"✓ Applied Gaussian falloff (σ={sigma:.1f})")

        # === STEP 4: Filter by density threshold ===
        aprint(f"\nFiltering points (threshold={density_threshold})...")

        # Keep only points above threshold (creates wisps and gaps)
        mask = combined_density > density_threshold
        positions = positions_raw[mask]
        density_values = combined_density[mask]

        # Check if we have any points left
        if len(positions) == 0:
            aprint(
                f"⚠️  WARNING: Threshold {density_threshold} filtered out ALL points!"
            )
            aprint("   Lowering threshold to 0.25 and retrying...")
            density_threshold = 0.25
            mask = combined_density > density_threshold
            positions = positions_raw[mask]
            density_values = combined_density[mask]

        if len(positions) == 0:
            raise ValueError(
                f"No points remain after filtering! Try lowering density_threshold (current: {density_threshold})"
            )

        aprint(
            f"✓ Kept {len(positions):,} points ({len(positions) / n_candidate_points * 100:.1f}% of candidates)"
        )
        aprint("  This creates the wispy, cloud-like structure")

        # === STEP 5: Size points based on density ===
        aprint("Assigning point sizes based on local density...")

        # Normalize density values to [0, 1] for the kept points
        density_norm = (density_values - density_values.min()) / (
            density_values.max() - density_values.min() + 1e-8
        )

        # Map density to radius with more dramatic variation
        # Use power function to make size differences more pronounced
        size_factor = np.power(
            density_norm, 0.7
        )  # Emphasize larger points in dense areas
        radii = min_radius + size_factor * (max_radius - min_radius)

        # Add some randomness to sizes for more natural look
        size_noise = rng.uniform(0.85, 1.15, len(positions))
        radii = radii * size_noise
        radii = np.clip(radii, min_radius, max_radius).astype(np.float32)

        aprint(f"✓ Point sizes: {min_radius} to {max_radius} units (with variation)")
        aprint(f"  Mean radius: {np.mean(radii):.3f}")
        aprint(f"  Std dev: {np.std(radii):.3f}")
        aprint("  Dramatic size variation for volumetric depth")

        # === STEP 6: Create soft, cloud-like colors ===
        aprint("Generating cloud colors...")

        # Base color: white/light gray (high brightness for visibility)
        # Add subtle variation based on position and density
        base_brightness = 0.6 + density_norm * 0.4  # Range: 0.6 to 1.0 (visible!)

        # Add very subtle color tint (slight blue/gray variation)
        # This gives depth and makes it less flat
        color_variation = rng.uniform(-0.1, 0.1, len(positions))

        colors = np.zeros((len(positions), 3), dtype=np.float32)
        colors[:, 0] = np.clip(base_brightness + color_variation, 0, 1)  # R
        colors[:, 1] = np.clip(base_brightness + color_variation * 0.5, 0, 1)  # G
        colors[:, 2] = np.clip(
            base_brightness + color_variation * 1.5, 0, 1
        )  # B (slight blue tint)

        aprint("✓ Generated white/gray colors with subtle variation")

        # === STEP 7: Very soft sharpness for cloud-like appearance ===
        # Clouds are extremely soft and fluffy, not sharp at all
        # sharpness is a normalized [0, 1] knob: low values -> peakier/softer cusp
        # Inverse relationship: denser areas are actually SOFTER (more diffuse light scattering)
        sharpness = 0.2 + (1.0 - density_norm) * 0.15  # Range: 0.2 to 0.35 (very soft!)

        # Add slight randomness for natural variation
        sharpness_noise = rng.uniform(0.9, 1.1, len(positions))
        sharpness = sharpness * sharpness_noise
        sharpness = np.clip(sharpness, 0.15, 0.4).astype(np.float32)

        aprint("✓ Very soft sharpness: 0.2 to 0.35 (cloud-like softness)")
        aprint(f"  Mean: {np.mean(sharpness):.2f} (softer than default)")
        aprint("  Inverse to density (dense areas softer = more diffuse)")

    # === STEP 8: Write to Zarr ===
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

            scene.add_points(
                "VolumetricCloud",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                opacity=0.95,  # Slightly transparent
                blending_mode="additive",  # Clouds accumulate light
                intensity=0.016,
            )

            # Overlay annotations
            scene.add_text(
                "Volumetric Cloud",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Fractal noise density",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written {len(positions):,} points to {output_path}")
        aprint(
            f"✓ Dataset size: ~{len(positions) * 40 / 1024 / 1024:.1f} MB (uncompressed)"
        )


def main() -> None:
    """Main demo entry point."""
    # Parse simple command line args (optional)
    n_candidate_points = 800000  # Start with many candidates (will be filtered)
    if len(sys.argv) > 1 and sys.argv[1].startswith("--points="):
        n_candidate_points = int(sys.argv[1].split("=")[1])

    aprint("=" * 70)
    aprint("VOLUMETRIC CLOUD DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a realistic cloud with fractal structure")
    aprint(f"Candidate points: {n_candidate_points:,} (will be filtered by density)")
    aprint("Features:")
    aprint("  - 7-octave fractal noise for detail at multiple scales")
    aprint("  - Turbulence/curl for wispy tendrils and structure")
    aprint("  - Dramatic size variation (0.03 to 0.5 units)")
    aprint("  - Very soft appearance (sharpness 0.2-0.35)")
    aprint("  - Dense areas are SOFTER (diffuse light scattering)")
    aprint("  - Higher density threshold for wispy, irregular boundaries")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "cloud.luxar.zarr"
        generate_volumetric_cloud(output_path, n_candidate_points=n_candidate_points)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_cloud_") as tmpdir:
        output_path = Path(tmpdir) / "cloud.luxar.zarr"

        # Generate the dataset (all code in this file!)
        generate_volumetric_cloud(output_path, n_candidate_points=n_candidate_points)

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")
        aprint("VIEWING TIPS:")
        aprint("   - Rotate slowly to appreciate the 3D volumetric structure")
        aprint("   - Notice fractal detail: large puffs contain smaller wisps")
        aprint("   - Very soft edges create realistic cloud appearance")
        aprint("   - Point sizes vary dramatically (10x range) for depth")
        aprint("   - Dense areas appear softer (inverse sharpness)")
        aprint("   - Turbulence creates wispy tendrils and irregular shape")
        aprint("   - Internal gaps from high density threshold")
        aprint("   - Try zooming in to see individual 'cloud particles'")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically
    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
