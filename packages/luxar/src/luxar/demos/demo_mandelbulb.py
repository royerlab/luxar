#!/usr/bin/env python3
"""Self-Contained Demo: Mandelbulb Fractal Visualization

This demo demonstrates:
- Volumetric representation of the famous Mandelbulb 3D fractal
- Distance estimation for surface detection
- Orbit-trap coloring for visual structure
- Distance-field ambient occlusion and baked key lighting
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
    "description": "Volumetric Mandelbulb with orbit-trap colors and distance-field shading.",
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
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.colormaps import scalars_to_colors
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._lod_policy import stream_ladder
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
    distances, iterations, _ = _mandelbulb_distance_core(
        points,
        power=power,
        max_iterations=max_iterations,
        bailout=bailout,
        track_orbit_trap=False,
    )
    return distances, iterations


def _mandelbulb_distance_and_orbit_trap(
    points: np.ndarray,
    power: int = 8,
    max_iterations: int = 128,
    bailout: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance, escape iterations, and a plane orbit trap."""
    distances, iterations, orbit_trap = _mandelbulb_distance_core(
        points,
        power=power,
        max_iterations=max_iterations,
        bailout=bailout,
        track_orbit_trap=True,
    )
    if orbit_trap is None:
        raise RuntimeError("Orbit trap tracking unexpectedly disabled")
    return distances, iterations, orbit_trap


def _mandelbulb_distance_core(
    points: np.ndarray,
    power: int,
    max_iterations: int,
    bailout: float,
    *,
    track_orbit_trap: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run the Mandelbulb escape loop with optional orbit-trap tracking."""
    n_points = points.shape[0]

    # Initialize
    z = points.copy()  # Start at the point itself
    dr = np.ones(n_points)  # Derivative for distance estimation
    r = np.sqrt(np.sum(z**2, axis=1))

    iterations = np.zeros(n_points, dtype=np.int32)
    escaped = np.zeros(n_points, dtype=bool)
    orbit_trap = np.abs(z[:, 0]).copy() if track_orbit_trap else None

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
        if orbit_trap is not None:
            # A plane orbit trap records the closest approach of each orbit to
            # x=0. Unlike escape iterations, it varies across nearby surface
            # points and exposes the Mandelbulb's internal folds.
            active_indices = np.flatnonzero(active)
            bounded = r[active] <= bailout
            bounded_indices = active_indices[bounded]
            orbit_trap[bounded_indices] = np.minimum(
                orbit_trap[bounded_indices], np.abs(z_active_new[bounded, 0])
            )

        # Check for escape
        newly_escaped = (r > bailout) & active
        escaped[newly_escaped] = True
        iterations[active] += 1

    # Distance estimate: 0.5 * r * log(r) / |dr|
    # Clamp dr to avoid division by very small numbers
    dr_clamped = np.maximum(dr, 1e-6)
    distances = 0.5 * r * np.log(np.maximum(r, 1e-6)) / dr_clamped

    return distances, iterations, orbit_trap


def _equalize_orbit_trap(orbit_trap: np.ndarray) -> np.ndarray:
    """Map an orbit trap to a tie-preserving empirical CDF in [0, 1]."""
    if orbit_trap.size == 0:
        return np.empty(0, dtype=np.float32)

    _, inverse, counts = np.unique(orbit_trap, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    quantiles = (cumulative - 0.5 * counts) / orbit_trap.size
    return quantiles[inverse].astype(np.float32)


def _mandelbulb_surface_normals(
    positions: np.ndarray,
    *,
    power: int,
    epsilon: float,
) -> np.ndarray:
    """Estimate outward normals from central differences of the DE field."""
    gradients = np.empty_like(positions, dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = epsilon
        distance_plus, _ = mandelbulb_distance_estimate(positions + offset, power=power)
        distance_minus, _ = mandelbulb_distance_estimate(
            positions - offset, power=power
        )
        gradients[:, axis] = distance_plus - distance_minus

    lengths = np.linalg.norm(gradients, axis=1)
    normals = np.zeros_like(gradients)
    valid = lengths > 1e-12
    normals[valid] = gradients[valid] / lengths[valid, None]

    if np.any(~valid):
        invalid_indices = np.flatnonzero(~valid)
        radial = positions[~valid]
        radial_lengths = np.linalg.norm(radial, axis=1)
        radial_valid = radial_lengths > 1e-12
        normals[invalid_indices[radial_valid]] = (
            radial[radial_valid] / radial_lengths[radial_valid, None]
        )

    return normals


def _mandelbulb_ambient_occlusion(
    positions: np.ndarray,
    surface_distances: np.ndarray,
    normals: np.ndarray,
    *,
    power: int,
    step: float,
) -> np.ndarray:
    """Bake directional distance-field AO along each outward normal.

    The distance field makes occlusion directional and independent of point
    sampling density; a KD-tree neighbour count cannot distinguish a flat sheet
    from a crevice with the same local point count.
    """
    occlusion = np.zeros(len(positions), dtype=np.float64)
    total_weight = 0.0
    # Four full-depth taps preserve the reference lighting (mean delta 0.0014,
    # p99 0.0073 at 48^3); the fifth tap costs another DE pass for no visible gain.
    for index in range(4):
        distance_along_normal = step * 2**index
        sample_positions = positions + normals * distance_along_normal
        sampled_distances, _ = mandelbulb_distance_estimate(
            sample_positions, power=power
        )
        expected_distances = surface_distances + distance_along_normal
        deficit = np.clip(
            (expected_distances - sampled_distances) / distance_along_normal,
            0.0,
            1.0,
        )
        weight = 0.5**index
        occlusion += deficit * weight
        total_weight += weight

    # This DE is a lower bound rather than a unit-gradient SDF, so even open-space
    # probes retain a 0.13-0.15 deficit. Strength 0.85 and floor 0.15 preserve
    # crevice contrast; the authored display range absorbs the global darkening.
    return np.clip(1.0 - 0.85 * occlusion / total_weight, 0.15, 1.0)


def _mandelbulb_surface_appearance(
    positions: np.ndarray,
    surface_distances: np.ndarray,
    orbit_trap: np.ndarray,
    *,
    power: int = 8,
    max_distance: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create orbit-trap colours with baked key light and distance-field AO."""
    if len(positions) == 0:
        empty_colors = np.empty((0, 3), dtype=np.float32)
        empty_scalars = np.empty(0, dtype=np.float32)
        return empty_colors, empty_scalars, empty_scalars

    trap_quantiles = _equalize_orbit_trap(orbit_trap)
    base_colors = scalars_to_colors(trap_quantiles, "magma", vmin=0.0, vmax=1.0)

    normals = _mandelbulb_surface_normals(
        positions,
        power=power,
        epsilon=max_distance * 0.5,
    )
    ambient_occlusion = _mandelbulb_ambient_occlusion(
        positions,
        surface_distances,
        normals,
        power=power,
        step=max_distance,
    )

    # Deliberately bake a world-fixed key into emissive point colour: it reveals
    # the upper/front folds but does not follow the camera during the orbit.
    key_direction = np.array([-0.45, -0.35, 0.82])
    key_direction /= np.linalg.norm(key_direction)
    lambert = np.clip(normals @ key_direction, 0.0, 1.0)
    # Ambient 0.32 keeps the far side legible while retaining directional relief.
    lighting = (0.32 + 0.68 * lambert) * ambient_occlusion
    colors = np.clip(base_colors * lighting[:, None], 0.0, 1.0).astype(np.float32)
    return colors, lighting.astype(np.float32), ambient_occlusion.astype(np.float32)


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
        distances, _, orbit_trap = _mandelbulb_distance_and_orbit_trap(
            sample_points,
            power=power,
        )

        aprint(f"✓ Computed distance estimates for {len(sample_points):,} points")

        # Filter to points near the surface
        # Keep points with distance estimate below threshold
        near_surface = (distances < max_distance) & (distances > 0)

        positions = sample_points[near_surface].astype(np.float32)
        surface_distances = distances[near_surface]
        surface_orbit_trap = orbit_trap[near_surface]

        aprint(f"✓ Found {len(positions):,} points near fractal surface")
        aprint(f"  Surface density: {len(positions) / resolution**3 * 100:.2f}%")

        if len(positions) == 0:
            aprint("⚠️  No points found! Try increasing max_distance or resolution")
            return 0

        aprint("Generating orbit-trap colors and distance-field shading...")
        colors, lighting, ambient_occlusion = _mandelbulb_surface_appearance(
            positions,
            surface_distances,
            surface_orbit_trap,
            power=power,
            max_distance=max_distance,
        )
        aprint("✓ Applied magma orbit-trap palette with baked key light and DE AO")
        aprint(
            f"  Lighting range: [{lighting.min():.3f}, {lighting.max():.3f}], "
            f"AO range: [{ambient_occlusion.min():.3f}, {ambient_occlusion.max():.3f}]"
        )

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
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=ViewerConfig(cinematic_mode=True)
            )

            # Volumetric emission–absorption (VOLUMETRIC_BLENDING_SPEC.md
            # phase 3): the dense fractal shell self-occludes instead of
            # blowing out, so the historical additive anti-blowout
            # workarounds (colors ×= 0.1 AND intensity = 0.0625) are gone
            # — real depth cueing rather than a flat-clipped glow.
            #
            # κ = 0.04 is what leaves the surface glowing: higher κ
            # over-self-screens toward a dim solid, lower κ loses the depth
            # cue. (κ was 8 before the 2026-08-02 ray-mass unification, when
            # the point shader still multiplied τ by a world radius; κ is now
            # dimensionless and comparable across geometry types, so the
            # numeric value changed while the render did not.)
            #
            # The display range and κ below were tuned in the viewer's Layers
            # panel and baked back — that is what `layer=True` is for.
            #
            # intensity is the EXPOSURE-NEUTRAL authoring: the
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
                # Tuned in the Layers panel and baked back. The mechanical
                # conversion for the ray-mass unification (kappa * r_mean *
                # chord = 0.083) was only a starting point — it preserves total
                # optical depth but not the old per-point weighting, where a
                # bigger point absorbed proportionally more. 0.04 is the value
                # that actually reads right.
                absorption=0.04,
                # The darker baked lighting needs a tighter display range than
                # the old unshaded palette. The window maps to the
                # shader uniforms as intensity = 1/(max-min), offset =
                # -min/(max-min) (rendering/display-range.ts::computeUniforms),
                # so a min of 0 leaves offset at its identity and the max is
                # simply 1/intensity. A max of 26 preserves the new shadow
                # range. The final gallery framing selected a +1.53 EV adjustment to
                # converge on its exposure target; that is not residual clipping
                # or an uncompensated authoring offset.
                intensity=1.0 / 26.0,
                additive_lod=stream_ladder(len(positions)),
                # Expose the node in the viewer's Layers panel so the
                # appearance above is live-tunable — in volumetric mode the
                # panel shows the Absorption (kappa) slider alongside opacity /
                # display range / gamma / blend, which is how these values were
                # arrived at in the first place.
                layer=True,
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
            add_demo_caption(
                scene, "Power 8 \u2022 Distance estimation", DEMO_META.get("citation")
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
    aprint("  5. Color by orbit traps and bake distance-field shading")
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
        aprint("  • Orbit-trap colors reveal the fractal's internal structure")
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
