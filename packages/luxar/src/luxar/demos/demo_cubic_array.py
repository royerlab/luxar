#!/usr/bin/env python3
"""Self-Contained Demo: Cubic Array with Depth-of-Field Visualization

This demo demonstrates:
- Creating a dense 100³ cubic array of points (1,000,000 points total)
- Color gradient indicating depth for better visualization
- Sharp disc-like points optimized for depth-of-field testing
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all generation code is in this file.
It uses the luxar CLI for serving, which handles server lifecycle automatically.

Usage:
    python demo_cubic_array.py

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "cubic_array",
    "title": "Cubic Array with Depth-of-Field Visualization",
    "description": "A dense 100³ cubic lattice of 1M depth-colored points on a star field, for depth-of-field testing.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["cubic_array"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._lod_policy import stream_ladder
from luxar.utils.paths import get_demos_output_dir

# Opening framing. The star field deliberately spans +/-500 units while the
# lattice itself is only ~50 across, so the default whole-scene camera fit backs
# off far enough to contain the STARS and leaves the cube — the actual subject —
# as a speck in the middle of the frame. The scene therefore authors its own
# opening camera, solved from the lattice's own size.
#
# The distance is derived rather than hardcoded so it tracks `grid_size` and
# `spacing`: a body of bounding-sphere radius R seen from distance d subtends a
# half-angle asin(R/d). Asking for CAMERA_FILL of the FULL field of view (rather
# than the 0.5 that would exactly circumscribe it) lets the near corners of the
# lattice reach the edges of the frame.
#
# The solve is done AT the cinematic preset's own 35 mm lens
# (`CINEMATIC_FOV_DEG`), so `fov` is deliberately NOT pinned on the
# CameraConfig: pinning it would make the preset leave the lens alone as a unit
# and there would be nothing left for the preset to contribute. Deriving the
# distance from the preset's own constant is the house convention — see
# `demos/_cinematic_camera.py`, and the `test_demos_cinematic_mode` gate.
CAMERA_FOV_DEG = CINEMATIC_FOV_DEG
CAMERA_FILL = 0.58
# View direction, in units of the orbit distance. A three-quarter view so three
# faces of the lattice are visible and it reads as a cube rather than a square.
CAMERA_DIRECTION = (1.0, 0.62, 1.0)

# Auto-rotation, on from the first frame. 0.5 — twice the viewer's own
# presentation default — chosen live against this scene: the lattice's moire
# interference pattern is the thing worth watching here, and it only resolves
# as the view angle sweeps, so a slower turn just makes it look static.
AUTO_ROTATE_SPEED = 0.5

# Scene exposure, in LOG2 STOPS. A million additive points plus half a million
# stars come up hot; -1.5 pulls the lattice back off the top of the curve.
EXPOSURE_EV = -1.5


def generate_background_stars(
    n_stars: int = 500000,
    extent: float = 500.0,
    seed: int = 42,
) -> tuple:
    """Generate a background star field - completely self-contained.

    Creates a sparse field of stars with varying colors (white to orange-red)
    and sizes to provide depth context and aesthetic background.

    Args:
        n_stars: Number of stars
        extent: Stars distributed in cube [-extent, +extent]
        seed: Random seed for reproducibility

    Returns:
        Tuple of (positions, colors, radii, sharpness)
    """
    rng = np.random.default_rng(seed)

    # Random positions in large volume (way beyond the cube)
    positions = rng.uniform(-extent, extent, (n_stars, 3)).astype(np.float32)

    # Vectorized star color generation (instead of loop)
    # Star temperature determines color: hot=white/blue, cool=red/orange
    # Most stars are white, some yellow, fewer orange/red
    temp = rng.random(n_stars)

    # Initialize color array
    colors = np.ones((n_stars, 3), dtype=np.float32)

    # White stars (temp < 0.3) - already all 1.0
    # Yellow-white stars (0.3 <= temp < 0.6)
    mask = (temp >= 0.3) & (temp < 0.6)
    colors[mask] = [1.0, 1.0, 0.78]

    # Yellow stars (0.6 <= temp < 0.8)
    mask = (temp >= 0.6) & (temp < 0.8)
    colors[mask] = [1.0, 0.86, 0.59]

    # Orange-red stars (temp >= 0.8)
    mask = temp >= 0.8
    colors[mask] = [1.0, 0.71, 0.47]

    # Variable star sizes (small, twinkling effect from variation)
    radii = rng.uniform(0.01, 0.05, n_stars).astype(np.float32)

    # Gaussian-to-pointy profiles for stars (0.5 = true Gaussian, lower =
    # peaky cusp) so they read as glowy point sources, not hard discs
    sharpness = rng.uniform(0.3, 0.5, n_stars).astype(np.float32)

    return positions, colors, radii, sharpness


def generate_cubic_array(
    output_path: Path,
    grid_size: int = 100,
    spacing: float = 0.5,
    radius: float = 0.05,
    sharpness: float = 0.8,
) -> None:
    """Generate a cubic array of sharp disc-like points.

    This function contains ALL the generation logic - completely self-contained.

    Args:
        output_path: Where to write the zarr store
        grid_size: Number of points along each axis (100 = 1M points total)
        spacing: Distance between adjacent points
        radius: Radius of each point (small for disc-like appearance)
        sharpness: Sharpness value (high for sharp edges, good for DOF testing)
    """
    with asection(f"Generating Cubic Array ({grid_size}³ = {grid_size**3:,} points)"):
        # Create 3D grid coordinates using linspace for even spacing
        aprint(f"Creating {grid_size}x{grid_size}x{grid_size} grid...")
        axis = np.linspace(
            -(grid_size - 1) * spacing / 2,
            (grid_size - 1) * spacing / 2,
            grid_size,
            dtype=np.float32,
        )

        # Create meshgrid for all 3 dimensions
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

        # Flatten to create position array
        positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        aprint(f"✓ Created {positions.shape[0]:,} positions")

        # Create depth-based color gradient for visualization
        # Near points = warm colors (red/yellow)
        # Far points = cool colors (blue/cyan)
        aprint("Generating depth-based color gradient...")
        z_normalized = (z.ravel() - z.min()) / (z.max() - z.min() + 1e-8)

        colors = np.zeros((positions.shape[0], 3), dtype=np.float32)
        colors[:, 0] = 1.0 - z_normalized * 0.7  # Red: 1.0 → 0.3
        colors[:, 1] = 0.3 + 0.4 * np.sin(z_normalized * np.pi)  # Green: varies
        colors[:, 2] = 0.2 + z_normalized * 0.8  # Blue: 0.2 → 1.0

        # Add subtle position-based variation for visual interest
        position_hash = np.sin(x.ravel() * 12.345) * np.cos(y.ravel() * 67.89)
        color_variation = position_hash * 0.1

        colors[:, 0] = np.clip(colors[:, 0] + color_variation, 0, 1)
        colors[:, 1] = np.clip(colors[:, 1] - color_variation * 0.5, 0, 1)
        colors[:, 2] = np.clip(colors[:, 2] + color_variation * 0.5, 0, 1)
        aprint("✓ Generated colors with depth gradient")

        # Create uniform radii and sharpness arrays
        radii = np.full(positions.shape[0], radius, dtype=np.float32)
        sharpness_array = np.full(positions.shape[0], sharpness, dtype=np.float32)
        aprint(f"✓ Set radius={radius}, sharpness={sharpness}")

    # Write to zarr using luxar
    with asection("Writing to Zarr"):
        # Define 3D dimensions with metric units
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Frame the LATTICE, not the star field (see CAMERA_FILL above).
        cube_half_extent = (grid_size - 1) * spacing / 2
        cube_radius = float(cube_half_extent * np.sqrt(3.0))  # body half-diagonal
        cam_distance = float(
            cube_radius / np.sin(np.radians(CAMERA_FOV_DEG * CAMERA_FILL))
        )
        direction = np.asarray(CAMERA_DIRECTION, dtype=np.float64)
        cam_pos = tuple(
            float(v) for v in direction / np.linalg.norm(direction) * cam_distance
        )
        aprint(f"Camera: distance {cam_distance:.1f} at {np.round(cam_pos, 1)}")

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    exposure=EXPOSURE_EV,
                    auto_rotate=True,
                    auto_rotate_speed=AUTO_ROTATE_SPEED,
                    camera=CameraConfig(
                        position=cam_pos,
                        # The lattice's own bbox centre, resolved at load time —
                        # so the orbit pivot is the cube even though the scene
                        # bounds are dominated by the stars.
                        target_node="CubicArray",
                    ),
                ),
            )

            # Write main cubic array
            scene.add_points(
                "CubicArray",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness_array,
                opacity=1.0,
                blending_mode="additive",
                intensity=0.384,
                layer=True,
                additive_lod=stream_ladder(len(positions)),
            )
            aprint(f"✓ Added {len(positions):,} points (cubic array)")

            # Add background stars for context and aesthetics
            aprint("Creating background star field...")
            star_pos, star_colors, star_radii, star_sharp = generate_background_stars(
                n_stars=500000, extent=500.0
            )

            scene.add_points(
                "BackgroundStars",
                star_pos,
                colors=star_colors,
                radii=star_radii,
                sharpness=star_sharp,
                opacity=0.4,  # Semi-transparent
                blending_mode="normal",  # Normal blending for stars
                layer=True,
                additive_lod=stream_ladder(len(star_pos)),
            )
            aprint(f"✓ Added {len(star_pos):,} background stars")

            # Overlay annotations
            scene.add_text(
                "Cubic Array",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene, "100\u00b3 grid \u2022 Depth gradient", DEMO_META.get("citation")
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Total: {len(positions) + len(star_pos):,} points (cube + stars)")


def main() -> None:
    """Main demo entry point - generates data and launches viewer."""
    aprint("=" * 70)
    aprint("CUBIC ARRAY DEMO - Depth-of-Field Visualization")
    aprint("=" * 70)
    aprint("")
    aprint("This demo generates a 100x100x100 cubic array of points")
    aprint("with depth-based color gradient, perfect for testing DOF effects.")
    aprint("")
    aprint("Scene includes:")
    aprint("  - 1,000,000 sharp points in regular grid (warm→cool gradient)")
    aprint("  - 500,000 background stars (white→yellow→orange, semi-transparent)")
    aprint("  - Total: 1.5M points")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "cubic_array.luxar.zarr"
        generate_cubic_array(output_path)
        aprint(f"✓ Dataset generated at {output_path}")
        return

    # Create temporary directory for the demo (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_cubic_") as tmpdir:
        output_path = Path(tmpdir) / "cubic_array.luxar.zarr"

        # Generate the dataset (all generation code above)
        generate_cubic_array(output_path)

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Press Ctrl+C when done to stop servers and cleanup")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically when tmpdir context exits
    aprint("✓ Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
