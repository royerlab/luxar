#!/usr/bin/env python3
"""Self-Contained Demo: Rainbow Sphere with Fibonacci Spiral

This demo demonstrates:
- Generating evenly distributed points on a sphere using Fibonacci spiral
- Smooth rainbow color gradient flowing along the spiral
- Automatic point spacing calculation for optimal density
- High-quality rendering with sharp points
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all generation code is in this file.

Mathematical Background:
    The Fibonacci (golden angle) spiral provides near-optimal even distribution
    of points on a sphere surface. The golden angle (≈137.5°) ensures points
    don't align in regular patterns, avoiding visual artifacts.

Usage:
    python demo_rainbow_sphere.py [--points=N]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "rainbow_sphere",
    "title": "Rainbow Sphere with Fibonacci Spiral",
    "description": "200k points on a Fibonacci-spiral sphere with a smooth rainbow gradient flowing along the spiral.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["rainbow_sphere"],
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

# Opening framing. The default fit backs off until the whole bounding sphere is
# comfortably inside the frame, which leaves this demo as a small ball in the
# middle of a lot of black. It is a single decorative object, so it should
# instead fill the canvas and spill past its edges.
#
# The camera distance is derived from the sphere radius rather than hardcoded so
# it survives a change to `sphere_radius`: a sphere of radius R seen from
# distance d subtends a half-angle asin(R/d), and the frame's own half-angle is
# fov/2. Asking for a half-angle of 0.72 * fov (rather than the 0.5 * fov that
# would exactly touch top and bottom) makes the sphere overflow by ~45%.
# The solve is done AT the cinematic preset's own 35 mm lens
# (`CINEMATIC_FOV_DEG`), so `fov` is deliberately NOT pinned on the
# CameraConfig — deriving the distance from the preset's own constant is the
# house convention (`demos/_cinematic_camera.py`, and the
# `test_demos_cinematic_mode` gate).
CAMERA_FOV_DEG = CINEMATIC_FOV_DEG
CAMERA_OVERFLOW = 0.72  # fraction of the FULL fov the sphere's half-angle fills
# Elevation of the camera above the equator, as a fraction of the orbit
# distance. A little above the equator reads better than dead-on while the
# scene auto-rotates.
CAMERA_ELEVATION = 0.28

# Auto-rotation. On by default so the sphere is already turning when the demo
# opens; 0.25 is the viewer's own presentation speed — roughly one revolution
# every 25 s, slow enough to read and fast enough to register immediately.
AUTO_ROTATE_SPEED = 0.25


def generate_rainbow_sphere(
    output_path: Path,
    n_points: int = 200000,
    sphere_radius: float = 10.0,
    sharpness: float = 0.85,
) -> None:
    """Generate a dense sphere with rainbow colors along a spiral.

    This function contains ALL the generation logic - completely self-contained.

    The sphere uses a Fibonacci (golden angle) spiral for even point distribution.
    Colors flow smoothly along the spiral creating a rainbow effect.

    Args:
        output_path: Where to write the zarr store
        n_points: Number of points on sphere surface (default: 400k for high density)
        sphere_radius: Radius of the sphere in world units
        sharpness: Point sharpness (higher = crisper edges)
    """
    with asection(f"Generating Rainbow Sphere ({n_points:,} points)"):
        aprint(f"Sphere radius: {sphere_radius} units")
        aprint(f"Point sharpness: {sharpness}")

        # === STEP 1: Generate spherical spiral positions ===
        aprint("\nGenerating Fibonacci spiral on sphere...")

        # Use Fibonacci spiral (golden angle) for even distribution
        # This avoids the pole clustering that uniform random sampling creates
        indices = np.arange(0, n_points, dtype=float) + 0.5

        # Golden angle in radians (~137.5°)
        # This is the angle that provides optimal spiral distribution
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))

        # Generate spherical coordinates:
        # theta: azimuthal angle (rotates around sphere)
        # y: vertical position (from top to bottom)
        theta = indices * golden_angle  # Spiral angle
        y = 1 - (indices / float(n_points - 1)) * 2  # y ∈ [-1, 1]
        y = np.clip(y, -1.0, 1.0)  # Clamp to avoid numerical issues

        # Radius at each y-level (forms circles at different heights)
        radius_at_y = np.sqrt(1 - y * y)  # Pythagorean theorem

        # Convert to Cartesian coordinates (x, y, z)
        x = np.cos(theta) * radius_at_y * sphere_radius
        z = np.sin(theta) * radius_at_y * sphere_radius
        y = y * sphere_radius

        positions = np.column_stack([x, y, z]).astype(np.float32)
        aprint("✓ Created spherical spiral with even distribution")

        # === STEP 2: Generate rainbow colors ===
        aprint("Generating smooth rainbow gradient...")

        # Parameter t progresses along the spiral (0 → 1)
        t = np.linspace(0, 1, n_points)

        # Create rainbow using phase-shifted sine waves
        # Each color channel offset by 2π/3 (120°) for RGB sequence
        # This creates smooth transitions through all hues
        r = np.sin(2 * np.pi * t) * 0.5 + 0.5  # Red channel
        g = np.sin(2 * np.pi * t + 2 * np.pi / 3) * 0.5 + 0.5  # Green (120° offset)
        b = np.sin(2 * np.pi * t + 4 * np.pi / 3) * 0.5 + 0.5  # Blue (240° offset)

        colors = np.column_stack([r, g, b]).astype(np.float32)
        colors *= 3  # HDR intensity boost so points glow under additive blending
        aprint("✓ Generated smooth rainbow gradient (R→G→B→R)")

        # === STEP 3: Calculate optimal point radius ===
        aprint("\nCalculating optimal point spacing...")

        # For evenly distributed points on sphere, calculate spacing
        sphere_area = 4 * np.pi * sphere_radius**2
        area_per_point = sphere_area / n_points
        avg_distance = np.sqrt(area_per_point)

        # Point radius should be about half the average distance
        # This gives ~1 radius of space between points
        point_radius = avg_distance / 2
        aprint(f"Sphere surface area: {sphere_area:.2f} sq units")
        aprint(f"Area per point: {area_per_point:.6f} sq units")
        aprint(f"Average neighbor distance: {avg_distance:.4f} units")
        aprint(f"Point radius: {point_radius:.4f} units")
        aprint(f"Spacing ratio: ~{avg_distance / point_radius:.1f}x radius")

        # Create uniform radii and sharpness
        radii = np.full(n_points, point_radius, dtype=np.float32)
        sharpness_array = np.full(n_points, sharpness, dtype=np.float32)

    # === STEP 4: Write to Zarr ===
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
            ]
        )

        # Solve the orbit distance that makes the sphere overflow the frame
        # (see CAMERA_OVERFLOW above), then lift the camera off the equator.
        half_angle = np.radians(CAMERA_FOV_DEG * CAMERA_OVERFLOW)
        cam_distance = float(sphere_radius / np.sin(half_angle))
        cam_y = cam_distance * CAMERA_ELEVATION
        cam_z = float(np.sqrt(max(cam_distance**2 - cam_y**2, 0.0)))
        aprint(f"Camera: distance {cam_distance:.2f}, elevation {cam_y:.2f}")

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    auto_rotate=True,
                    auto_rotate_speed=AUTO_ROTATE_SPEED,
                    camera=CameraConfig(
                        position=(0.0, cam_y, cam_z),
                        target=(0.0, 0.0, 0.0),
                    ),
                ),
            )

            scene.add_points(
                "RainbowSphere",
                positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness_array,
                opacity=1.0,
                blending_mode="additive",
                # 400k points packed one radius apart on a thin shell means
                # every pixel sums dozens of them, so the authored gain has to
                # be small for the sphere to sit in range at exposure 0 (a
                # gain of 0.5 needed the viewer pushed down ~4.4 stops).
                # Halved from 0.024 when the opening framing moved in close: a
                # sphere that fills the canvas sums far more hues per pixel, and
                # at the old gain the middle of the ball washed out to pastel
                # grey. Nothing clips at either value — this is about keeping
                # the rainbow saturated, not about staying in range.
                intensity=0.012,
                additive_lod=stream_ladder(n_points),
                layer=True,
            )

            # Overlay annotations
            scene.add_text(
                "Rainbow Sphere",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                f"Fibonacci spiral \u2022 {n_points / 1000:.0f}K points",
                DEMO_META.get("citation"),
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{n_points * 40 / 1024 / 1024:.1f} MB (uncompressed)")


def main() -> None:
    """Main demo entry point."""
    # Parse simple command line args (optional)
    n_points = 400000  # Default: 400k points for high quality
    if len(sys.argv) > 1 and sys.argv[1].startswith("--points="):
        n_points = int(sys.argv[1].split("=")[1])

    aprint("=" * 70)
    aprint("RAINBOW SPHERE DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Generating a perfect sphere with Fibonacci spiral distribution")
    aprint(f"Points: {n_points:,}")
    aprint("Colors: Smooth rainbow gradient along spiral")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "rainbow_sphere.luxar.zarr"
        generate_rainbow_sphere(output_path, n_points=n_points)
        aprint(f"✓ Dataset generated at {output_path}")
        return

    # Use temporary directory for demo data (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_rainbow_") as tmpdir:
        output_path = Path(tmpdir) / "rainbow_sphere.luxar.zarr"

        # Generate the dataset (all code in this file!)
        generate_rainbow_sphere(output_path, n_points=n_points)

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")
        aprint(
            f"💡 TIP: This demo has high point density ({n_points / 1000:.0f}k points)"
        )
        aprint("   - Initial load may take a moment")
        aprint("   - Zoom in to see individual points clearly")
        aprint("   - Notice how evenly distributed the points are")
        aprint("   - Rainbow flows smoothly along the spiral")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically
    aprint("✓ Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
