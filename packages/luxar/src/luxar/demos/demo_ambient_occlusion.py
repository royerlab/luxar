#!/usr/bin/env python3
"""Self-Contained Demo: Baked Ambient Occlusion, side by side

This demo demonstrates:
- Why purely emissive geometry loses shape, and what baked AO gives back
- `luxar.shading.bake_ambient_occlusion` on an analytic surface
- The difference between full-sphere and cosine-hemisphere occlusion
- Complete workflow: generate → serve → view → cleanup

Three copies of the SAME point cloud, side by side, with identical flat colour,
identical radii and identical blending. The only thing that differs is the
per-point multiplier baked into the colour:

    left    no occlusion            what the renderer gives you unaided
    middle  full-sphere AO          no normals needed
    right   cosine-hemisphere AO    normals supplied

Read it left to right. The left copy is the control, and it is meant to look
disappointing: additive emissive points carry no shape information whatsoever, so
a structure with deep interpenetrating channels renders as an even glow. Nothing
is wrong with the data — there is simply no shading term in an emissive shader.

The subject is a **gyroid**, the triply-periodic minimal surface

    sin(x)·cos(y) + sin(y)·cos(z) + sin(z)·cos(x) = 0

chosen because it is the clearest possible test of an occlusion term. It divides
space into two interlocking labyrinths that never touch, so every point sits in a
channel, and how enclosed that channel is varies continuously across the surface.
It also has an exact analytic gradient, so the normals handed to the third copy
are the true surface normals rather than an estimate — which is what makes the
middle-versus-right comparison a fair one.

Why the two AO copies differ, and by how much: over the full sphere, a point on a
thin surface is surrounded by the same in-plane material as every other point,
and that shared contribution dominates the average. Restricting the integral to
the hemisphere the surface actually faces removes it. Scored against an
independent reference (the Mandelbulb demo's distance-estimator AO, on its own
geometry) the hemisphere form correlates roughly twice as well: +0.61 against
+0.31 at a small radius. Full-sphere is still the right default for genuinely
volumetric data — a light-sheet fit, a cloud — where there is no surface to face.

Usage:
    python demo_ambient_occlusion.py [--resolution=N] [--ao-radius=R]

Controls:
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
    - Toggle the three copies in the Layers panel to compare them one at a time
"""

DEMO_META = {
    "key": "ambient_occlusion",
    "title": "Ambient Occlusion (A/B)",
    "description": "A gyroid surface three times over: unshaded, full-sphere AO, hemisphere AO.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["ambient_occlusion"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.shading import bake_ambient_occlusion
from luxar.utils.paths import get_demos_output_dir

#: Unit cells per axis. Two is enough to read as a repeating labyrinth rather
#: than a single ambiguous blob, and keeps the sample grid affordable.
CELLS = 2.0

#: Shell half-thickness, in world units of |f| / |grad f| — an approximate
#: distance to the surface, so the shell has even thickness everywhere rather
#: than bulging where the implicit function happens to be flat.
SHELL_THICKNESS = 0.06

#: Occlusion radius. The gyroid's channels are about pi across, so this looks
#: across a channel but not through the neighbouring one.
AO_RADIUS = 1.2

#: Gap between the three copies, as a fraction of one copy's width. Wide enough
#: that no copy occludes its neighbour — the AO bake runs on ONE copy and its
#: result is reused, but the gap also keeps the three from reading as one object.
COPY_GAP = 0.35

#: Flat albedo shared by all three copies, in LINEAR light. Deliberately one
#: colour with no variation: any structure you can see is the occlusion term and
#: nothing else.
BASE_COLOR = np.array([0.62, 0.72, 0.95], dtype=np.float32)

#: Point radius as a MULTIPLE OF THE SAMPLE SPACING, so the sprites always
#: overlap and the surface reads as a continuous sheet at any ``--resolution``.
#:
#: This has to be derived and not hardcoded, and getting it wrong hides the whole
#: point of the demo. A fixed 0.035 against a spacing of 4*pi/159 = 0.0790 gave
#: sprites whose DIAMETER (0.070) was smaller than the gap between their centres:
#: they never touched, so the surface rendered as a dot screen. Stipple like that
#: is per-pixel on/off contrast, an order of magnitude stronger than the smooth
#: occlusion gradient underneath it, so the eye reads noise and the shading
#: becomes invisible however hard it is pushed. 0.7 puts the diameter at 1.4x the
#: spacing — comfortably overlapping without smearing the fine channel walls.
SPRITE_OVERLAP = 0.7


def point_radius(resolution: int) -> float:
    """Render radius that keeps the sampled surface visually continuous.

    Args:
        resolution: Grid samples per axis, as passed to
            :func:`sample_gyroid_surface`.

    Returns:
        World-space point radius.
    """
    spacing = 2.0 * CELLS * np.pi / max(resolution - 1, 1)
    return SPRITE_OVERLAP * spacing


#: Occlusion strength: all of the ambient treated as direct, so none of it is
#: left as an indirect floor. Full strength rather than the library's 0.7 default
#: because this scene exists to show the term as clearly as it can be shown, and
#: the auto-exposure below absorbs the extra darkening. Measured contrast (std of
#: the multiplier) at the default radius: 0.070 full-sphere, 0.147 hemisphere.
AO_STRENGTH = 1.0

#: Target peak accumulation for the auto-exposure, in linear light. Additive
#: emission SUMS along the ray, and a saturated render clips the shading flat and
#: hides the very thing the demo is about — the same trap
#: `demo_volumetric_cloud` documents for its three radiance terms. 0.7 leaves
#: headroom for the estimate below being approximate.
TARGET_PEAK = 0.7

#: The gyroid is a SURFACE, so its material is opaque rather than a medium: one
#: wall blocks a direction and a second wall behind it changes nothing. Under the
#: default Beer-Lambert reading a one-cell-thick shell only attenuates by
#: exp(-k), so a wall would pass roughly half the light — measured, the
#: saturating mapping carries about a quarter more contrast here at a matched
#: median.
OCCLUDER = "opaque"

#: Additive, not volumetric, and that is the whole point. Volumetric blending
#: self-occludes, which would supply depth cueing of its own and make the
#: control copy look partly shaded — the comparison would then be measuring two
#: effects at once. Additive ignores depth entirely, so the left copy carries
#: exactly zero shape information and every difference is attributable to AO.
BLENDING = "additive"


def gyroid_field(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the gyroid implicit function and its exact gradient.

    The gradient is analytic rather than a finite difference, so the normals are
    exact to floating point. That matters for this demo specifically: the
    hemisphere copy is supposed to show what correct normals buy, and estimated
    normals would confound that with their own error.

    Args:
        points: ``(N, 3)`` sample positions.

    Returns:
        ``(values, gradients)`` of shapes ``(N,)`` and ``(N, 3)``.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sin_x, cos_x = np.sin(x), np.cos(x)
    sin_y, cos_y = np.sin(y), np.cos(y)
    sin_z, cos_z = np.sin(z), np.cos(z)

    values = sin_x * cos_y + sin_y * cos_z + sin_z * cos_x

    gradients = np.empty_like(points)
    gradients[:, 0] = cos_x * cos_y - sin_z * sin_x
    gradients[:, 1] = -sin_x * sin_y + cos_y * cos_z
    gradients[:, 2] = -sin_y * sin_z + cos_z * cos_x
    return values, gradients


def sample_gyroid_surface(
    resolution: int, thickness: float = SHELL_THICKNESS, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points on the gyroid surface, with their unit normals.

    Args:
        resolution: Grid samples per axis before the surface filter.
        thickness: Half-thickness of the retained shell, in world units.
        seed: Seed for the anti-aliasing jitter RNG (reproducible output).

    Returns:
        ``(positions, normals)`` of shapes ``(N, 3)`` float32 and ``(N, 3)``.
    """
    extent = CELLS * np.pi
    coords = np.linspace(-extent, extent, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(coords, coords, coords, indexing="ij")
    samples = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

    # Jitter off the regular lattice, as demo_mandelbulb does: a grid-aligned
    # sample set of a smooth surface produces visible moire terracing.
    spacing = float(coords[1] - coords[0])
    rng = np.random.default_rng(seed)
    samples += rng.uniform(-0.3 * spacing, 0.3 * spacing, samples.shape)

    values, gradients = gyroid_field(samples)

    # |f| / |grad f| is the first-order distance to the zero set, so thresholding
    # it gives an even shell. Thresholding |f| alone would make the shell thick
    # wherever the field is flat and thin where it is steep.
    lengths = np.linalg.norm(gradients, axis=1)
    distance = np.abs(values) / np.maximum(lengths, 1e-9)
    on_surface = distance < thickness

    positions = samples[on_surface].astype(np.float32)
    normals = gradients[on_surface] / np.maximum(lengths[on_surface, None], 1e-9)
    return positions, normals


def auto_exposure(positions: np.ndarray, radius: float) -> Tuple[float, int]:
    """Node gain that keeps the deepest additive sightline under white.

    Derived rather than hardcoded, because the answer moves with
    ``--resolution``: the deepest column through the labyrinth measures 20 points
    at 100³ and 62 at 200³, so any fixed gain is either clipped at one end of
    that range or needlessly dim at the other. Emitted radiance is linear in
    ``intensity`` (the point shader multiplies colour by ``uIntensity``), so
    scaling by the measured depth holds the peak roughly fixed.

    Approximate on purpose: this counts points per column, whereas the real
    per-point contribution also depends on opacity and the Gaussian profile.
    Hence :data:`TARGET_PEAK` well below 1.0 rather than right at it.

    Args:
        positions: ``(N, 3)`` points of a single panel.
        radius: Per-point radius, which sets the column footprint.

    Returns:
        ``(intensity, deepest_column)``.
    """
    # Points per (x, y) column: what a view down the z axis composites per pixel.
    cell = 2.0 * radius
    keys = np.floor(positions[:, :2] / cell).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    deepest = max(int(counts.max()), 1)
    intensity = TARGET_PEAK / (deepest * float(BASE_COLOR.max()))
    return intensity, deepest


def generate_ambient_occlusion_demo(
    output_path: Path,
    resolution: int = 160,
    ao_radius: float = AO_RADIUS,
) -> int:
    """Generate the three-copy ambient-occlusion comparison scene.

    Args:
        output_path: Where to write the zarr store.
        resolution: Grid samples per axis.
        ao_radius: World-space occlusion radius.

    Returns:
        Total number of points written across all three copies.
    """
    with asection(f"Sampling the gyroid surface ({resolution}³ grid)"):
        positions, normals = sample_gyroid_surface(resolution)
        aprint(f"✓ {len(positions):,} surface points of {resolution**3:,} samples")
        if len(positions) == 0:
            aprint("⚠️  No surface points — raise --resolution")
            return 0
        aprint(f"  Surface density: {len(positions) / resolution**3 * 100:.2f}%")
        radius = point_radius(resolution)
        aprint(f"  Point radius {radius:.4f} = {SPRITE_OVERLAP} x sample spacing")
        intensity, deepest = auto_exposure(positions, radius)
        aprint(
            f"  Deepest sightline {deepest} points -> intensity {intensity:.4f} "
            f"(peak ~{TARGET_PEAK:.2f}, under white)"
        )

    with asection("Baking ambient occlusion"):
        # Both bakes run on the SAME single copy. Baking after the copies were
        # laid out side by side would let each copy occlude its neighbours, which
        # is not what any of the three panels is meant to show.
        sphere_ao = bake_ambient_occlusion(
            positions, radius=ao_radius, strength=AO_STRENGTH, occluder=OCCLUDER
        )
        aprint(
            f"✓ Full sphere:      [{sphere_ao.min():.3f}, {sphere_ao.max():.3f}] "
            f"mean {sphere_ao.mean():.3f}"
        )
        hemisphere_ao = bake_ambient_occlusion(
            positions,
            normals=normals,
            radius=ao_radius,
            strength=AO_STRENGTH,
            occluder=OCCLUDER,
        )
        aprint(
            f"✓ Cosine hemisphere: [{hemisphere_ao.min():.3f}, "
            f"{hemisphere_ao.max():.3f}] mean {hemisphere_ao.mean():.3f}"
        )
        # The spread is the figure of merit, not the mean: a bake that merely
        # darkens everything uniformly has told you nothing about the shape.
        aprint(
            f"  Contrast (std): sphere {sphere_ao.std():.3f} vs "
            f"hemisphere {hemisphere_ao.std():.3f} "
            f"({hemisphere_ao.std() / max(sphere_ao.std(), 1e-9):.1f}x)"
        )

    with asection("Writing to Zarr"):
        width = float(positions[:, 0].max() - positions[:, 0].min())
        stride = width * (1.0 + COPY_GAP)
        panels = [
            ("Emissive only (no AO)", -stride, None),
            ("AO — full sphere", 0.0, sphere_ao),
            ("AO — hemisphere (normals)", stride, hemisphere_ao),
        ]

        # Ranges must cover every copy, so they are derived from the laid-out
        # extent rather than from one copy's own bounds.
        margin = radius
        x_range = [
            float(positions[:, 0].min() - stride - margin),
            float(positions[:, 0].max() + stride + margin),
        ]
        yz_range = [
            float(positions[:, 1:].min() - margin),
            float(positions[:, 1:].max() + margin),
        ]
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True, range=x_range),
                Dimension("y", unit="units", display=True, range=yz_range),
                Dimension("z", unit="units", display=True, range=yz_range),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=ViewerConfig(cinematic_mode=True)
            )

            for name, offset, occlusion in panels:
                shifted = positions.copy()
                shifted[:, 0] += offset

                # A flat albedo, multiplied by the occlusion term where there is
                # one. Premultiplying into colour is the only route available
                # today; a per-element `shade` attribute the viewer could scale
                # live would be the better contract, and does not exist yet.
                # For a fixed side-by-side comparison it costs nothing, because
                # each panel wants its own baked-in answer anyway.
                tint = BASE_COLOR[None, :]
                if occlusion is not None:
                    tint = tint * occlusion[:, None]
                colors = np.broadcast_to(tint, (len(shifted), 3)).astype(np.float32)

                scene.add_points(
                    name,
                    shifted,
                    colors=np.ascontiguousarray(colors),
                    radii=np.full(len(shifted), radius, dtype=np.float32),
                    opacity=0.9,
                    blending_mode=BLENDING,
                    intensity=intensity,
                    # Each panel is its own layer so the Layers panel can isolate
                    # one at a time — comparing two of three is easier than
                    # comparing three at once.
                    layer=True,
                )

            scene.add_text(
                "Ambient Occlusion",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "Gyroid • none / sphere / hemisphere",
                DEMO_META.get("citation"),
            )

        total = 3 * len(positions)
        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ {total:,} points across 3 panels")

    return total


def main() -> None:
    """Main demo entry point."""
    resolution = 160
    ao_radius = AO_RADIUS

    for arg in sys.argv[1:]:
        if arg.startswith("--resolution="):
            resolution = int(arg.split("=")[1])
        elif arg.startswith("--ao-radius="):
            ao_radius = float(arg.split("=")[1])

    aprint("=" * 70)
    aprint("AMBIENT OCCLUSION DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("The same gyroid surface three times, with identical colour and")
    aprint("blending. Only the baked occlusion multiplier differs:")
    aprint("")
    aprint("  left    no occlusion          (the control — expect a flat glow)")
    aprint("  middle  full-sphere AO        (no normals needed)")
    aprint("  right   cosine-hemisphere AO  (exact analytic normals)")
    aprint("")
    aprint(f"Resolution: {resolution}³ = {resolution**3:,} samples")
    aprint(f"AO radius: {ao_radius}")
    aprint("")

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "ambient_occlusion.luxar.zarr"
        if generate_ambient_occlusion_demo(output_path, resolution, ao_radius) == 0:
            aprint("\n❌ No points generated - try a higher --resolution")
            return
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_ambient_occlusion_") as tmpdir:
        output_path = Path(tmpdir) / "ambient_occlusion.luxar.zarr"
        if generate_ambient_occlusion_demo(output_path, resolution, ao_radius) == 0:
            aprint("\n❌ No points generated - try a higher --resolution")
            return

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("Once the viewer opens:")
        aprint("  • Compare the three panels left to right")
        aprint("  • Use the Layers panel to show one panel at a time")
        aprint("  • Orbit: the occlusion never changes as the camera moves,")
        aprint("    because it is a property of the geometry, not of the view")
        aprint("  • The channels of the labyrinth are only legible on the right")
        aprint("")
        aprint("Try:")
        aprint("  --ao-radius=0.3  → only the tightest creases darken")
        aprint("  --ao-radius=3.0  → looks through the whole labyrinth, washing out")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
