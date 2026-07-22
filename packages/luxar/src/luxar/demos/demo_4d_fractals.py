#!/usr/bin/env python3
"""Self-Contained Demo: 4D Geometric Fractal Explorer

This demo demonstrates:
- Simple 4D geometric fractals (XOR, Menger, Sierpinski, etc.)
- Categorical dimension (select different fractal patterns)
- 4th spatial dimension (W) navigation showing fractal slices
- Large dataset (~100M+ points) with spatial indexing
- Only ~1M points visible per configuration
- Complete workflow: generate → serve → view → cleanup

Mathematical Background:
    Geometric fractals use simple rules applied to 4D grids:

    - XOR Fractal: (x⊕y⊕z⊕w) creates self-similar patterns from bitwise XOR
    - Menger Sponge 4D: Recursive cube subdivision with holes
    - Sierpinski 4D: Points where (x+y+z+w) satisfies modular arithmetic
    - Cantor Dust 4D: Product of 1D Cantor sets in each dimension
    - Checkerboard: (x+y+z+w) mod 2 creates hypercheckerboard
    - Diamond: |x|+|y|+|z|+|w| creates symmetric patterns

Performance:
    - INSTANT generation - simple conditions, no iteration
    - Fully vectorized numpy operations
    - All 6 fractals in ~5-15 seconds total

Usage:
    python demo_4d_fractals.py [--grid=N]

Controls:
    - Press '1' to select FRACTAL TYPE, then [/] to switch fractals
    - Press '5' to select W dimension, then [/] to navigate 4D slices
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "fractals_4d",
    "title": "4D Geometric Fractal Explorer",
    "description": "Six 4D geometric fractals (XOR, Menger, Sierpinski, ...) with categorical + W-axis navigation.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["fractals_4d"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def xor_fractal_4d(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """XOR fractal: (w⊕x⊕y⊕z) creates beautiful self-similar patterns.

    Args:
        W, X, Y, Z: Integer coordinate grids

    Returns:
        XOR values (for coloring)
    """
    # Convert to integers and compute bitwise XOR
    wi: np.ndarray = W.astype(np.int32)
    xi: np.ndarray = X.astype(np.int32)
    yi: np.ndarray = Y.astype(np.int32)
    zi: np.ndarray = Z.astype(np.int32)

    result = wi ^ xi ^ yi ^ zi
    return result  # type: ignore[no-any-return]


def menger_sponge_4d(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, level: int = 3
) -> np.ndarray:
    """4D Menger sponge - recursive hypercube with holes.

    Args:
        W, X, Y, Z: Coordinate grids (scaled to [0, 3^level])
        level: Recursion depth

    Returns:
        Boolean mask (True = solid, False = hole)
    """
    # Scale to [0, 3^level]
    scale = 3**level
    wi = ((W + 1) * scale / 2).astype(np.int32) % scale
    xi = ((X + 1) * scale / 2).astype(np.int32) % scale
    yi = ((Y + 1) * scale / 2).astype(np.int32) % scale
    zi = ((Z + 1) * scale / 2).astype(np.int32) % scale

    # Check if in hole at any level
    solid = np.ones(W.shape, dtype=bool)

    for lev in range(level):
        div = 3**lev
        # Count how many coords are in middle third
        wm = (wi // div) % 3 == 1
        xm = (xi // div) % 3 == 1
        ym = (yi // div) % 3 == 1
        zm = (zi // div) % 3 == 1

        # Hole if 2 or more coords in middle third
        middle_count = wm.astype(int) + xm.astype(int) + ym.astype(int) + zm.astype(int)
        solid &= middle_count < 2

    return solid.astype(np.int16)  # type: ignore[no-any-return]


def sierpinski_4d(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """4D Sierpinski - points where (w&x&y&z)==0 in binary.

    Args:
        W, X, Y, Z: Integer coordinate grids

    Returns:
        Sierpinski values
    """
    wi: np.ndarray = W.astype(np.int32)
    xi: np.ndarray = X.astype(np.int32)
    yi: np.ndarray = Y.astype(np.int32)
    zi: np.ndarray = Z.astype(np.int32)

    # Sierpinski condition: bitwise AND of all coordinates
    result = (wi & xi & yi & zi) == 0
    return result.astype(np.int16)  # type: ignore[no-any-return]


def cantor_dust_4d(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, level: int = 4
) -> np.ndarray:
    """4D Cantor dust - product of Cantor sets.

    Args:
        W, X, Y, Z: Coordinate grids
        level: Recursion depth

    Returns:
        Boolean mask
    """
    # Scale to [0, 3^level]
    scale = 3**level
    wi = ((W + 1) * scale / 2).astype(np.int32)
    xi = ((X + 1) * scale / 2).astype(np.int32)
    yi = ((Y + 1) * scale / 2).astype(np.int32)
    zi = ((Z + 1) * scale / 2).astype(np.int32)

    # Check if in Cantor set for each dimension
    def in_cantor(coord, lev):  # type: ignore[no-untyped-def]
        in_set = np.ones(coord.shape, dtype=bool)
        for level_idx in range(lev):
            div = 3**level_idx
            in_set &= ((coord // div) % 3) != 1  # Not in middle third
        return in_set

    # Point is in 4D Cantor dust if in Cantor set in all 4 dimensions
    result = (
        in_cantor(wi, level)
        & in_cantor(xi, level)
        & in_cantor(yi, level)
        & in_cantor(zi, level)
    )
    return result.astype(np.int16)  # type: ignore[no-any-return]


def checkerboard_4d(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, scale: int = 2
) -> np.ndarray:
    """4D hypercheckerboard pattern.

    Args:
        W, X, Y, Z: Coordinate grids
        scale: Checker size

    Returns:
        Pattern values
    """
    wi: np.ndarray = (W * scale).astype(np.int32)
    xi: np.ndarray = (X * scale).astype(np.int32)
    yi: np.ndarray = (Y * scale).astype(np.int32)
    zi: np.ndarray = (Z * scale).astype(np.int32)

    result = (wi + xi + yi + zi) % 2
    return result  # type: ignore[no-any-return]


def diamond_fractal_4d(
    W: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
) -> np.ndarray:
    """4D diamond/taxicab fractal based on L1 distance.

    Args:
        W, X, Y, Z: Coordinate grids

    Returns:
        Distance-based pattern values
    """
    # L1 distance from origin
    dist = np.abs(W) + np.abs(X) + np.abs(Y) + np.abs(Z)

    # Create fractal pattern from distance
    result = (dist * 5).astype(np.int32) % 7

    return result  # type: ignore[no-any-return]


def generate_4d_fractal(
    fractal_type: int,
    grid_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one 4D geometric fractal on a grid (FAST - no iteration!).

    Args:
        fractal_type: 0-5 indicating which fractal
        grid_size: Grid resolution (N^4 points)

    Returns:
        Tuple of (positions, values) for points in the fractal
    """
    aprint(f"  Grid: {grid_size}^4 = {grid_size**4:,} points")

    # Create integer grid for geometric fractals
    aprint("  Creating 4D integer grid...")
    coords_int = np.arange(grid_size, dtype=np.int32)

    # Create meshgrid efficiently
    W, X, Y, Z = np.meshgrid(
        coords_int, coords_int, coords_int, coords_int, indexing="ij"
    )

    # Fractal type computation (INSTANT - just simple conditions!)
    fractal_funcs = {
        0: ("XOR Fractal", xor_fractal_4d),
        1: (
            "Menger Sponge 4D",
            lambda w, x, y, z: menger_sponge_4d(w, x, y, z, level=3),
        ),
        2: ("Sierpinski 4D", sierpinski_4d),
        3: ("Cantor Dust 4D", lambda w, x, y, z: cantor_dust_4d(w, x, y, z, level=4)),
        4: (
            "Hypercheckerboard",
            lambda w, x, y, z: checkerboard_4d(w, x, y, z, scale=4),
        ),
        5: ("Diamond Fractal", diamond_fractal_4d),
    }

    fractal_name, fractal_func = fractal_funcs[fractal_type]
    aprint(f"  Computing {fractal_name}...")

    # Compute fractal values (VERY FAST - no iteration!)
    values = fractal_func(W, X, Y, Z)

    # VALUE-BASED thresholding per fractal type
    # Calculate target density to stay under 2M points
    target_max = 2_000_000
    target_density = target_max / grid_size**4  # ~2%

    if fractal_type == 0:  # XOR - threshold on XOR value
        # Keep points with high XOR values (most interesting patterns)
        # Aim for top 2% of grid
        threshold_percentile = max(50, (1 - target_density) * 100)
        threshold = np.percentile(values, threshold_percentile)
        keep_mask = values >= threshold
        aprint(
            f"    XOR threshold: {threshold:.0f} (top {100 - threshold_percentile:.1f}%)"
        )

    elif fractal_type == 1:  # Menger - keep solid, already sparse
        keep_mask = values > 0

    elif fractal_type == 2:  # Sierpinski - threshold on point count if needed
        initial_mask = values > 0
        n_initial: int = int(np.sum(initial_mask))
        if n_initial > target_max:
            # For Sierpinski, use distance from origin to threshold
            dist = np.sqrt(W**2 + X**2 + Y**2 + Z**2)
            # Keep points with intermediate distances (most interesting)
            dist_valid = dist[initial_mask]
            threshold_percentile = (1 - target_max / n_initial) * 100
            threshold = np.percentile(dist_valid, threshold_percentile)
            keep_mask = initial_mask & (dist >= threshold)
            aprint(f"    Sierpinski distance threshold: {threshold:.2f}")
        else:
            keep_mask = initial_mask

    elif fractal_type == 3:  # Cantor - already very sparse
        keep_mask = values > 0

    elif fractal_type == 4:  # Checkerboard - threshold on parity
        # Only keep "1" values, and if still too many, subsample by position
        initial_mask = values > 0
        n_initial = np.sum(initial_mask)
        if n_initial > target_max:
            # Keep points where (W+X+Y+Z) > threshold
            sum_coords = (W + X + Y + Z)[initial_mask]
            threshold_percentile = (1 - target_max / n_initial) * 100
            threshold = np.percentile(sum_coords, threshold_percentile)
            keep_mask = initial_mask & ((W + X + Y + Z) >= threshold)
            aprint(f"    Checkerboard coordinate sum threshold: {threshold:.0f}")
        else:
            keep_mask = initial_mask

    else:  # Diamond - threshold on distance value
        # Keep points with high L1 distance (outer shells)
        threshold_percentile = max(50, (1 - target_density) * 100)
        threshold = np.percentile(values, threshold_percentile)
        keep_mask = values >= threshold
        aprint(
            f"    Diamond distance threshold: {threshold:.0f} (top {100 - threshold_percentile:.1f}%)"
        )

    n_points_final: int = int(np.sum(keep_mask))
    aprint(
        f"    ✓ Kept: {n_points_final:,} points ({n_points_final / grid_size**4 * 100:.2f}% density)"
    )

    # Safety check
    if np.sum(keep_mask) == 0:
        aprint("    ⚠️  No points - using fallback")
        # Keep random 1M points
        n_fallback = min(1_000_000, grid_size**4)
        indices = np.random.choice(grid_size**4, n_fallback, replace=False)
        keep_mask_flat = np.zeros(grid_size**4, dtype=bool)
        keep_mask_flat[indices] = True
        keep_mask = keep_mask_flat.reshape(keep_mask.shape)

    # Convert to world coordinates [-1, 1]
    w_coords = (W[keep_mask].astype(np.float32) / grid_size - 0.5) * 2
    x_coords = (X[keep_mask].astype(np.float32) / grid_size - 0.5) * 2
    y_coords = (Y[keep_mask].astype(np.float32) / grid_size - 0.5) * 2
    z_coords = (Z[keep_mask].astype(np.float32) / grid_size - 0.5) * 2

    positions = np.column_stack([w_coords, x_coords, y_coords, z_coords])
    pattern_values = values[keep_mask]

    aprint(f"  ✓ {fractal_name}: {len(positions):,} points")
    aprint(f"    Density: {len(positions) / grid_size**4 * 100:.2f}% of grid")

    return positions, pattern_values


def pattern_values_to_colors(values: np.ndarray) -> np.ndarray:
    """Convert fractal pattern values to rainbow colors.

    Args:
        values: Pattern value array

    Returns:
        RGB colors
    """
    # Handle empty array case
    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.float32)  # type: ignore[no-any-return]

    # Normalize to [0, 1]
    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        t = (values.astype(np.float32) - v_min) / (v_max - v_min)
    else:
        t = np.zeros(len(values), dtype=np.float32)

    # Create rainbow
    colors = np.zeros((len(values), 3), dtype=np.float32)
    colors[:, 0] = np.abs(np.sin(2 * np.pi * t))
    colors[:, 1] = np.abs(np.sin(2 * np.pi * t + 2 * np.pi / 3))
    colors[:, 2] = np.abs(np.sin(2 * np.pi * t + 4 * np.pi / 3))

    # Moderate brightness to prevent saturation with additive blending
    colors *= 0.8

    return colors  # type: ignore[no-any-return]


def generate_4d_fractal_dataset(
    output_path: Path,
    grid_size: int = 100,
) -> int:
    """Generate complete 4D fractal dataset with multiple types.

    Args:
        output_path: Where to write zarr
        grid_size: Grid resolution (default 100 → 1M points per 3D slice)

    Returns:
        Total points generated
    """
    # Simple geometric fractal types
    fractal_names = [
        "XOR Fractal (w⊕x⊕y⊕z)",
        "4D Menger Sponge",
        "4D Sierpinski",
        "4D Cantor Dust",
        "4D Hypercheckerboard",
        "4D Diamond Fractal",
    ]

    all_positions = []
    all_colors = []
    all_fractal_ids = []

    with asection(f"Generating 6 × 4D Fractals (grid: {grid_size}^4)"):
        for frac_id, frac_name in enumerate(fractal_names):
            with asection(f"Fractal {frac_id}: {frac_name}"):
                # Generate fractal (FAST!)
                positions, pattern_values = generate_4d_fractal(
                    frac_id,
                    grid_size=grid_size,
                )

                # Generate colors from pattern values
                colors = pattern_values_to_colors(pattern_values)

                # Add fractal type ID as first coordinate
                fractal_ids = np.full(len(positions), frac_id, dtype=np.float32)

                all_positions.append(positions)
                all_colors.append(colors)
                all_fractal_ids.append(fractal_ids)

    # Combine all fractals
    with asection("Combining all fractals"):
        positions = np.vstack(all_positions)
        colors = np.vstack(all_colors)
        fractal_ids = np.concatenate(all_fractal_ids)

        # Reorder: [fractal_id, w, x, y, z]
        positions_5d = np.column_stack(
            [
                fractal_ids,
                positions[:, 0],  # W
                positions[:, 1],  # X
                positions[:, 2],  # Y
                positions[:, 3],  # Z
            ]
        )

        aprint(f"✓ Total points: {len(positions_5d):,}")
        aprint(f"  Per fractal: {len(positions_5d) // 6:,} avg")
        aprint(f"  Per 3D slice: ~{grid_size**3:,} points")

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension(
                    "fractal",
                    unit="",
                    categories=[
                        "XOR",
                        "Menger",
                        "Sierpinski",
                        "Cantor",
                        "Checkerboard",
                        "Diamond",
                    ],
                    display=False,
                    description="Fractal type - geometric 4D fractals",
                ),
                Dimension(
                    "w",
                    unit="",
                    range=(-2.5, 2.5),
                    step=5.0 / grid_size,
                    display=False,
                    discrete=False,
                    description="4th spatial dimension (navigate to see slices!)",
                ),
                Dimension("x", unit="", range=(-2.5, 2.5), display=True),
                Dimension("y", unit="", range=(-2.5, 2.5), display=True),
                Dimension("z", unit="", range=(-2.5, 2.5), display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add all points with small uniform radius
            radii = np.full(len(positions_5d), 0.025, dtype=np.float32)
            sharpnesses = np.full(len(positions_5d), 0.55, dtype=np.float32)

            scene.add_points(
                "Fractals4D",
                positions_5d,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
                intensity=0.0625,
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "4D Geometric Fractals",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Dimension-aware labels for fractal type
            fractal_labels = [
                "XOR",
                "Menger",
                "Sierpinski",
                "Cantor",
                "Checkerboard",
                "Diamond",
            ]
            for i, label in enumerate(fractal_labels):
                scene.add_text(
                    label,
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"fractal": float(i)},
                    transition="fade",
                    transition_duration=0.15,
                )

            # Info
            scene.add_text(
                "6 fractal types \u2022 4D space",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{len(positions_5d) * 40 / 1024 / 1024:.0f} MB")

    return len(positions_5d)


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    # NOTE: grid_size=100 creates 57M+ points and 31K+ chunks, which overwhelms
    # browser HTTP connection limits. grid_size=50 gives ~6M points (~3K chunks).
    grid_size = 50  # 50^4 = 6.25M per fractal max, ~1M after filtering

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--grid="):
                grid_size = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("4D FRACTAL EXPLORER DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Explore 6 different 4D fractals in quaternion space!")
    aprint(f"Grid: {grid_size}^4 = {grid_size**4:,} samples per fractal")
    aprint(f"Visible per slice: ~{grid_size**3:,} points (up to ~1M per fractal)")
    aprint("")
    aprint("Fractal types:")
    aprint("  0: XOR Fractal - Bitwise XOR creates self-similar patterns")
    aprint("  1: 4D Menger Sponge - Recursive hypercube with holes")
    aprint("  2: 4D Sierpinski - Bitwise AND condition")
    aprint("  3: 4D Cantor Dust - Product of Cantor sets")
    aprint("  4: 4D Hypercheckerboard - Alternating pattern")
    aprint("  5: 4D Diamond Fractal - L1 distance patterns")
    aprint("")
    aprint("⏱️  Generation time: ~10-30 seconds for all 6 fractals")
    aprint("   (No iteration - instant geometric computation!)")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "fractals_4d.luxar.zarr"
        generate_4d_fractal_dataset(output_path, grid_size=grid_size)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_4d_fractals_") as tmpdir:
        output_path = Path(tmpdir) / "fractals_4d.luxar.zarr"

        # Generate all fractals
        generate_4d_fractal_dataset(output_path, grid_size=grid_size)

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION - EXPLORE 4D FRACTAL SPACE!")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  1. Press '1' → Select FRACTAL TYPE")
        aprint("     Press ']' to cycle through 6 different fractals")
        aprint("     Each has unique 4D structure!")
        aprint("")
        aprint("  2. Press '5' → Select W dimension")
        aprint("     Press '['/']' to navigate through 4D slices")
        aprint("     See how the fractal changes in the 4th dimension!")
        aprint("")
        aprint("What you're seeing:")
        aprint("  • 3D slices through 4D fractals")
        aprint("  • Colors show iteration depth (complexity)")
        aprint("  • Navigate W to see how structure evolves")
        aprint("  • Switch fractal type to compare geometries")
        aprint("")
        aprint("Try this:")
        aprint("  1. Start with XOR fractal (fractal=0)")
        aprint("  2. Navigate W dimension with [/]")
        aprint("  3. Switch to other fractals (1=Menger, 2=Sierpinski, etc.)")
        aprint("  4. Notice how each fractal has unique 4D structure")
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
