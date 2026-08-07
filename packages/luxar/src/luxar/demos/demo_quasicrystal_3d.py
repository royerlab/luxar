#!/usr/bin/env python3
"""Self-Contained Demo: 3D Aperiodic Quasicrystal (3D Penrose Tiling)

This demo demonstrates:
- 3D quasicrystal with icosahedral symmetry (5-fold!)
- Aperiodic tiling (never repeats, but ordered)
- Cut-and-project method from 6D to 3D
- Color-coded by perpendicular space coordinates
- Beautiful mathematical structure impossible in periodic crystals
- Complete workflow: generate → serve → view → cleanup

The demo is completely self-contained - all quasicrystal math is here!

Mathematical Background - Cut-and-Project Method:

    The 3D icosahedral quasicrystal is constructed by projecting a 6D cubic
    lattice onto an irrational 3D subspace:

    1. **6D Lattice**: Start with Z^6 (all 6D integer coordinates)
       - This is completely periodic in 6D

    2. **Decomposition**: Split 6D space into two orthogonal 3D subspaces:
       - E_∥ (parallel/physical): The 3D space we see
       - E_⊥ (perpendicular): A 3D filtering space

    3. **Projection Matrices**: Use golden ratio φ = (1+√5)/2 for irrationality
       - P_∥: Projects Z^6 → E_∥ (produces 3D positions)
       - P_⊥: Projects Z^6 → E_⊥ (used for filtering)
       - These must be orthogonal: P_∥ · P_⊥^T = 0

    4. **Acceptance Window**: For each v ∈ Z^6:
       - v_∥ = P_∥ · v  (physical 3D position)
       - v_⊥ = P_⊥ · v  (perpendicular 3D position)
       - Accept point if |v_⊥| < r (inside window)

    5. **Why It Works**:
       - Irrational slopes (golden ratio) → never periodic
       - Filtering creates finite density
       - Icosahedral symmetry group (forbidden in crystals!)

    The resulting point set is:
    - **Aperiodic**: Pattern never repeats (proven using irrationality of φ)
    - **Quasiperiodic**: Has long-range order despite aperiodicity
    - **Dense**: Points densely fill space (unlike fractals)
    - **Self-similar**: Local structure repeats at different scales

    Icosahedral Symmetry (Point Group I_h):
    - 6 five-fold axes (through opposite vertices)
    - 10 three-fold axes (through opposite faces)
    - 15 two-fold axes (through opposite edges)
    - Total: 120 symmetry operations

    Physical Realization:
    - Real quasicrystals (Al-Mn, Al-Cu-Fe) have atomic positions
      following this exact mathematical structure
    - Diffraction patterns show sharp Bragg peaks (ordered) but with
      forbidden 5-fold symmetry (aperiodic)

References:
    - de Bruijn, N.G. (1981). "Algebraic theory of Penrose's non-periodic tilings"
      Koninklijke Nederlandse Akademie van Wetenschappen
    - Shechtman et al. (1984). "Metallic Phase with Long-Range Orientational Order"
      Physical Review Letters 53 (20): 1951–1953
    - Wikipedia: https://en.wikipedia.org/wiki/Quasicrystal
    - Wikipedia: https://en.wikipedia.org/wiki/Cut-and-project_method
    - Senechal, M. (1995). "Quasicrystals and Geometry"
      Cambridge University Press

Nobel Prize:
    Dan Shechtman received the 2011 Nobel Prize in Chemistry for discovering
    quasicrystals, initially met with skepticism because 5-fold symmetry was
    "impossible" in periodic crystals. Quote: "There is no such thing as
    quasicrystals, only quasi-scientists." - Linus Pauling (he was wrong!)

Usage:
    python demo_quasicrystal_3d.py [--points=N]

Controls:
    - Rotate to see 5-fold symmetry axes
    - Zoom to see self-similar structure at all scales
    - Colors show 3D perpendicular space position
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "quasicrystal",
    "title": "Quasicrystal",
    "description": "3D icosahedral quasicrystal built by cut-and-project from a 6D lattice.",
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["quasicrystal"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def create_icosahedral_projection_matrices():  # type: ignore[no-untyped-def]
    """Create projection matrices for icosahedral quasicrystal.

    Uses golden ratio φ = (1+√5)/2 to create irrational slopes, ensuring
    aperiodicity. The specific matrix entries are chosen to give icosahedral
    (I_h) symmetry group, which includes 5-fold rotation axes.

    Mathematical Construction:
        The matrices are based on the vertex coordinates of a 6D hypercube
        projected to create icosahedral symmetry in 3D. The golden ratio
        appears because the icosahedron's geometry inherently involves φ
        (e.g., vertices of icosahedron have coordinates like (±1, ±φ, 0)).

    Returns:
        Tuple of (P_parallel, P_perp) where:
        - P_parallel: 3×6 matrix projecting 6D → 3D physical space E_∥
        - P_perp: 3×6 matrix projecting 6D → 3D perpendicular space E_⊥
        - These matrices are orthogonal: P_∥ · P_⊥^T = 0

    Reference:
        Kramer & Neri (1984). "On periodic and non-periodic space fillings"
        Acta Crystallographica A40: 580-587
    """
    φ = (1 + np.sqrt(5)) / 2  # Golden ratio

    # Parallel space projection (defines physical 3D)
    # Uses golden ratio to create icosahedral symmetry
    P_parallel = np.array(
        [
            [1, φ, 0, -1, φ, 0],
            [φ, 0, 1, φ, 0, -1],
            [0, 1, φ, 0, -1, φ],
        ],
        dtype=np.float64,
    )

    # Normalize rows to unit vectors
    P_parallel = P_parallel / np.linalg.norm(P_parallel, axis=1, keepdims=True)

    # Perpendicular space: Compute true orthogonal complement using SVD
    # The perpendicular space is the null space (right singular vectors) of P_parallel
    U, S, Vt = np.linalg.svd(P_parallel)

    # Last 3 rows of V^T span the null space (perpendicular to P_parallel)
    P_perp = Vt[3:6, :]  # 3×6 matrix (last 3 singular vectors)

    # Verify orthogonality
    dot_product: float = float(np.sum(np.abs(P_parallel @ P_perp.T)))
    if dot_product < 1e-10:
        aprint(f"  ✓ Projection spaces are orthogonal (dot={dot_product:.2e})")
    else:
        aprint(
            f"  ⚠️  Warning: Projections not perfectly orthogonal (dot={dot_product:.2e})"
        )

    return P_parallel, P_perp


def generate_quasicrystal_3d(
    output_path: Path,
    target_points: int = 1_000_000,  # 1M points target
    lattice_radius: int = 18,  # Reasonable: (2×18+1)^6 = 2.3B points
    box_size: float = 45.0,  # Scale box proportionally
    seed: int = 0,
) -> int:
    """Generate 3D icosahedral quasicrystal using cut-and-project.

    Args:
        output_path: Where to write zarr
        target_points: Target number of points (~500k recommended)
        lattice_radius: Radius of 6D lattice to sample
        box_size: Physical 3D box size
        seed: Seed for the subsampling RNG (reproducible output)

    Returns:
        Actual number of points generated
    """
    with asection(f"Generating 3D Quasicrystal (target: {target_points:,} points)"):
        # Create projection matrices
        aprint("Setting up icosahedral projections...")
        P_parallel, P_perp = create_icosahedral_projection_matrices()
        φ = (1 + np.sqrt(5)) / 2
        aprint(f"  Golden ratio φ: {φ:.6f}")
        aprint("  Parallel space: 3D physical (what we see)")
        aprint("  Perpendicular space: 3D filter (controls density)")

        # Use much larger window and lattice to get 500k points
        aprint(f"\nSampling 6D lattice (radius={lattice_radius})...")
        total_6d_points = (2 * lattice_radius + 1) ** 6
        aprint(f"  Potential points: {total_6d_points:,}")

        # Larger window to accept more points
        # Scale with lattice_radius: r_window ~ lattice_radius / 10
        r_window = lattice_radius / 10.0  # Adaptive: 20/10 = 2.0
        aprint(f"  Acceptance window radius: {r_window:.3f} (adaptive)")

        # Collect all positions
        all_physical_positions = []
        all_perp_positions = []

        # Efficient vectorized 6D lattice generation
        aprint("  Generating 6D lattice points...")
        coords_1d = np.arange(-lattice_radius, lattice_radius + 1, dtype=np.float64)

        # Generate all 6D lattice points efficiently using meshgrid
        # This is memory-intensive but much faster than loops
        total_processed = 0

        # Process in many small chunks for fine-grained progress and efficiency
        # More chunks = better rejection of unproductive regions
        # Target: ~100 chunks, each ~25B/100 ≈ 250M points
        # Split 6D space by splitting multiple dimensions

        # Split on FIRST dimension only, reasonable chunk size
        # For radius=18: 37 coords → ~12 chunks of ~3 coords each
        # Each chunk: 3 × 37^5 ≈ 209M points (large but vectorized = fast)
        chunk_size_1d = 3  # Process 3 coordinates at a time
        chunk_ranges = [
            coords_1d[i : i + chunk_size_1d]
            for i in range(0, len(coords_1d), chunk_size_1d)
        ]
        n_chunks = len(chunk_ranges)

        aprint(
            f"  Processing in {n_chunks} chunks "
            "(~3 × (2·lattice_radius+1)^5 points per chunk)"
        )

        chunk_num = 0
        for coords_chunk in chunk_ranges:
            chunk_num += 1
            aprint(f"    Chunk {chunk_num}/{n_chunks}...")

            # Generate 6D meshgrid (split only first dim for speed)
            grids = np.meshgrid(
                coords_chunk,
                coords_1d,
                coords_1d,
                coords_1d,
                coords_1d,
                coords_1d,
                indexing="ij",
            )

            # Stack into (N, 6) array
            chunk_array = np.stack([g.ravel() for g in grids], axis=1)
            total_processed += len(chunk_array)

            # Project to parallel (physical 3D)
            parallel = chunk_array @ P_parallel.T

            # Project to perpendicular (filter 3D)
            perp = chunk_array @ P_perp.T

            # Filter by perpendicular distance and box bounds
            perp_dist = np.linalg.norm(perp, axis=1)
            keep = (perp_dist < r_window) & (
                np.max(np.abs(parallel), axis=1) < box_size / 2
            )

            n_kept_chunk: int = int(np.sum(keep))

            if n_kept_chunk > 0:
                all_physical_positions.append(parallel[keep])
                all_perp_positions.append(perp[keep])

            current_points = sum(len(p) for p in all_physical_positions)
            aprint(f"      → {n_kept_chunk:,} kept ({current_points:,} total)")

        # Combine results
        if not all_physical_positions:
            aprint("  ⚠️  No points found - try larger window")
            return 0

        positions_3d = np.vstack(all_physical_positions).astype(np.float32)
        perp_coords = np.vstack(all_perp_positions).astype(np.float32)

        aprint(f"✓ Generated {len(positions_3d):,} quasicrystal points")
        aprint(f"  Acceptance rate: {len(positions_3d) / total_processed * 100:.3f}%")

        # If too many points, subsample
        if len(positions_3d) > target_points:
            aprint(f"  Subsampling to {target_points:,} points...")
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(positions_3d), target_points, replace=False)
            positions_3d = positions_3d[indices]
            perp_coords = perp_coords[indices]

        # Generate colors from perpendicular space coordinates
        aprint("Generating colors from perpendicular space...")

        # Map perpendicular 3D coordinates (v_⊥) to RGB colors
        # This encodes WHERE in the 3D perpendicular space each point came from
        # Points with similar perpendicular positions get similar colors
        perp_norm = perp_coords / (r_window * 2)  # Normalize to ~[-0.5, 0.5]
        colors = np.abs(perp_norm)  # Map to [0, 1] for RGB
        colors = colors.astype(np.float32)

        aprint("✓ Colors encode perpendicular space E_⊥ position:")
        aprint("   R = |v_⊥_x|, G = |v_⊥_y|, B = |v_⊥_z|")
        aprint("   (Similar colors = similar perpendicular space origin)")

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="a", display=True),
                Dimension("y", unit="a", display=True),
                Dimension("z", unit="a", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Small points for dense quasicrystal appearance
            radii = np.full(len(positions_3d), 0.08, dtype=np.float32)
            sharpnesses = np.full(len(positions_3d), 0.8, dtype=np.float32)

            scene.add_points(
                "Quasicrystal",
                positions_3d,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.9,
                intensity=0.125,
                layer=True,
            )

            # Overlay annotations
            scene.add_text(
                "3D Quasicrystal",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Icosahedral aperiodic tiling",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Written to {output_path}")
        aprint(f"✓ Dataset size: ~{len(positions_3d) * 36 / 1024 / 1024:.0f} MB")

    return len(positions_3d)


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    target_points = 500_000

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--points="):
                target_points = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("3D QUASICRYSTAL DEMO - APERIODIC TILING")
    aprint("=" * 70)
    aprint("")
    aprint("Generate a 3D Penrose-like quasicrystal!")
    aprint(f"Target points: {target_points:,}")
    aprint("")
    aprint("What is a quasicrystal?")
    aprint("  • Ordered but NEVER repeating (aperiodic)")
    aprint("  • 5-fold rotational symmetry (impossible in normal crystals!)")
    aprint("  • Created using golden ratio projections from 6D")
    aprint("  • Discovered in 1984 - Nobel Prize 2011")
    aprint("")
    aprint("How it works:")
    aprint("  1. Sample 6D integer lattice (periodic)")
    aprint("  2. Project to 3D using golden ratio slopes")
    aprint("  3. Filter by perpendicular distance")
    aprint("  4. Result: Aperiodic 3D tiling!")
    aprint("")
    aprint("Generation time: ~30-60 seconds")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "quasicrystal.luxar.zarr"
        actual_points = generate_quasicrystal_3d(
            output_path,
            target_points=target_points,
        )
        if actual_points == 0:
            aprint("\nGeneration failed - no points created")
            return
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total points: {actual_points:,}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_quasicrystal_") as tmpdir:
        output_path = Path(tmpdir) / "quasicrystal.luxar.zarr"

        # Generate with scaled parameters
        actual_points = generate_quasicrystal_3d(
            output_path,
            target_points=target_points,
            # Uses defaults from function: lattice_radius=18, box_size=45
        )

        if actual_points == 0:
            aprint("\nGeneration failed - no points created")
            return

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING - DISCOVER APERIODIC ORDER!")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  - Rotate slowly - look for 5-fold symmetry axes")
        aprint("  - Zoom in - patterns are self-similar at all scales")
        aprint("  - No periodic repetition - pattern never repeats!")
        aprint("  - Colors encode position in 'perpendicular space'")
        aprint("")
        aprint("Things to notice:")
        aprint("  - Dense, ordered structure (not random)")
        aprint("  - Local patterns repeat, but global pattern doesn't")
        aprint("  - 5-fold rotation symmetry (rotate 72 degrees)")
        aprint("  - Similar to real atomic quasicrystals!")
        aprint("")
        aprint("Real-world: Al-Mn quasicrystals have this exact structure")
        aprint("  (discovered by Dan Shechtman, Nobel Prize 2011)")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
