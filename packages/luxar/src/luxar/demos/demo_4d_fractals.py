#!/usr/bin/env python3
"""Self-Contained Demo: 4D Geometric Fractal Explorer

This demo demonstrates:
- Simple 4D geometric fractals (XOR, Menger, Sierpinski, etc.)
- Categorical dimension (select different fractal patterns)
- 4th spatial dimension (W) navigation showing fractal slices
- Millions of points with spatial indexing; each (fractal, w) slice
  shows its own subset, and every w slider stop shows structure
- Complete workflow: generate → serve → view → cleanup

Mathematical Background:
    Geometric fractals use simple rules applied to 4D grids. All rules are
    evaluated on integer grid indices (or ternary cell coordinates derived
    from them), and every rule is chosen so that EVERY w-slice of the 4D
    set is non-empty — the w slider always shows structure:

    - XOR Fractal: high values of (x⊕y⊕z⊕w) form self-similar shells
    - Menger Sponge 4D: recursive hypercube, holes where ≥2 ternary
      digits are in the middle third
    - Sierpinski 4D: no THREE coordinates share a common binary bit
      (the triple-AND condition) — w morphs a carved cube into a sparse
      Sierpinski-simplex-like dust
    - Cantor Dust 4D: 3D Cantor dust in (x,y,z) whose recursion depth
      grows with w — scrubbing w plays the Cantor construction
    - Hypercheckerboard: odd parity of 4D cell coordinates
    - Diamond Fractal: thin concentric L1-distance (taxicab) shells

Performance:
    - Per-w-plane generation bounds the lattice workspace at grid^3; the grouped
      normal-aware AO pass is output-sized (23.7 GB peak RSS at the default grid)
    - All 6 fractals, grouped AO, and writing take about 25 minutes at the default grid

Usage:
    python demo_4d_fractals.py [--grid=N]

Controls:
    - Press '1' to select FRACTAL TYPE, then [/] to switch fractals
    - Press '2' to select W dimension, then [/] to navigate 4D slices
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
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["fractals_4d"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.shading import bake_ambient_occlusion
from luxar.utils.paths import get_demos_output_dir

#: Lattice resolution per axis. Four times the former 50 — the resolution bump
#: this demo was asked for. Affordable only because generation is now per
#: w-plane and only surface voxels are kept; the old whole-lattice generator
#: would need 26 GB here.
GRID_SIZE_DEFAULT = 200
#: Materialise every Nth w-plane. 200/4 = 50 slider stops, the same slider the
#: demo has always had, with four times the detail inside each slice.
W_STRIDE = 4
#: PER-PLANE point budget. Deliberately per-plane: the global cap it replaces
#: spread a fixed total across every plane, so a finer grid made each slice
#: SPARSER — the opposite of the intended effect. 150k covers about half the
#: surface voxels of the densest fractal at grid 200; at 80k the flat faces
#: still read as stippled. The store compresses well (lattice coordinates), so
#: the measured AO-shaded default output is about 147 MB on disk.
TARGET_MAX_POINTS_PER_PLANE = 150_000
#: Direction budget for the grouped surface AO bake. Twenty-four is half the
#: library's measured floor for normal-weighted AO, chosen to bound peak memory;
#: on representative slices it adds about 20% excess AO variance versus 48.
AO_N_DIRECTIONS = 24
#: Layers-panel display window re-measured after AO. The grouped bake's mean
#: multiplier is 0.570 at the default grid, so 1.971 × 0.570 preserves the
#: previously authored mean emission while keeping the AO contrast.
DISPLAY_MAX = 1.123


def axis_world_values(grid_size: int) -> np.ndarray:
    """World coordinate of each grid index, on the viewer's snap grid.

    The viewer anchors discrete navigation at the declared range minimum.
    The scene range is built from the first and last materialised planes, so
    declaring their spacing as ``step`` makes every slider stop land exactly
    on a data plane.

    Args:
        grid_size: Number of samples per axis

    Returns:
        Float64 array of world coordinates in [-1, 1), exact step multiples
    """
    step = 2.0 / grid_size
    k = np.arange(grid_size) - grid_size // 2
    return k * step


def xor_fractal_4d(
    IW: np.ndarray, IX: np.ndarray, IY: np.ndarray, IZ: np.ndarray, grid_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """XOR fractal: high values of (w⊕x⊕y⊕z) form self-similar shells.

    For any fixed w, ``w ⊕ (x⊕y⊕z)`` still spans (almost) the full value
    range, so the kept top-quartile is present in every w-slice.

    Args:
        IW, IX, IY, IZ: Integer grid-index arrays
        grid_size: Unused here (the XOR rule is scale-free); kept only so
            every fractal shares the dispatch signature in generate_4d_fractal

    Returns:
        Tuple of (keep mask, color values)
    """
    values = IW ^ IX ^ IY ^ IZ
    threshold = np.percentile(values, 75)
    keep = values >= threshold
    return keep, values


def menger_sponge_4d(
    IW: np.ndarray,
    IX: np.ndarray,
    IY: np.ndarray,
    IZ: np.ndarray,
    grid_size: int,
    level: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """4D Menger sponge - recursive hypercube with holes.

    Grid indices are mapped to ternary cell coordinates in [0, 3^level);
    a point is a hole if, at any recursion level, two or more of its four
    ternary digits are in the middle third. Every w-slice is solid
    somewhere: whatever digits w has, (x, y, z) cells with no middle
    digits always survive.

    Args:
        IW, IX, IY, IZ: Integer grid-index arrays
        grid_size: Grid resolution
        level: Recursion depth

    Returns:
        Tuple of (keep mask, color values = shallowest near-hole level)
    """
    scale = 3**level
    tw = (IW * scale) // grid_size
    tx = (IX * scale) // grid_size
    ty = (IY * scale) // grid_size
    tz = (IZ * scale) // grid_size

    solid = np.ones(IW.shape, dtype=bool)
    # Color: the coarsest level at which the point sits next to a hole
    # (exactly one middle digit) — paints the recursive surface structure.
    depth = np.zeros(IW.shape, dtype=np.int8)

    for lev in range(level):
        div = 3 ** (level - 1 - lev)  # coarsest subdivision first
        wm = (tw // div) % 3 == 1
        xm = (tx // div) % 3 == 1
        ym = (ty // div) % 3 == 1
        zm = (tz // div) % 3 == 1

        middle_count = (
            wm.astype(np.int8)
            + xm.astype(np.int8)
            + ym.astype(np.int8)
            + zm.astype(np.int8)
        )
        solid &= middle_count < 2
        depth = np.where((depth == 0) & (middle_count == 1), lev + 1, depth)

    return solid, depth


def sierpinski_4d(
    IW: np.ndarray, IX: np.ndarray, IY: np.ndarray, IZ: np.ndarray, grid_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """4D Sierpinski: no three coordinates share a common binary bit.

    The triple-AND condition (the 4D analogue of the Sierpinski
    tetrahedron's pairwise condition, one rung denser so a 50^4 grid
    stays visibly populated). At w=0 it reduces to the 3D condition
    (x&y&z)==0 (a carved cube); as w gains bits the slice thins towards
    a sparse simplex dust — but never empties, since (x, y, z) with
    pairwise-disjoint bits satisfy the condition for ANY w.

    Args:
        IW, IX, IY, IZ: Integer grid-index arrays
        grid_size: Grid resolution

    Returns:
        Tuple of (keep mask, color values = popcount of OR, a depth proxy)
    """
    triple = (IW & IX & IY) | (IW & IX & IZ) | (IW & IY & IZ) | (IX & IY & IZ)
    keep = triple == 0

    combined = IW | IX | IY | IZ
    n_bits = max(1, int(np.ceil(np.log2(grid_size))))
    values = np.zeros(IW.shape, dtype=np.int8)
    for b in range(n_bits):
        values += ((combined >> b) & 1).astype(np.int8)

    return keep, values


def cantor_dust_4d(
    IW: np.ndarray,
    IX: np.ndarray,
    IY: np.ndarray,
    IZ: np.ndarray,
    grid_size: int,
    max_level: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cantor dust whose recursion depth grows along w.

    A TRUE 4D Cantor dust (product of four Cantor sets) is empty on ~80%
    of w-slices — the middle thirds removed along w — which makes the w
    slider mostly blank. Instead this shows the 3D Cantor dust in
    (x, y, z) at a recursion depth that increases with w, from depth 1
    (coarse 2/3-blocks) to ``max_level`` (fine dust). Fractional depths
    fade the about-to-be-removed blocks by random subsampling, so every
    single w step visibly advances the construction.

    Args:
        IW, IX, IY, IZ: Integer grid-index arrays
        grid_size: Grid resolution
        max_level: Deepest recursion level (reached at w max)
        rng: Seeded generator for the fractional-depth fade

    Returns:
        Tuple of (keep mask, color values = per-point survival depth)
    """
    if rng is None:
        rng = np.random.default_rng(0)
    scale = 3**max_level

    def survival(idx: np.ndarray) -> np.ndarray:
        """Deepest level whose ternary digit avoids the middle third."""
        c = (idx * scale) // grid_size
        surv = np.full(idx.shape, max_level, dtype=np.int8)
        # Finest level first so coarser failures override (a point removed
        # at level 1 has survival 0 no matter its finer digits).
        for lev in range(max_level, 0, -1):
            digit = (c // 3 ** (max_level - lev)) % 3
            surv = np.where(digit == 1, lev - 1, surv).astype(np.int8)
        return surv

    sp = np.minimum(np.minimum(survival(IX), survival(IY)), survival(IZ))

    # Depth shown at each w-plane: 1 → max_level across the w range.
    t = 1.0 + (max_level - 1.0) * IW / max(1, grid_size - 1)
    t_floor = np.floor(t)
    frac = t - t_floor

    keep = sp >= np.ceil(t)
    fading = (sp == t_floor) & (frac > 0)
    if fading.any():
        u = rng.random(IW.shape, dtype=np.float32)
        keep |= fading & (u >= frac)

    return keep, sp


def checkerboard_4d(
    IW: np.ndarray,
    IX: np.ndarray,
    IY: np.ndarray,
    IZ: np.ndarray,
    grid_size: int,
    cells: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """4D hypercheckerboard: odd-parity cells of a cells^4 grid.

    Exactly half the cells of every w-slice are lit, and the lit pattern
    inverts each time w crosses a cell boundary.

    Args:
        IW, IX, IY, IZ: Integer grid-index arrays
        grid_size: Grid resolution
        cells: Checker cells per axis

    Returns:
        Tuple of (keep mask, color values = 4D cell diagonal index)
    """
    cw = (IW * cells) // grid_size
    cx = (IX * cells) // grid_size
    cy = (IY * cells) // grid_size
    cz = (IZ * cells) // grid_size

    diag = cw + cx + cy + cz
    keep = diag % 2 == 1
    return keep, diag


def diamond_fractal_4d(
    IW: np.ndarray,
    IX: np.ndarray,
    IY: np.ndarray,
    IZ: np.ndarray,
    grid_size: int,
    shells_per_unit: float = 2.5,
    duty: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Concentric L1-distance (taxicab) shells in world coordinates.

    Keeps thin shells where the fractional part of ``dist × shells_per_unit``
    is below ``duty``. Any w-slice spans ≥3 units of L1 distance, so it
    always cuts through many shells; growing |w| sweeps the shells inward.

    Args:
        IW, IX, IY, IZ: Integer grid-index arrays
        grid_size: Grid resolution
        shells_per_unit: Shell spatial frequency
        duty: Kept fraction of each shell period (shell thickness)

    Returns:
        Tuple of (keep mask, color values = shell index)
    """
    axis = np.abs(axis_world_values(grid_size)).astype(np.float32)
    dist = axis[IW] + axis[IX] + axis[IY] + axis[IZ]

    phase = dist * shells_per_unit
    keep = (phase % 1.0) < duty
    values = phase.astype(np.int16)
    return keep, values


def _fractal_rule(fractal_type: int, rng: np.random.Generator):
    """``(name, fn)`` for one fractal type. ``fn(IW, IX, IY, IZ, n) -> (keep, values)``."""
    rules = {
        0: ("XOR Fractal", xor_fractal_4d),
        1: (
            "Menger Sponge 4D",
            lambda w, x, y, z, n: menger_sponge_4d(w, x, y, z, n, level=3),
        ),
        2: ("Sierpinski 4D", sierpinski_4d),
        3: (
            "Cantor Dust 4D",
            lambda w, x, y, z, n: cantor_dust_4d(w, x, y, z, n, max_level=4, rng=rng),
        ),
        4: (
            "Hypercheckerboard",
            lambda w, x, y, z, n: checkerboard_4d(w, x, y, z, n, cells=5),
        ),
        5: ("Diamond Fractal", diamond_fractal_4d),
    }
    return rules[fractal_type]


def surface_of(mask: np.ndarray) -> np.ndarray:
    """The boundary voxels of a 3D solid: kept, with a 6-neighbour outside it.

    Interior voxels of a solid are never visible and are what made these
    fractals read as fuzzy: with a 25%-filling set, every ray summed dozens of
    hidden points into a haze that buried the surface it was sitting behind.
    Dropping them sharpens the render AND is what makes a 4x finer grid
    affordable — measured at grid 200 it removes 85% of the XOR set, 86% of the
    hypercheckerboard and 82% of the diamond, while leaving the already-thin
    Sierpinski and Cantor sets almost untouched (they are nearly all surface).

    Voxels on the array's own face count as exposed: a solid cut by the edge of
    the grid has a real boundary there.
    """
    interior = np.ones_like(mask)
    for axis in range(3):
        for shift in (1, -1):
            neighbour = np.roll(mask, shift, axis=axis)
            face = [slice(None)] * 3
            face[axis] = 0 if shift == 1 else -1
            neighbour[tuple(face)] = False
            interior &= neighbour
    return mask & ~interior


def surface_normals(mask: np.ndarray) -> np.ndarray:
    """Outward unit normals from the occupancy gradient of a 3D solid.

    The gradient is measured before :func:`surface_of` removes the interior, so
    it describes the solid the visible points came from rather than the
    one-voxel shell left for rendering. Degenerate rows stay zero and therefore
    use the AO implementation's documented full-sphere fallback.
    """
    gradients = np.gradient(mask.astype(np.float32))
    normals = -np.stack(gradients, axis=-1)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    return np.divide(
        normals,
        lengths,
        out=np.zeros_like(normals),
        where=lengths > 1e-12,
    )


def materialised_w_planes(grid_size: int, stride: int) -> np.ndarray:
    """Indices of the w-planes actually written, as a strided subset.

    The fractal RULE is evaluated on the full ``grid_size`` lattice, so the
    geometry inside every slice is at full resolution; only the number of
    SLIDER STOPS is reduced. At the default grid 200 / stride 4 that is 50
    stops — the same slider the demo has always had — with four times the
    linear detail in each one. The starting phase keeps the historical plane
    selection stable across supported grid sizes.
    """
    phase = grid_size // 2 % stride
    return np.arange(phase, grid_size, stride, dtype=np.int64)


def generate_4d_fractal(
    fractal_type: int,
    grid_size: int = GRID_SIZE_DEFAULT,
    *,
    w_stride: int = W_STRIDE,
    surface_only: bool = True,
    max_points_per_plane: int = TARGET_MAX_POINTS_PER_PLANE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one 4D geometric fractal, one w-plane at a time.

    Evaluated PER W-PLANE rather than over the whole 4D lattice at once. The
    old version built four ``grid_size**4`` meshgrids up front, which is 100 MB
    at grid 50 and 26 GB at grid 200 — the reason the grid was stuck at 50 and
    the slices came out coarse. Per-plane the working set is ``grid_size**3``,
    so a four-times-finer grid costs a few hundred MB instead of being
    impossible.

    Args:
        fractal_type: 0-5 indicating which fractal.
        grid_size: Lattice resolution per axis.
        w_stride: Materialise every ``w_stride``-th w-plane (slider stops).
        surface_only: Keep only boundary voxels (see :func:`surface_of`).
        max_points_per_plane: Per-plane budget. A PER-PLANE cap, not a global
            one: the global cap it replaces divided a fixed budget across every
            plane, so raising the grid made each individual slice sparser —
            exactly backwards for "make it less fuzzy".

    Returns:
        ``(positions, values, normals)`` for points in the fractal.

    Raises:
        ValueError: If ``grid_size < 3`` (below that, some fractal rules have
            no odd-parity cells and cannot populate every w-plane).
        RuntimeError: If any materialised w-plane is empty (every slider stop
            must show structure — this is the demo's core contract).
    """
    if grid_size < 3:
        raise ValueError(f"grid_size must be >= 3, got {grid_size}")

    rng = np.random.default_rng(42 + fractal_type)
    fractal_name, fractal_func = _fractal_rule(fractal_type, rng)
    planes = materialised_w_planes(grid_size, w_stride)
    aprint(f"  Grid: {grid_size}^4, {len(planes)} materialised w-planes")
    aprint(f"  Computing {fractal_name} (surface_only={surface_only})...")

    axis = axis_world_values(grid_size).astype(np.float32)
    coords = np.arange(grid_size, dtype=np.int32)
    IX, IY, IZ = np.meshgrid(coords, coords, coords, indexing="ij")

    per_plane_counts: list[int] = []
    chunks_pos: list[np.ndarray] = []
    chunks_val: list[np.ndarray] = []
    chunks_normal: list[np.ndarray] = []

    for w_index in planes:
        IW = np.full_like(IX, w_index)
        keep, values = fractal_func(IW, IX, IY, IZ, grid_size)
        normal_field = surface_normals(keep)
        if surface_only:
            keep = surface_of(keep)

        n_kept = int(keep.sum())
        if n_kept > max_points_per_plane:
            flat = np.flatnonzero(keep)
            chosen = rng.choice(flat, size=max_points_per_plane, replace=False)
            keep = np.zeros(keep.shape, dtype=bool)
            keep.ravel()[chosen] = True
            n_kept = max_points_per_plane
        per_plane_counts.append(n_kept)
        if n_kept == 0:
            continue

        chunks_pos.append(
            np.column_stack(
                [
                    np.full(n_kept, axis[w_index], dtype=np.float32),
                    axis[IX[keep]],
                    axis[IY[keep]],
                    axis[IZ[keep]],
                ]
            )
        )
        chunks_val.append(values[keep])
        chunks_normal.append(normal_field[keep])

    counts = np.asarray(per_plane_counts)
    if (counts == 0).any():
        empty = planes[np.flatnonzero(counts == 0)].tolist()
        raise RuntimeError(
            f"{fractal_name}: empty w-planes {empty} — every slider "
            f"stop must show structure"
        )
    aprint(
        f"    Per-w-plane points: min={counts.min():,} "
        f"median={int(np.median(counts)):,} max={counts.max():,}"
    )

    positions = np.vstack(chunks_pos)
    pattern_values = np.concatenate(chunks_val)
    normals = np.vstack(chunks_normal)
    aprint(f"  ✓ {fractal_name}: {len(positions):,} points")
    return positions, pattern_values, normals


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


def apply_fractal_ambient_occlusion(
    positions_5d: np.ndarray,
    normals: np.ndarray,
    base_colors: np.ndarray,
) -> np.ndarray:
    """Bake surface AO without crossing the hidden fractal or w dimensions.

    Rows must remain grouped by ``(fractal, w)`` as emitted by the generator;
    run-based labels avoid sorting the 44.5-million-row production array.
    """
    joint_keys = positions_5d[:, :2]
    group_starts = np.ones(len(positions_5d), dtype=bool)
    group_starts[1:] = np.any(joint_keys[1:] != joint_keys[:-1], axis=1)
    group_by = np.cumsum(group_starts, dtype=np.int32) - 1

    shade = bake_ambient_occlusion(
        positions_5d,
        normals=normals,
        occluder="opaque",
        n_directions=AO_N_DIRECTIONS,
        spatial_dims=(2, 3, 4),
        group_by=group_by,
    )
    aprint(
        f"  Ambient occlusion: {int(group_by[-1]) + 1} fractal/w groups, "
        f"range [{shade.min():.3f}, {shade.max():.3f}], mean {shade.mean():.3f}"
    )
    return (base_colors * shade[:, None]).astype(np.float32)


def generate_4d_fractal_dataset(
    output_path: Path,
    grid_size: int = GRID_SIZE_DEFAULT,
) -> int:
    """Generate complete 4D fractal dataset with multiple types.

    Args:
        output_path: Where to write zarr
        grid_size: Grid resolution (default 200; every W_STRIDE-th w-plane is
            materialised, at up to TARGET_MAX_POINTS_PER_PLANE points each)

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
    all_normals = []

    with asection(f"Generating 6 × 4D Fractals (grid: {grid_size}^4)"):
        for frac_id, frac_name in enumerate(fractal_names):
            with asection(f"Fractal {frac_id}: {frac_name}"):
                # Generate fractal (FAST!)
                positions, pattern_values, normals = generate_4d_fractal(
                    frac_id,
                    grid_size=grid_size,
                )

                # Generate colors from pattern values
                colors = pattern_values_to_colors(pattern_values)

                # Add fractal type ID as first coordinate
                fractal_ids = np.full(len(positions), frac_id, dtype=np.float32)

                all_positions.append(
                    np.column_stack(
                        [
                            fractal_ids,
                            positions[:, 0],  # W
                            positions[:, 1],  # X
                            positions[:, 2],  # Y
                            positions[:, 3],  # Z
                        ]
                    )
                )
                all_colors.append(colors)
                all_normals.append(normals)

    # Combine all fractals
    with asection("Combining all fractals"):
        positions_5d = np.vstack(all_positions)
        all_positions.clear()
        normals = np.vstack(all_normals)
        all_normals.clear()
        base_colors = np.vstack(all_colors)
        all_colors.clear()
        colors = apply_fractal_ambient_occlusion(
            positions_5d,
            normals,
            base_colors,
        )

        aprint(f"✓ Total points: {len(positions_5d):,}")
        aprint(f"  Per fractal: {len(positions_5d) // 6:,} avg")

    # Write to Zarr
    with asection("Writing to Zarr"):
        # The w Dimension must mirror the MATERIALISED planes, not the full
        # lattice: only every W_STRIDE-th plane is written, so the slider's snap
        # grid is `stride x 2/grid_size` and its range ends on the first and
        # last plane that actually holds data. Declaring the full-lattice step
        # here would put most slider stops on empty space.
        axis = axis_world_values(grid_size)
        planes = materialised_w_planes(grid_size, W_STRIDE)
        axis = axis[planes]
        w_step = W_STRIDE * 2.0 / grid_size

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
                    range=(float(axis[0]), float(axis[-1])),
                    step=w_step,
                    display=False,
                    discrete=True,
                    description="4th spatial dimension (navigate to see slices!)",
                ),
                Dimension("x", unit="", range=(-1.0, 1.0), display=True),
                Dimension("y", unit="", range=(-1.0, 1.0), display=True),
                Dimension("z", unit="", range=(-1.0, 1.0), display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
            )

            # Point radius DERIVED from the lattice step (2/grid_size), not
            # pinned. The old constant 0.025 was 62% of the step at grid 50; at
            # grid 200 the step is 0.01 and the same constant would draw each
            # voxel two and a half times oversized — the exact "fuzzy" the
            # higher resolution is meant to remove. 0.62 of a step keeps the
            # surface closed without smearing it.
            voxel_step = 2.0 / grid_size
            point_radius = 0.62 * voxel_step
            aprint(f"Point radius: {point_radius:.5f} (step {voxel_step:.5f})")
            radii = np.full(len(positions_5d), point_radius, dtype=np.float32)
            # Crisper profile than the old 0.55: with the interior voxels gone
            # there is nothing behind a surface point that a soft edge is
            # helping to blend into.
            sharpnesses = np.full(len(positions_5d), 0.75, dtype=np.float32)

            scene.add_points(
                "Fractals4D",
                positions_5d,
                colors=colors,
                radii=radii,
                sharpness=sharpnesses,
                # Dialled in live in the Layers panel and copied back here, so
                # the demo OPENS on the settings someone actually chose rather
                # than on defaults they then have to rediscover.
                #
                # `volumetric` (emission-absorption) rather than the previous
                # additive default: these are dense solid shells, and summing
                # along the ray flattened the near and far faces of a slice into
                # one silhouette. With absorption the near surface occludes and
                # the fractal reads as a SOLID.
                blending_mode="volumetric",
                opacity=0.43,
                absorption=1.23,
                gamma=1.0,
                # `intensity` is the display WINDOW, not a gain: the viewer
                # recovers [-offset/i, (1-offset)/i]. AO lowers mean emission
                # to 0.570 of the unshaded value, so the previous 1.971 window
                # becomes 1.123 to preserve the authored mean brightness.
                intensity=1.0 / DISPLAY_MAX,
                layer=True,
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
            add_demo_caption(
                scene, "6 fractal types • 4D space", DEMO_META.get("citation")
            )

        aprint(f"✓ Written to {output_path}")
        disk_bytes = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        aprint(f"✓ Dataset size: {disk_bytes / 1024 / 1024:.0f} MB on disk")

    return len(positions_5d)


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    grid_size = GRID_SIZE_DEFAULT

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--grid="):
                grid_size = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("4D FRACTAL EXPLORER DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Explore 6 different 4D geometric fractals!")
    aprint(f"Grid: {grid_size}^4 = {grid_size**4:,} samples per fractal")
    aprint(f"Candidate points per 3D slice: ~{grid_size**3:,}")
    aprint("")
    aprint("Fractal types:")
    aprint("  0: XOR Fractal - Bitwise XOR creates self-similar shells")
    aprint("  1: 4D Menger Sponge - Recursive hypercube with holes")
    aprint("  2: 4D Sierpinski - No 3 coordinates share a binary bit")
    aprint("  3: 4D Cantor Dust - Recursion depth grows along w")
    aprint("  4: 4D Hypercheckerboard - Alternating parity cells")
    aprint("  5: 4D Diamond Fractal - Concentric taxicab shells")
    aprint("")
    aprint("⏱️  Generation time: ~25 minutes at the default grid")
    aprint("   Peak memory: ~24 GB; use a smaller --grid to reduce time and memory.")
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
        aprint("  2. Press '2' → Select W dimension")
        aprint("     Press '['/']' to navigate through 4D slices")
        aprint("     Every stop shows structure - watch it evolve!")
        aprint("")
        aprint("What you're seeing:")
        aprint("  • 3D slices through 4D fractals")
        aprint("  • Colors show local structure (depth, shell, cell index)")
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
