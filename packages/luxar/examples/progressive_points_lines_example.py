#!/usr/bin/env python3
"""Progressive Points and Lines Example — multi-additive LOD loading.

This example demonstrates the multi-additive-LOD (progressive) loading
path landed in PR #318. The Python writer produces ``additive_<i>/``
zarr subgroups under a single Points / Lines node; the viewer loads
each subgroup in coarse-to-fine order yielded by
``requestAnimationFrame``, so the user sees a coarse first paint that
refines over a few frames.

This example demonstrates:
- ``add_points(additive_lod=...)`` with the default ``random`` method.
- ``add_lines(additive_lod=...)`` with the ``spatial-uniform`` method
  (per-polyline stratified-grid sampling) — visually proves the
  refinement covers the dataset evenly at every level.
- Two side-by-side nodes so you can compare progressive vs flat layers
  in the layers panel.

Educational value:
- Watch the viewer paint coarse-first and refine to full detail in
  a few rAF ticks. Open the layer panel to confirm both progressive
  layers report ``N additive LODs`` on their type badge.
- Learn the ``additive_lod=True`` shorthand and the
  ``additive_lod=dict(n_lods=N, method=...)`` long form.

PR α (#318) — Progressive multi-additive-LOD loading.
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_dense_point_cloud(n: int = 200_000, seed: int = 0):
    """A large enough cloud that progressive refinement is visible."""
    rng = np.random.default_rng(seed)
    # Mixture of a dense central blob + a thin orbiting ring → tests
    # that all density modes show up at every LOD level.
    blob = rng.normal(0, 1.5, (n // 2, 3))
    theta = rng.uniform(0, 2 * np.pi, n // 2)
    ring = np.stack(
        [
            6.0 * np.cos(theta) + rng.normal(0, 0.1, n // 2),
            6.0 * np.sin(theta) + rng.normal(0, 0.1, n // 2),
            rng.normal(0, 0.3, n // 2),
        ],
        axis=1,
    )
    return np.concatenate([blob, ring], axis=0).astype(np.float32)


def make_polyline_bundle(n_polylines: int = 800, vertices_per: int = 8, seed: int = 1):
    """Many short polylines, scattered across the volume → polyline-level LOD."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-6, 6, (n_polylines, 3))
    offsets = rng.normal(0, 0.2, (n_polylines, vertices_per, 3))
    verts = (centers[:, None, :] + offsets).reshape(-1, 3).astype(np.float32)
    # Flatten as segments-typed lines: vertex pairs per polyline. With
    # vertices_per=8 we get 4 segments per polyline → 3200 segments.
    # The progressive loader treats each polyline atomically.
    seg_verts = verts.reshape(n_polylines, vertices_per, 3)
    pairs = []
    for poly in seg_verts:
        for i in range(0, vertices_per - 1, 2):
            pairs.append(poly[i])
            pairs.append(poly[i + 1])
    return np.array(pairs, dtype=np.float32)


def main() -> None:
    output_path = get_examples_output_dir() / "progressive_points_lines_example.zarr"

    with LuxarZarrCompiler(str(output_path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # === 1. Progressive Points (random ordering) ===
        # 200k points across 4 LOD levels. The viewer paints the first
        # ~50k immediately, then refines through the remaining 3 levels
        # on subsequent animation frames.
        positions = make_dense_point_cloud(n=200_000)
        # Per-point cyan (tinted by Z so the LOD refinement is visible
        # — uniform color across a dense cloud hides which points
        # arrived at which level).
        z_norm = (positions[:, 2] - positions[:, 2].min()) / (
            positions[:, 2].max() - positions[:, 2].min() + 1e-9
        )
        colors_pts = np.stack(
            [0.3 + 0.0 * z_norm, 0.5 + 0.5 * z_norm, np.full_like(z_norm, 1.0)],
            axis=1,
        ).astype(np.float32)
        scene.add_points(
            "progressive_points",
            positions,
            colors=colors_pts,
            radii=0.04,
            layer=True,
            additive_lod=True,  # shorthand: 4 random-ordered LOD levels
        )

        # === 2. Progressive Lines (spatial-uniform ordering) ===
        # The spatial-uniform method bins polyline centroids into a
        # stratified grid so every LOD level covers the full bbox.
        # Visually: each refinement adds detail uniformly rather than
        # in one cluster first.
        vertices = make_polyline_bundle(n_polylines=800, vertices_per=8)
        # Per-vertex orange tinted by X so each LOD level's refinement
        # is visually distinguishable.
        x_norm = (vertices[:, 0] - vertices[:, 0].min()) / (
            vertices[:, 0].max() - vertices[:, 0].min() + 1e-9
        )
        colors_lines = np.stack(
            [
                np.full_like(x_norm, 1.0),
                0.3 + 0.4 * x_norm,
                np.full_like(x_norm, 0.2),
            ],
            axis=1,
        ).astype(np.float32)
        scene.add_lines(
            "progressive_lines",
            vertices=vertices,
            widths=0.04,
            colors=colors_lines,
            line_type="segments",
            layer=True,
            additive_lod=dict(
                method="spatial-uniform",
                n_lods=4,
            ),
        )

        aprint(
            f"Created progressive scene with {len(positions):,} points + "
            f"{len(vertices) // 2:,} line segments across 4 LOD levels each."
        )

    aprint(f"\nScene saved to: {output_path}")
    aprint(f"To view: luxar serve --viewer {output_path}")
    aprint(
        "  → Open the layers panel. Both nodes show their N-LODs badge.\n"
        "  → On load you should see a coarse first paint, then ~3 frames "
        "of refinement to full detail."
    )


if __name__ == "__main__":
    main()
