#!/usr/bin/env python3
"""Lines Split + Sampling Example — polyline BSP, Poisson-disk, SAH.

Demonstrates the three opt-in feature extensions landed in PR #320:

1. **``add_lines(split=...)``** — polyline-centroid BSP. Whole polylines
   stay atomic; the BSP runs over per-polyline centroids so each part
   is a clean spatial chunk of the original line bundle.
2. **Poisson-disk LOD sampling** — ``additive_lod=dict(method='poisson-disk')``
   uses a Bridson sampler with cell-grid acceleration. Compare visually
   against the default ``random`` and the deterministic
   ``spatial-uniform`` methods.
3. **SAH BSP variant** — ``split=dict(rule='sah', ...)`` swaps the
   default midpoint BSP for a surface-area-heuristic partitioner.
   Useful on heavily skewed datasets (one dense cluster + a long
   streamer).

Layout: three side-by-side line bundles to compare splitting + sampling
methods on the same input shape.

Educational value:
- ``split=`` on lines never breaks a polyline across part boundaries.
- Poisson-disk LOD subsets show visibly more uniform spacing than
  ``method='random'`` (zoom in: random clusters; poisson-disk doesn't).
- SAH and midpoint BSPs produce different part shapes on skewed data;
  SAH typically yields fewer but more frustum-friendly parts.

PR γ (#320) — Lines split + Poisson-disk + SAH BSP.
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_line_bundle(n_segments: int, bbox: tuple, seed: int = 0):
    """N short random line segments inside ``bbox``."""
    rng = np.random.default_rng(seed)
    lo, hi = bbox
    starts = rng.uniform(lo, hi, (n_segments, 3))
    # Short, slightly noisy: each segment is a 0.5-unit jitter from its start.
    ends = starts + rng.normal(0, 0.5, (n_segments, 3))
    verts = np.empty((2 * n_segments, 3), dtype=np.float32)
    verts[0::2] = starts
    verts[1::2] = ends
    return verts


def make_skewed_bundle(n_segments: int, seed: int = 0):
    """A bundle skewed strongly along X — interesting for SAH vs midpoint."""
    rng = np.random.default_rng(seed)
    # Half the segments in a long thin streamer, half in a dense cluster.
    n_streamer = n_segments // 2
    n_cluster = n_segments - n_streamer
    starts_streamer = np.stack(
        [
            rng.uniform(-15, 15, n_streamer),
            rng.uniform(-0.3, 0.3, n_streamer),
            rng.uniform(-0.3, 0.3, n_streamer),
        ],
        axis=1,
    )
    starts_cluster = rng.normal([0, 0, 0], [0.8, 0.8, 0.8], (n_cluster, 3))
    starts = np.concatenate([starts_streamer, starts_cluster], axis=0)
    ends = starts + rng.normal(0, 0.2, starts.shape)
    verts = np.empty((2 * n_segments, 3), dtype=np.float32)
    verts[0::2] = starts
    verts[1::2] = ends
    return verts.astype(np.float32)


def main() -> None:
    output_path = get_examples_output_dir() / "lines_split_and_sampling_example.zarr"

    with LuxarZarrCompiler(str(output_path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # === 1. add_lines(split=...) — polyline-atomic BSP ===
        # 5000 segments across a wide spatial range → BSP carves spatial
        # parts. Polylines are atomic: every vertex of a segment ends
        # up in exactly one part.
        verts_split = make_line_bundle(
            n_segments=5000, bbox=(-10, 10), seed=0
        )
        # Translate to its own region (X-).
        verts_split[:, 0] -= 25.0
        # Per-vertex green colors (uniform); the writer's sort path
        # requires (N, 3) when split= is active.
        colors_split = np.full(
            (verts_split.shape[0], 3), [0.2, 0.9, 0.4], dtype=np.float32
        )
        scene.add_lines(
            "split_lines",
            vertices=verts_split,
            widths=0.04,
            colors=colors_split,
            line_type="segments",
            split=dict(max_elements=1500),  # produces ~4 parts
            layer=True,
        )

        # === 2. Poisson-disk LOD sampling ===
        # Same shape as #1 but with a 4-level progressive ladder using
        # the Poisson-disk method. Zoom in to compare its blue-noise
        # subset against the random default.
        verts_poisson = make_line_bundle(
            n_segments=5000, bbox=(-10, 10), seed=1
        )
        colors_poisson = np.full(
            (verts_poisson.shape[0], 3), [0.9, 0.3, 0.6], dtype=np.float32
        )
        scene.add_lines(
            "poisson_disk_lines",
            vertices=verts_poisson,
            widths=0.04,
            colors=colors_poisson,
            line_type="segments",
            additive_lod=dict(method="poisson-disk", n_lods=4),
            layer=True,
        )

        # === 3. SAH BSP variant on a skewed dataset ===
        # The streamer-plus-cluster shape is exactly what SAH was
        # designed for. Both runs use the same input; only the rule
        # differs. SAH typically produces fewer parts and respects the
        # streamer's anisotropy better.
        verts_skewed = make_skewed_bundle(n_segments=5000, seed=2)
        # Translate to X+ region.
        verts_skewed[:, 0] += 25.0
        colors_skewed = np.full(
            (verts_skewed.shape[0], 3), [0.4, 0.5, 1.0], dtype=np.float32
        )
        scene.add_lines(
            "sah_split_lines",
            vertices=verts_skewed,
            widths=0.04,
            colors=colors_skewed,
            line_type="segments",
            split=dict(max_elements=1500, rule="sah"),
            layer=True,
        )

        aprint(
            "Created three lines bundles:\n"
            "  - split_lines: 5k segments → ~N polyline-atomic BSP parts\n"
            "  - poisson_disk_lines: 5k segments → 4-level Poisson-disk LOD\n"
            "  - sah_split_lines: 5k skewed segments → SAH BSP partition"
        )

    aprint(f"\nScene saved to: {output_path}")
    aprint(f"To view: luxar serve --viewer {output_path}")
    aprint(
        "  → Three layers visible in the panel.\n"
        "  → 'split_lines' and 'sah_split_lines' show 'N parts' badges.\n"
        "  → 'poisson_disk_lines' shows a '4 additive LODs' badge; on\n"
        "     load you should see coarse-first refinement with a blue-\n"
        "     noise spacing pattern (no obvious clusters).\n"
        "  → Hover any line: tooltip reports the WRAPPER path for the\n"
        "     two split layers (split-aware picking from PR β)."
    )


if __name__ == "__main__":
    main()
