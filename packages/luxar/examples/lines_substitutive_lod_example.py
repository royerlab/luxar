#!/usr/bin/env python3
"""Lines Substitutive LOD Example — coarse levels rendered as Gaussian splats.

The three-geometry-symmetry counterpart of ``points_substitutive_lod_example.py``.
Lines coarsen *additively* by decimation (drop whole polylines), which thins a
dense line field at coarse zoom. ``substitutive_lod=`` instead synthesises
**mass-preserving Gaussian splats**: each segment is lifted to a string of
isotropic "bead" gaussians (view-independent, summing to a smooth tube), the
gsplat substitutive pipeline builds fewer-but-larger representatives, and those
become the coarse levels. The finest LOD child stays the original Lines node.

Layout: a dense bundle of short curved fibres authored two ways —
- ``substitutive``: coarse levels are synthesised gsplats (this feature).
- ``additive``: coarse levels are decimated polylines (the baseline).

Zoom OUT on ``substitutive`` and the fibre field stays smooth and equally bright
as the viewer swaps in the coarse gsplat levels; on ``additive`` the fibres thin.
Both report type "lines" in the layers panel.
"""

import numpy as np
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_fibres(n_fibres: int = 4000, seed: int = 0):
    """A bundle of short 3-vertex curved fibres (polylines) over a volume."""
    rng = np.random.default_rng(seed)
    starts = rng.uniform(-60, 60, (n_fibres, 3))
    dirs = rng.normal(0, 1, (n_fibres, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bend = rng.normal(0, 2.5, (n_fibres, 3))
    # 3 vertices per fibre: start, mid (bent), end.
    mids = starts + dirs * 4.0 + bend
    ends = starts + dirs * 8.0
    verts = np.empty((n_fibres * 3, 3), dtype=np.float32)
    verts[0::3], verts[1::3], verts[2::3] = starts, mids, ends
    # indexed edges: (3i,3i+1) and (3i+1,3i+2) per fibre.
    idx = np.empty((n_fibres * 2, 2), dtype=np.int64)
    base = np.arange(n_fibres) * 3
    idx[0::2] = np.stack([base, base + 1], 1)
    idx[1::2] = np.stack([base + 1, base + 2], 1)
    widths = np.full(verts.shape[0], 0.6, dtype=np.float32)
    colors = np.tile(np.array([0.2, 0.8, 1.0], np.float32), (verts.shape[0], 1))
    return verts, idx, widths, colors


def main() -> None:
    output_path = (
        get_examples_output_dir() / "lines_substitutive_lod_example.luxar.zarr"
    )
    with LuxarZarrCompiler(str(output_path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        verts, idx, widths, colors = make_fibres()

        # 1. Substitutive LOD — coarse levels are synthesised gsplats.
        scene.add_lines(
            "substitutive", verts, widths, colors=colors,
            indices=idx, line_type="indexed", layer=True,
            substitutive_lod=dict(compression_factor=4, levels=3, device="cpu"),
        )

        # 2. Additive LOD (baseline) — coarse levels are decimated polylines.
        shifted = verts.copy()
        shifted[:, 0] += 160.0
        scene.add_lines(
            "additive", shifted, widths, colors=colors,
            indices=idx, line_type="indexed", layer=True,
            additive_lod=dict(n_lods=4, method="random", seed=0),
        )

    aprint(f"\nScene saved to: {output_path}")
    aprint(f"To view: luxar serve --viewer {output_path}")
    aprint(
        "  → Zoom OUT on 'substitutive': coarse gsplat levels keep the fibre\n"
        "     field smooth and equally bright (mass-preserving).\n"
        "  → Zoom OUT on 'additive': decimated fibres thin out.\n"
        "  → Both report type 'lines' in the layers panel."
    )


if __name__ == "__main__":
    main()
