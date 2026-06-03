#!/usr/bin/env python3
"""GSplats LOD Example — substitutive level-of-detail on a fitted splat set.

This example demonstrates the **substitutive LOD** path for Gaussian
splats: a coarse-to-fine pyramid where each coarser level *replaces*
its finer level with a smaller set of synthesized representative
splats. The viewer picks a level based on projected pixel size.

This complements the four existing LOD examples
(``progressive_points_lines_example.py``, ``partition_of_lod_example.py``,
``lines_partition_and_sampling_example.py``, ``energy_breakpoints_example.py``)
which all focus on **additive** LOD (each level adds new elements on
top of the previous one). Substitutive and additive are independent
axes — together they form the 2-D LOD pyramid described in the
gsplats specs.

Pipeline:
1. Fit a small synthetic volume to get a base ``GSplatData``.
2. Build a substitutive ladder explicitly with ``make_substitutive_lod``
   using the ``greedy_lloyd`` method (bottom-up Runnalls-style merge
   warm start + cost-increment Lloyd refinement), producing 3 levels
   (the original + 2 coarsened copies).
3. Paint each substitutive level a distinct **debug color** so it is
   obvious in the viewer which level is currently being rendered.
4. Add the house-style explainer card plus a per-level color legend
   (live splat counts, each line tinted to match its level) that
   explain what the colors mean.
5. The viewer renders the appropriate level given the camera distance;
   zooming in switches to finer levels automatically — and the color
   on screen changes as it does.

Educational value:
- See substitutive LOD authored from the lower-level
  ``make_substitutive_lod`` API, then handed to the scene via
  ``add_gsplats_from_data(..., lod_group=True)``.
- Understand the ``compression_factor`` × ``levels`` parameters.
- Use per-level coloring + overlays as a debugging aid to *see* the
  level-switching the ``pixel_size`` selector performs.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.gsplats import fit_gaussian_splats, make_substitutive_lod
from luxar.gsplats.gsplat_data import (
    AdditiveSubLOD,
    GSplatData,
    SubstitutiveLevel,
)
from luxar.utils.paths import get_examples_output_dir

# Distinct, high-contrast debug colors, indexed by substitutive level
# (finest = index 0). RGB floats in [0, 1]. Coarse → warm/red so a
# coarse level (few big splats, seen when zoomed out) reads as "alarm".
LEVEL_COLORS: list[tuple[float, float, float]] = [
    (0.25, 1.00, 0.45),  # level 0 — finest   → green
    (1.00, 0.78, 0.20),  # level 1 — middle   → amber
    (1.00, 0.30, 0.32),  # level 2 — coarsest → red
]


def make_synthetic_volume(size: int = 48, seed: int = 0) -> np.ndarray:
    """Build a 3D volume with ~12 randomly-placed Gaussian blobs."""
    rng = np.random.default_rng(seed)
    n_blobs = 12
    centers = rng.integers(size // 5, 4 * size // 5, size=(n_blobs, 3))
    sigmas = rng.uniform(1.5, 3.0, size=n_blobs)
    amps = rng.uniform(0.5, 1.0, size=n_blobs)
    z_idx, y_idx, x_idx = np.ogrid[:size, :size, :size]
    volume = np.zeros((size, size, size), dtype=np.float32)
    for (cz, cy, cx), sigma, amp in zip(centers, sigmas, amps):
        d2 = (z_idx - cz) ** 2 + (y_idx - cy) ** 2 + (x_idx - cx) ** 2
        volume += amp * np.exp(-d2 / (2.0 * sigma**2))
    return volume.astype(np.float32)


def colorize_levels(ladder: GSplatData) -> GSplatData:
    """Return a copy of ``ladder`` with a solid debug color per level.

    Each substitutive level's lone additive sub-LOD is rebuilt with a
    flat ``(N, 3)`` color array set to ``LEVEL_COLORS[level_index]``.
    ``AdditiveSubLOD`` / ``SubstitutiveLevel`` are frozen, so we
    construct fresh instances rather than mutate in place.
    """
    colored_levels: list[SubstitutiveLevel] = []
    for level in ladder.substitutive_levels:
        sub = level.additive_sublods[0]
        rgb = LEVEL_COLORS[level.level_index % len(LEVEL_COLORS)]
        colors = np.tile(np.asarray(rgb, dtype=np.float32), (sub.n_splats, 1))
        colored_sub = AdditiveSubLOD(
            centers=sub.centers,
            amplitudes=sub.amplitudes,
            cholesky_factors=sub.cholesky_factors,
            colors=colors,
            stats=dict(sub.stats),
            truncation_radius=sub.truncation_radius,
        )
        colored_levels.append(
            SubstitutiveLevel(
                additive_sublods=[colored_sub],
                compression_factor=level.compression_factor,
                parent_method=level.parent_method,
                level_index=level.level_index,
                stats=dict(level.stats),
            )
        )
    return GSplatData.from_substitutive_levels(colored_levels, stats=dict(ladder.stats))


def main() -> None:
    """Fit a tiny volume and ship it as a colored substitutive LOD pyramid."""
    output_path = get_examples_output_dir() / "gsplats_lod_example.zarr"
    aprint(f"Writing substitutive-LOD example to {output_path}")

    volume = make_synthetic_volume(size=48, seed=0)
    aprint(f"Synthetic volume: shape={volume.shape}")

    # 80 seeds × 80 iterations is enough to produce a non-trivial splat
    # set for a 12-blob volume on CPU in a few seconds.
    result = fit_gaussian_splats(
        volume,
        seeds=80,
        n_iters=80,
        device="cpu",
        verbose=False,
    )
    aprint(f"Base fit: {result.centers.shape[0]} splats")

    # Build the substitutive ladder explicitly so we can paint each level.
    # compression_factor=4, levels=2 → each coarser level keeps ~N/4 splats,
    # giving the original + 2 coarsened copies = 3 substitutive levels.
    # method="greedy_lloyd": bottom-up Runnalls-style pairwise merge warm
    # start, then cost-increment Lloyd refinement (quality-leaning at small N).
    ladder = make_substitutive_lod(
        result,
        compression_factor=4,
        levels=2,
        method="greedy_lloyd",
        device="cpu",
        seed=0,
        verbose=False,
    )
    counts = [lvl.n_splats_total for lvl in ladder.substitutive_levels]
    aprint(f"Substitutive levels (finest→coarsest): {counts} splats")

    colored = colorize_levels(ladder)

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        # lod_group=True → use the stored substitutive levels as-is (don't
        # recompute); the writer emits a kind="lod" group with one colored
        # gsplats child per level, coarsest→finest.
        scene.add_gsplats_from_data(
            "lod_blobs",
            colored,
            lod_group=True,
            opacity=1.0,
            blending_mode="additive",
        )

        # ── Explanatory overlays ────────────────────────────────────────
        # Per-level color legend (finest → coarsest), each line tinted to
        # match the splats of that level. This carries the live per-level
        # splat counts, which the generic explainer card cannot — so it
        # stays as a dedicated legend (the single-card house rule allows a
        # legend alongside the explainer).
        labels = ["finest", "middle", "coarsest"]
        factors = [1, 4, 16]  # K^level for compression_factor=4
        for level in colored.substitutive_levels:
            i = level.level_index
            r, g, b = (int(round(c * 255)) for c in LEVEL_COLORS[i])
            css = f"rgb({r},{g},{b})"
            n = level.n_splats_total
            scene.add_text(
                f"● level {i} ({labels[i]}): {n} splats  ÷{factors[i]}",
                position=(0.02, 0.80 + 0.05 * i),
                font_size=0.022,
                font="mono",
                color=css,
                stroke_color="black",
                stroke_width=0.0018,
            )

        # House-style explainer card at the conventional top-left inset; the
        # bottom-left colour legend sits well clear of it.
        add_explainer(
            scene,
            title="Substitutive Gaussian LOD",
            body=(
                "Each coarser level <strong>replaces</strong> the finer one with "
                "fewer synthesized splats (built by "
                "<code>make_substitutive_lod</code>). The viewer auto-picks a "
                "level by projected pixel size; debug colours mark which level "
                "is on screen."
            ),
            observe=[
                "Zoom out and splats turn red (coarsest, fewest splats).",
                "Zoom in and they turn green (finest, full splat count).",
                "The amber middle level appears at intermediate distances.",
                "Splat count drops by ~4x per coarser level.",
            ],
            observe_label="Verify",
        )

    aprint(f"Done. View with: luxar serve {output_path} --viewer")
    aprint("Zoom in/out — splat color changes as the substitutive level switches.")


if __name__ == "__main__":
    main()
