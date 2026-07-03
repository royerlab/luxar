#!/usr/bin/env python3
"""Energy-Breakpoints Example — perceptual LOD ordering on Points and Lines.

Demonstrates the LOD refinements landed in PR #321:

1. **``salience_kind='energy'``** — for ``method='salience'``, sort
   elements by perceptual energy instead of raw size:

   - Points: ``luminance × radius^3``
   - Lines: ``mean_luminance × Σ(seg_length × width^2)``

   Luminance is Rec.709 over colors (falls back to scalars, then to 1).

2. **``counts='energy:0.5,0.9,0.99,1.0'``** — cumulative-energy
   breakpoints. Each fraction names the cumulative perceptual energy
   the LOD level captures: L0 holds the elements that together carry
   ≥50% of total energy, L1 → 90%, L2 → 99%, L3 → the long tail.

3. **``coverage_fraction=``** — a hand-built ``add_lod_group`` where each
   child sets its own viewport-relative switch threshold (0.0 coarsest →
   1.0 finest); the finest child shows when the object fills the screen.

4. **Compiler-side ``display_type`` back-fill** — explicit-builder
   lod_groups without an authored ``display_type`` get one filled in
   at finalize time from the finest child. Demonstrated implicitly:
   the ``custom_lod`` layer below never sets it, yet the layer panel
   correctly shows ``points`` as the type.

Layout: a single "constellation" of points where a handful are bright
big stars and the rest are dim dust. The energy: breakpoints surface
the stars in L0 and the dust progressively at coarser→finer levels.

Educational value:
- Watch the bright big elements paint FIRST (L0 by energy = highest
  luminance × volume). The dim dust fills in later, by visible mass
  rather than count.
- Compare against a sibling layer using the default size-only salience:
  the largest elements paint first regardless of brightness.
- Tweak the per-child ``coverage_fraction=`` values in code and re-run to
  see how the LOD ladder's switching thresholds change.

PR δ (#321) — LOD refinements + energy breakpoints.
"""

import numpy as np
from _overlay_style import add_explainer
from arbol import aprint

from luxar import Dimensions, LuxarZarrCompiler
from luxar.utils.paths import get_examples_output_dir


def make_constellation(
    n_dim: int = 8000,
    n_bright: int = 30,
    seed: int = 0,
):
    """A "constellation": many dim small points + a few bright big ones.

    Returns ``(positions, colors, radii)``. The bright + big elements
    together carry ~80% of perceptual energy under
    ``luminance × radius^3``, so energy: breakpoints cut neatly.
    """
    rng = np.random.default_rng(seed)
    # Dim dust spread across a 20-unit sphere.
    dust_pos = rng.normal(0, 8.0, (n_dim, 3)).astype(np.float32)
    dust_colors = np.full((n_dim, 3), 0.10, dtype=np.float32)  # dim grey
    dust_radii = np.full(n_dim, 0.05, dtype=np.float32)  # tiny

    # A few bright big stars — well-spaced to be picked out individually.
    star_pos = rng.uniform(-10, 10, (n_bright, 3)).astype(np.float32)
    # Saturated colors: red, green, blue, yellow alternating.
    palette = np.array(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.3],
            [0.3, 0.5, 1.0],
            [1.0, 0.9, 0.2],
        ],
        dtype=np.float32,
    )
    star_colors = palette[np.arange(n_bright) % len(palette)]
    star_radii = np.full(n_bright, 0.6, dtype=np.float32)  # ~12× dust

    positions = np.concatenate([dust_pos, star_pos], axis=0)
    colors = np.concatenate([dust_colors, star_colors], axis=0)
    radii = np.concatenate([dust_radii, star_radii], axis=0)
    return positions, colors, radii


def main() -> None:
    output_path = get_examples_output_dir() / "energy_breakpoints_example.luxar.zarr"

    with LuxarZarrCompiler(str(output_path)) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        positions, colors, radii = make_constellation(n_dim=8000, n_bright=30)

        # === 1. Energy-ordered Points ===
        # salience_kind='energy' sorts by luminance × radius^3, so the
        # bright big stars rank first. Energy: breakpoints partition
        # cumulative energy → L0 captures ≥ 50% (the stars), L1 → 90%,
        # L2 → 99%, L3 → the long tail.
        scene.add_points(
            "energy_ordered",
            positions,
            colors=colors,
            radii=radii,
            layer=True,
            additive_lod=dict(
                method="salience",
                salience_kind="energy",
                counts="energy:0.5,0.9,0.99,1.0",
            ),
        )

        # === 2. Size-ordered Points (baseline for comparison) ===
        # Same input, but salience_kind defaults to 'size' (legacy
        # behavior — sort by radius alone, ignoring color/luminance).
        # The stars still come first (they're also the largest), but
        # the cumulative cuts use equal-count instead of energy.
        size_pos = positions.copy()
        size_pos[:, 0] += 30.0  # translate so the two layers don't overlap
        scene.add_points(
            "size_ordered",
            size_pos,
            colors=colors,
            radii=radii,
            layer=True,
            additive_lod=dict(method="salience"),  # salience_kind='size'
        )

        # === 3. Hand-built lod_group with explicit coverage_fraction thresholds ===
        # Each child sets its own viewport-relative switch threshold: the coarse
        # subset is the always-eligible floor (0.0) and the fine level only takes
        # over as the object approaches filling the screen (1.0). Also exercises
        # the display_type back-fill: we never set `display_type=`, but the layers
        # panel still reads "points" because the compiler fills it from the finest
        # child.
        custom = scene.add_lod_group("custom_lod", layer=True)
        # Coarse subset = every 16th point.
        custom.add_points(
            "lod_coarse",
            positions[::16],
            colors=colors[::16],
            radii=radii[::16] * 1.5,
            coverage_fraction=0.0,
        )
        # Fine = everything.
        custom.add_points(
            "lod_fine",
            positions,
            colors=colors,
            radii=radii,
            coverage_fraction=1.0,
        )

        add_explainer(
            scene,
            title="Energy-Ordered LOD Breakpoints",
            body=(
                "<code>salience_kind='energy'</code> orders elements by "
                "luminance x radius^3, and <code>counts='energy:...'</code> cuts "
                "LOD levels at cumulative-energy fractions, so bright, large "
                "splats land in the coarsest level first."
            ),
            observe=[
                "On 'energy_ordered' the bright coloured stars paint first, then dust fills in.",
                "'size_ordered' paints largest-first regardless of colour.",
                "'custom_lod' reports type 'points' (display_type back-fill).",
                "custom_lod's fine level (coverage_fraction=1.0) shows only near fills-screen.",
            ],
            observe_label="Look for",
        )

        aprint(
            "Created three points layers:\n"
            "  - energy_ordered: salience_kind='energy' + energy: breakpoints\n"
            "  - size_ordered: default salience (size only)\n"
            "  - custom_lod: 2-level kind=lod with explicit coverage_fraction + "
            "no authored display_type"
        )

    aprint(f"\nScene saved to: {output_path}")
    aprint(f"To view: luxar serve --viewer {output_path}")
    aprint(
        "  → All three layers show up in the panel.\n"
        "  → On 'energy_ordered' you should see the COLORED STARS paint\n"
        "     first (they carry the most perceptual energy), then dust\n"
        "     fills in around them across the next refinement frames.\n"
        "  → 'size_ordered' paints big-first regardless of color.\n"
        "  → 'custom_lod' should report type 'points' on its badge —\n"
        "     proof of the compiler-side display_type back-fill.\n"
        "  → 'custom_lod's fine level (coverage_fraction=1.0) only shows\n"
        "     near fills-screen; zoom out to see the coarse subset take over."
    )


if __name__ == "__main__":
    main()
