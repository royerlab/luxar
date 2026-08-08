#!/usr/bin/env python3
"""GSplats Demo: Adaptive Level of Detail on a real Tribolium embryo (Light-Sheet)

Takes the ~256K-splat fit of the *Tribolium castaneum* embryo (the same
precomputed dataset as ``demo_gsplats_3d_tribolium_embryo.py``) and builds an
**adaptive Level of Detail (LOD)** pyramid on top of it, then ships it to the
viewer with per-level debug colors so the detail switching is visible.

Level of Detail (LOD) means storing the embryo at several resolutions and
letting the viewer show the simplest one that still looks right at the current
zoom: a detailed version up close, a coarser version when you zoom out. This
demo uses *substitutive* LOD — each coarser level fully *replaces* the finer one
with fewer, larger splats (genuine geometry compression, not just a load order).

================================================================================
WHAT THIS DEMONSTRATES — ADAPTIVE LEVEL OF DETAIL AT SCALE
================================================================================

The tiny ``examples/gsplats_lod_example.py`` shows this on a 12-blob synthetic
volume.  This demo does the same thing on a *real* quarter-million-splat
microscopy dataset, which is where adaptive detail actually earns its keep:

- **Substitutive levels**: each coarser level *replaces* the finer one with a
  smaller set of synthesized representative splats (built by
  ``make_substitutive_lod``).  This is genuine geometry/memory compression, not
  just a streaming order.  The viewer auto-picks a level by viewport-relative
  ``coverage_fraction`` (``sqrt(N_i/N_finest)``, derived from per-level splat
  counts) — the finest level shows when the embryo fills the screen and coarser
  levels step in as it shrinks: zoom out → coarse, zoom in → fine.
- **Debug colors** make the level-switching obvious: each level is painted a
  distinct color on a green → amber → red ramp (finest → coarsest).  As you
  zoom, the embryo changes color when the active level changes.
- **auto** method (the default): per level it uses kmeans_lloyd (Morton
  warm start + cost-increment Lloyd) for the large early levels and greedy
  (Runnalls) for the small coarse levels where greedy is both fastest and
  highest quality.

Pipeline:
1. **Load** precomputed ~256K-splat Tribolium fit (Git LFS / local cache)
2. **Center + scale** intensity (so every synthesized level is consistent)
3. **Build** a substitutive ladder with ``make_substitutive_lod``.
   The number of ÷4 levels needed scales as log_4(N): the precomputed
   cached fit is ~256K splats, so auto-leveling builds 7 levels
   (≈256K / 64K / 16K / 4K / 1K / 250 / 62 splats, finest→coarsest)
   down to a handful of blobs. The level count is auto-derived from the
   actual base splat count (override with --levels=N).
4. **Colorize** each level (green → amber → red) for visible level-switching
5. **Export** as a kind="lod" group via ``add_gsplats_from_data(lod_group=True)``
6. **Visualize** — zoom in/out, watch the color (and splat count) change

DATA SOURCE & CITATIONS:
========================
Source:  Cell Tracking Challenge / Zenodo record 5270323 (GIANI paper)
Imaging: Zeiss LightSheet Z.1, Tribolium castaneum, 0.381 um isotropic
Cite:    Yin et al. (2022). GIANI. J. Cell Sci. 135(5), jcs259022.
         Maska et al. (2023). Cell Tracking Challenge. Nat. Methods 20, 1010-1020.

USAGE:
======
    python demo_gsplats_lod_tribolium.py [--recompute] [--no-serve] [--serve-only]
                                         [--levels=N] [--factor=K] [--method=NAME]

Options:
    --recompute:   Force re-fitting the base splats from scratch (download + GPU)
    --no-serve:    Generate scene without launching viewer
    --serve-only:  Just serve a previously generated scene (skips LOD rebuild)
    --levels=N:    Number of coarser levels (default: auto from base count)
    --factor=K:    Per-level compression factor (default: 4)
    --method=NAME: auto (default) | kmeans_lloyd | greedy_lloyd | kmeans | greedy

By default the base splats are loaded from package data (Git LFS).  Use
--recompute to re-fit from scratch (requires network + GPU).  The substitutive
ladder is always built fresh from the base splats (that is the point of the
demo); use --serve-only to reopen the last generated scene without rebuilding.

Output:
    - Scene saved to:  datasets/demos/gsplats_lod_tribolium.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_lod_tribolium",
    "title": "Adaptive Level of Detail on a real Tribolium embryo (Light-Sheet)",
    "description": "A ~256K-splat Tribolium embryo with an adaptive substitutive LOD ladder for zoom detail.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 3,
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_tribolium"],
    "outputs": ["gsplats_lod_tribolium"],
}

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats import make_substitutive_lod
from luxar.gsplats.gsplat_data import (
    AdditiveSubLOD,
    GSplatData,
    SubstitutiveLevel,
)
from luxar.utils.demos import (
    detect_device,
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Substitutive LOD parameters.
#
# Each coarser level keeps ~N/K splats, so the number of useful levels scales
# as log_K(N): the precomputed cached fit is ~256K splats, which auto-derives
# to 7 levels (≈256K / 64K / 16K / 4K / 1K / 250 / 62, finest→coarsest) down to
# a handful of blobs. LEVELS=None auto-derives the count from the actual base
# splat count so the coarsest level lands near MIN_COARSEST splats; override
# with --levels=N.
COMPRESSION_FACTOR = 4  # K: each coarser level keeps ~N/K splats
LEVELS: int | None = None  # None → auto from base count; int → forced
MIN_COARSEST = 24  # auto-levels target: stop before the coarsest level dips below this
METHOD = "auto"  # per-level: kmeans_lloyd for large levels, greedy for small


def auto_levels(n_base: int, k: int = COMPRESSION_FACTOR) -> int:
    """Pick #coarser levels so the coarsest stays >= MIN_COARSEST splats."""
    levels = 0
    while n_base // (k ** (levels + 1)) >= MIN_COARSEST:
        levels += 1
    return max(1, levels)


# Color ramp anchors (finest → coarsest), RGB floats in [0, 1].
# Coarse → warm/red so a coarse level (few big splats, seen zoomed out)
# reads as "alarm"; fine → green. Matches gsplats_lod_example.py.
_RAMP_ANCHORS: list[tuple[float, float, float]] = [
    (0.25, 1.00, 0.45),  # finest   → green
    (1.00, 0.78, 0.20),  # middle   → amber
    (1.00, 0.30, 0.32),  # coarsest → red
]

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

for _arg in sys.argv:
    if _arg.startswith("--levels="):
        LEVELS = int(_arg.split("=")[1])
    elif _arg.startswith("--factor="):
        COMPRESSION_FACTOR = int(_arg.split("=")[1])
    elif _arg.startswith("--method="):
        METHOD = _arg.split("=")[1]

# Arbol logging depth
Arbol.max_depth = 5


# =============================================================================
# Level coloring
# =============================================================================


def level_color(t: float) -> tuple[float, float, float]:
    """Piecewise-linear green → amber → red ramp. ``t`` in [0, 1]."""
    if t <= 0.0:
        return _RAMP_ANCHORS[0]
    if t >= 1.0:
        return _RAMP_ANCHORS[-1]
    # Two segments: [0, 0.5] green→amber, [0.5, 1] amber→red.
    pos = t * (len(_RAMP_ANCHORS) - 1)
    i = int(pos)
    frac = pos - i
    a = _RAMP_ANCHORS[i]
    b = _RAMP_ANCHORS[i + 1]
    return tuple(a[k] + (b[k] - a[k]) * frac for k in range(3))  # type: ignore[return-value]


def level_colors(n_levels: int) -> list[tuple[float, float, float]]:
    """Distinct debug color per substitutive level (finest=index 0)."""
    if n_levels == 1:
        return [_RAMP_ANCHORS[0]]
    return [level_color(i / (n_levels - 1)) for i in range(n_levels)]


def colorize_levels(ladder: GSplatData) -> GSplatData:
    """Return a copy of ``ladder`` with a solid debug color per level.

    Each substitutive level's lone additive sub-LOD is rebuilt with a flat
    ``(N, 3)`` color array. ``AdditiveSubLOD`` / ``SubstitutiveLevel`` are
    frozen, so we construct fresh instances rather than mutate in place.
    """
    palette = level_colors(len(ladder.substitutive_levels))
    colored_levels: list[SubstitutiveLevel] = []
    for level in ladder.substitutive_levels:
        sub = level.additive_sublods[0]
        rgb = palette[level.level_index]
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


# =============================================================================
# LOD construction
# =============================================================================


def build_lod_ladder(base: GSplatData) -> GSplatData:
    """Center, scale, and build the substitutive LOD ladder from base splats."""
    # Center at intensity-weighted centroid + dim the amplitudes, BEFORE building
    # the ladder so every synthesized level inherits consistent coords/intensity.
    base = base.translate(-base.centers.T @ base.amplitudes / base.amplitudes.sum())
    base = base.scale_intensity(0.03)

    n_base = len(base.amplitudes)
    n_levels = LEVELS if LEVELS is not None else auto_levels(n_base)
    device = detect_device()
    with asection(
        f"Building substitutive LOD (method={METHOD}, "
        f"K={COMPRESSION_FACTOR}, levels={n_levels})"
    ):
        aprint(f"Base splats: {n_base:,}")
        if LEVELS is None:
            aprint(
                f"Auto levels: {n_levels} (coarsest ≈ {n_base // COMPRESSION_FACTOR**n_levels} splats)"
            )
        aprint(f"Device: {device}")
        ladder = make_substitutive_lod(
            base,
            compression_factor=COMPRESSION_FACTOR,
            levels=n_levels,
            method=METHOD,  # type: ignore[arg-type]
            device=device,
            seed=0,
            verbose=True,
        )
        counts = [lvl.n_splats_total for lvl in ladder.substitutive_levels]
        aprint(f"Substitutive levels (finest→coarsest): {counts} splats")
    return colorize_levels(ladder)


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(colored: GSplatData, output_path: Path) -> Path:
    """Create the LOD scene from the colored substitutive ladder."""
    with asection("Creating 3D Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        dims = Dimensions(
            [
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                # ACES, set explicitly + matched intensity, consistent with the
                # other gsplat demos. NOTE: the per-level debug colours
                # (green -> amber -> red) must stay tellable apart; ACES shifts
                # hues, so check them if this demo's level cues get muddy.
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )

            scene.attrs["title"] = (
                "GSplats: Adaptive Level of Detail — Tribolium Embryo"
            )
            scene.attrs["description"] = """
Adaptive Level of Detail — Tribolium castaneum Embryo (Light-Sheet)
===================================================================

A ~256K-splat fit of a beetle embryo, stored at several resolutions.

Level of Detail (LOD) means the viewer shows the simplest version of the
embryo that still looks right at the current zoom: a detailed version up
close, a coarser one when you zoom out. Each coarser level REPLACES the
finer one with fewer, larger splats, so it never draws more detail than
the screen can show. Debug colors (green→amber→red, finest→coarsest) mark
which level is currently on screen.

Data: Cell Tracking Challenge / Zenodo 5270323, Zeiss LightSheet Z.1.
Cite: Yin et al. (2022) J. Cell Sci. 135(5); Maska et al. (2023) Nat. Methods 20.

Navigation:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
  - Zoom out → coarse (red, few splats); zoom in → fine (green, full count)
            """

            # lod_group=True → write the stored substitutive levels as-is into a
            # kind="lod" group (one colored gsplats child per level).
            scene.add_gsplats_from_data(
                "tribolium_lod",
                colored,
                lod_group=True,
                opacity=1.0,
                absorption=1.0,
                blending_mode="volumetric",
                # `layer` is a compositing attr, so it rides on the kind=lod
                # wrapper: one panel row for the ladder, not one per level.
                layer=True,
            )

            # ── Overlays ────────────────────────────────────────────────────
            # Canonical title (top-left) + data-source caption (bottom-right),
            # matching the other gsplat demos.
            scene.add_text(
                "Tribolium Embryo — Adaptive Level of Detail",
                position=(0.02, 0.02),
                font_size=0.048,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Light-sheet microscopy • adaptive level of detail",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

            # Explanatory panel, placed below the title so the two don't overlap.
            scene.add_text(
                "Level of Detail (LOD): the embryo is stored at several\n"
                "resolutions. The viewer shows the simplest one that still looks\n"
                "right at your current zoom — a detailed version up close, a\n"
                "coarser one (fewer, bigger blobs) when you zoom out.\n"
                "Zoom out → coarse (red); zoom in → fine (green).",
                position=(0.02, 0.10),
                font_size=0.020,
                font="mono",
                color="white",
                width=0.46,
                line_height=1.45,
                background="rgba(0,0,0,0.55)",
                padding=0.012,
            )

            # Per-level color legend (finest → coarsest), each line tinted to
            # match the splats of that level. Layout grows upward from a fixed
            # bottom so it stays on-screen for any number of levels.
            palette = level_colors(len(colored.substitutive_levels))
            n_levels = len(colored.substitutive_levels)
            spacing = 0.034
            start_y = 0.94 - spacing * (n_levels - 1)
            for level in colored.substitutive_levels:
                i = level.level_index
                r, g, b = (int(round(c * 255)) for c in palette[i])
                css = f"rgb({r},{g},{b})"
                n = level.n_splats_total
                factor = COMPRESSION_FACTOR**i
                if i == 0:
                    label = "finest"
                elif i == n_levels - 1:
                    label = "coarsest"
                else:
                    label = f"level {i}"
                scene.add_text(
                    f"● {label}: {n:,} splats  ÷{factor}",
                    position=(0.02, start_y + spacing * i),
                    font_size=0.020,
                    font="mono",
                    color=css,
                    stroke_color="black",
                    stroke_width=0.0018,
                )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: Adaptive Level of Detail — Tribolium castaneum Embryo")
    aprint("=" * 70)
    aprint("~256K real microscopy splats -> adaptive level-of-detail pyramid")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_lod_tribolium.luxar.zarr"

    # Serve-only mode: reopen the last generated scene without rebuilding.
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Load precomputed base splats (or re-fit on --recompute).
    precomputed = load_precomputed_gsplats(
        "gsplats_tribolium",
        ["tribolium.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )

    if precomputed is not None:
        base = precomputed[0]
    else:
        # --recompute path: re-fit the base splats from the raw volume by
        # delegating to the base Tribolium demo's fitting pipeline.
        warn_if_no_cuda_gpu()
        from luxar.demos.demo_gsplats_3d_tribolium_embryo import (
            fit_tribolium,
            load_tribolium_volume,
        )

        base = fit_tribolium(load_tribolium_volume())

    # Build the substitutive ladder and colorize it.
    colored = build_lod_ladder(base)

    # Report
    with asection("Summary"):
        counts = [lvl.n_splats_total for lvl in colored.substitutive_levels]
        aprint(f"Levels:  {len(counts)} ({COMPRESSION_FACTOR}x compression each)")
        aprint(f"Splats:  {counts} (finest→coarsest)")

    # Create scene + launch viewer.
    scene_path = create_luxar_scene(colored, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        aprint("Zoom in/out — splat color changes as the detail level switches.")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
