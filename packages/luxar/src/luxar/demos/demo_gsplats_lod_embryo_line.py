#!/usr/bin/env python3
"""GSplats Demo: Near-Unlimited Scaling with Adaptive Level of Detail

Lays out ``--count`` copies (default 100) of one Tribolium embryo along a
straight line. You start near the middle of the line and can fly anywhere.

The trick that keeps this fast is **Level of Detail (LOD)**: every object is
kept at several resolutions at once, and the viewer picks — per object, every
frame — the coarsest version that still looks right at its current on-screen
size. Embryos near the camera render at full detail; the ones receding into the
distance collapse to a handful of big splats. So the detail actually drawn stays
roughly bounded by what the screen can resolve — you could line up thousands of
embryos and the rendering cost would barely grow. (Each embryo is the
"substitutive" LOD ladder built by ``demo_gsplats_lod_tribolium.py``, where each
coarser level *replaces* the finer one with fewer, larger splats.)

================================================================================
WHAT THIS DEMONSTRATES — ADAPTIVE DETAIL MAKES HUGE SCENES TRACTABLE
================================================================================

``demo_gsplats_lod_tribolium.py`` shows adaptive Level of Detail on a *single*
embryo. This demo shows why that matters at scale:

- **Per-object detail selection**: the viewer evaluates every embryo
  independently each frame by its viewport-relative ``coverage_fraction``
  (``sqrt(N_i/N_finest)``; ``lod-group-registry.ts``) — effectively how much of
  the screen it covers. A near embryo gets the finest level (green, full
  detail); a far one gets the coarsest (red, a few big splats). You never pay to
  draw detail you cannot see.
- **"Never more than the screen can show"**: with the camera in the middle, the
  bulk of the line is far away and therefore coarse. The number of splats
  actually drawn (visible in the data-loading monitor) stays modest even with
  100+ embryos — the long tail of distant embryos costs almost nothing.
- **Shared geometry on disk**: all copies are *byte-identical*, so the Luxar
  encoder writes the geometry **once** and stores cheap pointers for the rest
  (``array_ref``, ``encoding/registry.py``). 100 embryos cost about 1 embryo of
  storage. Each embryo's individual look (orientation, jitter) lives entirely in
  **scene-graph transforms**, never in the splat coordinates — which is what
  keeps the arrays identical and the pointers valid.
- **Color by detail level** (green → amber → red, finest → coarsest, inherited
  from the single-embryo demo) makes the switching unmistakable: as the line
  recedes, the embryos visibly shift from green to red.

Pipeline:
1. **Load** the precomputed Tribolium fit and build ONE colored substitutive
   ladder (delegates to ``demo_gsplats_lod_tribolium.build_lod_ladder``).
2. **Lay out** ``--count`` embryos along X. Each embryo is the *same* ladder
   placed inside its own group with a transform = random rotation (about the
   embryo's own center) ∘ translation to its slot on the line, plus small jitter
   so the line reads as a population rather than photocopies.
3. **Aim the camera** in front of and near the middle of the line, looking down
   it so it recedes into the distance.
4. **Export** one scene; the encoder deduplicates the shared arrays.
5. **Visualize** — fly down the line, watch distant embryos go coarse/red while
   near ones stay fine/green.

DATA SOURCE & CITATIONS:
========================
Source:  Cell Tracking Challenge / Zenodo record 5270323 (GIANI paper)
Imaging: Zeiss LightSheet Z.1, Tribolium castaneum, 0.381 um isotropic
Cite:    Yin et al. (2022). GIANI. J. Cell Sci. 135(5), jcs259022.
         Maska et al. (2023). Cell Tracking Challenge. Nat. Methods 20, 1010-1020.

USAGE:
======
    python demo_gsplats_lod_embryo_line.py [--count=N] [--recompute]
                                           [--no-serve] [--serve-only]
                                           [--levels=N] [--factor=K] [--method=NAME]

Options:
    --count=N:     Number of embryos along the line (default: 100)
    --recompute:   Force re-fitting the base splats from scratch (download + GPU)
    --no-serve:    Generate scene without launching viewer
    --serve-only:  Just serve a previously generated scene (skips rebuild)
    --levels=N:    Number of coarser LOD levels (default: auto from base count)
    --factor=K:    Per-level compression factor (default: 4)
    --method=NAME: auto (default) | kmeans_lloyd | greedy_lloyd | kmeans | greedy

Output:
    - Scene saved to:  datasets/demos/gsplats_lod_embryo_line.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_lod_embryo_line",
    "title": "Embryo LOD (splats + lines)",
    "description": "100 copies of a Tribolium embryo splat-fit laid along a line — adaptive level-of-detail at scale.",
    "category": "microscopy",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 50,  # approx
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_tribolium"],
    "outputs": ["gsplats_lod_embryo_line"],
}

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import (
    CameraConfig,
    Dimension,
    Dimensions,
    LuxarZarrCompiler,
    ViewerConfig,
)
from luxar.core import transforms

# Reuse the single-embryo LOD demo's ladder builder + palette (build_lod_ladder,
# level_colors, COMPRESSION_FACTOR) as a normal sibling import rather than
# duplicating the LOD logic.
from luxar.demos import demo_gsplats_lod_tribolium as _LOD
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

COUNT = 100  # number of embryos along the line (override with --count=N)
SPACING_FACTOR = 1.5  # center-to-center spacing as a multiple of embryo diameter
JITTER_FRACTION = 0.18  # lateral/along-line jitter as a fraction of spacing
SEED = 0  # RNG seed so the layout is reproducible

# Parse shared demo flags + this demo's extras.
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

for _arg in sys.argv:
    if _arg.startswith("--count="):
        COUNT = int(_arg.split("=")[1])

Arbol.max_depth = 5


def load_base_splats() -> GSplatData:
    """Load the precomputed Tribolium fit (or re-fit on --recompute)."""
    precomputed = load_precomputed_gsplats(
        "gsplats_tribolium",
        ["tribolium.gsplats.zarr.zip"],
        recompute=RECOMPUTE,
    )
    if precomputed is not None:
        return precomputed[0]

    # --recompute path: re-fit from the raw volume via the base Tribolium demo.
    warn_if_no_cuda_gpu()
    from luxar.demos.demo_gsplats_3d_tribolium_embryo import (
        fit_tribolium,
        load_tribolium_volume,
    )

    return fit_tribolium(load_tribolium_volume())


# =============================================================================
# Layout
# =============================================================================


def embryo_diameter(colored: GSplatData) -> float:
    """Bounding-box diagonal of the finest level (the embryo's full extent)."""
    finest = colored.substitutive_levels[0].additive_sublods[0].centers
    extent = finest.max(axis=0) - finest.min(axis=0)
    return float(np.linalg.norm(extent))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-random 4x4 rotation about the embryo's own center.

    Applied at the SCENE-GRAPH level so the splat arrays stay byte-identical
    across all embryos (which is what lets the encoder deduplicate them).
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) or 1.0
    angle = float(rng.uniform(0.0, 360.0))
    return transforms.rotate(angle, axis)


def embryo_transforms(
    n: int, spacing: float, rng: np.random.Generator
) -> list[np.ndarray]:
    """Per-embryo transform: rotate-about-origin, then translate onto the line.

    ``compose(rot, trans)`` applies ``rot`` first (the embryo is centered at the
    origin) then ``trans`` — i.e. spin in place, then move to the slot.
    """
    jitter = JITTER_FRACTION * spacing
    half = (n - 1) / 2.0
    out: list[np.ndarray] = []
    for i in range(n):
        x = (i - half) * spacing + rng.uniform(-jitter, jitter)
        y = rng.uniform(-jitter, jitter)
        z = rng.uniform(-jitter, jitter)
        out.append(
            transforms.compose(random_rotation(rng), transforms.translate(x, y, z))
        )
    return out


def camera_for_line(n: int, spacing: float, diameter: float) -> CameraConfig:
    """Stand just off the NEAR end of the line and look ALONG its axis.

    The camera sits a hair before the first embryo, shifted sideways by about
    one embryo diameter (and slightly raised), aiming down the +X axis at the
    far end — so the whole row of embryos recedes into the distance rather than
    the front one occluding the rest.
    """
    radius = diameter / 2.0
    total_len = (n - 1) * spacing
    # Stand back off the near end (−X), offset sideways ~2 diameters in +Z and
    # raised ~1 diameter in +Y, and aim at the MIDDLE of the row — so the whole
    # sequence recedes diagonally and stays centred (standing right at the first
    # embryo blows it out and crams the line into a corner).
    position = (-total_len * 0.12, diameter * 1.0, diameter * 2.2)
    target = (total_len * 0.45, 0.0, 0.0)
    return CameraConfig(
        position=position,
        target=target,
        up=(0.0, 1.0, 0.0),
        fov=50.0,
        near=max(0.1, radius * 0.02),
        far=total_len * 1.5 + radius * 10.0,
    )


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    colored: GSplatData, xforms: list[np.ndarray], camera: CameraConfig, out: Path
) -> Path:
    """Build the line-of-embryos scene from one shared colored ladder."""
    n = len(xforms)
    with asection(f"Creating scene: {n} embryos on a line"):
        aprint(f"Output: {out.name}")

        dims = Dimensions(
            [
                Dimension("x", unit="px", display=True),
                Dimension("y", unit="px", display=True),
                Dimension("z", unit="px", display=True),
            ]
        )

        with LuxarZarrCompiler(out, encoding_mode=EncodingMode.PRECISION) as compiler:
            scene = compiler.create_scene(
                dimensions=dims, viewer_config=ViewerConfig(camera=camera)
            )

            scene.attrs["title"] = (
                f"GSplats: Adaptive Level of Detail — {n} Tribolium embryos"
            )
            scene.attrs["description"] = f"""
Near-Unlimited Scaling with Adaptive Level of Detail
====================================================

{n} copies of one Tribolium embryo laid along a straight line. You start near
the middle, looking down the line.

Level of Detail (LOD) is the idea that keeps this fast: each embryo is kept at
several resolutions, and the viewer chooses — for each embryo, every frame —
the simplest version that still looks right at its on-screen size. Near embryos
are drawn in full detail (green); far ones collapse to a few big blobs (red).
So the detail actually drawn stays roughly bounded by what the screen can show,
no matter how long the line — you could add thousands of embryos and the cost
would barely grow.

On disk the {n} embryos are byte-identical, so the encoder stores the geometry
once and writes cheap references for the rest. Each embryo's individual look
(its rotation and position) lives in the scene graph, not in the geometry.

Data: Cell Tracking Challenge / Zenodo 5270323, Zeiss LightSheet Z.1.
Cite: Yin et al. (2022) J. Cell Sci. 135(5); Maska et al. (2023) Nat. Methods 20.

Navigation:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
  - Fly down the line: near = fine/green, far = coarse/red
            """

            # Place the SAME colored ladder under one group per embryo, each
            # carrying its own transform. The arrays are identical across
            # embryos, so the encoder deduplicates them via array references.
            with asection(f"Placing {n} embryos (shared arrays → array_ref dedup)"):
                # One `layer=True` group over the whole line: 100 per-embryo
                # rows would drown the Layers panel, and the embryos are copies
                # of one ladder anyway, so the useful control is "the line".
                # The wrapper restates the descendants' blending mode rather
                # than leaving it unset: the panel falls back to a group's
                # DEFAULT mode (additive) when none is authored, which would
                # mislabel the row and hide the absorption slider that only
                # `volumetric` layers get. Every embryo below is volumetric, so
                # the mode the wrapper owns is the one they already had.
                line = scene.add_group(
                    "embryo_line", layer=True, blending_mode="volumetric"
                )
                for i, xform in enumerate(xforms):
                    group = line.add_group(f"embryo_{i:03d}", transform=xform)
                    group.add_gsplats_from_data(
                        "lod",
                        colored,
                        lod_group=True,
                        opacity=1.0,
                        absorption=1.0,
                        blending_mode="volumetric",
                    )

            # ── Overlays ────────────────────────────────────────────────────
            # Canonical title (top-left) + data-source caption (bottom-right),
            # matching the other gsplat demos.
            scene.add_text(
                "Near-Unlimited Scaling with Adaptive Level of Detail",
                position=(0.02, 0.02),
                font_size=0.040,
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
                f"Level of Detail (LOD): {n} embryos on one line.\n"
                "Each embryo is drawn at the detail its on-screen size needs —\n"
                "near ones stay fine (green), far ones drop to a few big blobs\n"
                "(red). The detail actually drawn never exceeds what you can see,\n"
                "so the line could be far longer at almost no extra cost.",
                position=(0.02, 0.10),
                font_size=0.020,
                font="mono",
                color="white",
                width=0.50,
                line_height=1.45,
                background="rgba(0,0,0,0.55)",
                padding=0.012,
            )

            # Per-level color legend (finest → coarsest), reusing the single
            # embryo demo's palette so the colors match the splats on screen.
            palette = _LOD.level_colors(len(colored.substitutive_levels))
            n_levels = len(colored.substitutive_levels)
            spacing_y = 0.034
            start_y = 0.94 - spacing_y * (n_levels - 1)
            for level in colored.substitutive_levels:
                idx = level.level_index
                r, g, b = (int(round(c * 255)) for c in palette[idx])
                if idx == 0:
                    label = "finest (near)"
                elif idx == n_levels - 1:
                    label = "coarsest (far)"
                else:
                    label = f"level {idx}"
                factor = _LOD.COMPRESSION_FACTOR**idx
                scene.add_text(
                    f"● {label}: {level.n_splats_total:,} splats  ÷{factor}",
                    position=(0.02, start_y + spacing_y * idx),
                    font_size=0.020,
                    font="mono",
                    color=f"rgb({r},{g},{b})",
                    stroke_color="black",
                    stroke_width=0.0018,
                )

        aprint(f"Scene saved: {out}")
        return out


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main demo execution."""
    aprint("=" * 70)
    aprint(
        "GSplats Demo: Near-Unlimited Scaling with Adaptive Level of Detail "
        f"({COUNT} Tribolium embryos)"
    )
    aprint("=" * 70)

    output_path = get_demos_output_dir() / "gsplats_lod_embryo_line.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    # Build ONE colored substitutive ladder (reuses the single-embryo demo).
    base = load_base_splats()
    colored = _LOD.build_lod_ladder(base)

    diameter = embryo_diameter(colored)
    spacing = diameter * SPACING_FACTOR
    rng = np.random.default_rng(SEED)
    xforms = embryo_transforms(COUNT, spacing, rng)
    camera = camera_for_line(COUNT, spacing, diameter)

    with asection("Summary"):
        counts = [lvl.n_splats_total for lvl in colored.substitutive_levels]
        aprint(f"Embryos:        {COUNT} on a line")
        aprint(f"Embryo extent:  {diameter:,.1f} px (diagonal)")
        aprint(f"Line length:    {(COUNT - 1) * spacing:,.1f} px")
        aprint(f"Detail levels:  {len(counts)} → {counts} splats (fine→coarse)")
        aprint("Stored once:    arrays shared across embryos via array_ref dedup")

    scene_path = create_luxar_scene(colored, xforms, camera, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        aprint("Fly down the line — far embryos go coarse/red, near ones fine/green.")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
