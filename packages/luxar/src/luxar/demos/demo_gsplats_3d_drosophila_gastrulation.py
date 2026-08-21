#!/usr/bin/env python
"""GSplats Demo: Drosophila Gastrulation (SiMView Light-Sheet).

A single 3D light-sheet stack of a *Drosophila melanogaster* embryo caught at
**gastrulation** — the ~20-minute window when the blastoderm's uniform shell of
nuclei buckles into the first body-plan structures. Two of them are visible in
this stack: the **cephalic furrow** (the transverse groove near the anterior
third) and the **posterior midgut invagination** (the pit at the posterior pole).

The embryo carries a ubiquitous histone label (``His2Av::mRFP1``), so every
nucleus is a bright blob — roughly 6,000 of them at this stage. That is exactly
the structure Gaussian splatting represents well: a dense field of compact,
locally-ellipsoidal emitters. The fit below spends ~200,000 splats on it.

DATA SOURCE & CITATIONS:
    Royer & Keller labs. Imaged on a SiMView multi-view light-sheet microscope
    under the AutoPilot adaptive framework:

      Royer, L.A., Lemon, W.C., Chhetri, R.K., Wan, Y., Coleman, M., Myers, E.W.
      & Keller, P.J. "Adaptive light-sheet microscopy for long-term,
      high-resolution imaging in living organisms."
      Nat. Biotechnol. 34, 1267-1278 (2016). doi:10.1038/nbt.3708

    Specimen: ``w; His2Av::mRFP1; +`` (Bloomington stock #23560).

VOXEL CALIBRATION (why the Z scale is what it is):
    The raw stack is anisotropic: 108 z-slices x 1352 x 532 px. The lateral
    pixel size follows from the instrument — Nikon 16x/0.8 detection objectives
    on Hamamatsu Orca Flash 4.0 cameras (6.5 um pitch), i.e. 6.5/16 =
    **0.40625 um/px** (Royer et al. 2016, Online Methods).

    The axial step is not recorded in the array metadata, so it was measured
    from the embryo's own geometry: a Drosophila embryo is a prolate ellipsoid,
    so its mid-length cross-section must be circular. The section spans 91
    z-slices and 433 lateral px, giving a ratio of 4.76 and therefore a z-step
    of 4.76 x 0.40625 = **1.93 um** (the standard ~2 um SiMView sampling, and
    consistent with the paper's 1.75 um detection depth of focus).

    Cross-checks, both independent of that ratio: at this calibration the embryo
    measures 521 um long and 190 um wide, against a textbook Drosophila embryo of
    ~500 x 180 um; and after scaling, the two cross-sectional axes agree to 2.5%
    (195 um axial vs 190 um lateral) — i.e. the section really is circular.

PIPELINE (how the bundled gsplats were produced — provenance, NOT re-run here):
    1. Extract timepoint 150 of the 500-timepoint recording.
    2. Calibrate K* by Noise2Self blind-spot cross-validation (``gsplat cal``):
       the curve is ``signal_limited`` (K* = 512,000) with diminishing returns
       flagged at 256,000 — the operating point used here.
    3. ``gsplat fit --tiling none --seeds 256000`` -> 200,155 splats.
       (The recorded run pinned ``--tiling none``, which at the time was
       required: ``--seeds`` was then applied *per tile*, so auto-tiling this
       stack into 21 tiles multiplied the calibrated budget by 21. Since #1556
       an integer ``--seeds`` is a whole-volume budget that a tiled fit divides
       across its tiles, so the pin is no longer needed to keep the budget
       honest — it is kept here only because it is what actually produced the
       numbers below.)
    4. Quality vs the original volume: **39.89 dB PSNR, 0.911 SSIM**.
    5. ``gsplat transform --scale 1.93,0.40625,0.40625`` -> physical microns.
    6. ``gsplat lod --recipe stream --target-ms 200`` -> progressive ladder.

USAGE:
    python demo_gsplats_3d_drosophila_gastrulation.py [--no-serve] [--serve-only]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_3d_drosophila_gastrulation.luxar.zarr
    - Opens in the browser; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_3d_drosophila_gastrulation",
    "title": "3D Drosophila Gastrulation",
    "description": (
        "A Drosophila embryo at gastrulation (cephalic furrow + posterior midgut "
        "invagination) as 200k Gaussian splats, from SiMView light-sheet."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 3,
        "compute": "light",
        "gpu": "none",
        # Pending the Zenodo upload (R17 step 2) the data resolves from the
        # local cache only, so this is a manual-file demo today. It becomes a
        # plain download (local_data None) once the record URL is populated.
        "local_data": "manual-file",
    },
    "caches": ["gsplats_3d_drosophila_gastrulation"],
    "outputs": ["gsplats_3d_drosophila_gastrulation"],
    "citation": {
        "short": "Royer et al. 2016",
        "doi": "10.1038/nbt.3708",
        "license": "CC BY 4.0",
    },
}

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import ensure_dataset, launch_viewer, parse_demo_flags
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_3d_drosophila_gastrulation"

SCENE_NAME = "gsplats_3d_drosophila_gastrulation.luxar.zarr"

# Physical voxel size (Z, Y, X) in microns — already baked into the centers by
# the fit pipeline (step 5 above). Kept here because the scene's dimension
# ranges are read off the data, but the UNITS are ours to declare.
VOXEL_UM = (1.93, 0.40625, 0.40625)

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]


# =============================================================================
# Data loading
# =============================================================================
def resolve_data() -> Path:
    """Resolve the fitted gsplats: cache -> in-repo copy -> Zenodo.

    ``ensure_dataset`` verifies the manifest sha256 at every step, so a partial
    or corrupted copy is never handed back.
    """
    with asection("Resolving Drosophila gsplats"):
        paths = ensure_dataset(DATASET)
        aprint(f"Data: {paths[0]}")
        return paths[0]


# =============================================================================
# Scene construction
# =============================================================================
def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 3D scene from the pre-fitted, physically-scaled gsplats."""
    with asection("Creating Drosophila gastrulation scene"):
        node, _ = load_gsplat_node(str(data_path))
        bmin, bmax = center_bounds(node)
        aprint(f"Scene bounds (um): min={np.round(bmin, 1)} max={np.round(bmax, 1)}")
        aprint(
            f"Embryo extent: {bmax[1] - bmin[1]:.0f} um long, "
            f"{bmax[2] - bmin[2]:.0f} um wide"
        )

        # Center columns are (Z, Y, X) — the fit's array order — so the
        # Dimensions list must follow that exact order. All three are spatial
        # and displayed; the stack is a single timepoint, so there is no
        # hidden axis.
        dims = Dimensions(
            [
                Dimension(
                    "Z", unit="µm", display=True, range=(float(bmin[0]), float(bmax[0]))
                ),
                Dimension(
                    "Y", unit="µm", display=True, range=(float(bmin[1]), float(bmax[1]))
                ),
                Dimension(
                    "X", unit="µm", display=True, range=(float(bmin[2]), float(bmax[2]))
                ),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES"),
                citation=DEMO_META["citation"],
            )
            scene.attrs["title"] = "GSplats: Drosophila Gastrulation (SiMView)"
            scene.attrs["description"] = (
                "Drosophila melanogaster embryo (His2Av::mRFP1) at gastrulation, "
                "imaged on a SiMView adaptive light-sheet microscope and fitted as "
                "200k Gaussian splats. The transverse groove is the cephalic furrow; "
                "the posterior pit is the midgut invagination. Voxels are physical "
                "microns (Z 1.93, XY 0.40625). Press L for the Layers panel."
            )

            with asection("Adding gsplats"):
                scene.add_gsplats_from_file(
                    name="drosophila_nuclei",
                    path=str(data_path),
                    # `normal` (alpha-over), not an accumulating mode. This is a
                    # dense shell of nuclei about a bright yolk: every summing
                    # mode integrates the far side through the near side and the
                    # embryo reads as one flat silhouette. Measured on this
                    # dataset — `volumetric` needs absorption >= 12 before the
                    # far side stops bleeding through, by which point the whole
                    # object is nearly black. Alpha-over keeps the surface nuclei
                    # crisp. (Same conclusion the Tribolium demo reached.)
                    blending_mode="normal",
                    colormap="magma",
                    # `intensity` is a WINDOW whose top clips: measured here,
                    # anything above ~1.0 drives the bulk of the splats past the
                    # LUT's top and the embryo goes uniformly white (lowering
                    # `opacity` to compensate does NOT undo it — it is the window
                    # that clips, not the alpha). So leave the window alone and
                    # shape the midtones with `gamma`, which is a curve: 2.2 lifts
                    # the nuclei out of magma's dark-violet foot into its magenta
                    # band without touching the highlights. Below ~1.6 the embryo
                    # is too dark to read; above ~3 it washes out.
                    intensity=1.0,
                    gamma=2.2,
                    layer=True,
                )

            scene.add_text(
                "Drosophila • gastrulation",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "SiMView light-sheet • His2Av::mRFP1",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve the data, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: Drosophila Gastrulation (SiMView Light-Sheet)")
    aprint("=" * 70)
    aprint("200k splats • cephalic furrow + posterior midgut invagination")
    aprint("")

    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: launching viewer…")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    data_path = resolve_data()
    scene_path = create_luxar_scene(data_path, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer…")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
