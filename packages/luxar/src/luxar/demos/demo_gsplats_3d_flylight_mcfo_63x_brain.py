#!/usr/bin/env python
"""GSplats Demo: Drosophila Whole Brain, MultiColor FlpOut (FlyLight 63x).

A whole female *Drosophila melanogaster* central brain with both optic lobes,
labelled by **MultiColor FlpOut (MCFO)** and imaged at 63x on a confocal
microscope. MCFO is a stochastic three-epitope label: each cell that flips out
expresses a random combination of HA / V5 / FLAG tags, so individual neurons
come up in distinguishable colours against an almost entirely empty volume.

This is the **sparse-and-huge** shape. The source mosaic is 2573 x 2707 x 463
voxels — 3.22 Gvoxel per channel, 12.9 Gvoxel over the four channels — of which
roughly **0.03% carries signal**. Fitted to 418,722 Gaussian splats that is
about **7,700:1** against the source voxels, which is the regime splats are for:
the empty 99.97% costs nothing.

WHY THE COMPOSITE IS FITTED ONCE, NOT PER CHANNEL:
    A neuron's MCFO hue is the RATIO of the three channels at the SAME voxels.
    Fitting each channel separately gives three splat sets whose centres do not
    co-locate, and every axon renders candy-striped. So the fit runs on the
    per-voxel maximum of the three (gain-balanced at p99.99 first, or whichever
    channel is brightest wins nearly every voxel), and each splat then reads its
    colour from the three channels at its own centre.

DATA SOURCE & CITATIONS:
    Janelia FlyLight Gen1 MCFO collection, driver line VT019012, slide code
    20140423_20_D5, female, 63x. Imagery is CC BY 4.0.

    Browse:  https://gen1mcfo.janelia.org
    Bulk:    s3://janelia-flylight-imagery/  (public, anonymous access)
    Docs:    https://github.com/JaneliaSciComp/open-data-flylight

    Please credit the FlyLight Project Team at Janelia Research Campus and cite:

    - Meissner, G. W. et al. "A searchable image resource of Drosophila GAL4
      driver expression patterns with single neuron resolution." eLife 12,
      e80660 (2023). doi:10.7554/eLife.80660
    - Nern, A., Pfeiffer, B. D. & Rubin, G. M. "Optimized tools for multicolor
      stochastic labeling reveal diverse stereotyped cell arrangements in the
      fly visual system." PNAS 112 (22), 2015. doi:10.1073/pnas.1506763112
    - Tirian, L. & Dickson, B. J. "The VT GAL4, LexA, and split-GAL4 driver line
      collections for targeted expression in the Drosophila nervous system."
      bioRxiv 198648 (2017). doi:10.1101/198648

VOXEL SIZE:
    Recorded by the instrument, not inferred: the LSM metadata gives
    ``VoxelSizeX/Y`` = 1.8826889e-07 m and ``VoxelSizeZ`` = 3.8e-07 m, i.e.
    **0.18826889 um** laterally and **0.38 um** axially. The scale applied in
    step 9 rounds the lateral pitch to 0.1883 um (a 0.017% error, ~0.08 um over
    the full 475 um width — far below the ~0.4 um optical resolution).

    The shipped centres are therefore already in micrometres, and the brain
    measures 475 x 501 x 168 um: the right envelope for an adult Drosophila
    central brain plus optic lobes, which is an independent check that the
    calibration and the axis permutation are both right.

PIPELINE (how the bundled gsplats were produced — provenance, NOT re-run here):
    1. Pull the 10 raw ``.lsm.bz2`` tiles for this specimen from the public S3
       bucket (~7.2 GB compressed; 5 tile positions x 2 acquisitions).
    2. Recover tile placement. The raw LSMs carry NO stage coordinates —
       ``Positions``, ``TilePositions`` and ``OriginX/Y/Z`` are all zero — so the
       mosaic geometry exists only in Janelia's stitched H5J. Each tile is
       registered against that H5J, which is used purely as a COORDINATE
       reference; its lossy 8-bit voxels never enter the output.
    3. Blend to a 16-bit mosaic with a separable Hann feather (a plain max keeps
       the brighter tile's shading step at the seam; a plain mean halves
       single-tile-thick coverage).
    4. Gain-balance the three signal channels at p99.99 and take the per-voxel
       maximum as the fit target. The fourth channel (nc82 neuropil reference)
       is EXCLUDED — it is 10-19% occupied, i.e. not sparse, and would dominate.
    5. ``gsplat cal --auto-region --feature-metric edges --k-star-metric gain``
       -> K* ~ 27.9k on a 320^3 content-rich region (a real held-out peak: the
       curve turns over, unlike stitched/denoised data which never peaks).
    6. ``gsplat fit --tiling content --cal … --flat --floor p99`` -> 418,722
       splats over 66 content-balanced boxes. ``--floor p99`` rather than
       ``auto``: auto removes the pedestal but leaves neuropil autofluorescence,
       which fills the brain silhouette and saturates in every blending mode.
    7. Colour each splat by sampling the three channels at its own centre.
    8. ``gsplat lod --recipe stream --target-ms 200`` -> progressive ladder.
    9. ``gsplat transform --rotate-y 90 --scale 0.1883,0.1883,0.38 --center``
       -> face-on default view, physical micrometres.

KNOWN LIMITATION — RESIDUAL GHOSTING:
    Thin neurites show some doubling where tiles overlap. This is NOT fixable by
    better translation, and the demo ships with it knowingly.

    Each of the 5 tile positions is a separate acquisition and the sample
    deforms between them. Measuring the residual misalignment in four
    independent sub-blocks per overlapping pair (7 pairs) splits them cleanly
    into two regimes, in y:

        left_dorsal      + ventral            spread 7.5   [ 2.2, -0.2, -5.2, -0.5]
        left_dorsal      + left_optic_lobe    spread 7.0   [ 4.0,  0.2,  7.2,  0.5]
        left_dorsal      + right_dorsal       spread 5.2   [-2.2, -1.8, -7.0, -3.0]
        right_dorsal     + ventral            spread 5.2   [ 2.2, -1.0,  4.2,  0.5]
        right_dorsal     + right_optic_lobe   spread 4.2   [-0.2, -1.2, -4.5, -3.5]
        right_optic_lobe + ventral            spread 2.8   [12.8, 10.0, 12.8, 10.0]
        left_optic_lobe  + ventral            spread 0.5   [-8.8, -8.8, -9.2, -9.2]

    The five dorsal/ventral pairs have residuals that VARY by 4.2-7.5 voxels
    across a single overlap — a rigid model predicts the same residual
    everywhere, so no per-tile offset can align those. The two optic-lobe pairs
    are the opposite case: nearly constant (spread 0.5 and 2.8) but offset by a
    large 9-11 voxels, which IS rigidly correctable and is not corrected here.
    Janelia's own pipeline advertises "stitched AND distortion corrected"
    precisely because the correction it applies is non-rigid.

    Removing it properly needs a piecewise/elastic fit, or accepting Janelia's
    8-bit stitched product (geometrically correct, but lossy). This demo keeps
    the full 16-bit dynamic range and lives with the seams.

USAGE:
    python demo_gsplats_3d_flylight_mcfo_63x_brain.py [--no-serve] [--serve-only]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.

OUTPUT:
    - Scene saved to: datasets/demos/gsplats_3d_flylight_mcfo_63x_brain.luxar.zarr
    - Opens in the browser; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_3d_flylight_mcfo_63x_brain",
    "title": "3D Drosophila Whole Brain (FlyLight MCFO, 63x)",
    "description": (
        "A whole fly brain's individually-coloured neurons as 419k Gaussian "
        "splats — 7,700:1 against a 3.2 Gvoxel, 99.97%-empty confocal mosaic."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 5,
        "compute": "light",
        "gpu": "none",
        # Ships in-repo under git-LFS, like every other bundled gsplat demo.
        # `ensure_dataset` still prefers the checksum-verified cache, and will
        # switch to the Zenodo leg by itself once that record URL is populated.
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_flylight_mcfo_63x"],
    "outputs": ["gsplats_3d_flylight_mcfo_63x_brain"],
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

DATASET = "gsplats_flylight_mcfo_63x"

SCENE_NAME = "gsplats_3d_flylight_mcfo_63x_brain.luxar.zarr"

# Physical voxel size (X, Y, Z) in microns, from the LSM metadata. Already baked
# into the shipped centers by pipeline step 9; kept here because the dimension
# RANGES are read off the data but the UNITS are ours to declare.
VOXEL_UM = (0.1883, 0.1883, 0.38)

# The DISPLAY RANGE the Layers panel shows, dialled in against THIS store.
#
# `intensity`/`offset` are the stored form of that window, not a gain:
#     intensity = 1 / (hi - lo)        offset = -lo / (hi - lo)
# so a 0-1.101 window is intensity 0.908, NOT 1.101. Passing the max directly
# stores a window ~8x too narrow and the scene renders blown out.
#
# The window is in STORED-AMPLITUDE units, so it is only meaningful for this
# exact store — any refit or rescale moves it. Re-tune in the panel and copy the
# numbers back here if the data is ever rebuilt.
DISPLAY_LO, DISPLAY_HI = 0.0, 1.101

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
    with asection("Resolving FlyLight MCFO gsplats"):
        paths = ensure_dataset(DATASET)
        aprint(f"Data: {paths[0]}")
        return paths[0]


# =============================================================================
# Scene construction
# =============================================================================
def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 3D scene from the pre-fitted, physically-scaled gsplats."""
    with asection("Creating FlyLight MCFO whole-brain scene"):
        node, _ = load_gsplat_node(str(data_path))
        bmin, bmax = center_bounds(node)
        aprint(f"Scene bounds (um): min={np.round(bmin, 1)} max={np.round(bmax, 1)}")
        aprint(
            f"Brain extent: {bmax[0] - bmin[0]:.0f} x {bmax[1] - bmin[1]:.0f} x "
            f"{bmax[2] - bmin[2]:.0f} um"
        )

        # Center columns are (X, Y, Z) after the pipeline's rotate-y: the thin
        # axial extent ends up last, which is also what puts the brain face-on
        # at the viewer's default camera instead of edge-on.
        dims = Dimensions(
            [
                Dimension(
                    "X", unit="µm", display=True, range=(float(bmin[0]), float(bmax[0]))
                ),
                Dimension(
                    "Y", unit="µm", display=True, range=(float(bmin[1]), float(bmax[1]))
                ),
                Dimension(
                    "Z", unit="µm", display=True, range=(float(bmin[2]), float(bmax[2]))
                ),
            ]
        )

        span = DISPLAY_HI - DISPLAY_LO

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )
            scene.attrs["title"] = "GSplats: Drosophila Whole Brain (FlyLight MCFO)"
            scene.attrs["description"] = (
                "A whole female Drosophila central brain and optic lobes labelled by "
                "MultiColor FlpOut, so individually-resolved neurons carry distinct "
                "hues. Ten raw 63x confocal tiles were registered, blended into a "
                "2573x2707x463 16-bit mosaic, and fitted as 418,722 Gaussian splats "
                "— about 7,700:1 against a volume that is 99.97% empty. Colour is "
                "sampled per-splat from the three MCFO channels. Press L for the "
                "Layers panel."
            )

            with asection("Adding gsplats"):
                scene.add_gsplats_from_file(
                    name="mcfo_neurons",
                    path=str(data_path),
                    # `volumetric` — emission–absorption. The neurons are sparse
                    # but the brain is 168 um deep, so additive summing along the
                    # ray saturates every dense arbor to white and the MCFO hues,
                    # which are the whole point of the label, disappear.
                    blending_mode="volumetric",
                    # ~10x lower than the bioimaging default of 1.0: volumetric
                    # alpha is optical depth and ACCUMULATES along the ray, so the
                    # intuitive value over-attenuates badly at this depth.
                    absorption=0.10,
                    # Tiny on purpose — this is the exposure lever. Scaling the
                    # amplitudes instead does nothing, because the viewer
                    # normalises by the stored maximum.
                    opacity=0.02,
                    # Per-splat RGB is already baked from the three channels, so
                    # no colormap: a LUT would overwrite the MCFO hues with a
                    # scalar ramp.
                    intensity=1.0 / span,
                    offset=-DISPLAY_LO / span,
                    gamma=1.0,
                    layer=True,
                )

            scene.add_text(
                "Drosophila • MultiColor FlpOut",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Janelia FlyLight • 63x confocal • VT019012 • 418,722 splats",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


def main() -> None:
    """Resolve the data, build the scene, and serve it.

    ``--serve-only`` skips straight to the viewer on an already-built scene;
    ``--no-serve`` stops after writing it.
    """
    output_path = get_demos_output_dir() / SCENE_NAME

    if not SERVE_ONLY:
        data_path = resolve_data()
        create_luxar_scene(data_path, output_path)

    if not NO_SERVE:
        launch_viewer(output_path, title="FlyLight MCFO 63x Brain")


if __name__ == "__main__":
    main()
