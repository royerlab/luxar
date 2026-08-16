#!/usr/bin/env python
"""GSplats Demo: Drosophila Whole Brain, MultiColor FlpOut (FlyLight 63x).

A whole female *Drosophila melanogaster* central brain with both optic lobes,
labelled by **MultiColor FlpOut (MCFO)** and imaged at 63x on a confocal
microscope. MCFO is a stochastic three-epitope label: each cell that flips out
expresses a random combination of HA / V5 / FLAG tags, so individual neurons
come up in distinguishable colours against an almost entirely empty volume.

This is the **sparse-and-huge** shape. The source is 2573 x 2707 x 463 voxels —
3.22 Gvoxel per channel, 12.9 Gvoxel over the four channels — of which only
**0.014%** rises above 1% of the peak intensity (the volume's own 99th
percentile sits at 0.05% of peak). Fitted to **653,759 Gaussian splats** that is
about **4,900:1** against the source voxels, which is the regime splats are for:
the empty 99.99% costs nothing.

WHY THE COMPOSITE IS FITTED ONCE, NOT PER CHANNEL:
    A neuron's MCFO hue is the RATIO of the three channels at the SAME voxels.
    Fitting each channel separately gives three splat sets whose centres do not
    co-locate, and every axon renders candy-striped. So the fit runs on the
    per-voxel maximum of the three (gain-balanced at p99.99 first, or whichever
    channel is brightest wins nearly every voxel), and each splat then reads its
    colour from the three channels at its own centre. Result here: a near-even
    41 / 27 / 31 % three-way colour split, and 0.06% of splats uncoloured.

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

WHY JANELIA'S STITCHED PRODUCT, NOT THE RAW TILES:
    FlyLight distributes each 63x brain three ways, and none is both stitched
    and 16-bit:

        *.lsm.bz2            per-tile, 16-bit, ~7.2 GB for the ten files
        unaligned_stack.h5j  stitched, 8-bit HEVC, 199 MB   <- used here
        aligned_stack.h5j    stitched + template-aligned, 8-bit

    This demo takes the stitched H5J. Re-stitching the raw tiles ourselves was
    tried and abandoned: the raw LSMs carry NO stage coordinates (``Positions``,
    ``TilePositions`` and ``OriginX/Y/Z`` are all zero), and once placement is
    recovered by registration the result still shows two artefacts that
    Janelia's pipeline — "stitched AND distortion corrected" — already solves:

    * **Ghosting.** The five tile positions are separate acquisitions and the
      sample deforms between them. Residual misalignment measured in four
      sub-blocks per overlapping pair VARIES by 4.2-7.5 voxels across a single
      overlap, so no per-tile translation can align them; it needs a non-rigid
      fit.
    * **Coverage seams.** Blending N overlapping tiles averages N independent
      noise realisations, so doubly-covered regions are quieter by ~sqrt(2)
      (measured 1.442 against 1.414 predicted). Coverage count is
      piecewise-constant with rectangular boundaries, so the fit sees
      box-shaped steps.

    The cost of the H5J is dynamic range in the dim tail, not resolution: the
    raw tiles union to 462 x 2712 x 2577 against the H5J's 463 x 2707 x 2573,
    i.e. the same voxel grid to within 0.2%, at the same 0.19 um sampling.

VOXEL SIZE:
    0.19 x 0.19 um laterally, 0.38 um axially, from the H5J metadata
    (``voxel_size``) — recorded by the instrument, not inferred. The shipped
    centres are already in micrometres and the brain measures 663 x 303 x 167
    um, the right envelope for an adult Drosophila central brain plus optic
    lobes, which independently checks both the calibration and the orientation.

PIPELINE (how the bundled gsplats were produced — provenance, NOT re-run here):
    1. Fetch ``VT019012-20140423_20_D5-f-63x-brain-GAL4-unaligned_stack.h5j``
       (199 MB) from the public S3 bucket.
    2. Decode it. H5J is an HDF5 container holding one H.265 elementary stream
       per channel, padded to macroblock bounds; decode with ffmpeg, crop the
       padding (``pad_right``/``pad_bottom``), write zarr.
    3. Gain-balance the three signal channels at p99.99 and take the per-voxel
       maximum as the fit target. Channel_3 (nc82 neuropil reference) is
       EXCLUDED — at 10-19% occupancy it is not sparse and would dominate.
    4. ``gsplat cal --auto-region --feature-metric edges --k-star-metric gain``
       -> a region-scoped density (K* 128,000, confidence 13.35 dB).
    5. ``gsplat fit --tiling content --cal … --flat --floor auto`` -> 653,759
       splats over 75 content-balanced boxes.
    6. Colour each splat by sampling the three channels at its own centre.
    7. ``gsplat lod --recipe stream --target-ms 200`` -> progressive ladder.
    8. ``gsplat transform --rotate-y 90`` (face-on), then a SECOND call for
       ``--scale 0.19,0.19,0.38``: one invocation applies ``--scale`` BEFORE
       ``--rotate-*``, which would put the axial pitch on a lateral axis.
    9. ``gsplat transform --rotate-z 48.84`` -> level. The specimen sits
       diagonally on the imaging canvas (the canvas is square because it is the
       union of five square tile positions, not because the brain is). The
       angle is the amplitude-weighted principal axis of the splat cloud in the
       view plane; levelling takes the bounding box from 483 x 508 to 663 x 303
       um, so the viewer frames the brain instead of empty corners.

    The shipped centres are therefore in micrometres about an arbitrary origin
    (the brain's centre lands near (5, 36, 43) um, not at 0) — which is why the
    camera below is derived from the loaded bounds rather than hard-coded.

WHY ``--floor auto`` AND NOT A PERCENTILE:
    A ``pN`` floor subtracts the Nth percentile of ALL voxels, which on sparse
    data lands wherever the sparsity puts it rather than where the noise ends.

    The sweep below is a 96 x 640 x 640 CROP of this specimen (39 Mvoxel), not
    the shipped 654k-splat whole-volume fit — the point is the RANKING, and the
    crop is small enough to fit five arms at a fixed seed budget. In it, 1.01%
    of voxels are foreground (above 10% of max) and 12.9% fall in the dim band
    (1-10%, where thin faint neurites live), so a p99 floor lands squarely
    inside signal — at 1.34% of the crop's max. Every arm scored against the
    UNFLOORED original:

        floor   splats   global   foreground   dim-band mass recovered
        none    47,172   41.90    28.49 dB     42.0%
        auto    46,020   41.76    28.24 dB     40.7%
        p95     39,859   40.33    27.04 dB     23.0%
        p99     15,483   35.81    18.86 dB      0.6%

    ``auto`` is within 0.25 dB of no floor at all, so pedestal removal is
    essentially free; all the damage comes from raising the floor. A p99-floored
    fit also LOOKS better in a MIP (the haze is gone and the render is crisper
    than its own source) — that is the trap. Judge a floor on foreground /
    dim-band PSNR against unfloored data, never on how the render looks.

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
        "A whole fly brain's individually-coloured neurons as 654k Gaussian "
        "splats — 4,900:1 against a 3.2 Gvoxel, 99.99%-empty confocal stack."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 8,
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

import math
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import ensure_dataset, launch_viewer, parse_demo_flags
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_flylight_mcfo_63x"

SCENE_NAME = "gsplats_3d_flylight_mcfo_63x_brain.luxar.zarr"

# Physical voxel size (X, Y, Z) in microns, from the H5J metadata. Already baked
# into the shipped centers by pipeline step 8; kept here because the dimension
# RANGES are read off the data but the UNITS are ours to declare.
VOXEL_UM = (0.19, 0.19, 0.38)

# The DISPLAY RANGE the Layers panel shows, dialled in against THIS store.
#
# `intensity`/`offset` are the stored form of that window, not a gain:
#     intensity = 1 / (hi - lo)        offset = -lo / (hi - lo)
# so a 0-2.723 window is intensity 0.367, NOT 2.723. Passing the max directly
# stores a window ~7x too narrow and the scene renders blown out.
#
# The window is in STORED-AMPLITUDE units, so it is only meaningful for this
# exact store — any refit or rescale moves it. Re-tune in the panel and copy the
# numbers back here if the data is ever rebuilt.
DISPLAY_LO, DISPLAY_HI = 0.0, 2.723

# Opacity is the exposure lever and wants to be tiny; scaling the amplitudes
# instead does nothing, because the viewer normalises by the stored maximum.
# Absorption is much higher here (0.81) than on a hazier volume: with the
# background gone, depth cueing can be strong without muddying anything.
OPACITY = 0.02
ABSORPTION = 0.81

# Camera framing. The viewer's default fits a CUBE of the box's LARGEST dimension
# into 75% of the frame and then adds a further 20% margin
# (`calculateCameraDistance`), i.e. ~1.6 x maxDim / (2 tan(fov/2)) — about 1220 um
# on this 663 x 303 brain, which leaves it small. Place the camera explicitly.
#
#     visible_height(d) = 2 d tan(fov/2)      visible_width(d) = that * aspect
#
# Two corrections over the naive fit:
#
# 1. Measure at the NEAR FACE, not the target plane. The frustum narrows towards
#    the camera, so the widest part of a 167 um-deep object to worry about is its
#    camera-facing side at `d - depth/2`. Fitting at the centre plane over-fills
#    the near face and clips its corners (7% of the width, here, before this).
#    Hence the `+ depth/2` term.
# 2. `fill` is a fraction of the frame the object should occupy, so the extent is
#    divided by it BEFORE the fit.
#
# WHAT THE ASPECT MEANS. `fov` is VERTICAL, so visible width scales with the live
# viewport aspect while a baked distance cannot. This brain is 2.19:1 — far wider
# than any window — so width binds at every realistic aspect and no single
# distance can fill it everywhere. `CAMERA_ASPECT` is therefore a declared
# calibration point, not a safety margin: at exactly this aspect the brain
# occupies `CAMERA_FILL` of the width; a WIDER window leaves more margin, a
# NARROWER one eats into it and below ~1.29 (= 1.4 x `CAMERA_FILL`) starts
# cropping the outer optic lobes. 1.4 is the landscape floor this demo is
# calibrated for. Framing for a square window instead — the never-crop choice —
# would put the camera at ~912 um: closer than the viewer's own ~1220 um default,
# but the brain would then occupy only ~0.66 of the width at 1.4 and ~0.52 at
# 16:9, instead of 0.92. That headroom nobody sees is most of what the explicit
# camera is here to recover.
CAMERA_FOV = 47.0
CAMERA_FILL = 0.92
CAMERA_ASPECT = 1.4

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]


# =============================================================================
# Camera
# =============================================================================
def camera_distance(width: float, height: float, half_depth: float) -> float:
    """Distance from the bbox CENTRE that fills ``CAMERA_FILL`` of the frame.

    The fit is done at the object's camera-facing side (``half_depth`` in front
    of the centre) and the half-depth added back, so it is the NEAR face —
    where the frustum is narrowest — that lands inside the frame.
    """
    vh = 2.0 * math.tan(math.radians(CAMERA_FOV / 2.0))
    fit_w = (width / CAMERA_FILL) / (vh * CAMERA_ASPECT)
    fit_h = (height / CAMERA_FILL) / vh
    return max(fit_w, fit_h) + half_depth


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
    """Build the 3D scene from the pre-fitted, levelled, physically-scaled gsplats."""
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
        cx, cy, cz = ((float(bmin[i]) + float(bmax[i])) / 2 for i in range(3))
        width = float(bmax[0] - bmin[0])
        height = float(bmax[1] - bmin[1])
        half_depth = float(bmax[2] - bmin[2]) / 2.0
        distance = camera_distance(width, height, half_depth)
        aprint(
            f"Camera: fov {CAMERA_FOV}, distance {distance:.0f} um "
            f"(near face at {distance - half_depth:.0f} um)"
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    camera=CameraConfig(
                        position=(cx, cy, cz + distance),
                        target=(cx, cy, cz),
                        up=(0.0, 1.0, 0.0),
                        fov=CAMERA_FOV,
                    ),
                ),
            )
            scene.attrs["title"] = "GSplats: Drosophila Whole Brain (FlyLight MCFO)"
            scene.attrs["description"] = (
                "A whole female Drosophila central brain and optic lobes labelled by "
                "MultiColor FlpOut, so individually-resolved neurons carry distinct "
                "hues. Janelia's stitched 63x confocal stack — 2573x2707x463, with "
                "99.99% of it below 1% of peak — fitted as 653,759 Gaussian splats, "
                "roughly 4,900:1. "
                "Colour is sampled per-splat from the three MCFO channels. Press L "
                "for the Layers panel."
            )

            with asection("Adding gsplats"):
                scene.add_gsplats_from_file(
                    name="mcfo_neurons",
                    path=str(data_path),
                    # `volumetric` — emission–absorption. The neurons are sparse
                    # but the brain is 167 um deep, so additive summing along the
                    # ray saturates every dense arbor to white and the MCFO hues,
                    # which are the whole point of the label, disappear.
                    blending_mode="volumetric",
                    absorption=ABSORPTION,
                    opacity=OPACITY,
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
                "Janelia FlyLight • 63x confocal • VT019012 • 653,759 splats",
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

    if SERVE_ONLY:
        if not output_path.exists():
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
            return
    else:
        data_path = resolve_data()
        create_luxar_scene(data_path, output_path)

    if not NO_SERVE:
        launch_viewer(output_path)


if __name__ == "__main__":
    main()
