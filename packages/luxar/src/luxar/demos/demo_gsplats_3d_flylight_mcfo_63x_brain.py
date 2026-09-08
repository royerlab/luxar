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
percentile sits at 0.05% of peak). The hosted refit contains **660,035 Gaussian
splats**; the bundled fallback contains 653,759. The runtime caption reads the
selected archive, so either is described accurately. Both are about **4,900:1**
against the source voxels, which is the regime splats are for: the empty 99.99%
costs nothing.

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

PIPELINE (how the hosted refit was produced; re-run with ``--recompute``):
    1. Fetch ``VT019012-20140423_20_D5-f-63x-brain-GAL4-unaligned_stack.h5j``
       (199 MB) from the public S3 bucket.
    2. Decode it. H5J is an HDF5 container holding one H.265 elementary stream
       per channel, padded to macroblock bounds; decode with ffmpeg, crop the
       padding (``pad_right``/``pad_bottom``), write zarr.
    3. Gain-balance the three signal channels at p99.99 and take the per-voxel
       maximum as the fit target. Channel_3 (nc82 neuropil reference) is
       EXCLUDED — at 10-19% occupancy it is not sparse and would dominate.
    4. ``gsplat cal --auto-region --feature-metric edges``
       -> a region-scoped density (K* 128,000, confidence 13.35 dB).
    5. ``gsplat fit --tiling content --cal … --flat --floor auto`` -> 660,035
       splats over 75 content-balanced boxes.
    6. Colour each splat by sampling the three channels at its own centre.
    7. ``gsplat lod --recipe stream --target-ms 200`` -> progressive ladder.
    8. ``gsplat transform --rotate-y 90`` (face-on), then a SECOND call for
       ``--scale 0.19,0.19,0.38``: one invocation applies ``--scale`` BEFORE
       ``--rotate-*``, which would put the axial pitch on a lateral axis.
    9. ``gsplat transform --rotate-z 48.71`` -> level. The specimen sits
       diagonally on the imaging canvas (the canvas is square because it is the
       union of five square tile positions, not because the brain is). The
       angle is the amplitude-weighted principal axis of the splat cloud in the
       view plane and is recomputed for each fit. On the bundled fallback, the
       analogous levelling takes the bounding box from 483 x 508 to 663 x 303
       um, so the viewer frames the brain instead of empty corners.

    The shipped centres are therefore in micrometres about an arbitrary origin
    (the brain's centre lands near (5, 36, 43) um, not at 0) — which is why the
    camera below is derived from the loaded bounds rather than hard-coded.

WHY ``--floor auto`` AND NOT A PERCENTILE:
    A ``pN`` floor subtracts the Nth percentile of non-zero voxels, which on sparse
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
    python demo_gsplats_3d_flylight_mcfo_63x_brain.py [--no-serve] [--serve-only] [--recompute]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.
    --recompute:   Rebuild the archive from Janelia's pinned raw H5J.

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
        "local_data": None,
    },
    "caches": ["gsplats_flylight_mcfo_63x"],
    "outputs": ["gsplats_3d_flylight_mcfo_63x_brain"],
    "citation": {
        "short": "Janelia FlyLight Project Team, HHMI (Meissner et al. 2023)",
        "ref": "Meissner et al. 2023",
        "doi": "10.7554/eLife.80660",
        "license": "CC BY 4.0",
    },
}

import math
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    detect_device,
    download_with_checksum,
    ensure_dataset,
    launch_viewer,
    local_fit_path,
    parse_demo_flags,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._h5j import (
    decode_h5j_channel,
    reference_channel_index,
    signal_channel_indices,
)
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds, total_splats
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
# The window is in raw amplitude units for this direct-colour node, so it is only
# meaningful for this exact store — any refit or rescale moves it. Re-tune in the
# panel and copy the numbers back here if the data is ever rebuilt.
#
# It is NOT a percentile of the stored amplitudes, and it is worth knowing that
# before trying to derive it. Measured on the 1000-iteration build (653,759
# splats), which this one replaces:
#
#     min 0.0048   median 7.51   mean 10.52   p99.9 79.16   max 188.81
#
# 2.723 is below the MEDIAN, so no percentile rule lands anywhere near it — the
# number is a judgement made in the panel against the rendered image, not a
# statistic, and it has to be re-made by eye rather than recomputed.
#
# Two consequences worth flagging together, because they compound:
#   * A REFIT changes the stored distribution, so this constant is stale the
#     moment the archive is rebuilt, independently of any viewer change.
#   * This node carries direct colour and no colormap, so `intensity` acts as a
#     gain on radiance rather than as a display window over a LUT. Any change
#     that rescales amplitudes at scene-insertion time therefore shifts the
#     effective exposure here, and this constant does not compensate.
#
# SETTLED by rendering both builds at a fixed camera through this demo's own
# scene builder (2026-08-26). Raw amplitude units DO reach the shader here: had
# the viewer normalised by the stamped maximum, the two renders would have been
# near-identical, and they are not.
#
# But neither the max nor the median governs exposure -- the amplitude SUM does,
# because total emitted radiance is a sum over splats:
#
#     amplitudes   median -9.4%   p99.9 +0.6%   max -8.0%   SUM -1.4%
#     rendered     total light -2.1%   mean fg -1.4%   p99 -0.7%
#                  median lit pixel -7.5%
#
# So overall exposure moved ~1.4-2.1%, tracking the sum, and the bright arbors
# are unchanged (p99 -0.7%). The -7.5% sits in the median lit pixel, i.e. the
# faint neuropil, tracking the amplitude median. Nothing clips either way
# (p99 ~= 0.66). 2.723 therefore stands for this build, and the earlier guess
# that the 8% max drop would dim the scene by 8% was simply the wrong statistic.
#
# Keep the constant suspect after a refit that moves the SUM materially; a refit
# that moves only the max is not a reason to touch it.
DISPLAY_LO, DISPLAY_HI = 0.0, 2.723

# These values were tuned against the archive's raw amplitude units. The scene
# therefore opts out of insertion-time amplitude normalisation below; otherwise
# the archive's p99.9 ~= 79 scale would dim both radiance and optical depth by
# that factor while leaving this window and opacity unchanged.

# Splat count of the hosted refit. The bundled fallback is an older generation,
# so runtime descriptions and captions derive their count from whichever archive
# is actually loaded. This pin remains as a cross-check against the committed
# measurements sidecar and the hosted-generation recipe in the docstring.
N_SPLATS = 660_035

# Opacity is the exposure lever and wants to be tiny. Scaling amplitudes also
# changes exposure because raw amplitude units reach this direct-colour node,
# but opacity is the clearer scene-level control.
# Absorption is much higher here (0.81) than on a hazier volume: with the
# background gone, depth cueing can be strong without muddying anything.
OPACITY = 0.02
ABSORPTION = 0.81

# Camera framing. The viewer's default fits the larger screen-plane extent into
# 75% of the shorter viewport axis at the box's near face
# (`calculateCameraDistance`) — about 805 um on this 663 x 303 x 167 brain,
# which still leaves it small. Place the camera explicitly for this composition.
#
#     visible_height(d) = 2 d tan(fov/2)      visible_width(d) = that * aspect
#
# Two requirements for the explicit fit:
#
# 1. Keep the viewer default's NEAR-FACE convention. The frustum narrows towards
#    the camera, so the widest part of a 167 um-deep object to worry about is its
#    camera-facing side at `d - depth/2`. Fitting at the centre plane would
#    over-fill the near face and clip its corners. Hence the `+ depth/2` term.
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
# would put the camera at ~671 um: closer than the viewer's own ~805 um default,
# but the brain would then occupy only ~0.66 of the width at 1.4 and ~0.52 at
# 16:9, instead of 0.92. That headroom nobody sees is most of what the explicit
# camera is here to recover.
CAMERA_FOV = CINEMATIC_FOV_DEG
CAMERA_FILL = 0.92
CAMERA_ASPECT = 1.4

# Reference cage. MCFO labels only a handful of neurons out of a whole brain, so
# the scene is mostly empty and there is no cue for how big the specimen is,
# where the edges of the imaged stack are, or how much of the black is "no
# label" rather than "outside the data". A faint box around the stack bounds,
# ruled at round micron intervals, supplies all three.
#
# Drawn ADDITIVE at low opacity on purpose: additive never occludes the neurons
# behind it, so the cage can cross the specimen without hiding any of it, and at
# this opacity it reads as a faint scaffold rather than as scene content. The
# grid lines are half the width and about a third the radiance of the box edges,
# so the bounds stay legible as the outer shape.
GRID_STEP_UM = 100.0  # ruling interval — "decimal" in the data's own units
BOX_WIDTH_UM = 0.9  # ~1.5 px at the authored framing
GRID_WIDTH_UM = 0.45  # ~1 px — the "thin" of a thin grid
BOX_RGB = (0.42, 0.54, 0.72)
GRID_RGB = (0.14, 0.19, 0.27)
CAGE_OPACITY = 0.18

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]


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
# Recompute (--recompute): rebuild the archive from Janelia's raw H5J
# =============================================================================
# This pipeline used to exist only as prose in the docstring above, which is why
# the shipped archive carried no PSNR: once the working directory looked gone
# there was no way to re-measure it. A levelled, micron-scaled archive CANNOT be
# scored after the fact against a voxel-grid volume, so the quality figures have
# to be stamped at fit time — which means the fit has to be reproducible.
#
#: Public S3, no credentials. This URL plus the code below is the whole recipe.
H5J_URL = (
    "https://janelia-flylight-imagery.s3.amazonaws.com/Annotator%20Gen1%20MCFO/"
    "VT019012/VT019012-20140423_20_D5-f-63x-brain-GAL4-unaligned_stack.h5j"
)
H5J_NAME = "VT019012-20140423_20_D5-f-63x-brain-GAL4-unaligned_stack.h5j"
H5J_SHA256 = "189595b1013af62f71158556fab75ae535c9a8d3765138fa9a462583f09eed98"

#: Percentile the signal channels are balanced at, for the composite and again
#: for the per-splat colour. High enough to sit in real signal, low enough not
#: to ride on one hot voxel.
BALANCE_PERCENTILE = 99.99

RECOMPUTE_DIR = local_fit_path(DATASET, H5J_NAME).parent


def _luxar(*args: str) -> None:
    """Run one ``luxar`` CLI command in-process.

    The recompute path drives the CLI rather than the fitting API deliberately.
    ``gsplat fit --tiling content`` resolves the background floor ONCE against
    the whole volume before it plans boxes; handing ``auto`` to the planner
    directly re-estimates it per box crop, and abutting boxes that subtract
    different pedestals show up as brightness steps at box boundaries.
    Reproducing that resolution here would be a second copy of it, free to drift.
    """
    from luxar.cli import app

    aprint(f"$ luxar {' '.join(args)}")
    try:
        app(list(args))
    except SystemExit as exc:  # the CLI exits even on success
        if exc.code not in (0, None):
            raise RuntimeError(
                f"`luxar {' '.join(args)}` failed with exit code {exc.code}"
            ) from exc


def fetch_h5j() -> Path:
    """Download Janelia's stitched H5J, or reuse a verified cached copy.

    ``download_with_checksum`` retries with backoff and RESUMES a partial
    transfer against a validated ETag, which matters for a single
    multi-gigabyte object over a link that may drop, and it verifies the digest
    on every path -- cache hit included -- deleting the file if it does not
    match.

    Checking a cached copy is the point, not an extra: a file at this name is
    not evidence of the right file. It may be a truncated earlier attempt or a
    different sample someone left there, and the pinned hash exists because
    this demo describes ONE specimen.
    """
    RECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)
    target = RECOMPUTE_DIR / H5J_NAME
    with asection(f"Fetching {H5J_NAME}"):
        aprint(f"from {H5J_URL}")
        download_with_checksum(H5J_URL, target, expected_sha256=H5J_SHA256)
    aprint(f"H5J: {target} ({target.stat().st_size:,} bytes)")
    return target


def build_composite(h5j_path: Path, out_zarr: Path) -> None:
    """Gain-balance the SIGNAL channels and store their per-voxel maximum.

    The composite is fitted ONCE rather than each channel separately. An MCFO
    hue is the RATIO of the three channels at the SAME voxels; three independent
    fits scatter the three colours across three different splat sets and the hue
    — the entire point of MultiColor FlpOut — is destroyed.

    The reference channel is excluded, identified from the file's own
    ``channel_spec`` rather than by index: at 10-19% occupancy against ~0.1% for
    the signal, sweeping nc82 in would let it dominate the splat budget.

    Gains are DERIVED here rather than hardcoded. The shipped run recorded
    ``[1.3133, 1.0, 1.1720]`` for this sample, but a constant would silently be
    wrong for any other — and an earlier attempt's recorded gains
    (``[1.2339, 1.5959, 1.0]``, from p99.99 values in the *thousands*) came from
    16-bit raw tiles, not from this 8-bit H5J at all.
    """
    from luxar._zarr_compat import create_array, open_group
    from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor

    with asection("Building the gain-balanced composite"):
        signal = signal_channel_indices(h5j_path)
        if len(signal) != 3:
            raise ValueError(
                f"MCFO colouring requires exactly 3 signal channels, got {signal}"
            )
        aprint(
            f"signal channels {signal}, "
            f"reference channel {reference_channel_index(h5j_path)} (excluded)"
        )

        tops = []
        shape = None
        for channel in signal:
            volume = decode_h5j_channel(h5j_path, channel)
            if shape is None:
                shape = volume.shape
            tops.append(float(np.percentile(volume, BALANCE_PERCENTILE)))
            del volume
        ceiling = max(tops)
        gains = [ceiling / t if t > 0 else 1.0 for t in tops]
        aprint(
            f"p{BALANCE_PERCENTILE} {[round(t, 1) for t in tops]} "
            f"gains {[round(g, 4) for g in gains]}"
        )

        assert shape is not None
        composite = np.zeros(shape, dtype=np.uint16)
        for channel, gain in zip(signal, gains):
            volume = decode_h5j_channel(h5j_path, channel)
            for start in range(0, shape[0], 16):
                slab = slice(start, min(start + 16, shape[0]))
                scaled = np.rint(volume[slab].astype(np.float32) * np.float32(gain))
                np.maximum(
                    composite[slab], scaled.astype(np.uint16), out=composite[slab]
                )
            del volume
        aprint(
            f"composite {composite.shape} max {composite.max():.1f} "
            f"nonzero {100.0 * np.count_nonzero(composite) / composite.size:.2f}%"
        )
        with asection(f"Writing {out_zarr.name}"):
            grp = open_group(str(out_zarr), mode="w")
            create_array(
                grp,
                "composite",
                data=composite,
                chunks=(64, 256, 256),
                compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, np.uint16),
            )


def colour_from_channels(h5j_path: Path, fit_path: Path, out_path: Path) -> None:
    """Give every splat the MCFO hue of the voxel it sits on.

    The fit is monochrome — it was made against the composite — so colour is
    applied afterwards by sampling the three signal channels at each splat's own
    centre. Sampling per splat is what preserves the hue: an MCFO colour is the
    ratio of the three channels at ONE location.

    Channels are sampled one at a time and released; three 3.2 Gvoxel float
    channels held together would be ~39 GB.
    """
    from luxar._zarr_compat import open_group
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.save_gsplats import save_gsplats

    with asection("Colouring splats from the three signal channels"):
        data = GSplatData.load(str(fit_path))
        centres = np.asarray(data.centers, dtype=np.float64)
        n = centres.shape[0]
        aprint(f"{n:,} splats to colour")

        signal = signal_channel_indices(h5j_path)
        if len(signal) != 3:
            raise ValueError(
                f"MCFO colouring requires exactly 3 signal channels, got {signal}"
            )
        rgb = np.zeros((n, len(signal)), dtype=np.float32)
        idx = np.rint(centres).astype(np.int64)
        valid: np.ndarray | None = None
        tops: list[float] = []
        for slot, ch in enumerate(signal):
            vol = decode_h5j_channel(h5j_path, ch)
            if valid is None:
                valid = np.all((idx >= 0) & (idx < np.asarray(vol.shape)), axis=1)
            sel = idx[valid]
            rgb[valid, slot] = vol[sel[:, 0], sel[:, 1], sel[:, 2]]
            # Take the balance percentile from the VOLUME while it is in hand.
            # A channel gain describes the CHANNEL, not the splat sample: taking
            # it from the sampled values instead re-weights the hue balance by
            # where the splats happen to sit, which measurably tilted the colour
            # means to [118, 146, 117] against the shipped [141, 136, 145].
            tops.append(float(np.percentile(vol, BALANCE_PERCENTILE)))
            del vol

        # Balance the channels against each other first, so a neuron labelled in
        # a globally dimmer channel is not systematically darker in hue. The
        # gains come from the volume percentiles measured above, against a shared
        # ceiling, which is how the shipped run derived them.
        ceiling = max(tops)
        aprint(
            f"p{BALANCE_PERCENTILE} per channel {[round(t, 1) for t in tops]} "
            f"-> gains {[round(ceiling / t, 4) if t > 0 else 1.0 for t in tops]}"
        )
        # Each channel becomes a fraction of its OWN robust maximum. The shipped
        # run expressed the same thing as gains against a shared ceiling; the two
        # differ only by one global factor, which the per-splat normalisation
        # below removes, so the resulting hue is identical.
        for slot, top in enumerate(tops):
            if top > 0:
                rgb[:, slot] /= top
        # Then normalise each splat to FULL BRIGHTNESS, so colour carries HUE ONLY
        # and every scrap of magnitude lives in the amplitude.
        #
        # This is not cosmetic. The shader multiplies emission by the colour, so
        # leaving the sampled magnitude in the colour applies intensity TWICE and
        # throws away most of the light: measured against the previously shipped
        # archive, keeping raw magnitudes gave a mean peak channel of 36/255
        # against its 255/255, and 5.1x less total emitted light. Loic saw it
        # immediately as "dimmer, and the hues less saturated, more grayish" --
        # the greyness being a CONSEQUENCE of the dimness, since ACES desaturates
        # the dark end. Per-splat saturation was in fact already slightly higher
        # (0.862 vs 0.785); it simply had no brightness to show it at.
        peak = rgb.max(axis=1)
        lit = peak > 0
        rgb[lit] /= peak[lit, None]

        uncoloured = int(np.count_nonzero(rgb.max(axis=1) == 0))
        share = rgb.sum(axis=0)
        share = share / max(share.sum(), 1e-9)
        aprint(
            f"colour share {np.round(100.0 * share, 1).tolist()}%, "
            f"{100.0 * uncoloured / n:.3f}% uncoloured"
        )

        # Demo colours are LINEAR light, not sRGB - the viewer applies the
        # transfer curve itself.
        colours = np.rint(rgb * 255.0).astype(np.uint8)

        # Write through save_gsplats rather than assigning `data.colors`.
        # `GSplatData.colors` is a DERIVED view -- "cached concatenation of all
        # LOD colors" -- so assigning it updates a cache that the save path does
        # not read, and the colours vanish with no error and no warning. The
        # symptom is has_colors=False on the written archive, which then survives
        # every downstream stage.
        source = open_group(str(fit_path), mode="r")
        fitting = dict(source["fitting"].attrs) if "fitting" in source else None
        fitting_config = (
            dict(source["fitting/config"].attrs) if "fitting/config" in source else None
        )
        provenance = (
            dict(source["provenance"].attrs) if "provenance" in source else None
        )
        save_gsplats(
            str(out_path),
            centers=np.asarray(data.centers),
            amplitudes=np.asarray(data.amplitudes),
            cholesky_factors=np.asarray(data.cholesky_factors),
            colors=colours,
            fitting_info=fitting,
            fitting_config=fitting_config,
            provenance_info=provenance,
            description=source.attrs.get("description"),
            truncation_radius=data.truncation_radius,
        )
        written = GSplatData.load(str(out_path))
        if written.colors is None:
            raise RuntimeError(
                f"{out_path.name} came back with no colours; the MCFO hue is the "
                "point of this demo, so refusing to continue silently."
            )
        aprint(f"wrote {out_path} with colours {written.colors.shape}")


def levelling_angle_deg(node) -> float:
    """Degrees about the view axis that bring the specimen level.

    The brain lies DIAGONALLY on the imaging canvas — the canvas is square
    because it is the union of five square tile positions, not because the
    specimen is — so an unlevelled scene frames empty corners.

    The angle is the amplitude-weighted principal axis of the splat cloud in the
    view plane. It is DATA-DERIVED and moves with any refit, so it must be
    recomputed rather than carried as a constant (the bundled fallback used
    48.84; the hosted refit used 48.71).
    """
    from luxar.gsplats.tree import iter_leaves

    # Columns 0 and 1, because that is the plane `transform --rotate-z` acts on
    # (--spatial-dims defaults to 0,1,2 and the listed order assigns X/Y/Z).
    # Picking "the two widest axes" instead is WRONG and silently so: past ~45
    # degrees of tilt the second axis becomes the wider one, the roles swap, and
    # the returned angle is off by exactly 90 degrees. Measured on synthetic
    # clouds, a 70-degree tilt came back as -20 and levelling made the in-plane
    # extent ratio WORSE (2.30 -> 1.15) instead of better.
    xs, ys, ws, thin = [], [], [], []
    for leaf in iter_leaves(node):
        for s in leaf.additive_sublods:
            c = np.asarray(s.centers, dtype=np.float64)
            xs.append(c[:, 0])
            ys.append(c[:, 1])
            ws.append(np.asarray(s.amplitudes, dtype=np.float64).ravel())
            thin.append(c.max(axis=0) - c.min(axis=0))
    spread = np.max(np.asarray(thin), axis=0)
    if int(np.argmin(spread)) != 2:
        aprint(
            f"  WARNING: column 2 is not the narrowest axis (extents "
            f"{np.round(spread, 0).tolist()}), so columns 0/1 may not be the "
            "view plane; the levelling angle would rotate the wrong pair."
        )
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    w = w / w.sum()
    x = x - (w * x).sum()
    y = y - (w * y).sum()
    cxx = float((w * x * x).sum())
    cyy = float((w * y * y).sum())
    cxy = float((w * x * y).sum())
    return -math.degrees(0.5 * math.atan2(2.0 * cxy, cxx - cyy))


def recompute_archive() -> Path:
    """Rebuild the fitted archive from the raw H5J and return its path.

    Each stage writes into the demo's ``local/`` cache, so an interrupted run resumes at the
    first missing artifact instead of starting over. The FIT is the stage that
    matters for provenance: it stamps ``psnr_db`` and ``foreground_psnr_db``.
    """
    composite_zarr = RECOMPUTE_DIR / "composite.zarr"
    cal_json = RECOMPUTE_DIR / "cal_h5j.json"
    fit_path = RECOMPUTE_DIR / "fit.gsplats.zarr"
    coloured = RECOMPUTE_DIR / "fit_coloured.gsplats.zarr"
    laddered = RECOMPUTE_DIR / "fit_stream.gsplats.zarr"
    faced = RECOMPUTE_DIR / "fit_faceon.gsplats.zarr"
    scaled = RECOMPUTE_DIR / "fit_um.gsplats.zarr"
    final = RECOMPUTE_DIR / "flylight_mcfo_63x.gsplats.zarr"

    with asection("Recomputing the FlyLight 63x archive from the raw H5J"):
        h5j = fetch_h5j()

        if not composite_zarr.exists():
            build_composite(h5j, composite_zarr)
        else:
            aprint(f"composite present: {composite_zarr}")

        if not cal_json.exists():
            _luxar(
                "gsplat",
                "cal",
                str(composite_zarr),
                str(cal_json),
                "--array-key",
                "composite",
                "--auto-region",
                "--feature-metric",
                "edges",
                "--device",
                detect_device(),
            )

        if not fit_path.exists():
            _luxar(
                "gsplat",
                "fit",
                str(composite_zarr),
                str(fit_path),
                "--array-key",
                "composite",
                "--tiling",
                "content",
                "--cal",
                str(cal_json),
                "--flat",
                "--floor",
                "auto",
                "--device",
                detect_device(),
            )

        if not coloured.exists():
            colour_from_channels(h5j, fit_path, coloured)

        if not laddered.exists():
            _luxar(
                "gsplat",
                "lod",
                str(coloured),
                str(laddered),
                "--recipe",
                "stream",
                "--target-ms",
                "200",
            )

        # Orientation: rotate FIRST, scale in a SECOND call. A single
        # invocation applies --scale BEFORE --rotate-*, which would put the
        # 0.38 um axial pitch onto a lateral axis.
        if not faced.exists():
            _luxar("gsplat", "transform", str(laddered), str(faced), "--rotate-y", "90")
        if not scaled.exists():
            _luxar(
                "gsplat",
                "transform",
                str(faced),
                str(scaled),
                "--scale",
                ",".join(str(v) for v in VOXEL_UM),
            )

        if not final.exists():
            node, _ = load_gsplat_node(str(scaled))
            angle = levelling_angle_deg(node)
            bmin, bmax = center_bounds(node)
            aprint(
                f"levelling angle {angle:.2f} deg "
                "(bundled fallback: 48.84; hosted refit: 48.71)"
            )
            _luxar(
                "gsplat",
                "transform",
                str(scaled),
                str(final),
                "--rotate-z",
                f"{angle:.2f}",
            )
            levelled, _ = load_gsplat_node(str(final))
            lmin, lmax = center_bounds(levelled)
            ext = np.round(np.asarray(lmax) - np.asarray(lmin), 0)
            aprint(
                f"bbox {np.round(np.asarray(bmax) - np.asarray(bmin), 0)} -> {ext} um "
                "(bundled fallback levelled 483x508 -> 663x303x167)"
            )
            wide = np.sort(ext)[-2:]
            if wide[1] / max(wide[0], 1.0) < 1.5:
                aprint(
                    "  WARNING: the levelled bbox is not elongated. Check the "
                    "--rotate-z sign against the 663x303 reference before shipping."
                )
        aprint(f"Archive: {final}")
        return final


# =============================================================================
# Data loading
# =============================================================================
def resolve_data() -> Path:
    """Resolve the fitted gsplats: recompute, else cache -> in-repo -> Zenodo.

    ``ensure_dataset`` verifies the manifest sha256 at every step, so a partial
    or corrupted copy is never handed back. With ``--recompute`` the archive is
    rebuilt from Janelia's raw H5J instead (see :func:`recompute_archive`) —
    which is the only way to obtain its quality figures, since they are stamped
    at fit time and cannot be recovered from the levelled store afterwards.
    """
    if RECOMPUTE:
        return recompute_archive()
    with asection("Resolving FlyLight MCFO gsplats"):
        paths = ensure_dataset(DATASET)
        aprint(f"Data: {paths[0]}")
        return paths[0]


# =============================================================================
# Scene construction
# =============================================================================
def reference_cage(bmin, bmax, step=GRID_STEP_UM):
    """Box edges plus a ruled grid over the six faces of the stack bounds.

    One node rather than two: the box and the grid are the same object to a
    reader, and per-vertex widths and colours are enough to keep the bounds
    reading as the stronger of the two.

    Grid lines sit at absolute multiples of ``step`` — 100, 200, 300 um and so
    on in the scene's own coordinates — not at fractions of the extent, so the
    spacing means a fixed physical distance and stays comparable if the bounds
    ever change. A face whose extent is shorter than one step simply gets no
    interior rules on that axis.

    Returns:
        ``(vertices, widths, colors)`` for ``line_type="segments"`` — consecutive
        PAIRS of rows are independent segments. Segments rather than an indexed
        polyline because nothing here is a connected path: 12 box edges and a set
        of disjoint rules, none of which share a joint that would benefit from a
        shared vertex.
    """
    lo = np.asarray(bmin, dtype=np.float64)
    hi = np.asarray(bmax, dtype=np.float64)

    verts: list[tuple[float, float, float]] = []
    is_box: list[bool] = []

    def seg(a, b, box):
        verts.append(tuple(float(v) for v in a))
        verts.append(tuple(float(v) for v in b))
        is_box.append(box)

    # --- 12 box edges -----------------------------------------------------
    for axis in range(3):
        u, v = [d for d in range(3) if d != axis]
        for cu in (lo[u], hi[u]):
            for cv in (lo[v], hi[v]):
                a = np.empty(3)
                b = np.empty(3)
                a[u] = b[u] = cu
                a[v] = b[v] = cv
                a[axis], b[axis] = lo[axis], hi[axis]
                seg(a, b, True)

    # --- ruled grid on each of the six faces -------------------------------
    def rules(step_axis, run_axis, face_axis, face_value):
        """Lines parallel to ``run_axis``, spaced along ``step_axis``, on one face."""
        first = np.ceil(lo[step_axis] / step) * step
        for t in np.arange(first, hi[step_axis] + 1e-9, step):
            if t <= lo[step_axis] + 1e-9 or t >= hi[step_axis] - 1e-9:
                continue  # coincides with a box edge — do not double-draw
            a = np.empty(3)
            b = np.empty(3)
            a[step_axis] = b[step_axis] = t
            a[face_axis] = b[face_axis] = face_value
            a[run_axis], b[run_axis] = lo[run_axis], hi[run_axis]
            seg(a, b, False)

    for face_axis in range(3):
        u, v = [d for d in range(3) if d != face_axis]
        for face_value in (lo[face_axis], hi[face_axis]):
            rules(u, v, face_axis, face_value)
            rules(v, u, face_axis, face_value)

    vertices = np.asarray(verts, dtype=np.float32)
    box_mask = np.repeat(np.asarray(is_box, dtype=bool), 2)  # per VERTEX
    widths = np.where(box_mask, BOX_WIDTH_UM, GRID_WIDTH_UM).astype(np.float32)
    colors = np.where(
        box_mask[:, None],
        np.asarray(BOX_RGB, dtype=np.float32),
        np.asarray(GRID_RGB, dtype=np.float32),
    ).astype(np.float32)
    return vertices, widths, colors


def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 3D scene from the pre-fitted, levelled, physically-scaled gsplats."""
    with asection("Creating FlyLight MCFO whole-brain scene"):
        node, _ = load_gsplat_node(str(data_path))
        n_splats = total_splats(node)
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
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    camera=CameraConfig(
                        position=(cx, cy, cz + distance),
                        target=(cx, cy, cz),
                        up=(0.0, 1.0, 0.0),
                    ),
                ),
                citation=DEMO_META["citation"],
            )
            scene.attrs["title"] = "GSplats: Drosophila Whole Brain (FlyLight MCFO)"
            scene.attrs["description"] = (
                "A whole female Drosophila central brain and optic lobes labelled by "
                "MultiColor FlpOut, so individually-resolved neurons carry distinct "
                "hues. Janelia's stitched 63x confocal stack — 2573x2707x463, with "
                "99.99% of it below 1% of peak — fitted as "
                f"{n_splats:,} Gaussian splats, roughly 4,900:1. "
                "Colour is sampled per-splat from the three MCFO channels. Press L "
                "for the Layers panel."
            )

            with asection("Adding gsplats"):
                scene.add_gsplats_from_file(
                    name="mcfo_neurons",
                    path=str(data_path),
                    normalize_amplitudes=False,
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

            with asection("Adding the reference cage"):
                cage_v, cage_w, cage_c = reference_cage(bmin, bmax)
                aprint(
                    f"Cage: {len(cage_v) // 2} segments, {GRID_STEP_UM:.0f} um rules"
                )
                scene.add_lines(
                    "bounds & grid",
                    vertices=cage_v,
                    widths=cage_w,
                    colors=cage_c,
                    # Consecutive PAIRS are independent segments; a polyline
                    # would join the end of one box edge to the start of the
                    # next and draw diagonals across the specimen.
                    line_type="segments",
                    dim_order=["X", "Y", "Z"],
                    # Additive so the cage never occludes a neuron behind it —
                    # it can cross the brain without hiding any of it.
                    blending_mode="additive",
                    opacity=CAGE_OPACITY,
                    sharpness=0.8,
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
            add_demo_caption(
                scene,
                f"Janelia FlyLight • 63x confocal • VT019012 • {n_splats:,} splats",
                DEMO_META.get("citation"),
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
