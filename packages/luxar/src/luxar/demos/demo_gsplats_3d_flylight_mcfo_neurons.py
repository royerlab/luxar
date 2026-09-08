#!/usr/bin/env python3
"""GSplats Demo: MCFO Fly Brain Neurons in a Volume-Rendered Neuropil

Labelled *Drosophila* neurons — long, thin, widely branching — threaded through
the brain they live in, as a single volume-rendered Gaussian splat cloud.

================================================================================
WHAT THIS DEMO IS FOR
================================================================================

Two things, and they pull in opposite directions:

1. **Thin filaments are the hard case for a Gaussian basis.** Most microscopy
   gsplat demos here fit *blobby* content — nuclei, cells, tissue. MCFO neurons
   are thin (single neurites approach the resolution limit), long-range (one
   arbor spans the brain), sparse (~0.1% of voxels), and interwoven. That is
   what makes FISBe a benchmark dataset.

2. **The neuropil is the showcase for volume rendering.** The counterstained
   brain is a dense, semi-transparent medium. Under ``blending_mode=
   "volumetric"`` (emission-absorption, Max 1995) it composites front-to-back
   with real occlusion: the brain reads as a solid body, and neurites genuinely
   pass behind it and are dimmed by it, rather than glowing through as they
   would under additive. Sparse filaments inside a dense medium, in one node,
   is exactly the case that separates volumetric from additive compositing.

--------------------------------------------------------------------------------
1. ``gsplat cal`` CANNOT calibrate K here — the data is noise-free
--------------------------------------------------------------------------------

The blind-spot cross-validation behind ``luxar gsplat cal`` needs image noise
to locate a held-out peak. FISBe's raw has effectively none: the Janelia
Workstation stitches and distortion-corrects the tiles, which scrubs the pixel
noise even though values remain 12-bit. A sweep returns
``curve_type=signal_limited``, ``still_climbing=True``, sigma_hat ~1e-10 — its
"K*" is just the top of whatever grid you supplied. Luxar warns that alternate
metrics may help on sparse data; we drove K off a measured quality curve instead.

--------------------------------------------------------------------------------
2. THE FLOOR IS THE MOST CONSEQUENTIAL KNOB — and a percentile floor is brutal
--------------------------------------------------------------------------------

``--floor`` subtracts a background level before fitting. What it removes here:

  floor       value    % voxels zeroed   % of image energy removed
  auto      0.00293             60.7%                       35.1%
  p90       0.01783             90.3%                       89.0%
  p95       0.02198             95.1%                       93.4%
  p99       0.03370             99.0%                       97.5%

A percentile floor looks like a mild cleanup and is not: ``p99`` deletes 97.5%
of the image's energy before a single splat is placed, and most of the faint
neurites go with it. No splat count recovers them. Measured on the annotated
neuron, against the RAW data:

  floor  retention  splats     fg PSNR   energy kept   faintest fifth lost
  p99      0.999    32,709      24.70        0.57x                  87.8%
  p97      0.9999   62,213      26.47        0.66x                  68.6%
  p95      0.9999   90,506      26.79        0.69x                  55.8%
  p90      0.9999  170,835      27.49        0.73x                  35.1%
  auto     0.9999  786,986      28.01        0.85x                   0.0%
  0        0.9999 1,117,196     27.64        0.87x                   0.0%

``auto`` — the CLI default — beats a zero floor on the foreground while using
30% fewer splats, and takes the dim-band dropout to zero.

Beware which reference you score against: measured against the floor-SUPPRESSED
target instead of the raw data, the p99 fit scores a respectable 28.10 dB and
the floor's damage is invisible, because the reference has had the same signal
deleted from it. That is how the p99 configuration survived review here.

--------------------------------------------------------------------------------
3. GLOBAL PSNR IS THE WRONG NUMBER — score the foreground
--------------------------------------------------------------------------------

Only ~1.5% of the raw composite's energy lies inside the annotated neurons —
41% once the floor is suppressed — so global PSNR mostly scores empty space.
(Both figures use the same mask, the union of the ground-truth instances.)
Averages also hide the failure that matters: at low splat counts the neurites
break into disconnected beads, which no PSNR variant flags. Band the mask by
intensity and look at a zoomed MIP.

Seeds are not the splat count either: the optimiser works a fixed pool of
``seeds`` and then culls by cumulative amplitude mass (``cull_retention``).
Shrinking seeds to hit a target count throws away the search that makes the
survivors good — 128,000 seeds at retention 1.0 scored *worse* (fg 25.97) than
18,623 splats drawn from a 1.2M pool (fg 27.60). Keep seeds high; tune
retention.

--------------------------------------------------------------------------------
4. ONE NODE, and alpha is OPTICAL DEPTH
--------------------------------------------------------------------------------

Neurons and neuropil occupy the same volume. As two nodes there is no correct
draw order — whichever draws first occludes the other — so they are merged into
a single node with per-splat RGBA and depth-sorted together. Alpha keeps them
distinguishable: the neuropil gets a low per-splat alpha, the neurons a high
one.

Under volumetric compositing alpha is *optical depth*, and it accumulates: with
~356K neuropil splats across a specimen 172 x 466 x 399 um, alpha 0.12 makes the
brain effectively
opaque and buries interior neurites. That is the intended look here — a solid
body with neurons emerging from it. Lower ``NEUROPIL_ALPHA`` toward ~0.01 for a
translucent haze with every neurite visible; both are one constant apart and
worth trying.

Alpha is also how the BACKGROUND is suppressed: each neuron splat's alpha ramps
with its own amplitude (``NEURON_ALPHA_*_AMP_PCT``), so haze goes optically thin
while neurites stay opaque. Doing that job with a floor instead is what cost
97.5% of the image (finding 2) — and unlike a floor this is reversible, because
the faint splats are still in the scene for the display range to recover. The
ramp therefore bottoms out at ``NEURON_ALPHA_MIN``, not at zero: alpha 0 is
folded into the splat's contribution before the shader's discard, so it is as
irreversible as never having fitted the splat at all.

The cost of one node: the Layers panel can no longer fade the neuropil
independently, because there is no second layer. That trade is deliberate.

The initial camera is measured, not defaulted. FISBe ships the UNALIGNED
FlyLight stack, so the specimen sits at whatever angle it was mounted (~52 deg
here). The demo measures its principal axis and rolls the camera to match,
framed close; rotating the splats instead would desynchronise them from FISBe's
annotations and require rotating every covariance.

--------------------------------------------------------------------------------
5. FIT THE COMPOSITE ONCE, THEN COLOUR
--------------------------------------------------------------------------------

MCFO is a *stochastic* multicolour label: a neuron's colour is a fixed ratio of
the three signal channels **at the same voxels**. Fitting each channel
separately produces splat sets that do not co-locate, and the composite becomes
candy-stripe — adjacent red, green and blue splats along a single axon that
should be one hue. So: fit the per-voxel channel maximum once, then read each
splat's colour from the channels at its own centre, balancing the channels by
their own robust maxima first (unbalanced, the brightest channel wins 96% of
splats and the whole brain reads red).

6. THE PYTHON API'S DEFAULTS ARE NOT THE CLI'S PRESETS — and thin filaments bead
--------------------------------------------------------------------------------

Axons in the first version of this demo rendered as chains of beads. Some of
that is REAL: MCFO axons have varicosities, and walking the filament skeleton
shows the *original data* dipping below half its local ridge on 35.9% of the
skeleton and below a quarter on 11.9%. But the fit doubled the deep dips, to
22.2%.

The cause is that ``fit_gaussian_splats`` defaults to ``n_iters=1000`` — below
the CLI's ``draft`` preset (2000) and a fifth of ``standard`` (5000). Calling
the Python API without a schedule is therefore NOT "the default quality"; it is
below the lowest preset the CLI will give you. Nor is this only a Python-API
trap: ``--preset`` has no default either, so a bare ``luxar gsplat fit`` falls
through to the same 1000 iterations. At 1000 iterations the splats
here never left their seed shape: edge seeding initialises them isotropic at
sigma = 1.0 voxel, and the fitted result measured sigma 0.87-1.16 voxels with a
median axis ratio of 1.30. A 1-voxel-wide axon rebuilt from 1-voxel spheres
spaced 2.5 sigma apart beads by construction.

Six defaults have to move together (``NEURON_FIT_SCHEDULE``). The three that
dominate, measured cumulatively on this sample (skeleton dips <25%, and
foreground PSNR against the raw data):

    shipped originally  1000 iters                    22.2%   22.94 dB
    + 5000 iters                                      16.6%   21.86 dB
    + no relocation     enable_dynamic_ops=False      16.0%   23.54 dB
    + 10000 iters       NEURON_FIT_SCHEDULE           16.5%   25.85 dB
      20000 iters       (not used: +0.21 dB, +46% time) 16.3%  26.06 dB

The two effects decouple, which is why the schedule stops at 10000: BEADING
converges by 5000 (16.6 -> 16.5 -> 16.3 is noise) while FIDELITY keeps climbing
to 10000 and then flattens.

Relocation is why raising ``n_iters`` alone LOSES a dB: it periodically resets
splats to isotropic sigma=0.5 with off-diagonals zeroed, undoing the shapes the
extra iterations just bought. And ``max_eccentricity`` (default 10.0, an axis
ratio of sqrt(10)) is inert at 1000 iterations but binds once converged — 14.9%
of splats pile up against that ceiling, costing 2.65 dB.

What does NOT work, all measured rather than assumed:

  - MORE SPLATS. Doubling seeds to 2.4M made the geometry WORSE (splat spacing
    2.45 -> 3.22 sigma), because finer subdivision shrinks sigma faster than it
    shrinks the gaps. It bought 0.4 dB for 73% more data.
  - FEWER SPLATS. 300k seeds gave the tightest spacing measured (2.03 sigma) and
    the roundest-to-longest shapes (2.75), and beading still got worse — below
    ~1.2M the thin branches lose coverage for a different reason.
  - A WIDER RENDER CUTOFF. Widening truncation from 2.75 to 6.0 sigma moved the
    deep dips by 1.2 points: the splats genuinely do not reach each other.

The residual ~4 points of beading over the data's own is the honest cost of a
localized-Gaussian basis on a filament, and it is now much closer to the
varicosity the specimen actually has.

DATA SOURCE & CITATIONS:
========================

Neurons: FISBe (FlyLight Instance Segmentation Benchmark)
         Zenodo 10875063 -- https://doi.org/10.5281/zenodo.10875063
         Sample VT047848-20171020_66_I3 (``completely`` split, train)
         (C, Z, Y, X) = (3, 390, 1058, 907), uint16, 12-bit valued

Neuropil: Janelia FlyLight Gen1 MCFO, ``janelia-flylight-imagery`` on S3.
         FISBe distributes only the three signal channels; this sample's
         ``channel_spec`` is ``sssr`` -- three **s**ignal plus one
         **r**eference channel. That reference (neuropil counterstain) is what
         makes the image read as neurons *inside a brain*, and it exists only
         in the FlyLight release.

Voxel:   0.44 um isotropic, confirmed two ways -- FISBe paper Sec. 3, and the
         H5J's own root attrs (``voxel_size``, ``unit='micron'``)
Optics:  Zeiss LSM 710/780 confocal, Plan-Apochromat 40x/1.3 Oil DIC M27
Genotype: VT047848 BJD_118E08_AE_01 (female)
License: CC BY 4.0 (both sources)

Where the data comes from (both downloads are automatic):

  FISBe dataset page   https://kainmueller-lab.github.io/fisbe
  FISBe archive        https://doi.org/10.5281/zenodo.10875063
  FlyLight Gen1 MCFO   https://gen1mcfo.janelia.org
  FlyLight imagery     s3://janelia-flylight-imagery  (anonymous HTTP works)
  FlyLight project     https://www.janelia.org/project-team/flylight

How to Cite:
------------
If you use this data, cite all three. The first covers the benchmark and its
annotations; the second the imagery; the third the driver line.

  Mais L, Hirsch P, Managan C, Kandarpa R, Rumberger JL, Reinke A,
  Maier-Hein L, Ihrke G, Kainmueller D. "FISBe: A real-world benchmark
  dataset for instance segmentation of long-range thin filamentous
  structures." CVPR 2024.
  https://arxiv.org/abs/2404.00130

  Meissner GW, et al. "A searchable image resource of Drosophila GAL4
  driver expression patterns with single neuron resolution."
  eLife (2023) 12:e80660.
  https://doi.org/10.7554/eLife.80660

  Tirian L, Dickson BJ. "The VT GAL4, LexA, and split-GAL4 driver line
  collections for targeted expression in the Drosophila nervous system."
  bioRxiv (2017).
  https://doi.org/10.1101/198648

Credit the FlyLight Project Team, Janelia Research Campus, HHMI. Both
sources are CC BY 4.0, which requires attribution and that changes be
indicated — this demo fits splats to the imagery, which is a change.

WORKFLOW:
=========

1. **Fetch** one FISBe sample (~415 MB) from the 7.1 GB Zenodo archive by HTTP
   range request -- the archive is never downloaded whole
2. **Fetch + decode** the reference channel from the FlyLight H5J (~60 MB;
   HEVC streams inside HDF5, needs ffmpeg -- skipped gracefully if absent)
3. **Fit** the MCFO composite (neurons) and the reference channel (neuropil)
4. **Colour** each neuron splat from the three channels at its own centre
5. **Merge** into one node with per-splat RGBA and render volumetrically

USAGE:
======
    python demo_gsplats_3d_flylight_mcfo_neurons.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:      Force re-fetch and re-fit from scratch
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --sample=NAME:    Fit a different FISBe sample from the 'completely' split
    --no-neuropil:    Skip the reference channel (neurons on black)

Output:
    - Scene saved to: datasets/demos/gsplats_3d_flylight_mcfo_neurons.luxar.zarr
    - Automatically opens in your browser on the demo's own derived port
"""

DEMO_META = {
    "key": "gsplats_3d_flylight_mcfo_neurons",
    "title": "3D FlyLight MCFO Neurons",
    "description": "MCFO fly brain neurons in a volume-rendered neuropil, as Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        # ~415 MB range-extracted from the FISBe archive (never fetched whole)
        # plus a ~60 MB H5J for the reference channel.
        "download_mb": 475,
        "compute": "heavy",
        # Redistributable (CC BY 4.0) but not hosted by us yet, so a first run
        # fetches and refits — which needs a GPU.
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_flylight_mcfo"],
    "outputs": ["gsplats_3d_flylight_mcfo_neurons"],
    "citation": {
        "short": "Mais et al. 2024",
        "doi": "10.5281/zenodo.10875063",
        "license": "CC BY 4.0",
    },
}

import hashlib
import importlib.util
import io
import ntpath
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    launch_viewer,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._h5j import decode_h5j_channel, reference_channel_index
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZENODO_RECORD = "10875063"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
ARCHIVE_NAME = "fisbe_v1.0_completely.zip"
SAMPLE_LIST_URL = f"{ZENODO_BASE}/sample_list_per_split.txt?download=1"

DEFAULT_SAMPLE = "VT047848-20171020_66_I3"
# The archive splits the 30 completely-labelled samples across train/val/test
# (18/5/7). Which split a sample sits in is a machine-learning detail with no
# bearing on rendering it, so the demo locates a sample in ANY of them rather
# than pinning one — hardcoding ``completely/train`` silently made 12 of the 30
# unreachable.
SAMPLE_MEMBER_ROOT = "completely"

# The FlyLight H5J carrying the reference (neuropil) channel FISBe drops.
FLYLIGHT_BUCKET = "https://janelia-flylight-imagery.s3.amazonaws.com"
FLYLIGHT_H5J = {
    DEFAULT_SAMPLE: (
        "Gen1 MCFO/VT047848/"
        "VT047848-20171020_66_I3-f-40x-brain-GAL4-unaligned_stack.h5j"
    ),
}

VOXEL_SIZE_ZYX = (0.44, 0.44, 0.44)

# Neurons. SEEDS is the optimiser's pool, CULL_RETENTION decides how much of it
# survives — see finding 3. Do NOT lower SEEDS to shrink the output.
SEEDS = 1_200_000
CULL_RETENTION = 0.9999

# The CLI's own default, and it matters more than any other constant here —
# see finding 2. ``auto`` subtracts a histogram-mode estimate (~0.0029 on this
# sample, removing ~35% of total energy: the pedestal, and little else). A
# percentile floor looks superficially similar and is not: ``p99`` subtracts
# 0.0337 and removes 97.5% of the image's energy BEFORE a single splat is
# placed, taking most of the neurites with it.
FLOOR = "auto"

# How long the neurons are optimised, and what is allowed to happen to their
# SHAPE while they are — see finding 6. Every one of these overrides a
# ``fit_gaussian_splats`` default, and together they are worth ~3 dB of
# foreground PSNR and a third of the beading on thin axons.
#
# ``n_iters`` is the one that matters most, and the trap is that the Python
# API's default is 1000 — BELOW the CLI's own ``draft`` preset (2000) and a
# fifth of ``standard`` (5000). At 1000 the splats never leave their isotropic
# seed shape, so a 1-voxel-wide axon is rebuilt as a chain of 1-voxel spheres
# and reads as beaded. The other five exist because raising ``n_iters`` alone is
# NOT enough: the two patience defaults decay the shape LR and stop the fit long
# before 10,000 iterations are ever reached, relocation resets shapes back to
# isotropic, and the eccentricity cap and L1 diagonal penalty then clip and pull
# back what convergence finally earned.
NEURON_FIT_SCHEDULE = {
    "n_iters": 10_000,
    "patience": 200,  # plateau LR decay, vs 15 — 15 decays away the shape LR
    "early_stop_patience": 2000,  # vs 300, which stops before shapes settle
    "enable_dynamic_ops": False,  # relocation re-isotropises splats mid-fit
    "max_eccentricity": None,  # default 10.0 caps the axis ratio at sqrt(10)
    "l1_diag": 0.0,  # default penalty "encourages ... more isotropic splats"
}

# Neuropil. Fewer splats than the neurons need, because it is a smooth medium.
# Deliberately left on the stock schedule: it is a diffuse counterstain with no
# filaments to bead, the chosen "solid brain" look depends on its current
# character, and a converged fit of 600k seeds would cost more time than the
# neurons' own.
NEUROPIL_SEEDS = 600_000
NEUROPIL_ALPHA = 0.12  # optical depth per splat — see finding 4
NEUROPIL_AMP = 0.6
NEUROPIL_RGB = (0.15, 0.25, 1.0)

# Background suppression happens HERE, at render time, not in the fit: a splat's
# alpha ramps from NEURON_ALPHA_MIN to 1 between these two percentiles of splat
# AMPLITUDE. (Percentiles of amplitude — not of voxel intensity like FLOOR. Two
# different quantities; keeping the names distinct is deliberate.)
NEURON_ALPHA_LO_AMP_PCT = 90.0
NEURON_ALPHA_HI_AMP_PCT = 99.5
# The ramp never reaches exactly 0, and that is the whole difference from a
# floor. The shader folds alpha into a splat's contribution and then discards
# anything negligible, so an alpha of exactly 0 makes a splat emit and absorb
# nothing NO MATTER what the display range does afterwards — as irreversible as
# deleting it. 0.01 is optical depth −ln(1 − a) ≈ 0.01: ~600x thinner than an
# opaque neurite and ~12x thinner than a neuropil splat, so the haze is
# invisible at the default display range and comes back when the range is
# NARROWED. (Narrowed, not widened: the viewer's gain is 1/(max - min), so a
# tighter window is a brighter one — see `display-range.ts::computeUniforms`.)
NEURON_ALPHA_MIN = 0.01

COLOR_BALANCE_PERCENTILE = 99.99

# Initial framing. <1 starts closer than a just-fits view of the bounding
# sphere; the specimen tilt is MEASURED per sample, never hard-coded.
CAMERA_FOV_DEG = CINEMATIC_FOV_DEG
CAMERA_FRAMING = 0.60

CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_flylight_mcfo"

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
NO_NEUROPIL = "--no-neuropil" in sys.argv

# FISBe sample names are plain basenames (``VT047848-20171020_66_I3``,
# ``JRC_SS04989-20160318_24_B1``). Pinning that shape keeps --sample out of the
# path-traversal business: the value lands in cache paths, the archive member
# prefix and the output filename, so a ``../`` or an absolute path would let it
# read and write outside the demo's directories — and --serve-only would then
# happily serve whatever .luxar.zarr it pointed at.
_SAMPLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_sample_name(sample: str) -> str:
    """Return ``sample`` if it is a bare FISBe sample name, else raise."""
    if not _SAMPLE_RE.match(sample):
        raise ValueError(
            f"Invalid --sample {sample!r}: expected a bare FISBe sample name "
            "matching [A-Za-z0-9][A-Za-z0-9_-]* (no path separators, no '..'). "
            f"See {SAMPLE_LIST_URL} for valid names."
        )
    return sample


SAMPLE = DEFAULT_SAMPLE
for _arg in sys.argv:
    if _arg.startswith("--sample="):
        SAMPLE = validate_sample_name(_arg.split("=", 1)[1])

Arbol.max_depth = 6
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = None


# =============================================================================
# Remote zip access (HTTP range requests)
# =============================================================================


_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/")


def _served_range_span(content_range: str, start: int, end: int, url: str) -> int:
    """Length of the range a 206 says it carries, refusing anything else.

    A 206 is only usable here if it answers the interval that was asked for:
    the block is cached at ``start`` regardless of what arrived, so bytes from
    some other offset would be assembled into the archive at the wrong place.
    A range ENDING short of ``end`` is accepted — :meth:`_HttpRangeFile.read`
    simply fetches the remainder — but one starting elsewhere, or running past
    ``end``, is refused.
    """
    m = _CONTENT_RANGE_RE.match(content_range.strip())
    if m is None or int(m.group(1)) != start or not start <= int(m.group(2)) <= end:
        raise RuntimeError(
            f"{url} answered a byte-range request for bytes {start}-{end} with "
            f"Content-Range {content_range!r}. The bytes would be cached at the "
            "wrong offset and the archive assembled out of the wrong data."
        )
    return int(m.group(2)) - start + 1


def _read_at_most(resp, span: int, url: str) -> bytes:
    """Read a streamed body, refusing one longer than ``span`` bytes.

    ``resp.content`` would buffer whatever the host chose to send, which for a
    host that answers a range with the whole 7.1 GB archive is exactly the
    download this class exists to avoid. Reading in chunks bounds it.
    """
    body = bytearray()
    for part in resp.iter_content(chunk_size=1 << 16):
        body += part
        if len(body) > span:
            raise RuntimeError(
                f"{url} sent more than the {span} bytes its Content-Range "
                "advertised, so it is not honouring range requests as claimed; "
                "refusing before the whole archive is buffered."
            )
    return bytes(body)


class _HttpRangeFile(io.RawIOBase):
    """A minimal seekable read-only file over an HTTP resource.

    ``zipfile.ZipFile`` only needs ``seek``/``read``/``tell``, and it seeks to
    the end of the file to find the central directory. Serving those seeks with
    HTTP range requests means one member can be extracted from a multi-gigabyte
    remote archive while transferring only that member plus the directory.
    """

    def __init__(self, url: str, size: int, session, chunk_size: int = 1 << 20):
        """Wrap ``url`` as a seekable file of ``size`` bytes served by ranges."""
        self._url = url
        self._size = size
        self._session = session
        self._pos = 0
        self._chunk_size = chunk_size
        # One-block read-ahead: zipfile does many small adjacent reads while
        # parsing the central directory, and without this each becomes its own
        # HTTP round trip.
        self._cache_start = 0
        self._cache = b""

    def readable(self) -> bool:
        """This file is read-only, and readable."""
        return True

    def close(self) -> None:
        """Close the file *and* the HTTP session that served its ranges.

        ``zipfile.ZipFile`` never closes a file object it was handed, so
        nothing else in the chain releases the session — its pooled
        connection would stay open for the rest of the run (a fit that takes
        minutes) and only be reclaimed whenever the collector got round to it.
        """
        session, self._session = self._session, None
        try:
            if session is not None:
                session.close()
        finally:
            super().close()

    def seekable(self) -> bool:
        """Seeking is what makes remote zip access possible."""
        return True

    def tell(self) -> int:
        """Return the current byte offset."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read position, clamping to the bounds of the resource."""
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        """Read ``size`` bytes (or to EOF), fetching blocks as needed."""
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size <= 0:
            return b""

        out = bytearray()
        while size > 0:
            block = self._block_for(self._pos)
            offset = self._pos - self._cache_start
            take = min(size, len(block) - offset)
            if take <= 0:  # pragma: no cover - defensive
                break
            out += block[offset : offset + take]
            self._pos += take
            size -= take
        return bytes(out)

    def readinto(self, b) -> int:
        """Read into a pre-allocated buffer (the ``RawIOBase`` contract)."""
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def _block_for(self, pos: int) -> bytes:
        """Return the cached block covering ``pos``, range-fetching it if needed."""
        if self._cache and self._cache_start <= pos < self._cache_start + len(
            self._cache
        ):
            return self._cache
        start = pos
        end = min(start + self._chunk_size, self._size) - 1
        # ``stream=True`` so the status is checked BEFORE the body is read. A
        # host that ignores the Range header answers 200 with the WHOLE 7.1 GB
        # archive, and a non-streaming ``get`` buffers all of it in memory
        # before this function can refuse it — the refusal has to be cheap.
        resp = self._session.get(
            self._url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=60,
            stream=True,
        )
        try:
            if resp.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206 (partial content) from {self._url}, got "
                    f"{resp.status_code}. The host has stopped honouring range "
                    "requests, so a single sample can no longer be extracted "
                    "without downloading the whole archive."
                )
            # A 206 alone is not enough: the block is filed at ``start``
            # whatever the host actually sent, so a response covering a
            # DIFFERENT interval would have the archive assembled out of the
            # wrong bytes — an offset error surfacing much later as a corrupt
            # zip. Read the interval back out of ``Content-Range`` and bound
            # the body to it, because a 206 carrying more than it advertises is
            # the same multi-gigabyte buffer the status check above refuses.
            span = _served_range_span(
                resp.headers.get("Content-Range", ""), start, end, self._url
            )
            self._cache_start = start
            self._cache = _read_at_most(resp, span, self._url)
        finally:
            resp.close()
        return self._cache


def _open_remote_zip(url: str):
    """Open a remote zip for partial extraction. Returns ``(ZipFile, handle)``.

    The caller owns both: closing the ``ZipFile`` is not enough, because it was
    handed a file object and so leaves it open. Close the handle too (see
    :meth:`_HttpRangeFile.close`).
    """
    import zipfile

    session = requests.Session()
    # Any failure below — a refused range, a body that is not a zip — has to
    # take the session with it, or a fetch that never got off the ground
    # strands an open connection for the rest of the run.
    try:
        # Probe with a one-byte ranged GET rather than trusting HEAD: Zenodo's
        # HEAD answers 200 with no ``Accept-Ranges`` header even though ranged
        # GETs are honoured, so a HEAD-based check would refuse a host that
        # works fine. The 206 also carries the total size in ``Content-Range``,
        # so this is one request instead of two.
        # ``stream=True`` because the whole point of the probe is to find out
        # whether the body is one byte or 7.1 GB: a host that ignores the Range
        # header answers 200 with the entire archive, and a non-streaming
        # ``get`` downloads all of it into memory before the check below can
        # refuse it. Headers (status, Content-Range, resolved URL) are
        # available without touching the body, so nothing is transferred on
        # the refusal path.
        probe = session.get(
            url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=True,
            timeout=60,
            stream=True,
        )
        try:
            probe.raise_for_status()
            content_range = probe.headers.get("Content-Range", "")
            # A 206 may legally report an unknown total (``bytes 0-0/*``),
            # which is no more usable here than a refused range: without the
            # size there is nothing to seek against. Insist on a digit total so
            # an odd host produces the explanation below instead of a bare
            # ValueError from int().
            total = content_range.rsplit("/", 1)[-1].strip()
            if probe.status_code != 206 or not total.isdigit():
                raise RuntimeError(
                    f"{url} did not honour a byte-range request (status "
                    f"{probe.status_code}), so extracting one sample would "
                    "require downloading the full 7.1 GB archive. Download it "
                    f"manually and extract the sample into {CACHE_DIR} as "
                    "<sample>.zarr instead."
                )
            probe_url = probe.url
        finally:
            probe.close()

        handle = _HttpRangeFile(probe_url, int(total), session)
        return zipfile.ZipFile(io.BufferedReader(handle, buffer_size=1 << 20)), handle
    except BaseException:
        session.close()
        raise


def _safe_extract_path(root: Path, rel: str, member: str) -> Path:
    """Resolve ``rel`` under ``root``, refusing anything that escapes it.

    Archive member names are attacker-controlled in the general case, and a
    member spelled ``../../…`` (or with an absolute path) would otherwise be
    written outside the cache directory — the classic zip-slip. Refuse rather
    than sanitise, so a malformed archive is loud instead of silently partial.
    """
    if PurePosixPath(rel).is_absolute() or ntpath.isabs(rel):
        raise RuntimeError(f"Refusing absolute path in archive member: {member!r}")
    # ZIP names are specified to use forward slashes, so a backslash is both
    # non-conformant and a traversal on Windows (``..\..\x`` is one harmless
    # filename on POSIX and an escape there). Refuse it everywhere rather than
    # let the guard's behaviour depend on the host OS.
    if "\\" in rel:
        raise RuntimeError(f"Refusing backslash in archive member: {member!r}")
    dest = (root / rel).resolve()
    if dest != root and root not in dest.parents:
        raise RuntimeError(
            f"Refusing archive member that escapes the extraction root: {member!r}"
        )
    return dest


def _extract_members(zf, members, member_prefix: str, tmp: Path) -> None:
    """Write every archive member below ``member_prefix`` into ``tmp``."""
    root = tmp.resolve()
    for i, name in enumerate(members):
        rel = name[len(member_prefix) :]
        if not rel:
            continue
        dest = _safe_extract_path(root, rel, name)
        if name.endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Copy in chunks rather than materialising the member: a zarr chunk is
        # usually small, but nothing here bounds it, and the whole point of the
        # range extraction is to stay well under the archive's size in memory.
        with zf.open(name) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        if i % 200 == 0:
            aprint(f"  {i}/{len(members)} members")


def _install_store(tmp: Path, target: Path) -> None:
    """Move a freshly extracted store into place, preserving any existing one.

    Renaming onto a populated directory raises ENOTEMPTY, so an existing store
    is moved aside first — and restored if installing the new one fails.
    Deleting it unconditionally would lose BOTH copies on failure, leaving no
    usable sample at all.
    """
    if not target.exists():
        tmp.rename(target)
        return

    stale = target.with_suffix(".zarr.stale")
    if stale.exists():
        shutil.rmtree(stale)
    target.rename(stale)
    try:
        tmp.rename(target)
    except BaseException:
        if not target.exists():
            stale.rename(target)
        raise
    shutil.rmtree(stale, ignore_errors=True)


def find_member_prefix(names, sample: str) -> str | None:
    """Locate ``<sample>.zarr/`` under any split of the archive.

    Returns the member prefix (including the trailing slash), or None if the
    sample is not in this archive.
    """
    pattern = re.compile(
        rf"^{re.escape(SAMPLE_MEMBER_ROOT)}/[^/]+/{re.escape(sample)}\.zarr/"
    )
    for name in names:
        m = pattern.match(name)
        if m:
            return m.group(0)
    return None


def fetch_sample(sample: str) -> Path:
    """Range-extract one FISBe sample's zarr store into the demo cache."""
    target = CACHE_DIR / f"{sample}.zarr"
    # The store is renamed into place only once every member is written, so a
    # present target directory always means a complete extraction.
    if target.exists() and not RECOMPUTE:
        aprint(f"Sample already extracted: {target.name}")
        return target

    url = f"{ZENODO_BASE}/{ARCHIVE_NAME}?download=1"

    with asection(f"Range-extracting {sample} from {ARCHIVE_NAME}"):
        aprint("The 7.1 GB archive is NOT downloaded whole — only this sample.")
        zf, handle = _open_remote_zip(url)
        try:
            names = zf.namelist()
            member_prefix = find_member_prefix(names, sample)
            if member_prefix is None:
                raise RuntimeError(
                    f"Sample {sample!r} not found in {ARCHIVE_NAME} (searched "
                    f"every split under {SAMPLE_MEMBER_ROOT}/). See "
                    f"{SAMPLE_LIST_URL} for valid names — note this archive "
                    "holds the 'completely' labelled samples only."
                )
            members = [n for n in names if n.startswith(member_prefix)]
            payload = sum(zf.getinfo(n).compress_size for n in members)
            aprint(f"{len(members)} members, {payload / 1e6:.0f} MB compressed")

            # A previous interrupted run can leave a stale partial directory,
            # and --recompute reaches here with `target` already populated.
            tmp = target.with_suffix(".zarr.partial")
            if tmp.exists():
                shutil.rmtree(tmp)

            _extract_members(zf, members, member_prefix, tmp)
            _install_store(tmp, target)
        finally:
            # Both, in this order: the ZipFile leaves the file object it was
            # handed open, so the handle is what releases the HTTP session.
            zf.close()
            handle.close()

    aprint(f"Extracted to {target}")
    return target


# =============================================================================
# Reference (neuropil) channel — FlyLight H5J
# =============================================================================


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temporary sibling + rename.

    The sibling is removed if anything goes wrong: a disk-full or interrupted
    write would otherwise strand a large ``.part`` file that nothing ever
    cleans up, occupying exactly the space the retry needs.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        tmp.write_bytes(payload)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """``np.save`` to a temporary sibling, then rename into place.

    Saving through an open handle rather than a path: ``np.save`` appends
    ``.npy`` to any *name* that does not already end in it, so a temporary
    called ``vol.npy.part`` would silently be written as ``vol.npy.part.npy``
    and the rename would then fail on a missing file.

    As in :func:`_atomic_write`, a failed save takes its sibling with it — the
    partial here is a whole decoded volume, hundreds of megabytes of it.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, array)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def neuropil_grid_mismatch(ref_shape, neuron_shape) -> str | None:
    """Return a warning if the two sources are not on the same voxel grid.

    The two clouds are fitted independently, in physical units anchored at the
    volume origin, so the merge only registers if the reference channel has the
    same (Z, Y, X) shape as the MCFO composite. FISBe derives from the same
    FlyLight stack and the default sample matches, but a newly mapped sample
    that does not would place the brain off its own neurons — an error easy to
    read as a fitting artefact rather than as a mismatch of inputs.
    """
    if tuple(ref_shape) == tuple(neuron_shape):
        return None
    return (
        f"⚠️  Reference channel is {tuple(ref_shape)} but the MCFO composite is "
        f"{tuple(neuron_shape)}: the two are on different voxel grids, so the "
        "neuropil cannot be registered against the neurons — skipping it. "
        "Check this sample's FLYLIGHT_H5J mapping."
    )


def neuropil_reference(sample: str, neuron_shape):
    """Return a reference channel that registers against the neurons, or None.

    The registration check has to *decide*, not just narrate: a mismatch means
    the merged scene would be wrong, and rendering it anyway spends a
    multi-minute fit to publish a brain sitting off its own neurons. So a
    mismatch degrades exactly like a missing ffmpeg does — neurons only, with
    the mapping to fix named — rather than warning and carrying on.
    """
    ref = fetch_neuropil(sample)
    if ref is None:
        return None
    mismatch = neuropil_grid_mismatch(ref.shape, neuron_shape)
    if mismatch:
        aprint(mismatch)
        return None
    return ref


def fetch_neuropil(sample: str):
    """Fetch and decode the reference channel, or return None if unavailable.

    Missing ffmpeg (or an unmapped sample) degrades to a neurons-only scene
    rather than failing: the demo is still worth running without the neuropil,
    it just loses the volume-rendered context.
    """
    key = FLYLIGHT_H5J.get(sample)
    if key is None:
        aprint(f"No FlyLight H5J mapped for {sample}; skipping the neuropil.")
        return None
    # Check the decoded cache FIRST: loading it needs neither ffmpeg nor h5py,
    # so gating on those beforehand would throw away a perfectly good warm
    # cache and silently drop to a neurons-only scene.
    cached = CACHE_DIR / f"{sample}_neuropil.npy"
    if cached.exists() and not RECOMPUTE:
        aprint(f"Neuropil already decoded: {cached.name}")
        return np.load(cached)

    # Both remaining dependencies are optional, and BOTH must degrade the same
    # way: the documented fallback is a neurons-only scene, so a missing one
    # must not raise out of a demo that promises to keep going.
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (H5J stores the channel as HEVC)")
    if importlib.util.find_spec("h5py") is None:
        missing.append("h5py (pip install 'luxar[demos]')")
    if missing:
        aprint(
            f"Skipping the neuropil channel — missing {', '.join(missing)}. "
            "The neurons render fine without it; install the above for the "
            "volume-rendered brain."
        )
        return None

    h5j = CACHE_DIR / f"{sample}.h5j"
    if not h5j.exists() or RECOMPUTE:
        url = f"{FLYLIGHT_BUCKET}/{requests.utils.quote(key)}"
        with asection("Fetching FlyLight H5J (reference channel)"):
            aprint(url)
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            # Publish atomically: an interrupted write would otherwise leave a
            # truncated file that every later run trusts and fails to decode,
            # recoverable only by knowing to pass --recompute.
            _atomic_write(h5j, resp.content)
            aprint(f"  {len(resp.content) / 1e6:.0f} MB")

    # The reference channel's position comes from the file, not a constant.
    vol = decode_h5j_channel(h5j, reference_channel_index(h5j))
    _atomic_save_npy(cached, vol)
    return vol


# =============================================================================
# Data Loading
# =============================================================================


def load_fisbe_sample(sample: str):
    """Load the three MCFO channels, their per-voxel maximum, and the source dtype.

    The channels are widened to float32 to be fitted, and after that cast the
    ACQUISITION's element size is unrecoverable. So it is carried out
    separately: everything downstream that says what was compressed — the fit's
    own recorded provenance, the summary's ratio — otherwise measures the
    float32 working copy and flatters itself by the cast's 2x.
    """
    store_path = fetch_sample(sample)

    with asection(f"Loading {sample}"):
        raw = zarr.open(str(store_path), mode="r")["volumes"]["raw"]
        aprint(f"raw: shape={raw.shape} dtype={raw.dtype}")
        source_dtype = str(raw.dtype)
        if raw.shape[0] != 3:
            raise RuntimeError(f"Expected 3 MCFO channels, got {raw.shape[0]}")

        channels = [np.asarray(raw[c]).astype(np.float32) for c in range(3)]
        shared_max = max(float(v.max()) for v in channels) or 1.0
        # Scale IN PLACE. Each channel is 1.5 GB here, and
        # ``[v / shared_max for v in channels]`` would hold the old and the new
        # list simultaneously — 4.5 GB of avoidable peak on a demo people are
        # expected to run on a laptop.
        for v in channels:
            v /= shared_max
        for c, v in enumerate(channels):
            aprint(f"  ch{c}: mean={v.mean():.6f} p99.9={np.percentile(v, 99.9):.5f}")

        # Likewise accumulate the composite into one buffer rather than letting
        # the nested np.maximum allocate an intermediate.
        combined = channels[0].copy()
        np.maximum(combined, channels[1], out=combined)
        np.maximum(combined, channels[2], out=combined)
        extent = tuple(round(n * s, 1) for n, s in zip(combined.shape, VOXEL_SIZE_ZYX))
        aprint(f"  composite mean={combined.mean():.6f}")
        aprint(f"  physical extent (Z, Y, X): {extent} um")
        return channels, combined, source_dtype


# =============================================================================
# Fitting and colouring
# =============================================================================


def fit_cache_key(
    seeds: int, floor, retention: float, schedule: dict | None = None
) -> str:
    """Short digest of everything that changes a fit's result.

    Cache filenames keyed only by sample would silently reuse an incompatible
    fit whenever one of the tuning constants above is edited — and those
    constants are exactly what this demo invites you to tune. Folding them into
    the name means changing one produces a different file, so the refit is
    automatic rather than dependent on remembering ``--recompute``.

    ``schedule`` is in here for the same reason and is easy to forget, because
    unlike seeds/floor/retention it does not change the splat COUNT — only the
    shapes. A key blind to it would hand back the old under-converged, beaded
    fit and make an edit to :data:`NEURON_FIT_SCHEDULE` look like a no-op.
    """
    shape = "|".join(f"{k}={schedule[k]}" for k in sorted(schedule or {}))
    payload = f"v2|{seeds}|{floor}|{retention}|{tuple(VOXEL_SIZE_ZYX)}|{shape}"
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _fit_cache_path(
    component: str, seeds: int, floor, retention: float, schedule: dict | None = None
) -> Path:
    """Cache path for one fitted component, keyed by its fit parameters."""
    key = fit_cache_key(seeds, floor, retention, schedule)
    return CACHE_DIR / f"{SAMPLE}_{component}_{key}.gsplats.zarr.zip"


def _device():
    """Resolve (and memoise) the fitting device."""
    global DEVICE
    if DEVICE is None:
        from luxar.demos import detect_device

        DEVICE = detect_device()
    return DEVICE


def fit_volume(
    volume,
    cache_file: Path,
    seeds: int,
    floor,
    retention: float,
    label: str,
    schedule: dict | None = None,
    source_dtype: str | None = None,
):
    """Fit one volume to splats, using the cache when present.

    ``schedule`` carries the optimiser overrides (iterations, and what may
    happen to splat shapes); ``None`` means the library defaults, which is what
    the neuropil wants and what the neurons emphatically do not.

    ``source_dtype`` is what the volume was ACQUIRED as. Both components reach
    here already widened to float32 — the MCFO channels from uint16, the
    neuropil from the decoder's uint8 — so without it the recorded provenance
    would quote compression against the working copy rather than the data.
    """
    if cache_file.exists() and not RECOMPUTE:
        aprint(f"Loading cached {label} fit")
        try:
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"  Loaded {len(result.amplitudes):,} cached splats")
            return result
        except Exception as exc:
            aprint(f"  Cache load failed: {exc}, re-fitting...")

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {label} (seeds={seeds:,}, floor={floor}, retention={retention})")
    if schedule:
        aprint(f"  schedule: {schedule}")
    result = fit_gaussian_splats(
        volume,
        seeds=seeds,
        floor=floor,
        cull_retention=retention,
        device=_device(),
        verbose=True,
        voxel_size=VOXEL_SIZE_ZYX,
        source_dtype=source_dtype,
        **(schedule or {}),
    )
    aprint(f"  Fitted {len(result.amplitudes):,} splats (from {seeds:,} seeds)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_with_lod(
        result,
        cache_file,
        recipe="stream",
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    return GSplatData.load(cache_file, include_stats=False)


def splat_colors(centers, channels, voxel_size=VOXEL_SIZE_ZYX):
    """Read each splat's MCFO colour out of the three channel volumes.

    Args:
        centers: (N, 3) splat centres in PHYSICAL units (Z, Y, X micrometres).
        channels: three (Z, Y, X) float32 volumes.
        voxel_size: (Z, Y, X) micrometres per voxel.

    Returns:
        (N, 3) float32 RGB in [0, 1].
    """
    shape = channels[0].shape
    # ``fit_gaussian_splats(..., voxel_size=...)`` returns centres in physical
    # units, so they must be divided back out before they can index the volume.
    # Indexing microns as voxels silently samples the wrong places (compressed
    # toward the origin by 1/voxel_size) and quietly corrupts every colour.
    idx = np.rint(centers / np.asarray(voxel_size, dtype=np.float32)).astype(np.int64)
    for d in range(3):
        np.clip(idx[:, d], 0, shape[d] - 1, out=idx[:, d])

    triplet = np.empty((len(centers), 3), dtype=np.float32)
    for c, vol in enumerate(channels):
        # Balance channels before compositing — see finding 5.
        gain = float(np.percentile(vol, COLOR_BALANCE_PERCENTILE)) or 1.0
        triplet[:, c] = vol[idx[:, 0], idx[:, 1], idx[:, 2]] / gain
        aprint(f"  ch{c} balance gain (p{COLOR_BALANCE_PERCENTILE}) = {gain:.5f}")

    # Hue only: brightness is carried by the splat amplitude, so normalising
    # each splat by its own strongest channel keeps the MCFO colour ratio
    # without double-counting intensity.
    peak = triplet.max(axis=1, keepdims=True)
    rgb = np.clip(
        np.divide(triplet, peak, out=np.zeros_like(triplet), where=peak > 0), 0.0, 1.0
    )
    if len(triplet):
        dominant = triplet.argmax(axis=1)
        aprint(
            "  dominant channel: "
            + ", ".join(f"ch{c}={100 * (dominant == c).mean():.1f}%" for c in range(3))
        )
    return rgb


def neuron_alpha(amplitudes):
    """Map splat amplitude to per-splat alpha (optical depth).

    This is where background suppression belongs. Doing it with a FLOOR instead
    deletes the faint signal before fitting, and no splat count recovers it —
    measured, a p99 floor lost 88% of the neuron's faintest fifth while ``auto``
    loses none. Here the haze is merely made optically thin: the splats are
    still in the scene, so the display range can bring them back.

    Which is why the floor of the ramp is ``NEURON_ALPHA_MIN`` and not zero. A
    zero-alpha splat neither emits nor absorbs and the shader discards it, so
    clipping the bottom decile to 0 would be exactly the irreversible deletion
    this function exists to avoid — one shader stage later than the floor does
    it, and no more recoverable.
    """
    amp = np.asarray(amplitudes, dtype=np.float32)
    if amp.size == 0:
        return amp.reshape(0)
    lo = float(np.percentile(amp, NEURON_ALPHA_LO_AMP_PCT))
    hi = float(np.percentile(amp, NEURON_ALPHA_HI_AMP_PCT))
    ramp = np.clip((amp - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    a = (NEURON_ALPHA_MIN + (1.0 - NEURON_ALPHA_MIN) * ramp).astype(np.float32)
    aprint(
        f"  alpha ramp: amp p{NEURON_ALPHA_LO_AMP_PCT}={lo:.5f} -> "
        f"p{NEURON_ALPHA_HI_AMP_PCT}={hi:.5f}; {100 * (a < 0.05).mean():.1f}% "
        "of splats below alpha 0.05"
    )
    return a


def measure_tilt_deg(centers, amplitudes):
    """Angle of the specimen's long axis, in degrees, in the display plane.

    FISBe ships the UNALIGNED FlyLight stack — the brain as mounted, which for
    this sample sits at ~52 degrees. The angle is per-specimen, so it is
    measured rather than hard-coded; another ``--sample`` will differ.

    Centres arrive as (Z, Y, X); the display plane is the (Y, X) one.
    """
    c = np.asarray(centers, dtype=np.float64)
    w = np.asarray(amplitudes, dtype=np.float64)
    total = w.sum()
    if len(c) < 2 or total <= 0:
        return 0.0
    w = w / total
    d = c - (c * w[:, None]).sum(0)
    cov = (d * w[:, None]).T @ d
    ev, evec = np.linalg.eigh(cov[np.ix_([1, 2], [1, 2])])
    major = evec[:, int(np.argmax(ev))]
    return float(np.degrees(np.arctan2(major[0], major[1])))


def merge_for_render(neurons, neuron_rgb, neuropil):
    """Merge neurons and neuropil into one splat set with per-splat RGBA.

    Returns ``(centers, amplitudes, cholesky, rgba)``. The neuropil is emitted
    first purely for readability; ordering within a node is resolved by the
    renderer's depth sort, which is the whole reason for using one node.
    """
    alpha = neuron_alpha(neurons.amplitudes)
    neuron_rgba = np.concatenate([neuron_rgb, alpha[:, None]], axis=1).astype(
        np.float32
    )
    if neuropil is None:
        return (
            neurons.centers,
            neurons.amplitudes,
            neurons.cholesky_factors,
            neuron_rgba,
        )

    p = len(neuropil.amplitudes)
    neuropil_rgba = np.empty((p, 4), dtype=np.float32)
    neuropil_rgba[:, :3] = np.asarray(NEUROPIL_RGB, dtype=np.float32)
    neuropil_rgba[:, 3] = NEUROPIL_ALPHA

    return (
        np.concatenate([neuropil.centers, neurons.centers]).astype(np.float32),
        np.concatenate([neuropil.amplitudes * NEUROPIL_AMP, neurons.amplitudes]).astype(
            np.float32
        ),
        np.concatenate([neuropil.cholesky_factors, neurons.cholesky_factors]).astype(
            np.float32
        ),
        np.concatenate([neuropil_rgba, neuron_rgba]).astype(np.float32),
    )


# =============================================================================
# Scene Creation
# =============================================================================


def scene_description(has_neuropil: bool) -> str:
    """The scene's own description of itself — neuropil or neurons only.

    Four ordinary conditions produce a neurons-only scene (no ffmpeg, no h5py,
    an unmapped sample, a reference channel on a different voxel grid) and
    ``--no-neuropil`` asks for one outright, so the text cannot assume the
    counterstain is there: a scene that describes a volume-rendered brain it
    never fitted sends the reader looking for the demo's whole point.
    """
    if has_neuropil:
        heading = "MCFO Fly Brain Neurons in a Volume-Rendered Neuropil"
        medium = """The neuropil counterstain is a dense semi-transparent medium; under volumetric
(emission-absorption) compositing it occludes front-to-back, so the brain reads
as a solid body and neurites genuinely pass behind it. Neurons and neuropil
share ONE node with per-splat RGBA — as two nodes covering the same volume
there would be no correct draw order."""
        source = (
            "  - Neuropil: Janelia FlyLight Gen1 MCFO reference channel "
            "(CC BY 4.0),\n    which FISBe does not distribute"
        )
    else:
        heading = "MCFO Fly Brain Neurons (neurons only, no neuropil)"
        medium = """There is no neuropil in this scene. FISBe distributes only the signal channels,
and the FlyLight reference channel was unavailable on this run — missing ffmpeg
or h5py, an unmapped sample, a reference channel on a different voxel grid, or
--no-neuropil. The neurons still composite volumetrically; they simply have no
brain around them."""
        source = "  - Neuropil: not included in this scene"

    return f"""
{heading}
{"=" * len(heading)}

Labelled Drosophila neurons — long, thin, widely branching — threaded through
the brain they live in, as one volume-rendered Gaussian splat cloud.

{medium}

Data Source:
  - Neurons: FISBe v1.0, Zenodo 10.5281/zenodo.10875063 (CC BY 4.0)
    Sample {SAMPLE}, 'completely' split
{source}
  - Zeiss LSM 710/780 confocal, 40x/1.3 Oil, 0.44 um isotropic

Cite: Mais et al. (FISBe, CVPR 2024); Meissner et al. (eLife 2023
12:e80660); Tirian & Dickson (2017). Credit the FlyLight Project Team,
Janelia Research Campus, HHMI.

Controls:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
"""


def create_luxar_scene(
    centers, amplitudes, cholesky, rgba, output_path=None, has_neuropil: bool = True
):
    """Create the 3D scene: one volume-rendered node with per-splat RGBA.

    ``has_neuropil`` is what the scene DESCRIBES itself as, so it has to be the
    truth: the neuropil is absent whenever ffmpeg or h5py is missing, the sample
    is unmapped, its grid does not register, or ``--no-neuropil`` was passed, and
    a scene that still claims a volume-rendered counterstain in that case sends
    the reader looking for a brain that was never fitted.
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_flylight_mcfo_neurons.luxar.zarr"
        )

    with asection("Creating 3D Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # Roll the CAMERA to compensate for how the specimen was mounted,
            # rather than rotating the data. Rotating splats would desynchronise
            # them from FISBe's annotations (which live in this unaligned space)
            # and would mean rotating every covariance too — easy to get subtly
            # wrong. The camera costs nothing and the user can undo it.
            centroid = (centers.T @ amplitudes) / max(float(amplitudes.sum()), 1e-9)
            tilt = np.radians(measure_tilt_deg(centers, amplitudes))
            radius = float(np.linalg.norm(centers - centroid, axis=1).max())
            dist = radius / np.tan(np.radians(CAMERA_FOV_DEG / 2)) * CAMERA_FRAMING
            aprint(
                f"Camera: tilt {np.degrees(tilt):.1f} deg, radius {radius:.0f}, "
                f"distance {dist:.0f}"
            )

            scene = compiler.create_scene(
                dimensions=dims,
                # A film look suits this scene: it IS a microscope image, so a
                # little grain and a soft glow read as photographic rather than
                # as decoration, and the bloom gives the bright neurites the
                # halo they have in the raw data.
                #
                # ``cinematic_mode=True`` now carries that look on its own — the
                # zarr bridge expands the preset into every field a scene leaves
                # unset (#1591), which it did NOT when this demo was written.
                # The effects below are spelled out anyway, and they hold the
                # preset's own values, so they are redundant rather than wrong:
                # they keep this scene's grain and glow pinned at what it was
                # tuned against if the shared preset is ever re-graded.
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    cinematic_mode=True,
                    bloom_enabled=True,
                    bloom_threshold=0.01,
                    bloom_strength=0.05,
                    bloom_radius=1.0,
                    bloom_levels=8,
                    vignette_enabled=True,
                    detector_noise_enabled=True,
                    detector_noise_readout_sigma=0.002,
                    detector_noise_photon_gain=0.002,
                    detector_noise_fpn_sigma=0.001,
                    camera=CameraConfig(
                        position=(0.0, 0.0, dist),
                        target=(0.0, 0.0, 0.0),
                        # PERPENDICULAR to the specimen's long axis, so that
                        # axis lands horizontal: (-sin, cos), not (sin, cos).
                        # The un-negated version rolls the wrong way and still
                        # looks plausible — diagonal, just mirrored.
                        up=(float(-np.sin(tilt)), float(np.cos(tilt)), 0.0),
                    ),
                ),
                citation=DEMO_META["citation"],
            )

            scene.attrs["title"] = "GSplats: MCFO Fly Brain Neurons"
            scene.attrs["sample"] = SAMPLE
            scene.attrs["description"] = scene_description(has_neuropil)

            aprint(f"Centroid: {centroid}")

            scene.add_gsplats(
                name="gsplats_mcfo",
                centers=centers - centroid,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=rgba,
                dim_order=["z", "y", "x"],
                opacity=1.0,
                absorption=1.0,
                blending_mode="volumetric",
                layer=True,
            )
            aprint(f"Added {len(amplitudes):,} splats with per-splat RGBA")

            scene.add_text(
                "MCFO Fly Brain Neurons",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "Confocal • 0.44 μm isotropic • FISBe / FlyLight",
                DEMO_META.get("citation"),
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: MCFO Fly Brain Neurons (FlyLight / FISBe)")
    aprint("=" * 70)
    aprint(f"Sample: {SAMPLE}")
    aprint("Thin filaments + a volume-rendered neuropil, in one splat cloud")
    aprint("")

    # A non-default --sample writes beside the default scene rather than
    # silently replacing it, so the two are never confused for one another.
    stem = "gsplats_3d_flylight_mcfo_neurons"
    if SAMPLE != DEFAULT_SAMPLE:
        stem = f"{stem}_{SAMPLE}"
    output_path = get_demos_output_dir() / f"{stem}.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    warn_if_no_cuda_gpu()

    # Colouring needs the individual channels, so the sample is loaded even
    # when the fit itself is cached.
    channels, combined, source_dtype = load_fisbe_sample(SAMPLE)

    with asection("Fitting neurons"):
        neurons = fit_volume(
            combined,
            _fit_cache_path(
                "neurons", SEEDS, FLOOR, CULL_RETENTION, NEURON_FIT_SCHEDULE
            ),
            SEEDS,
            FLOOR,
            CULL_RETENTION,
            "neurons",
            schedule=NEURON_FIT_SCHEDULE,
            source_dtype=source_dtype,
        )

    neuropil = None
    if not NO_NEUROPIL:
        with asection("Neuropil (reference channel)"):
            ref = neuropil_reference(SAMPLE, combined.shape)
            if ref is not None:
                neuropil = fit_volume(
                    ref.astype(np.float32) / 255.0,
                    _fit_cache_path("neuropil", NEUROPIL_SEEDS, "auto", 0.95),
                    NEUROPIL_SEEDS,
                    "auto",
                    0.95,
                    "neuropil",
                    source_dtype=str(ref.dtype),
                )

    with asection("Colouring neurons from MCFO channels"):
        neuron_rgb = splat_colors(neurons.centers, channels)

    centers, amps, chol, rgba = merge_for_render(neurons, neuron_rgb, neuropil)

    with asection("Summary"):
        aprint(f"  Neurons:  {len(neurons.amplitudes):,} splats")
        if neuropil is not None:
            aprint(f"  Neuropil: {len(neuropil.amplitudes):,} splats")
        else:
            aprint("  Neuropil: skipped")
        aprint(f"  Total:    {len(amps):,} splats")
        # 15 floats per splat: the 11 the other gsplat demos count (3 centers +
        # 6 Cholesky + amplitude + pad) plus the 4 baked RGBA columns, which are
        # part of what ships for this scene — colour is per-splat here, not a
        # colormap applied at render time. The numerator is the ACQUISITION's
        # element size, not ``combined``'s: the composite is a float32 working
        # copy of uint16 data, and measuring it doubles the ratio for free.
        source_bytes = combined.size * np.dtype(source_dtype).itemsize
        aprint(f"  Compression: {source_bytes / (len(amps) * 15 * 4):.0f}:1")

    scene_path = create_luxar_scene(
        centers, amps, chol, rgba, output_path, has_neuropil=neuropil is not None
    )

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
