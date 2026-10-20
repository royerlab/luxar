#!/usr/bin/env python
"""GSplats Demo: 3D Spatial Transcriptomics (EASI-FISH, Mouse Lateral Hypothalamus).

A slab of mouse lateral hypothalamus, expanded and imaged by **EASI-FISH**
(Expansion-Assisted Iterative Fluorescence In Situ Hybridization) on a Zeiss
Lightsheet Z.1. Four channels: DAPI-stained nuclei and the single mRNA
molecules of three genes, each a diffraction-limited punctum.

    channel  laser   target    what it marks
    c0       488 nm  Gpr83     a G-protein-coupled receptor; a subset of cells
    c1       669 nm  Slc17a6   VGLUT2, the glutamatergic (excitatory) neurons
    c2       405 nm  DAPI      every nucleus
    c3       546 nm  Bdnf      brain-derived neurotrophic factor

THE STORY IS TWO SPATIAL SCALES:
    At the opening view the slab is a field of grey nuclei, and the cells that
    express each gene stand out in that gene's colour: where a cell holds many
    transcripts, its puncta merge into a coloured cloud. Zoom in on one of those
    cells and the cloud resolves into individual mRNA molecules, each a separate
    dot. That is the read-out EASI-FISH exists for (transcript counts per cell,
    in 3D tissue), and it is what a partitioned, per-part level-of-detail scene
    is built for: coarse merged splats while a part is small on screen, the
    finest fitted puncta once it fills it.

DATA SOURCE & CITATIONS:
    EASI-FISH example data (multi-round FISH, round 3), sample LHA3, Janelia
    Research Campus. CC BY 4.0.
    figshare DOI 10.25378/janelia.13622819 (file ``LHA3_R3_medium.czi``,
    34.9 GB, two tiles x 4 channels x 591 x 1920 x 1920, uint16).

    Please cite:

    - Wang, Y., Eddison, M., Fleishman, G., Weigert, M., Xu, S., Wang, T.,
      Rokicki, K., Goina, C., Henry, F. E., Lemire, A. L., Schmidt, U.,
      Yang, H., Svoboda, K., Myers, E. W., Saalfeld, S., Korff, W.,
      Sternson, S. M. & Tillberg, P. W. "EASI-FISH for thick tissue defines
      lateral hypothalamus spatio-molecular organization." Cell 184 (26),
      6361-6377 (2021). doi:10.1016/j.cell.2021.11.024
      (Tillberg and Sternson labs, HHMI Janelia.)

STITCHING (the dataset ships two unstitched tiles):
    No stitched product of this file is published, so the demo stitches the two
    views itself. The recipe follows the dataset's own layout: the CZI records a
    stage ``PositionX`` per view and the acquisition used an 8% tile overlap
    along x, so the nominal offset is read from the metadata and then refined by
    normalised cross-correlation of the two DAPI max-projections over the
    overlap (search +-80 px in x, +-16 px in y). The overlap is blended with a
    linear ramp. Measured on this file: dx 1732 px, dy 14 px, NCC 0.93. The
    stitching code was first written for a large-scale Luxar test of this
    dataset and is reproduced here so the build runs from the public source.

    It is rigid and per-file: one translation for all four channels. The
    sub-voxel DAPI-vs-FISH offset a chromatic registration would correct is
    left in. The raw data also carries faint vertical stripes (~3 px period);
    the fit reproduces them about 10x attenuated.

BIOLOGICAL MICROMETRES (the tissue was expanded):
    EASI-FISH images an expanded hydrogel, so the CZI's voxel size (0.42 x
    0.23 x 0.23 um) is in EXPANDED-tissue units. The authors' own analysis code
    divides by a linear expansion factor of 2 (``roi_prop(..., ex=2)``,
    "linear expansion factor, default is 2", in github.com/multiFISH/EASI-FISH,
    ``easi_fish/roi_prop.py``), and their published LHA tables are "in
    pre-expansion units" (figshare 10.25378/janelia.13749154). The fits are
    therefore scaled by 1/2 as a last step, so the scene reads in biological
    micrometres: the slab is ~124 x 217 x 417 um of tissue (expanded: ~249 x
    435 x 834 um). The fits and their quality figures are on the expanded grid.

FIT RECIPE (every step caches under the demo's ``local/`` dir; a re-run resumes
at the first missing one):
    1. Fetch ``LHA3_R3_medium.czi`` from figshare (sha256-pinned, resumable),
       or use a copy passed with ``--czi PATH`` (checked against the same hash).
    2. Stitch, then write a local OME-Zarr with two levels: xy/2 (0.42 x 0.46 x
       0.46 expanded um, the FISH fits) and z/2 xy/4 (0.84 x 0.91 x 0.91, DAPI).
    3. Resolve each channel's background floor (``auto``) on its own array.
    4. ``gsplat batch-fit run`` per mRNA channel: content tiling with pinned
       density knobs, the ``draft`` preset, ``--min-extent 0``, and a per-part
       ``levels`` LOD (3 coarser levels) built by the streaming merge.
    5. ``gsplat batch-fit run`` for DAPI: uniform 256-voxel tiles, 600,000
       seeds, the ``standard`` preset, the same per-part ``levels`` merge.
    6. ``gsplat additive -b equi-energy:4 -m self_energy`` re-ladders every
       leaf so its first rung is the few splats carrying the most light (the
       merge's ``levels`` recipe only takes equal-count ladders), then
       ``gsplat transform --scale 0.5`` to biological micrometres.
    7. Build the 4-layer scene, then ``luxar optimize --pack``.

WHY ``--min-extent 0`` FOR THE mRNA CHANNELS:
    The default post-fit pruning (``--min-extent 0.6``) removes splats whose
    largest sigma is under 0.6 voxels. Most of what it removes are specks at
    the reseed size, but a few carry single molecules, so for spot-level
    fidelity the mRNA channels keep every splat. The gain is small: on Gpr83
    crops fitted at 30,000 seeds, spot recall (detected puncta recovered at
    50% peak) was 0.992 unpruned against 0.983 pruned at full resolution, and
    1.00 either way at the xy/2 level fitted here, with up to ~1 dB of
    foreground PSNR. DAPI keeps the default: nuclei are many voxels across.

    Those figures, and every quality figure quoted for this demo, assume the
    pedestal-seeding fix of PR #3226. Without it most seeds start dead on this
    dataset's 209-237 count background pedestal, which costs far more recall
    and many more splats than the pruning does. This demo is meant to merge
    after that PR.

WHY THE EXPOSURE IS SET BY HAND:
    Each layer's brightness and colormap window are authored below (see
    ``Channel``). With the default normalisation every layer's 99.9th
    percentile lands at 1.0, and four additive layers summed through a 250 um
    slab wash out to a beige-white in which the genes cannot be told apart.
    DAPI, dense and many voxels thick, needs ~200x less light than the puncta:
    here it is a dim grey context and the three genes are saturated, distinct
    hues (magenta, green, yellow), all under ACES. The values were chosen by
    rendering the scene, not derived from statistics.

USAGE:
    python demo_gsplats_3d_easifish_hypothalamus.py [--no-serve] [--serve-only]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.
    --czi PATH:    Use a local copy of LHA3_R3_medium.czi instead of
                   downloading it (verified against the pinned sha256).
    --keep-stale:  Reuse an existing scene even if this demo's code changed.

    No hosted copy of the fits is published yet, so the first run builds them.
    Measured on an NVIDIA GB10 (128 GB unified memory):

    - Disk: the 34.9 GB CZI plus ~5 GB of derived files (3.2 GB OME-Zarr,
      ~0.4 GB of fits, the 0.26 GB scene). The CZI stays in the cache after
      the build; see the demos README for how to remove it.
    - Host RAM: >= 32 GB. Stitching holds ~20 GB (two 4.4 GB views plus the
      stitched channel and its levels); each mRNA fit runs a ~13 GB parent
      plus ~6.5 GB per worker, up to three workers (~33 GB).
    - GPU: CUDA. Workers are sized with ``--jobs-per-gpu auto`` and capped at
      three; a card with less memory simply runs fewer at once.
    - Time: about 2.5 hours of fitting. Later runs reuse the cached fits and
      rebuild the scene only when the demo's code or a fit changed.

OUTPUT:
    - Scene saved to: datasets/demos/gsplats_3d_easifish_hypothalamus.luxar.zarr
    - Opens in the browser; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_3d_easifish_hypothalamus",
    "title": "3D Spatial Transcriptomics (EASI-FISH, Mouse Hypothalamus)",
    "description": (
        "Four-channel EASI-FISH of mouse lateral hypothalamus: nuclei plus the "
        "mRNA of three genes, from cell types at the overview to single "
        "molecules up close."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        # The 34.9 GB raw CZI. No hosted fit exists yet (see the manifest
        # entry), so a first run downloads it and refits.
        "download_mb": 34862,
        "compute": "heavy",
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_easifish_lha"],
    "outputs": ["gsplats_3d_easifish_hypothalamus"],
    "citation": {
        "short": "Wang et al. 2021 (Tillberg and Sternson labs, HHMI Janelia)",
        "ref": "Wang et al. 2021",
        "doi": "10.1016/j.cell.2021.11.024",
        "license": "CC BY 4.0",
        "url": "https://doi.org/10.25378/janelia.13622819",
    },
}

import hashlib
import math
import os
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    BUILDER_FINGERPRINT_ATTR,
    add_demo_caption,
    demo_source_fingerprint,
    download_with_checksum,
    launch_viewer,
    local_fit_path,
    parse_demo_flags,
    parse_path_arg,
    print_data_provenance,
    require_module,
    run_luxar_cli,
    scene_is_current,
    stamp_input_digests,
    warn_if_no_cuda_gpu,
    window_attrs,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds, iter_default_leaves
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_easifish_lha"

SCENE_NAME = "gsplats_3d_easifish_hypothalamus.luxar.zarr"

#: The public source. figshare publishes an md5 (9358626ab10464716cd7ffa00c70d272);
#: the sha256 below was computed from a copy that matched it.
CZI_URL = "https://ndownloader.figshare.com/files/26510777"
CZI_NAME = "LHA3_R3_medium.czi"
CZI_SHA256 = "3152e32d30983046f714c2a482a8706aa89492fdc2ca20a38661ef39b6903fcc"
CZI_BYTES = 34_861_369_536

RECOMPUTE_DIR = local_fit_path(DATASET, CZI_NAME).parent
OME_ZARR = RECOMPUTE_DIR / "lha3_r3_stitched.ome.zarr"

#: Linear expansion of the EASI-FISH hydrogel. The CZI's voxel size is in
#: expanded units; the scene divides by this to read in biological um. Source:
#: the authors' analysis code, github.com/multiFISH/EASI-FISH
#: ``easi_fish/roi_prop.py`` ("ex: linear expansion factor, default is 2").
EXPANSION_FACTOR = 2.0

#: Fit workers per GPU are sized by ``--jobs-per-gpu auto`` and capped here
#: through ``LUXAR_AUTO_WORKER_HARD_CAP``. Uncapped, the auto sizer picked eight
#: workers on a 128 GB unified-memory GB10 and ran it down to 3 GB free (it read
#: device and host memory as two pools); three ran with room to spare (13 GB
#: parent + ~6.5 GB per worker). The cap only lowers what auto would choose, so
#: a smaller card still gets its own, smaller count.
WORKER_CAP = 3

#: Every leaf's additive ladder: four rungs at equal shares of cumulative
#: self-energy, so the first rung of each part is the handful of splats that
#: carry the most light and first paint is a small, visually heavy subset.
LADDER_ARGS = ("-b", "equi-energy:4", "-m", "self_energy")


@dataclass(frozen=True)
class Channel:
    """One CZI channel: how it is identified, fitted and drawn.

    ``emission_nm`` is checked against the CZI metadata before anything is
    fitted, because the channel ORDER is the only thing tying a gene name to an
    index. ``level`` is the OME-Zarr path the channel is fitted from.
    """

    index: int
    key: str
    label: str
    emission_nm: float
    level: str
    colormap: str
    #: Emitted brightness: the layer's amplitudes are scaled so their 99.9th
    #: percentile lands here (``normalize_amplitudes=<target>``). Radiance is
    #: linear in amplitude and sums along the ray, so this, not the window, is
    #: the exposure knob.
    brightness: float
    #: Colormap window over the amplitudes in units of that same percentile
    #: (1.0 = the 99.9th). It only picks the LUT colour; a raised low end
    #: renders the faint diffuse floor black, so puncta stay crisp dots.
    window: tuple[float, float]
    #: ``batch-fit run`` arguments that differ per channel (see the table notes).
    fit_args: tuple[str, ...]


# Content-tiling density knobs, pinned per mRNA channel. They transfer a splat
# density measured by `gsplat cal` on a region of each channel (K* = 32,000
# splats for a reference region holding `--n-features-ref` features above
# `--feature-threshold`) to every content box. They are pinned rather than
# recalibrated on each build because calibration on sparse puncta is slow
# (20-26 min per K=64k fit on a GB10) and its held-out metric is dominated by
# background, so it is not a better estimate than these values.
_K_STAR_REF = "32000"

# The mRNA channels and DAPI share a merge recipe: each content box (or tile)
# becomes one part of a `kind=partition`, and each part gets its own `levels`
# ladder of 3 coarser levels below the fit. That is the per-part LOD the story
# needs: parts small on screen draw merged splats, parts the camera is close to
# draw the fitted puncta.
_MERGE_ARGS = ("--merge-recipe", "levels", "--merge-levels", "3")

_FISH_COMMON = (
    "--tiling",
    "content",
    "--k-star-ref",
    _K_STAR_REF,
    "--preset",
    "draft",
    # Keep every splat (a ~1% spot-recall gain). See "WHY --min-extent 0" above.
    "--min-extent",
    "0",
)

CHANNELS: tuple[Channel, ...] = (
    Channel(
        index=2,
        key="dapi",
        label="DAPI (nuclei)",
        emission_nm=415.0,
        level="2",
        colormap="gray",
        brightness=0.025,
        # Wider than the mRNA windows: the 99.9th percentile sits at LUT 0.4 of
        # the grey ramp, which keeps DAPI a dim grey context (and a stored
        # window is capped at intensity 100, which a narrow one would exceed).
        window=(0.0, 2.5),
        fit_args=(
            "--tiling",
            "uniform",
            "--tile-size",
            "256",
            "--seeds",
            "600000",
            "--preset",
            "standard",
        ),
    ),
    Channel(
        index=1,
        key="slc17a6",
        label="Slc17a6 mRNA",
        emission_nm=648.0,
        level="1",
        colormap="magenta",
        brightness=3.5,
        window=(0.05, 0.6),
        fit_args=(
            *_FISH_COMMON,
            "--n-features-ref",
            "222",
            "--feature-threshold",
            "44.5755",
        ),
    ),
    Channel(
        index=0,
        key="gpr83",
        label="Gpr83 mRNA",
        emission_nm=498.0,
        level="1",
        colormap="green",
        brightness=5.0,
        window=(0.05, 0.6),
        fit_args=(
            *_FISH_COMMON,
            "--n-features-ref",
            "89",
            "--feature-threshold",
            "60.475",
        ),
    ),
    Channel(
        index=3,
        key="bdnf",
        label="Bdnf mRNA",
        emission_nm=571.0,
        level="1",
        colormap="yellow",
        brightness=2.5,
        # Bdnf carries a diffuse cytoplasmic signal under its puncta. A raised
        # low end renders that floor black and a short window saturates the
        # LUT, so Bdnf+ cells read clean yellow rather than khaki haze.
        window=(0.2, 0.7),
        fit_args=(
            *_FISH_COMMON,
            "--n-features-ref",
            "1235",
            "--feature-threshold",
            "65.1955",
        ),
    ),
)

# Stitch search window around the nominal (stage-derived) offset, in pixels.
_SEARCH_DX = 80
_SEARCH_DY = 16
#: Tile overlap the acquisition was set to, used only when the CZI does not
#: carry one stage position per view.
_NOMINAL_OVERLAP = 0.08

#: Lowest NCC accepted for the tile registration (0.93 measured on this file).
_MIN_STITCH_NCC = 0.5

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
KEEP_STALE = FLAGS["keep_stale"]
#: ``--czi PATH``: a copy of the CZI the user already has.
CZI_ARG = parse_path_arg("czi")
FINGERPRINT = demo_source_fingerprint(__file__)


# =============================================================================
# Recompute step 1: fetch the CZI
# =============================================================================
def _sha256(path: Path) -> str:
    """Streaming sha256 of a (35 GB) file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(64 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def user_czi(path: Path) -> Path:
    """Check a ``--czi`` copy against the pinned hash, without touching it.

    The download helper deletes a file whose digest does not match, which is
    right for its own cache and wrong for a file the user owns, so a user copy
    is hashed here and refused (not removed) on a mismatch.
    """
    if not path.is_file():
        raise SystemExit(f"--czi {path}: no such file")
    with asection(f"Verifying {path} (sha256 of {CZI_BYTES / 1e9:.1f} GB)"):
        got = _sha256(path)
    if got != CZI_SHA256:
        raise SystemExit(
            f"--czi {path} has sha256 {got}, not the pinned {CZI_SHA256} of "
            f"{CZI_NAME}. The figshare record also holds a 12 GB 'small' and a "
            "'tiny' acquisition; this demo needs the 34.9 GB 'medium' one."
        )
    return path


def fetch_czi() -> Path:
    """Download the round-3 CZI from figshare, or reuse a verified copy.

    ``download_with_checksum`` resumes a partial transfer and verifies the
    sha256 on every path, cache hit included. A file at this name is not
    evidence of the right file: the same record also holds a 12 GB two-channel
    ``small`` acquisition and a ``tiny`` crop.
    """
    if CZI_ARG is not None:
        return user_czi(CZI_ARG)
    RECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)
    target = RECOMPUTE_DIR / CZI_NAME
    print_data_provenance(
        title="EASI-FISH example data (multi-round FISH, round 3), LHA3",
        source="Janelia Research Campus (Wang et al., Cell 2021)",
        license="CC BY 4.0",
        url="https://doi.org/10.25378/janelia.13622819",
        note=f"{CZI_BYTES / 1e9:.1f} GB raw CZI; resumable, sha256-verified.",
    )
    with asection(f"Fetching {CZI_NAME}"):
        download_with_checksum(
            CZI_URL, target, expected_sha256=CZI_SHA256, expected_size=CZI_BYTES
        )
    aprint(f"CZI: {target} ({target.stat().st_size:,} bytes)")
    return target


# =============================================================================
# Recompute step 2: stitch the two views and write the OME-Zarr
# =============================================================================
@dataclass(frozen=True)
class TileOffset:
    """Where the right-hand view lands relative to the left-hand one."""

    order: tuple[int, int]  # (left view, right view)
    dx: int
    dy: int
    ncc: float


def _ncc(p: np.ndarray, q: np.ndarray) -> float:
    """Normalised cross-correlation of two equal-shaped images."""
    p = p - p.mean()
    q = q - q.mean()
    denom = math.sqrt(float((p * p).sum()) * float((q * q).sum())) + 1e-9
    return float((p * q).sum()) / denom


def _overlap(
    left: np.ndarray, right: np.ndarray, dx: int, dy: int
) -> tuple[np.ndarray, np.ndarray]:
    """The overlapping strips of two (H, W) images for a candidate offset.

    The right image starts ``dx`` columns into the left one and ``dy`` rows
    below it; rows that either image lacks are cropped from both.
    """
    h, w = left.shape
    rows_l = slice(max(0, dy), h + min(0, dy))
    rows_r = slice(max(0, -dy), h - max(0, dy))
    return left[rows_l, dx:], right[rows_r, : w - dx]


def _best_offset(
    left: np.ndarray,
    right: np.ndarray,
    dxs: range,
    dys: range,
    best: tuple[float, int, int],
) -> tuple[float, int, int]:
    """Return the ``(ncc, dx, dy)`` maximising NCC over a grid of offsets."""
    w = left.shape[1]
    for dx in dxs:
        if w - dx < 40:
            continue
        for dy in dys:
            score = _ncc(*_overlap(left, right, dx, dy))
            if score > best[0]:
                best = (score, dx, dy)
    return best


def find_tile_offset(mips: list[np.ndarray], nominal_dx: int) -> TileOffset:
    """Register two max-projections by NCC around the stage-derived offset.

    The view ORDER is searched too (two candidates): whether stage +x runs
    along image +x is a property of the instrument setup, not of the file.
    Coarse (2 px in x, 4 px in y), then +-3 / +-5 px at full resolution.
    """
    found: Optional[TileOffset] = None
    for order in ((0, 1), (1, 0)):
        left, right = mips[order[0]], mips[order[1]]
        coarse = _best_offset(
            left,
            right,
            range(nominal_dx - _SEARCH_DX, nominal_dx + _SEARCH_DX + 1, 2),
            range(-_SEARCH_DY, _SEARCH_DY + 1, 4),
            (-2.0, 0, 0),
        )
        fine = _best_offset(
            left,
            right,
            range(coarse[1] - 3, coarse[1] + 4),
            range(coarse[2] - 5, coarse[2] + 6),
            coarse,
        )
        if found is None or fine[0] > found.ncc:
            found = TileOffset(order=order, dx=fine[1], dy=fine[2], ncc=fine[0])
    assert found is not None
    return found


def _nominal_dx(metadata: str, n_views: int, width: int, pixel_um: float) -> int:
    """Stage-derived x offset between the two views, in pixels.

    The CZI records one ``<PositionX>`` (um) per view. When it does not, fall
    back to the 8% overlap the acquisition was set to.
    """
    positions = [float(v) for v in re.findall(r"<PositionX>([^<]+)<", metadata)]
    if len(positions) == n_views:
        return int(round(abs(positions[1] - positions[0]) / pixel_um))
    aprint(f"  {len(positions)} stage positions for {n_views} views; using 8%")
    return int(round(width * (1.0 - _NOMINAL_OVERLAP)))


def _check_layout(sizes: dict[str, int], image: Any) -> None:
    """Refuse a CZI that is not the two-view, four-channel acquisition."""
    if sizes.get("V") != 2:
        raise RuntimeError(f"expected 2 tile views, the CZI has {sizes.get('V')}")
    names = list(image.attrs["channels"])
    # A dict cannot hold a duplicate name, so a collapsed one shows as a count
    # mismatch: two channels sharing a name would make the order ambiguous.
    if len(names) != sizes.get("C") or len(set(names)) != len(names):
        raise RuntimeError(
            f"expected {sizes.get('C')} uniquely named channels, got {names}"
        )


def _check_offset(offset: TileOffset, width: int) -> None:
    """Refuse a registration that cannot be the 8%-overlap tile pair."""
    if not 0 < offset.dx < width:
        raise RuntimeError(f"registered dx {offset.dx} is outside (0, {width})")
    if offset.ncc < _MIN_STITCH_NCC:
        raise RuntimeError(
            f"tile registration NCC {offset.ncc:.3f} < {_MIN_STITCH_NCC}: the "
            "two views do not overlap as the stage positions say"
        )


def _check_channels(image: Any) -> None:
    """Refuse a CZI whose channel order does not match :data:`CHANNELS`."""
    emissions = [
        float(meta.get("EmissionWavelength", 0.0))
        for meta in image.attrs["channels"].values()
    ]
    for ch in CHANNELS:
        got = emissions[ch.index]
        if abs(got - ch.emission_nm) > 2.0:
            raise RuntimeError(
                f"CZI channel {ch.index} emits at {got:.0f} nm; {ch.label} "
                f"should be {ch.emission_nm:.0f} nm. Wrong file or channel order."
            )


def _blend_views(left: np.ndarray, right: np.ndarray, offset: TileOffset) -> np.ndarray:
    """Paste two (Z, H, W) views at ``offset`` with a linear ramp across the overlap."""
    z, h, w = left.shape
    dx, dy = offset.dx, offset.dy
    rows_l = slice(max(0, dy), h + min(0, dy))
    rows_r = slice(max(0, -dy), h - max(0, dy))
    lc, rc = left[:, rows_l, :], right[:, rows_r, :]
    overlap = w - dx
    out = np.empty((z, lc.shape[1], dx + w), dtype=np.uint16)
    out[:, :, :dx] = lc[:, :, :dx]
    out[:, :, w:] = rc[:, :, overlap:]
    ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
    for start in range(0, z, 32):  # slabs bound the float32 temporaries
        zs = slice(start, min(start + 32, z))
        mixed = lc[zs, :, dx:] * (1.0 - ramp) + rc[zs, :, :overlap] * ramp
        out[zs, :, dx:w] = np.rint(mixed).astype(np.uint16)
    return out


def _downsample(volume: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    """Block-mean downsample (Z, Y, X) by integer factors, cropping remainders."""
    fz, fy, fx = factors
    z, y, x = (s // f for s, f in zip(volume.shape, factors, strict=True))
    blocks = volume[: z * fz, : y * fy, : x * fx].reshape(z, fz, y, fy, x, fx)
    return np.rint(blocks.mean(axis=(1, 3, 5), dtype=np.float32)).astype(np.uint16)


#: OME-Zarr levels written, keyed by their path: block-mean factors (z, y, x)
#: applied to the stitched full-resolution channel. Full resolution itself is
#: not stored: no fit reads it, and it would add ~16 GB.
_LEVELS = {"1": (1, 2, 2), "2": (2, 4, 4)}


def stitch_czi(czi_path: Path, out_zarr: Path) -> None:
    """Stitch the two CZI views and write a two-level OME-Zarr (c, z, y, x).

    One channel at a time: each view is ~4.4 GB of uint16, so the peak is about
    20 GB. The background floor of every channel at every level is resolved
    here, while its array is in memory, and recorded in the store's attrs.
    """
    czifile = require_module("czifile")
    from luxar._zarr_compat import create_array, open_group
    from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor
    from luxar.gsplats.fitting.preprocessing import resolve_volume_floor

    with asection("Stitching the two CZI views"), czifile.CziFile(str(czi_path)) as czi:
        image = czi.scenes[0]
        sizes = image.sizes
        aprint(f"CZI: {dict(sizes)}")
        _check_layout(dict(sizes), image)
        _check_channels(image)
        scales = image.attrs["coord_scales"]
        voxel_um = (scales["Z"] * 1e6, scales["Y"] * 1e6, scales["X"] * 1e6)

        dapi = next(ch.index for ch in CHANNELS if ch.key == "dapi")
        mips = [
            image(V=v, C=dapi).asarray().max(axis=0).astype(np.float32)
            for v in range(sizes["V"])
        ]
        nominal = _nominal_dx(czi.metadata(), sizes["V"], sizes["X"], voxel_um[2])
        offset = find_tile_offset(mips, nominal)
        _check_offset(offset, sizes["X"])
        aprint(
            f"nominal dx {nominal} px -> registered dx {offset.dx}, "
            f"dy {offset.dy}, NCC {offset.ncc:.3f}, order {offset.order}"
        )

        group = open_group(str(out_zarr), mode="w")
        compressor = resolve_compressor(WIDTH_AWARE_DEFAULT, np.uint16)
        floors: dict[str, list[Optional[float]]] = {k: [] for k in _LEVELS}
        arrays: dict[str, Any] = {}
        for c in range(sizes["C"]):
            left = image(V=offset.order[0], C=c).asarray()
            right = image(V=offset.order[1], C=c).asarray()
            stitched = _blend_views(left, right, offset)
            del left, right
            for path, factors in _LEVELS.items():
                level = _downsample(stitched, factors)
                if path not in arrays:
                    arrays[path] = create_array(
                        group,
                        path,
                        shape=(sizes["C"], *level.shape),
                        dtype=np.uint16,
                        chunks=(1, 64, 256, 256),
                        compressor=compressor,
                    )
                arrays[path][c] = level
                floors[path].append(resolve_volume_floor(level, "auto"))
            aprint(
                f"channel {c}: stitched {stitched.shape}, floors "
                f"{[floors[p][-1] for p in _LEVELS]}"
            )
            del stitched

        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": "LHA3_R3_medium",
                "axes": [
                    {"name": "c", "type": "channel"},
                    *(
                        {"name": a, "type": "space", "unit": "micrometer"}
                        for a in ("z", "y", "x")
                    ),
                ],
                "datasets": [
                    {
                        "path": path,
                        "coordinateTransformations": [
                            {
                                "type": "scale",
                                "scale": [
                                    1.0,
                                    *(
                                        v * f
                                        for v, f in zip(voxel_um, factors, strict=True)
                                    ),
                                ],
                            }
                        ],
                    }
                    for path, factors in _LEVELS.items()
                ],
            }
        ]
        group.attrs["easifish_stitch"] = {
            "order": list(offset.order),
            "dx": offset.dx,
            "dy": offset.dy,
            "ncc": offset.ncc,
            "floors": floors,
            # Written LAST: a store without it is a partial stitch, rebuilt.
            "complete": True,
        }


def _stitch_record(ome_zarr: Path) -> Optional[dict[str, Any]]:
    """The completed stitch record of ``ome_zarr``, or None if absent/partial."""
    from luxar._zarr_compat import open_group

    if not ome_zarr.exists():
        return None
    record = dict(open_group(str(ome_zarr), mode="r").attrs).get("easifish_stitch")
    return record if record and record.get("complete") else None


# =============================================================================
# Recompute steps 3-5: fit every channel
# =============================================================================
@contextmanager
def _worker_cap(cap: int) -> Iterator[None]:
    """Cap ``--jobs-per-gpu auto`` at ``cap`` workers for the duration.

    ``batch-fit run`` reads ``LUXAR_AUTO_WORKER_HARD_CAP`` when it sizes its
    workers, so the cap is set in this process (whose environment the workers
    inherit) and the previous value restored afterwards.
    """
    key = "LUXAR_AUTO_WORKER_HARD_CAP"
    previous = os.environ.get(key)
    os.environ[key] = str(cap)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def merged_path(ch: Channel) -> Path:
    """The streaming merge's output for one channel (expanded-tissue um)."""
    return RECOMPUTE_DIR / f"fit_{ch.key}" / "merged" / "final.gsplats.zarr"


def archive_path(ch: Channel) -> Path:
    """The finished per-channel archive the scene is built from."""
    return RECOMPUTE_DIR / f"fit_{ch.key}" / f"easifish_{ch.key}.gsplats.zarr"


def fit_channel(ch: Channel, floor: Optional[float]) -> Path:
    """Fit one channel with ``batch-fit run`` and return its merged archive.

    The floor is passed as a number, resolved on THIS channel's array. Passing
    ``--floor auto`` with ``--channels`` would resolve one shared level across
    all four channels (the minimum), which leaves the brighter-floored channels
    with a pedestal of haze.

    ``batch-fit run`` resumes: tiles already on disk are skipped.
    """
    final = merged_path(ch)
    if final.exists():
        aprint(f"{ch.label}: fit present ({final})")
        return final
    with (
        asection(f"Fitting {ch.label} (channel {ch.index}, level {ch.level})"),
        _worker_cap(WORKER_CAP),
    ):
        run_luxar_cli(
            "gsplat",
            "batch-fit",
            "run",
            str(OME_ZARR),
            str(final.parent.parent),
            "--array-key",
            ch.level,
            "--channels",
            f"{ch.index}:{ch.index + 1}",
            # Centres in (expanded-tissue) micrometres, from the OME-Zarr
            # scale: the two levels then land in one physical frame.
            "--physical",
            "--floor",
            "none" if floor is None else f"{floor:.6g}",
            "--gpus",
            "auto",
            "--jobs-per-gpu",
            "auto",
            *ch.fit_args,
            *_MERGE_ARGS,
        )
    if not final.exists():
        raise RuntimeError(f"batch-fit produced no {final}")
    return final


def finish_channel(ch: Channel, merged: Path) -> Path:
    """Front-load every leaf's ladder, then scale to biological micrometres.

    Both steps rewrite the tree structure-preservingly (partition parts and
    their ``levels`` groups keep their shape) and touch no fit, so a change to
    either costs about a minute per channel, never a refit.
    """
    final = archive_path(ch)
    if final.exists() and final.stat().st_mtime >= merged.stat().st_mtime:
        return final
    laddered = final.with_name(f"laddered_{ch.key}.gsplats.zarr")
    with asection(f"Finishing {ch.label}"):
        # `batch-fit run --merge-recipe levels` only builds equal-count ladders
        # (its ladder knobs belong to the `stream` recipe), so the ladder is
        # rebuilt from each leaf's union here.
        run_luxar_cli(
            "gsplat",
            "additive",
            str(merged),
            str(laddered),
            *LADDER_ARGS,
            "--overwrite",
        )
        scale = ",".join([f"{1.0 / EXPANSION_FACTOR:g}"] * 3)
        if final.exists():  # stale: `transform` has no --overwrite
            shutil.rmtree(final)
        run_luxar_cli(
            "gsplat", "transform", str(laddered), str(final), "--scale", scale
        )
        shutil.rmtree(laddered)
    return final


def recompute_channel_paths() -> dict[str, Path]:
    """Run the whole recipe from the figshare CZI; return archives by key."""
    warn_if_no_cuda_gpu()
    with asection("Recomputing the EASI-FISH fits from the figshare CZI"):
        record = _stitch_record(OME_ZARR)
        if record is None:
            # Before the 35 GB download, not after it: a missing reader should
            # cost a message, not an hour of transfer.
            require_module("czifile")
            stitch_czi(fetch_czi(), OME_ZARR)
            record = _stitch_record(OME_ZARR)
            assert record is not None
        else:
            aprint(f"stitched OME-Zarr present: {OME_ZARR}")
        floors = record["floors"]
        return {
            ch.key: finish_channel(ch, fit_channel(ch, floors[ch.level][ch.index]))
            for ch in CHANNELS
        }


# =============================================================================
# Data loading
# =============================================================================
def resolve_channel_paths() -> dict[str, Path]:
    """The four per-channel archives, built locally from the figshare CZI.

    The dataset is CC BY 4.0 and redistributable, but its fits are not hosted
    yet (the manifest lists it as ``local-compute``), so they are always built
    here and cached. Once a record carries them, the manifest entry moves to the
    ``zenodo`` bucket and this function resolves them with ``ensure_dataset``.
    """
    return recompute_channel_paths()


# =============================================================================
# Scene construction
# =============================================================================
def _camera(bmin: np.ndarray, bmax: np.ndarray) -> CameraConfig:
    """A tilted overview of the slab, image convention (y down).

    Scene dimensions keep the archives' column order (z, y, x): a grafted
    partition cannot be remapped. So world X is depth (z), world Y is image y
    and world Z is image x. The camera looks down the depth axis at the XY face
    (as Fiji/napari would), with world -Y up so image rows count down, then
    tilts about image x so the slab's thickness reads in 3D.
    """
    centre = (bmin + bmax) / 2.0
    extent = bmax - bmin
    # Fit the long image-x extent (world Z) across the frame width at 16:9.
    vh = 2.0 * math.tan(math.radians(CINEMATIC_FOV_DEG / 2.0))
    distance = max(extent[2] / (0.85 * vh * 16 / 9), extent[1] / (0.85 * vh))
    distance += extent[0] / 2.0
    tilt = math.radians(CAMERA_TILT_DEG)
    position = centre + distance * np.array([-math.cos(tilt), math.sin(tilt), 0.0])
    return CameraConfig(
        position=tuple(float(v) for v in position),
        target=tuple(float(v) for v in centre),
        up=(0.0, -1.0, 0.0),
    )


#: Tilt of the opening camera away from face-on, in degrees.
CAMERA_TILT_DEG = 30.0


def create_luxar_scene(paths: dict[str, Path], output_path: Path) -> None:
    """Build the four-layer scene from the per-channel archives."""
    with asection("Creating the EASI-FISH scene"):
        nodes = {ch.key: load_gsplat_node(str(paths[ch.key]))[0] for ch in CHANNELS}
        # Finest-level splats: the coarser per-part levels are merged stand-ins
        # for the same molecules, so counting them would inflate the figure.
        counts = {
            key: sum(leaf.n_splats for leaf in iter_default_leaves(node))
            for key, node in nodes.items()
        }
        bounds = [center_bounds(node) for node in nodes.values()]
        bmin = np.min([np.asarray(b[0]) for b in bounds], axis=0)
        bmax = np.max([np.asarray(b[1]) for b in bounds], axis=0)
        del nodes
        aprint(f"bounds (um): {np.round(bmin, 1)} .. {np.round(bmax, 1)}")
        aprint(f"finest-level splats per channel: {counts}")

        dims = Dimensions(
            [
                Dimension(name, unit="µm", display=True, range=(lo, hi))
                for name, lo, hi in zip(
                    ("z", "y", "x"), bmin.tolist(), bmax.tolist(), strict=True
                )
            ]
        )
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    camera=_camera(bmin, bmax),
                    # Scientific-fidelity override: the cinematic 35 mm lens
                    # smears and colour-fringes puncta away from the frame
                    # centre, and a single mRNA molecule IS a punctum. The rest
                    # of the cinematic look (bloom, vignette, noise) stays.
                    chromatic_lens_distortion_enabled=False,
                ),
                citation=DEMO_META["citation"],
            )
            stamp_input_digests(scene)
            scene.attrs[BUILDER_FINGERPRINT_ATTR] = FINGERPRINT
            scene.attrs["title"] = "GSplats: EASI-FISH Mouse Lateral Hypothalamus"
            scene.attrs["description"] = (
                "EASI-FISH spatial transcriptomics in mouse lateral hypothalamus "
                "(Wang et al., Cell 2021): DAPI nuclei plus single mRNA molecules "
                "of Slc17a6, Gpr83 and Bdnf, fitted as "
                f"{sum(counts.values()):,} Gaussian splats with per-part "
                "level of detail. Coordinates are biological micrometres (the "
                "tissue was imaged after 2x linear expansion). Zoom in on a "
                "coloured cell to resolve its transcripts. Press L for the "
                "Layers panel."
            )
            for ch in CHANNELS:
                with asection(f"Adding {ch.label} ({ch.colormap})"):
                    scene.add_gsplats_from_file(
                        name=ch.label,
                        path=str(paths[ch.key]),
                        # Additive: four co-located channels must SUM, not
                        # occlude one another, and additive is commutative, so
                        # no layer order has to be declared.
                        blending_mode="additive",
                        opacity=1.0,
                        colormap=ch.colormap,
                        normalize_amplitudes=ch.brightness,
                        # The window is authored relative to the 99.9th
                        # percentile; the stored amplitudes now put that
                        # percentile at `brightness`.
                        **window_attrs(
                            (ch.window[0] * ch.brightness, ch.window[1] * ch.brightness)
                        ),
                        layer=True,
                    )
            scene.add_text(
                "EASI-FISH • mouse lateral hypothalamus",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "DAPI grey • Slc17a6 magenta • Gpr83 green • Bdnf yellow • "
                f"{sum(counts.values()):,} splats",
                DEMO_META.get("citation"),
            )
    aprint(f"Scene saved: {output_path}")


def pack_scene(built: Path, output_path: Path) -> None:
    """Re-chunk nothing, but add chunk packs: one request per small node.

    The fits are hundreds of small parts and levels whose chunks are a few KB
    each, so without packs the opening view costs thousands of requests
    (measured: 3,447 unpacked against 887 packed).
    """
    run_luxar_cli("optimize", str(built), str(output_path), "--pack", "--overwrite")


def scene_is_fresh(output_path: Path, archives: dict[str, Path]) -> bool:
    """Whether the built scene can be served as-is.

    Current means two things: :func:`scene_is_current` (this demo's code and
    the zarr writer are unchanged, ``--keep-stale`` honoured) AND the scene is
    newer than every per-channel archive, so a re-finished fit rebuilds it.
    """
    if not scene_is_current(output_path, FINGERPRINT, keep_stale=KEEP_STALE):
        return False
    built = output_path.stat().st_mtime
    return all(path.stat().st_mtime <= built for path in archives.values())


def main() -> None:
    """Resolve the fits, build and pack the scene, and serve it."""
    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if not output_path.exists():
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
            return
    else:
        paths = resolve_channel_paths()
        if scene_is_fresh(output_path, paths):
            aprint(f"Using existing scene: {output_path}")
        else:
            built = local_fit_path(DATASET, "scene_unpacked.luxar.zarr")
            if built.exists():
                shutil.rmtree(built)
            create_luxar_scene(paths, built)
            pack_scene(built, output_path)
            shutil.rmtree(built)

    if not NO_SERVE:
        launch_viewer(output_path)


if __name__ == "__main__":
    main()
