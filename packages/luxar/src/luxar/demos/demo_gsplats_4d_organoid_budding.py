#!/usr/bin/env python3
"""GSplats Demo: 4D Budding Mouse Intestinal Organoid with its Cell Lineage

A mouse intestinal organoid grows from ONE cell into a budding mini-gut over
about 110 hours of live light-sheet imaging. Two fluorescent channels are fitted
as Gaussian splats, frame by frame, and the authors' tracked lineage of the
founder cell is drawn over them as fading Lines trails and Points markers.

================================================================================
FROM ONE CELL TO A BUDDING ORGANOID
================================================================================

Intestinal organoids are self-organising: a single stem cell, embedded in a
gel, divides into a hollow sphere of epithelium around a lumen, and the sphere
then breaks symmetry and buds, which is how crypts form. This recording follows
one organoid through the whole sequence. Scrub the Time slider from the start
and a single cell divides into a cyst; the cyst grows and, near the end of the
recording, pushes out a bud.

The two layers are the two fluorescent markers of the recording:

  - **Membranes** (mem9-GFP, 488 nm channel) - the epithelial cell outlines.
  - **Nuclei**    (H2B-mCherry, 561 nm channel) - one blob per cell.

The lineage is the founder cell's tree from the authors' partial,
semi-automatic tracking in Mastodon (shipped as a MaMuT XML file): 24,018
nuclear positions and 286 divisions in one tree (the file also holds one
isolated position). It does NOT cover every cell of the late organoid: at most
115 tracked cells are present at once. Each tracked cell
carries a comet tail of its last three hours, coloured by lineage: the hue
interval of a mother cell is split between her daughters at every division, so
sister clones share neighbouring hues and the founder's progeny separate into
up to eight colour families by the third generation.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Zenodo record 6828906, "Multiscale light-sheet organoid imaging
         framework" (light-sheet_recordings.zip, 11.95 GB)
URL:     https://zenodo.org/records/6828906
DOI:     10.5281/zenodo.6828906
License: CC BY 4.0
Recording: 002-P21-25X-Budding (one of seven main-figure recordings)

Only the parts this demo uses are fetched, by HTTP range requests into the
archive: every fourth ``rgb_grid/grid-T*.vti`` frame (about 345 MB) and the
lineage ``mamut_deconv.xml`` (0.9 MB). Each member is CRC-checked against the
archive directory and, as a set, against the SHA-256 digest pinned below.

What the frames are: the record does not carry the raw light-sheet stacks (the
paper's ~700 GB of imaging is available from the authors on request). The
``rgb_grid`` frames are the authors' own export of the DECONVOLVED two-channel
volume, cropped to the organoid, scaled to 8 bits and stored as RGB voxels.
Green (duplicated into blue) is the nuclei channel. Red is NOT the membrane
channel alone: it is membranes plus nuclei, added and clipped at 255, so every
nucleus appears in red too. The demo unmixes the membranes as red - green
(clipped at 0) and fits both channels at their native 2.0 x 0.26 x 0.26 um
voxel size. Evidence for the additive model, on frames T0301 and T0601: red is
never below green; inside nuclei (green > 60) red equals green in 99% / 94% of
voxels; and where both channels are present, red - green does not depend on
green while red itself rises with it.

Imaging:
--------
Microscope:  LS1-Live dual-illumination, inverted-detection light-sheet
             (Viventis Microscopy); recording 002 is a 25x acquisition
Organism:    mouse intestinal organoid, H2B-mCherry / mem9-GFP
Voxel size:  2.0 x 0.26 x 0.26 um (ZYX)
Temporal:    667 frames, 10 min apart (experiment.json records 0.1667 h)
Shown:       167 frames, every fourth (40 min apart)

How to Cite:
------------
de Medeiros, G., Ortiz, R., Strnad, P., Boni, A., Moos, F., Repina, N.,
Challet Meylan, L., Maurer, F. & Liberali, P. (2022). Multiscale light-sheet
organoid imaging framework. Nature Communications 13, 4864.
DOI: 10.1038/s41467-022-32465-z

WORKFLOW:
=========

1. **Fetch**   the 167 frames + lineage by range requests (cached, checksummed)
2. **Decode**  each VTK ImageData frame (zlib + base64, stdlib only)
3. **Fit**     each frame and channel (membranes unmixed as red - green) as 3D
               Gaussian splats on the shared 8-bit
               intensity scale, with a splat budget proportional to the
               organoid's occupied voxels (cached per frame)
4. **Place**   every frame at its crop origin, in acquisition microns
5. **Stack**   each channel into one 4D node (time = last centre column) with
               a slice-aware progressive ladder
6. **Lineage** fading trails (Lines) + current positions (Points) per frame
7. **View**    scrub or play Time; press L for the Layers panel

USAGE:
======
    python demo_gsplats_4d_organoid_budding.py [options]

Options:
    --recompute:   Re-fetch and refit from the Zenodo source (CUDA GPU strongly
                   recommended: about an hour of fitting on an 8 GB RTX 3070,
                   plus ~15 min to fetch; on CPU the late frames take minutes
                   each)
    --no-serve:    Generate the scene without launching the viewer
    --serve-only:  Just serve a previously generated scene

The per-frame fits are not hosted yet, so the demo refits from the Zenodo
source and caches every frame; a later run reuses the cache without touching
the network.

Output:
    - Scene saved to:  datasets/demos/gsplats_4d_organoid_budding.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_4d_organoid_budding",
    "title": "4D Budding Intestinal Organoid (lineage)",
    "description": "A mouse intestinal organoid growing from one cell to a budding mini-gut: two light-sheet channels as 4D splats, with the founder cell's lineage as trails.",
    "category": "microscopy",
    "geometry": "mixed",
    "requirements": {
        # The Zenodo source (167 frames + lineage) the fallback refit needs while
        # the precomputed fits are not hosted yet; see DATASET_NAME.
        "download_mb": 345,
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["gsplats_4d_organoid_budding"],
    "outputs": ["gsplats_4d_organoid_budding"],
    "citation": {
        "short": "de Medeiros et al., Nat Commun 2022 (Liberali lab)",
        "ref": "de Medeiros et al. 2022",
        "doi": "10.1038/s41467-022-32465-z",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/records/6828906",
    },
}

import base64
import hashlib
import struct
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, DimensionsConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    detect_device,
    download_zip_members,
    hsv_to_rgb,
    launch_viewer,
    local_fit_path,
    parse_demo_flags,
    print_data_provenance,
    stamp_input_digests,
    warn_if_no_cuda_gpu,
    window_attrs,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.merged_quality import collect_part_provenance
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

#: Manifest key (and cache directory) for the per-frame fits. The manifest lists
#: it as local-compute: the fits are not hosted, so the demo refits from the
#: Zenodo source.
DATASET_NAME = "gsplats_4d_organoid_budding"
#: Bundle of every per-frame fit cache, named exactly as `_frame_cache_name`.
#: TODO(Loic): once a cc-by record carries this bundle and the manifest entry is
#: promoted to bucket="zenodo", resolve it in `main` before the refit with
#: ``load_dataset_bundle(DATASET_NAME, BUNDLE_NAME, names)`` (names in CHANNELS
#: then FRAMES order; split the returned list per channel). Calling it now would
#: return None for a local-compute dataset, which a demo gate forbids.
BUNDLE_NAME = "organoid_budding_frames.gsplats.zarr.zip"
CACHE_DIR = Path.home() / ".cache" / "luxar" / DATASET_NAME

SOURCE_URL = (
    "https://zenodo.org/api/records/6828906/files/light-sheet_recordings.zip/content"
)
SOURCE_RECORD_URL = "https://zenodo.org/records/6828906"
RECORDING = "002-P21-25X-Budding"
LINEAGE_MEMBER = f"{RECORDING}/mamut_deconv.xml"

#: Acquisition frames in the recording (T0001..T0667, 10 min apart).
N_SOURCE_FRAMES = 667
#: Every FRAME_STEP-th frame is shown: 167 frames, 40 min apart.
FRAME_STEP = 4
#: Zero-based acquisition frames shown (T0001, T0005, ..., T0665).
FRAMES = tuple(range(0, N_SOURCE_FRAMES, FRAME_STEP))
#: Shown frames whose export is empty in the record, mapped to the acquisition
#: frame drawn in their place. T0245 is a 2 x 15 x 15 crop of zeros in both
#: channels; without a stand-in, playback blinks to an empty scene there. T0244,
#: 10 min earlier, supplies the splats AND the lineage overlay for that slot,
#: which keeps its nominal time (2440 min).
FRAME_STAND_INS = {244: 243}
#: Nominal frame interval. experiment.json records 0.1667 h, i.e. 10 min
#: rounded to four places; the axis uses the round value so every slider stop,
#: including the last, sits exactly on k * step.
FRAME_INTERVAL_MIN = 10.0
TIME_STEP_MIN = FRAME_STEP * FRAME_INTERVAL_MIN

#: SHA-256 over the per-member SHA-256 lines of everything fetched (the 167
#: frames, then the lineage XML), as computed by `_source_set_digest`. Pins the
#: source: the archive's own CRC32 only proves the bytes match its directory.
SOURCE_SET_SHA256 = "a2b5cb751a489f092cd57f99064114582b416c9ff44a17a5cd5922537775bad9"
#: SHA-256 of the lineage XML on its own: it is the one member a run with every
#: fit already cached (or a hosted bundle) still needs.
LINEAGE_SHA256 = "9e10c8bc7c5eb66459a158258390f6654314c5881483fe2996b5f5b52d022bd9"

#: Voxel size of every rgb_grid frame, (z, y, x) in microns.
VOXEL_SIZE_ZYX = (2.0, 0.26, 0.26)

#: The export is an additive composite: green and blue both carry the nuclei n,
#: and red carries the membranes m PLUS the nuclei, red = min(255, m + n). So the
#: membranes are unmixed as clip(red - green, 0); see `channel_volume`.
#: ``voxels_per_seed`` / ``splats_max`` set each channel's splat budget.
CHANNELS = (
    {
        "name": "membranes",
        "unmix": "red-minus-green",
        "voxels_per_seed": 12.0,
        # Thin sheets need more splats than nuclei: see the budget note below.
        "splats_max": 120_000,
        "marker": "mem9-GFP, 488 nm (membranes)",
        "colormap": "bop_orange",
        "window": (0.0, 1.0),
        "gamma": 1.0,
        "opacity": 0.15,
        "layer_order": 10,
    },
    {
        "name": "nuclei",
        "unmix": "green",
        "voxels_per_seed": 12.0,
        "splats_max": 60_000,
        "marker": "H2B-mCherry, 561 nm (nuclei)",
        "colormap": "bop_blue",
        "window": (0.0, 1.0),
        "gamma": 1.0,
        "opacity": 0.15,
        "layer_order": 20,
    },
)

# ---- Fit recipe ---------------------------------------------------------------
#: A voxel counts as occupied above this 8-bit value (it sets the budget only).
OCCUPIED_THRESHOLD = 16
#: Splat budget = occupied voxels / the channel's ``voxels_per_seed``, clipped
#: to [SPLATS_MIN, the channel's ``splats_max``]. The export is deconvolved, so
#: there is no noise peak to calibrate K* against; budgets were set on
#: foreground PSNR against the frame (1000 iterations, RTX 3070):
#:
#:   nuclei, frame 332:    4k / 8k / 16k seeds   39.8 / 44.0 / 47.0 dB
#:   membranes, frame 320: 1 per 24 / 12 / 6 voxels   38.4 / 40.5 / 42.9 dB
#:   membranes, frame 480: 33k / 60k / 67k / 120k     41.3 / 44.0 / 46.4 / 46.5
#:   membranes, frame 600: 60k / 82k / 120k / 165k    38.6 / 40.4 / 44.3 / 44.7
#:
#: One seed per 12 voxels holds ~40-46 dB on both channels. A 60k cap cost the
#: large late membrane frames ~6 dB (frame 600), so membranes cap at 120k:
#: about 1.9x the bytes on those frames (0.57 -> 1.06 MB) for +5.7 dB. Raising
#: it further bought 0.4 dB.
#: SPLATS_MIN: the founder-cell frames are small enough that the per-voxel rule
#: gives a few hundred seeds; frame 24's membranes scored 21-22 dB at 500 and
#: 1,000 seeds and 48 dB at 2,000. The min-extent cull still removes up to
#: two thirds of those seeds on the smallest frames, so early membrane frames
#: stay uneven (25.6-48.3 dB foreground; frame 64 keeps 641 splats).
SPLATS_MIN = 2000
N_ITERS = 1000

# ---- Lineage overlay --------------------------------------------------------
#: Comet-tail length in acquisition frames (10 min each): the last 3 hours.
TRAIL_HISTORY = 18
TRAIL_MIN_FADE = 0.15
TRAIL_LINE_WIDTH = 0.25  # um
TRAIL_OPACITY = 0.45
#: Trail vertices are averaged over this many frames either side along the
#: cell's own path (never across a division). The tracked positions jitter by
#: about a micron from frame to frame, comparable to the real motion, so raw
#: trails zigzag. Markers stay at the raw positions, on the nuclei.
TRAIL_SMOOTH_FRAMES = 2
POINT_RADIUS = 1.0  # um, well under a ~6 um nucleus
POINT_OPACITY = 0.8
#: Lineage colours stop splitting at this generation (2**3 = 8 families).
COLOUR_GENERATIONS = 3

# ---- Appearance --------------------------------------------------------------
#: Additive light grows with the organoid's thickness, so the exposure is set
#: for the late, budding frames (SwiftShader and RTX 3070 renders): at 0 or -1
#: stops they wash out to white, at -2.5 the bud and the lineage both read and
#: the single founder cell is still visible.
EXPOSURE_STOPS = -2.5
BLENDING_MODE = "additive"
#: Volumetric absorption (used only when BLENDING_MODE is "volumetric").
ABSORPTION = 1.0
#: Opening frame (zero-based acquisition frame, must be one of FRAMES): 6000
#: min, when the bud is clear and before the lumen fills the core with light.
OPENING_FRAME = 600
CAMERA_FRAME_FILL = 0.7

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

SCENE_NAME = "gsplats_4d_organoid_budding.luxar.zarr"

DEVICE: Optional[str] = None


# =============================================================================
# Source data
# =============================================================================


def source_frame(frame: int) -> int:
    """The acquisition frame drawn for a shown frame (see FRAME_STAND_INS)."""
    return FRAME_STAND_INS.get(frame, frame)


def frame_member(frame: int) -> str:
    """Archive member of one zero-based acquisition frame (T is one-based)."""
    return f"{RECORDING}/rgb_grid/grid-T{frame + 1:04d}.vti"


def _source_path(member: str) -> Path:
    return CACHE_DIR / "source" / member


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_set_digest(members: list[str]) -> str:
    """One SHA-256 over ``"<member> <sha256>"`` lines, in the given order."""
    lines = "".join(f"{m} {_sha256(_source_path(m))}\n" for m in members)
    return hashlib.sha256(lines.encode()).hexdigest()


def _source_is_cached(members: list[str]) -> bool:
    return all(_source_path(m).exists() for m in members) and (
        _source_set_digest(members) == SOURCE_SET_SHA256
    )


def fetch_lineage() -> Path:
    """The lineage XML, fetched only when missing; verified against its pin."""
    path = _source_path(LINEAGE_MEMBER)
    if not (path.exists() and _sha256(path) == LINEAGE_SHA256):
        path.unlink(missing_ok=True)  # a same-size corrupt copy would be skipped
        download_zip_members(SOURCE_URL, {LINEAGE_MEMBER: path})
    digest = _sha256(path)
    if digest != LINEAGE_SHA256:
        raise ValueError(
            f"Lineage digest mismatch: got {digest}, pinned {LINEAGE_SHA256}. "
            f"Delete {path} to re-fetch, or update the pin deliberately."
        )
    return path


def fetch_source() -> tuple[list[Path], Path]:
    """Fetch the shown frames and the lineage XML; verify the pinned digest.

    A complete, verified cache is used as it is, without touching the network.
    """
    members = [frame_member(source_frame(f)) for f in FRAMES] + [LINEAGE_MEMBER]
    if _source_is_cached(members):
        aprint(f"Using the cached source ({len(members)} members, digest verified)")
        return [_source_path(m) for m in members[:-1]], _source_path(LINEAGE_MEMBER)
    print_data_provenance(
        title=f"LSTree light-sheet recordings, {RECORDING}",
        source=SOURCE_RECORD_URL,
        license="CC BY 4.0 (de Medeiros et al., Nat Commun 2022)",
        url=SOURCE_URL,
        note=f"{len(FRAMES)} frames + lineage, ~345 MB by range requests",
    )
    download_zip_members(SOURCE_URL, {m: _source_path(m) for m in members})
    digest = _source_set_digest(members)
    if digest != SOURCE_SET_SHA256:
        raise ValueError(
            f"Source digest mismatch for {RECORDING}: got {digest}, pinned "
            f"{SOURCE_SET_SHA256}. The archive at {SOURCE_RECORD_URL} no longer "
            f"holds the bytes this demo was built from; delete "
            f"{CACHE_DIR / 'source'} to re-fetch, or update the pin deliberately."
        )
    return [_source_path(m) for m in members[:-1]], _source_path(LINEAGE_MEMBER)


def read_rgb_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode one LSTree ``rgb_grid`` frame (VTK XML ImageData, zlib, base64).

    Returns:
        ``(voxels, origin_zyx)``: uint8 ``(Z, Y, X, 3)`` RGB voxels and the crop
        origin in microns, in the recording's acquisition frame.
    """
    root = ET.parse(path).getroot()
    if root.get("type") != "ImageData" or root.get("byte_order") != "LittleEndian":
        raise ValueError(f"{path.name}: not little-endian VTK ImageData")
    if root.get("compressor") != "vtkZLibDataCompressor":
        raise ValueError(f"{path.name}: unexpected compressor {root.get('compressor')}")
    image = root.find("ImageData")
    array = root.find(".//PointData/DataArray")
    if image is None or array is None or array.text is None:
        raise ValueError(f"{path.name}: no ImageData point array")
    spacing = tuple(float(v) for v in image.get("Spacing", "").split())
    if spacing[::-1] != VOXEL_SIZE_ZYX:
        raise ValueError(f"{path.name}: spacing {spacing}, want {VOXEL_SIZE_ZYX[::-1]}")
    if image.get("Direction", "1 0 0 0 1 0 0 0 1").split() != list("100010001"):
        raise ValueError(f"{path.name}: rotated grids are not supported")
    if array.get("type") != "UInt8" or array.get("NumberOfComponents") != "3":
        raise ValueError(f"{path.name}: expected 3-component UInt8 voxels")
    x0, x1, y0, y1, z0, z1 = (int(v) for v in image.get("WholeExtent", "").split())
    shape = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1, 3)

    # VTK's compressed inline binary: a base64 header of UInt32
    # (n_blocks, block_size, last_block_size, compressed sizes...), then the
    # base64 of the concatenated zlib blocks.
    data = "".join(array.text.split()).encode("ascii")
    n_blocks = struct.unpack("<I", base64.b64decode(data[:8])[:4])[0]
    header_b64 = 4 * -(-(3 + n_blocks) * 4 // 3)
    header = struct.unpack(f"<{3 + n_blocks}I", base64.b64decode(data[:header_b64]))
    payload = base64.b64decode(data[header_b64:])
    blocks, offset = [], 0
    for size in header[3:]:
        blocks.append(zlib.decompress(payload[offset : offset + size]))
        offset += size
    raw = b"".join(blocks)
    if len(raw) != int(np.prod(shape)):
        raise ValueError(f"{path.name}: {len(raw)} bytes for shape {shape}")
    voxels = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    origin_xyz = np.array([float(v) for v in image.get("Origin", "").split()])
    # Extent offsets shift the origin by whole voxels (always 0 in this export).
    origin_xyz += np.array([x0, y0, z0]) * np.array(spacing)
    return voxels, origin_xyz[::-1].copy()


# =============================================================================
# Fitting
# =============================================================================


def _channel(name: str) -> dict:
    return next(ch for ch in CHANNELS if ch["name"] == name)


def channel_volume(voxels: np.ndarray, name: str) -> np.ndarray:
    """One channel's uint8 volume out of the export's RGB voxels.

    Green is the nuclei. Red is membranes + nuclei, added and clipped at 255,
    so the membranes are red - green (zero where the nuclei saturate red).
    Masking instead (keeping red only where it exceeds green) would keep the
    nuclear signal added at every membrane voxel that borders a nucleus.
    """
    unmix = _channel(name)["unmix"]
    red = voxels[..., 0].astype(np.int16)
    green = voxels[..., 1].astype(np.int16)
    if unmix == "green":
        return voxels[..., 1].copy()
    if unmix == "red-minus-green":
        return np.clip(red - green, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown unmix {unmix!r}")


def splat_budget(volume: np.ndarray, name: str) -> int:
    """Splats for one frame/channel: proportional to the occupied voxels."""
    ch = _channel(name)
    occupied = int((volume > OCCUPIED_THRESHOLD).sum())
    seeds = round(occupied / ch["voxels_per_seed"])
    return int(np.clip(seeds, SPLATS_MIN, ch["splats_max"]))


def _frame_cache_name(channel: str, frame: int) -> str:
    """Per-frame fit cache name, keyed by the acquisition frame actually fitted
    and by everything that changes the fit; the hosted bundle's members carry
    these names."""
    ch = _channel(channel)
    return (
        f"organoid_{channel}_{ch['unmix']}_vps{ch['voxels_per_seed']:g}"
        f"_min{SPLATS_MIN}_max{ch['splats_max']}_it{N_ITERS}"
        f"_t{source_frame(frame) + 1:04d}.gsplats.zarr.zip"
    )


def fit_frame(
    volume: np.ndarray, origin_zyx: np.ndarray, cache_file: Path, channel: str
) -> GSplatData:
    """Fit one frame/channel in acquisition microns, cached."""
    global DEVICE
    if cache_file.exists() and not RECOMPUTE:
        return GSplatData.load(cache_file, include_stats=True)
    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    seeds = splat_budget(volume, channel)
    aprint(f"  {cache_file.name}: {volume.shape} voxels -> {seeds:,} seeds")
    result = fit_gaussian_splats(
        volume.astype(np.float32),
        seeds=seeds,
        n_iters=N_ITERS,
        # The export is already background-free outside the organoid (zeros),
        # and a frame-dependent floor would make a timelapse flicker.
        floor="none",
        # ONE intensity scale for every frame: the authors' 8-bit export is
        # scaled once for the whole recording, so 255 means the same in every
        # frame and the organoid's brightening and dimming survive the fit.
        norm_range=(0.0, 255.0),
        voxel_size=VOXEL_SIZE_ZYX,
        source_dtype="uint8",
        device=DEVICE,
        verbose=False,
    ).translate(origin_zyx)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    save_with_lod(
        result,
        cache_file,
        recipe="stream",
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    # Return what was STORED: the cache is written under a lossy encoding, so a
    # cold run and a warm run must both read it back to build the same scene.
    return GSplatData.load(cache_file, include_stats=True)


def _fit_cache(channel: str, frame: int) -> Path:
    return local_fit_path(DATASET_NAME, _frame_cache_name(channel, frame))


def cached_fits() -> Optional[dict[str, list[GSplatData]]]:
    """Every frame's fits from the local cache, or ``None`` if any is missing.

    A fully cached run then needs neither the source frames nor the network.
    """
    if RECOMPUTE or not all(
        _fit_cache(ch["name"], f).exists() for ch in CHANNELS for f in FRAMES
    ):
        return None
    with asection(f"Loading {len(FRAMES)} cached frames x {len(CHANNELS)} channels"):
        return {
            ch["name"]: [
                GSplatData.load(_fit_cache(ch["name"], f), include_stats=True)
                for f in FRAMES
            ]
            for ch in CHANNELS
        }


def fit_all_frames(frame_paths: list[Path]) -> dict[str, list[GSplatData]]:
    """Fit every shown frame for both channels, in FRAMES order."""
    fits: dict[str, list[GSplatData]] = {ch["name"]: [] for ch in CHANNELS}
    with asection(f"Fitting {len(frame_paths)} frames x {len(CHANNELS)} channels"):
        for frame, path in zip(FRAMES, frame_paths, strict=True):
            voxels, origin = read_rgb_grid(path)
            for ch in CHANNELS:
                name = ch["name"]
                volume = channel_volume(voxels, name)
                fits[name].append(
                    fit_frame(volume, origin, _fit_cache(name, frame), name)
                )
    return fits


def stack_channel(frames: list[GSplatData]) -> GSplatData:
    """Stack one channel's frames into a 4D dataset, time (minutes) last."""
    values = [frame * FRAME_INTERVAL_MIN for frame in FRAMES]
    part_provenance = collect_part_provenance(
        frames,
        values=values,
        fit_reference={
            "kind": "preprocessed",
            "note": (
                "LSTree rgb_grid export: deconvolved, cropped to the organoid "
                "and scaled to 8 bits by the authors"
            ),
        },
    )
    return GSplatData.combine_as_new_dimension(
        frames, values=values, sigma=0.0, part_provenance=part_provenance
    )


# =============================================================================
# Lineage
# =============================================================================


def load_lineage(
    path: Path,
) -> tuple[dict[int, tuple[int, np.ndarray]], dict[int, int]]:
    """Read the MaMuT lineage: spots and each spot's parent.

    Returns:
        ``(spots, parent)``: spot id -> (zero-based frame, (z, y, x) microns),
        and child id -> parent id. MaMuT stores some links later-to-earlier, so
        every edge is oriented by frame; a link within one frame is dropped.
    """
    root = ET.parse(path).getroot()
    spots: dict[int, tuple[int, np.ndarray]] = {}
    for spot in root.iter("Spot"):
        spots[int(spot.get("ID", -1))] = (
            int(spot.get("FRAME", -1)),
            np.array(
                [float(spot.get(f"POSITION_{axis}", "nan")) for axis in "ZYX"],
                dtype=np.float32,
            ),
        )
    parent: dict[int, int] = {}
    for edge in root.iter("Edge"):
        a, b = int(edge.get("SPOT_SOURCE_ID", -1)), int(edge.get("SPOT_TARGET_ID", -1))
        if spots[a][0] == spots[b][0]:
            continue
        early, late = (a, b) if spots[a][0] < spots[b][0] else (b, a)
        # A spot with two incoming links keeps the lower id, deterministically.
        parent[late] = min(early, parent.get(late, early))
    return spots, parent


def lineage_hues(spots: dict, parent: dict[int, int]) -> dict[int, float]:
    """Hue per spot: each division splits the mother's hue interval.

    The root owns [0, 1); at every division the daughters split their mother's
    interval equally (ordered by spot id), until COLOUR_GENERATIONS divisions.
    A spot's hue is the midpoint of its interval, so sisters stay close in hue.
    """
    children: dict[int, list[int]] = {}
    for child, mother in parent.items():
        children.setdefault(mother, []).append(child)
    hues: dict[int, float] = {}
    roots = sorted(s for s in spots if s not in parent)
    stack = [(r, i / len(roots), (i + 1) / len(roots), 0) for i, r in enumerate(roots)]
    while stack:
        spot, lo, hi, gen = stack.pop()
        hues[spot] = 0.5 * (lo + hi)
        kids = sorted(children.get(spot, []))
        split = len(kids) > 1 and gen < COLOUR_GENERATIONS
        for k, kid in enumerate(kids):
            if split:
                width = (hi - lo) / len(kids)
                stack.append((kid, lo + k * width, lo + (k + 1) * width, gen + 1))
            else:
                stack.append((kid, lo, hi, gen + (len(kids) > 1)))
    return hues


def _trail_links(
    current: list[int], spots: dict, parent: dict[int, int], frame: int
) -> list[tuple[int, int]]:
    """(mother, child) links of the trails ending at ``frame``: each current
    cell's ancestry back to TRAIL_HISTORY frames, shared stretches once."""
    links: list[tuple[int, int]] = []
    seen: set[int] = set()
    for sid in sorted(current):
        child = sid
        while child in parent and child not in seen:
            seen.add(child)
            mother = parent[child]
            if frame - spots[mother][0] > TRAIL_HISTORY:
                break
            links.append((mother, child))
            child = mother
    return links


def smoothed_positions(spots: dict, parent: dict[int, int]) -> dict[int, np.ndarray]:
    """Each spot's position averaged with up to TRAIL_SMOOTH_FRAMES neighbours
    on either side of its own path; the walk stops at a division or a gap."""
    children: dict[int, list[int]] = {}
    for child, mother in parent.items():
        children.setdefault(mother, []).append(child)
    smoothed = {}
    for sid, (_, position) in spots.items():
        window = [position]
        back = sid
        for _ in range(TRAIL_SMOOTH_FRAMES):
            back = parent.get(back, -1)
            if back < 0 or len(children.get(back, [])) != 1:
                break
            window.append(spots[back][1])
        ahead = sid
        for _ in range(TRAIL_SMOOTH_FRAMES):
            kids = children.get(ahead, [])
            if len(kids) != 1:
                break
            ahead = kids[0]
            window.append(spots[ahead][1])
        smoothed[sid] = np.mean(window, axis=0).astype(np.float32)
    return smoothed


def lineage_geometry(spots: dict, parent: dict[int, int]) -> dict[str, np.ndarray]:
    """Per shown frame: fading trails back through each cell's ancestry, and
    one marker per tracked cell. Vertices are (z, y, x, time_min)."""
    hues = lineage_hues(spots, parent)
    smooth = smoothed_positions(spots, parent)
    by_frame: dict[int, list[int]] = {}
    for sid, (frame, _) in spots.items():
        by_frame.setdefault(frame, []).append(sid)

    verts, colors, edges, pts, pt_colors = [], [], [], [], []
    for shown in FRAMES:
        t = shown * FRAME_INTERVAL_MIN
        frame = source_frame(shown)
        current = by_frame.get(frame, [])
        links = _trail_links(current, spots, parent, frame)
        # Each joint is authored ONCE per frame so consecutive links share it
        # (an indexed line reads as one trail, not a chain of beads).
        index: dict[int, int] = {}
        for sid in dict.fromkeys(s for link in links for s in link):
            age = frame - spots[sid][0]
            fade = 1.0 - (1.0 - TRAIL_MIN_FADE) * age / TRAIL_HISTORY
            index[sid] = len(verts)
            # The newest vertex sits on its marker; older ones are smoothed.
            verts.append([*(spots[sid][1] if age == 0 else smooth[sid]), t])
            colors.append(np.asarray(hsv_to_rgb(hues[sid], 0.8, 1.0)) * fade)
        edges.extend((index[a], index[b]) for a, b in links)
        for sid in sorted(current):
            pts.append([*spots[sid][1], t])
            pt_colors.append(hsv_to_rgb(hues[sid], 0.8, 1.0))
    return {
        "trail_vertices": np.asarray(verts, dtype=np.float32),
        "trail_colors": np.asarray(colors, dtype=np.float32),
        "trail_edges": np.asarray(edges, dtype=np.uint32).reshape(-1, 2),
        "points": np.asarray(pts, dtype=np.float32),
        "point_colors": np.asarray(pt_colors, dtype=np.float32),
    }


# =============================================================================
# Scene
# =============================================================================


def opening_camera(stacks: dict[str, GSplatData]) -> CameraConfig:
    """Frame the opening frame's organoid, looking down the stack axis."""
    t_open = OPENING_FRAME * FRAME_INTERVAL_MIN
    centers, weights = [], []
    for data in stacks.values():
        keep = np.abs(data.centers[:, 3] - t_open) < 0.5 * TIME_STEP_MIN
        centers.append(data.centers[keep, :3])
        weights.append(data.amplitudes[keep])
    c, w = np.concatenate(centers), np.concatenate(weights).astype(np.float64)
    target_zyx = np.average(c, axis=0, weights=w)
    radius = float(np.percentile(np.linalg.norm((c - target_zyx)[:, 1:], axis=1), 98))
    distance = radius / CAMERA_FRAME_FILL / np.tan(np.radians(CINEMATIC_FOV_DEG / 2))
    target = (float(target_zyx[2]), float(target_zyx[1]), float(target_zyx[0]))
    # Mostly along -z (the microscope's view), tilted to show the depth. With
    # up = -y the image reads as it does on the microscope: x right, y down.
    tilt = np.radians(25.0)
    eye = np.array(target) + distance * np.array([0.0, np.sin(tilt), -np.cos(tilt)])
    return CameraConfig(
        position=tuple(float(v) for v in eye), target=target, up=(0.0, -1.0, 0.0)
    )


def create_luxar_scene(
    stacks: dict[str, GSplatData], lineage: dict[str, np.ndarray], output_path: Path
) -> Path:
    """Write the 4D scene: two gsplat layers + lineage trails and markers."""
    t_max = FRAMES[-1] * FRAME_INTERVAL_MIN
    with asection("Creating 4D organoid scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
                Dimension(
                    "time",
                    unit="min",
                    display=False,
                    discrete=True,
                    range=(0.0, t_max),
                    step=TIME_STEP_MIN,
                ),
            ]
        )
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    exposure=EXPOSURE_STOPS,
                    camera=opening_camera(stacks),
                    dimensions=DimensionsConfig(
                        current_step=[0.0, 0.0, 0.0, OPENING_FRAME * FRAME_INTERVAL_MIN]
                    ),
                ),
            )
            stamp_input_digests(scene)
            scene.attrs["title"] = "GSplats: Budding Intestinal Organoid + Lineage"
            scene.attrs["description"] = (
                f"Mouse intestinal organoid ({RECORDING}, Zenodo 6828906), live "
                f"light-sheet, {len(FRAMES)} frames 40 min apart: from one cell to "
                "a budding mini-gut. Membranes (bop_orange) and nuclei (bop_blue) "
                "as Gaussian splats; the founder cell's tracked lineage as fading "
                "trails and markers coloured by lineage."
            )

            for ch in CHANNELS:
                with asection(f"Adding {ch['name']} layer ({ch['marker']})"):
                    # Centre columns are (z, y, x, time); `slice_dims=[3]` is the
                    # time column, so every rung carries an equal budget PER
                    # FRAME instead of a share of the whole stack (the late
                    # frames hold most of the splats). `recompute=True` is
                    # required: the stacked per-frame ladders would otherwise be
                    # merged as they are and the spec ignored.
                    scene.add_gsplats_from_data(
                        name=ch["name"],
                        result=stacks[ch["name"]],
                        dim_order=["z", "y", "x", "time"],
                        extend_to_all=[],
                        additive_lod=dict(n_lods=4, slice_dims=[3], recompute=True),
                        blending_mode=BLENDING_MODE,
                        layer_order=ch["layer_order"],
                        **(
                            {"absorption": ABSORPTION}
                            if BLENDING_MODE == "volumetric"
                            else {}
                        ),
                        colormap=ch["colormap"],
                        gamma=ch["gamma"],
                        opacity=ch["opacity"],
                        layer=True,
                        **window_attrs(ch["window"]),
                    )

            scene.add_lines(
                name="lineage_trails",
                vertices=lineage["trail_vertices"],
                widths=TRAIL_LINE_WIDTH,
                colors=lineage["trail_colors"],
                indices=lineage["trail_edges"],
                line_type="indexed",
                dim_order=["z", "y", "x", "time"],
                extend_to_all=[],
                opacity=TRAIL_OPACITY,
                layer=True,
            )
            scene.add_points(
                name="tracked_cells",
                positions=lineage["points"],
                colors=lineage["point_colors"],
                radii=POINT_RADIUS,
                dim_order=["z", "y", "x", "time"],
                extend_to_all=[],
                opacity=POINT_OPACITY,
                layer=True,
            )

            scene.add_text(
                "Budding intestinal organoid",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "Light-sheet • membranes + nuclei • lineage",
                DEMO_META.get("citation"),
            )
    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Resolve the fits and the lineage, build the scene, optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: 4D Budding Intestinal Organoid + Lineage")
    aprint("=" * 70)
    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    # A fully cached local fit needs no source and no network; otherwise refit
    # from the Zenodo source (see BUNDLE_NAME for the hosted path to come).
    fits = cached_fits()
    if fits is None:
        warn_if_no_cuda_gpu()
        frame_paths, lineage_path = fetch_source()
        fits = fit_all_frames(frame_paths)
    else:
        # Fits carry the splats only; the lineage comes from the original
        # record (0.9 MB), fetched once and pinned.
        lineage_path = fetch_lineage()

    with asection("Stacking frames"):
        stacks = {name: stack_channel(frames) for name, frames in fits.items()}
        for name, data in stacks.items():
            aprint(f"{name}: {data.n_splats:,} splats over {len(FRAMES)} frames")
    with asection("Lineage"):
        spots, parent = load_lineage(lineage_path)
        lineage = lineage_geometry(spots, parent)
        aprint(
            f"{len(spots):,} tracked positions; {len(lineage['trail_edges']):,} "
            f"trail segments and {len(lineage['points']):,} markers over "
            f"{len(FRAMES)} frames"
        )

    scene_path = create_luxar_scene(stacks, lineage, output_path)
    if not NO_SERVE:
        launch_viewer(scene_path)
    aprint("\nDone!")


if __name__ == "__main__":
    main()
