#!/usr/bin/env python3
"""GSplats Demo: 2D Whole-Slide Pathology Image (CMU-1, OpenSlide)

Visualises a large H&E-stained whole-slide pathology image as 2D Gaussian
splats with per-channel (R/G/B) layers, using tiled fitting to handle
the full gigapixel resolution.

================================================================================
2D GAUSSIAN SPLATTING — WHOLE-SLIDE PATHOLOGY IMAGE (TILED)
================================================================================

This demo shows the complete workflow for tiled multi-channel 2D Gaussian
splatting on a real gigapixel digital pathology dataset:
- Downloading the Aperio SVS whole-slide image from OpenSlide test data
- Splitting into R, G, B channels (with brightness inversion for brightfield)
- Tiled fitting of 2D Gaussian splats per channel (Hann cosine apodization)
- Adding each channel as a separate layer with colormap for visualisation

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  OpenSlide test data — CMU-1.svs
URL:    https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/
Format: Aperio SVS (pyramidal TIFF, JPEG-compressed 256x256 tiles)
Size:   46,000 x 32,914 pixels (1.5 gigapixels), ~169 MB download
Stain:  Hematoxylin & Eosin (H&E) — brightfield histology

Imaging:
--------
Scanner:      Aperio ScanScope
Magnification: 20x
Specimen:     Human tissue section

Channels (derived from RGB):
-----------------------------
  0: Red   — Eosin / cytoplasm / connective tissue
  1: Green — Intermediate (both stains contribute)
  2: Blue  — Hematoxylin / nuclei / basophilic structures

License:
--------
CC0 1.0 Universal (Public Domain Dedication)

How to Cite:
------------
If you use this dataset, please cite:
  OpenSlide: A Vendor-Neutral Software Foundation for Digital Pathology
  Adam Goode, Benjamin Gilbert, Jan Harkes, Drazen Jukic, M. Satyanarayanan
  Journal of Pathology Informatics 2013, 4:27
  https://openslide.org

WORKFLOW:
=========

1. **Download** SVS whole-slide image from OpenSlide (~169 MB)
2. **Read** full-resolution level from the pyramidal TIFF
3. **Split** into R, G, B channels and invert (brightfield -> dark-on-light)
4. **Tiled fit** 2D Gaussian splats per channel (Hann cosine apodization
   for seamless tile stitching) with GPU acceleration
5. **Add** each channel as a separate layer with colormap
6. **Visualise** in the Luxar web viewer

USAGE:
======
    python demo_gsplats_2d_cmu1_pathology.py [OPTIONS]

Options:
    --recompute:        Force re-fitting from scratch (download + GPU fitting)
    --no-serve:         Generate scene without launching viewer
    --serve-only:       Just serve a previously generated scene
    --show-roundtrip:   Show matplotlib comparison of original vs reconstructed images
    --target-size=N:    Downsample target for longest axis (default: 0 = full res)
    --tile-size=N:      Tile size in pixels for tiled fitting (default: 4096)
    --overlap=N:        Tile overlap in pixels (default: 512)
    --seeds-per-tile=N: Gaussian seeds per tile (default: 255000, per tile)

By default, the demo works at full resolution (46,000 x 32,914 pixels)
using tiled fitting.

IF THE HOSTED FIT CANNOT BE OBTAINED (#1618):
=============================================
The normal path loads the precomputed per-channel fits through the manifest.
If neither the published archive nor an in-repo copy can be obtained, the demo
next looks for THIS machine's own earlier refit in
``~/.cache/luxar/gsplats_cmu1_pathology/local/``, and failing that falls through
to exactly what ``--recompute`` does: the ~169 MB SVS download
and a three-channel tiled fit over the full 1.5-gigapixel image. That is a
long, unattended run on a first launch, and it is deliberate — the same
"compute your own stand-in" fallback every migrated gsplat demo has, and the
reason the result is cached in the local-fit namespace where nothing
quarantines it. Only that one routable absence is answered this way: a
checksum that will not verify, an unknown file name and a missing packaged
manifest are faults and still crash.

Output:
    - Scene saved to:  datasets/demos/gsplats_2d_cmu1_pathology.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_2d_cmu1_pathology",
    "title": "2D Whole-Slide Pathology Image (CMU-1, OpenSlide)",
    "description": "H&E whole-slide pathology image (CMU-1, 1.5 gigapixel) as tiled 2D Gaussian splats.",
    "category": "medical",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 150,  # approx
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["gsplats_cmu1_pathology"],
    "outputs": ["gsplats_2d_cmu1_pathology"],
    "citation": {
        "short": "OpenSlide test data (Goode et al. 2013)",
        "ref": "Goode et al. 2013",
        "doi": "10.4103/2153-3539.119005",
        "license": "CC0 1.0",
    },
}

# Enable MPS->CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import zipfile
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import UIConfig, ViewerConfig
from luxar.demos import (
    DatasetUnavailable,
    MissingDependencyError,
    add_demo_caption,
    download_with_checksum,
    ensure_dataset,
    launch_viewer,
    local_fit_path,
    parse_demo_flags,
    require_module,
    warn_if_no_cuda_gpu,
)
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# OpenSlide test data — direct HTTP download (no auth required)
DOWNLOAD_URL = (
    "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1.svs"
)
EXPECTED_FILE_SIZE = 177_552_579  # 169.3 MB
EXPECTED_SHA256 = "00a3d54482cd707abf254fe69dccc8d06b8ff757a1663f1290c23418c480eb30"

# Original image specs
ORIGINAL_WIDTH = 46_000
ORIGINAL_HEIGHT = 32_914
MAGNIFICATION = 20  # 20x objective

# Channel configuration — RGB split from brightfield H&E
N_CHANNELS = 3
CHANNELS = [
    {"index": 0, "name": "Red (Eosin/Tissue)", "color": (1.0, 0.2, 0.2)},
    {"index": 1, "name": "Green", "color": (0.2, 1.0, 0.2)},
    {"index": 2, "name": "Blue (Hematoxylin/Nuclei)", "color": (0.2, 0.2, 1.0)},
]

# Tiled fitting parameters
TILE_SIZE = 4096  # Pixels per tile axis — fits comfortably in GPU memory
OVERLAP = 512  # Overlap for Hann cosine apodization (seamless stitching)
SEEDS_PER_TILE = 500000  # Seeds per tile
N_ITERS = 6_000

# Manifest key for the precomputed per-channel artifacts, and the files it pins.
DATASET = "gsplats_cmu1_pathology"
GSPLATS_FILES = [f"cmu1_ch{i}.gsplats.zarr.zip" for i in range(N_CHANNELS)]

# Cache location. CACHE_DIR holds the DOWNLOADED slide; a local refit is OUR
# artifact, not a copy of the hosted one, so it goes to the demo's local-fit
# namespace (~/.cache/luxar/<name>/local/, see `local_fit_path`). Written under
# the manifest's own names it was quarantined by the next fetch for failing the
# pinned sha256, and the demo refit every time (#1618).
CACHE_DIR = Path.home() / ".cache" / "luxar" / DATASET

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Parse optional CLI overrides
TARGET_SIZE = 0  # Default: full resolution (46,000 x 32,914)
for _arg in sys.argv:
    if _arg.startswith("--target-size="):
        TARGET_SIZE = int(_arg.split("=")[1])
    elif _arg.startswith("--tile-size="):
        TILE_SIZE = int(_arg.split("=")[1])
    elif _arg.startswith("--overlap="):
        OVERLAP = int(_arg.split("=")[1])
    elif _arg.startswith("--seeds-per-tile="):
        SEEDS_PER_TILE = int(_arg.split("=")[1])

# Arbol logging depth
Arbol.max_depth = 10
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Auto-detected on first fit
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def load_cmu1_image() -> tuple:
    """Download and load the CMU-1 whole-slide image as RGB channels.

    Downloads the SVS file, reads the full-resolution level (or a suitable
    pyramid level if --target-size is set), splits into R/G/B, inverts
    (brightfield), and normalises.

    Returns:
        ``(channels, acquisition)`` -- 3 channel images (2D float32, normalised to
        [0, 1]), and the ``(shape, dtype)`` of ONE channel of the slide page as
        read, to be declared to the fit: the images handed back are a resized,
        inverted float32 copy of it.
    """
    tifffile = require_module("tifffile")

    from scipy.ndimage import zoom

    # Download
    svs_path = CACHE_DIR / "CMU-1.svs"
    download_with_checksum(
        DOWNLOAD_URL,
        svs_path,
        expected_sha256=EXPECTED_SHA256,
        expected_size=EXPECTED_FILE_SIZE,
    )

    # Read SVS — find best pyramid level
    with asection("Loading whole-slide image"):
        aprint(f"Reading {svs_path.name}...")

        with tifffile.TiffFile(str(svs_path)) as tif:
            aprint(f"Pages: {len(tif.pages)}")
            for i, page in enumerate(tif.pages):
                aprint(f"  Page {i}: {page.shape}, dtype={page.dtype}")

            if TARGET_SIZE > 0:
                # Find smallest page that is >= TARGET_SIZE on longest axis
                best_page_idx = 0
                best_shape = tif.pages[0].shape

                for i, page in enumerate(tif.pages):
                    shape = page.shape
                    if len(shape) < 2:
                        continue
                    longest = max(shape[0], shape[1])
                    if longest >= TARGET_SIZE:
                        best_longest = max(best_shape[0], best_shape[1])
                        if longest < best_longest:
                            best_page_idx = i
                            best_shape = shape
            else:
                # Full resolution — first page
                best_page_idx = 0
                best_shape = tif.pages[0].shape

            aprint(f"Selected page {best_page_idx}: {best_shape}")
            data = tif.pages[best_page_idx].asarray()
            aprint(f"Loaded: {data.shape}, dtype={data.dtype}")
            # The slide page as read, before the per-channel split, the resize
            # and the float32 cast below. One CHANNEL of it is what each fit
            # represents, so the channel axis is dropped from the declaration.
            acquisition = (tuple(int(x) for x in data.shape[:2]), str(data.dtype))

    # Convert to float32 and split channels
    with asection("Processing RGB channels"):
        if data.ndim == 2:
            data = np.stack([data, data, data], axis=-1)

        if data.ndim != 3 or data.shape[2] < 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {data.shape}")

        # Take only first 3 channels (ignore alpha if present)
        data = data[:, :, :3].astype(np.float32)

        # Normalise to [0, 1]
        if data.max() > 1.0:
            data = data / 255.0

        h, w = data.shape[0], data.shape[1]
        aprint(f"Image: {w}x{h} pixels, RGB float32")

        # Optional downsampling
        longest = max(h, w)
        if TARGET_SIZE > 0 and longest > TARGET_SIZE:
            scale = TARGET_SIZE / longest
            target_h = int(h * scale)
            target_w = int(w * scale)
            aprint(f"Downsampling to {target_w}x{target_h}")
        else:
            target_h, target_w = h, w
            scale = 1.0

        # Invert brightness: H&E brightfield has white background and dark tissue.
        # For gsplat fitting, we want tissue structures to be bright (high amplitude).
        aprint("Inverting brightness (brightfield -> dark background)")
        data = 1.0 - data

        # Split and optionally downsample each channel
        channels = []
        for ch_idx, ch_config in enumerate(CHANNELS):
            ch = data[:, :, ch_idx]

            if scale < 1.0:
                zoom_factors = (target_h / ch.shape[0], target_w / ch.shape[1])
                ch = zoom(ch, zoom_factors, order=1)

            # Re-normalise to [0, 1]
            vmin, vmax = ch.min(), ch.max()
            if vmax - vmin > 1e-8:
                ch = (ch - vmin) / (vmax - vmin)
            ch = ch.astype(np.float32)

            channels.append(ch)
            aprint(
                f"  Ch{ch_idx} ({ch_config['name']}): {ch.shape}, "
                f"range [{ch.min():.3f}, {ch.max():.3f}]"
            )

        del data  # Free memory

    return channels, acquisition


def resolve_data() -> list[Path]:
    """Resolve the per-channel gsplat artifacts: cache -> in-repo copy -> Zenodo.

    Colormaps are assigned by POSITION downstream, so an out-of-order fetch
    would paint hematoxylin red. ``ensure_dataset`` promises exactly the
    manifest's file list, in manifest order; this makes the demo check that
    promise rather than depend on it silently — a rename or an added sidecar
    trips it just as a reordering does.
    """
    cache_paths = ensure_dataset(DATASET)
    if [p.name for p in cache_paths] != GSPLATS_FILES:
        raise RuntimeError(
            f"Manifest file list for {DATASET} is {[p.name for p in cache_paths]}, "
            f"which does not match the expected {GSPLATS_FILES} exactly."
        )
    return cache_paths


def local_fit_paths() -> list[Path] | None:
    """This machine's own earlier refit, or None if it is absent/incomplete.

    Paths, not ``GSplatData``: a ``--recompute`` writes the ``adaptive``
    topology, a ``kind=partition`` tree with no flat matrix form (which is also
    why ``load_local_fit_gsplats`` cannot serve this demo — see
    ``create_luxar_scene``). Consulted only when the manifest fetch came up
    empty, and BEFORE refitting, which is what makes the refit one-time.

    A ZIP header check catches the one cheap, unambiguous failure here: a
    truncated save from a Ctrl-C. Deeper validation would pay the whole graft,
    so structurally valid archives are left for ``add_gsplats_from_file`` to
    inspect rather than risking an unnecessary whole-slide refit.
    """
    paths = [local_fit_path(DATASET, name) for name in GSPLATS_FILES]
    return paths if all(zipfile.is_zipfile(path) for path in paths) else None


# =============================================================================
# Tiled GSplats Fitting
# =============================================================================


def fit_channel_tiled(
    image: np.ndarray,
    channel_name: str,
    cache_file: Path,
    acquisition: tuple | None = None,
) -> GSplatData:
    """Fit 2D gsplats to a single channel using tiled fitting.

    Uses Hann cosine apodization for seamless tile stitching across the
    full-resolution image.

    Args:
        image: 2D float32 image (H, W), normalised to [0, 1].
        channel_name: Name for logging.
        cache_file: Where to cache the fitted result.

    Returns:
        Fitted GSplatData with 2D centers in pixel coordinates.
    """
    from luxar.gsplats import fit_tiled

    global DEVICE
    if DEVICE is None:
        from luxar.demos import detect_device

        DEVICE = detect_device()

    h, w = image.shape
    aprint(f"Tiled fitting {channel_name}: {w}x{h} px")
    aprint(f"  Tile size: {TILE_SIZE}, overlap: {OVERLAP}")
    aprint(f"  Seeds/tile: {SEEDS_PER_TILE:,}, iters: {N_ITERS}")
    aprint(f"  Device: {DEVICE}")

    src_shape, src_dtype = acquisition or (None, None)
    result = fit_tiled(
        image,
        # One channel of the slide page as read; the fitted image is a resized,
        # normalized float32 copy of it. fit_tiled applies this to the MERGED
        # result -- the tiles themselves see crops.
        source_shape=src_shape,
        source_dtype=src_dtype,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        seeds=SEEDS_PER_TILE,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats across all tiles")

    # Cache result
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file}")
    save_with_lod(
        result,
        cache_file,
        recipe="adaptive",
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return result


def fit_all_channels(
    images: list[np.ndarray],
    acquisition: tuple | None = None,
) -> tuple[list[Path], list[GSplatData]]:
    """Fit 2D gsplats to all RGB channels using tiled fitting.

    Always fits: ``main`` has already tried the manifest fetch and this
    machine's own earlier refit (:func:`local_fit_paths`) by the time it gets
    here, so a "reuse the cache" branch would be dead code — and reading the
    cache back is not free anyway, since the artifact is a ``kind=partition``
    tree with no flat ``GSplatData`` form.

    Returns:
        ``(cache_paths, gsplats)`` — the written artifacts, and the in-memory
        fits the round-trip comparison renders from.
    """
    with asection("Tiled fitting of 2D GSplats per channel"):
        cache_paths: list[Path] = []
        gsplats_list = []

        for i, (image, ch_config) in enumerate(zip(images, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = local_fit_path(DATASET, GSPLATS_FILES[i])

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel_tiled(
                    image, ch_name, cache_file, acquisition=acquisition
                )
                cache_paths.append(cache_file)
                gsplats_list.append(gsplats)

        return cache_paths, gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    cache_paths: list[Path], output_path: Path | None = None
) -> Path:
    """Create Luxar scene with per-channel 2D gsplats as separate layers.

    Each channel is handed over as a **path**, not loaded into a ``GSplatData``
    first, because a ``--recompute`` writes the ``adaptive`` topology — spatial
    tiles under the recipe's shared per-tile cap, each picking its own detail
    level, which is what a 46000x33000 slide that is panned and zoomed rather
    than orbited wants — and that is a ``kind=partition`` tree with no flat
    matrix form, so ``add_gsplats_from_file`` is the only entry point that takes
    one whole.

    That entry point is deliberately topology-agnostic, which matters here
    because the record and a local recompute do not agree. The record's archives are four
    top-level parts carrying no ``child_`` detail levels
    (``scripts/demo_archive_characteristics.json`` records the topology,
    measured on the pinned ``sha256``), so a ``--recompute`` differs from
    them in tile count — four against roughly sixty — as much as in per-tile
    levels. ``add_gsplats_from_file`` routes the record partition through the
    graft and a local matrix-shaped result down the ordinary data path. To tell
    which generation a given cached file is, hash it against the record's
    ``sha256`` pin in ``data_manifest.json``; a local ``--recompute`` matches
    neither, which is itself the diagnosis. Byte size is only suggestive.

    Args:
        cache_paths: Per-channel ``.gsplats.zarr[.zip]`` artifacts, in channel
            order (red, green, blue).
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_2d_cmu1_pathology.luxar.zarr"

    # Channel -> colormap mapping
    CHANNEL_COLORMAPS = ["red", "green", "blue"]

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="px", display=True),
                    Dimension("y", unit="px", display=True),
                ]
            )
            # 2D data: start in orthographic mode with scale bar visible.
            # ACES, set explicitly (the house default; its filmic rolloff suits
            # the bright slide background). ACES does shift hues, so if faithful
            # H&E stain colour ever matters more than the filmic look here,
            # "None" is the documented alternative (#1459) — an exact
            # passthrough, as long as the render sits inside [0, 1].
            viewer_config = ViewerConfig(
                cinematic_mode=True,
                control_type="ortho",
                tone_mapping="ACES",
                # Preserve the projection-only scale bar and measured intensity.
                bloom_enabled=False,
                chromatic_lens_distortion_enabled=False,
                detector_noise_enabled=False,
                vignette_enabled=False,
                ui=UIConfig(show_scale_bar=True),
            )
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=viewer_config,
            )

            scene.attrs["title"] = "GSplats 2D: Whole-Slide Pathology (CMU-1, H&E)"
            scene.attrs["description"] = f"""
2D Gaussian Splatting - Whole-Slide Pathology Image
======================================================

H&E-stained human tissue section from digital pathology,
represented as 2D Gaussian splats with per-channel layers.

Fitted using tiled Gaussian splatting with Hann cosine apodization
for seamless stitching across {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels.

Data Source:
  - OpenSlide test data: CMU-1.svs (CC0 public domain)
  - Aperio ScanScope, 20x magnification
  - Resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels (1.5 gigapixels)

Tiled Fitting:
  - Tile size: {TILE_SIZE} px, overlap: {OVERLAP} px
  - Seeds per tile: {SEEDS_PER_TILE:,}, iterations: {N_ITERS:,}

Channels (inverted brightfield RGB, each a separate layer):
  - Red:   Eosin / cytoplasm / connective tissue
  - Green: Mixed contribution from both stains
  - Blue:  Hematoxylin / nuclei / basophilic structures

Controls:
  - Press L to open the Layers panel
  - Click eye icon to toggle channel visibility
  - Adjust [min, max] display range per channel
  - Mouse drag to pan, scroll to zoom
            """

            # Coordinates stay in slide pixels — no re-centring on a shared
            # centroid, which this demo used to do "so channels stay aligned".
            # They are aligned by construction: all three are fits of the SAME
            # image on the same pixel grid, so subtracting one common offset
            # from all three never changed their relative position. Framing is
            # unaffected too (the viewer targets the bounding-box centre, not
            # the origin), and `unit="px"` now reads as true slide coordinates.
            # Add each channel as a layer-enabled gsplats node
            for i, (cache_path, ch_config) in enumerate(
                zip(cache_paths, CHANNELS[: len(cache_paths)])
            ):
                ch_name = ch_config["name"]
                colormap = CHANNEL_COLORMAPS[i]

                with asection(f"Adding {ch_name} (layer)"):
                    scene.add_gsplats_from_file(
                        name=f"gsplats_{colormap}",
                        path=str(cache_path),
                        # No `dim_order`: grafting a multi-part subtree refuses
                        # it (the file is already a full node tree, so there is
                        # nothing left to remap). Nothing is lost — the old
                        # ["x", "y"] was the identity anyway, the scene
                        # declaring x then y in exactly that column order.
                        opacity=1.0,
                        # Additive fits both the overlapping-layer rule and
                        # this demo's strictly 2D reconstruction: there is no
                        # depth structure for volumetric to resolve. Viewed
                        # face-on (the default under this
                        # demo's control_type="ortho") every splat shares one
                        # view-depth plane and the depth sorter takes its
                        # identity-ordering branch, so volumetric would just
                        # composite in storage order. Additive blending is
                        # order-independent, so for a 2D fit the additive sum
                        # IS the reconstruction regardless of the camera.
                        blending_mode="additive",
                        layer=True,
                        colormap=colormap,
                    )
                    # Not necessarily "grafted": a flat generation takes the
                    # ordinary data path instead (see this function's docstring).
                    aprint(f"  Added {cache_path.name} with colormap='{colormap}'")

            # --- Overlays ---
            # Title
            scene.add_text(
                "Whole-Slide Pathology (CMU-1)",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            add_demo_caption(
                scene,
                "46K\u00d732K \u2022 H&E stain \u2022 20x",
                DEMO_META.get("citation"),
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================


def show_roundtrip_comparison(
    images: list[np.ndarray],
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed 2D images side by side."""
    try:
        plt = require_module("matplotlib.pyplot")
    except MissingDependencyError as exc:
        aprint(f"Skipping --show-roundtrip: {exc}")
        return

    n_channels = len(images)

    with asection("Round-trip reconstruction comparison"):
        reconstructions = []
        for i, (image, gsplats, ch_config) in enumerate(
            zip(images, gsplats_list, CHANNELS[:n_channels])
        ):
            with asection(f"Rendering Ch{i}: {ch_config['name']}"):
                recon = gsplats.render_to_volume(shape=image.shape, device=DEVICE)
                reconstructions.append(recon)
                mse = float(np.mean((image - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(
            n_channels, 3, figsize=(14, 4.5 * n_channels), squeeze=False
        )

        for i, (image, recon, ch_config) in enumerate(
            zip(images, reconstructions, CHANNELS[:n_channels])
        ):
            diff = np.abs(image - recon)

            mse = float(np.mean((image - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original — {ch_config['name']}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff, cmap="inferno", vmin=0, vmax=0.3)
            axes[i, 2].set_title("|Difference|")
            axes[i, 2].axis("off")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Round-Trip Comparison  "
            f"({sum(len(g.amplitudes) for g in gsplats_list):,} total splats)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


# =============================================================================
# Main
# =============================================================================


def resolve_or_local() -> list[Path] | None:
    """The manifest fetch, then this machine's own earlier refit; None ⇒ build it.

    ``None`` under ``--recompute`` too, which is how ``main`` reaches its
    fit-from-scratch branch.

    Only ``DatasetUnavailable`` falls through to the local door — the narrow
    "these bytes are not obtainable from anywhere yet" case. An unknown file
    name, a missing packaged manifest or an in-repo copy failing its sha256 are
    faults, and must not be disguised as a routine multi-minute refit.
    """
    if RECOMPUTE:
        return None
    try:
        return resolve_data()
    except DatasetUnavailable as exc:
        aprint(f"Manifest fetch unavailable ({exc}).")
        cache_paths = local_fit_paths()
        if cache_paths is not None:
            aprint(f"Reusing this machine's own refit in {cache_paths[0].parent}")
        return cache_paths


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 2D Whole-Slide Pathology (CMU-1, H&E)")
    aprint("=" * 70)
    aprint("Tiled per-channel RGB fitting + per-channel layers + Web viewer")
    if TARGET_SIZE > 0:
        aprint(f"Target size: {TARGET_SIZE} px (longest axis)")
    else:
        aprint(f"Full resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} px")
    aprint(
        f"Tile: {TILE_SIZE} px, overlap: {OVERLAP} px, seeds/tile: {SEEDS_PER_TILE:,}"
    )
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_2d_cmu1_pathology.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Manifest-driven fetch (checksum-verified cache -> in-repo -> Zenodo).
    # PATHS, not GSplatData: which topology the artifacts carry depends on which
    # generation resolves, and a partition has no flat form (see
    # `create_luxar_scene`).
    images = None
    gsplats_list: list[GSplatData] = []

    cache_paths = resolve_or_local()

    if cache_paths is None:
        # --recompute path (or no data to be had): download raw data, fit from
        # scratch, and cache the fits in the local-fit namespace.
        warn_if_no_cuda_gpu()
        images, acquisition = load_cmu1_image()

        if len(images) < N_CHANNELS:
            aprint(f"Error: Need {N_CHANNELS} channels, got {len(images)}")
            return

        # Tiled fitting per channel
        cache_paths, gsplats_list = fit_all_channels(images, acquisition=acquisition)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if images is not None:
            show_roundtrip_comparison(images, gsplats_list)
        else:
            aprint(
                "Cannot show round-trip: original images not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Create scene with per-channel layers
    scene_path = create_luxar_scene(cache_paths, output_path)

    # Summary
    if images is not None:
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        total_pixels = sum(img.size for img in images)
        pixel_bytes = total_pixels * 4  # float32
        # Floats per 2D splat: d + d*(d+1)/2 + 2 + 3 (colors) = 2 + 3 + 2 + 3 = 10
        splats_bytes = total_splats * 10 * 4
        compression = pixel_bytes / splats_bytes if splats_bytes > 0 else 0

        aprint("")
        aprint("=" * 70)
        aprint("2D Tiled Multi-Channel Compression Summary")
        aprint("=" * 70)
        aprint(f"Channels:          {N_CHANNELS}")
        aprint(f"Total pixels:      {total_pixels:,}")
        aprint(f"Total splats:      {total_splats:,}")
        aprint(f"Raw pixel data:    {pixel_bytes / (1024 * 1024):.1f} MB")
        aprint(f"Splat data size:   {splats_bytes / 1024:.1f} KB")
        aprint(f"Compression ratio: {compression:.1f}:1")
        aprint("=" * 70)

    # Launch viewer
    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
