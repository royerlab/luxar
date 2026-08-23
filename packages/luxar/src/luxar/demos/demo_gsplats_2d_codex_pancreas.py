#!/usr/bin/env python3
"""GSplats Demo: 2D 12-Channel CODEX Pancreas (Multiplexed Fluorescence)

Visualises a 12-channel multiplexed immunofluorescence image of human pancreas
tissue as 2D Gaussian splats with per-channel layers, using tiled fitting.

================================================================================
2D GAUSSIAN SPLATTING — CODEX MULTIPLEXED PANCREAS (12 CHANNELS, TILED)
================================================================================

This demo shows the complete workflow for tiled multi-channel 2D Gaussian
splatting on a real multiplexed immunofluorescence dataset:
- Downloading 12-channel CODEX data from Zenodo
- Extracting and normalising each fluorescence channel
- Tiled fitting of 2D Gaussian splats per channel (Hann cosine apodization)
- Building an ``adaptive`` LOD topology per channel: spatial tiles, each with
  its own substitutive levels, each level with an additive streaming ladder
- Grafting each channel into the scene as a separate layer with colormap
- Visualising in the Luxar web viewer

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  Zenodo record 7742474
URL:    https://zenodo.org/records/7742474
Format: 12 separate 16-bit TIFF files in a ZIP archive
Size:   25,816 x 18,440 pixels (476 megapixels) per channel
Pixel:  0.325 x 0.325 um
Depth:  16-bit unsigned integer

Imaging:
--------
Method:     CODEX (CO-Detection by indEXing) multiplexed immunofluorescence
Microscope: Leica system with CODEX Processor v1.7.0.6
Specimen:   Human pancreas 5 um tissue section

Channels (12 protein markers):
-------------------------------
  Hoechst      — Nuclear stain (DNA, all cells)
  CGC          — Chromogranin C (endocrine granules)
  Beta-Catenin — Cell-cell junctions (Wnt signalling)
  CPEP         — C-peptide (insulin-producing beta cells)
  VIM          — Vimentin (mesenchymal / stromal cells)
  KRT19        — Keratin 19 (ductal epithelium)
  PECAM-1      — CD31 (endothelial cells / vasculature)
  SST          — Somatostatin (delta cells)
  E-Cadherin   — Epithelial cell junctions
  CHGA         — Chromogranin A (neuroendocrine)
  ACTA2        — Alpha smooth muscle actin
  IAPP         — Islet amyloid polypeptide (beta cells)

License:
--------
Creative Commons Attribution 4.0 International (CC BY 4.0)

WORKFLOW:
=========

1. **Download** ZIP archive from Zenodo (~5.9 GB)
2. **Extract** 12 TIFF channel files
3. **Normalise** each channel (uint16 -> float32 [0, 1])
4. **Tiled fit** 2D Gaussian splats per channel with GPU acceleration
5. **Ladder** each fit with the ``adaptive`` recipe and cache it as a
   ``kind=partition`` tree
6. **Graft** each channel into the scene as a separate layer with colormap
7. **Visualise** in the Luxar web viewer

USAGE:
======
    python demo_gsplats_2d_codex_pancreas.py [OPTIONS]

Options:
    --recompute:        Force re-fitting from scratch
    --no-serve:         Generate scene without launching viewer
    --serve-only:       Just serve a previously generated scene
    --show-roundtrip:   Show matplotlib comparison of original vs reconstructed images
    --target-size=N:    Downsample target for longest axis (default: 0 = full res)
    --tile-size=N:      Tile size in pixels for tiled fitting (default: 4096)
    --overlap=N:        Tile overlap in pixels (default: 512)
    --seeds-per-tile=N: Gaussian seeds per tile (default: 255000, per tile)

By default, the demo works at full resolution (25,816 x 18,440 pixels)
using tiled fitting. Each channel is fitted and cached independently,
so interrupted runs can resume from where they left off.

Output:
    - Scene saved to:  datasets/demos/gsplats_2d_codex_pancreas.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_2d_codex_pancreas",
    "title": "2D 12-Channel CODEX Pancreas (Multiplexed Fluorescence)",
    "description": "A 12-channel CODEX immunofluorescence image of human pancreas as 2D Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 5900,
        "compute": "heavy",
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_codex_pancreas"],
    "outputs": ["gsplats_2d_codex_pancreas"],
    "citation": {
        "short": "Björklund et al. 2023",
        "doi": "10.5281/zenodo.7742474",
        "license": "CC BY 4.0",
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
    MissingDependencyError,
    add_demo_caption,
    launch_viewer,
    parse_demo_flags,
    require_module,
    warn_if_no_cuda_gpu,
)
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.download import robust_download
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Zenodo download
DOWNLOAD_URL = "https://zenodo.org/records/7742474/files/human_pancreas_codex.zip"
EXPECTED_FILE_SIZE = 6_294_399_806  # 5.9 GB

# Image specs
ORIGINAL_WIDTH = 25_816
ORIGINAL_HEIGHT = 18_440
PIXEL_SIZE = 0.325  # um per pixel

# 12 fluorescence channels with biologically meaningful colours
N_CHANNELS = 12
CHANNELS = [
    {
        "file": "reg001_cyc001_ch001_Hoechst.tiff",
        "name": "Hoechst (Nuclei)",
        "color": (0.0, 0.4, 1.0),
        "colormap": "blue",
    },
    {
        "file": "reg001_cyc002_ch003_CGC.tiff",
        "name": "CGC (Endocrine)",
        "color": (1.0, 0.8, 0.0),
        "colormap": "yellow",
    },
    {
        "file": "reg001_cyc002_ch004_BETA-CATENIN.tiff",
        "name": "Beta-Catenin (Junctions)",
        "color": (0.0, 1.0, 0.6),
        "colormap": "green",
    },
    {
        "file": "reg001_cyc003_ch003_CPEP.tiff",
        "name": "CPEP (Beta Cells)",
        "color": (1.0, 0.0, 0.4),
        "colormap": "red",
    },
    {
        "file": "reg001_cyc003_ch004_VIM.tiff",
        "name": "VIM (Mesenchymal)",
        "color": (0.6, 0.0, 1.0),
        "colormap": "magenta",
    },
    {
        "file": "reg001_cyc004_ch003_KRT19.tiff",
        "name": "KRT19 (Ductal)",
        "color": (0.0, 1.0, 1.0),
        "colormap": "cyan",
    },
    {
        "file": "reg001_cyc005_ch002_PECAM-1.tiff",
        "name": "PECAM-1 (Vasculature)",
        "color": (1.0, 0.0, 0.0),
        "colormap": "red",
    },
    {
        "file": "reg001_cyc005_ch003_SST.tiff",
        "name": "SST (Delta Cells)",
        "color": (1.0, 0.5, 0.0),
        "colormap": "orange",
    },
    {
        "file": "reg001_cyc006_ch003_E-CADHERIN.tiff",
        "name": "E-Cadherin (Epithelial)",
        "color": (0.5, 1.0, 0.0),
        "colormap": "green",
    },
    {
        "file": "reg001_cyc006_ch004_CHGA.tiff",
        "name": "CHGA (Endocrine)",
        "color": (1.0, 1.0, 0.0),
        "colormap": "yellow",
    },
    {
        "file": "reg001_cyc007_ch002_ACTA2.tiff",
        "name": "ACTA2 (Smooth Muscle)",
        "color": (1.0, 0.0, 1.0),
        "colormap": "magenta",
    },
    {
        "file": "reg001_cyc007_ch003_IAPP.tiff",
        "name": "IAPP (Beta Cells)",
        "color": (0.0, 1.0, 0.3),
        "colormap": "green",
    },
]

# Tiled fitting parameters
TILE_SIZE = 4096
OVERLAP = 512
SEEDS_PER_TILE = 500000
N_ITERS = 6_000

# Cache location
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_codex_pancreas"

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Parse optional CLI overrides
TARGET_SIZE = 0  # Default: full resolution
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


def ensure_extracted() -> Path:
    """Download the ZIP and extract TIFF files to cache.

    Returns:
        Path to the directory containing extracted TIFF files.
    """
    extract_dir = CACHE_DIR / "tiffs"
    zip_path = CACHE_DIR / "human_pancreas_codex.zip"

    # Check if all TIFFs already extracted
    all_present = extract_dir.exists() and all(
        (extract_dir / ch["file"]).exists() for ch in CHANNELS
    )
    if all_present:
        aprint("All TIFF files already extracted")
        return extract_dir

    # Download if needed
    robust_download(
        DOWNLOAD_URL,
        zip_path,
        expected_size=EXPECTED_FILE_SIZE,
    )

    # Extract TIFFs (skip __MACOSX junk)
    with asection("Extracting TIFF files"):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            for ch in CHANNELS:
                fname = ch["file"]
                target = extract_dir / fname
                if target.exists():
                    aprint(f"  Already extracted: {fname}")
                    continue
                aprint(f"  Extracting {fname}...")
                with zf.open(fname) as src, open(target, "wb") as dst:
                    import shutil

                    shutil.copyfileobj(src, dst)
        aprint(f"Extracted {N_CHANNELS} channels to {extract_dir}")

    return extract_dir


def load_channel(tiff_dir: Path, ch_config: dict) -> np.ndarray:
    """Load a single fluorescence channel from TIFF.

    Args:
        tiff_dir: Directory containing extracted TIFF files.
        ch_config: Channel config dict with 'file' and 'name' keys.

    Returns:
        2D float32 image (H, W), normalised to [0, 1].
    """
    tifffile = require_module("tifffile")

    from scipy.ndimage import zoom

    fname = ch_config["file"]
    path = tiff_dir / fname

    data = tifffile.imread(str(path))
    aprint(f"  Loaded {fname}: {data.shape}, dtype={data.dtype}")

    # Convert to float32 and normalise to [0, 1]
    data = data.astype(np.float32)
    vmax = data.max()
    if vmax > 0:
        data = data / vmax
    # No inversion — fluorescence: bright = signal

    # Optional downsampling
    h, w = data.shape[:2]
    longest = max(h, w)
    if TARGET_SIZE > 0 and longest > TARGET_SIZE:
        scale = TARGET_SIZE / longest
        target_h = int(h * scale)
        target_w = int(w * scale)
        zoom_factors = (target_h / h, target_w / w)
        data = zoom(data, zoom_factors, order=1)

    data = data.astype(np.float32)
    aprint(
        f"  {ch_config['name']}: {data.shape}, "
        f"range [{data.min():.3f}, {data.max():.3f}]"
    )

    return data


# =============================================================================
# Tiled GSplats Fitting
# =============================================================================


def channel_cache_path(channel_index: int) -> Path:
    """Return the current-frame adaptive cache path for one CODEX channel."""
    return CACHE_DIR / f"codex_ch{channel_index:02d}.v2.gsplats.zarr.zip"


def fit_channel_tiled(
    image: np.ndarray,
    channel_name: str,
    cache_file: Path,
) -> GSplatData:
    """Fit 2D gsplats to a single channel using tiled fitting.

    Args:
        image: 2D float32 image (H, W), normalised to [0, 1].
        channel_name: Name for logging.
        cache_file: Where to cache the fitted result.

    Returns:
        Fitted GSplatData with 2D centers.
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

    # Fit the TRANSPOSED plane, so the fitted centers come out in the scene's
    # own (x, y) column order instead of the array's (row, col).
    #
    # This used to be a `dim_order=["y", "x"]` on the scene adder, which is not
    # available any more: the cached artifact is now a `kind=partition` tree and
    # `add_gsplats_from_file` refuses dim_order when grafting one (there is no
    # single matrix left to remap — see `test_grafting_a_partition_rejects_dim_order`).
    # Doing it here instead of permuting the fit afterwards keeps centers,
    # Cholesky factors and the recorded source grid in ONE frame: a covariance
    # permutation would have to re-decompose every splat and would leave the
    # provenance shape describing the other orientation. The pixel grid is
    # isotropic (0.325 um on both axes), so transposing costs no metric fidelity.
    image = np.ascontiguousarray(image.T)

    result = fit_tiled(
        image,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        seeds=SEEDS_PER_TILE,
        n_iters=N_ITERS,
        device=DEVICE,
        voxel_size=PIXEL_SIZE,
        verbose=True,
        enable_dynamic_ops=True,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats across all tiles")

    # Cache result. `adaptive` is the recipe a 476-megapixel plane wants: the
    # slide is panned and zoomed rather than orbited, so most of it is off
    # screen most of the time and a per-tile level beats one global one. BSP
    # splitting needs only two spatial axes, so a strictly 2D fit tiles the same
    # way a volume does. What lands on disk is a `kind=partition` tree: spatial
    # tiles, each carrying its own substitutive levels, each level carrying an
    # additive ladder underneath.
    aprint(f"  Caching to {cache_file.name}")
    save_with_lod(
        result,
        cache_file,
        recipe="adaptive",
        # MPS lacks the float64 the substitutive reduction wants and warns as it
        # falls back; CPU is the same answer without the noise. CUDA keeps the
        # reduction on the GPU that just did the fit.
        device="cpu" if DEVICE == "mps" else DEVICE,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return result


def fit_all_channels(tiff_dir: Path) -> tuple[list[Path], list[GSplatData | None]]:
    """Load and fit all 12 channels, one at a time to save memory.

    Returns the per-channel cache PATHS alongside the fitted data. The paths are
    what the scene is built from — an ``adaptive`` artifact is a
    ``kind=partition`` tree with no flat matrix form, so it is grafted with
    :meth:`~luxar.core.group.Group.add_gsplats_from_file` rather than loaded
    into a ``GSplatData`` first.

    The data slots exist only for ``--show-roundtrip``, and only for channels
    fitted in THIS run: a channel restored from cache yields ``None`` there,
    because the flat matrix loader refuses a partition store outright. That
    keeps the resume behaviour this demo advertises — a 12-channel GPU fit over
    a 5.9 GB download is routinely interrupted — without pretending the cache
    round-trips to a matrix.
    """
    with asection(f"Tiled fitting of {N_CHANNELS} fluorescence channels"):
        cache_paths: list[Path] = []
        gsplats_list: list[GSplatData | None] = []

        for i, ch_config in enumerate(CHANNELS):
            ch_name = ch_config["name"]
            cache_file = channel_cache_path(i)

            # Check per-channel cache
            if cache_file.exists() and not RECOMPUTE:
                with asection(f"Channel {i}: {ch_name} (cached)"):
                    aprint(f"Reusing cached fit: {cache_file.name}")
                    cache_paths.append(cache_file)
                    gsplats_list.append(None)
                    continue

            with asection(f"Channel {i}: {ch_name}"):
                # Load one channel at a time (908 MB each)
                image = load_channel(tiff_dir, ch_config)
                gsplats = fit_channel_tiled(image, ch_name, cache_file)
                cache_paths.append(cache_file)
                gsplats_list.append(gsplats)
                del image  # Free 908 MB

        return cache_paths, gsplats_list


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(
    cache_paths: list[Path], output_path: Path | None = None
) -> Path:
    """Create Luxar scene with per-channel 2D gsplat layers.

    Each channel is added as a separate gsplats node with ``layer=True``
    and a ``colormap``, so the viewer's Layers panel (press L) provides
    per-channel visibility, display-range, gamma, and colormap controls.

    Each channel is **grafted from its artifact** rather than loaded into a
    ``GSplatData`` first. The fit writes the ``adaptive`` topology — spatial
    tiles, each picking its own detail level, with an additive ladder under
    every level — which is what a 476-megapixel plane that is panned and zoomed
    rather than orbited wants. That is a ``kind=partition`` tree with no flat
    matrix form, so ``add_gsplats_from_file`` is the entry point that grafts one
    whole; ``add_gsplats`` would flatten all of it back to a single leaf.

    Args:
        cache_paths: Per-channel ``.gsplats.zarr.zip`` artifacts, in channel
            order.
        output_path: Output .zarr path (default: demos output dir).

    Returns:
        Path to saved scene.
    """
    if output_path is None:
        output_path = get_demos_output_dir() / "gsplats_2d_codex_pancreas.luxar.zarr"

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            dims = Dimensions(
                [
                    Dimension("x", unit="um", display=True),
                    Dimension("y", unit="um", display=True),
                ]
            )
            # 2D data: start in orthographic mode with scale bar visible.
            # ACES, set explicitly (the house default). It shifts LUT hues a
            # little; "None" is the alternative if exact per-marker colour
            # fidelity ever matters more than the filmic look (#1459) — an
            # exact passthrough, as long as the render is inside [0, 1].
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
                dimensions=dims,
                viewer_config=viewer_config,
                citation=DEMO_META["citation"],
            )

            ch_list = "\n".join(f"  - {ch['name']}" for ch in CHANNELS)
            scene.attrs["title"] = "GSplats 2D: CODEX Pancreas (12-Channel Multiplexed)"
            scene.attrs["description"] = f"""
2D Gaussian Splatting - CODEX Multiplexed Pancreas
======================================================

12-channel multiplexed immunofluorescence of human pancreas tissue,
represented as 2D Gaussian splats with per-channel layers.

Fitted using tiled Gaussian splatting with Hann cosine apodization
for seamless stitching across {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels.

Data Source:
  - Zenodo record 7742474 (CC BY 4.0)
  - CODEX multiplexed imaging, Leica microscope
  - Resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} pixels (476 MP)
  - Pixel size: {PIXEL_SIZE} um

Tiled Fitting:
  - Tile size: {TILE_SIZE} px, overlap: {OVERLAP} px
  - Seeds per tile: {SEEDS_PER_TILE:,}, iterations: {N_ITERS:,}

Channels ({N_CHANNELS} protein markers):
{ch_list}

Controls:
  - Press L to open the Layers panel
  - Toggle visibility, adjust display range, gamma, colormap per channel
  - Mouse drag to pan, scroll to zoom
            """

            # Coordinates stay in slide micrometres — no re-centring on a shared
            # centroid, which this demo used to do "so channels stay aligned".
            # They are aligned by construction: all twelve are fits of the SAME
            # 25,816 x 18,440 pixel grid, so subtracting one common offset from
            # all twelve never changed their relative position. Framing is
            # unaffected too (the viewer targets the bounding-box centre, not
            # the origin), and `unit="um"` now reads as true slide coordinates.
            # The old per-channel `scale_intensity(0.1)` is gone with it: a
            # colormapped gsplat layer is windowed by the node's own
            # `amplitude_data_range`, so a global amplitude scale cancels and
            # the render is identical either way. Both were transforms on a
            # loose GSplatData, which the graft below no longer has in hand.

            # Add each channel as a layer-enabled gsplats node
            for i, (cache_path, ch_config) in enumerate(zip(cache_paths, CHANNELS)):
                ch_name = ch_config["name"]
                colormap = ch_config["colormap"]

                with asection(f"Adding {ch_name} (layer)"):
                    scene.add_gsplats_from_file(
                        name=f"gsplats_{ch_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
                        path=str(cache_path),
                        # No `dim_order`: grafting a multi-part subtree refuses
                        # it. The fit already ran on the transposed plane, so
                        # the stored columns are the scene's (x, y) order.
                        opacity=1.0,
                        # Stays additive while the other bioimaging gsplat
                        # demos are volumetric: this fit is strictly 2D, so
                        # there is no depth structure for volumetric to
                        # resolve. Viewed face-on (the default under this
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
                    aprint(f"  Grafted {cache_path.name} with colormap='{colormap}'")

            # --- Overlays ---
            # Title
            scene.add_text(
                "CODEX Pancreas",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info
            add_demo_caption(
                scene, "12-channel multiplexed fluorescence", DEMO_META.get("citation")
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================

# Max channels to show in round-trip comparison (full set is 12, which is huge)
_ROUNDTRIP_MAX_CHANNELS = 3


def show_roundtrip_comparison(
    tiff_dir: Path,
    gsplats_list: list[GSplatData | None],
) -> None:
    """Show original vs round-trip reconstructed 2D images side by side.

    Reloads a subset of channels from disk to avoid keeping all 12 in memory.

    Only channels fitted in THIS run can be shown: a channel restored from cache
    has no in-memory ``GSplatData`` (its artifact is a partition tree, which does
    not read back flat), so it is skipped rather than faked. Re-run that channel
    with ``--recompute`` to include it.
    """
    try:
        plt = require_module("matplotlib.pyplot")
    except MissingDependencyError as exc:
        aprint(f"Skipping --show-roundtrip: {exc}")
        return

    fitted = [(i, g) for i, g in enumerate(gsplats_list) if g is not None]
    n_cached = len(gsplats_list) - len(fitted)
    if n_cached:
        aprint(
            f"--show-roundtrip: skipping {n_cached} channel(s) restored from "
            "cache (no in-memory fit to compare against)"
        )
    if not fitted:
        aprint("--show-roundtrip: nothing was fitted this run, nothing to compare")
        return

    shown = fitted[:_ROUNDTRIP_MAX_CHANNELS]
    n_show = len(shown)

    with asection(
        f"Round-trip reconstruction comparison ({n_show}/{len(gsplats_list)} channels)"
    ):
        images = []
        reconstructions = []
        titles = []
        for ch_index, gsplats in shown:
            ch_config = CHANNELS[ch_index]
            titles.append(ch_config["name"])

            with asection(f"Ch{ch_index}: {ch_config['name']}"):
                image = load_channel(tiff_dir, ch_config)
                # The fit ran on the transposed plane (scene (x, y) order), so
                # render in that frame and transpose back to compare against the
                # image as read.
                recon = gsplats.render_to_volume(
                    shape=(image.shape[1], image.shape[0]), device=DEVICE
                ).T
                images.append(image)
                reconstructions.append(recon)

                mse = float(np.mean((image - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(n_show, 3, figsize=(14, 4.5 * n_show), squeeze=False)

        for i in range(n_show):
            image = images[i]
            recon = reconstructions[i]
            diff = np.abs(image - recon)

            mse = float(np.mean((image - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[i, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original — {titles[i]}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff, cmap="inferno", vmin=0, vmax=0.3)
            axes[i, 2].set_title("|Difference|")
            axes[i, 2].axis("off")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Round-Trip Comparison ({n_show} of {N_CHANNELS} channels)  "
            f"({sum(len(g.amplitudes) for _, g in fitted):,} splats fitted this run)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 2D CODEX Pancreas (12-Channel Multiplexed)")
    aprint("=" * 70)
    aprint("Tiled per-channel fitting + per-channel layers + Web viewer")
    if TARGET_SIZE > 0:
        aprint(f"Target size: {TARGET_SIZE} px (longest axis)")
    else:
        aprint(f"Full resolution: {ORIGINAL_WIDTH:,} x {ORIGINAL_HEIGHT:,} px")
    aprint(f"Channels: {N_CHANNELS}")
    aprint(
        f"Tile: {TILE_SIZE} px, overlap: {OVERLAP} px, seeds/tile: {SEEDS_PER_TILE:,}"
    )
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_2d_codex_pancreas.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Download and extract
    warn_if_no_cuda_gpu()
    tiff_dir = ensure_extracted()

    # Tiled fitting per channel (one at a time to save memory)
    cache_paths, gsplats_list = fit_all_channels(tiff_dir)

    if len(cache_paths) < N_CHANNELS:
        aprint(f"Error: Need {N_CHANNELS} channels, got {len(cache_paths)}")
        return

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        show_roundtrip_comparison(tiff_dir, gsplats_list)

    # Create scene by grafting each channel's partition tree
    scene_path = create_luxar_scene(cache_paths, output_path)

    # Summary. Splat counts come from the in-memory fits, so a run that reused
    # cached channels can only account for the ones it fitted itself — the
    # cached artifacts are partition trees and counting them would mean opening
    # each one. The ratio is kept apples-to-apples by scaling the pixel side to
    # the same set of channels.
    counted = [g for g in gsplats_list if g is not None]
    n_counted = len(counted)
    total_splats = sum(len(g.amplitudes) for g in counted)
    total_pixels = ORIGINAL_WIDTH * ORIGINAL_HEIGHT * n_counted
    pixel_bytes = total_pixels * 2  # uint16 original
    # Floats per 2D splat: d + d*(d+1)/2 + 2 + 3 (colors) = 2 + 3 + 2 + 3 = 10
    splats_bytes = total_splats * 10 * 4
    compression = pixel_bytes / splats_bytes if splats_bytes > 0 else 0

    aprint("")
    aprint("=" * 70)
    aprint("12-Channel 2D Tiled Compression Summary")
    aprint("=" * 70)
    aprint(f"Channels:          {N_CHANNELS} ({n_counted} fitted this run)")
    aprint(f"Pixels/channel:    {ORIGINAL_WIDTH * ORIGINAL_HEIGHT:,}")
    if n_counted:
        aprint(f"Total pixels:      {total_pixels:,}")
        aprint(f"Total splats:      {total_splats:,}")
        aprint(f"Raw data (uint16): {pixel_bytes / (1024**2):.1f} MB")
        aprint(f"Splat data:        {splats_bytes / (1024**2):.1f} MB")
        aprint(f"Compression ratio: {compression:.1f}:1")
    else:
        aprint("All channels reused from cache — no fresh figures to report.")
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
