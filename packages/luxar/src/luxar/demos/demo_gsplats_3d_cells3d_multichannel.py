#!/usr/bin/env python3
"""GSplats Demo: 3D Multi-Channel Cells (scikit-image cells3d, Layers + BOP LUTs)

Two-channel scikit-image ``cells3d`` fluorescence volume (cell membranes +
nuclei) fitted per channel as 3D Gaussian splats and shown as independent,
toggleable **layers**, each coloured by a BOP (Blue-Orange-Purple) microscopy
LUT. Fully self-contained — skimage downloads the sample on first use — so it is
the cheapest end-to-end gsplat demo to run from scratch.

================================================================================
LAYERS + BOP LUT SHOWCASE
================================================================================

Each channel is fitted independently as 3D splats, centered on a shared
amplitude-weighted centroid so the channels stay aligned, and added as a
``layer=True`` gsplats node. The viewer's Layers panel (press **L**) exposes a
per-channel visibility toggle, display-range window, gamma, and blending mode.
Colours come from Luxar's built-in BOP LUTs (``bop_orange`` for membranes,
``bop_blue`` for nuclei), applied at display time so you can switch colormaps
interactively.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source: scikit-image sample data (``skimage.data.cells3d()``)
Shape: (60, 2, 256, 256) — (Z, Channel, Y, X), uint16
Channel 0: Cell membranes
Channel 1: Cell nuclei (fluorescent stain)
Origin: Allen Institute for Cell Science

How to Cite:
------------
scikit-image: image processing in Python.
van der Walt et al. (2014). PeerJ 2:e453. DOI: 10.7717/peerj.453

WORKFLOW:
=========

1. **Load** cells3d from scikit-image (60 × 2 × 256 × 256)
2. **Fit** each channel independently as 3D Gaussian splats
3. **Create 3D scene** with per-channel layer nodes
4. **Add splats** with ``layer=True`` and a BOP LUT per channel
5. **Visualize** — toggle layers in the Layers panel (press L)

USAGE:
======
    python demo_gsplats_3d_cells3d_multichannel.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:  Force re-fitting from scratch (requires GPU)
    --no-serve:   Don't auto-launch viewer after scene creation
    --serve-only: Skip loading/fitting, just serve existing scene

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch.

Output:
    - Scene saved to: demos/gsplats_3d_cells3d_multichannel.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_3d_cells3d_multichannel",
    "title": "Cells3D Multichannel Microscopy",
    "description": "Two-channel scikit-image cells3d volume (membranes + nuclei) as toggleable gsplat layers.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 1,
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["gsplats_cells3d"],
    "outputs": ["gsplats_3d_cells3d_multichannel"],
    # scikit-image ships the sample, but the images are the Allen Institute's
    # (see the Dataset block above and skimage's own cells3d docstring). The
    # van der Walt PeerJ paper credits the LIBRARY, so citing it here would
    # attribute someone else's microscopy to the software that loads it; the
    # header keeps that software citation where it belongs.
    "citation": {
        "short": "Allen Institute for Cell Science (scikit-image cells3d)",
        "ref": "Allen Institute for Cell Science",
    },
}


import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    DatasetUnavailable,
    add_demo_caption,
    launch_viewer,
    load_dataset_gsplats,
    load_local_fit_gsplats,
    local_fit_path,
    parse_demo_flags,
    require_module,
    stamp_input_digests,
    warn_if_no_cuda_gpu,
)
from luxar.demos._lod_policy import save_with_lod
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 25000

# Voxel spacing (Z, Y, X) in micrometres for cells3d
# Original: (0.29, 0.065, 0.065) µm, 4x downsampled in Y/X → (0.29, 0.26, 0.26) µm
VOXEL_SIZE_ZYX = (0.29, 0.26, 0.26)

# Channel configuration — each channel is fitted separately and shown as a
# toggleable layer coloured by a BOP (Blue-Orange-Purple) microscopy LUT.
# The viewer applies the colormap at display time (interactive switching).
CHANNELS = [
    # `window` is the Layers panel's DISPLAY RANGE, hand-tuned on the hosted
    # scene (2026-09-10) on top of LAYER_INTENSITY; authored as
    # intensity = 1/(hi-lo), offset = -lo/(hi-lo) (see `window_attrs`). The
    # non-zero floors drop the residual haze under each channel.
    {
        "index": 0,
        "name": "Membranes",
        "colormap": "bop_orange",
        "window": (0.006, 0.053),
    },
    {"index": 1, "name": "Nuclei", "colormap": "bop_blue", "window": (0.009, 0.114)},
]


def window_attrs(window: tuple[float, float]) -> dict[str, float]:
    """The intensity/offset pair a Layers-panel window ``[lo, hi]`` is stored as."""
    lo, hi = window
    return {"intensity": 1.0 / (hi - lo), "offset": -lo / (hi - lo)}


# Per-channel brightness multiplier applied before writing. Kept conservative
# because the two channels are emitters whose contributions should sum. Their
# additive blend is order-independent and saturates around ~0.6 — raise it only
# alongside the Layers panel's display range.
LAYER_INTENSITY = 0.4

# Manifest dataset + the files it pins, one per channel. A local refit is OUR
# artifact, not a copy of the hosted one, so it lives in the demo's local-fit
# namespace (~/.cache/luxar/<name>/local/, see `local_fit_path`). Writing it to
# ~/.cache/luxar/<name>/<file> — the path the manifest fetch owns — got it
# quarantined on the next launch for failing the pinned sha256 (#1618).
DEMO_NAME = "gsplats_cells3d"
GSPLATS_FILES = [
    "cells3d_ch0.gsplats.zarr.zip",
    "cells3d_ch1.gsplats.zarr.zip",
]

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

# Setup
Arbol.max_depth = 5

# Global device (auto-detected on first fit)
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def load_cells3d():
    """Load cells3d dataset from scikit-image.

    Returns:
        tuple: ``(volumes, source_dtype)`` -- one 3D volume per channel, shape
        (Z, Y, X), float32 [0, 1], and the element type the data was STORED in.
        The grid is untouched here, so only the dtype has to be declared to the
        fit; without it the compression ratio is quoted against the float32
        working copy rather than the acquisition.
    """
    with asection("Loading cells3d dataset"):
        cells3d = require_module("skimage.data").cells3d

        # cells3d() returns (60, 2, 256, 256) — (Z, Channel, Y, X), uint16
        try:
            raw = cells3d()
        except Exception as exc:
            # The sample data is FETCHED, not bundled in the scikit-image wheel,
            # and skimage's fetcher is pooch. A missing pooch is the usual cause
            # here, so re-gate it to get the constrained hint; if pooch IS
            # present the failure is something else (network), so say that.
            require_module("pooch")
            raise RuntimeError(
                f"Failed to fetch the cells3d sample data: {exc}\n"
                "It is downloaded on first use — check network access."
            ) from exc
        aprint(f"Raw data shape: {raw.shape}, dtype: {raw.dtype}")
        aprint(
            f"  Axes: (Z={raw.shape[0]}, C={raw.shape[1]}, "
            f"Y={raw.shape[2]}, X={raw.shape[3]})"
        )

        volumes = []
        for ch_config in CHANNELS:
            ch_idx = ch_config["index"]
            ch_name = ch_config["name"]

            V = raw[:, ch_idx, :, :].astype(np.float32)
            # Normalize to [0, 1]
            V = (V - V.min()) / (V.max() - V.min() + 1e-8)

            volumes.append(V)
            aprint(f"  {ch_name}: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

        aprint(f"Loaded {len(volumes)} channels")
        return volumes, str(raw.dtype)


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(volume, channel_name, cache_file, source_dtype=None):
    """Fit gsplats to a single channel (always fits — caller handles precomputed).

    Args:
        volume: 3D volume (Z, Y, X), float32 [0, 1]
        channel_name: Human-readable channel name
        cache_file: Path to .gsplats.zarr.zip cache file

    Returns:
        GSplatData with fitted 3D splats
    """
    # Auto-detect device
    global DEVICE
    if DEVICE is None:
        from luxar.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {channel_name} (fixed-K joint fit: seeds={MAX_SPLATS})...")

    result = fit_gaussian_splats(
        volume,
        seeds=MAX_SPLATS,
        # The grid is the acquisition's; only the element type was changed
        # on the way here, and that is the denominator of the ratio.
        source_dtype=source_dtype,
        device=DEVICE,
        seed_method="edges",
        verbose=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats")

    # Cache result in compressed zarr format (AUTO = certified near-lossless,
    # the current default; centers→u16, Cholesky→certified u8).
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file}")
    save_with_lod(
        result,
        cache_file,
        recipe="stream",
        encoding_mode=EncodingMode.AUTO,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return GSplatData.load(cache_file, include_stats=False)


def fit_all_channels(volumes, source_dtype=None):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = local_fit_path(DEMO_NAME, GSPLATS_FILES[i])

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(
                    volume, ch_name, cache_file, source_dtype=source_dtype
                )
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation — per-channel layers with BOP LUTs
# =============================================================================


def create_luxar_scene(gsplats_list, output_path=None):
    """Create a 3D Luxar scene with one layer-enabled gsplats node per channel.

    Each channel is centered on a shared amplitude-weighted centroid (so the
    channels stay aligned) and added with ``layer=True`` and a BOP LUT. The
    viewer's Layers panel toggles channels and adjusts their display range,
    gamma, and blending mode live.
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_cells3d_multichannel.luxar.zarr"
        )

    with asection("Creating 3D Luxar Scene (per-channel layers)"):
        aprint(f"Output: {output_path.name}")

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # ACES, set explicitly, for the per-channel BOP LUTs
            # (ACES is the house default; it shifts LUT hues slightly, which is
            # the accepted trade for its highlight rolloff).
            #
            # exposure is in LOG2 STOPS. Two thin volumetric channels over a
            # 60-slice stack integrate to very little radiance, so the scene
            # opened far too dim and had to be pushed +2.35 EV by hand in the
            # viewer before it read at all. That measured value is baked here.
            # It is the right lever rather than the amplitudes: the writer
            # derives each channel's display window FROM its amplitudes, so a
            # global rescale cancels out on screen and changes nothing.
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(
                    cinematic_mode=True, tone_mapping="ACES", exposure=2.35
                ),
                citation=DEMO_META["citation"],
            )
            stamp_input_digests(scene)

            scene.attrs["title"] = "GSplats: 3D Cells Multi-Channel (BOP layers)"
            scene.attrs["description"] = """
Multi-Channel Gaussian Splatting — cells3d (scikit-image)
=========================================================

Two-channel fluorescence microscopy rendered as Gaussian splats, one
toggleable layer per channel with a BOP (Blue-Orange-Purple) LUT.

Data Source:
  - scikit-image cells3d sample dataset
  - Allen Institute for Cell Science
  - Shape: (60, 2, 256, 256) — (Z, Channel, Y, X)

Channels (Layers panel — press L):
  - bop_orange: Cell membranes (Channel 0)
  - bop_blue:   Cell nuclei    (Channel 1)

Controls:
  - Press L for the Layers panel (toggle channels, display range, gamma)
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # Compute shared centroid across ALL channels so they stay aligned
            all_centers = [g.centers for g in gsplats_list]
            all_amps = [g.amplitudes for g in gsplats_list]
            total_amp = sum(a.sum() for a in all_amps)
            if total_amp > 0:
                shared_centroid = (
                    sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                )
            else:
                shared_centroid = np.mean(np.concatenate(all_centers, axis=0), axis=0)

            # Add each channel as a layer-enabled gsplats node with a BOP LUT
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                colormap = ch_config["colormap"]

                with asection(f"Adding {ch_name} (layer, {colormap})"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(LAYER_INTENSITY)

                    n_splats = len(gsplats.amplitudes)

                    # colormap= applies a BOP LUT at display time; layer=True
                    # exposes the node in the Layers panel. Data columns are
                    # [Z, Y, X] from fitting a (Z, Y, X) volume, mapped to the
                    # scene's [x, y, z] via dim_order.
                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower()}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        dim_order=["z", "y", "x"],
                        opacity=1.0,
                        # One global order slot per node cannot interleave these
                        # co-located volumes; additive is order-independent.
                        blending_mode="additive",
                        layer=True,
                        colormap=colormap,
                        **window_attrs(ch_config["window"]),
                    )
                    aprint(
                        f"  Added {n_splats:,} splats with colormap='{colormap}' "
                        f"window={ch_config['window']}"
                    )

            # --- Overlays ---
            scene.add_text(
                "Cells3D Multichannel",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
            )
            add_demo_caption(
                scene, "scikit-image • 2 channels • BOP LUTs", DEMO_META.get("citation")
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3D Multi-Channel Cells (Layers + BOP LUTs)")
    aprint("=" * 70)
    aprint("3D per-channel fitting + per-channel BOP-LUT layers")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_cells3d_multichannel.luxar.zarr"

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Try the manifest-driven fetch (checksum-verified cache -> in-repo -> Zenodo)
    try:
        precomputed = load_dataset_gsplats(
            DEMO_NAME,
            GSPLATS_FILES,
            recompute=RECOMPUTE,
        )
    except DatasetUnavailable as exc:
        aprint(f"Manifest fetch unavailable ({exc}).")
        precomputed = None
    if precomputed is None and not RECOMPUTE:
        # A fit this machine built earlier, in its own namespace — checked
        # BEFORE refitting, which is what makes the refit below one-time.
        precomputed = load_local_fit_gsplats(DEMO_NAME, GSPLATS_FILES)

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # --recompute path (or no data to be had): load raw data, fit from
        # scratch, and cache the fits in the local-fit namespace.
        warn_if_no_cuda_gpu()
        volumes, source_dtype = load_cells3d()
        gsplats_list = fit_all_channels(volumes, source_dtype=source_dtype)

    # Report
    with asection("Fitting Summary"):
        for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
            aprint(
                f"  {ch_config['name']}: {len(gsplats.amplitudes):,} splats, "
                f"{gsplats.centers.shape[1]}D"
            )

    # Create 3D scene with per-channel layers
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
