#!/usr/bin/env python3
"""GSplats Demo: 3D Multi-Channel Kidney with Layers Panel (scikit-image kidney)

Variant of the multichannel kidney demo that uses the **Layers panel** instead of
boolean toggle dimensions for per-channel visibility control.

================================================================================
LAYERS PANEL — PER-NODE VISIBILITY AND DISPLAY CONTROLS
================================================================================

This demo uses the kidney() dataset from scikit-image — a three-channel 3D
confocal fluorescence microscopy volume of mouse kidney tissue (FluoCells
Prepared Slide #3).

Instead of creating extra nD boolean toggle dimensions (which adds complexity
and increases the scene from 3D to 6D), this demo marks each channel's gsplats
node as ``layer=True``. The viewer's Layers panel (press **L**) then provides:

  - **Visibility toggle** — eye icon to show/hide each channel
  - **Display range** — [min, max] intensity sliders per channel
  - **Gamma** — per-channel gamma correction
  - **Blending mode** — per-channel blend (additive, normal, etc.)
  - **Multi-select** — Ctrl+click / Shift+click to adjust multiple channels

This keeps the scene at 3D (X, Y, Z only) while providing the same
independent channel control — plus richer per-channel display adjustments
that boolean toggles cannot offer.

================================================================================
LAYERS PANEL vs. MULTIDIMENSIONAL TOGGLES — DESIGN TRADEOFFS
================================================================================

Luxar offers two ways to control per-channel visibility. This demo uses the
**Layers panel** approach. A companion demo uses the **multidimensional toggle**
approach (see ``demo_gsplats_3d_kidney_multichannel_toggles.py``).

LAYERS PANEL (this demo):

  Channel visibility is encoded as **presentation** — the scene stays 3D, and
  each channel's node is marked ``layer=True``. The viewer's Layers panel
  (press L) provides per-channel controls: visibility toggle, continuous
  [min, max] display range, gamma correction, and blending mode.

  This is a **viewer UI feature** rather than a data modeling technique. It
  provides richer per-channel control than binary toggles — you can window the
  intensity range (like adjusting contrast in napari/ImageJ), apply per-channel
  gamma, and switch blending modes live. These controls are viewer-side only:
  they are not part of the data model, cannot be animated through the dimension
  system, and are not serialized in URLs.

  The scene stays low-dimensional, which makes it simpler to reason about and
  faster to navigate. There is no ``fill``/``extend_to_all`` plumbing needed.

MULTIDIMENSIONAL TOGGLES (companion demo):

  Channel visibility is encoded as **data** — each channel gets its own boolean
  coordinate dimension, making the scene 6D. The viewer's nD navigation
  machinery (sliders, keyboard shortcuts, animations, URL state) all work
  automatically because channel state is just another dimension.

  This is a general-purpose **data modeling technique**. It works for any
  discrete parameter — channels, timepoints, experimental conditions,
  replicates. The combinatorial state space is navigable through the same
  uniform nD interface. But the control is binary (on/off) — no continuous
  intensity adjustment, gamma, or blending mode per channel.

WHEN TO USE WHICH:

  - Use **layers** when you need fine-grained visual control per channel
    (intensity windowing, gamma, blending), when keeping dimensionality low
    matters, or when the channels are purely a display concern. This is the
    more natural choice for typical microscopy multi-channel viewing.

  - Use **multidimensional toggles** when channel state is part of the data
    semantics (e.g., comparing conditions), when you need to animate through
    combinations, or when composing with other nD features (time + channels).

  - Use **both** together when you want nD navigation for some dimensions
    (time, z-slicing) and layers for per-channel visual tuning.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source: scikit-image sample data (``skimage.data.kidney()``)
Shape: (16, 512, 512, 3) — (Z, Y, X, C), uint16
Channel 0 (450nm): DAPI — cell nuclei
Channel 1 (515nm): Alexa Fluor 488 WGA (wheat germ agglutinin) — glomeruli and tubules
Channel 2 (605nm): Alexa Fluor 568 Phalloidin — actin filaments
Voxel size: 1.25 µm (Z), 1.24 µm (Y), 1.24 µm (X)
Origin: FluoCells Prepared Slide #3 (Invitrogen F-24630)
        Mouse kidney cryostat section, confocal fluorescence microscopy
        Acquired by Genevieve Buckley at Monash Micro Imaging, 2018
License: CC0

How to Cite:
------------
scikit-image: image processing in Python.
van der Walt et al. (2014). PeerJ 2:e453. DOI: 10.7717/peerj.453

WORKFLOW:
=========

1. **Load** kidney dataset from scikit-image (16 x 512 x 512 x 3)
2. **Fit** each channel independently as 3D Gaussian splats
3. **Create 3D scene** with dimensions [X, Y, Z] only
4. **Add splats** with ``layer=True`` on each channel
5. **Visualize** — Press L to open Layers panel, toggle/adjust per channel

USAGE:
======
    python demo_gsplats_3d_kidney_multichannel_layers.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:      Force re-fitting from scratch (ignore precomputed/cached results)
    --no-serve:       Don't auto-launch viewer after scene creation
    --serve-only:     Skip loading/fitting, just serve existing scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volumes

Output:
    - Scene saved to: demos/gsplats_3d_kidney_multichannel_layers.luxar.zarr
    - Automatically opens in your browser on the demo's own derived port

"""

DEMO_META = {
    "key": "gsplats_3d_kidney_multichannel_layers",
    "title": "3D Kidney Multichannel Layers",
    "description": "3-channel confocal mouse kidney splats with per-channel Layers-panel control.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 2,
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["gsplats_kidney"],
    "outputs": ["gsplats_3d_kidney_multichannel_layers"],
    # scikit-image ships the sample, but the volume was acquired by Genevieve
    # Buckley (Monash Micro Imaging, 2018) and released CC0 -- both recorded in
    # skimage's own kidney docstring. The van der Walt PeerJ paper credits the
    # LIBRARY, so citing it here would attribute someone else's microscopy to
    # the software that loads it; the header keeps that citation where it belongs.
    "citation": {
        "short": "G. Buckley 2018 (scikit-image kidney)",
        "ref": "G. Buckley 2018",
        "license": "CC0 1.0",
    },
}

import sys

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
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
    warn_if_no_cuda_gpu,
)
from luxar.demos._lod_policy import save_with_lod
from luxar.demos._roundtrip_common import show_roundtrip_comparison
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 75000

# Voxel spacing (Z, Y, X) in micrometres for kidney dataset
VOXEL_SIZE_ZYX = (1.25, 1.24, 1.24)

# Channel configuration — ordered by emission wavelength
# Uses named colormaps instead of baked RGB colors — the viewer applies the
# colormap at display time, allowing interactive colormap switching.
CHANNELS = [
    {
        "index": 0,
        "name": "Nuclei",
        "colormap": "blue",  # DAPI, 450nm
        "emission": "450nm",
        "stain": "DAPI",
    },
    {
        "index": 1,
        "name": "WGA",
        "colormap": "green",  # Alexa Fluor 488 WGA, 515nm
        "emission": "515nm",
        "stain": "Alexa Fluor 488 WGA",
    },
    {
        "index": 2,
        "name": "Actin",
        "colormap": "red",  # Alexa Fluor 568 Phalloidin, 605nm
        "emission": "605nm",
        "stain": "Alexa Fluor 568 Phalloidin",
    },
]

# Display window: the fraction of the writer's own [min, p99.9] amplitude window
# each channel is displayed over. The writer's robust default is right for a
# single ADDITIVE layer, but three volumetric fluorescence channels composited
# together read far too dark on it — the median splat sits at ~15% of that
# window, so most of the tissue maps into the bottom fifth of the LUT. Measured
# on this dataset: 1.0 (the writer default) is the dim render this replaces,
# 0.25 washes the red/blue overlap out to magenta and merges structure, and 0.4
# — roughly the 92nd percentile per channel — lifts the mid-tones while the
# brightest cores only just begin to clip. The viewer's Layers panel (L) still
# moves it live; this only sets where it opens.
DISPLAY_WINDOW_FRACTION = 0.4

# Precomputed data configuration (same precomputed fits, and the same cache, as
# the toggle demo). A local refit is OUR artifact, not a copy of the hosted one,
# so it lives in the demo's local-fit namespace (~/.cache/luxar/<name>/local/,
# see `local_fit_path`). Writing it to ~/.cache/luxar/<name>/<file> — the path
# the manifest fetch owns — got it quarantined on the next launch for failing
# the pinned sha256 (#1618).
_PRECOMPUTED_DEMO_NAME = "gsplats_kidney"
_PRECOMPUTED_FILE_NAMES = [
    "kidney_ch0.gsplats.zarr.zip",
    "kidney_ch1.gsplats.zarr.zip",
    "kidney_ch2.gsplats.zarr.zip",
]

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv

# Setup
Arbol.max_depth = 5

# Global device (auto-detected on first fit)
DEVICE = None


# =============================================================================
# Data Loading
# =============================================================================


def load_kidney():
    """Load kidney dataset from scikit-image.

    Returns:
        tuple: ``(volumes, source_dtype)`` -- one 3D volume per channel, shape
        (Z, Y, X), float32 [0, 1], and the element type the data was STORED in.
        The grid is untouched here, so only the dtype has to be declared to the
        fit; without it the compression ratio is quoted against the float32
        working copy rather than the acquisition.
    """
    with asection("Loading kidney dataset"):
        kidney = require_module("skimage.data").kidney

        try:
            raw = kidney()
        except Exception as exc:
            # The sample data is FETCHED, not bundled in the scikit-image wheel,
            # and skimage's fetcher is pooch. A missing pooch is the usual cause
            # here, so re-gate it to get the constrained hint; if pooch IS
            # present the failure is something else (network), so say that.
            require_module("pooch")
            raise RuntimeError(
                f"Failed to fetch the kidney sample data: {exc}\n"
                "It is downloaded on first use — check network access."
            ) from exc
        aprint(f"Raw data shape: {raw.shape}, dtype: {raw.dtype}")
        aprint(
            f"  Axes: (Z={raw.shape[0]}, Y={raw.shape[1]}, X={raw.shape[2]}, C={raw.shape[3]})"
        )

        volumes = []
        for ch_config in CHANNELS:
            ch_idx = ch_config["index"]
            ch_name = ch_config["name"]
            ch_stain = ch_config["stain"]

            V = raw[:, :, :, ch_idx].astype(np.float32)
            # Normalize to [0, 1]
            V = (V - V.min()) / (V.max() - V.min() + 1e-8)

            volumes.append(V)
            aprint(
                f"  {ch_name} ({ch_stain}): {V.shape}, range [{V.min():.3f}, {V.max():.3f}]"
            )

        aprint(f"Loaded {len(volumes)} channels (stored as {raw.dtype})")
        return volumes, str(raw.dtype)


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(volume, channel_name, cache_file, source_dtype=None):
    """Fit gsplats to a single channel, using cache if available.

    Args:
        volume: 3D volume (Z, Y, X), float32 [0, 1]
        channel_name: Human-readable channel name
        cache_file: Path to .gsplats.zarr.zip cache file

    Returns:
        GSplatData with fitted 3D splats
    """
    # Check this machine's own earlier fit (a partial set: the caller already
    # tried the complete one).
    if cache_file.exists() and not RECOMPUTE:
        aprint(f"Loading cached fit for {channel_name}")
        try:
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"  Loaded {len(result.amplitudes):,} cached splats")
            return result
        except Exception as e:
            aprint(f"  Cache load failed: {e}, re-fitting...")

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
        # The grid is the acquisition's; only the element type was changed on
        # the way here, and that is the denominator of the compression ratio.
        source_dtype=source_dtype,
        device=DEVICE,
        verbose=True,
        # boundary_penalty=0.1,
        # clip_to_bounds=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats")

    # Cache result in compressed zarr format
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file}")
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


def fit_all_channels(volumes, source_dtype=None):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = local_fit_path(
                _PRECOMPUTED_DEMO_NAME, _PRECOMPUTED_FILE_NAMES[i]
            )

            with asection(f"Channel {i}: {ch_name} ({ch_config['stain']})"):
                gsplats = fit_channel(
                    volume, ch_name, cache_file, source_dtype=source_dtype
                )
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation — Layers panel (no extra dimensions)
# =============================================================================


def display_window(amplitudes):
    """The ``[lo, hi]`` scalar window one channel opens on.

    Mirrors the writer's own robust window — ``[min, p99.9]``, not ``[min,
    max]``, because gsplat amplitudes are heavily right-skewed — and then pulls
    the upper end down by :data:`DISPLAY_WINDOW_FRACTION` so the composited
    three-channel render is not dim. Derived per channel rather than hardcoded,
    so a refit (different K, a different floor) moves the window with the data
    instead of stranding it.
    """
    amps = np.asarray(amplitudes, dtype=np.float64)
    lo = float(amps.min())
    hi = lo + (float(np.percentile(amps, 99.9)) - lo) * DISPLAY_WINDOW_FRACTION
    if not hi > lo:  # degenerate (constant amplitudes) — fall back to the max
        hi = max(float(amps.max()), lo + 1e-6)
    return lo, hi


def create_luxar_scene(gsplats_list, output_path=None):
    """Create 3D Luxar scene with per-channel layers.

    Unlike the toggle-dimension variant, this version stays at 3D (X, Y, Z)
    and uses ``layer=True`` on each channel's gsplats node to expose them
    in the viewer's Layers panel (press L).

    The Layers panel provides per-channel:
      - Visibility toggle (eye icon)
      - Display range [min, max] sliders
      - Gamma correction
      - Blending mode selection
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_kidney_multichannel_layers.luxar.zarr"
        )

    with asection("Creating 3D Luxar Scene (layer controls)"):
        aprint(f"Output: {output_path.name}")

        # Define 3D scene: spatial dimensions only
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
            # ACES, set explicitly, for the per-channel colormaps
            # (ACES is the house default; it shifts LUT hues slightly, which is
            # the accepted trade for its highlight rolloff).
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
                citation=DEMO_META["citation"],
            )

            scene.attrs["title"] = "GSplats: 3D Kidney Multi-Channel (Layers panel)"
            scene.attrs["description"] = """
3D Multi-Channel Gaussian Splatting — Layers Panel
====================================================

Three-channel confocal fluorescence microscopy of mouse kidney tissue
with per-channel control via the Layers panel (press L).

Data Source:
  - scikit-image kidney() sample dataset
  - FluoCells Prepared Slide #3 (Invitrogen F-24630)
  - Mouse kidney cryostat section
  - Confocal microscopy (Nikon C1 inverted)
  - Acquired by Genevieve Buckley, Monash Micro Imaging, 2018

Channels (each is a layer):
  - Blue:  DAPI — cell nuclei (450nm)
  - Green: Alexa Fluor 488 WGA — glomeruli/tubules (515nm)
  - Red:   Alexa Fluor 568 Phalloidin — actin filaments (605nm)

Controls:
  - Press L to open the Layers panel
  - Click eye icon to toggle channel visibility
  - Adjust [min, max] display range per channel
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # Compute shared centroid across ALL channels so they stay aligned
            with asection("Computing shared centroid"):
                all_centers = [g.centers for g in gsplats_list]
                all_amps = [g.amplitudes for g in gsplats_list]
                total_amp = sum(a.sum() for a in all_amps)
                if total_amp > 0:
                    shared_centroid = (
                        sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                    )
                else:
                    shared_centroid = np.mean(
                        np.concatenate(all_centers, axis=0), axis=0
                    )
                aprint(f"  Shared centroid: {shared_centroid}")

            # Add each channel as a layer-enabled gsplats node
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                colormap = ch_config["colormap"]

                with asection(f"Adding {ch_name} (layer)"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)
                    lo, hi = display_window(gsplats.amplitudes)
                    span = hi - lo

                    # KEY: colormap= replaces baked RGB colors.
                    # The viewer applies the LUT at display time, enabling
                    # interactive colormap switching in the Layers panel.
                    # layer=True exposes this node in the Layers panel.
                    #
                    # intensity/offset on a COLORMAPPED node are the scalar
                    # display WINDOW, not a post-LUT gain (the viewer recovers
                    # [-offset/i, (1-offset)/i] — see rendering/display-range.ts
                    # ::computeDisplayRange), so this is what bakes a Layers-panel
                    # display range into the scene. Amplitude scaling cannot do
                    # it: the writer's own window is derived FROM the amplitudes,
                    # so a global rescale cancels out on screen.
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
                        intensity=1.0 / span,
                        offset=-lo / span,
                    )
                    aprint(
                        f"  Added {n_splats:,} splats with colormap='{colormap}', "
                        f"display range [{lo:.4f}, {hi:.4f}]"
                    )

            # Overlay annotations
            scene.add_text(
                "Kidney Multichannel",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene, "Fluorescence \u2022 Layer panel", DEMO_META.get("citation")
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3D Multi-Channel Kidney (Layers panel)")
    aprint("=" * 70)
    aprint("3D per-channel fitting + Layers panel for per-channel control")
    aprint("Dataset: Mouse kidney — DAPI (nuclei) + WGA (tubules) + Phalloidin (actin)")
    aprint("Press L in the viewer to open the Layers panel")
    aprint("")

    output_path = (
        get_demos_output_dir() / "gsplats_3d_kidney_multichannel_layers.luxar.zarr"
    )

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
        gsplats_list = load_dataset_gsplats(
            _PRECOMPUTED_DEMO_NAME,
            _PRECOMPUTED_FILE_NAMES,
            recompute=RECOMPUTE,
        )
    except DatasetUnavailable as exc:
        aprint(f"Manifest fetch unavailable ({exc}).")
        gsplats_list = None
    if gsplats_list is None and not RECOMPUTE:
        # A fit this machine built earlier, in its own namespace — checked
        # BEFORE refitting, which is what makes the refit below one-time.
        gsplats_list = load_local_fit_gsplats(
            _PRECOMPUTED_DEMO_NAME, _PRECOMPUTED_FILE_NAMES
        )

    volumes = None

    if gsplats_list is None:
        # Recompute path (or no data to be had): warn about GPU requirements,
        # load data, fit, and cache the fits in the local-fit namespace.
        warn_if_no_cuda_gpu()

        # Load data
        volumes, source_dtype = load_kidney()

        # Fit gsplats per channel (with caching)
        gsplats_list = fit_all_channels(volumes, source_dtype=source_dtype)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if volumes is not None:
            show_roundtrip_comparison(
                volumes,
                gsplats_list,
                [c["name"] for c in CHANNELS],
                device=DEVICE,
            )
        else:
            aprint(
                "Cannot show round-trip: original volumes not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Report
    with asection("Fitting Summary"):
        for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
            aprint(
                f"  {ch_config['name']} ({ch_config['stain']}): "
                f"{len(gsplats.amplitudes):,} splats, "
                f"{gsplats.centers.shape[1]}D"
            )

    # Create 3D scene with layers
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        aprint("Press L to open the Layers panel!")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
