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
    - Automatically opens in browser at http://localhost:8000

"""

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    launch_viewer,
    load_precomputed_gsplats,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Progressive fitting parameters
MAX_SPLATS = 140000
MAX_SPLATS_PER_PASS = 28000
ITERS_PER_PASS = 5000
PSNR_PATIENCE = 0.1

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

# Cache directory (same as toggle demo — fits are identical)
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_kidney"

# Precomputed data configuration (same precomputed fits as toggle demo)
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
        list[np.ndarray]: One 3D volume per channel, shape (Z, Y, X), float32 [0, 1].
    """
    with asection("Loading kidney dataset"):
        try:
            from skimage.data import kidney
        except ImportError:
            raise ImportError(
                "scikit-image is required for this demo.\n"
                "Install with: pip install scikit-image pooch"
            )

        try:
            raw = kidney()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load kidney dataset: {e}\n"
                "This dataset is downloaded on first use and requires 'pooch'.\n"
                "Install with: pip install pooch"
            ) from e
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

        aprint(f"Loaded {len(volumes)} channels")
        return volumes


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(volume, channel_name, cache_file):
    """Fit gsplats to a single channel, using cache if available.

    Args:
        volume: 3D volume (Z, Y, X), float32 [0, 1]
        channel_name: Human-readable channel name
        cache_file: Path to .gsplats.zarr.zip cache file

    Returns:
        GSplatData with fitted 3D splats
    """
    # Check cache
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
        from luxar.utils.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_progressive_gaussian_splats

    aprint(
        f"Fitting {channel_name} (progressive: "
        f"max_splats={MAX_SPLATS}, max_splats_per_pass={MAX_SPLATS_PER_PASS}, "
        f"iters_per_pass={ITERS_PER_PASS}, psnr_patience={PSNR_PATIENCE})..."
    )

    result = fit_progressive_gaussian_splats(
        volume,
        max_splats=MAX_SPLATS,
        max_splats_per_pass=MAX_SPLATS_PER_PASS,
        iters_per_pass=ITERS_PER_PASS,
        psnr_patience=PSNR_PATIENCE,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
        # boundary_penalty=0.1,
        # clip_to_bounds=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats:,} splats")

    # Cache result in compressed zarr format
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file.name}")
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return result


def fit_all_channels(volumes):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"kidney_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name} ({ch_config['stain']})"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation — Layers panel (no extra dimensions)
# =============================================================================


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
            scene = compiler.create_scene(dimensions=dims)

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

                    # KEY: colormap= replaces baked RGB colors.
                    # The viewer applies the LUT at display time, enabling
                    # interactive colormap switching in the Layers panel.
                    # layer=True exposes this node in the Layers panel.
                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower()}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        dim_order=["z", "y", "x"],
                        opacity=1.0,
                        blending_mode="additive",
                        layer=True,
                        colormap=colormap,
                    )
                    aprint(f"  Added {n_splats:,} splats with colormap='{colormap}'")

            # Overlay annotations
            scene.add_text(
                "Kidney Multichannel",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Fluorescence \u2022 Layer panel",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Round-Trip Visualisation
# =============================================================================


def show_roundtrip_comparison(
    volumes: list[np.ndarray],
    gsplats_list: list[GSplatData],
) -> None:
    """Show original vs round-trip reconstructed volumes side by side."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        aprint(
            "matplotlib is required for --show-roundtrip. Install with: pip install matplotlib"
        )
        return

    n_channels = len(volumes)

    with asection("Round-trip reconstruction comparison"):
        reconstructions = []
        for i, (volume, gsplats, ch_config) in enumerate(
            zip(volumes, gsplats_list, CHANNELS[:n_channels])
        ):
            with asection(f"Rendering Ch{i}: {ch_config['name']}"):
                recon = gsplats.render_to_volume(shape=volume.shape, device=DEVICE)
                reconstructions.append(recon)
                mse = float(np.mean((volume - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(
            n_channels, 3, figsize=(14, 4.5 * n_channels), squeeze=False
        )

        for i, (volume, recon, ch_config) in enumerate(
            zip(volumes, reconstructions, CHANNELS[:n_channels])
        ):
            mid_z = volume.shape[0] // 2
            orig_slice = volume[mid_z]
            recon_slice = recon[mid_z]
            diff_slice = np.abs(orig_slice - recon_slice)

            mse = float(np.mean((volume - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[i, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original — {ch_config['name']}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff_slice, cmap="inferno", vmin=0, vmax=0.3)
            axes[i, 2].set_title("|Difference|")
            axes[i, 2].axis("off")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Round-Trip Comparison — z-slice {mid_z}  "
            f"({sum(len(g.amplitudes) for g in gsplats_list):,} total splats)",
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

    # Try loading precomputed data from Git LFS / local cache
    gsplats_list = load_precomputed_gsplats(
        _PRECOMPUTED_DEMO_NAME,
        _PRECOMPUTED_FILE_NAMES,
        recompute=RECOMPUTE,
    )

    volumes = None

    if gsplats_list is None:
        # Recompute path: warn about GPU requirements, load data, fit
        warn_if_no_cuda_gpu()

        # Load data
        volumes = load_kidney()

        # Fit gsplats per channel (with caching)
        gsplats_list = fit_all_channels(volumes)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if volumes is not None:
            show_roundtrip_comparison(volumes, gsplats_list)
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
