#!/usr/bin/env python3
"""GSplats Demo: 3D Multi-Channel Kidney with Boolean Toggles (scikit-image kidney)

Variant of the multichannel toggle demo adapted for the napari/scikit-image
kidney sample dataset — a 3-channel confocal fluorescence microscopy volume
of mouse kidney tissue.

================================================================================
BOOLEAN TOGGLE DIMENSIONS — INDEPENDENT CHANNEL VISIBILITY
================================================================================

This demo uses the kidney() dataset from scikit-image — a three-channel 3D
confocal fluorescence microscopy volume of mouse kidney tissue (FluoCells
Prepared Slide #3).

Each channel gets its own boolean dimension for independent on/off control,
allowing all eight visibility combinations:

  - All three On:        Full overlay of nuclei + WGA + actin
  - Any two On:          Pair overlay (e.g. WGA + actin without nuclei)
  - Single channel On:   View one structure in isolation
  - All Off:             Nothing visible

Each channel's splats use ``extend_to_all`` on the *other* channels' dimensions
so they remain visible regardless of those toggles' positions.

================================================================================
MULTIDIMENSIONAL TOGGLES vs. LAYERS PANEL — DESIGN TRADEOFFS
================================================================================

Luxar offers two ways to control per-channel visibility. This demo uses the
**multidimensional toggle** approach. A companion demo uses the **Layers panel**
approach (see ``demo_gsplats_3d_kidney_multichannel_layers.py``).

MULTIDIMENSIONAL TOGGLES (this demo):

  Channel visibility is encoded as **data** — each channel gets its own boolean
  coordinate dimension, making the scene 6D (X, Y, Z, Nuclei, WGA, Actin).
  The viewer's nD navigation machinery treats these like any other dimension:
  keyboard shortcuts, sliders, URL serialization, and dimension animation all
  work without any special UI code.

  This approach is a general-purpose **data modeling technique**. It works for
  any discrete parameter, not just channels — you could use it for timepoints,
  experimental conditions, replicates, staining protocols, or any categorical
  variable. The combinatorial state space is navigable through the same uniform
  nD interface.

  The tradeoff: the scene becomes higher-dimensional, the ``fill`` +
  ``extend_to_all`` plumbing requires understanding the nD data model, and the
  control is binary (on/off) — no continuous intensity adjustment, gamma, or
  blending mode changes per channel.

LAYERS PANEL (companion demo):

  Channel visibility is encoded as **presentation** — the scene stays 3D, and
  each channel's node is marked ``layer=True``. The viewer's Layers panel
  (press L) provides per-channel controls: visibility toggle, continuous
  [min, max] display range, gamma, and blending mode.

  This approach is a **viewer UI feature** designed for the specific use case
  of adjusting per-node visual properties. It provides richer controls than
  binary toggles, but these controls are viewer-side only — they are not part
  of the data model, cannot be animated via the dimension system, and are not
  serialized in URLs.

WHEN TO USE WHICH:

  - Use **multidimensional toggles** when channel state is part of the data
    semantics (e.g., comparing conditions), when you need to animate through
    combinations, or when composing with other nD features (time + channels).

  - Use **layers** when you need fine-grained visual control per channel
    (intensity windowing, gamma, blending), when keeping dimensionality low
    matters, or when the channels are purely a display concern.

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

1. **Load** kidney dataset from scikit-image (16 × 512 × 512 × 3)
2. **Fit** each channel independently as 3D Gaussian splats
3. **Create 6D scene** with dimensions [X, Y, Z, Nuclei, WGA, Actin]
4. **Add splats** with per-channel boolean toggles via ``fill`` + ``extend_to_all``
5. **Visualize** — Toggle each channel independently

USAGE:
======
    python demo_gsplats_3d_kidney_multichannel_toggles.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:      Force re-fitting from scratch (ignore precomputed/cached results)
    --no-serve:       Don't auto-launch viewer after scene creation
    --serve-only:     Skip loading/fitting, just serve existing scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volumes

Output:
    - Scene saved to: demos/gsplats_6d_kidney_multichannel_toggles.luxar.zarr
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

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 75000

# Voxel spacing (Z, Y, X) in micrometres for kidney dataset
VOXEL_SIZE_ZYX = (1.25, 1.24, 1.24)

# Channel configuration — ordered by emission wavelength
CHANNELS = [
    {
        "index": 0,
        "name": "Nuclei",
        "color": (0.3, 0.4, 1.0),  # Blue (DAPI, 450nm)
        "emission": "450nm",
        "stain": "DAPI",
    },
    {
        "index": 1,
        "name": "WGA",
        "color": (0.0, 1.0, 0.3),  # Green (Alexa Fluor 488 WGA, 515nm)
        "emission": "515nm",
        "stain": "Alexa Fluor 488 WGA",
    },
    {
        "index": 2,
        "name": "Actin",
        "color": (1.0, 0.3, 0.2),  # Red (Alexa Fluor 568 Phalloidin, 605nm)
        "emission": "605nm",
        "stain": "Alexa Fluor 568 Phalloidin",
    },
]

# Cache directory
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_kidney"

# Precomputed data configuration
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

        # kidney() returns (16, 512, 512, 3) — (Z, Y, X, C), uint16
        # Unlike cells3d, kidney() downloads data via pooch on first call
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

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {channel_name} (fixed-K joint fit: seeds={MAX_SPLATS})...")

    result = fit_gaussian_splats(
        volume,
        lr=0.01,
        seeds=MAX_SPLATS,
        device=DEVICE,
        verbose=True,
        boundary_penalty=0.1,
        clip_to_bounds=True,
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
# Scene Creation — Boolean toggle dimensions
# =============================================================================


def create_luxar_scene(gsplats_list, output_path=None):
    """Create 6D Luxar scene with independent boolean toggle dimensions.

    Instead of a single Channel slider, each channel gets its own boolean
    dimension. Splats use ``extend_to_all`` on the *other* channels' dimensions
    so they are visible regardless of those toggles' states.

    Scene dimensions: [X, Y, Z, Nuclei, WGA, Actin]

    Nuclei splats:
      - fill={"nuclei": 1.0}  → visible when Nuclei=On
      - extend_to_all=["wga", "actin"] → visible regardless of other toggles

    WGA splats:
      - fill={"wga": 1.0}  → visible when WGA=On
      - extend_to_all=["nuclei", "actin"] → visible regardless of other toggles

    Actin splats:
      - fill={"actin": 1.0}  → visible when Actin=On
      - extend_to_all=["nuclei", "wga"] → visible regardless of other toggles
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_6d_kidney_multichannel_toggles.luxar.zarr"
        )

    # Map each channel to its dimension name
    CHANNEL_DIM_NAMES = [ch["name"].lower() for ch in CHANNELS]

    with asection("Creating 6D Luxar Scene (boolean toggles)"):
        aprint(f"Output: {output_path.name}")

        # Define 6D scene: 3 spatial + 3 boolean toggle dimensions
        dim_list = [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
        ]

        # Add one boolean dimension per channel
        for ch_config in CHANNELS:
            dim_name = ch_config["name"].lower()
            dim_list.append(
                Dimension(
                    dim_name,
                    display=False,
                    discrete=True,
                    range=(0, 1),
                    step=1.0,
                    categories=["Off", "On"],
                )
            )

        dims = Dimensions(dim_list)

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
            )

            scene.attrs["title"] = (
                "GSplats: 3D Kidney Multi-Channel (boolean toggle demo)"
            )
            scene.attrs["description"] = """
6D Multi-Channel Gaussian Splatting — Boolean Toggle Dimensions
================================================================

Three-channel confocal fluorescence microscopy of mouse kidney tissue
with independent boolean toggle dimensions for each channel.

Data Source:
  - scikit-image kidney() sample dataset
  - FluoCells Prepared Slide #3 (Invitrogen F-24630)
  - Mouse kidney cryostat section
  - Confocal microscopy (Nikon C1 inverted)
  - Acquired by Genevieve Buckley, Monash Micro Imaging, 2018

Channels:
  - Blue:  DAPI — cell nuclei (450nm)
  - Green: Alexa Fluor 488 WGA — glomeruli/tubules (515nm)
  - Red:   Alexa Fluor 568 Phalloidin — actin filaments (605nm)

Navigation:
  - Toggle Nuclei/WGA/Actin on/off independently
  - 8 visibility combinations (all on, pairs, singles, all off)
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

            # Add each channel with its own boolean toggle dimension
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                color = ch_config["color"]
                own_dim = CHANNEL_DIM_NAMES[i]
                other_dims = [d for j, d in enumerate(CHANNEL_DIM_NAMES) if j != i]

                with asection(f"Adding {ch_name} (toggle: {own_dim})"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    # Assign channel color to all splats
                    colors = np.tile(np.array(color, dtype=np.float32), (n_splats, 1))

                    # KEY: Each channel's splats are placed at own_dim=1 (On)
                    # and extend_to_all on the other channels' dimensions so
                    # they don't disappear when any other toggle changes.
                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower()}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        colors=colors,
                        dim_order=["z", "y", "x"],
                        fill={own_dim: 1.0},
                        fill_sigma={own_dim: 0},
                        extend_to_all=other_dims,
                        opacity=1.0,
                        blending_mode="additive",
                        layer=True,
                    )
                    aprint(
                        f"  Added {n_splats:,} splats with {own_dim}=On, "
                        f"extend_to_all={other_dims}"
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
            scene.add_text(
                "Fluorescence \u2022 Toggle channels",
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
    aprint("GSplats Demo: 6D Multi-Channel Kidney (boolean toggle dimensions)")
    aprint("=" * 70)
    aprint("3D per-channel fitting + independent boolean toggles per channel")
    aprint("Dataset: Mouse kidney — DAPI (nuclei) + WGA (tubules) + Phalloidin (actin)")
    aprint("")

    output_path = (
        get_demos_output_dir() / "gsplats_6d_kidney_multichannel_toggles.luxar.zarr"
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

    # Create 6D scene with boolean toggles
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
