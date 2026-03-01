#!/usr/bin/env python3
"""GSplats Demo: 5D Multi-Channel Cells with Boolean Toggles (scikit-image cells3d)

Variant of the multichannel demo that uses two independent boolean dimensions
instead of a single categorical Channel slider. This lets you toggle each
channel on/off independently.

================================================================================
BOOLEAN TOGGLE DIMENSIONS — INDEPENDENT CHANNEL VISIBILITY
================================================================================

This demo uses the cells3d dataset from scikit-image — a two-channel 3D
fluorescence microscopy volume of cells (membranes + nuclei).

Instead of a single Channel slider that switches between views, this version
creates two boolean dimensions: "Membranes" and "Nuclei". Each can be toggled
independently, allowing four combinations:

  - Both On:       See membranes + nuclei overlaid
  - Membranes On:  See only membranes
  - Nuclei On:     See only nuclei
  - Both Off:      Nothing visible

Each channel's splats use ``extend_to_all`` on the *other* channel's dimension
so they remain visible regardless of that toggle's position.

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
3. **Create 5D scene** with dimensions [X, Y, Z, Membranes, Nuclei]
4. **Add splats** with per-channel boolean toggles via ``fill`` + ``extend_to_all``
5. **Visualize** — Toggle each channel independently

USAGE:
======
    python demo_gsplats_4d_cells3d_multichannel_toggles.py [--no-cache] [--no-serve] [--serve-only]

Options:
    --no-cache:   Force re-fitting (ignore cached results)
    --no-serve:   Don't auto-launch viewer after scene creation
    --serve-only: Skip loading/fitting, just serve existing scene

Output:
    - Scene saved to: demos/gsplats_5d_cells3d_multichannel_toggles.zarr
    - Automatically opens in browser at http://localhost:8000

"""

import sys
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import launch_viewer, warn_if_no_cuda_gpu
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Fitting parameters
N_SEEDS = 15000  # Splats per channel
N_ITERS = 8000  # Optimization iterations

# Voxel spacing (Z, Y, X) in micrometres for cells3d
# Original: (0.29, 0.065, 0.065) µm, 4x downsampled in Y/X → (0.29, 0.26, 0.26) µm
VOXEL_SIZE_ZYX = (0.29, 0.26, 0.26)

# Channel configuration
CHANNELS = [
    {"index": 0, "name": "Membranes", "color": (0.0, 1.0, 0.3)},  # Green
    {"index": 1, "name": "Nuclei", "color": (0.5, 0.3, 1.0)},  # Purple
]

# Cache directory (shared with the other demo — same fitting params)
CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_cells3d"

# Parse command-line flags
NO_CACHE = "--no-cache" in sys.argv
NO_SERVE = "--no-serve" in sys.argv
SERVE_ONLY = "--serve-only" in sys.argv

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
        list[np.ndarray]: One 3D volume per channel, shape (Z, Y, X), float32 [0, 1].
    """
    with asection("Loading cells3d dataset"):
        try:
            from skimage.data import cells3d
        except ImportError:
            raise ImportError(
                "scikit-image is required for this demo.\n"
                "Install with: pip install scikit-image"
            )

        # cells3d() returns (60, 2, 256, 256) — (Z, Channel, Y, X), uint16
        raw = cells3d()
        aprint(f"Raw data shape: {raw.shape}, dtype: {raw.dtype}")
        aprint(
            f"  Axes: (Z={raw.shape[0]}, C={raw.shape[1]}, Y={raw.shape[2]}, X={raw.shape[3]})"
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
    if cache_file.exists() and not NO_CACHE:
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
        import torch

        if torch.cuda.is_available():
            DEVICE = "cuda"
            aprint("Using CUDA device")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            DEVICE = "mps"
            aprint("Using MPS device (Metal acceleration)")
        else:
            DEVICE = "cpu"
            aprint("Using CPU device")

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {channel_name} ({N_ITERS} iterations, {N_SEEDS} seeds)...")

    result = fit_gaussian_splats(
        volume,
        seeds=N_SEEDS,
        n_iters=N_ITERS,
        device=DEVICE,
        verbose=True,
        enable_dynamic_ops=True,
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
    )

    return result


def fit_all_channels(volumes):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            cache_file = CACHE_DIR / f"cells3d_ch{i}.gsplats.zarr.zip"

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(volume, ch_name, cache_file)
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Scene Creation — Boolean toggle dimensions
# =============================================================================


def create_luxar_scene(gsplats_list, output_path=None):
    """Create 5D Luxar scene with independent boolean toggle dimensions.

    Instead of a single Channel slider, each channel gets its own boolean
    dimension. Splats use ``extend_to_all`` on the *other* channel's dimension
    so they are visible regardless of that toggle's state.

    Scene dimensions: [X, Y, Z, Membranes, Nuclei]

    Membrane splats:
      - fill={"membranes": 1.0}  → visible when Membranes=On
      - extend_to_all=["nuclei"] → visible regardless of Nuclei toggle

    Nuclei splats:
      - fill={"nuclei": 1.0}      → visible when Nuclei=On
      - extend_to_all=["membranes"] → visible regardless of Membranes toggle
    """
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_5d_cells3d_multichannel_toggles.zarr"
        )

    # Map each channel to its own dimension name and the other channel's dim
    CHANNEL_DIM_NAMES = [ch["name"].lower() for ch in CHANNELS]

    with asection("Creating 5D Luxar Scene (boolean toggles)"):
        aprint(f"Output: {output_path.name}")

        # Define 5D scene: 3 spatial + 2 boolean toggle dimensions
        dim_list = [
            Dimension("x", unit="px", display=True),
            Dimension("y", unit="px", display=True),
            Dimension("z", unit="px", display=True),
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
            scene = compiler.create_scene(dimensions=dims)

            scene.attrs["title"] = "GSplats: 5D Cells (boolean toggle demo)"
            scene.attrs["description"] = """
5D Multi-Channel Gaussian Splatting — Boolean Toggle Dimensions
================================================================

Variant of the multichannel demo using independent boolean dimensions
instead of a single Channel slider. Each channel can be toggled on/off
independently, allowing all four visibility combinations.

Data Source:
  - scikit-image cells3d sample dataset
  - Allen Institute for Cell Science
  - Shape: (60, 2, 256, 256) — (Z, Channel, Y, X)

Channels:
  - Green: Cell membranes (Membranes dimension)
  - Purple: Cell nuclei (Nuclei dimension)

Navigation:
  - Toggle Membranes on/off to show/hide membrane splats
  - Toggle Nuclei on/off to show/hide nuclear splats
  - Both on: overlaid view; both off: nothing visible
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            # Compute shared centroid across ALL channels so they stay aligned
            with asection("Computing shared centroid"):
                all_centers = [g.centers for g in gsplats_list]
                all_amps = [g.amplitudes for g in gsplats_list]
                total_amp = sum(a.sum() for a in all_amps)
                shared_centroid = (
                    sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
                )
                aprint(f"  Shared centroid: {shared_centroid}")

            # Add each channel with its own boolean toggle dimension
            for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
                ch_name = ch_config["name"]
                color = ch_config["color"]
                own_dim = CHANNEL_DIM_NAMES[i]
                other_dim = CHANNEL_DIM_NAMES[1 - i]

                with asection(f"Adding {ch_name} (toggle: {own_dim})"):
                    # Transform: shared centroid so channels stay aligned
                    gsplats = gsplats.translate(-shared_centroid)
                    gsplats = gsplats.scale_intensity(0.1)

                    n_splats = len(gsplats.amplitudes)

                    # Assign channel color to all splats
                    colors = np.tile(np.array(color, dtype=np.float32), (n_splats, 1))

                    # KEY: Each channel's splats are placed at own_dim=1 (On)
                    # and extend_to_all on the other channel's dimension so
                    # they don't disappear when the other toggle changes.
                    scene.add_gsplats(
                        name=f"gsplats_{ch_name.lower()}",
                        centers=gsplats.centers,
                        amplitudes=gsplats.amplitudes,
                        cholesky_factors=gsplats.cholesky_factors,
                        colors=colors,
                        sharpness=gsplats.sharpnesses,
                        dim_order=["z", "y", "x"],
                        fill={own_dim: 1.0},
                        fill_sigma={own_dim: 0},
                        extend_to_all=[other_dim],
                        opacity=1.0,
                        blending_mode="additive",
                    )
                    aprint(
                        f"  Added {n_splats:,} splats with {own_dim}=On, extend_to_all=[{other_dim}]"
                    )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    warn_if_no_cuda_gpu()
    aprint("=" * 70)
    aprint("GSplats Demo: 5D Multi-Channel Cells (boolean toggle dimensions)")
    aprint("=" * 70)
    aprint("3D per-channel fitting + independent boolean toggles per channel")
    aprint("")

    output_path = (
        get_demos_output_dir() / "gsplats_5d_cells3d_multichannel_toggles.zarr"
    )

    # Serve-only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    # Load data
    volumes = load_cells3d()

    # Fit gsplats per channel (with caching)
    gsplats_list = fit_all_channels(volumes)

    # Report
    with asection("Fitting Summary"):
        for i, (gsplats, ch_config) in enumerate(zip(gsplats_list, CHANNELS)):
            aprint(
                f"  {ch_config['name']}: {len(gsplats.amplitudes):,} splats, "
                f"{gsplats.centers.shape[1]}D"
            )

    # Create 5D scene with boolean toggles
    scene_path = create_luxar_scene(gsplats_list)

    # Launch viewer
    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
