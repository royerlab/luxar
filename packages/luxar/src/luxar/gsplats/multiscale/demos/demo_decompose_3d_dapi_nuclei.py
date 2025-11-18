#!/usr/bin/env python3
"""
3D Multi-Scale Decomposition Demo with Real DAPI Microscopy Data from IDR.

This demo demonstrates multi-scale decomposition on real DAPI-stained nuclei from the
Image Data Resource (IDR). Unlike the gsplat demo, this uses FULL RESOLUTION data
(no downscaling) to show the decomposition at the original data quality.

Data source: https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr
"""

import sys

import napari
import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar.gsplats.multiscale import (
    decompose_image,
    show_optimization_movie,
    upsample_for_visualization,
)

# Check for --no-napari flag
NO_NAPARI = "--no-napari" in sys.argv
if NO_NAPARI:
    aprint("Running all computations without napari visualization...")"🧬 3D DAPI Multi-Scale Decomposition Demo (napari disabled)")
    aprint("Running all computations without napari visualization...")
    )

# ======= Demo knobs =======
SCALES = [1, 2, 4, 8]  # Scale factors to use
N_ITERS = 3000  # Number of optimization iterations
DEVICE = None  # None -> auto; or "cuda"/"cpu"/"mps"
ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
DAPI_CHANNEL = 1  # DAPI is typically channel 1 (0-indexed)
TIME_POINT = 0  # Use first time point
# NO DOWNSCALING - using full resolution!
# ==========================

# Setup Arbol
Arbol.max_depth = 3


with asection("3D DAPI Multi-Scale Decomposition Demo"):
    aprint("Running all computations without napari visualization...")"🧬 Real microscopy data: DAPI-stained nuclei from IDR")
    aprint("Running all computations without napari visualization...")f"📦 Data source: {ZARR_URL}")
    aprint("Running all computations without napari visualization...")"⚠️  Using FULL RESOLUTION (no downscaling) for best quality")

    with asection("Loading DAPI data from zarr"):
        aprint("Running all computations without napari visualization...")"Loading data from remote zarr store...")
        aprint("Running all computations without napari visualization...")"Note: Remote data loading may take a moment...")
        try:
            # Open remote zarr store via fsspec
            import fsspec

            mapper = fsspec.get_mapper(ZARR_URL)

            # Try opening as a group first
            try:
                store = zarr.open_group(mapper, mode="r")
                aprint("Running all computations without napari visualization...")"Zarr group opened successfully")
            except (zarr.errors.PathNotFoundError, zarr.errors.GroupNotFoundError):
                # Try as direct array
                store = zarr.open_array(mapper, mode="r")
                aprint("Running all computations without napari visualization...")"Zarr array opened successfully")

            # OME-ZARR format: access the '0' array (highest resolution)
            data = store["0"]
            full_shape = data.shape
            aprint("Running all computations without napari visualization...")f"OME-ZARR data shape: {full_shape}")
            aprint("Running all computations without napari visualization...")f"Data type: {data.dtype}")

            # OME-ZARR typically uses (T, C, Z, Y, X) format
            if len(full_shape) == 5:
                n_time, n_channels, z_size, y_size, x_size = full_shape
                aprint("Running all computations without napari visualization...")
                    f"Detected OME-ZARR 5D: T={n_time}, C={n_channels}, Z={z_size}, Y={y_size}, X={x_size}"
                )

                # Extract DAPI channel
                if DAPI_CHANNEL >= n_channels:
                    aprint("Running all computations without napari visualization...")
                        f"⚠ Warning: Requested channel {DAPI_CHANNEL} but only {n_channels} available"
                    )
                    aprint("Running all computations without napari visualization...")"Using channel 0 instead")
                    DAPI_CHANNEL = 0

                aprint("Running all computations without napari visualization...")
                    f"Extracting time={TIME_POINT}, channel={DAPI_CHANNEL} (DAPI)..."
                )

                # Load FULL RESOLUTION volume (no downscaling!)
                aprint("Running all computations without napari visualization...")
                    f"Loading FULL RESOLUTION volume: Z={z_size}, Y={y_size}, X={x_size}"
                )
                V = data[TIME_POINT, DAPI_CHANNEL, :, :, :]
                V = np.array(V, dtype=np.float32)
                aprint("Running all computations without napari visualization...")f"Loaded full resolution: {V.shape}")

            elif len(full_shape) == 4:
                # (C, Z, Y, X) format
                n_channels, z_size, y_size, x_size = full_shape
                aprint("Running all computations without napari visualization...")
                    f"Detected 4D: C={n_channels}, Z={z_size}, Y={y_size}, X={x_size}"
                )

                if DAPI_CHANNEL >= n_channels:
                    aprint("Running all computations without napari visualization...")f"⚠ Warning: Using channel 0 instead of {DAPI_CHANNEL}")
                    DAPI_CHANNEL = 0

                aprint("Running all computations without napari visualization...")f"Extracting channel={DAPI_CHANNEL} (DAPI)...")

                # Load FULL RESOLUTION volume (no downscaling!)
                aprint("Running all computations without napari visualization...")
                    f"Loading FULL RESOLUTION volume: Z={z_size}, Y={y_size}, X={x_size}"
                )
                V = data[DAPI_CHANNEL, :, :, :]
                V = np.array(V, dtype=np.float32)
                aprint("Running all computations without napari visualization...")f"Loaded full resolution: {V.shape}")

            elif len(full_shape) == 3:
                # Single channel, just ZYX
                z_size, y_size, x_size = full_shape
                aprint("Running all computations without napari visualization...")f"Detected 3D: Z={z_size}, Y={y_size}, X={x_size}")

                # Load FULL RESOLUTION volume (no downscaling!)
                aprint("Running all computations without napari visualization...")
                    f"Loading FULL RESOLUTION volume: Z={z_size}, Y={y_size}, X={x_size}"
                )
                V = data[:, :, :]
                V = np.array(V, dtype=np.float32)
                aprint("Running all computations without napari visualization...")f"Loaded full resolution: {V.shape}")
            else:
                raise ValueError(
                    f"Unexpected data shape: {full_shape}. Expected 3D, 4D, or 5D (OME-ZARR)."
                )

            # Normalize to [0, 1] range for consistency
            V_min, V_max = V.min(), V.max()
            if V_max > V_min:
                V = (V - V_min) / (V_max - V_min)
            else:
                aprint("Running all computations without napari visualization...")"⚠ Warning: Uniform data, using constant value")
                V = np.ones_like(V) * 0.5

            aprint("Running all computations without napari visualization...")f"Loaded DAPI volume: {V.shape} = {V.size:,} voxels")
            aprint("Running all computations without napari visualization...")f"Intensity range: [{V.min():.4f}, {V.max():.4f}]")
            aprint("Running all computations without napari visualization...")f"Volume size: {V.nbytes / 1024 / 1024:.2f} MB")

        except Exception as e:
            aprint("Running all computations without napari visualization...")f"❌ Error loading zarr data: {e}")
            aprint("Running all computations without napari visualization...")"Falling back to synthetic phantom data for demo purposes")

            # Create synthetic data as fallback
            shape_3d = (64, 64, 64)
            V = np.zeros(shape_3d, dtype=np.float32)

            # Add nucleus-like blobs
            n_nuclei = 10
            for i in range(n_nuclei):
                center = [np.random.uniform(5, s - 5) for s in shape_3d]
                sigma = np.random.uniform(4.0, 8.0)
                amplitude = np.random.uniform(0.6, 1.0)

                grids = np.meshgrid(*[np.arange(s) for s in shape_3d], indexing="ij")
                dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
                blob = amplitude * np.exp(-dist_sq / (2 * sigma**2))
                V += blob

            V = np.clip(V, 0, 1).astype(np.float32)
            aprint("Running all computations without napari visualization...")f"Created synthetic DAPI-like volume: {V.shape}")

    with asection(f"Decomposing into {len(SCALES)} scales"):
        aprint("Running all computations without napari visualization...")f"Scales: {SCALES}")
        aprint("Running all computations without napari visualization...")f"Parameters: n_iters={N_ITERS}, full resolution")
        aprint("Running all computations without napari visualization...")"Note: Full resolution may take longer to process")

        # Decompose with movie recording enabled
        scales_list, stats = decompose_image(
            V,
            scales=SCALES,
            n_iters=N_ITERS,
            napari_movie=(not NO_NAPARI),
            movie_every=20,  # Record every 20 iterations
            movie_max_frames=200,  # Limit to 200 frames
            device=DEVICE,
            verbose=True,
        )

        aprint("Running all computations without napari visualization...")"Decomposition complete!")
        aprint("Running all computations without napari visualization...")f"Final reconstruction error: {stats['final_error']:.6e}")
        aprint("Running all computations without napari visualization...")f"Best reconstruction error: {stats['best_error']:.6e}")
        aprint("Running all computations without napari visualization...")f"Time elapsed: {stats['time_seconds']:.2f} seconds")
        aprint("Running all computations without napari visualization...")f"Converged: {stats['converged']}")
        if stats["converged"]:
            aprint("Running all computations without napari visualization...")
                f"  → Early convergence at iteration {stats['actual_iters']}/{N_ITERS}"
            )

    with asection("Preparing visualization data"):
        # Upsample all scales to original resolution for visualization
        # Use same interpolation as optimization
        scales_upsampled = []
        interpolation_mode = stats.get("interpolation", "cubic")
        aprint("Running all computations without napari visualization...")
            f"  Using '{interpolation_mode}' interpolation for upsampling (matches optimization)"
        )
        for i, (scale, vol_scale) in enumerate(zip(SCALES, scales_list)):
            if vol_scale.shape != V.shape:
                aprint("Running all computations without napari visualization...")
                    f"  Upsampling scale {scale}x from {vol_scale.shape} to {V.shape}"
                )
                vol_upsampled = upsample_for_visualization(
                    vol_scale, V.shape, interpolation_mode
                )
            else:
                vol_upsampled = vol_scale
            scales_upsampled.append(vol_upsampled)

        # Compute reconstruction
        reconstruction = np.sum(scales_upsampled, axis=0)

        # Compute residual
        residual = V - reconstruction
        abs_residual = np.abs(residual)

        aprint("Running all computations without napari visualization...")f"Reconstruction MSE: {np.mean((reconstruction - V) ** 2):.6e}")
        aprint("Running all computations without napari visualization...")f"Max absolute residual: {abs_residual.max():.6e}")

    # Energy distribution analysis
    energy_dist = stats["energy_distribution"]
    aprint("Running all computations without napari visualization...")"\n" + "=" * 60)
    aprint("Running all computations without napari visualization...")"Energy Distribution (coarse → fine):")
    aprint("Running all computations without napari visualization...")"=" * 60)
    for i, (scale, energy_frac) in enumerate(zip(SCALES, energy_dist)):
        energy_pct = energy_frac * 100
        bar_length = int(energy_pct / 2)  # Scale for visualization
        bar = "█" * bar_length
        shape_str = "×".join(str(s) for s in scales_list[i].shape)
        aprint("Running all computations without napari visualization...")f"Scale {scale:2d}x: {energy_pct:5.1f}% {bar} [{shape_str}]")
    aprint("Running all computations without napari visualization...")f"Total: {sum(energy_dist) * 100:.1f}%")
    aprint("Running all computations without napari visualization...")"=" * 60)

    # Napari visualization
    aprint("Running all computations without napari visualization...")"\nLaunching 3D napari viewer...")
    viewer = napari.Viewer(
        title="3D DAPI Multi-Scale Decomposition (Full Resolution)", ndisplay=3
    )

    # Determine contrast limits from original volume
    contrast_limits = [0, float(V.max())]

    # Add original volume
    viewer.add_image(
        V,
        name="DAPI (input, full res)",
        colormap="gray",
        contrast_limits=contrast_limits,
        rendering="mip",  # Maximum intensity projection
    )

    # Add reconstruction
    viewer.add_image(
        reconstruction,
        name="reconstruction",
        colormap="cyan",
        contrast_limits=contrast_limits,
        rendering="mip",
        opacity=0.8,
        visible=True,
    )

    # Add each scale component (upsampled)
    for i, (scale, vol_upsampled) in enumerate(zip(SCALES, scales_upsampled)):
        energy_pct = energy_dist[i] * 100
        viewer.add_image(
            vol_upsampled,
            name=f"scale_{scale}x ({energy_pct:.1f}%)",
            colormap="turbo",
            contrast_limits=contrast_limits,
            blending="additive",
            opacity=0.7,
            rendering="mip",
            visible=True,
        )

    # Add residual
    viewer.add_image(
        abs_residual,
        name="absolute_residual",
        colormap="inferno",
        contrast_limits=[0, max(1e-12, float(abs_residual.max()))],
        rendering="mip",
        visible=True,
    )

    # Add signed residual
    viewer.add_image(
        residual,
        name="signed_residual",
        colormap="bwr",
        contrast_limits=[-abs_residual.max(), abs_residual.max()],
        rendering="mip",
        visible=False,  # Hidden by default
    )

    # Try to label axes
    try:
        viewer.dims.axis_labels = ["z", "y", "x"]
    except Exception:
        pass

    # Set better 3D camera view
    viewer.camera.angles = (45, 45, 45)
    viewer.camera.zoom = 1.5

    # Enable tile/grid mode for comparison
    viewer.grid.enabled = True
    viewer.grid.shape = (-1, 3)  # Auto rows, 3 columns

    # Set up text overlay
    viewer.text_overlay.visible = True
    viewer.text_overlay.text = (
        f"3D DAPI Multi-Scale Decomposition (FULL RESOLUTION) | "
        f"Volume: {V.shape[0]}×{V.shape[1]}×{V.shape[2]} | "
        f"Scales: {SCALES} | "
        f"Energy: {' → '.join([f'{e:.1%}' for e in energy_dist])} | "
        f"MSE: {stats['final_error']:.6e}"
    )

    # Console tips
    aprint("Running all computations without napari visualization...")"\n🎮 3D Navigation Tips:")
    aprint("Running all computations without napari visualization...")"  • Using FULL RESOLUTION data (no downscaling)")
    aprint("Running all computations without napari visualization...")"  • Opened in tile/grid mode for side-by-side comparison")
    aprint("Running all computations without napari visualization...")"  • All scales use same contrast limits for consistent comparison")
    aprint("Running all computations without napari visualization...")"  • Use mouse + Shift to rotate the 3D view")
    aprint("Running all computations without napari visualization...")"  • Toggle layers on/off to compare scales")
    aprint("Running all computations without napari visualization...")"  • 'DAPI (input, full res)' = original DAPI channel at full resolution")
    aprint("Running all computations without napari visualization...")"  • 'reconstruction' = sum of all scales")
    aprint("Running all computations without napari visualization...")f"  • 'scale_Nx' = individual scale components (N={SCALES})")
    aprint("Running all computations without napari visualization...")"  • MIP rendering = Maximum Intensity Projection")

    aprint("Running all computations without napari visualization...")"\n🎯 Energy Distribution Insights:")
    coarse_energy = energy_dist[-1]
    fine_energy = energy_dist[0]
    if coarse_energy > 0.5:
        aprint("Running all computations without napari visualization...")f"  ✓ Good: {coarse_energy:.1%} energy in coarsest scale")
        aprint("Running all computations without napari visualization...")
            "    → Low-frequency 3D nuclear structures captured at coarse resolution"
        )
    elif fine_energy > 0.5:
        aprint("Running all computations without napari visualization...")
            f"  ⚠ Warning: {fine_energy:.1%} energy in finest scale (trivial solution)"
        )
        aprint("Running all computations without napari visualization...")"    → Try increasing alpha or energy_weight")
    else:
        aprint("Running all computations without napari visualization...")"  → Energy well distributed across scales")

    aprint("Running all computations without napari visualization...")"\n📊 Compression Potential:")
    # Calculate storage requirements for each scale
    FLOAT_BITS = 32
    VOLUME_BITS = V.size * FLOAT_BITS
    total_compressed_bits = 0
    for i, scale in enumerate(SCALES):
        scale_voxels = scales_list[i].size
        scale_bits = scale_voxels * FLOAT_BITS
        total_compressed_bits += scale_bits
        compression_pct = 100.0 * (1.0 - (scale_bits / VOLUME_BITS))
        energy_pct = energy_dist[i] * 100
        aprint("Running all computations without napari visualization...")
            f"  Scale {scale:2d}x: {scale_voxels:>9,d} voxels | "
            f"Size vs full: {compression_pct:5.1f}% smaller | "
            f"Energy: {energy_pct:5.1f}%"
        )

    overall_compression = 100.0 * (1.0 - (total_compressed_bits / VOLUME_BITS))
    aprint("Running all computations without napari visualization...")"\n  Total multi-scale representation:")
    aprint("Running all computations without napari visualization...")f"    Original: {VOLUME_BITS:,} bits ({V.nbytes / 1024 / 1024:.2f} MB)")
    aprint("Running all computations without napari visualization...")
        f"    Multi-scale: {total_compressed_bits:,} bits ({total_compressed_bits / 8 / 1024 / 1024:.2f} MB)"
    )
    aprint("Running all computations without napari visualization...")f"    Compression: {overall_compression:.1f}% smaller")

    napari.run()

    # Show optimization convergence movie
    if stats["movie_frames"] is not None:
        aprint("Running all computations without napari visualization...")"\n🎬 Showing optimization convergence movie...")
        # Pass interpolation mode from stats to ensure movie matches optimization
        interpolation_mode = stats.get("interpolation", "cubic")
        show_optimization_movie(
            stats["movie_frames"], V.shape, interpolation=interpolation_mode
        )

# Console summary
aprint("Running all computations without napari visualization...")"\n" + "=" * 60)
aprint("Running all computations without napari visualization...")"3D DAPI Decomposition Summary (FULL RESOLUTION)")
aprint("Running all computations without napari visualization...")"=" * 60)
aprint("Running all computations without napari visualization...")f"Input volume: {V.shape[0]}×{V.shape[1]}×{V.shape[2]} = {V.size:,} voxels")
aprint("Running all computations without napari visualization...")f"Volume size: {V.nbytes / 1024 / 1024:.2f} MB")
aprint("Running all computations without napari visualization...")f"Scales used: {SCALES}")
aprint("Running all computations without napari visualization...")f"Optimization: {N_ITERS} iterations in {stats['time_seconds']:.2f}s")
aprint("Running all computations without napari visualization...")f"Final MSE: {stats['final_error']:.6e}")
aprint("Running all computations without napari visualization...")f"Best MSE: {stats['best_error']:.6e}")
aprint("Running all computations without napari visualization...")f"Converged: {stats['converged']}")
aprint("Running all computations without napari visualization...")"")
aprint("Running all computations without napari visualization...")"Energy distribution:")
for i, (scale, energy_frac) in enumerate(zip(SCALES, energy_dist)):
    aprint("Running all computations without napari visualization...")f"  {scale:2d}x: {energy_frac * 100:5.1f}%")
aprint("Running all computations without napari visualization...")"=" * 60)
aprint("Running all computations without napari visualization...")"✅ 3D DAPI multi-scale decomposition complete at FULL RESOLUTION!")
