#!/usr/bin/env python3
"""
Simple 3D Gaussian Splat Demo (No GUI)

This demo demonstrates 3D Gaussian splat fitting on synthetic volumetric data.
It creates a 3D volume with multiple blob structures, fits Gaussian splats,
and analyzes compression performance without requiring napari.
"""

import numpy as np
from arbol import aprint
from scipy import ndimage
from skimage import data

from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy
from luxar.gsplats.utils.trils import tril_size, unpack_tril


def create_3d_test_volume(size=32, seed=42):
    """Create synthetic 3D volume using binary_blobs (same approach as 2D version)."""
    blobs = data.binary_blobs(
        length=size, blob_size_fraction=0.08, n_dim=3, volume_fraction=0.15, rng=seed
    ).astype(float)
    volume = ndimage.gaussian_filter(blobs, sigma=1.8).astype(np.float32)
    return volume


def analyze_compression_performance(V, params_full, amps, n_frames=10):
    """Analyze compression performance across different sparsity levels."""
    aprint("\n📊 Compression Performance Analysis")
    aprint(f"{'=' * 70}")

    # Calculate energy-based ranking for compression
    d = 3
    L_packed = params_full[:, d:]
    L_full = unpack_tril(L_packed, d)
    diag_prod = np.prod(L_full.diagonal(axis1=1, axis2=2), axis=1)
    energy_score = (amps**2) * (np.sqrt(np.pi) ** d) * diag_prod
    order = np.argsort(-energy_score)

    # Compute compression frames
    N = len(amps)
    keep_counts = np.unique(
        np.linspace(1, N, num=min(n_frames, N), endpoint=True).astype(int)
    )

    # Bit accounting
    FLOAT_BITS = 32
    FLOATS_PER_SPLAT = d + tril_size(d) + 1  # 3 + 6 + 1 = 10 floats per splat
    VOLUME_BITS = V.size * FLOAT_BITS

    aprint(f"Original volume: {V.shape} = {V.size:,} voxels")
    aprint(f"Raw storage: {VOLUME_BITS:,} bits ({FLOAT_BITS} bits/voxel)")
    aprint(
        f"Fitted splats: {N} splats × {FLOATS_PER_SPLAT} floats = {N * FLOATS_PER_SPLAT * FLOAT_BITS:,} bits"
    )
    aprint(f"{'=' * 70}")

    results = []
    for i, K in enumerate(keep_counts):
        idx = order[:K]

        # Compute reconstruction
        Vk = render_gaussians_numpy(
            V.shape, params_full[idx], amps[idx], truncate=3.0
        )

        # Quality metrics
        rel_error = np.linalg.norm(V - Vk) / (np.linalg.norm(V) + 1e-12)

        # Compression metrics
        model_bits = K * FLOATS_PER_SPLAT * FLOAT_BITS
        compression_ratio = VOLUME_BITS / model_bits if model_bits > 0 else float("inf")
        bits_per_voxel = model_bits / V.size

        results.append(
            {
                "splats": K,
                "rel_error": rel_error,
                "model_bits": model_bits,
                "compression_ratio": compression_ratio,
                "bits_per_voxel": bits_per_voxel,
            }
        )

        aprint(
            f"Splats: {K:3d} | Error: {rel_error:.4f} | "
            f"Compression: {compression_ratio:6.1f}x | "
            f"BPV: {bits_per_voxel:6.3f}"
        )

    return results


def main():
    """Run 3D Gaussian splat demonstration."""
    aprint("🔬 3D Gaussian Splat Fitting Demo")
    aprint("=" * 50)

    # Create test volume
    aprint("1️⃣ Creating 3D test volume...")
    V = create_3d_test_volume(size=32)  # Small size for demo speed
    aprint(f"   Volume shape: {V.shape}")
    aprint(f"   Value range: [{V.min():.4f}, {V.max():.4f}]")

    # Find candidates
    aprint("\n2️⃣ Finding candidate centers...")
    centers = find_candidates_overcomplete_nd(
        V,
        scales=(0.6, 1.0, 1.5, 2.2),
        peaks_per_scale=200,  # Reasonable for demo
        percentile_thresh=75,
        min_dist=2.0,
        add_intensity_grid=True,
        grid_step=[2, 2, 2],
    )
    aprint(f"   Found {len(centers)} candidates")

    if len(centers) == 0:
        aprint("❌ No candidates found. Try adjusting parameters.")
        return

    # Fit Gaussian splats
    aprint("\n3️⃣ Fitting 3D Gaussian splats...")
    params_full, amps, stats = fit_gaussian_splats(
        V,
        centers_overcomplete=centers,
        init_sigma_vox=1.2,
        n_iters=200,  # Moderate for demo
        lr=0.12,
        loss_type="poisson",
        sigma_min_diag=[0.4, 0.4, 0.4],
        sigma_max_diag=[3.0, 3.0, 3.0],
        truncate=3.0,
        verbose=False,  # Reduce output for demo
    )

    aprint(f"   Successfully fitted {len(amps)} splats")

    if len(amps) == 0:
        aprint("❌ No splats fitted successfully.")
        return

    # Analyze results
    aprint("\n4️⃣ Analyzing splat characteristics...")
    d = 3
    L_packed = params_full[:, d:]
    L_full = unpack_tril(L_packed, d)

    # Statistics
    center_coords = params_full[:, :d]
    amp_stats = {
        "min": amps.min(),
        "max": amps.max(),
        "mean": amps.mean(),
        "std": amps.std(),
    }

    # Covariance analysis
    det_values = []
    for i in range(len(L_full)):
        det = np.linalg.det(L_full[i] @ L_full[i].T)
        det_values.append(det)
    det_values = np.array(det_values)

    aprint(
        f"   Amplitude stats: min={amp_stats['min']:.4f}, max={amp_stats['max']:.4f}, "
        f"mean={amp_stats['mean']:.4f}"
    )
    aprint(
        f"   Volume stats: min_det={det_values.min():.6f}, max_det={det_values.max():.6f}"
    )
    aprint(
        f"   Centers span: z=[{center_coords[:, 0].min():.1f}, {center_coords[:, 0].max():.1f}], "
        f"y=[{center_coords[:, 1].min():.1f}, {center_coords[:, 1].max():.1f}], "
        f"x=[{center_coords[:, 2].min():.1f}, {center_coords[:, 2].max():.1f}]"
    )

    # Compression analysis
    analyze_compression_performance(V, params_full, amps, n_frames=8)

    # Final reconstruction
    aprint("\n5️⃣ Final reconstruction quality...")
    V_recon = render_gaussians_numpy(V.shape, params_full, amps, truncate=3.0)

    final_error = np.linalg.norm(V - V_recon) / np.linalg.norm(V)
    peak_preserved = V_recon.max() / V.max()

    aprint(f"   Final relative L2 error: {final_error:.4f}")
    aprint(f"   Peak intensity preserved: {peak_preserved:.1%}")
    aprint(f"   Original volume: {V.size:,} voxels × 32 bits = {V.size * 32:,} bits")
    aprint(
        f"   Splat model: {len(amps)} splats × {3 + tril_size(3) + 1} floats × 32 bits = {len(amps) * 10 * 32:,} bits"
    )

    compression_ratio = (V.size * 32) / (len(amps) * 10 * 32)
    aprint(f"   🎯 Compression ratio: {compression_ratio:.1f}x")

    # Quality assessment
    if final_error < 0.2:
        quality = "Excellent"
    elif final_error < 0.4:
        quality = "Good"
    elif final_error < 0.6:
        quality = "Fair"
    else:
        quality = "Poor"

    aprint("\n🎉 Demo Complete!")
    aprint(f"   Quality: {quality} (rel_error = {final_error:.4f})")
    aprint(f"   Efficiency: {compression_ratio:.1f}x compression")
    aprint("\n💡 To visualize in 3D, run the full demo with napari:")
    aprint(
        "   hatch run python packages/luxar/src/luxar/gsplats/demo/demo_splats_3d_napari_example.py"
    )


if __name__ == "__main__":
    main()
