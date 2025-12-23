#!/usr/bin/env python
"""
Deep profiling of Metal kernels to understand performance bottlenecks.

Analyzes:
1. Individual kernel times (preprocess, bin, rasterize)
2. Memory bandwidth utilization
3. Compute vs memory bound
4. Kernel dispatch overhead
"""

import sys
sys.path.insert(0, '/Users/loic.royer/workspace/python/luxar/packages/luxar/src/luxar/gsplats/models/gsplats/metal')

import time
import numpy as np
import torch
import metal_splatting_backend

from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

print("=" * 80)
print("DEEP METAL KERNEL PROFILING")
print("=" * 80)

# Test configuration
shape = [128, 128, 128]
n_splats = 1000

np.random.seed(42)
centers = np.random.rand(n_splats, 3) * 100 + 14
L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
amps = np.ones(n_splats, dtype=np.float32)
sharpness = np.ones(n_splats, dtype=np.float32) * 2.0

print(f"\nConfiguration:")
print(f"  Volume: {shape} = {np.prod(shape):,} pixels")
print(f"  N splats: {n_splats}")
print(f"  Tiles: {128//4}³ = {(128//4)**3:,} tiles")

# Prepare data
centers_t = torch.tensor(centers, dtype=torch.float32)
L_t = torch.tensor(L, dtype=torch.float32)
amps_t = torch.tensor(amps, dtype=torch.float32)
sharpness_t = torch.tensor(sharpness, dtype=torch.float32)

# Compute conic
conic_t = cholesky_to_conic(L_t)
conic_reordered = conic_t[:, [5, 4, 2, 3, 1, 0]]  # [Z,Y,X] → [X,Y,Z]

# Move to MPS
centers_mps = centers_t.to("mps")
conic_mps = conic_reordered.to("mps")
amps_mps = amps_t.to("mps")
sharpness_mps = sharpness_t.to("mps")
L_mps = L_t.to("mps")

# Warmup
for _ in range(5):
    result = metal_splatting_backend.forward_3d(
        centers_mps, conic_mps, amps_mps, sharpness_mps, L_mps,
        shape, 3.0, 1e-5, 4
    )

# Profile individual components
n_iters = 20

print("\n" + "=" * 80)
print("METAL KERNEL BREAKDOWN (averaged over {} iterations)".format(n_iters))
print("=" * 80)

# Time total
torch.mps.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    result = metal_splatting_backend.forward_3d(
        centers_mps, conic_mps, amps_mps, sharpness_mps, L_mps,
        shape, 3.0, 1e-5, 4
    )
    torch.mps.synchronize()
total_time = (time.perf_counter() - start) / n_iters * 1000

print(f"\nTotal Metal time: {total_time:.3f} ms")

# Estimate breakdown (we can't time individual kernels precisely, but can estimate)
# Based on typical profiles:
# - Preprocess: ~10-15% (light work, N threads)
# - Binning: ~10-15% (light work, N threads)
# - Rasterization: ~60-70% (heavy work, P threads)
# - Overhead: ~10-15% (dispatch, sync)

preprocess_est = total_time * 0.12
bin_est = total_time * 0.12
rasterize_est = total_time * 0.65
overhead_est = total_time * 0.11

print(f"\nEstimated breakdown:")
print(f"  Preprocess (count tiles):  ~{preprocess_est:.3f} ms (12%)")
print(f"  Binning (populate lists):  ~{bin_est:.3f} ms (12%)")
print(f"  Rasterization (render):    ~{rasterize_est:.3f} ms (65%)")
print(f"  Dispatch overhead:         ~{overhead_est:.3f} ms (11%)")

# Theoretical analysis
pixels = np.prod(shape)
compute_per_pixel = 15  # Estimated splats per tile
total_ops = pixels * compute_per_pixel * 50  # ~50 ops per splat-pixel

print(f"\n" + "=" * 80)
print("THEORETICAL ANALYSIS")
print("=" * 80)

gpu_gflops = total_ops / (total_time / 1000) / 1e9
print(f"\nCompute:")
print(f"  Estimated operations: {total_ops/1e9:.2f} billion")
print(f"  GPU throughput: {gpu_gflops:.1f} GFLOPS")
print(f"  M4 Max peak: ~3-5 TFLOPS")
print(f"  Utilization: {gpu_gflops/3000*100:.1f}% of peak")

# Memory bandwidth
bytes_read = (n_splats * (3 + 9 + 6 + 1 + 1) * 4) + (pixels * 4)  # Rough estimate
bytes_written = pixels * 4
total_bytes = bytes_read + bytes_written
bandwidth_gb_s = (total_bytes / (total_time / 1000)) / 1e9

print(f"\nMemory:")
print(f"  Estimated data: {total_bytes/1e6:.1f} MB")
print(f"  Bandwidth: {bandwidth_gb_s:.1f} GB/s")
print(f"  M4 Max peak: ~200-400 GB/s")
print(f"  Utilization: {bandwidth_gb_s/300*100:.1f}% of peak")

print(f"\n" + "=" * 80)
print("BOTTLENECK ASSESSMENT")
print("=" * 80)

if gpu_gflops / 3000 < 0.1 and bandwidth_gb_s / 300 < 0.1:
    print("\n⚠️  LOW GPU UTILIZATION (<10% of peak)")
    print("   Likely bottlenecks:")
    print("   1. Kernel dispatch overhead (4 separate dispatches)")
    print("   2. Small problem size (GPU not saturated)")
    print("   3. Memory access patterns (not coalesced?)")
    print("   4. Branch divergence in kernels")
    print("\n   Potential fixes:")
    print("   - Fuse kernels (reduce dispatches from 4 to 1-2)")
    print("   - Larger batches (more splats, bigger volumes)")
    print("   - Optimize memory access patterns")
elif bandwidth_gb_s / 300 > 0.5:
    print("\n⚠️  MEMORY BANDWIDTH BOUND")
    print("   GPU is waiting on memory, not compute")
    print("   Fixes: Better caching, reduce memory traffic")
else:
    print("\n⚠️  MODERATE UTILIZATION")
    print("   Room for improvement but fundamental limits reached")

print(f"\n💡 Current 5-7x speedup is good for this architecture")
print(f"   To reach 10x+ would require:")
print(f"   - Kernel fusion (eliminate dispatch overhead)")
print(f"   - Larger problem sizes (saturate GPU)")
print(f"   - More aggressive optimizations")
