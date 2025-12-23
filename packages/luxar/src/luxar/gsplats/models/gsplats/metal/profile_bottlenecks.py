#!/usr/bin/env python
"""
Profile Metal backend to identify bottlenecks.

Breaks down time spent in each component to understand why we're only
getting 5-7x instead of the spec's target 10-50x.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("METAL BACKEND BOTTLENECK ANALYSIS")
print("=" * 80)

# Test configuration
shape = (128, 128, 128)
n_splats = 1000

np.random.seed(42)
centers = np.random.rand(n_splats, 3) * 100 + 14
L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
amps = np.ones(n_splats, dtype=np.float32)

print(f"\nConfiguration:")
print(f"  Volume: {shape}")
print(f"  N splats: {n_splats}")
print(f"  Pixels: {np.prod(shape):,}")

# Create models
model_metal = GaussianSplatModelMetal(
    shape=shape, centers0=centers, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], truncate=3.0, device="mps"
)

model_cpu = GaussianSplatModel(
    shape=shape, centers0=centers, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], truncate=3.0, device="cpu"
)

# Warmup
for _ in range(3):
    _ = model_metal()
    _ = model_cpu()

print("\n" + "=" * 80)
print("PROFILING METAL BACKEND")
print("=" * 80)

# Profile Metal forward pass with detailed timing
centers_param, Ls_param, amps_param, sharpness_param = model_metal.current_params()

# Time: Get parameters
t0 = time.perf_counter()
centers, Ls, amps, sharpness = model_metal.current_params()
torch.mps.synchronize()
t1 = time.perf_counter()
get_params_time = (t1 - t0) * 1000

# Time: Cholesky to conic (happens in Python)
from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

t0 = time.perf_counter()
Ls_for_conic = Ls.detach().clone().requires_grad_(True)
conic = cholesky_to_conic(Ls_for_conic)
torch.mps.synchronize()
t1 = time.perf_counter()
conic_time = (t1 - t0) * 1000

# Time: Conic reordering
t0 = time.perf_counter()
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
t1 = time.perf_counter()
reorder_time = (t1 - t0) * 1000

# Time: Data transfer to MPS
t0 = time.perf_counter()
centers_mps = centers.contiguous().to("mps")
conic_mps = conic_reordered.detach().contiguous().to("mps")
amps_mps = amps.contiguous().to("mps")
sharpness_mps = sharpness.contiguous().to("mps")
Ls_mps = Ls.contiguous().to("mps")
torch.mps.synchronize()
t1 = time.perf_counter()
transfer_time = (t1 - t0) * 1000

# Time: Metal kernels (total)
t0 = time.perf_counter()
output = model_metal()
torch.mps.synchronize()
t1 = time.perf_counter()
total_metal_time = (t1 - t0) * 1000

# Estimate Metal kernel time (subtract overhead)
metal_kernel_time = total_metal_time - get_params_time - conic_time - transfer_time - reorder_time

print(f"\nMetal Forward Pass Breakdown ({total_metal_time:.2f} ms total):")
print(f"  1. Get parameters:      {get_params_time:>6.2f} ms ({get_params_time/total_metal_time*100:>5.1f}%)")
print(f"  2. Cholesky→Conic:      {conic_time:>6.2f} ms ({conic_time/total_metal_time*100:>5.1f}%)")
print(f"  3. Conic reordering:    {reorder_time:>6.2f} ms ({reorder_time/total_metal_time*100:>5.1f}%)")
print(f"  4. Data transfer (MPS): {transfer_time:>6.2f} ms ({transfer_time/total_metal_time*100:>5.1f}%)")
print(f"  5. Metal kernels:       {metal_kernel_time:>6.2f} ms ({metal_kernel_time/total_metal_time*100:>5.1f}%)")

# Compare with CPU
t0 = time.perf_counter()
output_cpu = model_cpu()
t1 = time.perf_counter()
cpu_time = (t1 - t0) * 1000

speedup = cpu_time / total_metal_time

print(f"\n" + "=" * 80)
print(f"COMPARISON")
print(f"=" * 80)
print(f"  CPU total:   {cpu_time:>6.2f} ms")
print(f"  Metal total: {total_metal_time:>6.2f} ms")
print(f"  Speedup:     {speedup:>6.2f}x")

print(f"\n" + "=" * 80)
print(f"BOTTLENECK ANALYSIS")
print(f"=" * 80)

# Identify bottlenecks
overhead = get_params_time + conic_time + transfer_time + reorder_time
print(f"\nOverhead (non-Metal): {overhead:.2f} ms ({overhead/total_metal_time*100:.1f}%)")
print(f"Actual Metal work:    {metal_kernel_time:.2f} ms ({metal_kernel_time/total_metal_time*100:.1f}%)")

if conic_time > metal_kernel_time * 0.5:
    print(f"\n⚠️  BOTTLENECK: Cholesky→Conic taking {conic_time:.2f} ms")
    print(f"   This is done in PyTorch (not Metal)")
    print(f"   Potential speedup if moved to Metal: {total_metal_time/(total_metal_time-conic_time):.1f}x")

if transfer_time > metal_kernel_time * 0.3:
    print(f"\n⚠️  BOTTLENECK: Data transfer taking {transfer_time:.2f} ms")
    print(f"   This is MPS tensor allocation/copying overhead")

print(f"\n💡 Theoretical maximum speedup if overhead eliminated:")
print(f"   {cpu_time/metal_kernel_time:.1f}x (vs current {speedup:.1f}x)")

# Check if CPU is actually fast
pixels = np.prod(shape)
cpu_mpixels_per_sec = pixels / (cpu_time / 1000) / 1e6
metal_mpixels_per_sec = pixels / (total_metal_time / 1000) / 1e6

print(f"\n" + "=" * 80)
print(f"THROUGHPUT")
print(f"=" * 80)
print(f"  CPU:   {cpu_mpixels_per_sec:.1f} Mpixels/sec")
print(f"  Metal: {metal_mpixels_per_sec:.1f} Mpixels/sec")
