#!/usr/bin/env python
"""
Test Metal L→Conic fast path and measure speedup.

Compares:
1. Default: PyTorch L→Conic (safe, tested)
2. Fast path: Metal L→Conic (experimental, faster)
"""

import time
import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

print("=" * 80)
print("METAL L→CONIC FAST PATH BENCHMARK")
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

# Create models with different conic computation paths
model_pytorch_conic = GaussianSplatModelMetal(
    shape=shape, centers0=centers, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], truncate=3.0,
    use_metal_conic=False,  # PyTorch L→Conic (default)
    device="mps"
)

model_metal_conic = GaussianSplatModelMetal(
    shape=shape, centers0=centers, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], truncate=3.0,
    use_metal_conic=True,  # Metal L→Conic (fast path!)
    device="mps"
)

# Warmup
for _ in range(3):
    _ = model_pytorch_conic()
    _ = model_metal_conic()

# Benchmark PyTorch conic path
n_iters = 20
torch.mps.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    _ = model_pytorch_conic()
    torch.mps.synchronize()
pytorch_time = (time.perf_counter() - start) / n_iters * 1000

# Benchmark Metal conic path
torch.mps.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    _ = model_metal_conic()
    torch.mps.synchronize()
metal_time = (time.perf_counter() - start) / n_iters * 1000

speedup = pytorch_time / metal_time

print("\n" + "=" * 80)
print("FORWARD PASS BENCHMARK")
print("=" * 80)
print(f"  PyTorch L→Conic: {pytorch_time:>6.2f} ms/iter")
print(f"  Metal L→Conic:   {metal_time:>6.2f} ms/iter")
print(f"  Speedup:         {speedup:>6.2f}x")
print(f"  Time saved:      {pytorch_time - metal_time:>6.2f} ms")

# Test backward too
torch.mps.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    output = model_pytorch_conic()
    loss = output.sum()
    loss.backward()
    model_pytorch_conic.zero_grad()
    torch.mps.synchronize()
pytorch_bwd_time = (time.perf_counter() - start) / n_iters * 1000

torch.mps.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    output = model_metal_conic()
    loss = output.sum()
    loss.backward()
    model_metal_conic.zero_grad()
    torch.mps.synchronize()
metal_bwd_time = (time.perf_counter() - start) / n_iters * 1000

bwd_speedup = pytorch_bwd_time / metal_bwd_time

print("\n" + "=" * 80)
print("FORWARD + BACKWARD BENCHMARK")
print("=" * 80)
print(f"  PyTorch L→Conic: {pytorch_bwd_time:>6.2f} ms/iter")
print(f"  Metal L→Conic:   {metal_bwd_time:>6.2f} ms/iter")
print(f"  Speedup:         {bwd_speedup:>6.2f}x")

# Verify numerical accuracy
print("\n" + "=" * 80)
print("ACCURACY CHECK")
print("=" * 80)

out_pytorch = model_pytorch_conic().cpu()
out_metal = model_metal_conic().cpu()

max_diff = (out_pytorch - out_metal).abs().max().item()
mean_diff = (out_pytorch - out_metal).abs().mean().item()

print(f"  Max difference:  {max_diff:.6e}")
print(f"  Mean difference: {mean_diff:.6e}")

if max_diff < 1e-4:
    print(f"  ✓ Excellent agreement (< 1e-4)")
elif max_diff < 1e-3:
    print(f"  ✓ Good agreement (< 1e-3)")
elif max_diff < 0.01:
    print(f"  ⚠️  Minor differences (< 0.01)")
else:
    print(f"  ✗ Significant differences (> 0.01)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Metal L→Conic fast path provides:")
print(f"  • {speedup:.2f}x speedup on forward pass")
print(f"  • {bwd_speedup:.2f}x speedup on forward+backward")
print(f"  • {max_diff:.2e} max numerical difference")

if speedup > 1.3 and max_diff < 0.01:
    print(f"\n✅ RECOMMENDED: Enable use_metal_conic=True for {speedup:.1f}x speedup!")
else:
    print(f"\n⚠️  Keep default use_metal_conic=False for safety")
