#!/usr/bin/env python
"""
Benchmark different tile sizes to find optimal configuration.

Tests tile sizes: 2, 4, 6, 8 on 128³ volume with varying splat counts.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

print("=" * 80)
print("METAL TILE SIZE OPTIMIZATION BENCHMARK")
print("=" * 80)

# Query Metal device capabilities
print("\nQuerying Metal GPU capabilities...")
import subprocess
try:
    # Get Metal device info via system_profiler
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if "Apple" in result.stdout:
        print("  GPU: Apple Silicon (M-series)")
        # Apple Silicon GPUs typically support 1024 threads/threadgroup
        max_threads = 1024
    else:
        max_threads = 1024  # Conservative default
except:
    max_threads = 1024

print(f"  Estimated max threads/threadgroup: {max_threads}")

# Calculate maximum tile size
import math
max_tile_size = int(math.pow(max_threads, 1/3))  # Cube root
print(f"  Maximum tile size: {max_tile_size} ({max_tile_size}³ = {max_tile_size**3} threads)")

# Test configuration
VOLUME_SIZE = 128
SHAPE = (VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE)
# Test tile sizes up to the maximum
TILE_SIZES = [2, 4, 6, 8, 10, max_tile_size] if max_tile_size > 10 else [2, 4, 6, 8, 10]
TILE_SIZES = sorted(list(set(TILE_SIZES)))  # Remove duplicates and sort
SPLAT_COUNTS = [500, 1000]  # Test with different splat densities
N_WARMUP = 3
N_ITERS = 10

print(f"\nConfiguration:")
print(f"  Volume: {SHAPE}")
print(f"  Tile sizes to test: {TILE_SIZES}")
print(f"  Splat counts: {SPLAT_COUNTS}")
print(f"  Iterations: {N_ITERS} (after {N_WARMUP} warmup)")


def benchmark_tile_size(n_splats: int, tile_size: int) -> dict:
    """Benchmark one configuration. Returns None if tile size too large."""
    try:
        # Create test data
        np.random.seed(42)
        centers = np.random.rand(n_splats, 3) * (VOLUME_SIZE * 0.8) + (VOLUME_SIZE * 0.1)
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

        # Create model with specified tile size
        model = GaussianSplatModelMetal(
            shape=SHAPE,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            tile_size=tile_size,
            device="mps",
        )
    except Exception as e:
        print(f"    ✗ tile_size={tile_size} failed: {e}")
        return None

    # Warmup
    for _ in range(N_WARMUP):
        _ = model()
        torch.mps.synchronize()

    # Benchmark forward only
    start = time.perf_counter()
    for _ in range(N_ITERS):
        _ = model()
        torch.mps.synchronize()
    fwd_time = (time.perf_counter() - start) / N_ITERS * 1000  # ms

    # Benchmark forward + backward
    start = time.perf_counter()
    for _ in range(N_ITERS):
        output = model()
        loss = output.sum()
        loss.backward()
        model.zero_grad()
        torch.mps.synchronize()
    total_time = (time.perf_counter() - start) / N_ITERS * 1000  # ms

    bwd_time = total_time - fwd_time

    return {
        "fwd_ms": fwd_time,
        "bwd_ms": bwd_time,
        "total_ms": total_time,
    }


# Run benchmarks
print("\n" + "=" * 80)
print("BENCHMARK RESULTS")
print("=" * 80)

all_results = {}

for n_splats in SPLAT_COUNTS:
    print(f"\n{'=' * 80}")
    print(f"N_SPLATS = {n_splats}")
    print(f"{'=' * 80}")

    print(f"\n{'Tile Size':<12} {'Threads':<10} {'Forward':<12} {'Backward':<12} {'Total':<12} {'vs Tile=4':<12}")
    print("-" * 80)

    results_for_n = {}

    for tile_size in TILE_SIZES:
        threads = tile_size ** 3
        result = benchmark_tile_size(n_splats, tile_size)

        if result is None:
            # Tile size too large or other error
            print(f"{tile_size:<12} {threads:<10} {'FAILED':<12} {'(GPU limit exceeded?)':<25}")
            continue

        results_for_n[tile_size] = result

        # Compare to baseline (tile_size=4)
        baseline = results_for_n.get(4, result)
        speedup = baseline["total_ms"] / result["total_ms"]
        speedup_str = f"{speedup:.2f}x" if tile_size != 4 else "baseline"

        print(
            f"{tile_size:<12} {threads:<10} {result['fwd_ms']:>8.2f} ms {result['bwd_ms']:>8.2f} ms "
            f"{result['total_ms']:>8.2f} ms {speedup_str:>12}"
        )

    all_results[n_splats] = results_for_n

# Summary and recommendation
print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATION")
print("=" * 80)

for n_splats, results in all_results.items():
    fastest_tile = min(results.keys(), key=lambda t: results[t]["total_ms"])
    fastest_time = results[fastest_tile]["total_ms"]

    print(f"\nFor {n_splats} splats:")
    print(f"  Fastest: tile_size={fastest_tile} ({fastest_tile}³={fastest_tile**3} threads)")
    print(f"  Time: {fastest_time:.2f} ms/iter")

    if fastest_tile != 4:
        improvement = (results[4]["total_ms"] / fastest_time - 1) * 100
        print(f"  Improvement over default (4): {improvement:.1f}% faster")

# Overall recommendation
all_fastest = [min(r.keys(), key=lambda t: r[t]["total_ms"]) for r in all_results.values()]
if all(t == all_fastest[0] for t in all_fastest):
    print(f"\n{'=' * 80}")
    print(f"✨ RECOMMENDATION: Use tile_size={all_fastest[0]} for best performance!")
    print(f"{'=' * 80}")
else:
    print(f"\n{'=' * 80}")
    print(f"⚠️  Optimal tile size varies with splat count")
    print(f"   Keep default tile_size=4 for good balance")
    print(f"{'=' * 80}")
