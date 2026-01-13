#!/usr/bin/env python
"""
Performance benchmark comparing PyTorch CPU, PyTorch CUDA vanilla, and custom CUDA kernels.

This script measures forward pass performance across different configurations
to demonstrate the speedup achieved by the custom CUDA implementation.

Includes:
1. Main benchmark: CPU vs CUDA vanilla vs CUDA custom (FP32) vs CUDA custom (FP16)
2. Extended FP16 benchmark: Larger volumes to show memory bandwidth benefits

Usage:
    python -m luxar.gsplats.models.gsplats.cuda.benchmark

    # Or from the cuda directory:
    python benchmark.py

Results are printed to stdout in formatted tables.
"""

import time
from typing import Tuple

import numpy as np
import torch


def create_test_data(
    N: int, shape: Tuple[int, ...], seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random test data for benchmarking."""
    np.random.seed(seed)
    d = len(shape)
    centers = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 2) + 1
    L = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
    for i in range(N):
        L[i] *= np.random.uniform(0.5, 2.0)
    amps = np.random.rand(N).astype(np.float32) + 0.1
    return centers, L, amps


def benchmark(
    model: torch.nn.Module,
    n_warmup: int = 5,
    n_iters: int = 20,
    sync_cuda: bool = False,
) -> float:
    """
    Benchmark a model's forward pass.

    Returns:
        Average time per forward pass in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        _ = model()
        if sync_cuda:
            torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model()
        if sync_cuda:
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return elapsed


def run_benchmark(verbose: bool = True) -> dict:
    """
    Run the full benchmark suite.

    Args:
        verbose: If True, print results to stdout.

    Returns:
        Dictionary with benchmark results.
    """
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
        GaussianSplatModelCUDA,
    )
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run benchmark.")

    configs = [
        (100, (64, 64, 64), "Small 3D"),
        (500, (96, 96, 96), "Medium 3D"),
        (1000, (128, 128, 128), "Large 3D"),
        (100, (256, 256), "Small 2D"),
        (500, (512, 512), "Medium 2D"),
    ]

    results = {}

    if verbose:
        print("=" * 90)
        print(
            "PERFORMANCE BENCHMARK: CPU vs CUDA-vanilla vs CUDA-custom (FP32) vs CUDA-custom (FP16)"
        )
        print("=" * 90)

    for cfg in configs:
        N, shape, label = cfg
        centers, L, amps = create_test_data(N, shape)
        sigma_min = (0.5,) * len(shape)

        # CPU
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cpu",
        )
        cpu_time = benchmark(cpu_model, sync_cuda=False)

        # PyTorch CUDA vanilla (uses PyTorch ops on CUDA)
        cuda_vanilla_model = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
        )
        cuda_vanilla_time = benchmark(cuda_vanilla_model, sync_cuda=True)

        # Custom CUDA kernels (FP32)
        cuda_custom_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=False,
        )
        cuda_custom_time = benchmark(cuda_custom_model, sync_cuda=True)

        # Custom CUDA kernels (FP16)
        cuda_fp16_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=True,
        )
        cuda_fp16_time = benchmark(cuda_fp16_model, sync_cuda=True)

        # Calculate speedups
        vanilla_vs_cpu = cpu_time / cuda_vanilla_time
        custom_vs_cpu = cpu_time / cuda_custom_time
        custom_vs_vanilla = cuda_vanilla_time / cuda_custom_time
        fp16_vs_cpu = cpu_time / cuda_fp16_time
        fp16_vs_fp32 = cuda_custom_time / cuda_fp16_time

        voxels = int(np.prod(shape))

        results[label] = {
            "N": N,
            "shape": shape,
            "voxels": voxels,
            "cpu_ms": cpu_time,
            "cuda_vanilla_ms": cuda_vanilla_time,
            "cuda_custom_ms": cuda_custom_time,
            "cuda_fp16_ms": cuda_fp16_time,
            "vanilla_vs_cpu": vanilla_vs_cpu,
            "custom_vs_cpu": custom_vs_cpu,
            "custom_vs_vanilla": custom_vs_vanilla,
            "fp16_vs_cpu": fp16_vs_cpu,
            "fp16_vs_fp32": fp16_vs_fp32,
        }

        if verbose:
            print(f"\n{label} (N={N}, shape={shape}, {voxels:,} voxels):")
            print(f"  PyTorch CPU:          {cpu_time:8.2f} ms")
            print(
                f"  PyTorch CUDA vanilla: {cuda_vanilla_time:8.2f} ms  "
                f"({vanilla_vs_cpu:5.1f}x vs CPU)"
            )
            print(
                f"  Custom CUDA (FP32):   {cuda_custom_time:8.2f} ms  "
                f"({custom_vs_cpu:5.1f}x vs CPU, {custom_vs_vanilla:.1f}x vs vanilla)"
            )
            print(
                f"  Custom CUDA (FP16):   {cuda_fp16_time:8.2f} ms  "
                f"({fp16_vs_cpu:5.1f}x vs CPU, {fp16_vs_fp32:.2f}x vs FP32)"
            )

    if verbose:
        print("\n" + "=" * 90)
        print("\nSummary (times in ms):")
        print("-" * 90)
        print(
            f"{'Config':<12} {'CPU':<10} {'Vanilla':<10} {'FP32':<10} {'FP16':<10} "
            f"{'Speedup':<12} {'FP16/FP32':<10}"
        )
        print("-" * 90)
        for label, r in results.items():
            print(
                f"{label:<12} {r['cpu_ms']:<10.2f} {r['cuda_vanilla_ms']:<10.2f} "
                f"{r['cuda_custom_ms']:<10.2f} {r['cuda_fp16_ms']:<10.2f} "
                f"{r['custom_vs_cpu']:.1f}x vs CPU  {r['fp16_vs_fp32']:.2f}x"
            )
        print("-" * 90)

    return results


def run_fp16_benchmark(verbose: bool = True) -> dict:
    """
    Run FP16 vs FP32 benchmark suite.

    Args:
        verbose: If True, print results to stdout.

    Returns:
        Dictionary with benchmark results.
    """
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
        GaussianSplatModelCUDA,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run benchmark.")

    # Larger configs to show memory bandwidth benefits
    configs = [
        (100, (64, 64, 64), "Small 3D"),
        (500, (128, 128, 128), "Medium 3D"),
        (1000, (192, 192, 192), "Large 3D"),
        (2000, (256, 256, 256), "XLarge 3D"),
        (500, (512, 512), "Medium 2D"),
        (1000, (1024, 1024), "Large 2D"),
    ]

    results = {}

    if verbose:
        print("=" * 80)
        print("FP16 vs FP32 BENCHMARK: Custom CUDA Kernels")
        print("=" * 80)
        print("\nNote: FP16 converts inputs to half precision at API boundary.")
        print("      Output and gradients remain FP32 for numerical stability.\n")

    for cfg in configs:
        N, shape, label = cfg
        centers, L, amps = create_test_data(N, shape)
        sigma_min = (0.5,) * len(shape)

        # FP32 CUDA custom
        model_fp32 = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=False,
        )
        fp32_time = benchmark(model_fp32, sync_cuda=True)

        # FP16 CUDA custom
        model_fp16 = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=True,
        )
        fp16_time = benchmark(model_fp16, sync_cuda=True)

        # Calculate speedup
        fp16_speedup = fp32_time / fp16_time

        voxels = int(np.prod(shape))

        # Memory estimation (input tensors only)
        d = len(shape)
        conic_size = d * (d + 1) // 2
        fp32_input_bytes = N * (d + conic_size + 1 + 1) * 4  # centers, conic, amps, sharpness
        fp16_input_bytes = N * (d + conic_size + 1 + 1) * 2
        memory_reduction = (1 - fp16_input_bytes / fp32_input_bytes) * 100

        results[label] = {
            "N": N,
            "shape": shape,
            "voxels": voxels,
            "fp32_ms": fp32_time,
            "fp16_ms": fp16_time,
            "fp16_speedup": fp16_speedup,
            "fp32_input_kb": fp32_input_bytes / 1024,
            "fp16_input_kb": fp16_input_bytes / 1024,
            "memory_reduction_pct": memory_reduction,
        }

        if verbose:
            print(f"{label} (N={N}, shape={shape}, {voxels:,} voxels):")
            print(f"  FP32: {fp32_time:8.3f} ms")
            print(f"  FP16: {fp16_time:8.3f} ms  ({fp16_speedup:5.2f}x {'faster' if fp16_speedup > 1 else 'slower'})")
            print(f"  Input memory: {fp32_input_bytes/1024:.1f} KB -> {fp16_input_bytes/1024:.1f} KB ({memory_reduction:.0f}% reduction)")
            print()

    if verbose:
        print("=" * 80)
        print("\nSummary:")
        print("-" * 80)
        print(
            f"{'Config':<12} {'N':<6} {'Voxels':<12} {'FP32 (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10}"
        )
        print("-" * 80)
        for label, r in results.items():
            speedup_str = f"{r['fp16_speedup']:.2f}x"
            if r['fp16_speedup'] < 1:
                speedup_str = f"{r['fp16_speedup']:.2f}x (slower)"
            print(
                f"{label:<12} {r['N']:<6} {r['voxels']:<12,} {r['fp32_ms']:<12.3f} "
                f"{r['fp16_ms']:<12.3f} {speedup_str:<10}"
            )
        print("-" * 80)

        # Analysis
        print("\nAnalysis:")
        faster_configs = [l for l, r in results.items() if r['fp16_speedup'] > 1.05]
        slower_configs = [l for l, r in results.items() if r['fp16_speedup'] < 0.95]
        if faster_configs:
            print(f"  FP16 faster: {', '.join(faster_configs)}")
        if slower_configs:
            print(f"  FP16 slower: {', '.join(slower_configs)}")
        if not faster_configs and not slower_configs:
            print("  FP16 and FP32 have similar performance across all configs.")
        print("\n  Note: FP16 benefits are most visible when memory bandwidth is the bottleneck.")
        print("        The current Phase 1 implementation converts at API boundary, so")
        print("        benefits depend on the ratio of conversion overhead vs kernel time.")

    return results


def main():
    """Main entry point."""
    import sys

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system.")
        print("This benchmark requires an NVIDIA GPU with CUDA support.")
        sys.exit(1)

    try:
        import cuda_splatting_backend
    except ImportError:
        print("ERROR: CUDA splatting backend is not compiled.")
        print("Please build it first:")
        print("  make build-cuda")
        print("Or:")
        print(
            "  hatch run python packages/luxar/src/luxar/gsplats/models/gsplats/cuda/build.py"
        )
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print()

    # Run main benchmark (CPU vs CUDA vanilla vs CUDA custom FP32 vs CUDA custom FP16)
    run_benchmark(verbose=True)

    # Run extended FP16 benchmark with larger volumes
    print("\n\n")
    run_fp16_benchmark(verbose=True)


if __name__ == "__main__":
    main()
