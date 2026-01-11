#!/usr/bin/env python
"""
Performance benchmark comparing PyTorch CPU, PyTorch CUDA vanilla, and custom CUDA kernels.

This script measures forward pass performance across different configurations
to demonstrate the speedup achieved by the custom CUDA implementation.

Usage:
    python -m luxar.gsplats.models.gsplats.cuda.benchmark

    # Or from the cuda directory:
    python benchmark.py

Results are printed to stdout in a formatted table.
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
        print("=" * 75)
        print(
            "PERFORMANCE BENCHMARK: PyTorch-CPU vs PyTorch-CUDA-vanilla vs CUDA-custom"
        )
        print("=" * 75)

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

        # Custom CUDA kernels
        cuda_custom_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
        )
        cuda_custom_time = benchmark(cuda_custom_model, sync_cuda=True)

        # Calculate speedups
        vanilla_vs_cpu = cpu_time / cuda_vanilla_time
        custom_vs_cpu = cpu_time / cuda_custom_time
        custom_vs_vanilla = cuda_vanilla_time / cuda_custom_time

        voxels = int(np.prod(shape))

        results[label] = {
            "N": N,
            "shape": shape,
            "voxels": voxels,
            "cpu_ms": cpu_time,
            "cuda_vanilla_ms": cuda_vanilla_time,
            "cuda_custom_ms": cuda_custom_time,
            "vanilla_vs_cpu": vanilla_vs_cpu,
            "custom_vs_cpu": custom_vs_cpu,
            "custom_vs_vanilla": custom_vs_vanilla,
        }

        if verbose:
            print(f"\n{label} (N={N}, shape={shape}, {voxels:,} voxels):")
            print(f"  PyTorch CPU:          {cpu_time:8.2f} ms")
            print(
                f"  PyTorch CUDA vanilla: {cuda_vanilla_time:8.2f} ms  "
                f"({vanilla_vs_cpu:5.1f}x vs CPU)"
            )
            print(
                f"  Custom CUDA kernels:  {cuda_custom_time:8.2f} ms  "
                f"({custom_vs_cpu:5.1f}x vs CPU, {custom_vs_vanilla:.1f}x vs vanilla)"
            )

    if verbose:
        print("\n" + "=" * 75)
        print("\nSummary:")
        print("-" * 75)
        print(
            f"{'Config':<15} {'CPU (ms)':<12} {'Vanilla (ms)':<14} {'Custom (ms)':<12} {'Speedup':<10}"
        )
        print("-" * 75)
        for label, r in results.items():
            print(
                f"{label:<15} {r['cpu_ms']:<12.2f} {r['cuda_vanilla_ms']:<14.2f} "
                f"{r['cuda_custom_ms']:<12.2f} {r['custom_vs_cpu']:.1f}x"
            )
        print("-" * 75)

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

    run_benchmark(verbose=True)


if __name__ == "__main__":
    main()
