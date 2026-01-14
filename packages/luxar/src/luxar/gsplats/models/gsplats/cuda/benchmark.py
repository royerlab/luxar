#!/usr/bin/env python
"""
Performance benchmark for CUDA Gaussian splatting.

Compares CPU, PyTorch CUDA, and custom CUDA kernels (FP32, AMP, FP16) across
varied 2D and 3D configurations with different volume sizes and splat counts.

Usage:
    python -m luxar.gsplats.models.gsplats.cuda.benchmark

    # Or from the cuda directory:
    python benchmark.py

Results are printed to stdout in formatted tables.
"""

import time
from typing import List, Tuple

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


def benchmark_forward(
    model: torch.nn.Module,
    n_warmup: int = 5,
    n_iters: int = 20,
    sync_cuda: bool = False,
    use_amp: bool = False,
) -> float:
    """
    Benchmark a model's forward pass.

    Args:
        model: Model to benchmark.
        n_warmup: Number of warmup iterations.
        n_iters: Number of timed iterations.
        sync_cuda: Whether to synchronize CUDA after each iteration.
        use_amp: Whether to use torch.autocast() for mixed precision.

    Returns:
        Average time per forward pass in milliseconds.
    """
    # Warmup
    for _ in range(n_warmup):
        if use_amp:
            with torch.amp.autocast('cuda'):
                _ = model()
        else:
            _ = model()
        if sync_cuda:
            torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_amp:
            with torch.amp.autocast('cuda'):
                _ = model()
        else:
            _ = model()
        if sync_cuda:
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return elapsed


def run_benchmark(
    configs: List[Tuple[int, Tuple[int, ...], str]] = None,
    include_cpu: bool = True,
    include_vanilla: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Run comprehensive benchmark suite.

    Args:
        configs: List of (n_splats, shape, label) tuples. If None, uses defaults.
        include_cpu: Include CPU baseline (slow for large configs).
        include_vanilla: Include PyTorch CUDA vanilla baseline.
        verbose: Print results to stdout.

    Returns:
        Dictionary with benchmark results.
    """
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
        GaussianSplatModelCUDA,
    )
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run benchmark.")

    # Default comprehensive configuration
    if configs is None:
        configs = [
            # ===== 3D Configurations =====
            # Small volumes - few splats
            (50, (32, 32, 32), "3D 32³ 50"),
            (100, (64, 64, 64), "3D 64³ 100"),
            (500, (64, 64, 64), "3D 64³ 500"),
            # Medium volumes - varied splats
            (100, (128, 128, 128), "3D 128³ 100"),
            (500, (128, 128, 128), "3D 128³ 500"),
            (1000, (128, 128, 128), "3D 128³ 1K"),
            (2000, (128, 128, 128), "3D 128³ 2K"),
            # Large volumes - high splat counts
            (1000, (256, 256, 256), "3D 256³ 1K"),
            (5000, (256, 256, 256), "3D 256³ 5K"),
            (10000, (256, 256, 256), "3D 256³ 10K"),
            # Very large volumes
            (5000, (384, 384, 384), "3D 384³ 5K"),
            (10000, (512, 256, 256), "3D 512×256² 10K"),
            # Extreme splat counts
            (20000, (256, 256, 256), "3D 256³ 20K"),
            # ===== 2D Configurations =====
            # Small 2D
            (100, (256, 256), "2D 256² 100"),
            (500, (256, 256), "2D 256² 500"),
            # Medium 2D
            (500, (512, 512), "2D 512² 500"),
            (1000, (512, 512), "2D 512² 1K"),
            (2000, (512, 512), "2D 512² 2K"),
            # Large 2D
            (1000, (1024, 1024), "2D 1024² 1K"),
            (5000, (1024, 1024), "2D 1024² 5K"),
            (10000, (1024, 1024), "2D 1024² 10K"),
            # Very large 2D
            (5000, (2048, 2048), "2D 2048² 5K"),
            (10000, (2048, 2048), "2D 2048² 10K"),
            (20000, (2048, 2048), "2D 2048² 20K"),
        ]

    results = {}

    if verbose:
        print("=" * 120)
        print("CUDA GAUSSIAN SPLATTING BENCHMARK")
        print("=" * 120)
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print()

        # Print legend
        print("Backends:")
        print("  CPU        = PyTorch CPU (baseline)")
        print("  Vanilla    = PyTorch CUDA (standard ops)")
        print("  FP32       = Custom CUDA kernels (FP32)")
        print("  AMP        = Custom CUDA + torch.autocast() [BEST for training]")
        print("  FP16       = Custom CUDA with FP16 params [inference only]")
        print()

    for i, (N, shape, label) in enumerate(configs):
        centers, L, amps = create_test_data(N, shape)
        sigma_min = (0.5,) * len(shape)
        voxels = int(np.prod(shape))
        dim = len(shape)

        if verbose:
            print(f"[{i+1}/{len(configs)}] {label} (N={N:,}, shape={shape})")

        result = {
            "N": N,
            "shape": shape,
            "dim": dim,
            "voxels": voxels,
        }

        # CPU baseline (skip for very large configs)
        skip_cpu = not include_cpu or voxels > 4_000_000 or N > 5000
        if not skip_cpu:
            try:
                cpu_model = GaussianSplatModel(
                    shape=shape,
                    centers0=centers,
                    L0=L,
                    amps0=amps,
                    sigma_min_diag=sigma_min,
                    device="cpu",
                )
                cpu_time = benchmark_forward(cpu_model, sync_cuda=False, n_iters=5)
                result["cpu_ms"] = cpu_time
            except Exception as e:
                result["cpu_ms"] = None
                if verbose:
                    print(f"    CPU: skipped ({e})")
        else:
            result["cpu_ms"] = None

        # PyTorch CUDA vanilla (skip for very large configs)
        skip_vanilla = not include_vanilla or voxels > 16_000_000 or N > 10000
        if not skip_vanilla:
            try:
                cuda_vanilla_model = GaussianSplatModel(
                    shape=shape,
                    centers0=centers,
                    L0=L,
                    amps0=amps,
                    sigma_min_diag=sigma_min,
                    device="cuda",
                )
                vanilla_time = benchmark_forward(
                    cuda_vanilla_model, sync_cuda=True, n_iters=10
                )
                result["vanilla_ms"] = vanilla_time
            except Exception as e:
                result["vanilla_ms"] = None
                if verbose:
                    print(f"    Vanilla: skipped ({e})")
        else:
            result["vanilla_ms"] = None

        # Custom CUDA FP32
        cuda_fp32_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=False,
        )
        fp32_time = benchmark_forward(cuda_fp32_model, sync_cuda=True)
        result["fp32_ms"] = fp32_time

        # Custom CUDA with AMP (recommended for training)
        cuda_amp_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=False,
        )
        amp_time = benchmark_forward(cuda_amp_model, sync_cuda=True, use_amp=True)
        result["amp_ms"] = amp_time

        # Custom CUDA FP16 (inference only)
        cuda_fp16_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=True,
        )
        fp16_time = benchmark_forward(cuda_fp16_model, sync_cuda=True)
        result["fp16_ms"] = fp16_time

        # Calculate speedups
        if result["cpu_ms"] is not None:
            result["fp32_vs_cpu"] = result["cpu_ms"] / fp32_time
        else:
            result["fp32_vs_cpu"] = None

        if result["vanilla_ms"] is not None:
            result["fp32_vs_vanilla"] = result["vanilla_ms"] / fp32_time
        else:
            result["fp32_vs_vanilla"] = None

        result["amp_vs_fp32"] = fp32_time / amp_time
        result["fp16_vs_fp32"] = fp32_time / fp16_time

        results[label] = result

        if verbose:
            parts = []
            if result["cpu_ms"] is not None:
                parts.append(f"CPU={result['cpu_ms']:.1f}ms")
            if result["vanilla_ms"] is not None:
                parts.append(f"Vanilla={result['vanilla_ms']:.1f}ms")
            parts.append(f"FP32={fp32_time:.2f}ms")
            parts.append(f"AMP={amp_time:.2f}ms ({result['amp_vs_fp32']:.2f}x)")
            parts.append(f"FP16={fp16_time:.2f}ms ({result['fp16_vs_fp32']:.2f}x)")
            print(f"    {', '.join(parts)}")

    # Print summary table
    if verbose:
        print("\n" + "=" * 120)
        print("SUMMARY TABLE")
        print("=" * 120)

        # Group by dimension
        for dim_label, dim_val in [("3D", 3), ("2D", 2)]:
            dim_results = {k: v for k, v in results.items() if v["dim"] == dim_val}
            if not dim_results:
                continue

            print(f"\n{dim_label} Configurations:")
            print("-" * 120)
            header = (
                f"{'Config':<20} {'Splats':>8} {'Voxels':>12} "
                f"{'CPU':>10} {'Vanilla':>10} {'FP32':>10} {'AMP':>10} {'FP16':>10} "
                f"{'FP32/CPU':>10} {'AMP/FP32':>10}"
            )
            print(header)
            print("-" * 120)

            for label, r in dim_results.items():
                cpu_str = f"{r['cpu_ms']:.1f}" if r['cpu_ms'] else "-"
                vanilla_str = f"{r['vanilla_ms']:.1f}" if r['vanilla_ms'] else "-"
                fp32_cpu_str = f"{r['fp32_vs_cpu']:.1f}x" if r['fp32_vs_cpu'] else "-"

                row = (
                    f"{label:<20} {r['N']:>8,} {r['voxels']:>12,} "
                    f"{cpu_str:>10} {vanilla_str:>10} {r['fp32_ms']:>10.2f} "
                    f"{r['amp_ms']:>10.2f} {r['fp16_ms']:>10.2f} "
                    f"{fp32_cpu_str:>10} {r['amp_vs_fp32']:>9.2f}x"
                )
                print(row)

        print("-" * 120)
        print("\nNotes:")
        print("  - Times in milliseconds (ms). Lower is better.")
        print("  - FP32/CPU: Speedup of custom CUDA FP32 vs CPU baseline.")
        print("  - AMP/FP32: Speedup of AMP mode vs FP32 (>1.0 = faster).")
        print("  - '-' indicates skipped (too slow or OOM).")
        print("  - AMP is RECOMMENDED for training (FP32 params, FP16 compute).")
        print("  - FP16 is inference only (may overflow during training).")

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
        print("Or: hatch run python .../gsplats/models/gsplats/cuda/build.py")
        sys.exit(1)

    run_benchmark(verbose=True)


if __name__ == "__main__":
    main()
