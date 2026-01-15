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
            with torch.amp.autocast("cuda"):
                _ = model()
        else:
            _ = model()
        if sync_cuda:
            torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model()
        else:
            _ = model()
        if sync_cuda:
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return elapsed


def benchmark_training(
    model: torch.nn.Module,
    n_warmup: int = 5,
    n_iters: int = 20,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    """
    Benchmark a model's forward and backward pass separately.

    Args:
        model: Model to benchmark (must be in training mode).
        n_warmup: Number of warmup iterations.
        n_iters: Number of timed iterations.
        use_amp: Whether to use torch.autocast() for mixed precision.

    Returns:
        Tuple of (forward_ms, backward_ms, total_ms).
    """
    model.train()

    # Warmup with full forward+backward
    for _ in range(n_warmup):
        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model()
                loss = output.sum()
            loss.backward()
        else:
            output = model()
            loss = output.sum()
            loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()

    # Benchmark forward only
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model()
        else:
            output = model()
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) / n_iters * 1000

    # Benchmark forward + backward
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_amp:
            with torch.amp.autocast("cuda"):
                output = model()
                loss = output.sum()
            loss.backward()
        else:
            output = model()
            loss = output.sum()
            loss.backward()
        model.zero_grad()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) / n_iters * 1000

    # Note: backward time is derived by subtraction, which can be noisy
    # for very fast operations. Clamp to 0 to avoid confusing negative values.
    bwd_ms = max(0.0, total_ms - fwd_ms)
    return fwd_ms, bwd_ms, total_ms


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
    # Includes stress-test configurations to find GPU saturation point
    if configs is None:
        configs = [
            # ===== 3D Configurations =====
            # Small volumes - few splats (baseline)
            (100, (64, 64, 64), "3D 64³ 100"),
            (500, (64, 64, 64), "3D 64³ 500"),
            # Medium volumes - varied splats
            (500, (128, 128, 128), "3D 128³ 500"),
            (1000, (128, 128, 128), "3D 128³ 1K"),
            (2000, (128, 128, 128), "3D 128³ 2K"),
            # Large volumes - scaling test
            (5000, (256, 256, 256), "3D 256³ 5K"),
            (10000, (256, 256, 256), "3D 256³ 10K"),
            (20000, (256, 256, 256), "3D 256³ 20K"),
            # Very large volumes - GPU saturation test
            (10000, (384, 384, 384), "3D 384³ 10K"),
            (20000, (384, 384, 384), "3D 384³ 20K"),
            (50000, (384, 384, 384), "3D 384³ 50K"),
            # Extreme volumes - stress test
            (20000, (512, 512, 512), "3D 512³ 20K"),
            (50000, (512, 512, 512), "3D 512³ 50K"),
            (100000, (512, 512, 512), "3D 512³ 100K"),
            # Ultra stress test (may OOM on smaller GPUs)
            (50000, (768, 768, 768), "3D 768³ 50K"),
            # ===== 2D Configurations =====
            # Small 2D (baseline)
            (500, (512, 512), "2D 512² 500"),
            (1000, (512, 512), "2D 512² 1K"),
            # Medium 2D
            (5000, (1024, 1024), "2D 1024² 5K"),
            (10000, (1024, 1024), "2D 1024² 10K"),
            # Large 2D - scaling test
            (10000, (2048, 2048), "2D 2048² 10K"),
            (20000, (2048, 2048), "2D 2048² 20K"),
            (50000, (2048, 2048), "2D 2048² 50K"),
            # Very large 2D - GPU saturation test
            (20000, (4096, 4096), "2D 4096² 20K"),
            (50000, (4096, 4096), "2D 4096² 50K"),
            (100000, (4096, 4096), "2D 4096² 100K"),
            # Ultra stress test (may OOM on smaller GPUs)
            (100000, (8192, 8192), "2D 8192² 100K"),
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
            print(f"[{i + 1}/{len(configs)}] {label} (N={N:,}, shape={shape})")

        result = {
            "N": N,
            "shape": shape,
            "dim": dim,
            "voxels": voxels,
        }

        # CPU baseline (skip for large configs - too slow)
        skip_cpu = not include_cpu or voxels > 2_000_000 or N > 2000
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

        # PyTorch CUDA vanilla (skip for large configs - too slow)
        skip_vanilla = not include_vanilla or voxels > 8_000_000 or N > 5000
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

        # Custom CUDA FP32 - Forward only (for inference comparison)
        try:
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

            # Custom CUDA FP32 - Training (forward + backward)
            fp32_fwd, fp32_bwd, fp32_total = benchmark_training(
                cuda_fp32_model, use_amp=False
            )
            result["fp32_fwd_ms"] = fp32_fwd
            result["fp32_bwd_ms"] = fp32_bwd
            result["fp32_train_ms"] = fp32_total
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if verbose:
                print(f"    FP32: OOM ({e})")
            result["fp32_ms"] = None
            result["fp32_fwd_ms"] = None
            result["fp32_bwd_ms"] = None
            result["fp32_train_ms"] = None
            torch.cuda.empty_cache()
            results[label] = result
            continue

        # Custom CUDA with AMP (recommended for training) - Forward only
        try:
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

            # Custom CUDA with AMP - Training (forward + backward)
            amp_fwd, amp_bwd, amp_total = benchmark_training(
                cuda_amp_model, use_amp=True
            )
            result["amp_fwd_ms"] = amp_fwd
            result["amp_bwd_ms"] = amp_bwd
            result["amp_train_ms"] = amp_total
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if verbose:
                print(f"    AMP: OOM ({e})")
            result["amp_ms"] = None
            result["amp_fwd_ms"] = None
            result["amp_bwd_ms"] = None
            result["amp_train_ms"] = None
            torch.cuda.empty_cache()

        # Custom CUDA FP16 (inference only - no backward timing)
        try:
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
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if verbose:
                print(f"    FP16: OOM ({e})")
            result["fp16_ms"] = None
            torch.cuda.empty_cache()

        # Calculate speedups and throughput only if we have valid timings
        fp32_time = result.get("fp32_ms")
        amp_time = result.get("amp_ms")
        fp16_time = result.get("fp16_ms")
        fp32_total = result.get("fp32_train_ms")
        amp_total = result.get("amp_train_ms")

        if result["cpu_ms"] is not None and fp32_time is not None:
            result["fp32_vs_cpu"] = result["cpu_ms"] / fp32_time
        else:
            result["fp32_vs_cpu"] = None

        if result["vanilla_ms"] is not None and fp32_time is not None:
            result["fp32_vs_vanilla"] = result["vanilla_ms"] / fp32_time
        else:
            result["fp32_vs_vanilla"] = None

        if fp32_time is not None and amp_time is not None:
            result["amp_vs_fp32"] = fp32_time / amp_time
        else:
            result["amp_vs_fp32"] = None

        if fp32_time is not None and fp16_time is not None:
            result["fp16_vs_fp32"] = fp32_time / fp16_time
        else:
            result["fp16_vs_fp32"] = None

        if fp32_total is not None and amp_total is not None:
            result["amp_train_vs_fp32_train"] = fp32_total / amp_total
        else:
            result["amp_train_vs_fp32_train"] = None

        # Throughput metrics (helps identify GPU saturation)
        # GVoxel/s = Giga-voxels per second for inference
        if fp32_time is not None:
            result["gvoxel_per_s_fp32"] = (voxels / fp32_time) / 1e6  # GV/s
            result["splats_per_ms_fp32"] = N / fp32_time
        else:
            result["gvoxel_per_s_fp32"] = None
            result["splats_per_ms_fp32"] = None

        if amp_time is not None:
            result["gvoxel_per_s_amp"] = (voxels / amp_time) / 1e6
            result["splats_per_ms_amp"] = N / amp_time
        else:
            result["gvoxel_per_s_amp"] = None
            result["splats_per_ms_amp"] = None

        if fp16_time is not None:
            result["gvoxel_per_s_fp16"] = (voxels / fp16_time) / 1e6
            result["splats_per_ms_fp16"] = N / fp16_time
        else:
            result["gvoxel_per_s_fp16"] = None
            result["splats_per_ms_fp16"] = None

        results[label] = result

        if verbose:
            # Inference line
            parts = []
            if result["cpu_ms"] is not None:
                parts.append(f"CPU={result['cpu_ms']:.1f}ms")
            if result["vanilla_ms"] is not None:
                parts.append(f"Vanilla={result['vanilla_ms']:.1f}ms")
            if fp32_time is not None:
                parts.append(f"FP32={fp32_time:.2f}ms")
            if amp_time is not None and result["amp_vs_fp32"] is not None:
                parts.append(f"AMP={amp_time:.2f}ms ({result['amp_vs_fp32']:.2f}x)")
            if fp16_time is not None and result["fp16_vs_fp32"] is not None:
                parts.append(f"FP16={fp16_time:.2f}ms ({result['fp16_vs_fp32']:.2f}x)")
            print(f"    Inference: {', '.join(parts)}")
            # Throughput line (helps identify GPU saturation)
            if result.get("gvoxel_per_s_fp32") is not None:
                print(
                    f"    Throughput: FP32={result['gvoxel_per_s_fp32']:.1f} GV/s"
                    + (
                        f", AMP={result['gvoxel_per_s_amp']:.1f} GV/s"
                        if result.get("gvoxel_per_s_amp")
                        else ""
                    )
                    + (
                        f", FP16={result['gvoxel_per_s_fp16']:.1f} GV/s"
                        if result.get("gvoxel_per_s_fp16")
                        else ""
                    )
                )
            # Training line (forward + backward)
            fp32_fwd = result.get("fp32_fwd_ms")
            fp32_bwd = result.get("fp32_bwd_ms")
            amp_fwd = result.get("amp_fwd_ms")
            amp_bwd = result.get("amp_bwd_ms")
            if fp32_fwd is not None and fp32_total is not None:
                train_parts = [
                    f"FP32=[fwd={fp32_fwd:.2f}+bwd={fp32_bwd:.2f}={fp32_total:.2f}ms]"
                ]
                if amp_fwd is not None and amp_total is not None:
                    train_parts.append(
                        f"AMP=[fwd={amp_fwd:.2f}+bwd={amp_bwd:.2f}={amp_total:.2f}ms]"
                        + (
                            f" ({result['amp_train_vs_fp32_train']:.2f}x)"
                            if result.get("amp_train_vs_fp32_train")
                            else ""
                        )
                    )
                print(f"    Training:  {', '.join(train_parts)}")

    # Print summary table
    if verbose:
        print("\n" + "=" * 120)
        print("SUMMARY TABLE - INFERENCE")
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
                cpu_str = f"{r['cpu_ms']:.1f}" if r.get("cpu_ms") else "-"
                vanilla_str = f"{r['vanilla_ms']:.1f}" if r.get("vanilla_ms") else "-"
                fp32_str = f"{r['fp32_ms']:.2f}" if r.get("fp32_ms") else "OOM"
                amp_str = f"{r['amp_ms']:.2f}" if r.get("amp_ms") else "OOM"
                fp16_str = f"{r['fp16_ms']:.2f}" if r.get("fp16_ms") else "OOM"
                fp32_cpu_str = (
                    f"{r['fp32_vs_cpu']:.1f}x" if r.get("fp32_vs_cpu") else "-"
                )
                amp_fp32_str = (
                    f"{r['amp_vs_fp32']:.2f}x" if r.get("amp_vs_fp32") else "-"
                )

                row = (
                    f"{label:<20} {r['N']:>8,} {r['voxels']:>12,} "
                    f"{cpu_str:>10} {vanilla_str:>10} {fp32_str:>10} "
                    f"{amp_str:>10} {fp16_str:>10} "
                    f"{fp32_cpu_str:>10} {amp_fp32_str:>10}"
                )
                print(row)

        # Training summary table
        print("\n" + "=" * 120)
        print("SUMMARY TABLE - TRAINING (Forward + Backward)")
        print("=" * 120)

        for dim_label, dim_val in [("3D", 3), ("2D", 2)]:
            dim_results = {k: v for k, v in results.items() if v["dim"] == dim_val}
            if not dim_results:
                continue

            print(f"\n{dim_label} Configurations:")
            print("-" * 120)
            header = (
                f"{'Config':<20} {'Splats':>8} "
                f"{'FP32 Fwd':>10} {'FP32 Bwd':>10} {'FP32 Tot':>10} "
                f"{'AMP Fwd':>10} {'AMP Bwd':>10} {'AMP Tot':>10} "
                f"{'AMP/FP32':>10}"
            )
            print(header)
            print("-" * 120)

            for label, r in dim_results.items():
                # Handle None values for OOM configs
                fp32_fwd = f"{r['fp32_fwd_ms']:.2f}" if r.get("fp32_fwd_ms") else "OOM"
                fp32_bwd = f"{r['fp32_bwd_ms']:.2f}" if r.get("fp32_bwd_ms") else "-"
                fp32_tot = (
                    f"{r['fp32_train_ms']:.2f}" if r.get("fp32_train_ms") else "-"
                )
                amp_fwd = f"{r['amp_fwd_ms']:.2f}" if r.get("amp_fwd_ms") else "OOM"
                amp_bwd = f"{r['amp_bwd_ms']:.2f}" if r.get("amp_bwd_ms") else "-"
                amp_tot = f"{r['amp_train_ms']:.2f}" if r.get("amp_train_ms") else "-"
                ratio = (
                    f"{r['amp_train_vs_fp32_train']:.2f}x"
                    if r.get("amp_train_vs_fp32_train")
                    else "-"
                )

                row = (
                    f"{label:<20} {r['N']:>8,} "
                    f"{fp32_fwd:>10} {fp32_bwd:>10} {fp32_tot:>10} "
                    f"{amp_fwd:>10} {amp_bwd:>10} {amp_tot:>10} "
                    f"{ratio:>10}"
                )
                print(row)

        # Throughput summary table - shows GPU utilization scaling
        print("\n" + "=" * 120)
        print("SUMMARY TABLE - THROUGHPUT (GigaVoxels/second)")
        print("=" * 120)
        print("Higher is better. Constant throughput across configs = GPU saturated.")
        print("Increasing throughput = GPU underutilized at smaller workloads.")

        for dim_label, dim_val in [("3D", 3), ("2D", 2)]:
            dim_results = {k: v for k, v in results.items() if v["dim"] == dim_val}
            if not dim_results:
                continue

            print(f"\n{dim_label} Configurations:")
            print("-" * 120)
            header = (
                f"{'Config':<20} {'Splats':>8} {'Voxels':>14} "
                f"{'FP32 GV/s':>12} {'AMP GV/s':>12} {'FP16 GV/s':>12} "
                f"{'FP32 ms':>10} {'AMP ms':>10} {'FP16 ms':>10}"
            )
            print(header)
            print("-" * 120)

            for label, r in dim_results.items():
                # Handle None values for OOM configs
                gv_fp32 = (
                    f"{r['gvoxel_per_s_fp32']:.1f}"
                    if r.get("gvoxel_per_s_fp32")
                    else "OOM"
                )
                gv_amp = (
                    f"{r['gvoxel_per_s_amp']:.1f}"
                    if r.get("gvoxel_per_s_amp")
                    else "OOM"
                )
                gv_fp16 = (
                    f"{r['gvoxel_per_s_fp16']:.1f}"
                    if r.get("gvoxel_per_s_fp16")
                    else "OOM"
                )
                fp32_ms = f"{r['fp32_ms']:.2f}" if r.get("fp32_ms") else "-"
                amp_ms = f"{r['amp_ms']:.2f}" if r.get("amp_ms") else "-"
                fp16_ms = f"{r['fp16_ms']:.2f}" if r.get("fp16_ms") else "-"

                row = (
                    f"{label:<20} {r['N']:>8,} {r['voxels']:>14,} "
                    f"{gv_fp32:>12} {gv_amp:>12} {gv_fp16:>12} "
                    f"{fp32_ms:>10} {amp_ms:>10} {fp16_ms:>10}"
                )
                print(row)

        print("-" * 120)
        print("\nNotes:")
        print("  - Times in milliseconds (ms). Lower is better.")
        print("  - GV/s = GigaVoxels per second. Higher is better.")
        print("  - FP32/CPU: Speedup of custom CUDA FP32 vs CPU baseline.")
        print("  - AMP/FP32: Speedup of AMP mode vs FP32 (>1.0 = faster).")
        print("  - '-' indicates skipped (too slow or OOM).")
        print("  - AMP is RECOMMENDED for training (FP32 params, FP16 compute).")
        print("  - FP16 is inference only (may overflow during training).")
        print(
            "  - Backward pass uses Per-Tile Gradient Accumulation (16x fewer atomics)."
        )
        print("  - If throughput plateaus at large sizes, GPU is saturated.")
        print("  - If throughput keeps increasing, there's room for more parallelism.")

    return results


def main():
    """Main entry point."""
    import sys

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system.")
        print("This benchmark requires an NVIDIA GPU with CUDA support.")
        sys.exit(1)

    try:
        import cuda_splatting_backend  # noqa: F401
    except ImportError:
        print("ERROR: CUDA splatting backend is not compiled.")
        print("Please build it first:")
        print("  make build-cuda")
        print("Or: hatch run python .../gsplats/models/gsplats/cuda/build.py")
        sys.exit(1)

    run_benchmark(verbose=True)


if __name__ == "__main__":
    main()
