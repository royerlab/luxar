#!/usr/bin/env python
"""
Performance benchmark for CUDA Gaussian splatting.

Compares CPU, PyTorch CUDA, and custom CUDA kernels (FP32, AMP, FP16) across
varied 2D and 3D configurations with different volume sizes and splat counts.

Usage:
    python -m luxar.gsplats.models.gsplats.cuda.benchmark

    # Or from the cuda directory:
    python benchmark.py

    # Run with splat sweep (throughput vs splat count at peak volume size):
    python benchmark.py --sweep

    # Sweep at a specific volume size:
    python benchmark.py --sweep --shape 512 512 512

    # Custom sweep range:
    python benchmark.py --sweep --splats-min 500 --splats-max 200000 --splats-steps 64

Results are printed to stdout in formatted tables.
"""

import datetime
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _gpu_mem_info() -> Tuple[float, float]:
    """Return (free_gb, total_gb) for the current CUDA device."""
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


def _oom_cleanup() -> None:
    """Aggressively free GPU memory after an OOM."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


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
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            if use_amp:
                with torch.amp.autocast("cuda"):  # type: ignore[attr-defined, unused-ignore]
                    _ = model()
            else:
                _ = model()
            if sync_cuda:
                torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(n_iters):
            if use_amp:
                with torch.amp.autocast("cuda"):  # type: ignore[attr-defined, unused-ignore]
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
            with torch.amp.autocast("cuda"):  # type: ignore[attr-defined, unused-ignore]
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
            with torch.amp.autocast("cuda"):  # type: ignore[attr-defined, unused-ignore]
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
            with torch.amp.autocast("cuda"):  # type: ignore[attr-defined, unused-ignore]
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
    configs: Optional[List[Tuple[int, Tuple[int, ...], str]]] = None,
    include_cpu: bool = True,
    include_vanilla: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
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
            (100, (64, 64, 64), "3D 64\u00b3 100"),
            (500, (64, 64, 64), "3D 64\u00b3 500"),
            # Medium volumes - varied splats
            (500, (128, 128, 128), "3D 128\u00b3 500"),
            (1000, (128, 128, 128), "3D 128\u00b3 1K"),
            (2000, (128, 128, 128), "3D 128\u00b3 2K"),
            # Large volumes - scaling test
            (5000, (256, 256, 256), "3D 256\u00b3 5K"),
            (10000, (256, 256, 256), "3D 256\u00b3 10K"),
            (20000, (256, 256, 256), "3D 256\u00b3 20K"),
            # Very large volumes - GPU saturation test
            (10000, (384, 384, 384), "3D 384\u00b3 10K"),
            (20000, (384, 384, 384), "3D 384\u00b3 20K"),
            (50000, (384, 384, 384), "3D 384\u00b3 50K"),
            # Extreme volumes - stress test
            (20000, (512, 512, 512), "3D 512\u00b3 20K"),
            (50000, (512, 512, 512), "3D 512\u00b3 50K"),
            (100000, (512, 512, 512), "3D 512\u00b3 100K"),
            # Ultra-large volumes (output tensor: 768^3*4 = 1.8 GB)
            (50000, (768, 768, 768), "3D 768\u00b3 50K"),
            (100000, (768, 768, 768), "3D 768\u00b3 100K"),
            # Extreme volumes (output tensor: 1024^3*4 = 4.3 GB, needs ~10+ GB GPU)
            (50000, (1024, 1024, 1024), "3D 1024\u00b3 50K"),
            (100000, (1024, 1024, 1024), "3D 1024\u00b3 100K"),
            # Maximum volumes (output tensor: 1280^3*4 = 8.4 GB, needs ~18+ GB GPU)
            (50000, (1280, 1280, 1280), "3D 1280\u00b3 50K"),
            # Near-limit (output tensor: 1536^3*4 = 14.5 GB, needs ~24+ GB GPU)
            (50000, (1536, 1536, 1536), "3D 1536\u00b3 50K"),
            # 48+ GB GPUs (A6000, A100): output tensor = 2048^3*4 = 34.4 GB
            (50000, (2048, 2048, 2048), "3D 2048\u00b3 50K"),
            # 80 GB GPUs (A100-80G, H100): output tensor = 2560^3*4 = 67 GB
            (50000, (2560, 2560, 2560), "3D 2560\u00b3 50K"),
            # ===== 2D Configurations =====
            # Small 2D (baseline)
            (500, (512, 512), "2D 512\u00b2 500"),
            (1000, (512, 512), "2D 512\u00b2 1K"),
            # Medium 2D
            (5000, (1024, 1024), "2D 1024\u00b2 5K"),
            (10000, (1024, 1024), "2D 1024\u00b2 10K"),
            # Large 2D - scaling test
            (10000, (2048, 2048), "2D 2048\u00b2 10K"),
            (20000, (2048, 2048), "2D 2048\u00b2 20K"),
            (50000, (2048, 2048), "2D 2048\u00b2 50K"),
            # Very large 2D - GPU saturation test
            (20000, (4096, 4096), "2D 4096\u00b2 20K"),
            (50000, (4096, 4096), "2D 4096\u00b2 50K"),
            (100000, (4096, 4096), "2D 4096\u00b2 100K"),
            # Ultra stress test (may OOM on smaller GPUs)
            (100000, (8192, 8192), "2D 8192\u00b2 100K"),
        ]

    results = {}

    if verbose:
        print("=" * 120)
        print("CUDA GAUSSIAN SPLATTING BENCHMARK")
        print("=" * 120)
        free_gb, total_gb = _gpu_mem_info()
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
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

        result: Dict[str, Any] = {
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
                    device=torch.device("cpu"),
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
                    device=torch.device("cuda"),
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
        fp32_oom = False
        try:
            cuda_fp32_model = GaussianSplatModelCUDA(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=sigma_min,
                device=torch.device("cuda"),
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
            del cuda_fp32_model
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            free_gb, total_gb = _gpu_mem_info()
            if verbose:
                print(
                    f"    FP32: OOM ({free_gb:.1f} GB free / {total_gb:.1f} GB total)"
                )
            result["fp32_ms"] = None
            result["fp32_fwd_ms"] = None
            result["fp32_bwd_ms"] = None
            result["fp32_train_ms"] = None
            fp32_oom = True
            _oom_cleanup()

        # Skip AMP/FP16 if FP32 already OOM'd (they need similar or more memory)
        if fp32_oom:
            result["amp_ms"] = None
            result["amp_fwd_ms"] = None
            result["amp_bwd_ms"] = None
            result["amp_train_ms"] = None
            result["fp16_ms"] = None
            if verbose:
                print("    AMP/FP16: skipped (FP32 OOM)")
        else:
            # Custom CUDA with AMP (recommended for training) - Forward only
            try:
                cuda_amp_model = GaussianSplatModelCUDA(
                    shape=shape,
                    centers0=centers,
                    L0=L,
                    amps0=amps,
                    sigma_min_diag=sigma_min,
                    device=torch.device("cuda"),
                    use_fp16=False,
                )
                amp_time = benchmark_forward(
                    cuda_amp_model, sync_cuda=True, use_amp=True
                )
                result["amp_ms"] = amp_time

                # Custom CUDA with AMP - Training (forward + backward)
                amp_fwd, amp_bwd, amp_total = benchmark_training(
                    cuda_amp_model, use_amp=True
                )
                result["amp_fwd_ms"] = amp_fwd
                result["amp_bwd_ms"] = amp_bwd
                result["amp_train_ms"] = amp_total
                del cuda_amp_model
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                free_gb, total_gb = _gpu_mem_info()
                if verbose:
                    print(
                        f"    AMP: OOM ({free_gb:.1f} GB free"
                        f" / {total_gb:.1f} GB total)"
                    )
                result["amp_ms"] = None
                result["amp_fwd_ms"] = None
                result["amp_bwd_ms"] = None
                result["amp_train_ms"] = None
                _oom_cleanup()

            # Custom CUDA FP16 (inference only - no backward timing)
            try:
                cuda_fp16_model = GaussianSplatModelCUDA(
                    shape=shape,
                    centers0=centers,
                    L0=L,
                    amps0=amps,
                    sigma_min_diag=sigma_min,
                    device=torch.device("cuda"),
                    use_fp16=True,
                )
                fp16_time = benchmark_forward(cuda_fp16_model, sync_cuda=True)
                result["fp16_ms"] = fp16_time
                del cuda_fp16_model
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                free_gb, total_gb = _gpu_mem_info()
                if verbose:
                    print(
                        f"    FP16: OOM ({free_gb:.1f} GB free"
                        f" / {total_gb:.1f} GB total)"
                    )
                result["fp16_ms"] = None
                _oom_cleanup()

        # Calculate speedups and throughput only if we have valid timings
        r_fp32: Any = result.get("fp32_ms")
        r_amp: Any = result.get("amp_ms")
        r_fp16: Any = result.get("fp16_ms")
        r_fp32_total: Any = result.get("fp32_train_ms")
        r_amp_total: Any = result.get("amp_train_ms")

        if result["cpu_ms"] is not None and r_fp32 is not None:
            result["fp32_vs_cpu"] = result["cpu_ms"] / r_fp32
        else:
            result["fp32_vs_cpu"] = None

        if result["vanilla_ms"] is not None and r_fp32 is not None:
            result["fp32_vs_vanilla"] = result["vanilla_ms"] / r_fp32
        else:
            result["fp32_vs_vanilla"] = None

        if r_fp32 is not None and r_amp is not None:
            result["amp_vs_fp32"] = r_fp32 / r_amp
        else:
            result["amp_vs_fp32"] = None

        if r_fp32 is not None and r_fp16 is not None:
            result["fp16_vs_fp32"] = r_fp32 / r_fp16
        else:
            result["fp16_vs_fp32"] = None

        if r_fp32_total is not None and r_amp_total is not None:
            result["amp_train_vs_fp32_train"] = r_fp32_total / r_amp_total
        else:
            result["amp_train_vs_fp32_train"] = None

        # Throughput metrics (helps identify GPU saturation)
        # GVoxel/s = Giga-voxels per second for inference
        if r_fp32 is not None:
            result["gvoxel_per_s_fp32"] = (voxels / r_fp32) / 1e6  # GV/s
            result["splats_per_ms_fp32"] = N / r_fp32
        else:
            result["gvoxel_per_s_fp32"] = None
            result["splats_per_ms_fp32"] = None

        if r_amp is not None:
            result["gvoxel_per_s_amp"] = (voxels / r_amp) / 1e6
            result["splats_per_ms_amp"] = N / r_amp
        else:
            result["gvoxel_per_s_amp"] = None
            result["splats_per_ms_amp"] = None

        if r_fp16 is not None:
            result["gvoxel_per_s_fp16"] = (voxels / r_fp16) / 1e6
            result["splats_per_ms_fp16"] = N / r_fp16
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
            if r_fp32 is not None:
                parts.append(f"FP32={r_fp32:.2f}ms")
            if r_amp is not None and result["amp_vs_fp32"] is not None:
                parts.append(f"AMP={r_amp:.2f}ms ({result['amp_vs_fp32']:.2f}x)")
            if r_fp16 is not None and result["fp16_vs_fp32"] is not None:
                parts.append(f"FP16={r_fp16:.2f}ms ({result['fp16_vs_fp32']:.2f}x)")
            if parts:
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
            r_fp32_fwd: Any = result.get("fp32_fwd_ms")
            r_fp32_bwd: Any = result.get("fp32_bwd_ms")
            r_amp_fwd: Any = result.get("amp_fwd_ms")
            r_amp_bwd: Any = result.get("amp_bwd_ms")
            if r_fp32_fwd is not None and r_fp32_total is not None:
                train_parts = [
                    f"FP32=[fwd={r_fp32_fwd:.2f}+bwd={r_fp32_bwd:.2f}={r_fp32_total:.2f}ms]"
                ]
                if r_amp_fwd is not None and r_amp_total is not None:
                    train_parts.append(
                        f"AMP=[fwd={r_amp_fwd:.2f}+bwd={r_amp_bwd:.2f}={r_amp_total:.2f}ms]"
                        + (
                            f" ({result['amp_train_vs_fp32_train']:.2f}x)"
                            if result.get("amp_train_vs_fp32_train")
                            else ""
                        )
                    )
                print(f"    Training:  {', '.join(train_parts)}")

    # Print summary tables
    if verbose:
        _print_summary_tables(results)

    return results


def _print_summary_tables(results: dict) -> None:
    """Print all summary tables (inference, training, throughput)."""
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
            fp32_cpu_str = f"{r['fp32_vs_cpu']:.1f}x" if r.get("fp32_vs_cpu") else "-"
            amp_fp32_str = f"{r['amp_vs_fp32']:.2f}x" if r.get("amp_vs_fp32") else "-"

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
            fp32_tot = f"{r['fp32_train_ms']:.2f}" if r.get("fp32_train_ms") else "-"
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
                f"{r['gvoxel_per_s_fp32']:.1f}" if r.get("gvoxel_per_s_fp32") else "OOM"
            )
            gv_amp = (
                f"{r['gvoxel_per_s_amp']:.1f}" if r.get("gvoxel_per_s_amp") else "OOM"
            )
            gv_fp16 = (
                f"{r['gvoxel_per_s_fp16']:.1f}" if r.get("gvoxel_per_s_fp16") else "OOM"
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
    print("  - Backward pass uses Per-Tile Gradient Accumulation (16x fewer atomics).")
    print("  - If throughput plateaus at large sizes, GPU is saturated.")
    print("  - If throughput keeps increasing, there's room for more parallelism.")


def run_splat_sweep(
    shape: Tuple[int, ...],
    splats_min: int = 1000,
    splats_max: int = 100000,
    n_steps: int = 64,
    verbose: bool = True,
) -> dict:
    """
    Sweep splat count at a fixed volume size to characterize throughput vs splats.

    Runs FP32 custom CUDA for each splat count, measuring inference throughput.
    Stops early if OOM is encountered (larger counts will also OOM).

    Args:
        shape: Volume shape to benchmark at (e.g., (512, 512, 512)).
        splats_min: Minimum splat count.
        splats_max: Maximum splat count.
        n_steps: Number of log-spaced splat counts to test.
        verbose: Print results to stdout.

    Returns:
        Dictionary mapping splat count to result dict.
    """
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
        GaussianSplatModelCUDA,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot run benchmark.")

    # Generate log-spaced splat counts, deduplicated
    splat_counts = np.unique(np.geomspace(splats_min, splats_max, n_steps).astype(int))

    voxels = int(np.prod(shape))
    dim = len(shape)
    shape_str = "\u00d7".join(str(s) for s in shape)

    if verbose:
        print()
        print("=" * 90)
        print(f"SPLAT SWEEP: {dim}D {shape_str} ({voxels:,} voxels)")
        print("=" * 90)
        free_gb, total_gb = _gpu_mem_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
        print(
            f"Splat range: {splats_min:,} to {splats_max:,} ({len(splat_counts)} steps)"
        )
        print()
        print("-" * 90)
        header = (
            f"{'Splats':>10} {'FP32 ms':>10} {'GV/s':>10} {'Splats/ms':>12}"
            f" {'AMP ms':>10} {'AMP GV/s':>10}"
        )
        print(header)
        print("-" * 90)

    results = {}
    peak_gvs = 0.0
    peak_splats = 0
    oom_hit = False

    for N in splat_counts:
        N = int(N)
        if oom_hit:
            if verbose:
                print(f"{N:>10,} {'(skipped - OOM at lower count)':>50}")
            continue

        try:
            centers, L, amps = create_test_data(N, shape)
            sigma_min = (0.5,) * dim

            # FP32 inference
            model = GaussianSplatModelCUDA(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=sigma_min,
                device=torch.device("cuda"),
                use_fp16=False,
            )
            fp32_ms = benchmark_forward(model, sync_cuda=True, n_iters=15)
            gvs = (voxels / fp32_ms) / 1e6
            splats_per_ms = N / fp32_ms

            # AMP inference (reuse same model)
            amp_ms = benchmark_forward(model, sync_cuda=True, n_iters=15, use_amp=True)
            amp_gvs = (voxels / amp_ms) / 1e6

            del model
            _oom_cleanup()

            result = {
                "N": N,
                "fp32_ms": fp32_ms,
                "gvoxel_per_s": gvs,
                "splats_per_ms": splats_per_ms,
                "amp_ms": amp_ms,
                "amp_gvoxel_per_s": amp_gvs,
            }
            results[N] = result

            if gvs > peak_gvs:
                peak_gvs = gvs
                peak_splats = N

            if verbose:
                print(
                    f"{N:>10,} {fp32_ms:>10.2f} {gvs:>10.1f} {splats_per_ms:>12.0f}"
                    f" {amp_ms:>10.2f} {amp_gvs:>10.1f}"
                )

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            free_gb, total_gb = _gpu_mem_info()
            if verbose:
                print(
                    f"{N:>10,}        OOM ({free_gb:.1f} GB free"
                    f" / {total_gb:.1f} GB total)"
                )
            oom_hit = True
            _oom_cleanup()

    if verbose:
        print("-" * 90)
        if results:
            # Find the point where throughput drops significantly
            sorted_results = sorted(results.values(), key=lambda r: r["N"])
            print(f"\nPeak throughput: {peak_gvs:.1f} GV/s at {peak_splats:,} splats")

            # Show throughput retention curve
            if len(sorted_results) >= 2:
                first_gvs = sorted_results[0]["gvoxel_per_s"]
                last_gvs = sorted_results[-1]["gvoxel_per_s"]
                if first_gvs > 0:
                    drop_pct = (1 - last_gvs / first_gvs) * 100
                    print(
                        f"Throughput at max splats: {last_gvs:.1f} GV/s"
                        f" ({drop_pct:+.0f}% vs min splats)"
                    )
        if oom_hit:
            print("Note: OOM encountered - increase GPU memory for larger sweeps.")

    return results


def _luxar_config_dir() -> Path:
    """Return ~/.luxar/, creating it if necessary."""
    config_dir = Path.home() / ".luxar"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def build_profile_data(
    benchmark_results: dict,
    sweep_results: Optional[dict] = None,
    sweep_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """Build a benchmark profile data dict from raw results.

    This is the pure-data counterpart of the old ``generate_profile()`` —
    it computes all throughput tables, OOM boundaries, and recommendations
    but does **not** write anything to disk.

    Args:
        benchmark_results: Results from :func:`run_benchmark`.
        sweep_results: Optional results from :func:`run_splat_sweep`.
        sweep_shape: Volume shape used for the sweep.

    Returns:
        Profile dict with keys ``gpu``, ``timestamp``, ``throughput``,
        ``oom_boundaries``, ``splat_sweep`` (optional), ``recommendations``.
    """
    profile: Dict[str, Any] = {}

    # -- GPU info --
    gpu_info: Dict[str, Any] = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_gb, total_gb = _gpu_mem_info()
        gpu_info = {
            "name": props.name,
            "total_memory_gb": round(total_gb, 1),
            "free_memory_gb": round(free_gb, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "cuda_version": str(torch.version.cuda),
            "pytorch_version": str(torch.__version__),
        }
    profile["gpu"] = gpu_info
    profile["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # -- Per-config throughput table --
    configs_3d: List[Dict[str, Any]] = []
    configs_2d: List[Dict[str, Any]] = []
    peak_gvs_3d = 0.0
    peak_shape_3d: Optional[List[int]] = None
    peak_splats_3d = 0

    for label, r in benchmark_results.items():
        entry: Dict[str, Any] = {
            "label": label,
            "shape": list(r["shape"]),
            "voxels": r["voxels"],
            "splats": r["N"],
            "oom": r.get("fp32_ms") is None,
        }
        if not entry["oom"]:
            entry["fp32_ms"] = round(r["fp32_ms"], 2)
            entry["fp32_gvoxel_per_s"] = round(r["gvoxel_per_s_fp32"], 1)
            if r.get("amp_ms") is not None:
                entry["amp_ms"] = round(r["amp_ms"], 2)
                entry["amp_gvoxel_per_s"] = round(r["gvoxel_per_s_amp"], 1)
            if r.get("fp32_train_ms") is not None:
                entry["fp32_train_ms"] = round(r["fp32_train_ms"], 2)
            if r.get("amp_train_ms") is not None:
                entry["amp_train_ms"] = round(r["amp_train_ms"], 2)

        if r["dim"] == 3:
            configs_3d.append(entry)
            if not entry["oom"]:
                gvs = r.get("gvoxel_per_s_fp32", 0) or 0
                if gvs > peak_gvs_3d:
                    peak_gvs_3d = gvs
                    peak_shape_3d = list(r["shape"])
                    peak_splats_3d = r["N"]
        elif r["dim"] == 2:
            configs_2d.append(entry)

    profile["throughput"] = {}
    if configs_3d:
        profile["throughput"]["3d"] = configs_3d
    if configs_2d:
        profile["throughput"]["2d"] = configs_2d

    # -- OOM boundaries --
    # Find the largest successful and smallest OOM volume for each dim
    oom_info: Dict[str, Any] = {}
    for dim_key, _dim_val, configs in [("3d", 3, configs_3d), ("2d", 2, configs_2d)]:
        successful = [c for c in configs if not c["oom"]]
        failed = [c for c in configs if c["oom"]]
        if successful or failed:
            entry = {}
            if successful:
                largest_ok = max(successful, key=lambda c: c["voxels"])
                entry["max_successful_shape"] = largest_ok["shape"]
                entry["max_successful_voxels"] = largest_ok["voxels"]
            if failed:
                smallest_oom = min(failed, key=lambda c: c["voxels"])
                entry["min_oom_shape"] = smallest_oom["shape"]
                entry["min_oom_voxels"] = smallest_oom["voxels"]
            oom_info[dim_key] = entry
    profile["oom_boundaries"] = oom_info

    # -- Splat sweep curve --
    if sweep_results and sweep_shape:
        sorted_sweep = sorted(sweep_results.values(), key=lambda r: r["N"])
        sweep_data: Dict[str, Any] = {
            "shape": list(sweep_shape),
            "voxels": int(np.prod(sweep_shape)),
            "points": [],
        }
        sweep_peak_gvs = 0.0
        sweep_peak_n = 0
        for sr in sorted_sweep:
            point: Dict[str, Any] = {
                "splats": sr["N"],
                "fp32_ms": round(sr["fp32_ms"], 2),
                "fp32_gvoxel_per_s": round(sr["gvoxel_per_s"], 1),
            }
            if sr.get("amp_ms") is not None:
                point["amp_ms"] = round(sr["amp_ms"], 2)
                point["amp_gvoxel_per_s"] = round(sr["amp_gvoxel_per_s"], 1)
            sweep_data["points"].append(point)
            if sr["gvoxel_per_s"] > sweep_peak_gvs:
                sweep_peak_gvs = sr["gvoxel_per_s"]
                sweep_peak_n = sr["N"]

        sweep_data["peak_gvoxel_per_s"] = round(sweep_peak_gvs, 1)
        sweep_data["peak_splats"] = sweep_peak_n
        if len(sorted_sweep) >= 2:
            sweep_data["throughput_at_min_splats"] = round(
                sorted_sweep[0]["gvoxel_per_s"], 1
            )
            sweep_data["throughput_at_max_splats"] = round(
                sorted_sweep[-1]["gvoxel_per_s"], 1
            )
        profile["splat_sweep"] = sweep_data

    # -- Recommended operating points --
    recommendations: Dict[str, Any] = {}

    if peak_shape_3d:
        recommendations["peak_throughput_3d"] = {
            "shape": peak_shape_3d,
            "gvoxel_per_s": round(peak_gvs_3d, 1),
            "splats": peak_splats_3d,
        }

    if sweep_results and sweep_shape:
        sorted_sweep = sorted(sweep_results.values(), key=lambda r: r["N"])
        if sorted_sweep:
            local_peak = max(sr["gvoxel_per_s"] for sr in sorted_sweep)
            threshold_80 = local_peak * 0.80
            max_splats_80pct = sorted_sweep[0]["N"]
            for sr in sorted_sweep:
                if sr["gvoxel_per_s"] >= threshold_80:
                    max_splats_80pct = sr["N"]
            recommendations["max_splats_80pct_throughput"] = {
                "shape": list(sweep_shape),
                "max_splats": max_splats_80pct,
                "threshold_gvoxel_per_s": round(threshold_80, 1),
                "note": "Max splats keeping >=80% of peak throughput at this volume size",
            }

    if torch.cuda.is_available():
        free_gb, total_gb = _gpu_mem_info()
        safe_voxels_fp32 = int((free_gb * 0.6) * 1e9 / 4)
        recommendations["memory_safe_max_voxels_fp32"] = safe_voxels_fp32
        safe_side = int(safe_voxels_fp32 ** (1.0 / 3.0))
        recommendations["memory_safe_max_cube_side_3d"] = safe_side

    profile["recommendations"] = recommendations

    return profile


def generate_profile(
    benchmark_results: dict,
    sweep_results: Optional[dict] = None,
    sweep_shape: Optional[Tuple[int, ...]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Build profile data and save to the multi-GPU profile registry.

    This is the top-level function that both builds the profile data and
    persists it via :mod:`luxar.gsplats.gpu_profile`.

    Returns:
        Path to the written YAML file.
    """
    from luxar.gsplats.gpu_profile import PROFILE_PATH, append_run

    profile = build_profile_data(benchmark_results, sweep_results, sweep_shape)

    gpu_info_raw = profile.get("gpu", {})
    gpu_name = gpu_info_raw.get("name", "unknown")

    # Build static gpu_info (hardware-only fields)
    gpu_info = {
        k: v
        for k, v in gpu_info_raw.items()
        if k in ("total_memory_gb", "compute_capability", "sm_count")
    }

    # Build run_data (per-run fields)
    run_data = {
        "timestamp": profile["timestamp"],
        "cuda_version": gpu_info_raw.get("cuda_version", "unknown"),
        "pytorch_version": gpu_info_raw.get("pytorch_version", "unknown"),
        "free_memory_gb": gpu_info_raw.get("free_memory_gb", 0),
        "throughput": profile.get("throughput", {}),
        "oom_boundaries": profile.get("oom_boundaries", {}),
        "recommendations": profile.get("recommendations", {}),
    }
    if "splat_sweep" in profile:
        run_data["splat_sweep"] = profile["splat_sweep"]

    target_path = output_path if output_path is not None else PROFILE_PATH
    append_run(gpu_name, run_data, gpu_info, path=target_path)

    return target_path


def main() -> None:
    """Main entry point."""
    import argparse
    import sys

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available on this system.")
        print("This benchmark requires an NVIDIA GPU with CUDA support.")
        sys.exit(1)

    try:
        import cuda_splatting_backend  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        print("ERROR: CUDA splatting backend is not compiled.")
        print("Please build it first:")
        print("  make build-cuda")
        print("Or: hatch run python .../gsplats/models/gsplats/cuda/build.py")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="CUDA Gaussian splatting performance benchmark"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run splat count sweep after main benchmark",
    )
    parser.add_argument(
        "--sweep-only",
        action="store_true",
        help="Run only the splat sweep (skip main benchmark)",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=None,
        help="Volume shape for sweep (e.g., --shape 512 512 512). "
        "Default: auto-select peak throughput shape from main benchmark.",
    )
    parser.add_argument(
        "--splats-min",
        type=int,
        default=1000,
        help="Minimum splat count for sweep (default: 1000)",
    )
    parser.add_argument(
        "--splats-max",
        type=int,
        default=100000,
        help="Maximum splat count for sweep (default: 100000)",
    )
    parser.add_argument(
        "--splats-steps",
        type=int,
        default=64,
        help="Number of log-spaced steps in sweep (default: 64)",
    )
    args = parser.parse_args()

    # Run main benchmark
    results = {}
    if not args.sweep_only:
        results = run_benchmark(verbose=True)

    # Run splat sweep
    sweep_results = None
    sweep_shape: Optional[Tuple[int, ...]] = None
    if args.sweep or args.sweep_only:
        if args.shape:
            sweep_shape = tuple(args.shape)
        elif results:
            # Auto-select: find the 3D config with peak FP32 throughput
            best_gvs = 0.0
            for _label, r in results.items():
                if r["dim"] == 3 and r.get("gvoxel_per_s_fp32") is not None:
                    if r["gvoxel_per_s_fp32"] > best_gvs:
                        best_gvs = r["gvoxel_per_s_fp32"]
                        sweep_shape = r["shape"]
            if sweep_shape is not None:
                shape_str = "\u00d7".join(str(s) for s in sweep_shape)
                print(
                    f"\nAuto-selected sweep shape: {shape_str}"
                    f" (peak {best_gvs:.1f} GV/s)"
                )
        if sweep_shape is None:
            sweep_shape = (512, 512, 512)
            print(f"\nUsing default sweep shape: {sweep_shape}")

        sweep_results = run_splat_sweep(
            shape=sweep_shape,
            splats_min=args.splats_min,
            splats_max=args.splats_max,
            n_steps=args.splats_steps,
            verbose=True,
        )

    # Generate YAML profile (multi-GPU registry at ~/.luxar/gpu_profiles.yaml)
    if results or sweep_results:
        try:
            profile_path = generate_profile(
                benchmark_results=results or {},
                sweep_results=sweep_results,
                sweep_shape=sweep_shape,
            )
            print(f"\nProfile saved to: {profile_path}")
        except ImportError:
            print(
                "\nNote: Install PyYAML (pip install pyyaml) to generate"
                " the GPU benchmark profile."
            )


if __name__ == "__main__":
    main()
