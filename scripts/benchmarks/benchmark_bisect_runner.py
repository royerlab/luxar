#!/usr/bin/env python3
"""
Reduced benchmark runner for CUDA performance bisection.

Runs a subset of benchmark configurations to quickly measure
performance across commits. Outputs JSON for easy aggregation.

Usage:
    python benchmark_bisect_runner.py <worktree_path> <commit_sha> <output_json>
"""

import importlib
import json
import sys
import time
import traceback
from pathlib import Path


def setup_paths(worktree_path: str):
    """Set up Python path to use the worktree's source code."""
    worktree = Path(worktree_path)
    src_dir = worktree / "packages" / "luxar" / "src"
    cuda_dir = (
        worktree
        / "packages"
        / "luxar"
        / "src"
        / "luxar"
        / "gsplats"
        / "models"
        / "gsplats"
        / "cuda"
    )

    # Remove any existing luxar paths from sys.path
    sys.path = [p for p in sys.path if "luxar" not in p or "hatch" in p]

    # Insert worktree source and cuda dir at the front
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(cuda_dir))

    # Force reimport of luxar modules
    mods_to_remove = [k for k in sys.modules if k.startswith("luxar")]
    for mod in mods_to_remove:
        del sys.modules[mod]

    # Also remove cuda_splatting_backend if previously loaded
    if "cuda_splatting_backend" in sys.modules:
        del sys.modules["cuda_splatting_backend"]


def create_test_data(N, shape, seed=42):
    """Create random test data for benchmarking."""
    import numpy as np

    np.random.seed(seed)
    d = len(shape)
    centers = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 2) + 1
    L = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
    for i in range(N):
        L[i] *= np.random.uniform(0.5, 2.0)
    amps = np.random.rand(N).astype(np.float32) + 0.1
    return centers, L, amps


def benchmark_forward(model, n_warmup=5, n_iters=15, sync_cuda=True, use_amp=False):
    """Benchmark forward pass, return time in ms."""
    import torch

    for _ in range(n_warmup):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model()
        else:
            _ = model()
        if sync_cuda:
            torch.cuda.synchronize()

    torch.cuda.synchronize()
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


def benchmark_training(model, n_warmup=5, n_iters=15, use_amp=False):
    """Benchmark forward + backward, return (fwd_ms, bwd_ms, total_ms)."""
    import torch

    model.train()

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

    # Forward only
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model()
        else:
            _ = model()
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - start) / n_iters * 1000

    # Forward + backward
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

    bwd_ms = max(0.0, total_ms - fwd_ms)
    return fwd_ms, bwd_ms, total_ms


def create_model(model_class, shape, centers, L, amps, sigma_min, use_fp16=False):
    """Create model, handling different constructor signatures across commits."""
    try:
        return model_class(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
            use_fp16=use_fp16,
        )
    except TypeError:
        # Older commits don't have use_fp16
        return model_class(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=sigma_min,
            device="cuda",
        )


def run_benchmarks(worktree_path: str, commit_sha: str):
    """Run reduced benchmark suite and return results dict."""
    import numpy as np
    import torch

    setup_paths(worktree_path)

    # Import the CUDA model
    try:
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
    except ImportError as e:
        return {
            "error": f"Cannot import GaussianSplatModelCUDA: {e}",
            "commit": commit_sha,
        }

    # Reduced config set for fast bisection
    configs = [
        # Representative 3D configs
        (1000, (128, 128, 128), "3D_128_1K"),
        (10000, (256, 256, 256), "3D_256_10K"),
        (50000, (512, 512, 512), "3D_512_50K"),
        # Representative 2D configs
        (5000, (1024, 1024), "2D_1024_5K"),
        (50000, (4096, 4096), "2D_4096_50K"),
    ]

    results = {
        "commit": commit_sha,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "configs": {},
    }

    for N, shape, label in configs:
        print(f"  Benchmarking {label} (N={N:,}, shape={shape})...", flush=True)
        centers, L, amps = create_test_data(N, shape)
        sigma_min = (0.5,) * len(shape)
        voxels = int(np.prod(shape))

        config_result = {
            "N": N,
            "shape": list(shape),
            "voxels": voxels,
        }

        # FP32 inference
        try:
            torch.cuda.empty_cache()
            model = create_model(
                GaussianSplatModelCUDA,
                shape,
                centers,
                L,
                amps,
                sigma_min,
                use_fp16=False,
            )
            fp32_time = benchmark_forward(model, sync_cuda=True)
            config_result["fp32_inference_ms"] = round(fp32_time, 3)

            # FP32 training
            try:
                fwd, bwd, total = benchmark_training(model, use_amp=False)
                config_result["fp32_train_fwd_ms"] = round(fwd, 3)
                config_result["fp32_train_bwd_ms"] = round(bwd, 3)
                config_result["fp32_train_total_ms"] = round(total, 3)
            except Exception as e:
                config_result["fp32_train_error"] = str(e)

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            config_result["fp32_error"] = str(e)
            torch.cuda.empty_cache()

        # AMP inference
        try:
            torch.cuda.empty_cache()
            model = create_model(
                GaussianSplatModelCUDA,
                shape,
                centers,
                L,
                amps,
                sigma_min,
                use_fp16=False,
            )
            amp_time = benchmark_forward(model, sync_cuda=True, use_amp=True)
            config_result["amp_inference_ms"] = round(amp_time, 3)

            # AMP training
            try:
                fwd, bwd, total = benchmark_training(model, use_amp=True)
                config_result["amp_train_fwd_ms"] = round(fwd, 3)
                config_result["amp_train_bwd_ms"] = round(bwd, 3)
                config_result["amp_train_total_ms"] = round(total, 3)
            except Exception as e:
                config_result["amp_train_error"] = str(e)

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            config_result["amp_error"] = str(e)
            torch.cuda.empty_cache()

        # FP16 inference only
        try:
            torch.cuda.empty_cache()
            model = create_model(
                GaussianSplatModelCUDA,
                shape,
                centers,
                L,
                amps,
                sigma_min,
                use_fp16=True,
            )
            fp16_time = benchmark_forward(model, sync_cuda=True)
            config_result["fp16_inference_ms"] = round(fp16_time, 3)
            del model
            torch.cuda.empty_cache()
        except TypeError:
            config_result["fp16_note"] = "use_fp16 not supported in this commit"
        except Exception as e:
            config_result["fp16_error"] = str(e)
            torch.cuda.empty_cache()

        # Throughput
        fp32 = config_result.get("fp32_inference_ms")
        if fp32 and fp32 > 0:
            config_result["throughput_gvs_fp32"] = round((voxels / fp32) / 1e6, 2)

        amp = config_result.get("amp_inference_ms")
        if amp and amp > 0:
            config_result["throughput_gvs_amp"] = round((voxels / amp) / 1e6, 2)

        results["configs"][label] = config_result
        print(f"    FP32={fp32:.2f}ms" if fp32 else "    FP32=N/A", end="", flush=True)
        print(f"  AMP={amp:.2f}ms" if amp else "  AMP=N/A", flush=True)

    return results


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <worktree_path> <commit_sha> <output_json>")
        sys.exit(1)

    worktree_path = sys.argv[1]
    commit_sha = sys.argv[2]
    output_json = sys.argv[3]

    print(f"\n{'=' * 60}")
    print(f"Benchmarking commit {commit_sha}")
    print(f"Worktree: {worktree_path}")
    print(f"{'=' * 60}")

    try:
        results = run_benchmarks(worktree_path, commit_sha)
    except Exception as e:
        results = {
            "commit": commit_sha,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to {output_json}")


if __name__ == "__main__":
    main()
