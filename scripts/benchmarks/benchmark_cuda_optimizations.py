#!/usr/bin/env python3
"""
Benchmark runner for CUDA kernel optimization validation.

Runs a fixed set of representative configurations before and after
optimizations, outputting JSON for comparison. Supports --compare
mode to diff two result files and flag regressions.

Usage:
    # Run benchmark and save results
    hatch run python scripts/benchmarks/benchmark_cuda_optimizations.py \
        --label baseline --output benchmarks/cuda_opt_baseline.json

    # Compare two results
    hatch run python scripts/benchmarks/benchmark_cuda_optimizations.py \
        --compare benchmarks/cuda_opt_baseline.json benchmarks/cuda_opt1.json
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def create_test_data(N, shape, seed=42):
    """Create reproducible random test data for benchmarking."""
    np.random.seed(seed)
    d = len(shape)
    centers = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 2) + 1
    L = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
    for i in range(N):
        L[i] *= np.random.uniform(0.5, 2.0)
    amps = np.random.rand(N).astype(np.float32) + 0.1
    return centers, L, amps


def benchmark_forward(model, n_warmup=10, n_iters=25, use_amp=False):
    """Benchmark forward pass, return time in ms."""
    import torch

    for _ in range(n_warmup):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model()
        else:
            _ = model()
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_amp:
            with torch.amp.autocast("cuda"):
                _ = model()
        else:
            _ = model()
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1000
    return elapsed


def benchmark_training(model, n_warmup=10, n_iters=25, use_amp=False):
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

    # Forward only timing
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

    # Forward + backward timing
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


# Fixed representative configurations
CONFIGS = [
    (1000, (128, 128, 128), "3D_128_1K"),
    (10000, (256, 256, 256), "3D_256_10K"),
    (50000, (512, 512, 512), "3D_512_50K"),
    (5000, (1024, 1024), "2D_1024_5K"),
    (50000, (4096, 4096), "2D_4096_50K"),
]

# Metrics to compare (key, display_name, higher_is_worse)
METRICS = [
    ("fp32_inference_ms", "FP32 Infer", True),
    ("fp32_train_fwd_ms", "FP32 Fwd", True),
    ("fp32_train_bwd_ms", "FP32 Bwd", True),
    ("fp32_train_total_ms", "FP32 Train", True),
    ("amp_inference_ms", "AMP Infer", True),
    ("amp_train_fwd_ms", "AMP Fwd", True),
    ("amp_train_bwd_ms", "AMP Bwd", True),
    ("amp_train_total_ms", "AMP Train", True),
]


def run_benchmarks(label: str):
    """Run benchmark suite and return results dict."""
    import torch

    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
        GaussianSplatModelCUDA,
    )

    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

    results = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "configs": {},
    }

    for N, shape, config_label in CONFIGS:
        print(f"  {config_label} (N={N:,}, shape={shape})...", end="", flush=True)
        centers, L, amps = create_test_data(N, shape)
        sigma_min = (0.5,) * len(shape)
        voxels = int(np.prod(shape))

        config_result = {
            "N": N,
            "shape": list(shape),
            "voxels": voxels,
        }

        # FP32
        try:
            torch.cuda.empty_cache()
            model = GaussianSplatModelCUDA(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=sigma_min,
                device="cuda",
            )
            config_result["fp32_inference_ms"] = round(benchmark_forward(model), 3)

            fwd, bwd, total = benchmark_training(model, use_amp=False)
            config_result["fp32_train_fwd_ms"] = round(fwd, 3)
            config_result["fp32_train_bwd_ms"] = round(bwd, 3)
            config_result["fp32_train_total_ms"] = round(total, 3)

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            config_result["fp32_error"] = str(e)
            torch.cuda.empty_cache()

        # AMP
        try:
            torch.cuda.empty_cache()
            model = GaussianSplatModelCUDA(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=sigma_min,
                device="cuda",
            )
            config_result["amp_inference_ms"] = round(
                benchmark_forward(model, use_amp=True), 3
            )

            fwd, bwd, total = benchmark_training(model, use_amp=True)
            config_result["amp_train_fwd_ms"] = round(fwd, 3)
            config_result["amp_train_bwd_ms"] = round(bwd, 3)
            config_result["amp_train_total_ms"] = round(total, 3)

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            config_result["amp_error"] = str(e)
            torch.cuda.empty_cache()

        results["configs"][config_label] = config_result
        fp32 = config_result.get("fp32_inference_ms", "N/A")
        bwd = config_result.get("fp32_train_bwd_ms", "N/A")
        print(f"  FP32={fp32}ms  Bwd={bwd}ms", flush=True)

    return results


def compare_results(file_a: str, file_b: str):
    """Compare two benchmark result files and print formatted diff table."""
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    label_a = a.get("label", Path(file_a).stem)
    label_b = b.get("label", Path(file_b).stem)

    print(f"\nComparison: {label_a} vs {label_b}")
    print(f"  A: {file_a} (commit {a.get('commit', '?')})")
    print(f"  B: {file_b} (commit {b.get('commit', '?')})")
    print(f"  GPU: {b.get('gpu', '?')}")
    print()

    regressions = []
    improvements = []

    for metric_key, metric_name, higher_is_worse in METRICS:
        # Print header for each metric
        has_data = False
        rows = []

        for _, _, config_label in CONFIGS:
            ca = a.get("configs", {}).get(config_label, {})
            cb = b.get("configs", {}).get(config_label, {})

            val_a = ca.get(metric_key)
            val_b = cb.get(metric_key)

            if val_a is None or val_b is None:
                continue

            has_data = True
            if val_a > 0:
                pct_change = ((val_b - val_a) / val_a) * 100
            else:
                pct_change = 0.0

            # For time metrics, negative pct = improvement (faster)
            if higher_is_worse:
                is_regression = pct_change > 5.0
                is_improvement = pct_change < -5.0
            else:
                is_regression = pct_change < -5.0
                is_improvement = pct_change > 5.0

            flag = ""
            if is_regression:
                flag = " REGRESSION"
                regressions.append((config_label, metric_name, pct_change))
            elif is_improvement:
                flag = " IMPROVED"
                improvements.append((config_label, metric_name, pct_change))

            rows.append(
                f"    {config_label:<16s} {val_a:>8.3f} -> {val_b:>8.3f}  "
                f"({pct_change:+6.1f}%){flag}"
            )

        if has_data:
            print(f"  {metric_name}:")
            for row in rows:
                print(row)
            print()

    # Summary
    print("=" * 60)
    if regressions:
        print(f"  REGRESSIONS ({len(regressions)}):")
        for cfg, metric, pct in regressions:
            print(f"    {cfg} {metric}: {pct:+.1f}%")
    else:
        print("  No regressions (>5% slower)")

    if improvements:
        print(f"  IMPROVEMENTS ({len(improvements)}):")
        for cfg, metric, pct in improvements:
            print(f"    {cfg} {metric}: {pct:+.1f}%")
    else:
        print("  No significant improvements (>5% faster)")

    print("=" * 60)

    return len(regressions) == 0


def main():
    parser = argparse.ArgumentParser(description="CUDA optimization benchmark runner")
    parser.add_argument("--label", type=str, help="Label for this benchmark run")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two result JSON files",
    )

    args = parser.parse_args()

    if args.compare:
        ok = compare_results(args.compare[0], args.compare[1])
        sys.exit(0 if ok else 1)

    if not args.label or not args.output:
        parser.error("--label and --output are required when not using --compare")

    print(f"\nCUDA Optimization Benchmark: {args.label}")
    print("=" * 60)

    results = run_benchmarks(args.label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
