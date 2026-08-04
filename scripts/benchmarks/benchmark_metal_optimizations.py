#!/usr/bin/env python3
"""
Benchmark runner for Apple Metal (MPS) gsplat kernel optimization validation.

Runs a fixed set of representative configurations before and after kernel
changes, outputting JSON for comparison. Mirrors the interface of
``benchmark_cuda_optimizations.py`` so the same workflow applies on
M-series macs as on CUDA hosts.

The ``--stress`` mode runs many short forward+backward passes and reports
process RSS growth so MET-1 (autoreleasepool) leaks would show up as
linear RSS growth rather than a plateau.

Usage:
    # Run benchmark and save results
    hatch run python scripts/benchmarks/benchmark_metal_optimizations.py \\
        --label baseline --output benchmarks/metal_baseline.json

    # Compare two results
    hatch run python scripts/benchmarks/benchmark_metal_optimizations.py \\
        --compare benchmarks/metal_baseline.json benchmarks/metal_v2.json

    # Stress / leak-check mode (10 000 dispatches, RSS report)
    hatch run python scripts/benchmarks/benchmark_metal_optimizations.py \\
        --stress --label leak-check
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ============================================================================
# Environment guard
# ============================================================================


def require_metal() -> None:
    """Exit early if Metal/MPS isn't available — there's nothing to benchmark."""
    import torch

    if sys.platform != "darwin":
        print("Skipping: Metal benchmark requires macOS.", file=sys.stderr)
        sys.exit(0)
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        print(
            "Skipping: Metal benchmark requires PyTorch with MPS available.",
            file=sys.stderr,
        )
        sys.exit(0)


# ============================================================================
# Test data
# ============================================================================


def create_test_data(N: int, shape: tuple[int, ...], seed: int = 42):
    """Create reproducible random test data for benchmarking."""
    rng = np.random.default_rng(seed)
    d = len(shape)
    centers = rng.random((N, d), dtype=np.float32) * (np.array(shape) - 2) + 1
    L = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
    L *= rng.uniform(0.5, 2.0, size=(N, 1, 1)).astype(np.float32)
    amps = rng.random(N, dtype=np.float32) + 0.1
    return centers.astype(np.float32), L.astype(np.float32), amps.astype(np.float32)


# ============================================================================
# Per-call benchmarks (forward / training)
# ============================================================================


def benchmark_forward(model, n_warmup: int = 5, n_iters: int = 25) -> float:
    """Benchmark forward pass; returns mean wall-clock per iteration in ms."""
    import torch

    for _ in range(n_warmup):
        _ = model()
        torch.mps.synchronize()

    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model()
        torch.mps.synchronize()
    return (time.perf_counter() - start) / n_iters * 1000.0


def benchmark_training(
    model, n_warmup: int = 5, n_iters: int = 25
) -> tuple[float, float, float]:
    """Benchmark forward + backward; returns (fwd_ms, bwd_ms, total_ms)."""
    import torch

    model.train()

    # Warm-up + forward-only timing for fwd_ms
    for _ in range(n_warmup):
        _ = model()
        torch.mps.synchronize()

    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model()
        torch.mps.synchronize()
    fwd_ms = (time.perf_counter() - start) / n_iters * 1000.0

    # Total (forward + backward + zero_grad)
    for _ in range(n_warmup):
        out = model()
        out.sum().backward()
        model.zero_grad(set_to_none=True)
        torch.mps.synchronize()

    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        out = model()
        out.sum().backward()
        model.zero_grad(set_to_none=True)
        torch.mps.synchronize()
    total_ms = (time.perf_counter() - start) / n_iters * 1000.0

    bwd_ms = max(0.0, total_ms - fwd_ms)
    return fwd_ms, bwd_ms, total_ms


# ============================================================================
# Config and metrics
# ============================================================================

# Representative configurations — the headline 128³ @ 32k matches the
# performance numbers published in metal/README.md so the benchmark can
# quantify regressions against that claim.
CONFIGS: list[tuple[int, tuple[int, ...], str]] = [
    (1_000, (64, 64, 64), "3D_64_1K"),
    (32_000, (128, 128, 128), "3D_128_32K"),  # README claim baseline
    (10_000, (256, 256, 256), "3D_256_10K"),
]

METRICS: list[tuple[str, str, bool]] = [
    ("inference_ms", "Forward", True),
    ("train_fwd_ms", "Train Fwd", True),
    ("train_bwd_ms", "Train Bwd", True),
    ("train_total_ms", "Train Total", True),
]


# ============================================================================
# Main run
# ============================================================================


def run_benchmarks(label: str) -> dict:
    """Run benchmark suite and return results dict."""
    require_metal()
    import torch

    from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

    commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode()
        .strip()
    )

    results: dict = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pytorch_version": torch.__version__,
        "configs": {},
    }

    for N, shape, config_label in CONFIGS:
        print(f"  {config_label} (N={N:,}, shape={shape})...", end="", flush=True)
        centers, L, amps = create_test_data(N, shape)
        sigma_min = (0.5,) * len(shape)
        voxels = int(np.prod(shape))

        config_result: dict = {
            "N": N,
            "shape": list(shape),
            "voxels": voxels,
        }

        try:
            model = GaussianSplatModelMetal(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=sigma_min,
                device="mps",
            )
            config_result["inference_ms"] = round(benchmark_forward(model), 3)

            fwd, bwd, total = benchmark_training(model)
            config_result["train_fwd_ms"] = round(fwd, 3)
            config_result["train_bwd_ms"] = round(bwd, 3)
            config_result["train_total_ms"] = round(total, 3)

            del model
        except Exception as e:  # noqa: BLE001 — top-level reporting layer
            config_result["error"] = str(e)

        results["configs"][config_label] = config_result
        fwd = config_result.get("inference_ms", "N/A")
        bwd = config_result.get("train_bwd_ms", "N/A")
        print(f"  Fwd={fwd}ms  Bwd={bwd}ms", flush=True)

    return results


# ============================================================================
# Compare mode (mirrors CUDA benchmark)
# ============================================================================


def compare_results(file_a: str, file_b: str) -> bool:
    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    label_a = a.get("label", Path(file_a).stem)
    label_b = b.get("label", Path(file_b).stem)

    print(f"\nComparison: {label_a} vs {label_b}")
    print(f"  A: {file_a} (commit {a.get('commit', '?')})")
    print(f"  B: {file_b} (commit {b.get('commit', '?')})")
    print(f"  Machine: {b.get('machine', '?')} on {b.get('platform', '?')}")
    print()

    regressions: list[tuple[str, str, float]] = []
    improvements: list[tuple[str, str, float]] = []

    for metric_key, metric_name, higher_is_worse in METRICS:
        has_data = False
        rows: list[str] = []

        for _, _, config_label in CONFIGS:
            ca = a.get("configs", {}).get(config_label, {})
            cb = b.get("configs", {}).get(config_label, {})
            val_a = ca.get(metric_key)
            val_b = cb.get(metric_key)
            if val_a is None or val_b is None:
                continue
            has_data = True
            pct_change = ((val_b - val_a) / val_a) * 100 if val_a > 0 else 0.0

            is_regression = (
                (pct_change > 5.0) if higher_is_worse else (pct_change < -5.0)
            )
            is_improvement = (
                (pct_change < -5.0) if higher_is_worse else (pct_change > 5.0)
            )
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

    return not regressions


# ============================================================================
# Stress / leak-check mode (validates MET-1 @autoreleasepool)
# ============================================================================


def _rss_bytes() -> int:
    """Return current process RSS in bytes (macOS / Linux compatible)."""
    if sys.platform == "darwin":
        # `ps -o rss=` reports KB on macOS.
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
        return int(out.strip()) * 1024
    # Linux fallback — not the primary use case for this script.
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def run_stress(label: str, n_iters: int = 10_000) -> dict:
    """Run many short dispatches and report RSS at intervals.

    Pre-MET-1 (no @autoreleasepool), expect linear RSS growth as
    autoreleased Obj-C objects accumulate on the thread's outer pool.
    Post-fix, expect a plateau.
    """
    require_metal()
    import torch

    from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

    shape = (64, 64, 64)
    N = 4_096
    centers, L, amps = create_test_data(N, shape)
    sigma_min = (0.5,) * len(shape)

    model = GaussianSplatModelMetal(
        shape=shape,
        centers0=centers,
        L0=L,
        amps0=amps,
        sigma_min_diag=sigma_min,
        device="mps",
    )
    model.train()

    # Warm-up so JIT compilation / first-touch cost doesn't pollute the
    # baseline reading.
    for _ in range(50):
        _ = model()
        torch.mps.synchronize()

    samples = []
    interval = max(1, n_iters // 10)
    rss_start = _rss_bytes()
    samples.append({"iter": 0, "rss_bytes": rss_start})
    print(f"  iter      0   RSS = {rss_start / 1024 / 1024:8.2f} MB  (baseline)")

    for i in range(1, n_iters + 1):
        out = model()
        out.sum().backward()
        model.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        if i % interval == 0:
            rss = _rss_bytes()
            samples.append({"iter": i, "rss_bytes": rss})
            delta_mb = (rss - rss_start) / 1024 / 1024
            print(
                f"  iter {i:>6d}   RSS = {rss / 1024 / 1024:8.2f} MB  "
                f"(Δ {delta_mb:+7.2f} MB)"
            )

    rss_end = _rss_bytes()
    delta_mb_total = (rss_end - rss_start) / 1024 / 1024
    print()
    print(f"  Total RSS growth: {delta_mb_total:+.2f} MB over {n_iters} iterations")
    if abs(delta_mb_total) < 50:
        print("  PASS — RSS plateau is consistent with @autoreleasepool wrapping.")
    else:
        print(
            "  WARNING — significant RSS growth; investigate autorelease leaks "
            "(MET-1) or other accumulating allocations."
        )

    return {
        "label": label,
        "mode": "stress",
        "iterations": n_iters,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": rss_end,
        "rss_delta_mb": delta_mb_total,
        "samples": samples,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pytorch_version": torch.__version__,
    }


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Metal optimization benchmark runner")
    parser.add_argument("--label", type=str, help="Label for this benchmark run")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two result JSON files",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run leak-check stress mode (many short dispatches, RSS reporting)",
    )
    parser.add_argument(
        "--stress-iters",
        type=int,
        default=10_000,
        help="Number of stress iterations (default: 10000)",
    )

    args = parser.parse_args()

    if args.compare:
        ok = compare_results(args.compare[0], args.compare[1])
        sys.exit(0 if ok else 1)

    if args.stress:
        if not args.label:
            parser.error("--label is required with --stress")
        print(f"\nMetal Stress / Leak Check: {args.label}")
        print("=" * 60)
        results = run_stress(args.label, args.stress_iters)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults written to {args.output}")
        return

    if not args.label or not args.output:
        parser.error(
            "--label and --output are required when not using --compare or --stress"
        )

    print(f"\nMetal Optimization Benchmark: {args.label}")
    print("=" * 60)
    results = run_benchmarks(args.label)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
