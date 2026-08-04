#!/usr/bin/env python3
"""
Benchmark script for GPU-accelerated seeding strategies.

Compares CPU vs GPU performance for different volume sizes and methods.
Run with: hatch run python scripts/benchmarks/benchmark_seeding_gpu.py
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from arbol import aprint, asection

from luxar.gsplats.seeds import generate_seeds


def create_test_volume(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
    """Create synthetic test volume with structure."""
    rng = np.random.default_rng(seed)
    volume = rng.random(shape, dtype=np.float32)

    # Add some Gaussian blobs for structure
    for _ in range(10):
        center = tuple(rng.integers(0, s, size=1)[0] for s in shape)
        sigma = rng.uniform(5, 15)

        # Create Gaussian blob
        coords = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
        dist_sq = sum((c - center[i]) ** 2 for i, c in enumerate(coords))
        blob = np.exp(-dist_sq / (2 * sigma**2))
        volume += blob * rng.uniform(0.5, 1.5)

    return volume


def benchmark_method(
    volume: np.ndarray,
    method: str,
    device: str,
    n_runs: int = 3,
) -> Dict[str, float]:
    """Benchmark a single seeding method."""
    times = []

    for run in range(n_runs):
        # Clear cache
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        seeds = generate_seeds(volume, method=method, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        times.append(elapsed)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "n_seeds": len(seeds.centers),
    }


def format_time(seconds: float) -> str:
    """Format time in human-readable units."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.1f} µs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    # Check GPU availability
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    aprint("=" * 80)
    aprint("GPU-Accelerated Seeding Benchmark Suite")
    aprint("=" * 80)
    aprint(f"CUDA available: {cuda_available}")
    aprint(f"MPS available: {mps_available}")
    aprint("")

    if not cuda_available and not mps_available:
        aprint("⚠ No GPU detected. CPU-only benchmarks will be run.")
        aprint("")

    # Define test configurations
    configs = [
        ("2D Small", (128, 128)),
        ("2D Medium", (256, 256)),
        ("2D Large", (512, 512)),
        ("3D Small", (64, 64, 64)),
        ("3D Medium", (128, 128, 128)),
        ("3D Large", (256, 256, 256)),
    ]

    # Only test CUDA if available (MPS has similar performance characteristics)
    gpu_device = "cuda" if cuda_available else "mps" if mps_available else None

    # Test methods
    methods = ["edges", "grid", "auto"]

    results: List[Dict] = []

    for name, shape in configs:
        with asection(f"Benchmarking {name} {shape}"):
            # Create test volume
            volume = create_test_volume(shape)
            aprint(f"Volume size: {volume.nbytes / 1e6:.1f} MB")
            aprint("")

            for method in methods:
                aprint(f"Method: {method}")

                # CPU benchmark
                cpu_results = benchmark_method(volume, method, device="cpu", n_runs=3)
                cpu_time = cpu_results["mean"]
                aprint(
                    f"  CPU: {format_time(cpu_time)} ± {format_time(cpu_results['std'])} "
                    f"({cpu_results['n_seeds']} seeds)"
                )

                # GPU benchmark
                if gpu_device:
                    gpu_results = benchmark_method(
                        volume, method, device=gpu_device, n_runs=3
                    )
                    gpu_time = gpu_results["mean"]
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

                    aprint(
                        f"  GPU: {format_time(gpu_time)} ± {format_time(gpu_results['std'])} "
                        f"({gpu_results['n_seeds']} seeds)"
                    )
                    aprint(f"  Speedup: {speedup:.2f}x")

                    results.append(
                        {
                            "name": name,
                            "shape": shape,
                            "method": method,
                            "cpu_time": cpu_time,
                            "gpu_time": gpu_time,
                            "speedup": speedup,
                            "n_seeds": cpu_results["n_seeds"],
                        }
                    )
                else:
                    aprint("  GPU: N/A (no GPU available)")

                aprint("")

            aprint("")

    # Summary table
    if results:
        with asection("Performance Summary"):
            aprint(
                f"{'Volume':<15} {'Method':<12} {'CPU Time':<12} {'GPU Time':<12} {'Speedup':<10} {'Seeds':<8}"
            )
            aprint("-" * 80)

            for r in results:
                aprint(
                    f"{r['name']:<15} {r['method']:<12} "
                    f"{format_time(r['cpu_time']):<12} "
                    f"{format_time(r['gpu_time']):<12} "
                    f"{r['speedup']:>8.2f}x "
                    f"{r['n_seeds']:>8}"
                )

            aprint("")

            # Calculate average speedup by method
            aprint("Average Speedup by Method:")
            for method in methods:
                method_results = [r for r in results if r["method"] == method]
                if method_results:
                    avg_speedup = np.mean([r["speedup"] for r in method_results])
                    aprint(f"  {method:>12}: {avg_speedup:.2f}x")

            aprint("")

            # Calculate average speedup by size category
            aprint("Average Speedup by Volume Size:")
            size_categories = ["Small", "Medium", "Large"]
            for category in size_categories:
                cat_results = [r for r in results if category in r["name"]]
                if cat_results:
                    avg_speedup = np.mean([r["speedup"] for r in cat_results])
                    aprint(f"  {category:>12}: {avg_speedup:.2f}x")


def run_dimension_support_tests():
    """Test GPU support across different dimensions."""
    with asection("Dimension Support Tests"):
        aprint("Testing GPU support for 1D, 2D, 3D, 4D volumes...")
        aprint("")

        test_volumes = [
            ("1D", (256,)),
            ("2D", (128, 128)),
            ("3D", (64, 64, 64)),
            ("4D", (32, 32, 32, 32)),
        ]

        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"

        if not cuda_available:
            aprint("⚠ CUDA not available. Skipping GPU dimension tests.")
            return

        for name, shape in test_volumes:
            volume = create_test_volume(shape)
            aprint(f"{name} volume {shape}:")

            try:
                # Test edges method (uses Sobel + interpolation)
                start = time.perf_counter()
                seeds = generate_seeds(
                    volume, method="edges", device=device, n_seeds=50
                )
                elapsed = time.perf_counter() - start

                aprint(
                    f"  ✓ Edges seeding: {len(seeds.centers)} seeds in {format_time(elapsed)}"
                )
            except Exception as e:
                aprint(f"  ✗ Edges seeding failed: {e}")

            aprint("")


def main():
    """Run all benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark GPU-accelerated seeding")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer sizes, fewer runs)",
    )
    parser.add_argument(
        "--dims-only",
        action="store_true",
        help="Only test dimension support",
    )
    args = parser.parse_args()

    if args.dims_only:
        run_dimension_support_tests()
    else:
        run_benchmark_suite()
        aprint("")
        run_dimension_support_tests()


if __name__ == "__main__":
    main()
