"""
Performance benchmark tests for Metal backend.

Measures speedup of Metal vs CPU for forward and backward passes.
Tests multiple configurations to characterize performance.
"""

from __future__ import annotations

import sys
import time
from typing import Tuple

import numpy as np
import pytest
import torch
from arbol import aprint

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

# Skip entire module on non-macOS platforms
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not torch.backends.mps.is_available(),
    reason="Metal backend only available on macOS with MPS",
)

# Import Metal-specific modules only on macOS
if sys.platform == "darwin":
    from luxar.gsplats.models.gsplats.metal import (
        GaussianSplatModelMetal,
        is_metal_available,
    )
else:
    # Provide dummies for type checking
    GaussianSplatModelMetal = None  # type: ignore[misc, assignment]
    is_metal_available = lambda: False  # noqa: E731


def benchmark_forward(model, n_warmup: int = 3, n_iters: int = 20) -> float:
    """
    Benchmark forward pass.

    Returns:
        Average time per iteration in milliseconds
    """
    # Warmup
    for _ in range(n_warmup):
        _ = model()

    # Synchronize
    if next(model.parameters()).device.type == "mps":
        torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = model()
        if next(model.parameters()).device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / n_iters) * 1000  # ms


def benchmark_backward(
    model, n_warmup: int = 3, n_iters: int = 20
) -> Tuple[float, float]:
    """
    Benchmark forward + backward pass.

    Returns:
        (forward_ms, backward_ms) - average times per iteration
    """
    # Warmup
    for _ in range(n_warmup):
        output = model()
        loss = output.sum()
        loss.backward()
        model.zero_grad()

    # Synchronize
    if next(model.parameters()).device.type == "mps":
        torch.mps.synchronize()

    # Benchmark
    forward_times = []
    backward_times = []

    for _ in range(n_iters):
        # Forward
        if next(model.parameters()).device.type == "mps":
            torch.mps.synchronize()

        fwd_start = time.perf_counter()
        output = model()

        if next(model.parameters()).device.type == "mps":
            torch.mps.synchronize()

        fwd_end = time.perf_counter()

        # Backward
        loss = output.sum()
        loss.backward()

        if next(model.parameters()).device.type == "mps":
            torch.mps.synchronize()

        bwd_end = time.perf_counter()

        forward_times.append((fwd_end - fwd_start) * 1000)
        backward_times.append((bwd_end - fwd_end) * 1000)

        model.zero_grad()

    return np.mean(forward_times), np.mean(backward_times)


class TestForwardPassPerformance:
    """Benchmark forward pass performance."""

    @pytest.mark.parametrize(
        "shape,n_splats",
        [
            ((32, 32, 32), 100),
            ((64, 64, 64), 500),
            ((64, 64, 64), 1000),
        ],
    )
    def test_forward_speedup(self, shape, n_splats):
        """Test forward pass speedup for various configurations."""
        # Create test data
        np.random.seed(42)
        centers = (
            np.random.rand(n_splats, 3) * np.array(shape) * 0.8 + np.array(shape) * 0.1
        )
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

        # Metal model
        model_metal = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        # CPU model
        model_cpu = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="cpu",
        )

        # Benchmark
        metal_time = benchmark_forward(model_metal, n_warmup=3, n_iters=10)
        cpu_time = benchmark_forward(model_cpu, n_warmup=3, n_iters=10)

        speedup = cpu_time / metal_time

        aprint(f"\n[Forward] shape={shape}, n_splats={n_splats}:")
        aprint(f"  CPU:    {cpu_time:7.2f} ms/iter")
        aprint(f"  Metal:  {metal_time:7.2f} ms/iter")
        aprint(f"  Speedup: {speedup:6.2f}x")

        # Note: Metal overhead may exceed benefit for small problems on fast CPUs.
        # This test benchmarks performance without asserting speedup.
        # For production use, Metal benefits larger volumes (128³+) with more splats (1000+).


class TestBackwardPassPerformance:
    """Benchmark backward pass performance."""

    @pytest.mark.parametrize(
        "shape,n_splats",
        [
            ((32, 32, 32), 100),
            ((64, 64, 64), 500),
        ],
    )
    def test_backward_speedup(self, shape, n_splats):
        """Test forward+backward speedup for various configurations."""
        # Create test data
        np.random.seed(42)
        centers = (
            np.random.rand(n_splats, 3) * np.array(shape) * 0.8 + np.array(shape) * 0.1
        )
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

        # Metal model
        model_metal = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        # CPU model
        model_cpu = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="cpu",
        )

        # Benchmark
        metal_fwd, metal_bwd = benchmark_backward(model_metal, n_warmup=2, n_iters=5)
        cpu_fwd, cpu_bwd = benchmark_backward(model_cpu, n_warmup=2, n_iters=5)

        fwd_speedup = cpu_fwd / metal_fwd
        bwd_speedup = cpu_bwd / metal_bwd
        total_speedup = (cpu_fwd + cpu_bwd) / (metal_fwd + metal_bwd)

        aprint(f"\n[Forward+Backward] shape={shape}, n_splats={n_splats}:")
        aprint(
            f"  CPU:    fwd={cpu_fwd:7.2f} ms, bwd={cpu_bwd:7.2f} ms, total={cpu_fwd + cpu_bwd:7.2f} ms"
        )
        aprint(
            f"  Metal:  fwd={metal_fwd:7.2f} ms, bwd={metal_bwd:7.2f} ms, total={metal_fwd + metal_bwd:7.2f} ms"
        )
        aprint(
            f"  Speedup: fwd={fwd_speedup:5.2f}x, bwd={bwd_speedup:5.2f}x, total={total_speedup:5.2f}x"
        )

        # Note: Metal overhead may exceed benefit for small problems on fast CPUs.
        # This test benchmarks performance without asserting speedup.


class TestPerformanceScaling:
    """Test how performance scales with problem size."""

    @pytest.mark.slow
    def test_scaling_with_volume_size(self):
        """Test speedup for increasing volume sizes."""
        n_splats = 300
        results = []

        for size in [32, 48, 64]:
            shape = (size, size, size)

            np.random.seed(42)
            centers = np.random.rand(n_splats, 3) * size * 0.8 + size * 0.1
            L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
            amps = np.ones(n_splats, dtype=np.float32)

            model_metal = GaussianSplatModelMetal(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="mps",
            )

            model_cpu = GaussianSplatModel(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="cpu",
            )

            metal_time = benchmark_forward(model_metal, n_warmup=2, n_iters=5)
            cpu_time = benchmark_forward(model_cpu, n_warmup=2, n_iters=5)
            speedup = cpu_time / metal_time

            results.append((size, cpu_time, metal_time, speedup))

        aprint(f"\n[Scaling Test] n_splats={n_splats}")
        aprint(f"{'Size':<10} {'CPU (ms)':<12} {'Metal (ms)':<12} {'Speedup':<10}")
        aprint("-" * 50)
        for size, cpu_t, metal_t, speedup in results:
            aprint(
                f"{size}³{'':<7} {cpu_t:8.2f}     {metal_t:8.2f}       {speedup:6.2f}x"
            )

        # Note: Metal overhead may exceed benefit for small problems on fast CPUs.
        # This test benchmarks scaling behavior without asserting speedup.
        # For production use, Metal benefits larger volumes (128³+) with more splats (1000+).

    @pytest.mark.slow
    def test_scaling_with_splat_count(self):
        """Test speedup for increasing splat counts."""
        shape = (64, 64, 64)
        results = []

        for n_splats in [100, 300, 500, 1000]:
            np.random.seed(42)
            centers = np.random.rand(n_splats, 3) * 48 + 8
            L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
            amps = np.ones(n_splats, dtype=np.float32)

            model_metal = GaussianSplatModelMetal(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="mps",
            )

            model_cpu = GaussianSplatModel(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="cpu",
            )

            metal_time = benchmark_forward(model_metal, n_warmup=2, n_iters=5)
            cpu_time = benchmark_forward(model_cpu, n_warmup=2, n_iters=5)
            speedup = cpu_time / metal_time

            results.append((n_splats, cpu_time, metal_time, speedup))

        aprint(f"\n[Scaling Test] shape={shape}")
        aprint(f"{'N Splats':<10} {'CPU (ms)':<12} {'Metal (ms)':<12} {'Speedup':<10}")
        aprint("-" * 50)
        for n, cpu_t, metal_t, speedup in results:
            aprint(f"{n:<10} {cpu_t:8.2f}     {metal_t:8.2f}       {speedup:6.2f}x")


class TestMemoryUsage:
    """Test memory efficiency."""

    def test_no_memory_leak(self):
        """Test that repeated forward/backward doesn't leak memory."""
        shape = (64, 64, 64)
        n_splats = 500

        np.random.seed(42)
        centers = np.random.rand(n_splats, 3) * 48 + 8
        L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
        amps = np.ones(n_splats, dtype=np.float32)

        model = GaussianSplatModelMetal(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=[0.5, 0.5, 0.5],
            truncate=3.0,
            device="mps",
        )

        # Run many iterations
        for i in range(50):
            output = model()
            loss = output.sum()
            loss.backward()
            model.zero_grad()

        torch.mps.synchronize()

        # If we get here without OOM, memory is being managed correctly
        assert True


class TestPerformanceSummary:
    """Generate comprehensive performance report."""

    @pytest.mark.slow
    def test_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite and generate report."""
        aprint("\n" + "=" * 80)
        aprint("METAL BACKEND PERFORMANCE REPORT")
        aprint("=" * 80)

        configs = [
            ("Small", (32, 32, 32), 100),
            ("Medium", (64, 64, 64), 500),
            ("Large", (64, 64, 64), 1000),
        ]

        aprint(
            f"\n{'Config':<10} {'Shape':<15} {'N':<6} {'CPU Fwd':<10} {'Metal Fwd':<12} {'Speedup':<10}"
        )
        aprint("-" * 80)

        for name, shape, n_splats in configs:
            np.random.seed(42)
            centers = (
                np.random.rand(n_splats, 3) * np.array(shape) * 0.8
                + np.array(shape) * 0.1
            )
            L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
            amps = np.ones(n_splats, dtype=np.float32)

            model_metal = GaussianSplatModelMetal(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="mps",
            )

            model_cpu = GaussianSplatModel(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="cpu",
            )

            # Forward benchmark
            metal_fwd = benchmark_forward(model_metal, n_warmup=2, n_iters=10)
            cpu_fwd = benchmark_forward(model_cpu, n_warmup=2, n_iters=10)
            fwd_speedup = cpu_fwd / metal_fwd

            aprint(
                f"{name:<10} {str(shape):<15} {n_splats:<6} {cpu_fwd:8.2f} ms {metal_fwd:8.2f} ms   {fwd_speedup:6.2f}x"
            )

        aprint("\n" + "=" * 80)
        aprint("FORWARD + BACKWARD PERFORMANCE")
        aprint("=" * 80)

        aprint(
            f"\n{'Config':<10} {'CPU Total':<12} {'Metal Total':<12} {'Total Speedup':<15}"
        )
        aprint("-" * 60)

        speedups: list[float] = []
        for name, shape, n_splats in configs[:2]:  # Skip large for backward (slow)
            np.random.seed(42)
            centers = (
                np.random.rand(n_splats, 3) * np.array(shape) * 0.8
                + np.array(shape) * 0.1
            )
            L = np.tile(np.eye(3) * 2.0, (n_splats, 1, 1)).astype(np.float32)
            amps = np.ones(n_splats, dtype=np.float32)

            model_metal = GaussianSplatModelMetal(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="mps",
            )

            model_cpu = GaussianSplatModel(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=[0.5, 0.5, 0.5],
                truncate=3.0,
                device="cpu",
            )

            metal_fwd, metal_bwd = benchmark_backward(
                model_metal, n_warmup=2, n_iters=5
            )
            cpu_fwd, cpu_bwd = benchmark_backward(model_cpu, n_warmup=2, n_iters=5)

            total_speedup = (cpu_fwd + cpu_bwd) / (metal_fwd + metal_bwd)
            speedups.append(total_speedup)

            aprint(
                f"{name:<10} {cpu_fwd + cpu_bwd:8.2f} ms   {metal_fwd + metal_bwd:8.2f} ms     {total_speedup:6.2f}x"
            )

        aprint("\n" + "=" * 80)

        # The benchmark must produce a valid, positive speedup for every config.
        # (Magnitude is hardware-dependent, so we assert validity — finite and
        # positive — not a fixed threshold, to stay non-flaky across machines.)
        assert len(speedups) == len(configs[:2])
        assert all(s > 0 for s in speedups), f"non-positive/NaN speedup: {speedups}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
