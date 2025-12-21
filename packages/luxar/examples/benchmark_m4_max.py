#!/usr/bin/env python
"""
Benchmark Gaussian splat fitting on M4 Max: CPU vs MPS vs torch.compile()

This script compares:
1. CPU (baseline)
2. MPS (Apple Silicon GPU)
3. CPU + torch.compile()
4. MPS + torch.compile()

And profiles to identify bottlenecks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from arbol import aprint, asection

# Check torch version and available backends
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    device: str
    compiled: bool
    forward_time_ms: float
    backward_time_ms: float
    total_iter_time_ms: float
    n_iterations: int
    total_time_s: float
    final_loss: float


def create_test_volume(size: int = 64, n_blobs: int = 5, seed: int = 42) -> np.ndarray:
    """Create a synthetic test volume with Gaussian blobs."""
    np.random.seed(seed)
    volume = np.zeros((size, size, size), dtype=np.float32)

    for _ in range(n_blobs):
        # Random center
        center = np.random.randint(10, size - 10, size=3)
        # Random sigma
        sigma = np.random.uniform(3, 8)
        # Random amplitude
        amp = np.random.uniform(0.5, 1.0)

        # Create coordinate grids
        z, y, x = np.ogrid[:size, :size, :size]

        # Add Gaussian blob
        dist_sq = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2
        volume += amp * np.exp(-dist_sq / (2 * sigma**2))

    return volume


def benchmark_forward_backward(
    model: torch.nn.Module,
    target: torch.Tensor,
    n_warmup: int = 5,
    n_iterations: int = 50,
) -> tuple[float, float, float]:
    """
    Benchmark forward and backward pass times.

    Returns: (forward_ms, backward_ms, total_ms) averaged over iterations
    """
    # Loss function
    def compute_loss(pred: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)

    # Warmup
    for _ in range(n_warmup):
        pred = model()
        loss = compute_loss(pred)
        loss.backward()
        model.zero_grad()

    # Synchronize before timing
    if target.device.type == "mps":
        torch.mps.synchronize()
    elif target.device.type == "cuda":
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []

    for _ in range(n_iterations):
        # Forward timing
        start = time.perf_counter()
        pred = model()
        loss = compute_loss(pred)

        if target.device.type == "mps":
            torch.mps.synchronize()

        forward_end = time.perf_counter()

        # Backward timing
        loss.backward()

        if target.device.type == "mps":
            torch.mps.synchronize()

        backward_end = time.perf_counter()

        forward_times.append((forward_end - start) * 1000)
        backward_times.append((backward_end - forward_end) * 1000)

        model.zero_grad()

    avg_forward = np.mean(forward_times)
    avg_backward = np.mean(backward_times)

    return avg_forward, avg_backward, avg_forward + avg_backward


def run_optimization_benchmark(
    volume: np.ndarray,
    device_str: str,
    use_compile: bool = False,
    compile_backend: str = "inductor",
    n_iterations: int = 100,
    n_splats: int = 500,
    lr: float = 0.05,
) -> BenchmarkResult:
    """Run a full optimization benchmark."""

    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    device = torch.device(device_str)

    # Generate random seed positions
    np.random.seed(42)
    d = len(volume.shape)
    centers0 = np.random.rand(n_splats, d) * (np.array(volume.shape) - 1)

    # Initialize Cholesky factors (isotropic)
    L0 = np.zeros((n_splats, d, d), dtype=np.float32)
    for i in range(d):
        L0[:, i, i] = 2.0  # Initial sigma

    # Sample amplitudes from volume
    idx = np.clip(np.round(centers0).astype(int), 0, np.array(volume.shape) - 1)
    V_normalized = volume / (volume.max() + 1e-8)
    amps0 = V_normalized[tuple(idx.T)]

    # Create model
    model = GaussianSplatModel(
        shape=volume.shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5] * d,
        truncate=3.0,
        device=device,
    )

    # Optionally compile
    compiled_str = ""
    if use_compile:
        try:
            model = torch.compile(model, backend=compile_backend)
            compiled_str = f" + compile({compile_backend})"
        except Exception as e:
            aprint(f"Warning: torch.compile failed with {compile_backend}: {e}")
            aprint("Falling back to eager mode")
            use_compile = False

    # Target tensor
    target = torch.tensor(V_normalized, dtype=torch.float32, device=device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function
    def compute_loss(pred: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))  # L1 loss

    # Warmup (especially important for compiled models)
    with asection(f"Warmup {device_str}{compiled_str}"):
        for _ in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = compute_loss(pred)
            loss.backward()
            optimizer.step()

        # Sync
        if device_str == "mps":
            torch.mps.synchronize()

    # Benchmark
    forward_times = []
    backward_times = []
    step_times = []

    with asection(f"Benchmark {device_str}{compiled_str} ({n_iterations} iters)"):
        start_total = time.perf_counter()

        for it in range(n_iterations):
            iter_start = time.perf_counter()

            optimizer.zero_grad()

            # Forward
            fwd_start = time.perf_counter()
            pred = model()
            loss = compute_loss(pred)
            if device_str == "mps":
                torch.mps.synchronize()
            fwd_end = time.perf_counter()

            # Backward
            loss.backward()
            if device_str == "mps":
                torch.mps.synchronize()
            bwd_end = time.perf_counter()

            # Step
            optimizer.step()
            if device_str == "mps":
                torch.mps.synchronize()
            step_end = time.perf_counter()

            forward_times.append((fwd_end - fwd_start) * 1000)
            backward_times.append((bwd_end - fwd_end) * 1000)
            step_times.append((step_end - iter_start) * 1000)

            if (it + 1) % 25 == 0:
                aprint(f"  [{it+1:3d}/{n_iterations}] loss={loss.item():.5f}")

        end_total = time.perf_counter()
        total_time = end_total - start_total

    # Results
    avg_forward = np.mean(forward_times[10:])  # Skip first 10 for warmup effects
    avg_backward = np.mean(backward_times[10:])
    avg_total = np.mean(step_times[10:])

    return BenchmarkResult(
        device=device_str,
        compiled=use_compile,
        forward_time_ms=avg_forward,
        backward_time_ms=avg_backward,
        total_iter_time_ms=avg_total,
        n_iterations=n_iterations,
        total_time_s=total_time,
        final_loss=loss.item(),
    )


def profile_model(
    volume: np.ndarray,
    device_str: str,
    n_splats: int = 500,
    n_iterations: int = 20,
) -> None:
    """Profile the model to identify bottlenecks."""

    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    device = torch.device(device_str)

    # Setup model (same as benchmark)
    np.random.seed(42)
    d = len(volume.shape)
    centers0 = np.random.rand(n_splats, d) * (np.array(volume.shape) - 1)

    L0 = np.zeros((n_splats, d, d), dtype=np.float32)
    for i in range(d):
        L0[:, i, i] = 2.0

    idx = np.clip(np.round(centers0).astype(int), 0, np.array(volume.shape) - 1)
    V_normalized = volume / (volume.max() + 1e-8)
    amps0 = V_normalized[tuple(idx.T)]

    model = GaussianSplatModel(
        shape=volume.shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5] * d,
        truncate=3.0,
        device=device,
    )

    target = torch.tensor(V_normalized, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    def compute_loss(pred: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        pred = model()
        loss = compute_loss(pred)
        loss.backward()
        optimizer.step()

    if device_str == "mps":
        torch.mps.synchronize()

    # Profile
    with asection(f"Profiling {device_str}"):
        activities = [torch.profiler.ProfilerActivity.CPU]

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(n_iterations):
                optimizer.zero_grad()
                pred = model()
                loss = compute_loss(pred)
                loss.backward()
                optimizer.step()

            if device_str == "mps":
                torch.mps.synchronize()

        # Print results
        print("\n" + "=" * 80)
        print(f"PROFILE RESULTS ({device_str}) - Top 20 by CPU time")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        print("\n" + "=" * 80)
        print(f"PROFILE RESULTS ({device_str}) - Top 10 by Self CPU time")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


def main():
    """Run all benchmarks."""

    with asection("Creating test volume"):
        volume = create_test_volume(size=64, n_blobs=8)
        aprint(f"Volume shape: {volume.shape}")
        aprint(f"Volume range: [{volume.min():.3f}, {volume.max():.3f}]")

    # Configuration
    n_splats = 1000
    n_iterations = 100

    results: list[BenchmarkResult] = []

    # =========================================================================
    # Benchmark 1: CPU (baseline)
    # =========================================================================
    with asection("Benchmark 1: CPU (baseline)"):
        result = run_optimization_benchmark(
            volume, "cpu",
            use_compile=False,
            n_iterations=n_iterations,
            n_splats=n_splats,
        )
        results.append(result)
        aprint(f"Forward: {result.forward_time_ms:.2f} ms")
        aprint(f"Backward: {result.backward_time_ms:.2f} ms")
        aprint(f"Total/iter: {result.total_iter_time_ms:.2f} ms")

    # =========================================================================
    # Benchmark 2: MPS
    # =========================================================================
    if torch.backends.mps.is_available():
        with asection("Benchmark 2: MPS"):
            result = run_optimization_benchmark(
                volume, "mps",
                use_compile=False,
                n_iterations=n_iterations,
                n_splats=n_splats,
            )
            results.append(result)
            aprint(f"Forward: {result.forward_time_ms:.2f} ms")
            aprint(f"Backward: {result.backward_time_ms:.2f} ms")
            aprint(f"Total/iter: {result.total_iter_time_ms:.2f} ms")
    else:
        aprint("MPS not available, skipping")

    # =========================================================================
    # Benchmark 3: CPU + torch.compile (eager - safest)
    # Note: inductor backend has issues with paths containing spaces on macOS
    # =========================================================================
    with asection("Benchmark 3: CPU + torch.compile(eager)"):
        try:
            result = run_optimization_benchmark(
                volume, "cpu",
                use_compile=True,
                compile_backend="eager",
                n_iterations=n_iterations,
                n_splats=n_splats,
            )
            results.append(result)
            aprint(f"Forward: {result.forward_time_ms:.2f} ms")
            aprint(f"Backward: {result.backward_time_ms:.2f} ms")
            aprint(f"Total/iter: {result.total_iter_time_ms:.2f} ms")
        except Exception as e:
            aprint(f"CPU + compile(eager) failed: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"{'Configuration':<40} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<15} {'Speedup':<10}")
    print("-" * 100)

    baseline_time = results[0].total_iter_time_ms if results else 1.0

    for r in results:
        compiled_str = " + compile" if r.compiled else ""
        config = f"{r.device}{compiled_str}"
        speedup = baseline_time / r.total_iter_time_ms
        print(f"{config:<40} {r.forward_time_ms:<15.2f} {r.backward_time_ms:<15.2f} {r.total_iter_time_ms:<15.2f} {speedup:<10.2f}x")

    print("=" * 100)

    # =========================================================================
    # Profiling (CPU only - it's the fastest)
    # =========================================================================
    print("\n")
    with asection("Profiling CPU to identify bottlenecks"):
        profile_model(volume, "cpu", n_splats=n_splats, n_iterations=20)


if __name__ == "__main__":
    main()
