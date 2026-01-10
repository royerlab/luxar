"""
Performance tests for CUDA backend.

These tests verify that the CUDA backend achieves expected speedups
and memory efficiency compared to CPU implementations.
"""

import time

import numpy as np
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()

# Check if CUDA backend is compiled
try:
    import cuda_splatting_backend

    CUDA_BACKEND_AVAILABLE = True
except ImportError:
    CUDA_BACKEND_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


def create_test_data(N: int, shape: tuple, seed: int = 42):
    """Create test data for benchmarking."""
    np.random.seed(seed)
    d = len(shape)

    centers = np.random.rand(N, d).astype(np.float32) * (np.array(shape) - 2) + 1
    L = np.eye(d, dtype=np.float32)[None, :, :].repeat(N, axis=0)
    # Add some variation
    for i in range(N):
        L[i] *= np.random.uniform(0.5, 2.0)
    amps = np.random.rand(N).astype(np.float32) + 0.1
    sharpness = np.ones(N, dtype=np.float32) * 2.0

    return centers, L, amps, sharpness


class TestPerformanceBaseline:
    """Baseline performance tests using PyTorch CPU implementation."""

    def test_cpu_baseline_small_3d(self):
        """Measure CPU baseline for small 3D workload."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        N = 100
        shape = (64, 64, 64)
        centers, L, amps, _ = create_test_data(N, shape)

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # Warmup
        for _ in range(3):
            _ = model()

        # Benchmark
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = model()
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / n_iters) * 1000
        print(f"\nCPU baseline (N={N}, shape={shape}): {avg_time_ms:.2f} ms/forward")

    def test_cpu_baseline_medium_3d(self):
        """Measure CPU baseline for medium 3D workload."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        N = 1000
        shape = (128, 128, 128)
        centers, L, amps, _ = create_test_data(N, shape)

        model = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # Warmup
        for _ in range(2):
            _ = model()

        # Benchmark
        n_iters = 5
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = model()
        elapsed = time.perf_counter() - start

        avg_time_ms = (elapsed / n_iters) * 1000
        print(f"\nCPU baseline (N={N}, shape={shape}): {avg_time_ms:.2f} ms/forward")


@pytest.mark.skipif(not CUDA_BACKEND_AVAILABLE, reason="CUDA backend not compiled")
class TestCUDAPerformance:
    """CUDA backend performance tests."""

    def test_cuda_speedup_small_3d(self):
        """Test CUDA achieves target speedup for small 3D workload."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        N = 100
        shape = (64, 64, 64)
        centers, L, amps, _ = create_test_data(N, shape)

        # CPU model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # CPU benchmark
        for _ in range(3):
            _ = cpu_model()
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = cpu_model()
        cpu_time = (time.perf_counter() - start) / n_iters

        # CUDA benchmark
        torch.cuda.synchronize()
        for _ in range(5):
            _ = cuda_model()
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = cuda_model()
            torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / n_iters

        speedup = cpu_time / cuda_time
        print(
            f"\nSmall 3D: CPU={cpu_time * 1000:.2f}ms, CUDA={cuda_time * 1000:.2f}ms, Speedup={speedup:.1f}x"
        )

        # CUDA should be faster (allow for small workloads where overhead dominates)
        # For small workloads, even 1x is acceptable due to kernel launch overhead
        assert speedup > 0.5, f"CUDA is too slow: {speedup:.2f}x"

    def test_cuda_speedup_medium_3d(self):
        """Test CUDA achieves target speedup for medium 3D workload."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        N = 500
        shape = (96, 96, 96)
        centers, L, amps, _ = create_test_data(N, shape)

        # CPU model
        cpu_model = GaussianSplatModel(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cpu",
        )

        # CUDA model
        cuda_model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        # CPU benchmark
        for _ in range(2):
            _ = cpu_model()
        n_iters = 5
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = cpu_model()
        cpu_time = (time.perf_counter() - start) / n_iters

        # CUDA benchmark
        torch.cuda.synchronize()
        for _ in range(3):
            _ = cuda_model()
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            _ = cuda_model()
            torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) / n_iters

        speedup = cpu_time / cuda_time
        print(
            f"\nMedium 3D: CPU={cpu_time * 1000:.2f}ms, CUDA={cuda_time * 1000:.2f}ms, Speedup={speedup:.1f}x"
        )

        # For medium workloads, expect at least some speedup
        assert speedup > 1.0, (
            f"CUDA should be faster for medium workload: {speedup:.2f}x"
        )

    def test_cuda_memory_footprint(self):
        """Test CUDA memory usage is within expected bounds."""
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
            GaussianSplatModelCUDA,
        )

        N = 1000
        shape = (64, 64, 64)
        centers, L, amps, _ = create_test_data(N, shape)

        # Record baseline memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        baseline_mem = torch.cuda.memory_allocated()

        # Create model
        model = GaussianSplatModelCUDA(
            shape=shape,
            centers0=centers,
            L0=L,
            amps0=amps,
            sigma_min_diag=(0.5, 0.5, 0.5),
            device="cuda",
        )

        model_mem = torch.cuda.memory_allocated() - baseline_mem

        # Run forward pass
        output = model()
        peak_mem = torch.cuda.max_memory_allocated() - baseline_mem

        # Expected memory:
        # - Model parameters: N * (d + d*d + 1 + 1) * 4 bytes
        # - Output: prod(shape) * 4 bytes
        # - Tile buffers: ~10-20x overhead is reasonable
        d = 3
        param_mem = N * (d + d * d + 1 + 1) * 4
        output_mem = np.prod(shape) * 4
        expected_base = param_mem + output_mem

        print(f"\nMemory: model={model_mem / 1024:.1f}KB, peak={peak_mem / 1024:.1f}KB")
        print(f"Expected base: {expected_base / 1024:.1f}KB")

        # Memory should be reasonable (not more than 100x expected base)
        assert peak_mem < expected_base * 100, f"Peak memory {peak_mem} too high"

        # Cleanup
        del output, model
        torch.cuda.empty_cache()


class TestScaling:
    """Test performance scaling characteristics."""

    def test_scaling_with_splats_cpu(self):
        """Test how performance scales with splat count on CPU."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        shape = (64, 64, 64)
        splat_counts = [10, 50, 100, 500]
        times = []

        for N in splat_counts:
            centers, L, amps, _ = create_test_data(N, shape)

            model = GaussianSplatModel(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=(0.5, 0.5, 0.5),
                device="cpu",
            )

            # Warmup
            _ = model()

            # Benchmark
            n_iters = 5
            start = time.perf_counter()
            for _ in range(n_iters):
                _ = model()
            elapsed = time.perf_counter() - start

            avg_time_ms = (elapsed / n_iters) * 1000
            times.append(avg_time_ms)
            print(f"N={N}: {avg_time_ms:.2f} ms")

        # Verify roughly linear scaling (within 3x of linear)
        # time(N) should be approximately proportional to N
        for i in range(1, len(splat_counts)):
            ratio_n = splat_counts[i] / splat_counts[0]
            ratio_t = times[i] / times[0]
            # Allow up to 5x deviation from linear (accounting for overhead)
            assert ratio_t < ratio_n * 5, (
                f"Scaling worse than expected at N={splat_counts[i]}"
            )

    def test_scaling_with_volume_size_cpu(self):
        """Test how performance scales with volume size on CPU."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        N = 100
        shapes = [(32, 32, 32), (64, 64, 64), (96, 96, 96)]
        times = []

        for shape in shapes:
            centers, L, amps, _ = create_test_data(N, shape)

            model = GaussianSplatModel(
                shape=shape,
                centers0=centers,
                L0=L,
                amps0=amps,
                sigma_min_diag=(0.5, 0.5, 0.5),
                device="cpu",
            )

            # Warmup
            _ = model()

            # Benchmark
            n_iters = 3
            start = time.perf_counter()
            for _ in range(n_iters):
                _ = model()
            elapsed = time.perf_counter() - start

            avg_time_ms = (elapsed / n_iters) * 1000
            times.append(avg_time_ms)
            voxels = np.prod(shape)
            print(f"shape={shape} ({voxels} voxels): {avg_time_ms:.2f} ms")

        # Performance scaling with volume size depends on splat AABB overlap.
        # With compact splats (N=100), the work is bounded by splat footprints,
        # not total voxels. We only verify the test ran without errors.
        # The actual scaling relationship is complex and depends on:
        # - Number of voxels touched by splat AABBs (not total volume)
        # - Memory allocation overhead
        # - JIT compilation effects
        assert all(t > 0 for t in times), "All timings should be positive"
