"""
Pytest fixtures and configuration for CUDA backend tests.

This module provides reusable test fixtures for testing the CUDA splatting backend.

Usage:
    pytest tests/                           # Run all tests
    pytest tests/ -m "not slow"             # Skip slow tests
    pytest tests/ -m "not gpu"              # Skip tests requiring GPU
    pytest tests/ --cov=cuda_splatting      # With coverage
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

# =============================================================================
# MARKERS CONFIGURATION
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "numerical: marks numerical accuracy tests")


# =============================================================================
# GPU AVAILABILITY FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def cuda_device(cuda_available):
    """Return CUDA device for tests, skip if not available."""
    if not cuda_available:
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")


@pytest.fixture(scope="session")
def cuda_backend_available():
    """Check if CUDA backend is compiled and available."""
    try:
        import cuda_splatting_backend  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def require_cuda_backend(cuda_backend_available):
    """Skip test if CUDA backend is not compiled."""
    if not cuda_backend_available:
        pytest.skip("CUDA backend not compiled")


# =============================================================================
# RANDOM SEED FIXTURE
# =============================================================================


def seed_all(seed: int) -> None:
    """Seed numpy, torch CPU, and torch CUDA RNGs together.

    Tests that need explicit per-test determinism should call this instead
    of `np.random.seed(...)` alone — `torch.rand(...)` calls draw from
    torch's RNG, which numpy seeding does not touch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    seed_all(42)
    yield


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================


@pytest.fixture
def splat_params_2d() -> dict[str, Any]:
    """Generate random 2D splat parameters."""
    N = 100
    DIM = 2
    shape_size = 64

    centers = np.random.rand(N, DIM).astype(np.float32) * (shape_size - 2) + 1
    L = np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2.0
    amps = np.abs(np.random.randn(N).astype(np.float32)) + 0.5

    return {
        "centers": centers,
        "L": L,
        "amps": amps,
        "shape": (shape_size, shape_size),
        "sigma_min_diag": [0.5] * DIM,
        "dim": DIM,
    }


@pytest.fixture
def splat_params_3d() -> dict[str, Any]:
    """Generate random 3D splat parameters."""
    N = 100
    DIM = 3
    shape_size = 64

    centers = np.random.rand(N, DIM).astype(np.float32) * (shape_size - 2) + 1
    L = np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2.0
    amps = np.abs(np.random.randn(N).astype(np.float32)) + 0.5

    return {
        "centers": centers,
        "L": L,
        "amps": amps,
        "shape": (shape_size, shape_size, shape_size),
        "sigma_min_diag": [0.5] * DIM,
        "dim": DIM,
    }


@pytest.fixture(params=[2, 3, 4])
def splat_params_nd(request) -> dict[str, Any]:
    """Parameterized fixture for testing multiple dimensions."""
    DIM = request.param
    N = 50
    shape_size = max(8, 32 // DIM)  # Smaller shapes for higher dimensions

    centers = np.random.rand(N, DIM).astype(np.float32) * (shape_size - 2) + 1
    L = np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 1.5
    amps = np.abs(np.random.randn(N).astype(np.float32)) + 0.5

    return {
        "centers": centers,
        "L": L,
        "amps": amps,
        "shape": tuple([shape_size] * DIM),
        "sigma_min_diag": [0.5] * DIM,
        "dim": DIM,
    }


@pytest.fixture
def small_splat_params_3d() -> dict[str, Any]:
    """Small 3D test case for quick debugging."""
    N = 5
    DIM = 3
    shape_size = 16

    centers = np.array(
        [
            [8.0, 8.0, 8.0],  # Center
            [4.0, 4.0, 4.0],  # Corner region
            [12.0, 12.0, 12.0],  # Opposite corner
            [4.0, 8.0, 12.0],  # Mixed
            [8.0, 12.0, 4.0],  # Mixed
        ],
        dtype=np.float32,
    )

    L = np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 1.5
    amps = np.ones(N, dtype=np.float32)

    return {
        "centers": centers,
        "L": L,
        "amps": amps,
        "shape": (shape_size, shape_size, shape_size),
        "sigma_min_diag": [0.5] * DIM,
        "dim": DIM,
    }


# =============================================================================
# MODEL FACTORY FIXTURES
# =============================================================================


@pytest.fixture
def reference_model_factory(cuda_device):
    """
    Factory for creating PyTorch reference models.

    Usage:
        model = reference_model_factory(params)
    """
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    def factory(params: dict[str, Any], device=None):
        return GaussianSplatModel(
            shape=params["shape"],
            centers0=params["centers"],
            L0=params["L"],
            amps0=params["amps"],
            sigma_min_diag=params["sigma_min_diag"],
            device=device or cuda_device,
        )

    return factory


@pytest.fixture
def cuda_model_factory(cuda_device, require_cuda_backend):
    """
    Factory for creating CUDA-accelerated models.

    Usage:
        model = cuda_model_factory(params)
    """
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import (
        GaussianSplatModelCUDA,
    )

    def factory(params: dict[str, Any]):
        return GaussianSplatModelCUDA(
            shape=params["shape"],
            centers0=params["centers"],
            L0=params["L"],
            amps0=params["amps"],
            sigma_min_diag=params["sigma_min_diag"],
            device=cuda_device,
        )

    return factory


# =============================================================================
# TOLERANCE CONSTANTS
# =============================================================================


class Tolerances:
    """Tolerance levels for numerical comparisons.

    All tolerance values are centralized here. Tests should reference these
    constants rather than hardcoding values, so that tolerances can be
    tightened incrementally as the implementation improves.
    """

    # Forward pass element-wise comparison
    FORWARD_RTOL = 1e-4
    FORWARD_ATOL = 1e-6

    # Forward pass CUDA vs PyTorch aggregate comparison
    COMPARISON_MAX_REL_DIFF = 0.15  # Max relative difference
    COMPARISON_MEAN_REL_DIFF = 0.01  # Mean relative difference
    COMPARISON_MIN_CORRELATION = 0.99  # Minimum correlation coefficient
    COMPARISON_BOUNDARY_REL_DIFF = 0.30  # Boundary/cross-path comparisons (3D vs 4D)

    # Backward pass comparison (looser due to atomic accumulation)
    BACKWARD_RTOL = 1e-3
    BACKWARD_ATOL = 1e-5
    BACKWARD_SIGN_MATCH = 0.85  # Minimum fraction of gradient signs matching
    BACKWARD_MAG_RATIO = 10  # Maximum gradient magnitude ratio

    # FP16 comparison
    FP16_RTOL = 1e-2
    FP16_ATOL = 1e-4

    # Convergence comparison (for fitting tests)
    CONVERGENCE_RTOL = 0.1  # 10% relative difference in final loss


@pytest.fixture
def tolerances():
    """Return tolerance constants."""
    return Tolerances()


# =============================================================================
# UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def random_target_3d(splat_params_3d, cuda_device):
    """Generate a random target volume for 3D tests."""
    shape = splat_params_3d["shape"]
    return torch.rand(shape, device=cuda_device, dtype=torch.float32)


@pytest.fixture
def random_target_2d(splat_params_2d, cuda_device):
    """Generate a random target volume for 2D tests."""
    shape = splat_params_2d["shape"]
    return torch.rand(shape, device=cuda_device, dtype=torch.float32)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def params_to_tensors(params: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Convert numpy parameter arrays to torch tensors on device."""
    return {
        "centers": torch.tensor(params["centers"], device=device, dtype=torch.float32),
        "L": torch.tensor(params["L"], device=device, dtype=torch.float32),
        "amps": torch.tensor(params["amps"], device=device, dtype=torch.float32),
        "shape": params["shape"],
    }


@pytest.fixture
def params_to_tensors_fn(cuda_device):
    """Fixture that returns a helper function for converting params to tensors."""

    def convert(params: dict[str, Any]) -> dict[str, Any]:
        return params_to_tensors(params, cuda_device)

    return convert
