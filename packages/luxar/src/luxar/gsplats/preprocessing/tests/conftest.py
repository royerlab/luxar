"""Fixtures and constants for preprocessing tests."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch


def pytest_configure() -> None:
    """Fail instead of skipping when the cadence requires real CUDA coverage."""
    if os.environ.get("LUXAR_REQUIRE_CUDA") != "1":
        return

    from luxar.gsplats.preprocessing.cuda import NLM_CUDA_AVAILABLE

    if not torch.cuda.is_available():
        raise pytest.UsageError(
            "LUXAR_REQUIRE_CUDA=1 but PyTorch cannot access a CUDA device"
        )
    if not NLM_CUDA_AVAILABLE:
        raise pytest.UsageError(
            "LUXAR_REQUIRE_CUDA=1 but the NLM CUDA backend is unavailable"
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cuda_available():
    return torch.cuda.is_available()


@pytest.fixture
def require_cuda(cuda_available):
    if not cuda_available:
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------


class Tolerances:
    """Cross-backend comparison tolerances."""

    # PyTorch vs skimage: these are fundamentally different algorithms
    # (direct patch comparison vs integral images in fast_mode) with
    # different boundary handling.  Mean diff ~0.005 is typical; max
    # diff up to ~0.10 at borders is expected.
    PYTORCH_VS_SKIMAGE_ATOL = 0.10
    PYTORCH_VS_SKIMAGE_RTOL = 0.10

    # CUDA vs PyTorch (same algorithm, minor FP ordering)
    CUDA_VS_PYTORCH_ATOL = 1e-4

    # Minimum PSNR improvement for denoised vs noisy (dB)
    MIN_PSNR_IMPROVEMENT_DB = 2.0


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------


def _make_smooth_2d(H: int = 128, W: int = 128) -> torch.Tensor:
    """Create a smooth 2D image with Gaussian blobs."""
    y, x = torch.meshgrid(
        torch.linspace(-3, 3, H), torch.linspace(-3, 3, W), indexing="ij"
    )
    img = torch.exp(-(x**2 + y**2) / 2.0)
    img += 0.6 * torch.exp(-((x - 1.5) ** 2 + (y - 1.0) ** 2) / 0.8)
    img += 0.4 * torch.exp(-((x + 1.0) ** 2 + (y + 1.5) ** 2) / 0.5)
    # Normalise to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    return img


def _make_smooth_3d(D: int = 32, H: int = 32, W: int = 32) -> torch.Tensor:
    """Create a smooth 3D volume with Gaussian blobs."""
    z, y, x = torch.meshgrid(
        torch.linspace(-3, 3, D),
        torch.linspace(-3, 3, H),
        torch.linspace(-3, 3, W),
        indexing="ij",
    )
    vol = torch.exp(-(x**2 + y**2 + z**2) / 2.0)
    vol += 0.5 * torch.exp(-((x - 1) ** 2 + (y - 1) ** 2 + (z - 1) ** 2) / 0.8)
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    return vol


@pytest.fixture
def noisy_2d() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(clean, noisy)`` 2D image pair."""
    clean = _make_smooth_2d(128, 128)
    noisy = clean + torch.randn_like(clean) * 0.05
    return clean, noisy


@pytest.fixture
def noisy_3d() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(clean, noisy)`` 3D volume pair (small for fast tests)."""
    clean = _make_smooth_3d(32, 32, 32)
    noisy = clean + torch.randn_like(clean) * 0.05
    return clean, noisy


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def psnr(clean: torch.Tensor, estimate: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB (assumes data range [0, 1])."""
    mse = ((clean - estimate) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))
