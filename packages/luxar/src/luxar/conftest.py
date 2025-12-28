"""
Shared pytest fixtures for Luxar test suite.

This module provides reusable fixtures for testing across all modules.
Pytest automatically discovers and uses fixtures from conftest.py files.

Usage:
    Fixtures are automatically available in all test files under this directory.
    No imports needed - just use the fixture name as a parameter.

    Example:
        def test_something(cli_runner, sample_scene):
            result = cli_runner.invoke(app, ["info", str(sample_scene)])
            assert result.exit_code == 0
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# =============================================================================
# Optional Dependencies Detection
# =============================================================================

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass

HAS_SCIPY = False
try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    pass

HAS_CUDA = HAS_TORCH and torch.cuda.is_available()
HAS_MPS = (
    HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)

# =============================================================================
# Skip Markers - Use these to conditionally skip tests
# =============================================================================

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
requires_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not installed")
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
requires_mps = pytest.mark.skipif(not HAS_MPS, reason="MPS not available")
requires_gpu = pytest.mark.skipif(
    not (HAS_CUDA or HAS_MPS), reason="No GPU (CUDA or MPS) available"
)


# =============================================================================
# CLI Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Typer CLI test runner.

    Returns a CliRunner instance for testing CLI commands.

    Example:
        def test_cli_command(cli_runner):
            from luxar.cli import app
            result = cli_runner.invoke(app, ["info", "data.zarr"])
            assert result.exit_code == 0
    """
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture
def sample_scene(tmp_path: Path) -> Path:
    """Create a minimal Lorenz attractor scene for testing.

    Returns path to a .zarr store with 100 points.
    Uses seed=42 for reproducibility.

    Example:
        def test_scene_loading(sample_scene):
            import zarr
            store = zarr.open_group(sample_scene, mode="r")
            assert "scene" in store.attrs
    """
    from luxar.demos import create_lorenz_attractor

    store_path = tmp_path / "test_scene.zarr"
    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


@pytest.fixture
def sample_scene_large(tmp_path: Path) -> Path:
    """Create a larger Lorenz attractor scene (1000 points).

    Use for tests that need more data points.
    """
    from luxar.demos import create_lorenz_attractor

    store_path = tmp_path / "test_scene_large.zarr"
    create_lorenz_attractor(store_path, n_points=1000, seed=42)
    return store_path


# =============================================================================
# Gaussian Blob Fixtures (for GSplats testing)
# =============================================================================


@pytest.fixture
def gaussian_2d_21x21() -> np.ndarray:
    """21x21 2D Gaussian blob centered at origin.

    Standard deviation ~1.0, small background noise added.
    Returns float32 array.

    Example:
        def test_blob_fitting(gaussian_2d_21x21):
            assert gaussian_2d_21x21.shape == (21, 21)
            assert gaussian_2d_21x21.max() > 0.9  # Peak near 1.0
    """
    x, y = np.meshgrid(np.linspace(-3, 3, 21), np.linspace(-3, 3, 21))
    blob = np.exp(-(x**2 + y**2) / 2) + 0.1
    return blob.astype(np.float32)


@pytest.fixture
def gaussian_3d_15x15x15() -> np.ndarray:
    """15x15x15 3D Gaussian blob centered at origin.

    Standard deviation ~1.0, small background noise added.
    Returns float32 array.
    """
    x, y, z = np.meshgrid(
        np.linspace(-2, 2, 15),
        np.linspace(-2, 2, 15),
        np.linspace(-2, 2, 15),
    )
    blob = np.exp(-(x**2 + y**2 + z**2) / 2) + 0.05
    return blob.astype(np.float32)


@pytest.fixture
def multi_blob_2d_31x31() -> np.ndarray:
    """31x31 image with three Gaussian blobs at different locations.

    Blob 1: Center (-2, -2), amplitude 0.8
    Blob 2: Center (2, 2), amplitude 0.6
    Blob 3: Center (0, 3), amplitude 1.0
    """
    x, y = np.meshgrid(np.linspace(-5, 5, 31), np.linspace(-5, 5, 31))
    blob1 = 0.8 * np.exp(-((x + 2) ** 2 + (y + 2) ** 2) / 1.5)
    blob2 = 0.6 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 2.0)
    blob3 = 1.0 * np.exp(-(x**2 + (y - 3) ** 2) / 1.0)
    return (blob1 + blob2 + blob3 + 0.05).astype(np.float32)


# =============================================================================
# Point Cloud Fixtures
# =============================================================================


@pytest.fixture
def random_points_100() -> dict[str, Any]:
    """100 random 3D points with colors and radii.

    Uses seed=42 for reproducibility.

    Returns dict with:
        - positions: (100, 3) float32
        - colors: (100, 3) uint8
        - radii: (100,) float32

    Example:
        def test_point_rendering(random_points_100):
            positions = random_points_100["positions"]
            assert positions.shape == (100, 3)
    """
    np.random.seed(42)
    return {
        "positions": np.random.randn(100, 3).astype(np.float32),
        "colors": np.random.randint(0, 255, (100, 3), dtype=np.uint8),
        "radii": np.abs(np.random.randn(100)).astype(np.float32) * 0.1,
    }


@pytest.fixture
def random_points_1000() -> dict[str, Any]:
    """1000 random 3D points with colors and radii.

    Same structure as random_points_100 but larger.
    """
    np.random.seed(42)
    return {
        "positions": np.random.randn(1000, 3).astype(np.float32),
        "colors": np.random.randint(0, 255, (1000, 3), dtype=np.uint8),
        "radii": np.abs(np.random.randn(1000)).astype(np.float32) * 0.1,
    }


# =============================================================================
# Transform Fixtures
# =============================================================================


@pytest.fixture
def identity_matrix() -> np.ndarray:
    """4x4 identity matrix as float32."""
    return np.eye(4, dtype=np.float32)


@pytest.fixture
def translation_matrix() -> np.ndarray:
    """4x4 translation matrix (translate by [1, 2, 3])."""
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 3] = [1.0, 2.0, 3.0]
    return mat


@pytest.fixture
def scale_matrix() -> np.ndarray:
    """4x4 scale matrix (scale by [2, 2, 2])."""
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = 2.0
    mat[1, 1] = 2.0
    mat[2, 2] = 2.0
    return mat


# =============================================================================
# Dimension Fixtures
# =============================================================================


@pytest.fixture
def default_3d_dimensions():
    """Default 3D dimensions (x, y, z in micrometers).

    Example:
        def test_scene_creation(default_3d_dimensions):
            from luxar import LuxarZarrCompiler
            with LuxarZarrCompiler(path) as compiler:
                compiler.create_scene(dimensions=default_3d_dimensions)
    """
    from luxar import Dimensions

    return Dimensions.default_3d()


@pytest.fixture
def dimensions_4d():
    """4D dimensions (x, y, z spatial + time).

    x, y, z: display=True, unit="um"
    t: display=False, discrete=True
    """
    from luxar import Dimension, Dimensions

    return Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
            Dimension("t", unit="s", display=False, discrete=True, range=(0, 10)),
        ]
    )


@pytest.fixture
def dimensions_5d():
    """5D dimensions (x, y, z, channel, time).

    x, y, z: display=True
    channel: display=False, discrete=True
    time: display=False, discrete=True
    """
    from luxar import Dimension, Dimensions

    return Dimensions(
        [
            Dimension("x", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("z", unit="um", display=True),
            Dimension("channel", display=False, discrete=True, range=(0, 3)),
            Dimension("time", unit="s", display=False, discrete=True, range=(0, 100)),
        ]
    )


# =============================================================================
# Reproducibility Fixtures
# =============================================================================


@pytest.fixture(autouse=False)
def seed_random():
    """Set random seed for reproducibility.

    Use as fixture parameter to get reproducible random numbers:

        def test_random_stuff(seed_random):
            # numpy.random is now seeded to 42
            values = np.random.randn(100)
    """
    np.random.seed(42)
    if HAS_TORCH:
        torch.manual_seed(42)
        if HAS_CUDA:
            torch.cuda.manual_seed_all(42)
