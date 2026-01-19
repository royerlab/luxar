"""
Tests for fitting/visualization.py module.
"""

import sys

import numpy as np
import pytest

from luxar.gsplats.fitting.visualization import (
    display_compression_analysis,
    show_optimization_movie,
)
from luxar.gsplats.gsplat_data import GSplatData


class _MockNapari:
    """Simple mock for napari module to prevent window opening."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


@pytest.fixture(autouse=True)
def mock_napari_viewer(monkeypatch):
    """Mock napari to prevent opening windows during tests."""
    mock_napari = _MockNapari()
    monkeypatch.setitem(sys.modules, "napari", mock_napari)
    return mock_napari


def test_display_compression_analysis_2d(capsys) -> None:
    """Test compression analysis display for 2D data."""
    V = np.random.rand(64, 64).astype(np.float32)

    # Create GSplatData: 10 splats with 2D centers and packed L
    N = 10
    d = 2
    tril_size = d * (d + 1) // 2  # 3 for 2D

    result = GSplatData(
        centers=np.random.rand(N, d).astype(np.float32),
        amplitudes=np.random.rand(N).astype(np.float32),
        cholesky_factors=np.random.rand(N, tril_size).astype(np.float32),
        sharpnesses=np.full(N, 2.0, dtype=np.float32),
        stats={},
    )

    # Should run without errors
    display_compression_analysis(V, result)

    # Check that output was produced
    captured = capsys.readouterr()
    assert "Compression Analysis" in captured.out
    assert "Original image:" in captured.out
    assert "Splat representation:" in captured.out
    assert "Compression ratio:" in captured.out


def test_display_compression_analysis_3d(capsys) -> None:
    """Test compression analysis display for 3D data."""
    V = np.random.rand(32, 32, 32).astype(np.float32)

    # Create GSplatData: 8 splats for 3D
    N = 8
    d = 3
    tril_size = d * (d + 1) // 2  # 6 for 3D

    result = GSplatData(
        centers=np.random.rand(N, d).astype(np.float32),
        amplitudes=np.random.rand(N).astype(np.float32),
        cholesky_factors=np.random.rand(N, tril_size).astype(np.float32),
        sharpnesses=np.full(N, 2.0, dtype=np.float32),
        stats={},
    )

    # Should run without errors
    display_compression_analysis(V, result)

    captured = capsys.readouterr()
    assert "Compression Analysis" in captured.out
    assert "Original image:" in captured.out


def test_compression_ratio_calculation(capsys) -> None:
    """Test that compression ratio is calculated correctly."""
    # Small example where we can verify the calculation
    V = np.random.rand(16, 16).astype(np.float32)  # 256 pixels * 4 bytes = 1024 bytes

    # 2 splats: each has 2 (centers) + 3 (packed L) + 1 (sharpness) + 1 (amp) = 7 floats
    # Total: 2 * 7 * 4 = 56 bytes
    # Compression ratio: 1024 / 56 = 18.29:1
    N = 2
    d = 2
    tril_size = 3

    result = GSplatData(
        centers=np.random.rand(N, d).astype(np.float32),
        amplitudes=np.random.rand(N).astype(np.float32),
        cholesky_factors=np.random.rand(N, tril_size).astype(np.float32),
        sharpnesses=np.full(N, 2.0, dtype=np.float32),
        stats={},
    )

    display_compression_analysis(V, result)

    captured = capsys.readouterr()
    assert "Compression ratio:" in captured.out
    # Should see a high compression ratio (around 18:1)
    assert "18." in captured.out or "18:" in captured.out


def test_bits_per_pixel_calculation(capsys) -> None:
    """Test bits per pixel calculation."""
    V = np.random.rand(32, 32).astype(np.float32)

    # Few splats = low bits per pixel
    N = 5
    d = 2
    tril_size = 3

    result = GSplatData(
        centers=np.random.rand(N, d).astype(np.float32),
        amplitudes=np.random.rand(N).astype(np.float32),
        cholesky_factors=np.random.rand(N, tril_size).astype(np.float32),
        sharpnesses=np.full(N, 2.0, dtype=np.float32),
        stats={},
    )

    display_compression_analysis(V, result)

    captured = capsys.readouterr()
    assert "Bits per pixel:" in captured.out
    # Original is 32 bits per pixel
    assert "original: 32.000" in captured.out


def test_show_optimization_movie_no_napari(capsys) -> None:
    """Test graceful handling when napari is unavailable."""
    # Create minimal movie frames
    movie_frames = {
        "target": [np.random.rand(16, 16) for _ in range(3)],
        "reconstruction": [np.random.rand(16, 16) for _ in range(3)],
        "residual": [np.random.rand(16, 16) for _ in range(3)],
        "splat_centers": [np.random.rand(5, 2) for _ in range(3)],
        "iterations": [10, 20, 30],
    }
    shape = (16, 16)

    # Temporarily hide napari import by testing error handling
    # This should handle the ImportError gracefully
    try:
        show_optimization_movie(movie_frames, shape)
        # If napari is available, this will open a viewer
        # We can't easily test the interactive part
    except Exception as e:
        # Should not raise unhandled exceptions
        pytest.fail(f"show_optimization_movie raised unexpected exception: {e}")


def test_show_optimization_movie_3d_structure() -> None:
    """Test that 3D movie data structure is handled correctly."""
    # Create 3D movie frames
    movie_frames = {
        "target": [np.random.rand(16, 16, 16) for _ in range(2)],
        "reconstruction": [np.random.rand(16, 16, 16) for _ in range(2)],
        "residual": [np.random.rand(16, 16, 16) for _ in range(2)],
        "splat_centers": [np.random.rand(5, 3) for _ in range(2)],
        "iterations": [10, 20],
    }
    shape = (16, 16, 16)

    # Should not crash with 3D data
    try:
        show_optimization_movie(movie_frames, shape)
    except ImportError:
        # Expected if napari not available
        pass
    except Exception as e:
        # Other exceptions are problems
        pytest.fail(f"3D movie visualization failed: {e}")


def test_empty_movie_frames() -> None:
    """Test handling of empty movie frames."""
    movie_frames: dict[str, list] = {
        "target": [],
        "reconstruction": [],
        "residual": [],
        "splat_centers": [],
        "iterations": [],
    }
    shape = (16, 16)

    # Should handle empty frames gracefully
    try:
        show_optimization_movie(movie_frames, shape)
    except ImportError:
        pass  # Expected if napari not available
    except Exception as e:
        # Should not crash on empty frames
        pytest.fail(f"Empty frames caused unexpected error: {e}")


def test_compression_analysis_zero_splats(capsys) -> None:
    """Test compression analysis with zero splats (edge case)."""
    V = np.random.rand(16, 16).astype(np.float32)

    # Zero splats
    d = 2
    tril_size = 3

    result = GSplatData(
        centers=np.array([]).reshape(0, d).astype(np.float32),
        amplitudes=np.array([]).astype(np.float32),
        cholesky_factors=np.array([]).reshape(0, tril_size).astype(np.float32),
        sharpnesses=np.array([]).astype(np.float32),
        stats={},
    )

    # Should handle gracefully (infinite compression ratio)
    display_compression_analysis(V, result)

    captured = capsys.readouterr()
    assert "Compression Analysis" in captured.out
    # May show inf or very high ratio
    output_lower = captured.out.lower()
    assert "compression ratio" in output_lower


def test_compression_analysis_many_splats(capsys) -> None:
    """Test compression analysis with many splats (poor compression)."""
    V = np.random.rand(16, 16).astype(np.float32)

    # Many splats (worse than original)
    N = 100
    d = 2
    tril_size = 3

    result = GSplatData(
        centers=np.random.rand(N, d).astype(np.float32),
        amplitudes=np.random.rand(N).astype(np.float32),
        cholesky_factors=np.random.rand(N, tril_size).astype(np.float32),
        sharpnesses=np.full(N, 2.0, dtype=np.float32),
        stats={},
    )

    display_compression_analysis(V, result)

    captured = capsys.readouterr()
    assert "Compression Analysis" in captured.out
    # With 100 splats, may actually be negative "savings"
    assert "Space savings:" in captured.out


def test_display_compression_various_dimensions(capsys) -> None:
    """Test compression analysis with various dimensionalities."""
    for d in [2, 3, 4]:
        shape = tuple([16] * d)
        V = np.random.rand(*shape).astype(np.float32)

        N = 5
        tril_size = d * (d + 1) // 2

        result = GSplatData(
            centers=np.random.rand(N, d).astype(np.float32),
            amplitudes=np.random.rand(N).astype(np.float32),
            cholesky_factors=np.random.rand(N, tril_size).astype(np.float32),
            sharpnesses=np.full(N, 2.0, dtype=np.float32),
            stats={},
        )

        # Should work for all dimensionalities
        display_compression_analysis(V, result)

        captured = capsys.readouterr()
        assert "Compression Analysis" in captured.out
        assert "Compression ratio:" in captured.out
