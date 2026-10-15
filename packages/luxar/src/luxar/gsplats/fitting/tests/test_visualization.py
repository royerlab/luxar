"""
Tests for fitting/visualization.py module.
"""

import sys

import numpy as np
import pytest

from luxar.gsplats.fitting.visualization import (
    compression_lines,
    compression_ratio,
    display_compression_analysis,
    format_compression_ratio,
    show_optimization_movie,
)


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


def test_compression_ratio_is_source_bytes_over_stored_bytes() -> None:
    """One definition: decoded source bytes / bytes stored on disk."""
    assert compression_ratio(1000, 10) == pytest.approx(100.0)
    assert compression_ratio(10, 40) == pytest.approx(0.25)
    # Unknown or non-positive on either side -> no ratio, never a fake one.
    for src, stored in ((None, 10), (1000, None), (0, 10), (1000, 0), (-1, 5)):
        assert compression_ratio(src, stored) is None


def test_format_compression_ratio() -> None:
    assert format_compression_ratio(1234.4) == "1,234:1"
    assert format_compression_ratio(12.6) == "13:1"
    # A store bigger than its source must not round to a nonsensical "0:1".
    assert format_compression_ratio(0.25) == "0.25:1"


def test_compression_lines_name_both_bases() -> None:
    lines = compression_lines(2_000_000, 20_000, source_stored_bytes=1_000_000)
    assert len(lines) == 2
    assert lines[0].startswith("100:1 vs raw voxels")
    assert lines[1].startswith("50:1 vs the stored source")
    assert compression_lines(2_000_000, 20_000) == lines[:1]
    assert compression_lines(None, 20_000) == []


def test_display_compression_analysis_reports_the_shared_ratio(capsys) -> None:
    ratio = display_compression_analysis(
        2_000_000, 20_000, n_splats=123, source_voxels=1_000_000
    )
    assert ratio == pytest.approx(100.0)
    out = capsys.readouterr().out
    assert "Compression Analysis" in out
    assert "Compression ratio: 100:1 vs raw voxels" in out
    # 20,000 B * 8 bits / 1e6 voxels = 0.16; source 16 bits/voxel (uint16).
    assert "Bits per voxel: 0.160 (source: 16.000)" in out
    assert "Splats stored: 123" in out


def test_display_compression_analysis_unknown_source(capsys) -> None:
    assert display_compression_analysis(None, 20_000, n_splats=5) is None
    out = capsys.readouterr().out
    assert "unknown (source size not recorded)" in out


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
