"""
Validation tests for demo scripts.

These tests verify that demo scripts:
1. Execute without errors
2. Produce valid zarr output
3. Handle parameters correctly
4. Generate expected data structures
5. Follow naming conventions
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

# Find all demo scripts
DEMOS_DIR = Path(__file__).parent.parent
DEMO_SCRIPTS = list(DEMOS_DIR.glob("demo_*.py"))


@pytest.fixture
def demo_output_dir(tmp_path):
    """Create a temporary directory for demo outputs."""
    output_dir = tmp_path / "demo_outputs"
    output_dir.mkdir()
    return output_dir


class TestDemoUtilities:
    """Test demo utility functions (not CLI scripts)."""

    def test_lorenz_attractor_generation(self, demo_output_dir):
        """Test Lorenz attractor demo utility function."""
        from luxar.utils.demos import create_lorenz_attractor

        output_path = demo_output_dir / "lorenz_test.zarr"
        create_lorenz_attractor(output_path, n_points=100, seed=42)

        assert output_path.exists()

        store = zarr.open(str(output_path), mode="r")
        attrs = dict(store.attrs)
        assert "luxar_version" in attrs or "version" in attrs

        # Verify has Lorenz group with positions
        assert "Lorenz" in store or "LorenzAttractor" in store


class TestDemoOutputValidation:
    """Test that demo utility functions produce valid zarr stores."""

    def test_lorenz_produces_valid_zarr(self, demo_output_dir):
        """Test that Lorenz demo creates valid zarr with expected structure."""
        from luxar.utils.demos import create_lorenz_attractor

        output_path = demo_output_dir / "lorenz_output.zarr"
        create_lorenz_attractor(output_path, n_points=100, seed=42)

        # Validate output
        assert output_path.exists()

        # Open and validate zarr structure
        store = zarr.open(str(output_path), mode="r")

        # Check for required metadata
        attrs = dict(store.attrs)
        assert "luxar_version" in attrs or "version" in attrs, "Missing version info"

        # Verify has at least one group with points
        groups = list(store.group_keys())
        assert len(groups) > 0, "Empty zarr store"

        # Find and validate positions
        has_positions = False
        for group_name in groups:
            group = store[group_name]
            if "positions" in group:
                has_positions = True
                positions = group["positions"]
                assert positions.shape[1] >= 3, "Positions must be at least 3D"
                assert positions.shape[0] == 100, f"Expected 100 points, got {positions.shape[0]}"
                break

        assert has_positions, "No positions array found in zarr store"


class TestDemoParameterHandling:
    """Test that demos handle parameters correctly."""

    def test_demo_with_seed_reproducibility(self):
        """Test that same seed produces same output."""
        # Use the demo utility function directly for reproducibility testing
        from luxar.utils.demos import create_lorenz_attractor
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output1 = Path(tmpdir) / "seed_test1.zarr"
            output2 = Path(tmpdir) / "seed_test2.zarr"

            # Generate twice with same seed
            create_lorenz_attractor(output1, n_points=50, seed=123)
            create_lorenz_attractor(output2, n_points=50, seed=123)

            # Compare outputs
            store1 = zarr.open(str(output1), mode="r")
            store2 = zarr.open(str(output2), mode="r")

            # Find first positions array
            def find_positions(store):
                for key in store.group_keys():
                    if "positions" in store[key]:
                        return np.array(store[key]["positions"])
                return None

            pos1 = find_positions(store1)
            pos2 = find_positions(store2)

            assert pos1 is not None and pos2 is not None
            assert np.allclose(pos1, pos2), "Same seed should produce identical positions"

    def test_demo_point_count_parameter(self):
        """Test that demo generation respects point count."""
        # Use the demo utility function directly instead of CLI
        from luxar.utils.demos import create_lorenz_attractor
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.zarr"

            # Test with specific point counts
            for n_points in [10, 50, 100]:
                create_lorenz_attractor(output_path, n_points=n_points, seed=42)

                store = zarr.open(str(output_path), mode="r")

                # Find positions and verify count
                for key in store.group_keys():
                    if "positions" in store[key]:
                        positions = store[key]["positions"]
                        actual_count = positions.shape[0]
                        assert actual_count == n_points, f"Expected {n_points} points, got {actual_count}"
                        break

                # Clean up for next iteration
                import shutil
                if output_path.exists():
                    shutil.rmtree(output_path)


class TestDemoNamingConventions:
    """Test that demos follow naming conventions."""

    def test_all_demos_follow_naming_pattern(self):
        """Verify all demo files follow demo_*.py pattern."""
        all_py_files = list(DEMOS_DIR.glob("*.py"))
        demo_files = [f for f in all_py_files if f.name.startswith("demo_")]

        # Filter out test files and __init__
        non_demo_files = [
            f for f in all_py_files if not f.name.startswith("demo_") and f.name != "__init__.py"
        ]

        # Should have no non-demo python files (except __init__.py)
        test_files = [f for f in non_demo_files if "test" not in f.name.lower()]
        assert len(test_files) == 0, f"Found non-demo files: {[f.name for f in test_files]}"

    def test_demo_scripts_have_docstrings(self):
        """Verify demo scripts have module docstrings."""
        for demo_script in DEMO_SCRIPTS[:5]:  # Check first 5
            with open(demo_script) as f:
                content = f.read()
                # Should have triple-quoted docstring near top
                assert '"""' in content or "'''" in content, f"{demo_script.name} missing docstring"


class TestDemoErrorHandling:
    """Test demo error handling and edge cases."""

    def test_demo_with_invalid_path(self):
        """Test demo utility handles invalid output path gracefully."""
        from luxar.utils.demos import create_lorenz_attractor

        # Try to write to invalid path
        with pytest.raises((OSError, PermissionError, ValueError, KeyError)):
            create_lorenz_attractor("/invalid/nonexistent/path/output.zarr", n_points=10)

    def test_demo_with_zero_points(self, demo_output_dir):
        """Test demo handles zero points edge case."""
        from luxar.utils.demos import create_lorenz_attractor

        output_path = demo_output_dir / "zero_points.zarr"

        # Should either work with empty dataset or raise clear error
        try:
            create_lorenz_attractor(output_path, n_points=0)
            # If it succeeds, verify it created valid (possibly empty) zarr
            assert output_path.exists()
        except ValueError as e:
            # Or it may raise ValueError for zero points
            assert "point" in str(e).lower()


class TestDemoOutputQuality:
    """Test quality of demo outputs."""

    def test_demo_generates_non_empty_points(self, demo_output_dir):
        """Test that demos actually generate non-degenerate points."""
        from luxar.utils.demos import create_lorenz_attractor

        output_path = demo_output_dir / "quality_test.zarr"
        create_lorenz_attractor(output_path, n_points=100, seed=42)

        store = zarr.open(str(output_path), mode="r")

        # Find and validate positions
        for key in store.group_keys():
            if "positions" in store[key]:
                positions = np.array(store[key]["positions"])

                # Verify positions are not all zeros/identical
                assert np.std(positions) > 0.001, "Positions are degenerate (all same)"

                # Verify positions are finite
                assert np.all(np.isfinite(positions)), "Positions contain NaN/Inf"

                # If colors exist, verify they're valid
                if "colors" in store[key]:
                    colors = np.array(store[key]["colors"])
                    assert colors.shape[0] == positions.shape[0], "Color/position count mismatch"
                    assert np.all(colors >= 0), "Colors contain negative values"

                break
