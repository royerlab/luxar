"""
Tests for the luxar CLI commands.
"""

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar import LuxarZarrCompiler
from luxar.cli import app
from luxar.cli.main import _dfs


@pytest.fixture
def runner():
    """CLI test runner fixture."""
    return CliRunner()


@pytest.fixture
def sample_scene(tmp_path):
    """Create a sample scene for testing."""
    store_path = tmp_path / "test_scene.zarr"
    from luxar.demos import create_lorenz_attractor

    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


def test_demo_command_no_serve_success(runner, tmp_path):
    """Test successful demo generation without serving."""
    output_path = tmp_path / "demo_test.zarr"

    result = runner.invoke(
        app,
        [
            "demo",
            "--no-serve",
            "--output",
            str(output_path),
            "--points",
            "50",
            "--seed",
            "123",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Generated 50 points" in result.stdout

    # Verify the generated scene
    root = zarr.open_group(output_path, mode="r")
    assert "LorenzAttractor" in root
    assert root["LorenzAttractor"]["positions"].shape == (50, 3)
    assert root["LorenzAttractor"]["colors"].shape == (50, 3)


def test_demo_command_no_serve_with_defaults(runner, tmp_path):
    """Test demo command without serving with default parameters."""
    output_path = tmp_path / "default_demo.zarr"

    result = runner.invoke(app, ["demo", "--no-serve", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Generated 10,000 points" in result.stdout


def test_demo_command_no_serve_short_options(runner, tmp_path):
    """Test demo command without serving using short option flags."""
    output_path = tmp_path / "short_opts.zarr"

    result = runner.invoke(
        app, ["demo", "--no-serve", "-o", str(output_path), "-n", "25"]
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Generated 25 points" in result.stdout


def test_demo_command_no_serve_failure(runner, tmp_path):
    """Test demo command when scene creation fails."""
    # Use invalid path to trigger failure
    invalid_path = "/invalid/path/that/does/not/exist.zarr"

    result = runner.invoke(
        app, ["demo", "--no-serve", "--output", invalid_path, "--points", "10"]
    )

    assert result.exit_code == 1
    assert "Error:" in result.stdout


def test_info_command_success(runner, sample_scene):
    """Test successful info command on valid scene."""
    result = runner.invoke(app, ["info", str(sample_scene)])

    assert result.exit_code == 0
    # Check for new format with emojis
    assert "Root Attributes" in result.stdout or "🎯 Root Attributes" in result.stdout
    assert "Scene Hierarchy" in result.stdout or "🌳 Scene Hierarchy" in result.stdout
    assert (
        "Summary Statistics" in result.stdout
        or "📊 Summary Statistics" in result.stdout
    )
    assert (
        "Points objects: 1" in result.stdout or "⭕ Points objects: 1" in result.stdout
    )
    assert (
        "Total points: 100" in result.stdout or "✨ Total points: 100" in result.stdout
    )
    assert "luxar_version" in result.stdout
    assert "units" in result.stdout


def test_info_command_nonexistent_path(runner, tmp_path):
    """Test info command with non-existent path."""
    nonexistent_path = tmp_path / "does_not_exist.zarr"

    result = runner.invoke(app, ["info", str(nonexistent_path)])

    assert result.exit_code == 1
    assert "Path does not exist" in result.stdout


def test_info_command_complex_hierarchy(runner, tmp_path):
    """Test info command with complex scene hierarchy."""
    store_path = tmp_path / "complex_scene.zarr"

    with LuxarZarrCompiler(store_path) as compiler:
        scene = compiler.create_scene()

        # Create a complex hierarchy
        group1 = scene.add_group("Group1")
        scene.add_group("Group2")
        group1.add_group("SubGroup")

        # Add points to different levels
        pos1 = np.random.rand(10, 3).astype(np.float32)
        pos2 = np.random.rand(20, 3).astype(np.float32)
        pos3 = np.random.rand(15, 3).astype(np.float32)

        compiler.write_points("RootPoints", pos1)
        compiler.write_points("Group1/Group1Points", pos2)
        compiler.write_points("Group1/SubGroup/SubGroupPoints", pos3)

    result = runner.invoke(app, ["info", str(store_path)])

    assert result.exit_code == 0
    # Check for new format with emojis
    assert (
        "Points objects: 3" in result.stdout or "⭕ Points objects: 3" in result.stdout
    )
    assert "Total points: 45" in result.stdout or "✨ Total points: 45" in result.stdout
    assert "Group1" in result.stdout
    assert "Group2" in result.stdout
    assert "SubGroup" in result.stdout


def test_info_command_invalid_zarr_store(runner, tmp_path):
    """Test info command with invalid zarr store."""
    # Create a regular file instead of zarr store
    invalid_file = tmp_path / "not_a_zarr.txt"
    invalid_file.write_text("This is not a zarr store")

    result = runner.invoke(app, ["info", str(invalid_file)])

    assert result.exit_code == 1
    assert "Error reading info" in result.stdout


def test_serve_command_nonexistent_store(runner, tmp_path):
    """Test serve command with non-existent store."""
    nonexistent_path = tmp_path / "does_not_exist.zarr"

    result = runner.invoke(app, ["serve", str(nonexistent_path)])

    # Typer should handle the file existence validation
    assert result.exit_code != 0


def test_serve_command_zip_store_error(runner, tmp_path):
    """Test serve command with zip store (not supported)."""
    zip_path = tmp_path / "store.zip"
    zip_path.touch()  # Create empty zip file

    result = runner.invoke(app, ["serve", str(zip_path)])

    assert result.exit_code == 1
    # The serve command now checks if path is a directory
    assert "is not a directory" in result.stdout


def test_dfs_single_group(tmp_path):
    """Test _dfs with single group."""
    store_path = tmp_path / "single.zarr"
    with LuxarZarrCompiler(store_path) as compiler:
        compiler.create_scene()

    root = zarr.open_group(store_path, mode="r")
    groups = list(_dfs(root))

    assert len(groups) == 1
    assert groups[0][0] == 0  # depth
    assert groups[0][1] is root


def test_dfs_nested_groups(tmp_path):
    """Test _dfs with nested group structure."""
    store_path = tmp_path / "nested.zarr"
    with LuxarZarrCompiler(store_path) as compiler:
        scene = compiler.create_scene()

        group1 = scene.add_group("Group1")
        scene.add_group("Group2")
        group1.add_group("SubGroup")

    root = zarr.open_group(store_path, mode="r")
    groups = list(_dfs(root))

    # Should traverse: root -> Group1 -> SubGroup -> Group2
    assert len(groups) == 4

    # Check depths
    depths = [depth for depth, _ in groups]
    assert depths == [0, 1, 2, 1]  # root, Group1, SubGroup, Group2

    # Check group names
    names = [grp.basename or "/" for _, grp in groups]
    assert "/" in names
    assert "Group1" in names
    assert "Group2" in names
    assert "SubGroup" in names


def test_demo_then_info_workflow(runner, tmp_path):
    """Test complete workflow: generate demo scene then get info."""
    store_path = tmp_path / "workflow.zarr"

    # Step 1: Generate demo scene without serving
    result1 = runner.invoke(
        app,
        [
            "demo",
            "--no-serve",
            "--output",
            str(store_path),
            "--points",
            "75",
            "--seed",
            "999",
        ],
    )
    assert result1.exit_code == 0
    assert store_path.exists()

    # Step 2: Get info about the scene
    result2 = runner.invoke(app, ["info", str(store_path)])
    assert result2.exit_code == 0
    assert (
        "Total points: 75" in result2.stdout or "✨ Total points: 75" in result2.stdout
    )
    assert "LorenzAttractor" in result2.stdout


def test_cli_help_commands(runner):
    """Test help commands for all CLI functions."""
    # Test main help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "luxar – build and serve Zarr-backed 3-D scenes" in result.stdout

    # Test individual command help
    for command in ["demo", "serve", "info", "viewer"]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert command in result.stdout.lower()


def test_invalid_command(runner):
    """Test invalid command handling."""
    result = runner.invoke(app, ["invalid_command"])
    assert result.exit_code != 0
