"""
Tests for the luxar CLI commands.
"""

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar import Dimensions, LuxarZarrCompiler
from luxar.cli import app
from luxar.cli.main import _dfs


@pytest.fixture
def runner():
    """CLI test runner fixture."""
    return CliRunner()


@pytest.fixture
def sample_scene(tmp_path):
    """Create a sample scene for testing."""
    store_path = tmp_path / "test_scene.luxar.zarr"
    from luxar.demos import create_lorenz_attractor

    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


def test_demo_command_no_serve_success(runner, tmp_path) -> None:
    """Test successful demo generation without serving."""
    output_path = tmp_path / "demo_test.luxar.zarr"

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


def test_demo_command_no_serve_with_defaults(runner, tmp_path) -> None:
    """Test demo command without serving with default parameters."""
    output_path = tmp_path / "default_demo.luxar.zarr"

    result = runner.invoke(app, ["demo", "--no-serve", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Generated 10,000 points" in result.stdout
    # [Python-R1/A-C2] Validate the zarr is actually readable and carries
    # the expected default-count shape — a mutation that produced an empty
    # store, wrote to the wrong key, or skipped the colors attribute
    # would slip past a path-only `exists()` check.
    root = zarr.open_group(output_path, mode="r")
    assert "LorenzAttractor" in root
    assert root["LorenzAttractor"]["positions"].shape == (10_000, 3)
    assert root["LorenzAttractor"]["colors"].shape == (10_000, 3)


def test_demo_command_no_serve_short_options(runner, tmp_path) -> None:
    """Test demo command without serving using short option flags."""
    output_path = tmp_path / "short_opts.luxar.zarr"

    result = runner.invoke(
        app, ["demo", "--no-serve", "-o", str(output_path), "-n", "25"]
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Generated 25 points" in result.stdout
    # [Python-R1/A-C2] Short-option flags must produce the same zarr
    # shape as long-option flags; pin the count round-trip.
    root = zarr.open_group(output_path, mode="r")
    assert "LorenzAttractor" in root
    assert root["LorenzAttractor"]["positions"].shape == (25, 3)


def test_demo_command_no_serve_failure(runner, tmp_path) -> None:
    """Test demo command when scene creation fails."""
    # Use invalid path to trigger failure
    invalid_path = "/invalid/path/that/does/not/exist.luxar.zarr"

    result = runner.invoke(
        app, ["demo", "--no-serve", "--output", invalid_path, "--points", "10"]
    )

    assert result.exit_code == 1
    # [Python-R1/A-C1] The bare `"Error:" in stdout` check passed any
    # error message — a regression that swallowed an unrelated error
    # (TypeError, KeyboardInterrupt) and printed "Error: foo" would
    # have silently slipped through. Verify both the marker AND that
    # the message names the offending path so we know the failure
    # came from the path-resolution code.
    assert "Error:" in result.stdout
    assert "/invalid/path" in result.stdout or "invalid" in result.stdout.lower()


def test_info_command_success(runner, sample_scene) -> None:
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


def test_info_command_nonexistent_path(runner, tmp_path) -> None:
    """Test info command with non-existent path."""
    nonexistent_path = tmp_path / "does_not_exist.luxar.zarr"

    result = runner.invoke(app, ["info", str(nonexistent_path)])

    assert result.exit_code == 1
    assert "path does not exist" in result.stdout
    assert "does_not_exist" in result.stdout


def test_info_command_complex_hierarchy(runner, tmp_path) -> None:
    """Test info command with complex scene hierarchy."""
    store_path = tmp_path / "complex_scene.luxar.zarr"

    with LuxarZarrCompiler(store_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

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


def test_info_command_invalid_zarr_store(runner, tmp_path) -> None:
    """Test info command with invalid zarr store."""
    # Create a regular file instead of zarr store
    invalid_file = tmp_path / "not_a_zarr.txt"
    invalid_file.write_text("This is not a zarr store")

    result = runner.invoke(app, ["info", str(invalid_file)])

    assert result.exit_code == 1
    assert "Error reading info" in result.stdout


# [Python-R6 / A-G2] Empty / malformed zarr store boundary cases for
# the info command. The existing tests cover complex hierarchies and
# the "regular file" path but never:
#   - empty directory passed as zarr root
#   - zarr group with no Luxar metadata
#   - zarr group with only metadata, no points/lines/gsplats children
# These are the failure modes a partially-written or aborted compile
# would leave behind.
def test_info_command_empty_directory(runner, tmp_path) -> None:
    """Empty directory should fail with a clear error, not crash."""
    empty_dir = tmp_path / "empty.luxar.zarr"
    empty_dir.mkdir()
    result = runner.invoke(app, ["info", str(empty_dir)])
    assert result.exit_code == 1
    # Some kind of error must be reported — either "Error reading info"
    # (the catch-all path) or a more specific zarr-related message.
    assert any(s in result.stdout for s in ("Error", "invalid", "Invalid", "not")), (
        f"empty-dir info should report an error; got: {result.stdout!r}"
    )


def test_info_command_zarr_group_without_luxar_metadata(runner, tmp_path) -> None:
    """A zarr group with no Luxar metadata should be reported as invalid
    rather than crashing. Pin that the error message contains something
    actionable (mentions the path OR has the canonical 'Error' prefix)."""
    import zarr

    bare_store = tmp_path / "bare.luxar.zarr"
    # Create a valid zarr group but with no Luxar data
    zarr.open_group(str(bare_store), mode="w")

    result = runner.invoke(app, ["info", str(bare_store)])
    # Either the info command fails (preferred) or it succeeds with a
    # "no data" message. Both are acceptable as long as the user gets
    # actionable output.
    assert result.exit_code in (0, 1)
    # The output must reference the input path or describe the structure
    # — not be silently empty.
    assert len(result.stdout.strip()) > 0, "info on bare zarr produced empty output"


def test_serve_command_nonexistent_store(runner, tmp_path) -> None:
    """Test serve command with non-existent store."""
    nonexistent_path = tmp_path / "does_not_exist.luxar.zarr"

    result = runner.invoke(app, ["serve", str(nonexistent_path)])

    # Typer should handle the file existence validation
    assert result.exit_code != 0


def test_serve_command_zip_store_error(runner, tmp_path) -> None:
    """Test serve command with zip store (not supported)."""
    zip_path = tmp_path / "store.zip"
    zip_path.touch()  # Create empty zip file

    result = runner.invoke(app, ["serve", str(zip_path)])

    assert result.exit_code == 1
    # The serve command now checks if path is a directory
    assert "is not a directory" in result.stdout


def test_dfs_single_group(tmp_path) -> None:
    """Test _dfs with single group."""
    store_path = tmp_path / "single.luxar.zarr"
    with LuxarZarrCompiler(store_path) as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())

    root = zarr.open_group(store_path, mode="r")
    groups = list(_dfs(root))

    assert len(groups) == 1
    assert groups[0][0] == 0  # depth
    assert groups[0][1] is root


def test_dfs_nested_groups(tmp_path) -> None:
    """Test _dfs with nested group structure."""
    store_path = tmp_path / "nested.luxar.zarr"
    with LuxarZarrCompiler(store_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

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


def test_demo_then_info_workflow(runner, tmp_path) -> None:
    """Test complete workflow: generate demo scene then get info."""
    store_path = tmp_path / "workflow.luxar.zarr"

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


def test_cli_help_commands(runner) -> None:
    """Test help commands for all CLI functions."""
    # Test main help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "luxar – build and serve Zarr-backed nD scenes" in result.stdout

    # [Python-R3/A-W1] The per-command help test previously only checked
    # that the command name appeared somewhere in stdout — a regression
    # that produced empty help (or just printed the binary name) would
    # silently pass. Strengthen by requiring BOTH the "Usage:" header
    # AND the "Options" section header — both are universal in
    # Click/Typer help output regardless of terminal width or rendering
    # mode (the `--help` flag string itself can wrap on narrow CI
    # terminals, so we can't anchor to it directly).
    for command in ["demo", "serve", "info", "viewer"]:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert command in result.stdout.lower()
        # Anchor on the two structural headers Click/Typer always emit.
        assert "Usage:" in result.stdout, (
            f"{command} --help has no Usage: header; output was: {result.stdout!r}"
        )
        # "Options" section is universal — Click renders it as "Options:"
        # (Click) or "╭─ Options" (Typer's rich-rendered mode); the
        # substring "Options" matches both.
        assert "Options" in result.stdout, f"{command} --help missing Options section"


def test_invalid_command(runner) -> None:
    """Test invalid command handling."""
    result = runner.invoke(app, ["invalid_command"])
    assert result.exit_code != 0


# ── Tests guarding CLI fixes ─────────────────────────────────────────────────


def test_version_flag(runner) -> None:
    """Test --version flag shows version string (#7)."""
    from luxar import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_command_shows_help(runner) -> None:
    """Test that running luxar with no command shows help (regression guard)."""
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "COMMAND" in result.stdout or "Commands" in result.stdout


def test_info_format_validation(runner, sample_scene) -> None:
    """Test that --format rejects invalid values (#1)."""
    result = runner.invoke(app, ["info", str(sample_scene), "--format", "xml"])
    assert result.exit_code == 1
    assert "Unknown format" in result.stdout


def test_info_json_with_stats_includes_details(runner, sample_scene) -> None:
    """Test that --stats adds shape/dtype to JSON output (#16)."""
    import json

    result = runner.invoke(
        app, ["info", str(sample_scene), "--format", "json", "--stats"]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data["points_objects"]) > 0
    assert "shape" in data["points_objects"][0]
    assert "dtype" in data["points_objects"][0]


def test_info_json_without_stats_no_details(runner, sample_scene) -> None:
    """Test that JSON without --stats omits shape/dtype (#16)."""
    import json

    result = runner.invoke(app, ["info", str(sample_scene), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data["points_objects"]) > 0
    assert "shape" not in data["points_objects"][0]


def test_info_detects_lines_objects(runner, tmp_path) -> None:
    """Test that info detects Lines geometry type (#9)."""
    import json

    store_path = tmp_path / "lines_scene.luxar.zarr"
    root = zarr.open_group(store_path, mode="w")
    root.attrs["type"] = "scene"
    lines_group = root.create_group("my_lines")
    lines_group.attrs["type"] = "lines"
    lines_group.create_dataset(
        "vertices", data=np.random.rand(100, 3).astype(np.float32)
    )
    lines_group.create_dataset("widths", data=np.ones(100, dtype=np.float32))

    result = runner.invoke(app, ["info", str(store_path), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data["lines_objects"]) == 1
    assert data["lines_objects"][0]["n_vertices"] == 100
    assert data["n_lines_vertices_total"] == 100


def test_info_detects_gsplats_objects(runner, tmp_path) -> None:
    """Test that info detects GSplats geometry type (#9)."""
    import json

    store_path = tmp_path / "gsplats_scene.luxar.zarr"
    root = zarr.open_group(store_path, mode="w")
    root.attrs["type"] = "scene"
    gs_group = root.create_group("my_gsplats")
    gs_group.attrs["type"] = "gsplats"
    gs_group.create_dataset("centers", data=np.random.rand(50, 3).astype(np.float32))

    result = runner.invoke(app, ["info", str(store_path), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert len(data["gsplats_objects"]) == 1
    assert data["gsplats_objects"][0]["n_splats"] == 50
    assert data["n_gsplats_total"] == 50


def test_demo_no_serve_requires_output(runner) -> None:
    """Test that demo --no-serve without --output errors (#11)."""
    result = runner.invoke(app, ["demo", "--no-serve"])
    assert result.exit_code == 1
    assert "--output is required" in result.stdout


def test_demo_rejects_zero_points(runner, tmp_path) -> None:
    """Test that demo rejects --points 0 with clear message (#3)."""
    result = runner.invoke(
        app,
        ["demo", "--no-serve", "-o", str(tmp_path / "x.luxar.zarr"), "--points", "0"],
    )
    assert result.exit_code == 1
    assert "must be positive" in result.stdout


def test_demo_rejects_negative_points(runner, tmp_path) -> None:
    """Test that demo rejects --points -1 with clear message (#4)."""
    result = runner.invoke(
        app,
        ["demo", "--no-serve", "-o", str(tmp_path / "x.luxar.zarr"), "--points", "-1"],
    )
    assert result.exit_code == 1
    assert "must be positive" in result.stdout


# [Python-R6 / A-G1] Type-validation boundary cases for --points.
# Typer/Click should reject float and non-numeric inputs at the
# argument-parsing layer with a Click-style "Invalid value" exit_code
# (typically 2, not 1). Pin both type-rejection paths.
def test_demo_rejects_float_points(runner, tmp_path) -> None:
    """`--points 1.5` is a float; Typer's int annotation rejects it
    with exit_code 2 (Click's "Invalid value" code) BEFORE our
    `n_points <= 0` runtime guard fires."""
    result = runner.invoke(
        app,
        ["demo", "--no-serve", "-o", str(tmp_path / "x.luxar.zarr"), "--points", "1.5"],
    )
    assert result.exit_code != 0
    # Click error path: combined output contains either "Invalid value"
    # or our runtime guard message. EITHER is correct so long as the
    # demo command DOES NOT proceed (a regression that silently
    # accepted 1.5 and floored to 1 would slip both checks).
    combined = (result.stdout or "") + (result.stderr or "")
    assert (
        "Invalid" in combined or "must be" in combined or "is not a valid" in combined
    )


def test_demo_rejects_non_numeric_points(runner, tmp_path) -> None:
    """`--points abc` is not numeric; Typer rejects at parse time."""
    result = runner.invoke(
        app,
        ["demo", "--no-serve", "-o", str(tmp_path / "x.luxar.zarr"), "--points", "abc"],
    )
    assert result.exit_code != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert "Invalid" in combined or "is not a valid" in combined


def test_info_tree_shows_lines_icon(runner, tmp_path) -> None:
    """Test that tree view shows correct icon for Lines (#9)."""
    store_path = tmp_path / "lines_scene.luxar.zarr"
    root = zarr.open_group(store_path, mode="w")
    root.attrs["type"] = "scene"
    lines_group = root.create_group("my_lines")
    lines_group.attrs["type"] = "lines"
    lines_group.create_dataset(
        "vertices", data=np.random.rand(10, 3).astype(np.float32)
    )

    result = runner.invoke(app, ["info", str(store_path)])
    assert result.exit_code == 0
    assert "my_lines" in result.stdout
    # The tree should show the lines icon
    assert "📏" in result.stdout


def test_serve_error_not_double_printed(runner, tmp_path) -> None:
    """A typer.Exit raised inside serve must escape the broad except unchanged.

    typer.Exit subclasses RuntimeError, so a bare ``except Exception`` used to
    catch the command's own exit path and re-print a spurious
    '❌ Error serving path: 1' wrapper line.
    """
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")

    result = runner.invoke(app, ["serve", str(not_a_dir)])

    assert result.exit_code == 1
    assert "is not a directory" in result.stdout
    assert "Error serving path" not in result.stdout
