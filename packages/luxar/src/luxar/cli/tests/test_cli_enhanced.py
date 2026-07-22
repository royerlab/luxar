"""
Tests for the enhanced luxar CLI commands.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from luxar import Dimensions, LuxarZarrCompiler
from luxar.cli import app


@pytest.fixture
def runner():
    """CLI test runner fixture."""
    return CliRunner()


@pytest.fixture
def sample_scene(tmp_path):
    """Create a sample scene for testing."""
    store_path = tmp_path / "test_scene.luxar.zarr"
    from luxar.utils.demos import create_lorenz_attractor

    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


@pytest.fixture
def complex_scene(tmp_path):
    """Create a complex scene with hierarchy for testing."""
    store_path = tmp_path / "complex_scene.luxar.zarr"

    with LuxarZarrCompiler(store_path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())

        # Create hierarchy
        group1 = scene.add_group("Group1")
        scene.add_group("Group2")
        group1.add_group("SubGroup")

        # Add points at different levels
        pos1 = np.random.rand(50, 3).astype(np.float32)
        pos2 = np.random.rand(100, 3).astype(np.float32)
        pos3 = np.random.rand(75, 3).astype(np.float32)

        compiler.write_points("RootPoints", pos1)
        compiler.write_points("Group1/GroupPoints", pos2)
        compiler.write_points("Group1/SubGroup/SubPoints", pos3)

    return store_path


class TestViewerCommand:
    """Test the viewer command."""

    @patch("luxar.cli.main.check_viewer_built")
    @patch("luxar.cli.main._serve_viewer")
    def test_viewer_basic(self, mock_serve, mock_check, runner) -> None:
        """Test basic viewer command."""
        mock_check.return_value = True

        # _serve_viewer is mocked, so the command returns instead of blocking.
        # Invoke synchronously so the exit-code assertion actually runs on the
        # main thread — an assertion inside a daemon thread is swallowed by the
        # threading runtime and can never fail the test.
        result = runner.invoke(app, ["viewer", "--no-open"])

        assert result.exit_code == 0, result.output
        mock_check.assert_called_once()
        mock_serve.assert_called_once()

    @patch("luxar.cli.main.check_viewer_built")
    @patch("luxar.cli.main.build_viewer")
    def test_viewer_auto_build(self, mock_build, mock_check, runner) -> None:
        """Test viewer auto-builds if not built."""
        mock_check.return_value = False
        mock_build.return_value = True

        # This would normally block, so we just check the initial checks
        with patch("luxar.cli.main._serve_viewer"):
            runner.invoke(app, ["viewer", "--no-open"])
            # Build should be attempted
            mock_build.assert_called_once()

    @patch("luxar.cli.main.check_viewer_built")
    @patch("luxar.cli.main._serve_viewer")
    @patch("luxar.cli.main._serve_data")
    def test_viewer_with_data(
        self, mock_data, mock_viewer, mock_check, runner, sample_scene
    ) -> None:
        """Test viewer with data option."""
        mock_check.return_value = True

        result = runner.invoke(
            app, ["viewer", "--data", str(sample_scene), "--no-open"]
        )

        assert result.exit_code == 0, result.output
        # The viewer server is started synchronously; the data server runs in a
        # background thread (so mock_data is racy). Assert the viewer was served
        # once AND received a data_url at the data-server root — the store is
        # mounted AT the root (no store-name suffix, so sibling files are never
        # exposed) and the URL must carry no trailing slash.
        mock_viewer.assert_called_once()
        data_url = mock_viewer.call_args.args[2]
        assert data_url is not None
        assert sample_scene.name not in data_url
        assert not data_url.endswith("/")
        assert data_url.startswith("http://127.0.0.1:")


class TestDemoCommand:
    """Test the demo command."""

    @patch("luxar.cli.main.check_viewer_built")
    @patch("luxar.cli.main._serve_viewer")
    @patch("luxar.cli.main._serve_data")
    def test_demo_basic(self, mock_data, mock_viewer, mock_check, runner) -> None:
        """Test basic demo command."""
        mock_check.return_value = True

        with patch("luxar.utils.demos.create_lorenz_attractor") as mock_create:
            runner.invoke(app, ["demo", "--no-open", "--points", "100"])
            # Demo should be created
            assert mock_create.called
            call_args = mock_create.call_args
            assert call_args[1]["n_points"] == 100

    def test_demo_with_output(self, runner, tmp_path) -> None:
        """Test demo with specified output."""
        output = tmp_path / "my_demo.luxar.zarr"

        with patch("luxar.cli.main.check_viewer_built", return_value=True):
            with patch("luxar.cli.main._serve_viewer"):
                with patch("luxar.cli.main._serve_data"):
                    runner.invoke(
                        app,
                        [
                            "demo",
                            "--output",
                            str(output),
                            "--points",
                            "50",
                            "--no-open",
                        ],
                    )
                    # Output should exist
                    assert output.exists()

    def test_demo_invalid_type(self, runner) -> None:
        """Test demo with invalid type."""
        result = runner.invoke(app, ["demo", "--type", "invalid", "--no-open"])
        assert result.exit_code == 1
        assert "Unknown demo type" in result.stdout


class TestEnhancedInfoCommand:
    """Test the enhanced info command."""

    def test_info_tree_view(self, runner, complex_scene) -> None:
        """Test info command with tree view."""
        result = runner.invoke(app, ["info", str(complex_scene)])
        assert result.exit_code == 0

        # Check for tree elements
        assert (
            "🌳 Scene Hierarchy" in result.stdout or "Scene Hierarchy" in result.stdout
        )
        assert "Group1" in result.stdout
        assert "Group2" in result.stdout
        assert "SubGroup" in result.stdout
        assert "225" in result.stdout  # Total points

    def test_info_no_tree(self, runner, complex_scene) -> None:
        """Test info command without tree."""
        result = runner.invoke(app, ["info", str(complex_scene), "--no-tree"])
        assert result.exit_code == 0
        assert "Summary Statistics" in result.stdout

    def test_info_with_stats(self, runner, complex_scene) -> None:
        """Test info command with detailed stats."""
        result = runner.invoke(app, ["info", str(complex_scene), "--stats"])
        assert result.exit_code == 0
        assert "Points Objects Details" in result.stdout

    def test_info_json_format(self, runner, complex_scene) -> None:
        """Test info command with JSON output."""
        result = runner.invoke(app, ["info", str(complex_scene), "--format", "json"])
        assert result.exit_code == 0

        # Should be valid JSON
        data = json.loads(result.stdout)
        assert "n_points_total" in data
        assert data["n_points_total"] == 225

    def test_info_depth_limit(self, runner, complex_scene) -> None:
        """Test info command with depth limit."""
        result = runner.invoke(app, ["info", str(complex_scene), "--depth", "1"])
        assert result.exit_code == 0
        # SubGroup should not appear (depth 2)
        assert "Group1" in result.stdout
        # SubGroup might still appear in summary but not in tree


class TestEnhancedServeCommand:
    """Test the enhanced serve command."""

    @patch("luxar.cli.main.find_available_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    def test_serve_basic(self, mock_uvicorn, _mock_port, runner, sample_scene) -> None:
        """Test basic serve command (existing functionality)."""
        result = runner.invoke(app, ["serve", str(sample_scene)])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()

    @patch("luxar.cli.main.find_available_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.check_viewer_built")
    def test_serve_with_viewer(
        self, mock_check, mock_uvicorn, _mock_port, runner, sample_scene
    ) -> None:
        """Test serve with viewer option."""
        mock_check.return_value = True

        result = runner.invoke(app, ["serve", str(sample_scene), "--viewer"])
        assert result.exit_code == 0
        mock_check.assert_called()

    @patch("luxar.cli.main.find_available_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.open_browser_func")
    def test_serve_open_without_viewer_is_ignored(
        self, mock_browser, mock_uvicorn, _mock_port, runner, sample_scene
    ) -> None:
        """`serve --open` without `--viewer` is ignored — there is nothing to
        open, so the browser is never launched and a warning is printed."""
        result = runner.invoke(app, ["serve", str(sample_scene), "--open"])
        assert result.exit_code == 0
        mock_browser.assert_not_called()
        assert "--open requires --viewer" in result.stdout

    @patch("luxar.cli.main.find_available_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.open_browser_func")
    @patch("luxar.cli.main.check_viewer_built")
    def test_serve_with_viewer_and_open_opens_browser(
        self, mock_check, mock_browser, mock_uvicorn, _mock_port, runner, sample_scene
    ) -> None:
        """`serve --viewer --open` (viewer built) opens the browser exactly once
        — the positive twin of test_serve_with_viewer_not_built_skips_open."""
        mock_check.return_value = True
        result = runner.invoke(app, ["serve", str(sample_scene), "--viewer", "--open"])
        assert result.exit_code == 0, result.output
        mock_browser.assert_called_once()

    @patch("luxar.cli.main.find_available_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.open_browser_func")
    @patch("luxar.cli.main.check_viewer_built")
    def test_serve_with_viewer_not_built_skips_open(
        self, mock_check, mock_browser, mock_uvicorn, _mock_port, runner, sample_scene
    ) -> None:
        """Test serve with viewer requested but not built skips --open."""
        mock_check.return_value = False

        result = runner.invoke(app, ["serve", str(sample_scene), "--viewer", "--open"])

        assert result.exit_code == 0
        mock_browser.assert_not_called()

    @patch("luxar.cli.main.find_available_port", return_value=8000)
    @patch("luxar.cli.main.check_viewer_built")
    @patch("luxar.cli.main._serve_viewer")
    def test_serve_viewer_only(
        self, mock_serve, mock_check, _mock_port, runner
    ) -> None:
        """Test serve viewer only."""
        mock_check.return_value = True

        result = runner.invoke(app, ["serve", "--viewer-only"])
        assert result.exit_code == 0
        mock_check.assert_called()

    def test_serve_no_path_error(self, runner) -> None:
        """Test serve without path (and not viewer-only)."""
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1
        assert "Path required" in result.stdout


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_demo_no_serve_then_info_workflow(self, runner, tmp_path) -> None:
        """Test generating demo data without serving then viewing info."""
        output = tmp_path / "test.luxar.zarr"

        # Generate
        result1 = runner.invoke(
            app, ["demo", "--no-serve", "--output", str(output), "--points", "50"]
        )
        assert result1.exit_code == 0
        assert output.exists()

        # Info
        result2 = runner.invoke(app, ["info", str(output)])
        assert result2.exit_code == 0
        assert "50" in result2.stdout

    def test_demo_creates_temp_dir(self, runner) -> None:
        """Test that demo creates temp directory when no output specified."""
        with patch("luxar.cli.main.check_viewer_built", return_value=True):
            with patch("luxar.cli.main._serve_viewer"):
                with patch("luxar.cli.main._serve_data"):
                    with patch("tempfile.mkdtemp") as mock_temp:
                        mock_temp.return_value = "/tmp/test"
                        runner.invoke(app, ["demo", "--points", "10", "--no-open"])
                        mock_temp.assert_called_once()

    def test_all_commands_help(self, runner) -> None:
        """Test that all commands have proper help."""
        commands = ["serve", "info", "viewer", "demo"]

        for cmd in commands:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            assert cmd in result.stdout.lower()
            assert "Options" in result.stdout
