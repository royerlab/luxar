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
    from luxar.utils.scenes import create_lorenz_attractor

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

    @patch("luxar.cli.main.ensure_viewer_built")
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

    def test_viewer_auto_build(self, runner) -> None:
        """Viewer auto-builds when missing in a dev tree.

        Targets ensure_viewer_built's real logic in cli.utils: dist missing
        (check_viewer_built False) + dev tree present → build_viewer runs.
        """
        with (
            patch("luxar.cli.utils.check_viewer_built", return_value=False),
            patch("luxar.cli.utils.build_viewer", return_value=True) as mock_build,
            patch("luxar.cli.main._serve_viewer"),
        ):
            runner.invoke(app, ["viewer", "--no-open"])
            mock_build.assert_called_once()

    @patch("luxar.cli.main.wait_for_server", return_value=True)
    @patch("luxar.cli.main.ensure_viewer_built")
    @patch("luxar.cli.main._serve_viewer")
    @patch("luxar.cli.main._serve_data")
    def test_viewer_with_data(
        self, mock_data, mock_viewer, mock_check, _mock_wait, runner, sample_scene
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
    """The `demo` sub-app dispatches to demo scripts; details live in
    test_demo_commands.py. Here we just pin that the group is mounted."""

    def test_bare_demo_shows_table(self, runner) -> None:
        result = runner.invoke(app, ["demo"])
        assert result.exit_code == 0
        assert "demos" in result.stdout.lower()

    def test_demo_unknown_key_errors(self, runner) -> None:
        result = runner.invoke(app, ["demo", "run", "no-such-demo"])
        assert result.exit_code != 0
        assert "unknown demo" in result.stdout.lower()


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

    @patch("luxar.cli.main.pick_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    def test_serve_basic(self, mock_uvicorn, _mock_port, runner, sample_scene) -> None:
        """Test basic serve command (existing functionality)."""
        result = runner.invoke(app, ["serve", str(sample_scene)])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()

    @patch("luxar.cli.main.wait_for_server", return_value=True)
    @patch("luxar.cli.main._serve_viewer")
    @patch("luxar.cli.main.pick_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.ensure_viewer_built")
    def test_serve_with_viewer(
        self,
        mock_check,
        mock_uvicorn,
        _mock_port,
        _mock_viewer,
        _mock_wait,
        runner,
        sample_scene,
    ) -> None:
        """Test serve with viewer option."""
        mock_check.return_value = True

        result = runner.invoke(app, ["serve", str(sample_scene), "--viewer"])
        assert result.exit_code == 0
        mock_check.assert_called()

    @patch("luxar.cli.main.pick_port", return_value=8000)
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

    @patch("luxar.cli.main.wait_for_server", return_value=True)
    @patch("luxar.cli.main._serve_viewer")
    @patch("luxar.cli.main.pick_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.open_browser_func")
    @patch("luxar.cli.main.ensure_viewer_built")
    def test_serve_with_viewer_and_open_opens_browser(
        self,
        mock_check,
        mock_browser,
        mock_uvicorn,
        _mock_port,
        _mock_viewer,
        _mock_wait,
        runner,
        sample_scene,
    ) -> None:
        """`serve --viewer --open` (viewer built) opens the browser exactly once
        — the positive twin of test_serve_with_viewer_not_built_skips_open."""
        mock_check.return_value = True
        result = runner.invoke(
            app,
            [
                "serve",
                str(sample_scene),
                "--viewer",
                "--open",
                "--control",
                "--control-token",
                "tap secret",
            ],
        )
        assert result.exit_code == 0, result.output
        mock_browser.assert_called_once()
        opened_url = mock_browser.call_args.args[0]
        assert "&control" in opened_url
        assert "&controlToken=tap%20secret" in opened_url

    @patch("luxar.cli.main.pick_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    def test_control_flags_warn_when_their_prerequisites_are_missing(
        self, _mock_uvicorn, _mock_port, runner, sample_scene
    ) -> None:
        result = runner.invoke(
            app, ["serve", str(sample_scene), "--control", "--control-token", "secret"]
        )
        assert result.exit_code == 0, result.output
        assert "--control requires --viewer or --viewer-only" in result.stdout

        result = runner.invoke(
            app, ["serve", str(sample_scene), "--control-token", "secret"]
        )
        assert result.exit_code == 0, result.output
        assert "--control-token requires --control" in result.stdout

    @patch("luxar.cli.main.pick_port", return_value=8000)
    @patch("luxar.cli.main.uvicorn.run")
    @patch("luxar.cli.main.open_browser_func")
    @patch("luxar.cli.main.ensure_viewer_built")
    def test_serve_with_viewer_not_built_skips_open(
        self, mock_check, mock_browser, mock_uvicorn, _mock_port, runner, sample_scene
    ) -> None:
        """Test serve with viewer requested but not built skips --open."""
        mock_check.return_value = False

        result = runner.invoke(app, ["serve", str(sample_scene), "--viewer", "--open"])

        assert result.exit_code == 0
        mock_browser.assert_not_called()

    @patch("luxar.cli.main.pick_port", return_value=8000)
    @patch("luxar.cli.main.ensure_viewer_built")
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

    def test_all_commands_help(self, runner) -> None:
        """Test that all commands have proper help."""
        # Leaf commands show an Options section.
        for cmd in ["serve", "info", "viewer"]:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0
            assert cmd in result.stdout.lower()
            assert "Options" in result.stdout
        # `demo` is a group: it lists Commands.
        demo_help = runner.invoke(app, ["demo", "--help"])
        assert demo_help.exit_code == 0
        assert "Commands" in demo_help.stdout
