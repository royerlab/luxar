"""Tests for cli/utils.py module.

Tests cover:
- open_browser function
- check_port_available function
- find_available_port function
- check_viewer_built function
- get_viewer_dist_path function
- build_viewer function
- format_tree_node function
- format_memory_size function
- get_zarr_info function
- validate_zarr_store function
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import zarr

from luxar.cli.utils import (
    check_port_available,
    check_viewer_built,
    find_available_port,
    format_memory_size,
    format_tree_node,
    get_viewer_dist_path,
    get_zarr_info,
    open_browser,
    validate_zarr_store,
)


class TestOpenBrowser:
    """Tests for open_browser function."""

    def test_open_browser_success(self) -> None:
        """Test open_browser returns True on success."""
        with patch("webbrowser.open") as mock_open:
            mock_open.return_value = True
            result = open_browser("http://localhost:8000")
            assert result is True
            mock_open.assert_called_once_with("http://localhost:8000")

    def test_open_browser_failure(self) -> None:
        """Test open_browser returns False on failure."""
        with patch("webbrowser.open") as mock_open:
            mock_open.side_effect = Exception("Browser not available")
            result = open_browser("http://localhost:8000", suppress_errors=True)
            assert result is False

    def test_open_browser_failure_with_message(self) -> None:
        """Test open_browser prints error when not suppressed."""
        with patch("webbrowser.open") as mock_open:
            with patch("luxar.cli.utils.aprint") as mock_aprint:
                mock_open.side_effect = Exception("Browser not available")
                result = open_browser("http://localhost:8000", suppress_errors=False)
                assert result is False
                mock_aprint.assert_called_once()


class TestCheckPortAvailable:
    """Tests for check_port_available function."""

    def test_available_port(self) -> None:
        """Test check_port_available returns True for available port."""
        # Use a high port that's likely available
        result = check_port_available(59999)
        # This may vary depending on system state, but should work most times
        assert isinstance(result, bool)

    def test_unavailable_port(self) -> None:
        """Test check_port_available returns False for unavailable port."""
        import socket

        # Bind to an OS-assigned ephemeral port (port=0) rather than a
        # hardcoded one. A hardcoded port races against any other process
        # that happens to occupy that port on the CI runner (or against
        # leftover TIME_WAIT sockets between repeated test runs), causing
        # the test's own `sock.bind()` to raise
        # `OSError: [Errno 98] Address already in use` — a flake we saw
        # take down PR #285's python-tests-3.12. Port 0 lets the OS pick
        # an unused port, then we read it back and ask
        # check_port_available about that exact port.
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except PermissionError:
            pytest.skip("Socket operations not permitted in this environment")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", 0))
            bound_port = sock.getsockname()[1]
            result = check_port_available(bound_port)
            assert result is False
        finally:
            sock.close()

    def test_socket_closed_on_error(self) -> None:
        """Test check_port_available closes socket after bind failure."""
        with patch("luxar.cli.utils.socket.socket") as mock_socket:
            mock_sock = MagicMock()
            mock_sock.bind.side_effect = OSError("in use")
            mock_socket.return_value = mock_sock

            result = check_port_available(12345)

            assert result is False
            mock_sock.close.assert_called_once()


class TestFindAvailablePort:
    """Tests for find_available_port function."""

    def test_find_available_port_success(self) -> None:
        """Test find_available_port returns a port."""
        result = find_available_port(start_port=59900, max_attempts=100)
        # Should find an available port in the high range
        assert result is None or (isinstance(result, int) and result >= 59900)

    def test_find_available_port_max_attempts(self) -> None:
        """Test find_available_port returns None after max attempts."""
        # Mock check_port_available to always return False
        with patch("luxar.cli.utils.check_port_available") as mock_check:
            mock_check.return_value = False
            result = find_available_port(start_port=8000, max_attempts=3)
            assert result is None
            assert mock_check.call_count == 3

    def test_find_available_port_with_end_port(self) -> None:
        """Test find_available_port honors the explicit end_port keyword."""
        with patch("luxar.cli.utils.check_port_available") as mock_check:
            mock_check.side_effect = [False, True]
            result = find_available_port(9000, end_port=9001)
            assert result == 9001

    def test_find_available_port_end_port_below_start(self) -> None:
        """end_port below start_port finds nothing."""
        assert find_available_port(9000, end_port=8000) is None

    def test_find_available_port_end_port_capped(self) -> None:
        """end_port past the valid range is capped at 65535."""
        with patch("luxar.cli.utils.check_port_available", return_value=False):
            assert find_available_port(65534, end_port=99999) is None

    def test_find_available_port_threads_host(self) -> None:
        """The host argument reaches check_port_available."""
        with patch("luxar.cli.utils.check_port_available") as mock_check:
            mock_check.return_value = True
            find_available_port(9000, host="0.0.0.0")
            mock_check.assert_called_once_with(9000, "0.0.0.0")


class TestCheckViewerBuilt:
    """Tests for check_viewer_built function."""

    def test_viewer_not_built(self) -> None:
        """Test check_viewer_built returns False when dist doesn't exist."""
        with patch("luxar.cli.utils.get_viewer_dist_path") as mock_path:
            mock_dist = MagicMock()
            mock_dist.exists.return_value = False
            mock_path.return_value = mock_dist
            result = check_viewer_built()
            assert result is False

    def test_viewer_built_no_index(self) -> None:
        """Test check_viewer_built returns False when index.html missing."""
        with patch("luxar.cli.utils.get_viewer_dist_path") as mock_path:
            mock_dist = MagicMock()
            mock_dist.exists.return_value = True
            mock_index = MagicMock()
            mock_index.exists.return_value = False
            mock_dist.__truediv__ = lambda self, x: mock_index
            mock_path.return_value = mock_dist
            result = check_viewer_built()
            assert result is False


class TestGetViewerDistPath:
    """Tests for get_viewer_dist_path function."""

    def test_get_viewer_dist_path_returns_path(self) -> None:
        """Test get_viewer_dist_path returns a Path object."""
        result = get_viewer_dist_path()
        assert isinstance(result, Path)

    def test_get_viewer_dist_path_contains_luxar_viewer(self) -> None:
        """Test get_viewer_dist_path path includes luxar-viewer."""
        result = get_viewer_dist_path()
        assert "luxar-viewer" in str(result) or "dist" in str(result)

    def test_get_viewer_dist_path_parent_is_the_real_viewer_package(self) -> None:
        """The dist path must sit inside the actual viewer package.

        Regression tripwire for the fresh-clone auto-build bug: with no
        ``dist/`` built yet, the old code fell through to a fallback that
        resolved to ``<repo>/packages/packages/luxar-viewer`` (one ``.parent``
        short), so ``build_viewer()`` ran pnpm in a nonexistent directory and
        misreported "pnpm not found". In a dev tree (bundled ``_viewer_dist``
        absent) the returned path's parent must be the real viewer package —
        whether or not dist/ exists yet.
        """
        result = get_viewer_dist_path()
        if result.name == "_viewer_dist":  # installed-wheel bundled viewer
            pytest.skip("bundled _viewer_dist present; dev-tree layout n/a")
        assert result.parent.name == "luxar-viewer"
        assert (result.parent / "package.json").exists()


class TestFormatTreeNode:
    """Tests for format_tree_node function."""

    def test_root_node(self) -> None:
        """Test format_tree_node for root node."""
        result = format_tree_node("root", depth=0, is_last=True)
        assert "root" in result
        assert result.startswith("📊")

    def test_child_node_last(self) -> None:
        """Test format_tree_node for last child."""
        result = format_tree_node("child", depth=1, is_last=True, prefix="")
        assert "└─" in result
        assert "child" in result

    def test_child_node_not_last(self) -> None:
        """Test format_tree_node for non-last child."""
        result = format_tree_node("child", depth=1, is_last=False, prefix="")
        assert "├─" in result
        assert "child" in result

    def test_scene_type_indicator(self) -> None:
        """Test format_tree_node with scene type."""
        result = format_tree_node("scene", depth=0, is_last=True, node_type="scene")
        assert "🌐" in result

    def test_group_type_indicator(self) -> None:
        """Test format_tree_node with group type."""
        result = format_tree_node("group", depth=1, is_last=True, node_type="group")
        assert "📁" in result

    def test_points_type_indicator(self) -> None:
        """Test format_tree_node with points type."""
        result = format_tree_node("points", depth=1, is_last=True, node_type="points")
        assert "⚫" in result

    def test_with_attrs_n_points(self) -> None:
        """Test format_tree_node with n_points attribute."""
        result = format_tree_node(
            "points",
            depth=1,
            is_last=True,
            attrs={"n_points": 1000},
        )
        assert "n=1,000" in result

    def test_with_attrs_shape(self) -> None:
        """Test format_tree_node with shape attribute."""
        result = format_tree_node(
            "points",
            depth=1,
            is_last=True,
            attrs={"shape": (100, 3)},
        )
        assert "shape=(100, 3)" in result

    def test_with_attrs_dtype(self) -> None:
        """Test format_tree_node with dtype attribute."""
        result = format_tree_node(
            "points",
            depth=1,
            is_last=True,
            attrs={"dtype": "float32"},
        )
        assert "dtype=float32" in result

    def test_with_multiple_attrs(self) -> None:
        """Test format_tree_node with multiple attributes."""
        result = format_tree_node(
            "points",
            depth=1,
            is_last=True,
            attrs={"n_points": 500, "dtype": "float32"},
        )
        assert "n=500" in result
        assert "dtype=float32" in result

    def test_with_prefix(self) -> None:
        """Test format_tree_node with custom prefix."""
        result = format_tree_node("child", depth=1, is_last=True, prefix="│   ")
        assert "│   └─" in result


class TestFormatMemorySize:
    """Tests for format_memory_size function."""

    def test_bytes(self) -> None:
        """Test format_memory_size for bytes."""
        result = format_memory_size(500)
        assert "500.0 B" == result

    def test_kilobytes(self) -> None:
        """Test format_memory_size for kilobytes."""
        result = format_memory_size(2048)
        assert "2.0 KB" == result

    def test_megabytes(self) -> None:
        """Test format_memory_size for megabytes."""
        result = format_memory_size(5 * 1024 * 1024)
        assert "5.0 MB" == result

    def test_gigabytes(self) -> None:
        """Test format_memory_size for gigabytes."""
        result = format_memory_size(2 * 1024 * 1024 * 1024)
        assert "2.0 GB" == result

    def test_terabytes(self) -> None:
        """Test format_memory_size for terabytes."""
        result = format_memory_size(1.5 * 1024 * 1024 * 1024 * 1024)
        assert "1.5 TB" == result

    def test_petabytes(self) -> None:
        """Test format_memory_size for petabytes."""
        result = format_memory_size(2 * 1024 * 1024 * 1024 * 1024 * 1024)
        assert "2.0 PB" == result


class TestGetZarrInfo:
    """Tests for get_zarr_info function."""

    def test_nonexistent_path(self) -> None:
        """Test get_zarr_info for nonexistent path."""
        result = get_zarr_info(Path("/nonexistent/path.zarr"))
        assert result["exists"] is False

    def test_valid_zarr_store(self) -> None:
        """Test get_zarr_info for valid zarr store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            # Create a simple zarr store
            root = zarr.open_group(store_path, mode="w")
            root.attrs["version"] = "0.3"

            # Create positions array (simulates points)
            root.create_dataset(
                "positions",
                data=np.random.rand(100, 3).astype(np.float32),
            )

            result = get_zarr_info(store_path)
            assert result["exists"] is True
            assert result["n_groups"] >= 1
            assert result["n_arrays"] >= 1

    def test_zarr_store_with_points(self) -> None:
        """Test get_zarr_info counts points correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            # Create zarr store with points structure (type attr required)
            root = zarr.open_group(store_path, mode="w")
            root.attrs["type"] = "points"
            root.create_dataset(
                "positions",
                data=np.random.rand(50, 3).astype(np.float32),
            )
            root.create_dataset(
                "colors",
                data=np.random.randint(0, 255, (50, 3), dtype=np.uint8),
            )

            result = get_zarr_info(store_path)
            assert result["exists"] is True
            assert result["n_points_total"] == 50
            assert len(result["points_objects"]) == 1
            assert result["points_objects"][0]["has_colors"] is True

    def test_zarr_store_with_nested_groups(self) -> None:
        """Test get_zarr_info with nested groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "test.zarr"

            root = zarr.open_group(store_path, mode="w")
            child = root.create_group("child")
            child.attrs["type"] = "points"
            child.create_dataset(
                "positions",
                data=np.random.rand(25, 3).astype(np.float32),
            )

            result = get_zarr_info(store_path)
            assert result["n_groups"] >= 2  # root + child
            assert result["n_points_total"] == 25


class TestValidateZarrStore:
    """Tests for validate_zarr_store function."""

    def test_nonexistent_path(self) -> None:
        """Test validate_zarr_store for nonexistent path."""
        is_valid, error = validate_zarr_store(Path("/nonexistent/path.zarr"))
        assert is_valid is False
        assert error == "Path does not exist"

    def test_file_instead_of_dir(self) -> None:
        """Test validate_zarr_store for file instead of directory."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            f.flush()
            is_valid, error = validate_zarr_store(Path(f.name))
            assert is_valid is False
            assert error == "Path is not a directory"

    def test_invalid_zarr_store(self) -> None:
        """Test validate_zarr_store for invalid zarr directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "invalid.zarr"
            invalid_path.mkdir()
            (invalid_path / "not_zarr.txt").write_text("not zarr")

            is_valid, error = validate_zarr_store(invalid_path)
            assert is_valid is False
            assert error is not None
            assert "Not a valid Zarr store" in error

    def test_valid_zarr_store(self) -> None:
        """Test validate_zarr_store for valid zarr store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "valid.zarr"
            zarr.open_group(store_path, mode="w")

            is_valid, error = validate_zarr_store(store_path)
            assert is_valid is True
            assert error is None


class TestBuildViewer:
    """Tests for build_viewer function."""

    def test_build_viewer_pnpm_not_found(self) -> None:
        """Test build_viewer when pnpm is not found."""
        from luxar.cli.utils import build_viewer

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with patch("luxar.cli.utils.aprint"):
                result = build_viewer()
                assert result is False

    def test_build_viewer_success(self) -> None:
        """Test build_viewer when pnpm succeeds."""
        from luxar.cli.utils import build_viewer

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("luxar.cli.utils.aprint"):
                result = build_viewer()
                assert result is True

    def test_build_viewer_failure(self) -> None:
        """Test build_viewer when pnpm fails."""
        import subprocess

        from luxar.cli.utils import build_viewer

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "pnpm build", stderr="Build failed"
            )
            with patch("luxar.cli.utils.aprint"):
                result = build_viewer()
                assert result is False


class TestWaitForServer:
    """Tests for the poll-based server-readiness helper."""

    def test_returns_true_for_listening_server(self) -> None:
        """A live listener is detected well before the timeout."""
        import socket
        import threading

        from luxar.cli.utils import wait_for_server

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        thread = threading.Thread(target=lambda: None)
        thread.start()
        try:
            assert wait_for_server("127.0.0.1", port, timeout=2.0) is True
        finally:
            server.close()

    def test_returns_false_fast_when_thread_dies(self) -> None:
        """A dead server thread short-circuits the poll loop."""
        import threading
        import time

        from luxar.cli.utils import (
            find_available_port,
            wait_for_server,
        )

        dead = threading.Thread(target=lambda: None)
        dead.start()
        dead.join()

        port = find_available_port(59700)
        assert port is not None
        start = time.monotonic()
        assert wait_for_server("127.0.0.1", port, thread=dead, timeout=5.0) is False
        assert time.monotonic() - start < 1.0

    def test_returns_false_on_timeout(self) -> None:
        """Nothing listening and no thread → False after the timeout."""
        from luxar.cli.utils import find_available_port, wait_for_server

        port = find_available_port(59800)
        assert port is not None
        assert wait_for_server("127.0.0.1", port, timeout=0.3) is False

    def test_all_interfaces_host_probed_via_loopback(self) -> None:
        """0.0.0.0 binds are probed on 127.0.0.1."""
        import socket

        from luxar.cli.utils import wait_for_server

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            assert wait_for_server("0.0.0.0", port, timeout=2.0) is True
        finally:
            server.close()


class TestEnsureViewerBuilt:
    """Tests for the unified viewer-build policy."""

    def test_already_built(self) -> None:
        from luxar.cli.utils import ensure_viewer_built

        with patch("luxar.cli.utils.check_viewer_built", return_value=True):
            assert ensure_viewer_built() is True

    def test_dev_tree_auto_builds(self) -> None:
        from luxar.cli.utils import ensure_viewer_built

        with (
            patch("luxar.cli.utils.check_viewer_built", return_value=False),
            patch("luxar.cli.utils.build_viewer", return_value=True) as mock_build,
            patch("luxar.cli.utils.aprint"),
        ):
            assert ensure_viewer_built() is True
            mock_build.assert_called_once()

    def test_wheel_install_errors_without_building(self) -> None:
        """Outside a dev tree the missing viewer is a packaging error."""
        from luxar.cli.utils import ensure_viewer_built

        with (
            patch("luxar.cli.utils.check_viewer_built", return_value=False),
            patch("luxar.cli.utils._find_dev_repo_root", return_value=None),
            patch("luxar.cli.utils.build_viewer") as mock_build,
            patch("luxar.cli.utils.aprint"),
        ):
            assert ensure_viewer_built() is False
            mock_build.assert_not_called()


class TestPickPort:
    """Tests for the warn-on-shift port picker."""

    def test_returns_requested_port_silently(self) -> None:
        from luxar.cli.utils import pick_port

        with (
            patch("luxar.cli.utils.find_available_port", return_value=9000),
            patch("luxar.cli.utils.aprint") as mock_print,
        ):
            assert pick_port(9000) == 9000
            mock_print.assert_not_called()

    def test_warns_when_port_shifts(self) -> None:
        from luxar.cli.utils import pick_port

        with (
            patch("luxar.cli.utils.find_available_port", return_value=9001),
            patch("luxar.cli.utils.aprint") as mock_print,
        ):
            assert pick_port(9000, label="viewer") == 9001
            assert "busy" in mock_print.call_args.args[0]

    def test_returns_none_when_exhausted(self) -> None:
        from luxar.cli.utils import pick_port

        with (
            patch("luxar.cli.utils.find_available_port", return_value=None),
            patch("luxar.cli.utils.aprint") as mock_print,
        ):
            assert pick_port(9000) is None
            assert "No available ports" in mock_print.call_args.args[0]
