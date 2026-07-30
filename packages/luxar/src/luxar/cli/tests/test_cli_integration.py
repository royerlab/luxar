"""
Integration tests for CLI commands - NO MOCKING of internal server logic.

These tests actually start the server, make HTTP requests, and verify responses.
Following the principle from TESTING_GUIDELINES.md: mock only external dependencies.
"""

import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from luxar import Dimensions, LuxarZarrCompiler
from luxar.cli.utils import find_available_port
from luxar.utils.demos import create_lorenz_attractor


class _ImmediateThread:
    """threading.Thread stand-in that runs target inline; used to make CLI
    invocations deterministic in tests that exercise the data-server thread."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, *_a, **_k):
        return None

    def is_alive(self):
        # The target already ran inline, so the "thread" is done — this makes
        # wait_for_server() fail fast instead of polling to its timeout.
        return False


@pytest.fixture
def available_port():
    """Find an available port for testing."""
    return find_available_port(8000, end_port=9000)


@pytest.fixture
def sample_scene(tmp_path):
    """Create a sample scene for testing."""
    store_path = tmp_path / "test_scene.luxar.zarr"
    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


@pytest.fixture
def test_server(sample_scene, available_port):
    """Start a real test server in a background thread."""
    from luxar.cli.main import create_server_app

    # Create server app
    app = create_server_app(str(sample_scene), serve_viewer=False)

    # Start server in background thread
    import uvicorn

    server_thread = None
    server_started = threading.Event()

    def run_server():
        config = uvicorn.Config(
            app, host="127.0.0.1", port=available_port, log_level="error"
        )
        server = uvicorn.Server(config)

        # Signal that server is starting
        server_started.set()

        # Run server (this blocks)
        import asyncio

        asyncio.run(server.serve())

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to start
    server_started.wait(timeout=5)
    time.sleep(0.5)  # Give server time to bind to port

    # Verify server is running
    max_retries = 10
    for i in range(max_retries):
        try:
            response = requests.get(
                f"http://127.0.0.1:{available_port}/health", timeout=1
            )
            if response.status_code == 200:
                break
        except requests.exceptions.RequestException:
            if i == max_retries - 1:
                pytest.fail(
                    f"Server failed to start after {max_retries} retries — "
                    "a real regression, not a reason to skip."
                )
            time.sleep(0.5)

    yield f"http://127.0.0.1:{available_port}"

    # Server thread is daemon, will be cleaned up automatically


class TestServeIntegration:
    """Integration tests for the serve command."""

    def test_health_endpoint(self, test_server):
        """Test that health endpoint returns 200."""
        response = requests.get(f"{test_server}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

        # [Python-R6 / A-W2] Strengthen the health-endpoint contract:
        # response must be JSON Content-Type (a regression that returned
        # plain text "ok" would still satisfy the status==200 + json()
        # parse if the json() happened to succeed on the bytes), AND
        # response time must be sub-second on localhost (catches a
        # regression that added a synchronous heavy operation to the
        # /health handler).
        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"/health must return JSON; got Content-Type "
            f"{response.headers.get('Content-Type')!r}"
        )
        assert response.elapsed.total_seconds() < 2.0, (
            f"/health took {response.elapsed.total_seconds():.3f}s; should be < 2s"
        )

    def test_root_zarr_endpoint(self, test_server):
        """Test that root zarr endpoint returns correct metadata."""
        response = requests.get(f"{test_server}/.zattrs")
        assert response.status_code == 200
        data = response.json()
        assert "luxar_version" in data  # Changed from "version" to match implementation
        assert data["type"] == "scene"

    def test_scene_metadata(self, test_server):
        """Test retrieving scene metadata."""
        response = requests.get(f"{test_server}/.zattrs")
        assert response.status_code == 200
        metadata = response.json()

        # Verify expected metadata structure
        assert (
            metadata["luxar_version"] == "0.1"
        )  # Changed from "version" to "luxar_version"
        assert "scene_dimensions" in metadata

    def test_zarr_group_listing(self, test_server):
        """Test listing zarr groups."""
        response = requests.get(f"{test_server}/.zgroup")
        assert response.status_code == 200
        data = response.json()
        assert "zarr_format" in data

    def test_positions_array_access(self, test_server):
        """Test accessing point positions array."""
        # First, get the scene structure to find point nodes
        response = requests.get(f"{test_server}/.zattrs")
        assert response.status_code == 200

        # The Lorenz demo creates a node named "LorenzAttractor"
        response = requests.get(f"{test_server}/LorenzAttractor/positions/.zarray")
        assert response.status_code == 200, (
            f"Expected 200 for LorenzAttractor/positions/.zarray, "
            f"got {response.status_code}"
        )
        array_meta = response.json()
        assert "shape" in array_meta
        assert "dtype" in array_meta
        # Positions default to uint16 per-axis fixed-point (linear_perchannel_u16),
        # decoded to float32 in the viewer; PRECISION / large-extent scenes stay float32.
        assert array_meta["dtype"] in ["<u2", ">u2", "uint16", "<f4", ">f4", "float32"]

    def test_cors_headers(self, test_server):
        """Test that local CORS origins are allowed by default."""
        # CORS headers appear when Origin header is present (cross-origin request)
        origin = "http://localhost:5173"
        headers = {"Origin": origin}
        response = requests.get(f"{test_server}/health", headers=headers)
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == origin

    def test_no_cache_header_on_data_responses(self, test_server):
        """Every response must carry ``Cache-Control: no-cache``.

        StaticFiles sends ETag/Last-Modified but no Cache-Control, so
        browsers fall back to heuristic freshness and serve STALE chunks
        after a dataset is regenerated in place (same URLs, new bytes).
        ``no-cache`` forces ETag revalidation (unchanged chunks are still
        cheap 304s) so a regenerated dataset is always picked up.
        """
        for path in ("/.zattrs", "/health"):
            response = requests.get(f"{test_server}{path}")
            assert response.headers.get("Cache-Control") == "no-cache", path

    def test_non_local_cors_origin_rejected_by_default(self, sample_scene):
        """Default CORS policy should only allow loopback browser clients."""
        from luxar.cli.main import create_server_app

        client = TestClient(create_server_app(str(sample_scene)))
        response = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_wildcard_cors_requires_explicit_opt_in(self, sample_scene):
        """Wildcard CORS is still available, but credentials are disabled."""
        from luxar.cli.main import create_server_app

        client = TestClient(create_server_app(str(sample_scene), cors_origin="*"))
        response = client.get("/health", headers={"Origin": "https://example.org"})
        assert response.headers["Access-Control-Allow-Origin"] == "*"
        assert response.headers.get("Access-Control-Allow-Credentials") != "true"

    def test_viewer_command_threads_cors_origin(self, sample_scene, monkeypatch):
        """`luxar viewer --cors-origin X` must propagate X to both servers."""
        from typer.testing import CliRunner

        from luxar.cli import app
        from luxar.cli import main as cli_main

        captured: dict[str, str] = {}

        def fake_serve_viewer(
            host, port, data_url=None, open_browser_flag=True, cors_origin="local"
        ):
            captured["viewer"] = cors_origin

        def fake_serve_data(path, host, port, *args, **kwargs):
            cors_origin = kwargs.get("cors_origin")
            if cors_origin is None and len(args) >= 6:
                cors_origin = args[5]
            captured["data"] = cors_origin or "local"

        monkeypatch.setattr(cli_main, "_serve_viewer", fake_serve_viewer)
        monkeypatch.setattr(cli_main, "_serve_data", fake_serve_data)
        monkeypatch.setattr(cli_main, "ensure_viewer_built", lambda: True)
        monkeypatch.setattr(cli_main.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(cli_main, "wait_for_server", lambda *_a, **_k: True)

        result = CliRunner().invoke(
            app,
            [
                "viewer",
                "--data",
                str(sample_scene),
                "--cors-origin",
                "https://example.com",
                "--no-open",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert captured["viewer"] == "https://example.com"
        assert captured["data"] == "https://example.com"

    def test_directory_listing_blocks_parent_traversal(self, tmp_path):
        """Custom directory listings must not escape the served root."""
        from luxar.cli.main import create_server_app

        serve_root = tmp_path / "served"
        serve_root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")

        client = TestClient(create_server_app(str(serve_root)))
        response = client.get(
            "/%2e%2e/outside/", headers={"Accept": "application/json"}
        )
        assert response.status_code == 403

    def test_sensitive_system_path_requires_opt_in(self):
        """Serving filesystem roots is blocked unless explicitly allowed."""
        from luxar.cli.main import create_server_app

        with pytest.raises(ValueError, match="Refusing to serve sensitive system path"):
            create_server_app(Path(Path.cwd().anchor))

    def test_sensitive_system_paths_blocked(self):
        """System roots are flagged as sensitive."""
        from luxar.cli.main import _is_sensitive_serve_path

        # Posix-only paths — skipped on Windows where these aren't sensitive.
        if not Path("/etc").exists():
            pytest.skip("system /etc not present (probably Windows)")

        assert _is_sensitive_serve_path(Path("/etc")) is True
        assert _is_sensitive_serve_path(Path("/usr")) is True
        # Subdirectories of sensitive roots stay sensitive.
        assert _is_sensitive_serve_path(Path("/etc/passwd")) is True

    def test_tmpdir_not_sensitive(self):
        """Tempdirs (under /var on macOS, /tmp on Linux) must be servable."""
        import tempfile

        from luxar.cli.main import _is_sensitive_serve_path

        with tempfile.TemporaryDirectory() as td:
            assert _is_sensitive_serve_path(Path(td)) is False

    def test_user_directories_not_sensitive(self):
        """/home and /Users hold legitimate project data and must be servable."""
        from luxar.cli.main import _is_sensitive_serve_path

        # These paths may not exist on every machine — check that *if* they
        # exist, they are not classified as sensitive.
        for candidate in (Path("/Users"), Path("/home")):
            if candidate.exists():
                assert _is_sensitive_serve_path(candidate / "fakeuser") is False

    def test_lan_warning_silent_for_loopback_or_local_cors(self, capsys):
        """No LAN warning when bind is loopback or CORS is not wildcard."""
        from luxar.cli.main import _warn_if_lan_exposed

        # Loopback host with wildcard CORS — no warning.
        _warn_if_lan_exposed("127.0.0.1", "*")
        _warn_if_lan_exposed("localhost", "*")
        _warn_if_lan_exposed("::1", "*")
        # Non-loopback host with restricted CORS — no warning.
        _warn_if_lan_exposed("192.168.1.10", "local")
        _warn_if_lan_exposed("my-server.lan", "https://example.com")

        captured = capsys.readouterr()
        assert "Serving on host" not in captured.out

    def test_lan_warning_fires_for_lan_bind_with_wildcard(self, capsys):
        """LAN-routable bind + wildcard CORS triggers the warning."""
        from luxar.cli.main import _warn_if_lan_exposed

        _warn_if_lan_exposed("192.168.1.10", "*")
        captured = capsys.readouterr()
        assert "Serving on host=192.168.1.10" in captured.out
        assert "--cors-origin '*'" in captured.out

    def test_lan_warning_fires_for_all_interfaces_bind(self, capsys):
        """Binding all interfaces (0.0.0.0) with wildcard CORS must warn — it
        exposes the data on every network interface, not just loopback."""
        from luxar.cli.main import _warn_if_lan_exposed

        _warn_if_lan_exposed("0.0.0.0", "*")
        captured = capsys.readouterr()
        assert "Serving on host=0.0.0.0" in captured.out

    def test_404_for_nonexistent_path(self, test_server):
        """Test that nonexistent paths return 404."""
        response = requests.get(f"{test_server}/nonexistent/path")
        assert response.status_code == 404

    def test_concurrent_requests(self, test_server):
        """Test that server handles concurrent requests."""
        import concurrent.futures

        def make_request():
            response = requests.get(f"{test_server}/health", timeout=5)
            return response.status_code

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(code == 200 for code in results)

    def test_directory_listing_json(self, test_server, sample_scene):
        """Test directory listing in JSON format."""
        # Request JSON format via Accept header
        headers = {"Accept": "application/json"}
        response = requests.get(f"{test_server}/", headers=headers)

        # Server returns JSON directory listing when Accept: application/json
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "entries" in data
        entries = data["entries"]
        assert isinstance(entries, list)
        # Should contain zarr files
        assert len(entries) > 0
        # Verify structure
        assert any(e["type"] == "zarr" for e in entries)


class TestInfoCommand:
    """Integration tests for the info command (using CLI runner)."""

    def test_info_basic(self, sample_scene):
        """Test basic info command output."""
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info", str(sample_scene)])

        assert result.exit_code == 0
        assert "Version" in result.stdout or "version" in result.stdout
        assert "Points" in result.stdout or "points" in result.stdout

    def test_info_with_stats(self, sample_scene):
        """Test info command with --stats flag."""
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info", str(sample_scene), "--stats"])

        assert result.exit_code == 0
        # Should show more detailed statistics
        assert "Total" in result.stdout or "total" in result.stdout

    def test_info_with_tree(self, sample_scene):
        """Test info command with --tree flag."""
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info", str(sample_scene), "--tree"])

        assert result.exit_code == 0
        # Tree output should have hierarchy indicators
        assert "├─" in result.stdout or "└─" in result.stdout or "│" in result.stdout

    def test_info_nonexistent_store(self):
        """Test info command with nonexistent store."""
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["info", "/nonexistent/path.luxar.zarr"])

        assert result.exit_code != 0
        assert "Error" in result.stdout or "not found" in result.stdout.lower()


class TestProfilesCommand:
    """Integration tests for profiles command."""

    def test_profiles_list(self):
        """Test listing available network profiles."""
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["profiles"])

        assert result.exit_code == 0
        # Should list standard profiles
        assert "broadband" in result.stdout.lower() or "3g" in result.stdout.lower()
        # May have other profiles like satellite, 4g, 5g, etc.

    def test_profiles_output_format(self):
        """Test that profiles output is well-formatted."""
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["profiles"])

        assert result.exit_code == 0
        # Output should be structured (table or list format)
        lines = result.stdout.split("\n")
        assert len(lines) > 2  # At least header + 1 profile


class TestPortHandling:
    """Test port conflict handling and availability checking."""

    def test_find_available_port(self):
        """Test finding an available port."""
        port = find_available_port(9000, end_port=9100)
        assert 9000 <= port < 9100

        # Verify port is actually available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.close()
        except OSError:
            pytest.fail(f"Port {port} reported as available but couldn't bind")

    def test_port_conflict_handling(self, available_port):
        """Test that server handles port conflicts gracefully."""
        # Occupy the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", available_port))
        sock.listen(1)

        try:
            # Try to start server on occupied port
            from typer.testing import CliRunner

            from luxar.cli import app

            runner = CliRunner()
            # NOTE: serve has no --no-viewer flag (the viewer is opt-in via
            # --viewer); passing it here used to make this test vacuously
            # pass on the unknown-option exit code without ever exercising
            # the occupied-port path.
            result = runner.invoke(
                app,
                ["serve", "--port", str(available_port)],
                catch_exceptions=True,
            )

            # Should either fail with clear error or find alternative port
            # (Behavior depends on implementation)
            assert result.exit_code != 0 or "alternative" in result.stdout.lower()

        finally:
            sock.close()


@pytest.mark.slow
class TestServePerformance:
    """Performance tests for serve command (marked as slow)."""

    def test_large_dataset_serve(self, tmp_path, available_port):
        """Test serving a larger dataset."""
        # Create a larger scene
        store_path = tmp_path / "large_scene.luxar.zarr"
        with LuxarZarrCompiler(store_path) as compiler:
            _scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            positions = np.random.rand(10000, 3).astype(np.float32)
            compiler.write_points("LargePoints", positions)

        # Start server (using similar pattern to test_server fixture)
        # This is a simplified test - full implementation would start server and benchmark
        assert store_path.exists()

        # Verify zarr is readable
        import zarr

        store = zarr.open(str(store_path), mode="r")
        assert "LargePoints" in store

    def test_concurrent_load_performance(self, test_server):
        """Test performance under concurrent load."""
        import concurrent.futures
        import time

        def timed_request():
            start = time.time()
            response = requests.get(f"{test_server}/health", timeout=10)
            elapsed = time.time() - start
            return response.status_code, elapsed

        # Make 50 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(timed_request) for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        status_codes = [r[0] for r in results]
        assert all(code == 200 for code in status_codes)

        # Performance check: 95th percentile should be under 1 second
        times = sorted([r[1] for r in results])
        p95 = times[int(len(times) * 0.95)]
        assert p95 < 1.0, (
            f"95th percentile response time {p95:.2f}s exceeds 1s threshold"
        )


class TestDataServerMountRoot:
    """The data server must mount the dataset itself, never its parent."""

    def test_zarr_store_mounted_at_root_hides_siblings(self, sample_scene):
        """Sibling files of a served .zarr store are not exposed over HTTP."""
        from luxar.cli.serving import _build_data_app

        sibling = sample_scene.parent / "secret_sibling.txt"
        sibling.write_text("SECRET")

        client = TestClient(_build_data_app(sample_scene))

        # The store is served AT the root (data URLs carry no name suffix).
        assert client.get("/.zgroup").status_code == 200
        # Neither the sibling nor the old parent-mounted URL shape resolves.
        assert client.get("/secret_sibling.txt").status_code == 404
        assert client.get(f"/{sample_scene.name}/.zgroup").status_code == 404

    def test_resolve_mount_root(self, tmp_path):
        """Directories mount themselves; files never fall back to their parent."""
        from luxar.cli.serving import _resolve_mount_root

        store = tmp_path / "scene.luxar.zarr"
        store.mkdir()
        plain = tmp_path / "plain"
        plain.mkdir()
        lone_file = tmp_path / "volume.npy"
        lone_file.write_bytes(b"x")

        assert _resolve_mount_root(store) == store.resolve()
        assert _resolve_mount_root(plain) == plain.resolve()
        with pytest.raises(ValueError, match="must be a directory"):
            _resolve_mount_root(lone_file)

    def test_build_data_app_rechecks_directory_at_mount_time(self, tmp_path):
        """A directory replaced by a file before server startup is rejected."""
        from luxar.cli.serving import _build_data_app

        data_path = tmp_path / "scene.luxar.zarr"
        data_path.mkdir()
        data_path.rmdir()
        data_path.write_bytes(b"not a directory anymore")

        with pytest.raises(ValueError, match="must be a directory"):
            _build_data_app(data_path)

    def test_viewer_rejects_file_data_path(self, tmp_path):
        """`viewer --data <file>` is rejected before any server starts.

        A lone file store cannot be served over plain HTTP anyway, and serving
        it would mount its parent directory (every sibling file) at ``/``. The
        guard fires before ``uvicorn.run`` so the invocation returns immediately.
        """
        from typer.testing import CliRunner

        from luxar.cli import app
        from luxar.cli import main as cli_main

        lone_file = tmp_path / "scene.zarr.zip"
        lone_file.write_bytes(b"PK\x03\x04")
        (tmp_path / "secret_sibling.txt").write_text("SECRET")

        # ensure_viewer_built() runs before the guard; force True so a non-zero
        # exit can only come from the directory guard we are testing. Also stub
        # the blocking server plumbing so that if the guard ever regresses the
        # test fails cleanly on the message assertion instead of falling through
        # to uvicorn.run() and hanging the suite (while serving the tmp dir).
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(cli_main, "ensure_viewer_built", lambda: True)
        monkeypatch.setattr(cli_main, "_start_data_server_thread", lambda *a, **k: None)
        monkeypatch.setattr(cli_main, "_serve_viewer", lambda *a, **k: None)
        try:
            result = CliRunner().invoke(
                app,
                ["viewer", "--data", str(lone_file), "--no-open"],
            )
        finally:
            monkeypatch.undo()

        assert result.exit_code != 0, result.stdout
        assert "must be a directory" in result.stdout

    def test_viewer_accepts_directory_data_path(self, sample_scene, monkeypatch):
        """`viewer --data <.zarr dir>` passes the guard (no rejection)."""
        from typer.testing import CliRunner

        from luxar.cli import app
        from luxar.cli import main as cli_main

        # Stub out the blocking server plumbing so the command returns cleanly
        # once it is past the directory guard.
        monkeypatch.setattr(cli_main, "ensure_viewer_built", lambda: True)
        monkeypatch.setattr(cli_main, "_start_data_server_thread", lambda *a, **k: None)
        monkeypatch.setattr(cli_main, "_serve_viewer", lambda *a, **k: None)

        result = CliRunner().invoke(
            app,
            ["viewer", "--data", str(sample_scene), "--no-open"],
        )

        assert result.exit_code == 0, result.stdout
        assert "must be a directory" not in result.stdout
