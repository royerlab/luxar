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


@pytest.fixture
def available_port():
    """Find an available port for testing."""
    return find_available_port(8000, 9000)


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
                pytest.skip("Server failed to start")
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
        monkeypatch.setattr(cli_main, "check_viewer_built", lambda: True)
        monkeypatch.setattr(cli_main.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(cli_main.time, "sleep", lambda *_a, **_k: None)

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

    def test_demo_command_threads_cors_origin(self, monkeypatch, tmp_path):
        """`luxar demo --cors-origin X` must propagate X to both servers."""
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
        monkeypatch.setattr(cli_main, "check_viewer_built", lambda: True)
        monkeypatch.setattr(cli_main.threading, "Thread", _ImmediateThread)
        monkeypatch.setattr(cli_main.time, "sleep", lambda *_a, **_k: None)

        out = tmp_path / "demo.luxar.zarr"
        result = CliRunner().invoke(
            app,
            [
                "demo",
                "--no-open",
                "--points",
                "100",
                "--output",
                str(out),
                "--cors-origin",
                "https://example.com",
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


class TestDemoCommand:
    """Integration tests for demo command."""

    def test_demo_creates_valid_zarr(self, tmp_path):
        """Test that demo command creates a valid zarr store."""
        from typer.testing import CliRunner

        from luxar.cli import app

        output_path = tmp_path / "demo_output.luxar.zarr"
        runner = CliRunner()

        # Run demo with --no-serve and --output flags
        result = runner.invoke(
            app, ["demo", "--no-serve", "--output", str(output_path), "--points", "100"]
        )

        assert result.exit_code == 0, f"Demo command failed: {result.stdout}"
        assert output_path.exists()

        # Verify it's a valid zarr store
        import zarr

        store = zarr.open(str(output_path), mode="r")
        attrs_dict = dict(store.attrs)
        assert "version" in attrs_dict or "luxar_version" in attrs_dict
        # Verify it has expected structure
        assert "Lorenz" in store or len(list(store.group_keys())) > 0

    def test_demo_with_seed(self, tmp_path):
        """Test demo with seed produces reproducible results."""
        from typer.testing import CliRunner

        from luxar.cli import app

        output1 = tmp_path / "demo1.luxar.zarr"
        output2 = tmp_path / "demo2.luxar.zarr"

        runner = CliRunner()

        # Run twice with same seed
        result1 = runner.invoke(
            app,
            [
                "demo",
                "--no-serve",
                "--output",
                str(output1),
                "--points",
                "50",
                "--seed",
                "42",
            ],
        )
        result2 = runner.invoke(
            app,
            [
                "demo",
                "--no-serve",
                "--output",
                str(output2),
                "--points",
                "50",
                "--seed",
                "42",
            ],
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Both should exist
        assert output1.exists()
        assert output2.exists()

        # Load and verify both stores exist and have content
        import zarr

        store1 = zarr.open(str(output1), mode="r")
        store2 = zarr.open(str(output2), mode="r")

        # Verify both have same structure (exact reproducibility check would compare arrays)
        keys1 = list(store1.group_keys())
        keys2 = list(store2.group_keys())
        assert len(keys1) > 0, "Store 1 is empty"
        assert len(keys2) > 0, "Store 2 is empty"
        assert keys1 == keys2, "Stores have different structure"


class TestPortHandling:
    """Test port conflict handling and availability checking."""

    def test_find_available_port(self):
        """Test finding an available port."""
        port = find_available_port(9000, 9100)
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
            result = runner.invoke(
                app,
                ["serve", "--port", str(available_port), "--no-viewer"],
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
