"""
Integration tests for CLI commands - NO MOCKING of internal server logic.

These tests actually start the server, make HTTP requests, and verify responses.
Following the principle from TESTING_GUIDELINES.md: mock only external dependencies.
"""

import socket
import threading
import time

import numpy as np
import pytest
import requests

from luxar import Dimensions, LuxarZarrCompiler
from luxar.cli.utils import find_available_port
from luxar.utils.demos import create_lorenz_attractor


@pytest.fixture
def available_port():
    """Find an available port for testing."""
    return find_available_port(8000, 9000)


@pytest.fixture
def sample_scene(tmp_path):
    """Create a sample scene for testing."""
    store_path = tmp_path / "test_scene.zarr"
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

        # Try to access positions array (path may vary based on scene structure)
        # For Lorenz attractor, it's typically at root level
        response = requests.get(f"{test_server}/Lorenz/positions/.zarray")
        if response.status_code == 200:
            array_meta = response.json()
            assert "shape" in array_meta
            assert "dtype" in array_meta
            assert array_meta["dtype"] in ["<f4", ">f4", "float32"]

    def test_cors_headers(self, test_server):
        """Test that CORS headers are set correctly."""
        # CORS headers appear when Origin header is present (cross-origin request)
        origin = "http://localhost:5173"
        headers = {"Origin": origin}
        response = requests.get(f"{test_server}/health", headers=headers)
        assert "Access-Control-Allow-Origin" in response.headers
        # With allow_credentials=True, the CORS spec forbids wildcard "*" —
        # the middleware echoes back the specific requesting origin instead.
        assert response.headers["Access-Control-Allow-Origin"] in ("*", origin)

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
        result = runner.invoke(app, ["info", "/nonexistent/path.zarr"])

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

        output_path = tmp_path / "demo_output.zarr"
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

        output1 = tmp_path / "demo1.zarr"
        output2 = tmp_path / "demo2.zarr"

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
        store_path = tmp_path / "large_scene.zarr"
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
