"""
Integration tests for CLI commands - NO MOCKING of internal server logic.

These tests actually start the server, make HTTP requests, and verify responses.
Following the principle from TESTING_GUIDELINES.md: mock only external dependencies.
"""

import socket
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from luxar import Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import ZARR_FORMAT
from luxar.cli.utils import find_available_port
from luxar.utils.scenes import create_lorenz_attractor

#: The served document carrying a node's attributes / group record / array
#: record, for the format Luxar currently writes. Format 3 folds all three into
#: one ``zarr.json`` per node; format 2 keeps them apart. These are URL paths
#: fetched over HTTP by name — the server just serves files, so the test has to
#: ask for the document that actually exists.
_ATTRS_DOC = ".zattrs" if ZARR_FORMAT == 2 else "zarr.json"
_GROUP_DOC = ".zgroup" if ZARR_FORMAT == 2 else "zarr.json"
_ARRAY_DOC = ".zarray" if ZARR_FORMAT == 2 else "zarr.json"


def _node_attrs(payload: dict) -> dict:
    """A node's user attributes from either document shape.

    A ``.zattrs`` IS the attributes object; a ``zarr.json`` is the whole node
    record with attributes nested under ``attributes``, so reading a Luxar attr
    off the top level of the latter always yields ``None``. Mirrors the viewer's
    ``rootAttributes`` in ``types/zarr-documents.ts`` — both sides bypass the
    zarr library here and so both need this unwrap.

    A v3 record whose ``attributes`` is present but not a mapping yields ``{}``,
    matching the TypeScript side (whose ``typeof null === "object"`` check falls
    through to ``?? {}``). Returning ``payload`` there instead would hand back
    ``zarr_format`` and ``node_type`` as though they were the node's attributes.
    """
    if payload.get("zarr_format") == 3:
        attributes = payload.get("attributes")
        return attributes if isinstance(attributes, dict) else {}
    return payload


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
    """A port for this test to bind.

    Deliberately an OS-assigned EPHEMERAL port rather than
    ``find_available_port(8000, ...)``. That helper scans deterministically
    upward from its start port, so under ``pytest -n`` every worker probing at
    the same moment is handed 8000 and they collide — and the collision surfaces
    as ``pytest.fail`` in the ``test_server`` fixture, i.e. a hard red rather
    than a retry. The kernel's ephemeral allocator hands out distinct ports
    instead. (``test_export.py`` already uses this pattern.)

    Still a probe-then-close: the socket is closed so the caller can bind it,
    which every consumer here does — ``test_port_conflict_handling`` binds it
    ITSELF to manufacture a conflict, so the fixture cannot hold it open.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def sample_scene(tmp_path):
    """Create a sample scene for testing."""
    store_path = tmp_path / "test_scene.luxar.zarr"
    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


@pytest.fixture
def test_server(sample_scene, available_port):
    """Start a real test server in a background thread, and stop it afterwards."""
    import asyncio

    import uvicorn

    from luxar.cli.main import create_server_app

    app = create_server_app(str(sample_scene))
    # Build the Server OUTSIDE the thread so teardown has a handle to signal.
    config = uvicorn.Config(
        app, host="127.0.0.1", port=available_port, log_level="error"
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()), daemon=True
    )
    server_thread.start()

    # Poll /health as the readiness signal. The previous version set an Event
    # immediately BEFORE `serve()` (so it said nothing about readiness) and then
    # slept a flat 0.5 s to compensate — dead time on every one of the 11 tests
    # that take this fixture.
    deadline = time.monotonic() + 15.0
    while True:
        try:
            if (
                requests.get(
                    f"http://127.0.0.1:{available_port}/health", timeout=1
                ).status_code
                == 200
            ):
                break
        except requests.exceptions.RequestException:
            pass
        if time.monotonic() > deadline:
            pytest.fail(
                "Server failed to become healthy within 15s — a real regression."
            )
        time.sleep(0.05)

    yield f"http://127.0.0.1:{available_port}"

    # Actually shut down. The thread is a daemon, so a leaked server survived
    # until the process exited and kept its port bound; under `pytest -n` that
    # is one abandoned listener per server per worker.
    server.should_exit = True
    server_thread.join(timeout=10)


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
        response = requests.get(f"{test_server}/{_ATTRS_DOC}")
        assert response.status_code == 200
        data = _node_attrs(response.json())
        assert "format_version" in data
        assert data["format_type"] == "luxar_zarr"
        assert data["type"] == "scene"

    def test_scene_metadata(self, test_server):
        """Test retrieving scene metadata."""
        response = requests.get(f"{test_server}/{_ATTRS_DOC}")
        assert response.status_code == 200
        metadata = _node_attrs(response.json())

        # Verify expected metadata structure
        assert metadata["format_version"] == "0.2"
        assert "scene_dimensions" in metadata

    def test_zarr_group_listing(self, test_server):
        """Test listing zarr groups."""
        response = requests.get(f"{test_server}/{_GROUP_DOC}")
        assert response.status_code == 200
        data = response.json()
        assert "zarr_format" in data

    def test_positions_array_access(self, test_server):
        """Test accessing point positions array."""
        # First, get the scene structure to find point nodes
        response = requests.get(f"{test_server}/{_ATTRS_DOC}")
        assert response.status_code == 200

        # The Lorenz demo creates a node named "LorenzAttractor"
        url = f"{test_server}/LorenzAttractor/positions/{_ARRAY_DOC}"
        response = requests.get(url)
        assert response.status_code == 200, (
            f"Expected 200 for {url}, got {response.status_code}"
        )
        array_meta = response.json()
        assert "shape" in array_meta
        # The formats spell the element type differently — format 2 stores a
        # numpy dtype string (`<u2`), format 3 a `data_type` name (`uint16`) —
        # so accept either KEY, then assert the value against both spellings.
        dtype = array_meta.get("dtype", array_meta.get("data_type"))
        assert dtype is not None, f"no dtype key in {sorted(array_meta)}"
        # Positions default to uint16 per-axis fixed-point (linear_perchannel_u16),
        # decoded to float32 in the viewer; PRECISION / large-extent scenes stay float32.
        assert dtype in ["<u2", ">u2", "uint16", "<f4", ">f4", "float32"]

    def test_cors_headers(self, test_server):
        """Test that local CORS origins are allowed by default."""
        # CORS headers appear when Origin header is present (cross-origin request)
        origin = "http://localhost:5173"
        headers = {"Origin": origin}
        response = requests.get(f"{test_server}/health", headers=headers)
        assert "Access-Control-Allow-Origin" in response.headers
        assert response.headers["Access-Control-Allow-Origin"] == origin
        exposed = {
            header.strip()
            for header in response.headers["Access-Control-Expose-Headers"].split(",")
        }
        assert exposed == {"Content-Range", "Content-Length", "Accept-Ranges", "ETag"}
        # ETag matters specifically: it is not CORS-safelisted, so without it a
        # cross-origin viewer (the documented split-port layout) cannot read it and
        # a zipped store's "archive-etag" identity silently degrades to the
        # one-second granularity of Last-Modified.

    def test_no_cache_header_on_data_responses(self, test_server):
        """Every mutable data response must require browser revalidation."""
        response = requests.get(f"{test_server}/{_ATTRS_DOC}")
        assert response.headers.get("Cache-Control") == "no-cache"

    def test_no_cache_middleware_accepts_start_without_headers(self):
        """A spec-legal minimal response start still receives the policy."""
        from luxar.cli.serving import _NoCacheMiddleware

        async def minimal_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b"ok"})

        response = TestClient(_NoCacheMiddleware(minimal_app)).get("/")
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-cache"

    def test_no_cache_middleware_preserves_existing_policy(self):
        """Applications retain an explicit, more specific cache policy."""
        from luxar.cli.serving import _NoCacheMiddleware

        async def cacheable_app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"cache-control", b"public, max-age=60")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        response = TestClient(_NoCacheMiddleware(cacheable_app)).get("/")
        assert response.headers["Cache-Control"] == "public, max-age=60"

    def test_viewer_shell_revalidates_but_assets_do_not(self, tmp_path):
        """Unhashed shell revalidates; content-hashed assets stay cacheable."""
        from luxar.cli.serving import _build_viewer_app

        viewer = tmp_path / "viewer"
        assets = viewer / "assets"
        assets.mkdir(parents=True)
        (assets / "app.01234567.js").write_text("export {};")
        (viewer / "index.html").write_text("<html><body>shell</body></html>")

        client = TestClient(_build_viewer_app(viewer))

        # Content-hashed asset: immutable URL, must NOT be forced to revalidate.
        asset = client.get("/assets/app.01234567.js")
        assert asset.status_code == 200
        assert asset.headers.get("Cache-Control") != "no-cache"

        # Unhashed shell (both "/" and "/index.html"): replaced in place on
        # rebuild, so it MUST revalidate exactly like mutable dataset chunks.
        for shell_path in ("/", "/index.html"):
            response = client.get(shell_path)
            assert response.status_code == 200
            assert response.headers.get("Cache-Control") == "no-cache"

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
            host,
            port,
            data_url=None,
            open_browser_flag=True,
            cors_origin="local",
            title=None,
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

    def test_directory_listing_marks_zipped_stores_as_datasets(
        self, test_server, sample_scene
    ):
        """A ``.zarr.zip`` is a dataset, not a file to download.

        The viewer reads a zipped store in place over HTTP range requests, so it
        must appear in the listing as ``type: "zarr"`` — otherwise the dataset
        browser cannot offer a perfectly loadable scene, and the only way in is
        to hand-type its URL.
        """
        # `test_server` roots at the scene directory itself, so the archive has
        # to live inside it to appear in the listing at all.
        served_root = Path(sample_scene)
        archive = served_root / "zipped_scene.luxar.zarr.zip"
        plain_zarr = served_root / "plain.zarr"
        plain_zip = served_root / "results.zip"
        empty_file = served_root / "empty.txt"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("zarr.json", '{"zarr_format": 3, "node_type": "group"}')
        plain_zarr.mkdir()
        plain_zip.touch()
        empty_file.touch()

        try:
            response = requests.get(
                f"{test_server}/", headers={"Accept": "application/json"}
            )
            assert response.status_code == 200
            entries = {e["name"]: e["type"] for e in response.json()["entries"]}
            assert entries.get("zipped_scene.luxar.zarr.zip") == "zarr"
            assert entries.get("plain.zarr") == "zarr"
            assert entries.get("results.zip") == "file"

            response = requests.get(f"{test_server}/")
            assert response.status_code == 200
            expected_link = (
                '<a href="zipped_scene.luxar.zarr.zip">zipped_scene.luxar.zarr.zip</a>'
            )
            assert expected_link in response.text
            assert '<a href="plain.zarr/">plain.zarr/</a>' in response.text
            assert '<a href="results.zip">results.zip</a>' in response.text
            assert '<a href="empty.txt">empty.txt</a>' in response.text
        finally:
            archive.unlink(missing_ok=True)
            plain_zip.unlink(missing_ok=True)
            empty_file.unlink(missing_ok=True)
            plain_zarr.rmdir()


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
        assert 9000 <= port <= 9100  # end_port is INCLUSIVE

        # Verify port is actually available. SO_REUSEADDR because the probe's
        # contract is "bindable the way the SERVER binds" (uvicorn →
        # loop.create_server, i.e. reuse_address=True on POSIX) — verifying with a
        # stricter bind tests the wrong thing, and fails outright on a port left
        # in TIME_WAIT by an earlier run (the E2E suite parks one on 9000).
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            pytest.fail(f"Port {port} reported as available but couldn't bind")
        finally:
            sock.close()

    def test_port_conflict_handling(self, available_port, sample_scene):
        """`serve` shifts off an occupied port instead of failing.

        The invocation needs a real dataset path. Without one, `serve` exits on
        "Path required unless using --viewer-only" *before* it resolves a port,
        so the previous `exit_code != 0` assertion passed without ever reaching
        the occupied-port path. `uvicorn.run` is stubbed so the command returns
        instead of blocking, and the port it is handed is what gets asserted.
        """
        from typer.testing import CliRunner

        from luxar.cli import app
        from luxar.cli import main as cli_main

        # listen(), not a bare bind(): the probe binds with SO_REUSEADDR to
        # match uvicorn, and only a LISTENING socket conflicts with that.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", available_port))
        sock.listen(1)

        served_ports: list[int] = []
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            cli_main.uvicorn, "run", lambda *a, **kw: served_ports.append(kw["port"])
        )
        try:
            result = CliRunner().invoke(
                app,
                ["serve", str(sample_scene), "--port", str(available_port)],
                catch_exceptions=True,
            )
        finally:
            monkeypatch.undo()
            sock.close()

        assert result.exit_code == 0, result.stdout
        assert served_ports and served_ports[0] > available_port
        assert f"port {available_port} busy" in result.stdout.lower()


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

    def test_directory_mount_serves_file_byte_ranges(self, tmp_path):
        """Archive files support the exact range contract the viewer requires."""
        from luxar.cli.serving import _build_data_app

        archive = tmp_path / "scene.luxar.zarr.zip"
        archive.write_bytes(b"0123456789")
        client = TestClient(_build_data_app(tmp_path))

        response = client.get(archive.name, headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.headers["Content-Range"] == "bytes 2-5/10"
        assert response.content == b"2345"

        response = client.get(archive.name, headers={"Range": "bytes=20-30"})
        assert response.status_code == 416
        assert response.headers["Content-Range"] == "bytes */10"

    def test_zarr_store_mounted_at_root_hides_siblings(self, sample_scene):
        """Sibling files of a served .zarr store are not exposed over HTTP."""
        from luxar.cli.serving import _build_data_app

        sibling = sample_scene.parent / "secret_sibling.txt"
        sibling.write_text("SECRET")

        client = TestClient(_build_data_app(sample_scene))

        # The store is served AT the root (data URLs carry no name suffix).
        assert client.get(f"/{_GROUP_DOC}").status_code == 200
        # Neither the sibling nor the old parent-mounted URL shape resolves.
        assert client.get("/secret_sibling.txt").status_code == 404
        assert client.get(f"/{sample_scene.name}/{_GROUP_DOC}").status_code == 404

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
