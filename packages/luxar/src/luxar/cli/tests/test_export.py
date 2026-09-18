"""Tests for the luxar export command."""

from __future__ import annotations

import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from luxar import Dimensions, LuxarZarrCompiler
from luxar.cli import app
from luxar.cli import export as export_module
from luxar.cli.export import (
    _copy_viewer,
    _copy_zarr_data,
    _generate_readme,
    _generate_serve_script,
    _get_serve_script_content,
    export_scene,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def runner():
    """CLI test runner fixture."""
    return CliRunner()


@pytest.fixture
def sample_scene(tmp_path: Path) -> Path:
    """Create a minimal zarr scene for testing."""
    store_path = tmp_path / "test_scene.luxar.zarr"
    from luxar.demos import create_lorenz_attractor

    create_lorenz_attractor(store_path, n_points=100, seed=42)
    return store_path


@pytest.fixture
def mock_viewer_dist(tmp_path: Path) -> Path:
    """Create a mock viewer dist directory mimicking the real build output."""
    dist = tmp_path / "mock_viewer_dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!doctype html><html><head>"
        '<script type="module" crossorigin src="./assets/index-abc123.js"></script>'
        '<link rel="stylesheet" crossorigin href="./assets/index-def456.css">'
        "</head><body><canvas></canvas></body></html>"
    )
    assets = dist / "assets"
    assets.mkdir()
    (assets / "index-abc123.js").write_text("// bundled app")
    (assets / "index-def456.css").write_text("body { margin: 0; }")
    (assets / "data-worker-xyz789.js").write_text("// worker")
    wasm = dist / "wasm"
    wasm.mkdir()
    (wasm / "luxar_wasm.js").write_text("// wasm bindings")
    (wasm / "luxar_wasm_bg.wasm").write_bytes(b"\x00wasm")
    # The production build emits this, and `_copy_viewer` refuses a dist
    # without it: an export folder is redistribution, so it must carry the
    # third-party notices. A fixture that omits it models a bundle we should
    # not be able to produce.
    (dist / "THIRD_PARTY_LICENSES.txt").write_text(
        "THIRD-PARTY SOFTWARE NOTICES\n(test fixture)\n"
    )
    return dist


def _patch_viewer(mock_viewer_dist: Path):
    """Return context managers that mock viewer availability."""
    return (
        patch("luxar.cli.export.check_viewer_built", return_value=True),
        patch("luxar.cli.export.get_viewer_dist_path", return_value=mock_viewer_dist),
    )


# ─── TestExportScene ─────────────────────────────────────────────────────────


class TestExportScene:
    """Tests for the export_scene() function."""

    def test_creates_output_structure(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Verify export creates the expected folder structure."""
        output = tmp_path / "export_output"
        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2:
            export_scene(sample_scene, output)

        assert (output / "viewer").is_dir()
        assert (output / "viewer" / "index.html").exists()
        assert (output / "viewer" / "assets").is_dir()
        assert (output / "viewer" / "wasm").is_dir()
        assert (output / "data").is_dir()
        assert (output / "serve.py").exists()
        assert (output / "README.txt").exists()

    def test_preserves_zarr_data(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Verify zarr data is readable after export."""
        output = tmp_path / "export_output"
        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2:
            export_scene(sample_scene, output)

        root = zarr.open_group(output / "data", mode="r")
        assert "LorenzAttractor" in root
        assert root["LorenzAttractor"]["positions"].shape == (100, 3)
        assert root["LorenzAttractor"]["colors"].shape == (100, 3)

    def test_custom_data_dir_name(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Verify custom data directory name works."""
        output = tmp_path / "export_output"
        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2:
            export_scene(sample_scene, output, data_dir_name="my_dataset")

        assert (output / "my_dataset").is_dir()
        root = zarr.open_group(output / "my_dataset", mode="r")
        assert "LorenzAttractor" in root

    def test_fails_on_existing_output(self, sample_scene: Path, tmp_path: Path) -> None:
        """FileExistsError when output exists and overwrite=False."""
        output = tmp_path / "existing"
        output.mkdir()
        # [Python-R3/A-W3] Pin the contract more tightly: the original
        # output content must be PRESERVED when the export raises, AND
        # no partial new content can be written into it. Seed the dir
        # with a marker file, then verify both:
        #   (a) the marker survives intact
        #   (b) no luxar-output files (serve.py, viewer/, etc.) were
        #       created
        marker = output / "preserved.txt"
        marker.write_text("original content")

        with pytest.raises(FileExistsError, match="already exists"):
            export_scene(sample_scene, output)

        # Marker survives intact.
        assert marker.exists(), "FileExistsError path damaged original output"
        assert marker.read_text() == "original content"
        # No partial new content.
        assert not (output / "serve.py").exists()
        assert not (output / "viewer").exists()

    def test_overwrite_replaces_existing(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Verify overwrite=True removes old content and creates new."""
        output = tmp_path / "existing"
        output.mkdir()
        old_file = output / "stale_file.txt"
        old_file.write_text("should be removed")

        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2:
            export_scene(sample_scene, output, overwrite=True)

        assert not old_file.exists()
        assert (output / "serve.py").exists()
        assert (output / "viewer").is_dir()

    def test_missing_notices_does_not_wipe_existing_output(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Viewer preflight must finish before overwrite removes old output."""
        output = tmp_path / "existing"
        output.mkdir()
        sentinel = output / "important.txt"
        sentinel.write_text("user's prior export")
        (mock_viewer_dist / "THIRD_PARTY_LICENSES.txt").unlink()

        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2, pytest.raises(FileNotFoundError, match="THIRD_PARTY_LICENSES"):
            export_scene(sample_scene, output, overwrite=True)

        assert sentinel.read_text() == "user's prior export"
        assert not (output / "viewer").exists()

    def test_fails_on_invalid_zarr(self, tmp_path: Path) -> None:
        """ValueError for non-zarr directory."""
        source = tmp_path / "not_zarr"
        source.mkdir()
        (source / "random.txt").write_text("not zarr")
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="Invalid zarr store"):
            export_scene(source, output)

    def test_fails_on_nonexistent_source(self, tmp_path: Path) -> None:
        """ValueError when source doesn't exist."""
        source = tmp_path / "nonexistent.luxar.zarr"
        output = tmp_path / "output"

        with pytest.raises(ValueError, match="Invalid zarr store"):
            export_scene(source, output)

    # [Python-R5 / A-G3] Permission-denied output directory: when the
    # parent of the output path is not writable, export_scene should
    # surface a clean OSError rather than crash mid-copy with partial
    # state. The viewer-not-built path raises FileNotFoundError BEFORE
    # any write occurs, so this test exercises the post-viewer-check
    # path by mocking the viewer check to True and pointing output
    # into a read-only directory.
    def test_fails_on_readonly_output_parent(
        self,
        sample_scene: Path,
        mock_viewer_dist: Path,
        tmp_path: Path,
    ) -> None:
        readonly_parent = tmp_path / "readonly"
        readonly_parent.mkdir()
        readonly_parent.chmod(0o555)  # read + execute, no write
        try:
            output = readonly_parent / "export_attempt"
            p1, p2 = _patch_viewer(mock_viewer_dist)
            with p1, p2:
                with pytest.raises((PermissionError, OSError)):
                    export_scene(sample_scene, output)
            # The attempted output dir must not have been created.
            assert not output.exists()
        finally:
            # Restore permissions so pytest's tmp_path cleanup works.
            readonly_parent.chmod(0o755)

    def test_fails_when_viewer_not_built(
        self, sample_scene: Path, tmp_path: Path
    ) -> None:
        """FileNotFoundError when viewer dist doesn't exist."""
        output = tmp_path / "output"
        with patch("luxar.cli.export.check_viewer_built", return_value=False):
            with pytest.raises(FileNotFoundError, match="Viewer not built"):
                export_scene(sample_scene, output)

    def test_creates_parent_directories(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Verify nested output paths are created."""
        output = tmp_path / "nested" / "deep" / "export"
        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2:
            export_scene(sample_scene, output)

        assert (output / "serve.py").exists()

    def test_returns_output_path(
        self, sample_scene: Path, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """Verify export_scene returns the output path."""
        output = tmp_path / "export_output"
        p1, p2 = _patch_viewer(mock_viewer_dist)
        with p1, p2:
            result = export_scene(sample_scene, output)
        assert result == output


# ─── TestServeScript ─────────────────────────────────────────────────────────


class TestServeScript:
    """Tests for the generated serve.py script."""

    def test_is_valid_python(self) -> None:
        """Verify the generated script compiles without syntax errors."""
        content = _get_serve_script_content("data")
        compile(content, "serve.py", "exec")

    def test_is_executable(self, tmp_path: Path) -> None:
        """Verify the script file has executable permission bits."""
        _generate_serve_script(tmp_path, "data")
        serve_path = tmp_path / "serve.py"
        assert serve_path.exists()
        mode = serve_path.stat().st_mode
        assert mode & stat.S_IEXEC, "Script should be executable by owner"

    def test_contains_data_dir_name(self) -> None:
        """Verify the data directory name is embedded in the script."""
        content = _get_serve_script_content("my_custom_data")
        assert "my_custom_data" in content

    def test_uses_stdlib_only(self) -> None:
        """Verify the serve script only imports from Python stdlib.

        Checked against `sys.stdlib_module_names` rather than a hand-written
        whitelist, which is what this test used to be: a whitelist has to be
        EXTENDED for every legitimate new import, so the pressure is always to
        add the name and move on -- and an import of `luxar` or `websockets`
        would have been one edit away from passing. The real requirement is
        that the exported folder needs no `pip install` from someone who was
        emailed a zip, and this asserts exactly that.
        """
        import ast

        content = _get_serve_script_content("data")
        imported: set[str] = set()
        for node in ast.walk(ast.parse(content)):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imported.add((node.module or "").split(".")[0])

        assert imported, "parsed no imports at all -- the check would be vacuous"
        # `luxar_qr` is not stdlib, and it is not a pip install either: the
        # exporter COPIES it into the folder beside serve.py (see
        # `_copy_qr_module`), and serve.py imports it inside a try/except so a
        # folder without it still serves, printing URLs and no QR. Allowed by
        # name, not by relaxing the rule -- an import of `luxar` or
        # `websockets` must still fail this.
        shipped = {"luxar_qr"}
        foreign = imported - sys.stdlib_module_names - {"__future__"} - shipped
        assert not foreign, f"Non-stdlib imports found: {sorted(foreign)}"
        for name in shipped:
            assert (Path(export_module.__file__).parent / "_qr.py").exists(), name

    def test_relay_default_follows_the_scene(self) -> None:
        """The relay can be hosted, and whether it IS depends on the scene.

        A folder you double-click must not start listening for anything that
        wants to drive the display -- unless the scene was authored to be
        driven, in which case the alternative is an exhibit operator reading a
        README to make the tablet work. So the default is the scene's own
        `viewer_config.control_panel`, and `--no-control` turns it off.
        """
        content = _get_serve_script_content("data")
        assert "--control" in content
        assert "--control-token" in content
        # The switch is a BooleanOptionalAction, so `--no-control` exists.
        assert "action=argparse.BooleanOptionalAction" in content
        assert "default=HAS_CONTROL_PANEL" in content
        # A scene with no panel keeps the old, quiet behaviour.
        assert "HAS_CONTROL_PANEL = False" in content

    def test_substitution_failure_is_loud(self, tmp_path: Path) -> None:
        """A template that stopped carrying a substituted line must not export.

        Silently passing the template's own default through would produce a
        folder whose serve.py points at a data directory that is not there --
        which looks like a broken scene, not a broken export.
        """
        from luxar.cli import export as export_module

        original = export_module.SERVE_TEMPLATE
        try:
            export_module.SERVE_TEMPLATE = tmp_path / "_export_serve_template.py"
            export_module.SERVE_TEMPLATE.write_text("TITLE_QUERY = ''\n")
            with pytest.raises(RuntimeError, match="DATA_DIR_NAME"):
                _get_serve_script_content("data")
        finally:
            export_module.SERVE_TEMPLATE = original

    def test_has_shebang(self) -> None:
        """Verify the script starts with a proper shebang line."""
        content = _get_serve_script_content("data")
        assert content.startswith("#!/usr/bin/env python3")

    def test_no_cors_headers(self) -> None:
        """The serve script must NOT emit CORS headers: the viewer and its data
        are same-origin, so wildcard CORS would only widen local exposure."""
        content = _get_serve_script_content("data")
        assert "Access-Control-Allow-Origin" not in content

    def test_script_importable(self, tmp_path: Path) -> None:
        """Verify the serve script can be parsed by Python."""
        _generate_serve_script(tmp_path, "data")
        serve_path = tmp_path / "serve.py"
        import ast

        tree = ast.parse(serve_path.read_text())
        # Should have function definitions
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert "main" in func_names
        assert "find_port" in func_names

    def test_serve_script_serves_files(self, tmp_path: Path) -> None:
        """Integration test: start serve.py and verify it serves viewer + data
        files (same-origin, no CORS headers)."""
        # Create a minimal export structure
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        viewer_dir = export_dir / "viewer"
        viewer_dir.mkdir()
        (viewer_dir / "index.html").write_text("<html><body>test</body></html>")
        (viewer_dir / "assets").mkdir()
        (viewer_dir / "assets" / "app.01234567.js").write_text("export {};")
        data_dir = export_dir / "data"
        data_dir.mkdir()
        (data_dir / ".zattrs").write_text('{"type": "scene"}')

        _generate_serve_script(export_dir, "data")

        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        # Start serve.py as a subprocess
        proc = subprocess.Popen(
            [
                sys.executable,
                str(export_dir / "serve.py"),
                "--port",
                str(port),
                "--no-open",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Wait for server to start
            for _ in range(50):
                try:
                    urlopen(f"http://127.0.0.1:{port}/viewer/index.html", timeout=1)
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                pytest.fail("Server did not start within 5 seconds")

            # Verify viewer file is served
            resp = urlopen(f"http://127.0.0.1:{port}/viewer/index.html", timeout=5)
            assert resp.status == 200
            assert b"test" in resp.read()

            # Verify data directory is accessible
            resp = urlopen(f"http://127.0.0.1:{port}/data/.zattrs", timeout=5)
            assert resp.status == 200
            assert b"scene" in resp.read()

            # Verify NO CORS header is emitted (viewer + data are same-origin).
            resp = urlopen(f"http://127.0.0.1:{port}/data/.zattrs", timeout=5)
            cors = resp.headers.get("Access-Control-Allow-Origin")
            assert cors is None, f"Expected no CORS header, got {cors!r}"

            # Verify Cache-Control: no-cache is emitted. Without it, browsers
            # use heuristic freshness (SimpleHTTPRequestHandler sends only
            # Last-Modified) and serve STALE files after the folder is
            # re-exported in place.
            assert resp.headers.get("Cache-Control") == "no-cache"

            # Content-hashed viewer assets embed a build hash in their URL and
            # are immutable, so they must stay cacheable (NOT forced to
            # revalidate).
            resp = urlopen(
                f"http://127.0.0.1:{port}/viewer/assets/app.01234567.js", timeout=5
            )
            assert resp.status == 200
            assert resp.headers.get("Cache-Control") != "no-cache"

            # The unhashed viewer shell is replaced in place on re-export, so it
            # must revalidate exactly like mutable data.
            resp = urlopen(f"http://127.0.0.1:{port}/viewer/index.html", timeout=5)
            assert resp.headers.get("Cache-Control") == "no-cache"

            # A malformed request line must still get a clean 400 — end_headers
            # reads self.path, which is unset before parse_request assigns it
            # (a single-token line is classified HTTP/0.9 and rejected BEFORE
            # self.path is set), so the handler must degrade gracefully instead
            # of raising AttributeError. Pre-fix, that AttributeError aborted the
            # response mid-flight and the client saw an empty/reset reply; a
            # non-empty 400 error page proves it degraded cleanly. (HTTP/0.9
            # responses carry no status line, so we assert on the body.)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
                raw.sendall(b"BOGUS\r\n\r\n")
                chunks = []
                while True:
                    chunk = raw.recv(1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
            reply = b"".join(chunks)
            assert reply, "server crashed on a malformed request (empty reply)"
            assert b"400" in reply, f"expected a clean 400 response, got {reply!r}"
        finally:
            proc.terminate()
            proc.wait(timeout=5)


# ─── TestCopyZarrData ────────────────────────────────────────────────────────


class TestCopyZarrData:
    """Tests for zarr data copying."""

    def test_data_preserved(self, sample_scene: Path, tmp_path: Path) -> None:
        """Verify zarr data is identical after copy."""
        dest = tmp_path / "copied_data"
        _copy_zarr_data(sample_scene, dest)

        # Verify zarr structure
        root = zarr.open_group(dest, mode="r")
        assert "LorenzAttractor" in root
        assert root["LorenzAttractor"]["positions"].shape == (100, 3)

    def test_preserves_metadata(self, sample_scene: Path, tmp_path: Path) -> None:
        """Verify zarr metadata is preserved."""
        dest = tmp_path / "copied_data"
        _copy_zarr_data(sample_scene, dest)

        root = zarr.open_group(dest, mode="r")
        assert root.attrs["type"] == "scene"
        assert "luxar_version" in root.attrs

    def test_complex_hierarchy(self, tmp_path: Path) -> None:
        """Verify a scene with nested groups is copied correctly."""
        source = tmp_path / "complex.luxar.zarr"
        with LuxarZarrCompiler(source) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            group = scene.add_group("Group1")
            group.add_group("SubGroup")
            compiler.write_points(
                "Group1/points",
                np.random.rand(50, 3).astype(np.float32),
            )

        dest = tmp_path / "copied"
        _copy_zarr_data(source, dest)

        root = zarr.open_group(dest, mode="r")
        assert "Group1" in root
        assert "SubGroup" in root["Group1"]
        assert root["Group1"]["points"]["positions"].shape == (50, 3)


# ─── TestCopyViewer ──────────────────────────────────────────────────────────


class TestCopyViewer:
    """Tests for viewer copying."""

    def test_refuses_a_dist_without_third_party_notices(
        self, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """An export folder is redistribution; it must carry its notices.

        MIT / Apache-2.0 / MPL-2.0 / BSD all require their notices to accompany
        a binary redistribution, and an export folder is zipped and handed to
        people who never see this repository. Producing one silently without
        them is the failure this guards, so it must be a refusal rather than a
        warning: the folder otherwise looks complete.
        """
        (mock_viewer_dist / "THIRD_PARTY_LICENSES.txt").unlink()
        dest = tmp_path / "viewer_copy"
        with patch(
            "luxar.cli.export.get_viewer_dist_path", return_value=mock_viewer_dist
        ):
            with pytest.raises(FileNotFoundError, match="THIRD_PARTY_LICENSES"):
                _copy_viewer(dest)
        assert not dest.exists(), "refused export must leave no partial folder"

    def test_copies_the_third_party_notices(
        self, mock_viewer_dist: Path, tmp_path: Path
    ) -> None:
        """And when present, the notices actually travel into the output."""
        dest = tmp_path / "viewer_copy"
        with patch(
            "luxar.cli.export.get_viewer_dist_path", return_value=mock_viewer_dist
        ):
            _copy_viewer(dest)
        assert (dest / "THIRD_PARTY_LICENSES.txt").is_file()

    def test_copies_all_files(self, mock_viewer_dist: Path, tmp_path: Path) -> None:
        """Verify all viewer files are copied."""
        dest = tmp_path / "viewer_copy"
        with patch(
            "luxar.cli.export.get_viewer_dist_path", return_value=mock_viewer_dist
        ):
            _copy_viewer(dest)

        assert (dest / "index.html").exists()
        assert (dest / "assets" / "index-abc123.js").exists()
        assert (dest / "assets" / "index-def456.css").exists()
        assert (dest / "wasm" / "luxar_wasm.js").exists()
        assert (dest / "wasm" / "luxar_wasm_bg.wasm").exists()


# ─── TestReadme ──────────────────────────────────────────────────────────────


class TestReadme:
    """Tests for README generation."""

    def test_contains_key_sections(self, tmp_path: Path) -> None:
        """Verify README has essential sections."""
        _generate_readme(tmp_path, "data")
        content = (tmp_path / "README.txt").read_text()

        assert "Quick Start" in content
        assert "python serve.py" in content
        assert "Requirements" in content
        assert "Python 3" in content
        assert "Folder Structure" in content

    def test_references_data_dir(self, tmp_path: Path) -> None:
        """Verify README mentions the data directory name."""
        _generate_readme(tmp_path, "my_data")
        content = (tmp_path / "README.txt").read_text()
        assert "my_data" in content

    def test_names_both_pages_not_just_index(self, tmp_path: Path) -> None:
        """The file:// warning must cover the touch panel too.

        The folder now carries TWO pages, and the second one is the more
        tempting to double-click: it is small, it is called control.html, and
        it fails in exactly the same way.
        """
        _generate_readme(tmp_path, "data")
        content = (tmp_path / "README.txt").read_text()
        assert "viewer/index.html" in content
        assert "viewer/control.html" in content
        assert "file://" in content

    def test_documents_kiosk_mode_and_its_token(self, tmp_path: Path) -> None:
        """A reader must be able to run the kiosk, and know the token's limits.

        The token travels in the URL, so it is visible in the tablet's address
        bar -- that is fine for a LAN exhibit and worth saying out loud, since
        a reader who mistook it for a password would deploy it as one.
        """
        _generate_readme(tmp_path, "data")
        content = (tmp_path / "README.txt").read_text()
        assert "--control" in content
        assert "--control-token" in content
        assert "--host 0.0.0.0" in content
        assert "off by default" in content.lower()
        assert "address bar" in content


# ─── TestCLIExportCommand ────────────────────────────────────────────────────


class TestCLIExportCommand:
    """Integration tests for the CLI export command."""

    def test_help(self, runner: CliRunner) -> None:
        """Verify --help shows export command info."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.stdout)
        assert "export" in plain.lower()
        assert "--output" in plain
        assert "--overwrite" in plain

    def test_nonexistent_source(self, runner: CliRunner, tmp_path: Path) -> None:
        """Exit code != 0 for nonexistent source."""
        result = runner.invoke(
            app,
            [
                "export",
                str(tmp_path / "nope.luxar.zarr"),
                "-o",
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0

    def test_missing_output_flag(self, runner: CliRunner, sample_scene: Path) -> None:
        """Exit code != 0 when --output is not provided."""
        result = runner.invoke(app, ["export", str(sample_scene)])
        assert result.exit_code != 0

    def test_full_workflow(
        self,
        runner: CliRunner,
        sample_scene: Path,
        mock_viewer_dist: Path,
        tmp_path: Path,
    ) -> None:
        """End-to-end: export via CLI, verify output structure."""
        output = tmp_path / "cli_export"
        with (
            patch("luxar.cli.export.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.export.get_viewer_dist_path",
                return_value=mock_viewer_dist,
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "export",
                    str(sample_scene),
                    "-o",
                    str(output),
                ],
            )

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert (output / "viewer" / "index.html").exists()
        assert (output / "data").is_dir()
        assert (output / "serve.py").exists()
        assert (output / "README.txt").exists()

    def test_existing_output_without_overwrite(
        self,
        runner: CliRunner,
        sample_scene: Path,
        tmp_path: Path,
    ) -> None:
        """Exit code 1 when output exists and --overwrite not set."""
        output = tmp_path / "existing_output"
        output.mkdir()

        result = runner.invoke(
            app,
            ["export", str(sample_scene), "-o", str(output)],
        )
        assert result.exit_code == 1
        assert "already exists" in _strip_ansi(result.stdout)

    def test_existing_output_with_overwrite(
        self,
        runner: CliRunner,
        sample_scene: Path,
        mock_viewer_dist: Path,
        tmp_path: Path,
    ) -> None:
        """Overwrite flag replaces existing output."""
        output = tmp_path / "existing_output"
        output.mkdir()

        with (
            patch("luxar.cli.export.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.export.get_viewer_dist_path",
                return_value=mock_viewer_dist,
            ),
        ):
            result = runner.invoke(
                app,
                ["export", str(sample_scene), "-o", str(output), "--overwrite"],
            )

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert (output / "serve.py").exists()


class TestServeScriptTitle:
    """The exported serve.py bakes a tab title into its viewer URL."""

    def test_title_baked_pre_encoded(self) -> None:
        import ast

        from luxar.cli.export import _get_serve_script_content

        src = _get_serve_script_content("data", "Rivers of Earth & Fjords")
        ast.parse(src)  # generated script stays valid Python
        assert "&title=Rivers%20of%20Earth%20%26%20Fjords" in src

    def test_no_title_leaves_url_unchanged(self) -> None:
        import ast

        from luxar.cli.export import _get_serve_script_content

        src = _get_serve_script_content("data", None)
        ast.parse(src)
        assert "&title=" not in src

    def test_export_scene_derives_title_from_source_name(
        self, tmp_path, monkeypatch
    ) -> None:
        """export_scene passes the source stem through to the serve script."""
        from luxar.cli import export as export_mod

        captured: dict[str, object] = {}
        viewer_dist = tmp_path / "viewer_dist"
        viewer_dist.mkdir()
        (viewer_dist / "THIRD_PARTY_LICENSES.txt").write_text("test notices")
        monkeypatch.setattr(export_mod, "check_viewer_built", lambda: True)
        monkeypatch.setattr(export_mod, "get_viewer_dist_path", lambda: viewer_dist)
        monkeypatch.setattr(export_mod, "_copy_viewer", lambda dest: None)
        monkeypatch.setattr(export_mod, "_copy_zarr_data", lambda s, d: None)
        monkeypatch.setattr(
            export_mod, "_generate_readme", lambda o, d, facts=None: None
        )
        monkeypatch.setattr(export_mod, "validate_zarr_store", lambda p: (True, None))
        monkeypatch.setattr(
            export_mod,
            "_generate_serve_script",
            lambda out, ddn, title=None, facts=None: captured.update(title=title),
        )
        src = tmp_path / "my_scene.luxar.zarr"
        src.mkdir()
        export_mod.export_scene(src, tmp_path / "out")
        assert captured["title"] == "my_scene"


# ---------------------------------------------------------------------------
# Control-panel awareness (2026-09-17)
# ---------------------------------------------------------------------------


def _scene_attrs(control_panel: bool) -> dict:
    """Root attributes shaped like a real compiled scene."""
    attrs: dict = {
        "type": "scene",
        "citation": {"ref": "A Dataset, Someone et al. 2026"},
        "scene_dimensions": {
            "dimensions": [
                {"name": "story", "display": False},
                {"name": "x", "display": True},
                {"name": "y", "display": True},
                {"name": "z", "display": True},
            ]
        },
        "viewer_config": {"title": "A Tour"},
    }
    if control_panel:
        attrs["viewer_config"]["control_panel"] = {
            "chapter_dimension": "story",
            "chapters": {str(i): {"sublabel": f"stop {i}"} for i in range(1, 21)},
        }
    return attrs


def test_scene_facts_read_the_dimensions_from_the_right_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`scene_dimensions.dimensions`, not a top-level `dimensions`.

    The wrong key fails SILENTLY: the lookup finds nothing, no dimensions are
    reported, and the README simply omits those lines. That is exactly how the
    first version of this shipped a README with no dimensions in it.
    """
    monkeypatch.setattr(
        export_module, "read_node_attrs", lambda _p: _scene_attrs(control_panel=True)
    )
    facts = export_module.read_scene_facts(tmp_path)
    assert facts.dimensions == ("story", "x", "y", "z")
    assert facts.displayed == ("x", "y", "z")
    assert facts.title == "A Tour"
    assert facts.citation == "A Dataset, Someone et al. 2026"


def test_scene_facts_detect_a_control_panel_and_count_its_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for has_panel in (True, False):
        monkeypatch.setattr(
            export_module,
            "read_node_attrs",
            lambda _p, h=has_panel: _scene_attrs(control_panel=h),
        )
        facts = export_module.read_scene_facts(tmp_path)
        assert facts.has_control_panel is has_panel
        assert facts.chapter_count == (20 if has_panel else 0)
        assert facts.chapter_dimension == ("story" if has_panel else None)


def test_scene_facts_never_raise_on_an_unreadable_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Introspection is a nicety; a store it cannot read still exports."""

    def boom(_p: Path) -> dict:
        raise OSError("no metadata here")

    monkeypatch.setattr(export_module, "read_node_attrs", boom)
    facts = export_module.read_scene_facts(tmp_path)
    assert facts == export_module.SceneFacts()
    assert facts.has_control_panel is False

    monkeypatch.setattr(export_module, "read_node_attrs", lambda _p: None)
    assert export_module.read_scene_facts(tmp_path).has_control_panel is False


def test_the_serve_script_defaults_follow_the_scene() -> None:
    """A panel scene starts its relay; a plain scene does not.

    The substituted constant is what the exported script reads, so it is
    asserted in the generated TEXT rather than by importing the template.
    """
    panel = export_module._get_serve_script_content(
        "data", None, export_module.SceneFacts(has_control_panel=True)
    )
    plain = export_module._get_serve_script_content(
        "data", None, export_module.SceneFacts(has_control_panel=False)
    )
    assert "HAS_CONTROL_PANEL = True" in panel
    assert "HAS_CONTROL_PANEL = False" in plain
    # No facts at all is the conservative case, not an error.
    assert "HAS_CONTROL_PANEL = False" in export_module._get_serve_script_content(
        "data"
    )


def test_the_readme_leads_with_kiosk_mode_only_for_a_panel_scene(
    tmp_path: Path,
) -> None:
    panel_dir, plain_dir = tmp_path / "panel", tmp_path / "plain"
    panel_dir.mkdir()
    plain_dir.mkdir()
    export_module._generate_readme(
        panel_dir,
        "data",
        export_module.SceneFacts(
            title="A Tour",
            has_control_panel=True,
            chapter_dimension="story",
            chapter_count=20,
            dimensions=("story", "x", "y", "z"),
            displayed=("x", "y", "z"),
            citation="A Dataset",
        ),
    )
    export_module._generate_readme(plain_dir, "data", export_module.SceneFacts())

    panel = (panel_dir / "README.txt").read_text()
    plain = (plain_dir / "README.txt").read_text()

    assert "THIS SCENE HAS A CONTROL PANEL" in panel
    assert "THIS SCENE HAS A CONTROL PANEL" not in plain
    assert "declares no control panel" in plain
    # The scene summary only appears when there is something to say.
    assert "About this scene" in panel and "20 stops along 'story'" in panel
    assert "Steppable    story" in panel and "Displayed    x, y, z" in panel
    assert "About this scene" not in plain
    # Both name the QR encoder they ship, so a reader knows what the file is.
    for text in (panel, plain):
        assert "luxar_qr.py" in text


def test_the_export_ships_the_qr_encoder_beside_serve(tmp_path: Path) -> None:
    """serve.py imports `luxar_qr` by name, so the copy must use that name."""
    out = tmp_path / "out"
    out.mkdir()
    export_module._copy_qr_module(out)
    shipped = out / "luxar_qr.py"
    assert shipped.exists()
    assert shipped.read_text() == export_module.QR_MODULE_SOURCE.read_text()
    template = export_module.SERVE_TEMPLATE.read_text()
    assert "from luxar_qr import" in template
    assert f"from {shipped.stem} import" in template
