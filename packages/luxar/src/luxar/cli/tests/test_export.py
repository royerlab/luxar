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
        """Verify the serve script only imports from Python stdlib."""
        content = _get_serve_script_content("data")
        stdlib_modules = {
            "argparse",
            "http",
            "socket",
            "sys",
            "threading",
            "webbrowser",
            "functools",
            "pathlib",
        }
        import_lines = [
            line.strip()
            for line in content.split("\n")
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        for line in import_lines:
            if line.startswith("from "):
                module = line.split()[1].split(".")[0]
            else:
                module = line.split()[1].split(".")[0]
            assert module in stdlib_modules, f"Non-stdlib import found: {line}"

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
        monkeypatch.setattr(
            export_mod, "get_viewer_dist_path", lambda: viewer_dist
        )
        monkeypatch.setattr(export_mod, "_copy_viewer", lambda dest: None)
        monkeypatch.setattr(export_mod, "_copy_zarr_data", lambda s, d: None)
        monkeypatch.setattr(export_mod, "_generate_readme", lambda o, d: None)
        monkeypatch.setattr(export_mod, "validate_zarr_store", lambda p: (True, None))
        monkeypatch.setattr(
            export_mod,
            "_generate_serve_script",
            lambda out, ddn, title=None: captured.update(title=title),
        )
        src = tmp_path / "my_scene.luxar.zarr"
        src.mkdir()
        export_mod.export_scene(src, tmp_path / "out")
        assert captured["title"] == "my_scene"
