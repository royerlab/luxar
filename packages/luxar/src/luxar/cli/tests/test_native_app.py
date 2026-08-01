"""Tests for `luxar export --native` bundle producers.

The tests use a fake launcher binary in place of the real Go-built one so
they pass even on machines that have not run `make build-launchers`. The
two real test concerns are exercised regardless: bundle layout (paths,
permissions, plist contents) and the integration with the existing
viewer/zarr inputs.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import zarr
from typer.testing import CliRunner

from luxar.cli import app
from luxar.cli.native_app import (
    SUPPORTED_PLATFORMS,
    LauncherNotBuiltError,
    _macos_readme,
    bundle_linux_folder,
    bundle_macos_app,
    get_launcher_path,
    validate_bundle_name,
    zip_macos_app,
)


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def sample_scene(tmp_path: Path) -> Path:
    from luxar.demos import create_lorenz_attractor

    store_path = tmp_path / "demo_scene.luxar.zarr"
    create_lorenz_attractor(store_path, n_points=50, seed=1)
    return store_path


@pytest.fixture
def fake_viewer_dist(tmp_path: Path) -> Path:
    """Minimal viewer dist mimicking the Vite build output."""
    dist = tmp_path / "fake_dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html></html>")
    (dist / "assets").mkdir()
    (dist / "assets" / "index.js").write_text("// app")
    (dist / "wasm").mkdir()
    (dist / "wasm" / "luxar_wasm_bg.wasm").write_bytes(b"\x00wasm")
    return dist


@pytest.fixture
def fake_launchers_dir(tmp_path: Path) -> Path:
    """A _launchers directory populated with placeholder binaries.

    Each "binary" is a tiny shell script — enough for permissions and
    file-existence checks; no execution is attempted in unit tests.
    """
    launchers = tmp_path / "fake_launchers"
    launchers.mkdir()
    for name in ("darwin-universal", "linux-amd64", "linux-arm64"):
        path = launchers / name
        path.write_bytes(b"#!/bin/sh\necho fake\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return launchers


def _patch_launchers(fake_launchers_dir: Path):
    return patch("luxar.cli.native_app.LAUNCHERS_DIR", fake_launchers_dir)


# ─── Launcher resolution ─────────────────────────────────────────────────────


class TestGetLauncherPath:
    def test_returns_path_for_supported_platform(
        self, fake_launchers_dir: Path
    ) -> None:
        with _patch_launchers(fake_launchers_dir):
            for plat in SUPPORTED_PLATFORMS:
                assert get_launcher_path(plat).exists()

    def test_raises_for_unknown_platform(self) -> None:
        with pytest.raises(ValueError, match="Unknown native platform"):
            get_launcher_path("solaris")

    def test_raises_when_binary_missing(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with patch("luxar.cli.native_app.LAUNCHERS_DIR", empty_dir):
            with pytest.raises(LauncherNotBuiltError, match="make build-launchers"):
                get_launcher_path("macos")

    # [Python-R6 / A-W4] Pin that returned launcher paths point at FILES
    # (not directories) and that the parent is the LAUNCHERS_DIR. The
    # existing test only checked `.exists()` — a regression that pointed
    # at the parent directory (which also "exists") would slip past.
    def test_returned_path_is_file_in_launchers_dir(
        self, fake_launchers_dir: Path
    ) -> None:
        with _patch_launchers(fake_launchers_dir):
            for plat in SUPPORTED_PLATFORMS:
                path = get_launcher_path(plat)
                assert path.is_file(), f"{plat} launcher path is not a file: {path}"
                assert path.parent == fake_launchers_dir, (
                    f"{plat} launcher should be in LAUNCHERS_DIR, got parent "
                    f"{path.parent}"
                )


# ─── macOS .app bundle ───────────────────────────────────────────────────────


class TestBundleMacosApp:
    def test_creates_canonical_layout(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )

        assert app_path == output / "MyScene.app"
        assert (app_path / "Contents" / "Info.plist").is_file()
        launcher = app_path / "Contents" / "MacOS" / "launcher"
        assert launcher.is_file()
        assert launcher.stat().st_mode & stat.S_IEXEC
        assert (app_path / "Contents" / "Resources" / "viewer" / "index.html").is_file()
        # zarr data is preserved and re-readable
        assert (app_path / "Contents" / "Resources" / "data" / ".zgroup").is_file()
        zarr.open_group(app_path / "Contents" / "Resources" / "data", mode="r")

    def test_info_plist_is_valid(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="My Scene 1",
            )

        with (app_path / "Contents" / "Info.plist").open("rb") as f:
            plist = plistlib.load(f)
        assert plist["CFBundleExecutable"] == "launcher"
        assert plist["CFBundleName"] == "My Scene 1"
        assert plist["CFBundlePackageType"] == "APPL"
        # Bundle id is slugified and namespaced
        assert plist["CFBundleIdentifier"] == "org.czbiohub.luxar.my-scene-1"
        # The bundle is intentionally NOT LSUIElement: we want the app to
        # show in Dock + Cmd+Tab so users can Cmd+Q it explicitly.
        assert "LSUIElement" not in plist

    def test_icon_assets_are_distributed_with_package(self) -> None:
        """The committed icon assets live INSIDE the Python package so
        they ride along with `pip install luxar`. If this assertion ever
        starts failing, distribution is broken and silently iconless
        bundles will ship — exactly the regression the asset relocation
        was meant to prevent. This test purposely does NOT skip when the
        files are missing.
        """
        from luxar.cli.native_app import _ASSETS_DIR, LOGO_ICNS, LOGO_PNG

        assert _ASSETS_DIR.is_dir(), (
            f"Asset dir {_ASSETS_DIR} is missing — see README in that "
            "directory for regeneration instructions."
        )
        assert LOGO_PNG.is_file(), f"Missing committed asset: {LOGO_PNG}"
        assert LOGO_ICNS.is_file(), f"Missing committed asset: {LOGO_ICNS}"

    def test_icon_present_in_macos_bundle(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """The bundler copies the committed AppIcon.icns into the .app
        and the plist references it. Asserts the full chain: source
        asset present → copied into bundle → referenced from plist.
        """
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="Icony",
            )
        assert (app_path / "Contents" / "Resources" / "AppIcon.icns").is_file()
        with (app_path / "Contents" / "Info.plist").open("rb") as f:
            plist = plistlib.load(f)
        assert plist["CFBundleIconFile"] == "AppIcon"

    def test_linux_bundle_includes_icon(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            folder = bundle_linux_folder(
                arch="amd64",
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="Icony",
            )
        assert (folder / "Icony.png").is_file()


# ─── Linux portable folder ───────────────────────────────────────────────────


class TestBundleLinuxFolder:
    def test_creates_portable_layout(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            folder = bundle_linux_folder(
                arch="amd64",
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )

        assert folder == output / "MyScene-linux-amd64"
        launcher = folder / "luxar-launcher"
        assert launcher.is_file()
        assert launcher.stat().st_mode & stat.S_IEXEC
        assert (folder / "viewer" / "index.html").is_file()
        assert (folder / "data" / ".zgroup").is_file()
        assert "Quick start" in (folder / "README.txt").read_text()


# ─── Atomic staging (issue #687) ─────────────────────────────────────────────


class TestBundlerAtomicStaging:
    """A produced bundle must be COMPLETE or ABSENT: an interrupted/failed
    copy must never leave a partial bundle at the final path, nor a leftover
    `.tmp_*` staging dir (issue #687)."""

    @staticmethod
    def _fail_on_second_copytree():
        """Return a side_effect that runs the first copytree (viewer) for real
        and raises OSError on the second (the data copy)."""
        real_copytree = shutil.copytree
        calls = {"n": 0}

        def side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_copytree(*args, **kwargs)
            raise OSError("simulated ENOSPC during data copy")

        return side_effect

    def test_macos_partial_failure_leaves_no_output(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            with patch(
                "luxar.cli.native_app.shutil.copytree",
                side_effect=self._fail_on_second_copytree(),
            ):
                with pytest.raises(OSError, match="simulated ENOSPC"):
                    bundle_macos_app(
                        viewer_dist=fake_viewer_dist,
                        zarr_data=sample_scene,
                        output=output,
                        app_name="MyScene",
                    )
        # The final bundle must not exist.
        assert not (output / "MyScene.app").exists()
        # No leftover staging dir.
        assert not list(output.glob(".tmp_*")), "staging dir was not cleaned up"

    def test_linux_partial_failure_leaves_no_output(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            with patch(
                "luxar.cli.native_app.shutil.copytree",
                side_effect=self._fail_on_second_copytree(),
            ):
                with pytest.raises(OSError, match="simulated ENOSPC"):
                    bundle_linux_folder(
                        arch="amd64",
                        viewer_dist=fake_viewer_dist,
                        zarr_data=sample_scene,
                        output=output,
                        app_name="MyScene",
                    )
        assert not (output / "MyScene-linux-amd64").exists()
        assert not list(output.glob(".tmp_*")), "staging dir was not cleaned up"

    def test_macos_happy_path_leaves_no_staging_dir(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
        assert app_path.is_dir()
        assert (app_path / "Contents" / "MacOS" / "launcher").is_file()
        assert not list(output.glob(".tmp_*")), "staging dir left behind on success"

    def test_swap_failure_over_regular_file_leaves_no_staging_dir(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A pre-existing regular FILE at the final bundle path makes the swap
        phase fail (os.replace of a directory onto a file raises). The staging
        dir must still be cleaned up, and the file must survive untouched."""
        output = tmp_path / "out"
        output.mkdir()
        (output / "MyScene-linux-amd64").write_text("x")
        with _patch_launchers(fake_launchers_dir):
            with pytest.raises(OSError):
                bundle_linux_folder(
                    arch="amd64",
                    viewer_dist=fake_viewer_dist,
                    zarr_data=sample_scene,
                    output=output,
                    app_name="MyScene",
                )
        # No leftover staging dir despite the swap-phase failure.
        assert not list(output.glob(".tmp_*")), "staging dir leaked on swap failure"
        # The pre-existing file is untouched — a swap failure never destroys
        # pre-existing data.
        assert (output / "MyScene-linux-amd64").read_text() == "x"

    def test_keyboard_interrupt_cleans_up_staging(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A KeyboardInterrupt (Ctrl-C) mid-build must still clean up the
        staging dir and leave no output — pins the `except BaseException`
        (a weaker `except Exception` would let this leak)."""
        real_copytree = shutil.copytree
        calls = {"n": 0}

        def side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return real_copytree(*args, **kwargs)
            raise KeyboardInterrupt

        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            with patch("luxar.cli.native_app.shutil.copytree", side_effect=side_effect):
                with pytest.raises(KeyboardInterrupt):
                    bundle_linux_folder(
                        arch="amd64",
                        viewer_dist=fake_viewer_dist,
                        zarr_data=sample_scene,
                        output=output,
                        app_name="MyScene",
                    )
        assert not (output / "MyScene-linux-amd64").exists()
        assert not list(output.glob(".tmp_*")), "staging dir leaked on Ctrl-C"

    def test_zip_macos_app_partial_failure_leaves_no_archive(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """An interrupted zip (ENOSPC mid-stream) must leave NO archive and no
        temp behind — a truncated-but-openable .zip is exactly the #687 class."""
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
        archive_path = app_path.with_suffix(".app.zip")
        # Force the zipfile fallback (a macOS host would otherwise pick the
        # ditto branch, which never calls copyfileobj) and make the stream
        # fail partway.
        with (
            patch("luxar.cli.native_app.shutil.which", return_value=None),
            patch(
                "luxar.cli.native_app.shutil.copyfileobj",
                side_effect=OSError("simulated ENOSPC"),
            ),
        ):
            with pytest.raises(OSError, match="simulated ENOSPC"):
                zip_macos_app(app_path)
        assert not archive_path.exists(), "partial .app.zip left behind"
        assert not list(output.glob(".tmp_*.app.zip*")), "temp archive left behind"

    def test_overwrite_replaces_prior_bundle(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Rebuilding onto an existing bundle replaces it wholesale (no merge
        of stale files) and leaves no staging or backup dirs behind."""
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            folder = bundle_linux_folder(
                arch="amd64",
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
            (folder / "stale.txt").write_text("from the previous export")
            bundle_linux_folder(
                arch="amd64",
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
        assert (folder / "luxar-launcher").is_file()
        assert not (folder / "stale.txt").exists(), (
            "old bundle was merged, not replaced"
        )
        assert not list(output.glob(".tmp_*")), "staging/backup dir left behind"

    def test_swap_phase_failure_restores_prior_bundle(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """If the final swap fails while overwriting, the prior bundle is
        moved back into place — an overwrite never loses the old bundle."""
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            folder = bundle_linux_folder(
                arch="amd64",
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
            (folder / "marker.txt").write_text("prior bundle")

            real_replace = os.replace
            state = {"failed": False}

            def flaky_replace(src, dst, *args, **kwargs):
                # Fail exactly once, on the swap ONTO the final path (the
                # backup rename and the restore both target other paths).
                if not state["failed"] and Path(dst) == folder:
                    state["failed"] = True
                    raise OSError("simulated failure during swap")
                return real_replace(src, dst, *args, **kwargs)

            with patch("luxar.cli.native_app.os.replace", side_effect=flaky_replace):
                with pytest.raises(OSError, match="simulated failure during swap"):
                    bundle_linux_folder(
                        arch="amd64",
                        viewer_dist=fake_viewer_dist,
                        zarr_data=sample_scene,
                        output=output,
                        app_name="MyScene",
                    )
        # The prior bundle is back, complete, at the final path.
        assert (folder / "marker.txt").read_text() == "prior bundle"
        assert (folder / "luxar-launcher").is_file()
        assert not list(output.glob(".tmp_*")), "staging/backup dir left behind"

    @pytest.mark.skipif(
        os.geteuid() == 0, reason="permission bits are not enforced for root"
    )
    def test_zip_fallback_raises_on_unreadable_subdir(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """An unreadable subdirectory must fail the zip loudly — os.walk's
        default is to skip it silently, publishing a truncated archive."""
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
        locked = app_path / "Contents" / "Resources" / "viewer" / "assets"
        locked.chmod(0)
        try:
            with patch("luxar.cli.native_app.shutil.which", return_value=None):
                with pytest.raises(OSError):
                    zip_macos_app(app_path)
        finally:
            locked.chmod(0o755)
        assert not app_path.with_suffix(".app.zip").exists()
        assert not list(output.glob(".tmp_*.app.zip*")), "temp archive left behind"

    def test_zip_macos_app_happy_path_leaves_no_temp(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            app_path = bundle_macos_app(
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
        archive_path = zip_macos_app(app_path)
        assert archive_path.is_file()
        assert archive_path == app_path.with_suffix(".app.zip")
        assert not list(output.glob(".tmp_*.app.zip*")), "temp archive left behind"

    def test_linux_happy_path_leaves_no_staging_dir(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        with _patch_launchers(fake_launchers_dir):
            folder = bundle_linux_folder(
                arch="amd64",
                viewer_dist=fake_viewer_dist,
                zarr_data=sample_scene,
                output=output,
                app_name="MyScene",
            )
        assert folder.is_dir()
        assert (folder / "luxar-launcher").is_file()
        assert not list(output.glob(".tmp_*")), "staging dir left behind on success"


# ─── CLI integration ─────────────────────────────────────────────────────────


class TestCLINativeFlag:
    def test_help_advertises_native(self, runner: CliRunner) -> None:
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        plain = _strip_ansi(result.stdout)
        assert "--native" in plain
        assert "macos" in plain

    def test_unknown_platform_rejected(
        self, runner: CliRunner, sample_scene: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            [
                "export",
                str(sample_scene),
                "-o",
                str(tmp_path / "out"),
                "--native",
                "solaris",
            ],
        )
        assert result.exit_code == 1
        assert "Unknown --native platform" in _strip_ansi(result.stdout)

    def test_full_native_export(
        self,
        runner: CliRunner,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "native_out"
        with (
            patch("luxar.cli.main.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.utils.get_viewer_dist_path", return_value=fake_viewer_dist
            ),
            _patch_launchers(fake_launchers_dir),
        ):
            result = runner.invoke(
                app,
                [
                    "export",
                    str(sample_scene),
                    "-o",
                    str(output),
                    "--native",
                    "macos,linux-amd64,linux-arm64",
                    "--name",
                    "Demo",
                ],
            )

        assert result.exit_code == 0, _strip_ansi(result.stdout)
        assert (output / "Demo.app" / "Contents" / "MacOS" / "launcher").is_file()
        assert (output / "Demo-linux-amd64" / "luxar-launcher").is_file()
        assert (output / "Demo-linux-arm64" / "luxar-launcher").is_file()

    def test_missing_launcher_binary_does_not_wipe_existing_output(
        self,
        runner: CliRunner,
        sample_scene: Path,
        fake_viewer_dist: Path,
        tmp_path: Path,
    ) -> None:
        """If a requested launcher is missing, --overwrite must not destroy
        the user's prior export before discovering the failure."""
        output = tmp_path / "native_out"
        output.mkdir()
        sentinel = output / "important.txt"
        sentinel.write_text("user's prior export — must not be deleted")

        # _launchers dir with NO darwin-universal binary
        empty_launchers = tmp_path / "no_binaries"
        empty_launchers.mkdir()

        with (
            patch("luxar.cli.main.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.utils.get_viewer_dist_path", return_value=fake_viewer_dist
            ),
            patch("luxar.cli.native_app.LAUNCHERS_DIR", empty_launchers),
        ):
            result = runner.invoke(
                app,
                [
                    "export",
                    str(sample_scene),
                    "-o",
                    str(output),
                    "--native",
                    "macos",
                    "--overwrite",
                ],
            )
        assert result.exit_code == 1
        assert "make build-launchers" in _strip_ansi(result.stdout)
        # The important sentinel must still exist — pre-validation runs
        # before rmtree.
        assert sentinel.is_file(), "rmtree fired before launcher check (regression)"

    def test_name_defaults_to_zarr_stem(
        self,
        runner: CliRunner,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """With no --name, the bundle name defaults to source.stem (the
        zarr filename without the `.zarr` extension)."""
        output = tmp_path / "native_out"
        with (
            patch("luxar.cli.main.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.utils.get_viewer_dist_path", return_value=fake_viewer_dist
            ),
            _patch_launchers(fake_launchers_dir),
        ):
            result = runner.invoke(
                app,
                [
                    "export",
                    str(sample_scene),
                    "-o",
                    str(output),
                    "--native",
                    "macos",
                ],
            )
        assert result.exit_code == 0, _strip_ansi(result.stdout)
        # sample_scene fixture creates demo_scene.zarr → stem "demo_scene"
        assert (output / "demo_scene.app").is_dir()


# ─── Bundle name validation (issue #686) ─────────────────────────────────────


class TestValidateBundleName:
    @pytest.mark.parametrize(
        "bad",
        [
            "../../escaped",
            "foo/bar",
            "foo\\bar",
            "/abs/path",
            "..",
            ".",
            "",
            "   ",
            "foo\x00bar",
            "foo\tbar",
            "foo\nbar",
            "foo\x7fbar",  # DEL
            "foo\x85bar",  # C1 control (NEL)
        ],
    )
    def test_rejects_unsafe_names(self, bad: str) -> None:
        with pytest.raises(ValueError):
            validate_bundle_name(bad)

    @pytest.mark.parametrize(
        "good",
        ["MyScene", "My Scene 2", "Scéne", "My Scéne 2", "demo_scene", "a.b.c"],
    )
    def test_accepts_safe_names(self, good: str) -> None:
        # Returned unchanged when valid.
        assert validate_bundle_name(good) == good


class TestBundlerTraversalRejected:
    """Defense-in-depth: bundlers reject traversal names directly and
    produce NOTHING outside the requested output directory (issue #686)."""

    def test_linux_folder_rejects_traversal(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        before = sorted(tmp_path.rglob("*"))
        with _patch_launchers(fake_launchers_dir):
            with pytest.raises(ValueError):
                bundle_linux_folder(
                    arch="amd64",
                    viewer_dist=fake_viewer_dist,
                    zarr_data=sample_scene,
                    output=output,
                    app_name="../../escaped",
                )
        # No stray files created anywhere under tmp_path.
        assert sorted(tmp_path.rglob("*")) == before
        assert not (tmp_path.parent / "escaped-linux-amd64").exists()

    def test_macos_app_rejects_traversal(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        before = sorted(tmp_path.rglob("*"))
        with _patch_launchers(fake_launchers_dir):
            with pytest.raises(ValueError):
                bundle_macos_app(
                    viewer_dist=fake_viewer_dist,
                    zarr_data=sample_scene,
                    output=output,
                    app_name="../../escaped",
                )
        assert sorted(tmp_path.rglob("*")) == before
        assert not (tmp_path.parent / "escaped.app").exists()
        # The bundler also writes a sibling README next to the .app; a
        # README-only escape must be caught too.
        assert not (tmp_path.parent / "escaped-README.txt").exists()


class TestBundlerSymlinkEscapeRejected:
    """A valid bundle name whose target path is a symlink pointing outside
    the output directory must still be rejected by the post-resolve
    containment check — the guard `validate_bundle_name` alone cannot see.
    """

    def test_macos_app_rejects_symlink_escape(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (output / "MyScene.app").symlink_to(outside, target_is_directory=True)
        with _patch_launchers(fake_launchers_dir):
            with pytest.raises(ValueError, match="escapes the output directory"):
                bundle_macos_app(
                    viewer_dist=fake_viewer_dist,
                    zarr_data=sample_scene,
                    output=output,
                    app_name="MyScene",
                )

    def test_linux_folder_rejects_symlink_escape(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        output = tmp_path / "out"
        output.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (output / "MyScene-linux-amd64").symlink_to(outside, target_is_directory=True)
        with _patch_launchers(fake_launchers_dir):
            with pytest.raises(ValueError, match="escapes the output directory"):
                bundle_linux_folder(
                    arch="amd64",
                    viewer_dist=fake_viewer_dist,
                    zarr_data=sample_scene,
                    output=output,
                    app_name="MyScene",
                )

    def test_macos_app_rejects_readme_symlink_escape(
        self,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A pre-planted symlink at the sibling README path must not
        redirect the README write outside the output directory."""
        output = tmp_path / "out"
        output.mkdir()
        outside_file = tmp_path / "outside.txt"
        (output / "MyScene-README.txt").symlink_to(outside_file)
        with _patch_launchers(fake_launchers_dir):
            with pytest.raises(ValueError, match="escapes the output directory"):
                bundle_macos_app(
                    viewer_dist=fake_viewer_dist,
                    zarr_data=sample_scene,
                    output=output,
                    app_name="MyScene",
                )
        assert not outside_file.exists()


class TestMacosReadmeQuoting:
    """The macOS README's copy-paste Terminal commands must stay valid shell
    for any name `validate_bundle_name` accepts, including apostrophes."""

    def test_apostrophe_name_produces_valid_shell(self) -> None:
        readme = _macos_readme("O'Brien")
        # Each command line must shlex.split back to exactly two tokens, the
        # second being the bundle path — unbalanced quoting would raise or
        # split wrong.
        for cmd, verb in (("xattr", "xattr -cr"), ("open", "open")):
            line = next(
                ln.strip() for ln in readme.splitlines() if ln.strip().startswith(verb)
            )
            tokens = shlex.split(line)
            assert tokens[-1] == "./O'Brien.app", (cmd, tokens)

    def test_plain_name_is_unquoted(self) -> None:
        readme = _macos_readme("MyScene")
        assert "xattr -cr ./MyScene.app" in readme
        assert "open ./MyScene.app" in readme

    def test_dash_leading_name_is_not_parsed_as_an_option(self) -> None:
        # A dash-leading name is accepted by validate_bundle_name; the `./`
        # prefix must keep xattr/open from parsing the path as a flag.
        readme = _macos_readme("-R")
        for verb in ("xattr -cr", "open"):
            line = next(
                ln.strip() for ln in readme.splitlines() if ln.strip().startswith(verb)
            )
            tokens = shlex.split(line)
            assert tokens[-1] == "./-R.app", (verb, tokens)
            assert not tokens[-1].startswith("-"), (verb, tokens)


class TestCLITraversalNameRejected:
    def test_traversal_name_rejected_before_rmtree(
        self,
        runner: CliRunner,
        sample_scene: Path,
        fake_viewer_dist: Path,
        fake_launchers_dir: Path,
        tmp_path: Path,
    ) -> None:
        """A traversal --name must be rejected BEFORE the output directory
        is wiped: a pre-existing sentinel file survives even with
        --overwrite."""
        output = tmp_path / "native_out"
        output.mkdir()
        sentinel = output / "important.txt"
        sentinel.write_text("user's prior export — must not be deleted")

        with (
            patch("luxar.cli.main.check_viewer_built", return_value=True),
            patch(
                "luxar.cli.utils.get_viewer_dist_path", return_value=fake_viewer_dist
            ),
            _patch_launchers(fake_launchers_dir),
        ):
            result = runner.invoke(
                app,
                [
                    "export",
                    str(sample_scene),
                    "-o",
                    str(output),
                    "--native",
                    "macos",
                    "--name",
                    "../../escaped",
                    "--overwrite",
                ],
            )

        assert result.exit_code == 1
        # Nothing escaped the output directory. The pre-fix escape target
        # for `output = tmp_path / "native_out"` + `../../escaped` resolves
        # to `tmp_path.parent / "escaped.app"`, so assert THAT location.
        assert not (tmp_path.parent / "escaped.app").exists()
        # Validation ran before rmtree — the sentinel survives.
        assert sentinel.is_file(), (
            "rmtree fired before bundle-name validation (regression)"
        )
