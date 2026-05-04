"""Native bundle producers for ``luxar export --native``.

Wraps a copy of the viewer dist + a zarr scene around a precompiled Go
launcher (built via ``make build-launchers``) so end users can double-click
the result instead of running a Python server.

Two bundle formats are supported in the v1 prototype:

- **macOS .app**: standard ``<name>.app/Contents/{Info.plist, MacOS, Resources}``
  layout. The launcher lives at ``Contents/MacOS/launcher``; viewer and data
  at ``Contents/Resources/{viewer,data}``.
- **Linux portable folder**: ``<name>-linux-<arch>/`` containing the launcher
  binary alongside ``viewer/`` and ``data/`` plus a small README. Most
  desktop file managers happily run the binary on double-click; CLI users
  can run ``./luxar-launcher``.

Windows + AppImage are intentionally out of scope for this prototype.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

from arbol import aprint, asection

# Both directories live INSIDE the Python package so they ride along in
# any wheel/sdist build (no out-of-tree path resolution that breaks for
# `pip install`). Binaries land in `_launchers/` via `make build-launchers`;
# icons land in `_launcher_assets/` via build_logo.py + build_icons.sh.
LAUNCHERS_DIR = Path(__file__).parent / "_launchers"
_ASSETS_DIR = Path(__file__).parent / "_launcher_assets"
LOGO_PNG = _ASSETS_DIR / "luxar-logo.png"
LOGO_ICNS = _ASSETS_DIR / "AppIcon.icns"

# Map between user-facing --native names and the on-disk binary file names
# that the Makefile produces.
PLATFORM_BINARIES: dict[str, str] = {
    "macos": "darwin-universal",
    "linux-amd64": "linux-amd64",
    "linux-arm64": "linux-arm64",
}

SUPPORTED_PLATFORMS: tuple[str, ...] = tuple(PLATFORM_BINARIES.keys())


class LauncherNotBuiltError(FileNotFoundError):
    """Raised when a launcher binary is missing for the requested platform."""


def get_launcher_path(platform_name: str) -> Path:
    """Return the path to the launcher binary for ``platform_name``.

    Args:
        platform_name: One of ``SUPPORTED_PLATFORMS``.

    Raises:
        ValueError: If ``platform_name`` is not recognised.
        LauncherNotBuiltError: If the binary has not been built yet.
    """
    if platform_name not in PLATFORM_BINARIES:
        raise ValueError(
            f"Unknown native platform: {platform_name!r}. "
            f"Choices: {', '.join(SUPPORTED_PLATFORMS)}"
        )
    binary = LAUNCHERS_DIR / PLATFORM_BINARIES[platform_name]
    if not binary.exists():
        raise LauncherNotBuiltError(
            f"Native launcher not found at {binary}.\n"
            "Run: make build-launchers"
        )
    return binary


def bundle_macos_app(
    *,
    viewer_dist: Path,
    zarr_data: Path,
    output: Path,
    app_name: str,
) -> Path:
    """Build a macOS ``.app`` bundle at ``output / f"{app_name}.app"``.

    The launcher binary is the Go-compiled ``darwin-universal`` artifact;
    viewer + data live under ``Contents/Resources``.
    """
    launcher = get_launcher_path("macos")
    app_path = output / f"{app_name}.app"
    contents = app_path / "Contents"
    macos_dir = contents / "MacOS"
    resources = contents / "Resources"

    with asection(f"Building macOS .app bundle ({app_name}.app)"):
        macos_dir.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)

        bundled_launcher = macos_dir / "launcher"
        shutil.copy2(launcher, bundled_launcher)
        bundled_launcher.chmod(
            bundled_launcher.stat().st_mode
            | stat.S_IEXEC
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

        shutil.copytree(viewer_dist, resources / "viewer", dirs_exist_ok=True)
        shutil.copytree(zarr_data, resources / "data", dirs_exist_ok=True)

        icon_present = LOGO_ICNS.is_file()
        if icon_present:
            shutil.copy2(LOGO_ICNS, resources / "AppIcon.icns")
            aprint(f"Copied icon to {resources / 'AppIcon.icns'}")
        else:
            aprint(
                f"⚠️  {LOGO_ICNS} not found — bundle will use the default app icon. "
                "Run packages/luxar-launcher/assets/build_icons.sh to regenerate."
            )

        (contents / "Info.plist").write_text(
            _macos_info_plist(app_name, with_icon=icon_present)
        )
        aprint(f"Wrote {contents / 'Info.plist'}")

        # README is dropped next to the .app (NOT inside the bundle, so
        # Finder shows it as a sibling). Helps users hit the xattr -cr
        # workaround when Gatekeeper blocks an unsigned download.
        readme_path = output / f"{app_name}-README.txt"
        readme_path.write_text(_macos_readme(app_name))
        aprint(f"Wrote {readme_path}")

    return app_path


def bundle_linux_folder(
    *,
    arch: str,
    viewer_dist: Path,
    zarr_data: Path,
    output: Path,
    app_name: str,
) -> Path:
    """Build a portable Linux folder at ``output / f"{app_name}-linux-{arch}/"``.

    Args:
        arch: ``"amd64"`` or ``"arm64"``.
    """
    platform_name = f"linux-{arch}"
    launcher = get_launcher_path(platform_name)
    folder = output / f"{app_name}-{platform_name}"

    with asection(f"Building Linux folder bundle ({folder.name})"):
        folder.mkdir(parents=True, exist_ok=True)

        bundled_launcher = folder / "luxar-launcher"
        shutil.copy2(launcher, bundled_launcher)
        bundled_launcher.chmod(
            bundled_launcher.stat().st_mode
            | stat.S_IEXEC
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

        shutil.copytree(viewer_dist, folder / "viewer", dirs_exist_ok=True)
        shutil.copytree(zarr_data, folder / "data", dirs_exist_ok=True)

        if LOGO_PNG.is_file():
            # Standard FreeDesktop convention: ship as `<app>.png` so a
            # paired `.desktop` file (created by the user if they want a
            # desktop entry) can reference it via Icon=<absolute path>.
            shutil.copy2(LOGO_PNG, folder / f"{app_name}.png")
            aprint(f"Copied icon to {folder / f'{app_name}.png'}")

        (folder / "README.txt").write_text(_linux_readme(app_name))
        aprint(f"Wrote {folder / 'README.txt'}")

    return folder


def _macos_info_plist(app_name: str, *, with_icon: bool = True) -> str:
    """Generate a minimal ``Info.plist`` for a launcher-only app bundle.

    The bundle identifier uses ``org.czbiohub.luxar.<slug>`` so multiple
    exports do not collide in macOS' Launch Services database. If
    ``with_icon`` is True, the plist references ``AppIcon.icns`` (which
    must be present in ``Contents/Resources/``).
    """
    slug = "".join(c if c.isalnum() else "-" for c in app_name.lower()).strip("-")
    bundle_id = f"org.czbiohub.luxar.{slug or 'scene'}"
    icon_keys = ""
    if with_icon:
        icon_keys = (
            "    <key>CFBundleIconFile</key>\n"
            "    <string>AppIcon</string>\n"
        )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>{app_name}</string>
    <key>CFBundleDisplayName</key>
    <string>{app_name}</string>
    <key>CFBundleIdentifier</key>
    <string>{bundle_id}</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
{icon_keys}    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
"""


def _linux_readme(app_name: str) -> str:
    return f"""{app_name} — Luxar standalone scene
{"=" * (len(app_name) + 27)}

Quick start
-----------
    ./luxar-launcher

That starts a local HTTP server, opens a native viewer window, and
serves the bundled zarr scene. Close the window to shut everything
down (or press Ctrl+C in the terminal where it was launched).

Requirements
------------
The launcher uses the system WebView library:

    sudo apt-get install -y libwebkit2gtk-4.1
        (or libwebkit2gtk-4.0 on older distros)

These are present on every modern desktop Linux distribution. On
minimal/headless installs you can fall back to opening the system
browser instead:

    LUXAR_LAUNCHER_NO_WEBVIEW=1 ./luxar-launcher

Folder layout
-------------
    luxar-launcher    The Go-compiled native launcher
    viewer/           The Luxar viewer (HTML, JS, CSS, WASM)
    data/             The zarr dataset
    {app_name}.png    Icon (FreeDesktop convention)
    README.txt        This file
"""


def _macos_readme(app_name: str) -> str:
    return f"""{app_name}.app — Luxar standalone scene
{"=" * (len(app_name) + 31)}

Quick start
-----------
Double-click {app_name}.app. The viewer opens in a native window;
close the window (or press Cmd+Q) to shut everything down.

If macOS refuses to open it ("damaged" or "unidentified developer")
-------------------------------------------------------------------
That's macOS Gatekeeper blocking the unsigned binary it received via a
quarantining channel (web download, email, AirDrop, Slack, etc.).

Strip the quarantine attribute via Terminal:

    xattr -cr {app_name}.app
    open {app_name}.app

This is a one-time fix per copy of the app — once stripped, double-
click works normally. The app itself is a vanilla local HTTP server +
WKWebView; it makes no outbound network connections.
"""


def zip_macos_app(app_path: Path) -> Path:
    """Zip a macOS ``.app`` bundle next to itself.

    Prefers ``ditto -c -k --keepParent`` on Darwin so resource forks,
    extended attributes, and the bundle's directory layout are preserved
    exactly.  On non-Darwin hosts (e.g. Linux producing a macOS bundle as
    a cross-build), falls back to ``zipfile`` with executable-bit
    preservation so the embedded launcher remains runnable after
    extraction.

    The archive lands at ``<app_path>.zip`` (i.e. ``Foo.app.zip`` next to
    ``Foo.app``); any existing archive is overwritten.

    Returns the path to the produced ``.zip``.
    """
    if app_path.suffix != ".app" or not app_path.is_dir():
        raise ValueError(
            f"zip_macos_app expects a .app bundle directory, got {app_path}"
        )

    archive_path = app_path.with_suffix(".app.zip")
    if archive_path.exists():
        archive_path.unlink()

    ditto = shutil.which("ditto")
    if ditto and platform.system() == "Darwin":
        with asection(f"Zipping {app_path.name} with ditto"):
            subprocess.run(
                [
                    ditto,
                    "-c",
                    "-k",
                    "--keepParent",
                    str(app_path),
                    str(archive_path),
                ],
                check=True,
            )
            aprint(f"  ✓ {archive_path.name}")
    else:
        with asection(
            f"Zipping {app_path.name} with zipfile (ditto unavailable)"
        ):
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for root, _dirs, files in os.walk(app_path):
                    for fname in files:
                        full = Path(root) / fname
                        arcname = full.relative_to(app_path.parent)
                        # ``ZipInfo.from_file`` carries Unix permissions
                        # (executable bit on the launcher) over to the
                        # archive's external_attr field.
                        info = zipfile.ZipInfo.from_file(full, str(arcname))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        with open(full, "rb") as src:
                            zf.writestr(info, src.read())
            aprint(f"  ✓ {archive_path.name}")
    return archive_path
