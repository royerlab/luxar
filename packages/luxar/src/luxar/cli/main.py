"""luxar.cli – Command-line interface for building, serving, and inspecting Luxar Zarr scenes.

IMPORTANT: URL Construction
    When constructing data URLs for the viewer's ?src= parameter, DO NOT include
    trailing slashes. The viewer's fetch logic treats them differently:
    - CORRECT: http://host:port (joins correctly: http://host:port/.zmetadata)
    - WRONG:   http://host:port/ (creates double-slash: http://host:port//.zmetadata)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, cast

import typer
import uvicorn
from arbol import aprint, asection
from fastapi import FastAPI
from starlette.types import ASGIApp

from luxar import __version__
from luxar.utils.arbol_warnings import install_arbol_warnings

from ._traceback import exit_with_error
from .common_options import (
    AllowSensitivePathOption,
    BandwidthOption,
    CorsOriginOption,
    HostOption,
    JitterOption,
    LatencyOption,
    PacketLossOption,
    ProfileOption,
    make_port_option,
    parse_network_options_or_exit,
)
from .info_command import _dfs, register_info_command
from .network_simulation import (
    NETWORK_PROFILES,
    NetworkSimulationMiddleware,
    has_network_simulation,
    print_network_params,
)
from .optimise_command import register_optimise_command
from .restamp_lod_command import register_restamp_lod_command
from .serving import (
    DirectoryListingStaticFiles,
    _add_cors,
    _build_viewer_url,
    _is_sensitive_serve_path,
    _serve_data,
    _serve_viewer,
    _validate_serve_path,
    _warn_if_lan_exposed,
    create_server_app,
)
from .utils import (
    _DEFAULT_CORS_ORIGIN,
    append_title_param,
    check_viewer_built,
    dataset_title,
    ensure_viewer_built,
    pick_port,
    wait_for_server,
)
from .utils import (
    open_browser as open_browser_func,
)

# Server plumbing (CORS, path guards, DirectoryListingStaticFiles,
# create_server_app, _serve_data/_serve_viewer) lives in ``serving.py`` and the
# ``info`` command in ``info_command.py``; both are re-exported / registered
# here so ``luxar.cli.main`` stays the single CLI entry point and existing
# ``luxar.cli.main.*`` imports + mock/monkeypatch targets keep resolving.
# ``_DEFAULT_CORS_ORIGIN`` similarly lives in utils.py so subcommand modules can
# import it without forming a cycle through this file.
__all__ = [
    "app",
    "create_server_app",
    "_serve_data",
    "_serve_viewer",
    "_is_sensitive_serve_path",
    "_warn_if_lan_exposed",
    "_dfs",
]


def _version_callback(value: bool) -> None:
    if value:
        aprint(f"luxar {__version__}")
        raise typer.Exit()


app = typer.Typer(help="luxar – build and serve Zarr-backed nD scenes")


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """luxar – build and serve Zarr-backed nD scenes."""
    # Every warning raised anywhere in a CLI run displays as an arbol line when
    # arbol can show it, and falls back to Python's stock stderr display when it
    # cannot. Display-only: filters / -W / pytest.warns are unaffected, and the
    # install is skipped when a recorder is active.
    install_arbol_warnings()
    if not ctx.invoked_subcommand:
        aprint(ctx.get_help())
        raise typer.Exit(0)


# Add gsplat subcommands
from .gsplat_commands import app_gsplat  # noqa: E402

app.add_typer(app_gsplat, name="gsplat")

# Add mesh subcommands (import)
from .mesh_commands import app_mesh  # noqa: E402

app.add_typer(app_mesh, name="mesh")

# Add demo subcommands (list / info / run / run-all / cache)
from .demo_commands import app_demo  # noqa: E402

app.add_typer(app_demo, name="demo")

# Add env subcommands (bake / attach) — baked scene environments for physical meshes
from .env_commands import app_env  # noqa: E402

app.add_typer(app_env, name="env")

# Register the `info` inspection command (defined in info_command.py).
register_info_command(app)

# Register the `optimise` re-chunking command (defined in optimise_command.py).
register_optimise_command(app)

# Register the `restamp-lod` LOD-threshold re-derivation command (defined in
# restamp_lod_command.py).
register_restamp_lod_command(app)


def _start_data_server_thread(
    path: Path,
    host: str,
    port: int,
    bandwidth_mbps: Optional[float],
    latency_ms: Optional[float],
    jitter_percent: float,
    packet_loss_rate: float,
    allow_sensitive_path: bool,
    cors_origin: str,
    timeout: float = 5.0,
) -> threading.Thread:
    """Start the background data server thread shared by ``viewer`` and ``demo``.

    Readiness is polled (``wait_for_server``) instead of a fixed sleep, so a
    data server that dies on startup (e.g. loses a bind race) is reported
    instead of silently leaving the viewer pointing at a dead URL.

    Kept in main.py (not serving.py) so ``threading``, ``time``,
    ``_serve_data``, and ``wait_for_server`` resolve as *this module's*
    globals — preserving the existing test patch targets
    (``patch("luxar.cli.main._serve_data")``,
    ``patch("luxar.cli.main.wait_for_server")``,
    ``monkeypatch.setattr(cli_main.threading, "Thread", ...)``).
    """
    thread = threading.Thread(
        target=_serve_data,
        args=(
            path,
            host,
            port,
            bandwidth_mbps,
            latency_ms,
            jitter_percent,
            packet_loss_rate,
            allow_sensitive_path,
            cors_origin,
        ),
        daemon=True,
    )
    thread.start()
    if not wait_for_server(host, port, thread, timeout=timeout):
        aprint(f"⚠️  Data server on {host}:{port} did not become ready.")
    return thread


def _warn_control_flag_misuse(
    *, control: bool, control_token: Optional[str], viewer: bool, viewer_only: bool
) -> None:
    """Explain control options that cannot affect the selected serve mode."""
    if control and not (viewer or viewer_only):
        aprint("⚠️  --control requires --viewer or --viewer-only. Ignoring --control.")
    if control_token and not control:
        aprint("⚠️  --control-token requires --control. Ignoring --control-token.")


# ────────────────────────────── serve ────────────────────────────────────────
@app.command()
def serve(
    path: Optional[Path] = typer.Argument(None, exists=True, readable=True),
    host: HostOption = "127.0.0.1",
    port: int = make_port_option(8000),
    viewer: bool = typer.Option(False, "--viewer", help="Also serve the viewer"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Port for viewer"),
    open_browser: bool = typer.Option(False, "--open", help="Open browser"),
    viewer_only: bool = typer.Option(
        False, "--viewer-only", help="Serve only the viewer"
    ),
    # Network simulation parameters (shared declarations: common_options.py)
    profile: ProfileOption = None,
    bandwidth: BandwidthOption = None,
    latency: LatencyOption = None,
    jitter: JitterOption = None,
    packet_loss: PacketLossOption = None,
    cors_origin: CorsOriginOption = _DEFAULT_CORS_ORIGIN,
    allow_sensitive_path: AllowSensitivePathOption = False,
    control: bool = typer.Option(
        False,
        "--control",
        help="Expose the remote-control hub at /control (kiosk touch panels, agents)",
    ),
    control_token: Optional[str] = typer.Option(
        None,
        "--control-token",
        help="Require this shared secret as ?token= on every control socket",
    ),
) -> None:
    """Serve a directory, Zarr dataset, or viewer via HTTP.

    Network Simulation:
        Use network simulation options to test viewer performance under
        realistic network conditions. You can use a preset profile or
        specify individual parameters.

        Profiles: 3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested

        Individual parameters override profile defaults.

    Examples:
        # Simulate 3G mobile connection
        luxar serve data.luxar.zarr --profile 3g --viewer

        # Simulate custom slow connection
        luxar serve data.luxar.zarr --bandwidth 500kbps --latency 200ms

        # Use 4G profile with custom latency
        luxar serve data.luxar.zarr --profile 4g --latency 300ms

        # Test packet loss
        luxar serve data.luxar.zarr --bandwidth 10mbps --packet-loss 5%
    """
    try:
        # Warn about conflicting flags
        if viewer_only and path is not None:
            aprint("⚠️  --viewer-only ignores the path argument")
        if viewer_only and viewer:
            aprint(
                "⚠️  --viewer-only already includes the viewer; --viewer is redundant"
            )
        _warn_control_flag_misuse(
            control=control,
            control_token=control_token,
            viewer=viewer,
            viewer_only=viewer_only,
        )
        _warn_if_lan_exposed(host, cors_origin)

        # Handle viewer-only mode
        if viewer_only:
            if not ensure_viewer_built():
                raise typer.Exit(1)

            actual_viewer_port = pick_port(viewer_port, host, label="viewer")
            if actual_viewer_port is None:
                raise typer.Exit(1)

            _serve_viewer(
                host,
                actual_viewer_port,
                None,
                open_browser,
                cors_origin,
                control=control,
                control_token=control_token,
            )
            return

        # Require path for data serving
        if path is None:
            aprint("❌ Error: Path required unless using --viewer-only")
            raise typer.Exit(1)

        _validate_serve_path(path, allow_sensitive_path=allow_sensitive_path)

        # Determine what we're serving
        if path.is_dir():
            serve_path = path
            if path.name.endswith(".zarr"):
                aprint(f"📁 Serving Zarr dataset: {path}")
            else:
                aprint(f"📂 Serving directory: {path}")
        else:
            aprint(f"❌ Error: {path} is not a directory")
            raise typer.Exit(1)

        # Find available ports (auto-increment if requested ports are busy)
        actual_port = pick_port(port, host, label="data")
        if actual_port is None:
            raise typer.Exit(1)

        viewer_served = False
        if viewer:
            actual_viewer_port = pick_port(viewer_port, host, label="viewer")
            if actual_viewer_port is None:
                raise typer.Exit(1)
        else:
            actual_viewer_port = viewer_port

        # Parse network simulation parameters
        bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate = (
            parse_network_options_or_exit(
                profile, bandwidth, latency, jitter, packet_loss
            )
        )

        if has_network_simulation(
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate
        ):
            print_network_params(
                bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate
            )
            aprint(
                "⚠️  [Luxar] Responses will be throttled - this is intentional for testing"
            )

        api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)
        _add_cors(api, cors_origin)

        # Mount the static files handler with directory listing
        api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

        aprint(f"🛰️  Serving {serve_path} at http://{host}:{actual_port}")

        # Also serve viewer if requested
        if viewer:
            if not ensure_viewer_built():
                aprint("⚠️  Skipping viewer serving.")
            else:
                # Start viewer in a separate thread
                data_url = f"http://{host}:{actual_port}"  # No trailing slash
                viewer_thread = threading.Thread(
                    target=_serve_viewer,
                    args=(host, actual_viewer_port, data_url, False, cors_origin),
                    kwargs={
                        "title": dataset_title(serve_path),
                        "control": control,
                        "control_token": control_token,
                    },
                    daemon=True,
                )
                viewer_thread.start()
                if wait_for_server(host, actual_viewer_port, viewer_thread):
                    viewer_served = True
                else:
                    aprint("⚠️  Viewer server did not become ready.")
        else:
            hint_url = append_title_param(
                f"http://localhost:{viewer_port}/?src=http://{host}:{actual_port}",
                dataset_title(serve_path),
            )
            aprint(f"📊 Viewer URL: {hint_url}")

        # Open browser if requested
        if open_browser:
            if not viewer:
                aprint("⚠️  --open requires --viewer to also be set. Ignoring --open.")
            elif not viewer_served:
                aprint("⚠️  Viewer not served; skipping --open.")
            else:
                data_url = f"http://{host}:{actual_port}"
                viewer_url = _build_viewer_url(
                    host,
                    actual_viewer_port,
                    data_url,
                    title=dataset_title(serve_path),
                    control=control,
                    control_token=control_token,
                )
                open_browser_func(viewer_url)

        # Wrap the complete ASGI app with network simulation (if enabled)
        asgi_app: ASGIApp = api
        if has_network_simulation(
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate
        ):
            asgi_app = cast(
                ASGIApp,
                NetworkSimulationMiddleware(
                    api,
                    bandwidth_limit_mbps=bandwidth_mbps,
                    latency_ms=latency_ms,
                    jitter_percent=jitter_percent,
                    packet_loss_rate=packet_loss_rate,
                ),
            )

        uvicorn.run(
            asgi_app, host=host, port=actual_port, reload=False, log_level="warning"
        )
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down server...")
    except Exception as e:
        exit_with_error(f"❌ Error serving path: {e}", e)


# ────────────────────────────── viewer ──────────────────────────────────────
@app.command()
def viewer(
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Zarr data to load"),
    host: HostOption = "127.0.0.1",
    port: int = make_port_option(5173, "Port number"),
    data_port: int = typer.Option(8000, "--data-port", help="Port for data server"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
    # Network simulation parameters, data server only (see common_options.py)
    profile: ProfileOption = None,
    bandwidth: BandwidthOption = None,
    latency: LatencyOption = None,
    jitter: JitterOption = None,
    packet_loss: PacketLossOption = None,
    cors_origin: CorsOriginOption = _DEFAULT_CORS_ORIGIN,
    allow_sensitive_path: AllowSensitivePathOption = False,
) -> None:
    """Serve the Luxar viewer, optionally with data.

    Network Simulation:
        Network simulation options apply ONLY to the data server (zarr files),
        not the viewer HTML/JS/CSS. Use these to test viewer performance
        under various network conditions.

    Examples:
        # Serve viewer with data using 3G simulation
        luxar viewer --data foo.luxar.zarr --profile 3g

        # Serve viewer only (no simulation applies)
        luxar viewer
    """
    try:
        _warn_if_lan_exposed(host, cors_origin)

        if not ensure_viewer_built():
            raise typer.Exit(1)

        # Parse network simulation parameters (applies only if data is provided)
        bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate = (
            parse_network_options_or_exit(
                profile, bandwidth, latency, jitter, packet_loss
            )
        )

        if has_network_simulation(
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate
        ):
            if data is None:
                aprint(
                    "⚠️  [Luxar] Network simulation requires --data to be specified. "
                    "Simulation will be ignored."
                )
            else:
                print_network_params(
                    bandwidth_mbps,
                    latency_ms,
                    jitter_percent,
                    packet_loss_rate,
                    qualifier="data server only",
                )

        # If data provided, serve it in background
        data_url = None
        if data:
            if not data.exists():
                aprint(f"❌ Data path does not exist: {data}")
                raise typer.Exit(1)
            if not data.is_dir():
                aprint(
                    f"❌ Error: --data must be a directory (a .zarr store), "
                    f"got a file: {data}"
                )
                raise typer.Exit(1)
            _validate_serve_path(data, allow_sensitive_path=allow_sensitive_path)

            # Find available port for data server
            actual_data_port = pick_port(data_port, host, label="data")
            if actual_data_port is None:
                raise typer.Exit(1)

            # Start data server in background thread
            _start_data_server_thread(
                data,
                host,
                actual_data_port,
                bandwidth_mbps,
                latency_ms,
                jitter_percent,
                packet_loss_rate,
                allow_sensitive_path,
                cors_origin,
            )

            # The data server mounts the dataset itself at its root, so the
            # URL carries no store-name suffix. No trailing slash!
            data_url = f"http://{host}:{actual_data_port}"

        # Find available port for viewer
        actual_viewer_port = pick_port(port, host, label="viewer")
        if actual_viewer_port is None:
            raise typer.Exit(1)

        # Serve viewer
        _serve_viewer(
            host,
            actual_viewer_port,
            data_url,
            open_browser,
            cors_origin,
            title=dataset_title(data),
        )

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
    except Exception as e:
        exit_with_error(f"❌ Error: {e}", e)


# ─────────────────────────────── export ──────────────────────────────────────
@app.command()
def export(
    source: Path = typer.Argument(..., exists=True, readable=True, help="Zarr dataset"),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output folder for the standalone export"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output folder"
    ),
    open_browser: bool = typer.Option(
        False, "--open", help="Serve and open browser after export"
    ),
    port: int = make_port_option(8000, "Port for local server (with --open)"),
    native: Optional[str] = typer.Option(
        None,
        "--native",
        help=(
            "Comma-separated native bundles to produce instead of the "
            "Python serve.py folder. Choices: macos, linux-amd64, linux-arm64. "
            "Example: --native macos,linux-amd64"
        ),
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Bundle name (defaults to the zarr stem). Used with --native.",
    ),
    zip_app: bool = typer.Option(
        True,
        "--zip/--no-zip",
        help=(
            "When --native macos is requested, also produce a sibling "
            "<name>.app.zip via ditto (or zipfile fallback). On by default "
            "so users get a single shareable artifact with the .app's "
            "permissions and resource forks intact. Pass --no-zip to skip."
        ),
    ),
) -> None:
    """Export a zarr scene + viewer as a standalone offline folder.

    Creates a self-contained folder with the viewer, dataset, and a serve
    script. Anyone can view the scene with just Python 3 and a browser.

    With --native, produces double-clickable native bundles instead of the
    Python serve.py folder. Requires `make build-launchers` to have been
    run first.

    Examples:
        luxar export my_scene.luxar.zarr -o my_export/
        luxar export my_scene.luxar.zarr -o my_export/ --overwrite
        luxar export my_scene.luxar.zarr -o my_export/ --open
        luxar export my_scene.luxar.zarr -o out/ --native macos
        luxar export my_scene.luxar.zarr -o out/ --native macos,linux-amd64,linux-arm64
    """
    try:
        if native:
            _run_native_export(
                source=source,
                output=output,
                overwrite=overwrite,
                native=native,
                name=name,
                zip_app=zip_app,
            )
            return

        from .export import export_scene

        with asection("Luxar Export"):
            result = export_scene(
                source=source,
                output=output,
                overwrite=overwrite,
            )
            aprint(f"Exported to {result}")
            aprint(f"To view: cd {result} && python serve.py")

        if open_browser:
            import subprocess
            import sys

            serve_script = result / "serve.py"
            aprint(f"Starting local server on port {port}...")
            aprint("Press Ctrl+C to stop the server.")
            try:
                subprocess.run(
                    [sys.executable, str(serve_script), "--port", str(port)],
                    cwd=str(result),
                )
            except KeyboardInterrupt:
                aprint("\n🛑 Server stopped.")
    except typer.Exit:
        raise
    except FileExistsError as e:
        aprint(f"❌ {e}")
        aprint("Use --overwrite to replace existing output")
        raise typer.Exit(1)
    except (FileNotFoundError, ValueError) as e:
        aprint(f"❌ {e}")
        raise typer.Exit(1)
    except Exception as e:
        exit_with_error(f"❌ Error exporting scene: {e}", e)


def _run_native_export(
    *,
    source: Path,
    output: Path,
    overwrite: bool,
    native: str,
    name: Optional[str],
    zip_app: bool = True,
) -> None:
    """Helper for ``luxar export --native``: validate args, then bundle.

    All preconditions (zarr validity, platform spelling, viewer dist
    presence, *every* requested launcher binary) are checked BEFORE any
    destructive filesystem action runs, so a partial or missing build
    can never wipe a pre-existing output directory passed with
    --overwrite.
    """
    import shutil

    from .export import _require_third_party_notices
    from .native_app import (
        SUPPORTED_PLATFORMS,
        bundle_linux_folder,
        bundle_macos_app,
        get_launcher_path,
        validate_bundle_name,
        zip_macos_app,
    )
    from .utils import get_viewer_dist_path, validate_zarr_store

    is_valid, error = validate_zarr_store(source)
    if not is_valid:
        raise ValueError(f"Invalid zarr store: {error}")

    requested = [p.strip() for p in native.split(",") if p.strip()]
    unknown = [p for p in requested if p not in SUPPORTED_PLATFORMS]
    if unknown:
        raise ValueError(
            f"Unknown --native platform(s): {', '.join(unknown)}. "
            f"Choices: {', '.join(SUPPORTED_PLATFORMS)}"
        )
    if not requested:
        raise ValueError("--native requires at least one platform")

    if not check_viewer_built():
        raise FileNotFoundError(
            "Viewer not built. Run: cd packages/luxar-viewer && pnpm build"
        )
    viewer_dist = get_viewer_dist_path()
    _require_third_party_notices(viewer_dist)

    # Pre-validate every requested launcher binary before touching the
    # output directory. This prevents the rmtree-then-fail-on-second-
    # platform footgun where --overwrite would wipe the user's old
    # export only to discover a missing binary mid-flight.
    for plat in requested:
        get_launcher_path(plat)  # raises LauncherNotBuiltError on miss

    # Resolve and validate the bundle name BEFORE any destructive
    # filesystem action. A crafted/derived name containing path separators
    # or ``..`` could otherwise escape the output directory (issue #686),
    # and it must fail before the rmtree below ever runs.
    #
    # Strip the full (possibly compound) zarr suffix for the bundle name —
    # Path.stem only drops the last suffix, so ``foo.luxar.zarr`` would yield
    # ``foo.luxar`` rather than ``foo``.
    _bundle_stem = source.name
    for _suf in (".luxar.zarr", ".gsplats.zarr", ".zarr"):
        if _bundle_stem.endswith(_suf):
            _bundle_stem = _bundle_stem[: -len(_suf)]
            break
    bundle_name = name or _bundle_stem or "LuxarScene"
    validate_bundle_name(bundle_name)

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    with asection(f"Luxar Export (native: {', '.join(requested)})"):
        produced: list[Path] = []
        for plat in requested:
            if plat == "macos":
                app_path = bundle_macos_app(
                    viewer_dist=viewer_dist,
                    zarr_data=source,
                    output=output,
                    app_name=bundle_name,
                )
                produced.append(app_path)
                if zip_app:
                    produced.append(zip_macos_app(app_path))
            elif plat.startswith("linux-"):
                arch = plat.split("-", 1)[1]
                produced.append(
                    bundle_linux_folder(
                        arch=arch,
                        viewer_dist=viewer_dist,
                        zarr_data=source,
                        output=output,
                        app_name=bundle_name,
                    )
                )

        aprint("")
        aprint("✅ Native bundles produced:")
        for p in produced:
            aprint(f"   {p}")


# ─────────────────────────────── profiles ────────────────────────────────────
@app.command()
def profiles() -> None:
    """List available network simulation profiles.

    Display all preset network profiles with their parameters. Use these
    profiles with the --profile option in serve, viewer, and demo commands.
    """
    aprint("📊 [Luxar] Available Network Profiles\n")

    for profile_name in sorted(NETWORK_PROFILES.keys()):
        profile = NETWORK_PROFILES[profile_name]
        aprint(f"  {profile_name}")
        aprint(f"    Name: {profile['name']}")
        aprint(f"    Bandwidth: {profile['bandwidth']}")
        aprint(f"    Latency: {profile['latency']}")
        aprint(f"    Jitter: {profile['jitter'] * 100:.0f}%")
        aprint(f"    Packet Loss: {profile['packet_loss'] * 100:.1f}%")
        aprint(f"    Description: {profile['description']}")
        aprint("")

    aprint("Usage: luxar serve data.luxar.zarr --profile <profile-name>")
    aprint("       luxar viewer --data data.luxar.zarr --profile satellite")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
