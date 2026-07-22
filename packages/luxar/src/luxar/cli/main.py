"""luxar.cli – Command-line interface for building, serving, and inspecting Luxar Zarr scenes.

IMPORTANT: URL Construction
    When constructing data URLs for the viewer's ?src= parameter, DO NOT include
    trailing slashes. The viewer's fetch logic treats them differently:
    - CORRECT: http://host:port (joins correctly: http://host:port/.zmetadata)
    - WRONG:   http://host:port/ (creates double-slash: http://host:port//.zmetadata)
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, cast

import typer
import uvicorn
from arbol import aprint, asection
from fastapi import FastAPI
from starlette.types import ASGIApp

from luxar import __version__

from .info_command import _dfs, register_info_command
from .network_simulation import (
    NETWORK_PROFILES,
    NetworkSimulationMiddleware,
    has_network_simulation,
    parse_network_options,
    print_network_params,
)
from .serving import (
    DirectoryListingStaticFiles,
    _add_cors,
    _is_sensitive_serve_path,
    _serve_data,
    _serve_viewer,
    _validate_serve_path,
    _warn_if_lan_exposed,
    create_server_app,
)
from .utils import (
    _DEFAULT_CORS_ORIGIN,
    build_viewer,
    check_viewer_built,
    find_available_port,
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
    if not ctx.invoked_subcommand:
        aprint(ctx.get_help())
        raise typer.Exit(0)


# Add gsplat subcommands
from .gsplat_commands import app_gsplat  # noqa: E402

app.add_typer(app_gsplat, name="gsplat")

# Register the `info` inspection command (defined in info_command.py).
register_info_command(app)


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
    startup_delay: float = 1.0,
) -> threading.Thread:
    """Start the background data server thread shared by ``viewer`` and ``demo``.

    Kept in main.py (not serving.py) so ``threading``, ``time``, and
    ``_serve_data`` resolve as *this module's* globals — preserving the existing
    test patch targets (``patch("luxar.cli.main._serve_data")``,
    ``monkeypatch.setattr(cli_main.threading, "Thread", ...)``,
    ``monkeypatch.setattr(cli_main.time, "sleep", ...)``).
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
    time.sleep(startup_delay)
    return thread


# ────────────────────────────── serve ────────────────────────────────────────
@app.command()
def serve(
    path: Optional[Path] = typer.Argument(None, exists=True, readable=True),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host address to bind to (use 0.0.0.0 for all interfaces)",
    ),
    port: int = typer.Option(8000, "--port", "-p"),
    viewer: bool = typer.Option(False, "--viewer", help="Also serve the viewer"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Port for viewer"),
    open_browser: bool = typer.Option(False, "--open", help="Open browser"),
    viewer_only: bool = typer.Option(
        False, "--viewer-only", help="Serve only the viewer"
    ),
    # Network simulation parameters
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Network profile (3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested)",
    ),
    bandwidth: Optional[str] = typer.Option(
        None,
        "--bandwidth",
        "-b",
        help="Bandwidth limit (e.g., '1mbps', '500kbps', '10mbps')",
    ),
    latency: Optional[str] = typer.Option(
        None,
        "--latency",
        "-l",
        help="Network latency (e.g., '100ms', '500ms', '1s')",
    ),
    jitter: Optional[str] = typer.Option(
        None,
        "--jitter",
        "-j",
        help="Latency jitter as percentage (e.g., '10%', '0.1')",
    ),
    packet_loss: Optional[str] = typer.Option(
        None,
        "--packet-loss",
        help="Packet loss rate (e.g., '1%', '0.01', '5%')",
    ),
    cors_origin: str = typer.Option(
        _DEFAULT_CORS_ORIGIN,
        "--cors-origin",
        help=(
            "Allowed CORS origin. Default 'local' allows localhost/127.0.0.1/::1. "
            "Use '*' to allow any origin without credentials."
        ),
    ),
    allow_sensitive_path: bool = typer.Option(
        False,
        "--allow-sensitive-path",
        help="Allow serving obvious system paths such as /, /etc, /proc, /sys, /dev.",
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

    Args:
        path (Path, optional): Path to directory or Zarr dataset to serve.
        host (str, optional): Host address. Defaults to "127.0.0.1".
        port (int, optional): Port number. Defaults to 8000.
        viewer (bool, optional): Also serve the viewer. Defaults to False.
        viewer_port (int, optional): Port for viewer. Defaults to 5173.
        open_browser (bool, optional): Open browser. Defaults to False.
        viewer_only (bool, optional): Serve only the viewer. Defaults to False.
        profile (str, optional): Network profile name.
        bandwidth (str, optional): Bandwidth limit.
        latency (str, optional): Network latency.
        jitter (str, optional): Latency jitter percentage.
        packet_loss (str, optional): Packet loss rate.
        cors_origin (str, optional): Allowed CORS origin. Defaults to "local".
        allow_sensitive_path (bool, optional): Permit serving system paths.
    """
    try:
        # Warn about conflicting flags
        if viewer_only and path is not None:
            aprint("⚠️  --viewer-only ignores the path argument")
        if viewer_only and viewer:
            aprint(
                "⚠️  --viewer-only already includes the viewer; --viewer is redundant"
            )
        _warn_if_lan_exposed(host, cors_origin)

        # Handle viewer-only mode
        if viewer_only:
            if not check_viewer_built():
                aprint(
                    "❌ Viewer not built. Run: cd packages/luxar-viewer && pnpm build"
                )
                raise typer.Exit(1)

            # Find available port
            actual_viewer_port = find_available_port(viewer_port)
            if actual_viewer_port is None:
                aprint(f"❌ Error: No available ports found near {viewer_port}")
                raise typer.Exit(1)
            if actual_viewer_port != viewer_port:
                aprint(
                    f"⚠️  Viewer port {viewer_port} busy, using {actual_viewer_port} instead"
                )

            _serve_viewer(host, actual_viewer_port, None, open_browser, cors_origin)
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
        actual_port = find_available_port(port)
        if actual_port is None:
            aprint(f"❌ Error: No available ports found near {port}")
            raise typer.Exit(1)

        if actual_port != port:
            aprint(f"⚠️  Port {port} busy, using {actual_port} instead")

        viewer_served = False
        if viewer:
            actual_viewer_port = find_available_port(viewer_port)
            if actual_viewer_port is None:
                aprint(f"❌ Error: No available ports found near {viewer_port}")
                raise typer.Exit(1)
            if actual_viewer_port != viewer_port:
                aprint(
                    f"⚠️  Viewer port {viewer_port} busy, using {actual_viewer_port} instead"
                )
        else:
            actual_viewer_port = viewer_port

        # Parse network simulation parameters
        try:
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate = (
                parse_network_options(profile, bandwidth, latency, jitter, packet_loss)
            )
        except ValueError as e:
            aprint(f"❌ [Luxar] {e}")
            raise typer.Exit(code=1)

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
            if not check_viewer_built():
                aprint("⚠️  Viewer not built. Skipping viewer serving.")
                aprint("💡 To build: cd packages/luxar-viewer && pnpm build")
            else:
                # Start viewer in a separate thread
                data_url = f"http://{host}:{actual_port}"  # No trailing slash
                viewer_thread = threading.Thread(
                    target=_serve_viewer,
                    args=(host, actual_viewer_port, data_url, False, cors_origin),
                    daemon=True,
                )
                viewer_thread.start()
                time.sleep(1)  # Give viewer time to start
                viewer_served = True
        else:
            aprint(
                f"📊 Viewer URL: http://localhost:5173/?src=http://{host}:{actual_port}"
            )

        # Open browser if requested
        if open_browser:
            if not viewer:
                aprint("⚠️  --open requires --viewer to also be set. Ignoring --open.")
            elif not viewer_served:
                aprint("⚠️  Viewer not served; skipping --open.")
            else:
                data_url = f"http://{host}:{actual_port}"
                viewer_url = f"http://{host}:{actual_viewer_port}/?src={data_url}"
                time.sleep(1)  # Give servers time to start
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
        aprint(f"❌ Error serving path: {e}")
        raise typer.Exit(1)


# ────────────────────────────── viewer ──────────────────────────────────────
@app.command()
def viewer(
    data: Optional[Path] = typer.Option(None, "--data", "-d", help="Zarr data to load"),
    host: str = typer.Option("127.0.0.1", "--host", help="Host address"),
    port: int = typer.Option(5173, "--port", "-p", help="Port number"),
    data_port: int = typer.Option(8000, "--data-port", help="Port for data server"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
    # Network simulation parameters (apply to data server only)
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Network profile (3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested)",
    ),
    bandwidth: Optional[str] = typer.Option(
        None,
        "--bandwidth",
        "-b",
        help="Bandwidth limit (e.g., '1mbps', '500kbps')",
    ),
    latency: Optional[str] = typer.Option(
        None,
        "--latency",
        "-l",
        help="Network latency (e.g., '100ms', '500ms')",
    ),
    jitter: Optional[str] = typer.Option(
        None,
        "--jitter",
        "-j",
        help="Latency jitter percentage (e.g., '10%', '0.1')",
    ),
    packet_loss: Optional[str] = typer.Option(
        None,
        "--packet-loss",
        help="Packet loss rate (e.g., '1%', '0.01')",
    ),
    cors_origin: str = typer.Option(
        _DEFAULT_CORS_ORIGIN,
        "--cors-origin",
        help=(
            "Allowed CORS origin. Default 'local' allows localhost/127.0.0.1/::1. "
            "Use '*' to allow any origin without credentials."
        ),
    ),
    allow_sensitive_path: bool = typer.Option(
        False,
        "--allow-sensitive-path",
        help="Allow serving obvious system paths such as /, /etc, /proc, /sys, /dev.",
    ),
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

    Args:
        data (Path, optional): Zarr data to load.
        host (str, optional): Host address. Defaults to "127.0.0.1".
        port (int, optional): Port number. Defaults to 5173.
        data_port (int, optional): Port for data server. Defaults to 8000.
        open_browser (bool, optional): Open browser. Defaults to True.
        profile (str, optional): Network profile (applies to data server only).
        bandwidth (str, optional): Bandwidth limit (applies to data server only).
        latency (str, optional): Network latency (applies to data server only).
        jitter (str, optional): Latency jitter percentage (applies to data server only).
        packet_loss (str, optional): Packet loss rate (applies to data server only).
        cors_origin (str, optional): Allowed CORS origin for both viewer and
            data servers. Defaults to "local" (loopback only).
        allow_sensitive_path (bool, optional): Permit serving system paths.
    """
    try:
        _warn_if_lan_exposed(host, cors_origin)

        # Check if viewer is built
        if not check_viewer_built():
            aprint("❌ Viewer not built. Building now...")
            if not build_viewer():
                aprint("❌ Failed to build viewer")
                raise typer.Exit(1)

        # Parse network simulation parameters (applies only if data is provided)
        try:
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate = (
                parse_network_options(profile, bandwidth, latency, jitter, packet_loss)
            )
        except ValueError as e:
            aprint(f"❌ [Luxar] {e}")
            raise typer.Exit(code=1)

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
            _validate_serve_path(data, allow_sensitive_path=allow_sensitive_path)

            # Find available port for data server
            actual_data_port = find_available_port(data_port)
            if actual_data_port is None:
                aprint(f"❌ No available ports near {data_port}")
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
        actual_viewer_port = find_available_port(port)
        if actual_viewer_port is None:
            aprint(f"❌ No available ports near {port}")
            raise typer.Exit(1)
        if actual_viewer_port != port:
            aprint(f"⚠️  Viewer port {port} busy, using {actual_viewer_port} instead")

        # Serve viewer
        _serve_viewer(host, actual_viewer_port, data_url, open_browser, cors_origin)

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down viewer...")
    except Exception as e:
        aprint(f"❌ Error: {e}")
        raise typer.Exit(1)


# ────────────────────────────── demo ─────────────────────────────────────────
@app.command()
def demo(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path"),
    n_points: int = typer.Option(10000, "--points", "-n", help="Number of points"),
    demo_type: str = typer.Option(
        "lorenz",
        "--type",
        "-t",
        help="Demo type (currently only 'lorenz' supported)",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    serve: bool = typer.Option(True, "--serve/--no-serve", help="Serve with viewer"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser"),
    port: int = typer.Option(8000, "--port", "-p", help="Data server port"),
    viewer_port: int = typer.Option(5173, "--viewer-port", help="Viewer port"),
    # Network simulation parameters
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Network profile (3g, 4g, 5g, slow-broadband, broadband, fast-broadband, satellite, rural, congested)",
    ),
    bandwidth: Optional[str] = typer.Option(
        None,
        "--bandwidth",
        "-b",
        help="Bandwidth limit (e.g., '1mbps', '500kbps')",
    ),
    latency: Optional[str] = typer.Option(
        None,
        "--latency",
        "-l",
        help="Network latency (e.g., '100ms', '500ms')",
    ),
    jitter: Optional[str] = typer.Option(
        None,
        "--jitter",
        "-j",
        help="Latency jitter percentage (e.g., '10%', '0.1')",
    ),
    packet_loss: Optional[str] = typer.Option(
        None,
        "--packet-loss",
        help="Packet loss rate (e.g., '1%', '0.01')",
    ),
    cors_origin: str = typer.Option(
        _DEFAULT_CORS_ORIGIN,
        "--cors-origin",
        help=(
            "Allowed CORS origin. Default 'local' allows localhost/127.0.0.1/::1. "
            "Use '*' to allow any origin without credentials."
        ),
    ),
) -> None:
    """Generate a demo dataset and optionally serve with viewer.

    Network Simulation:
        Test viewer performance under various network conditions using
        --profile or individual simulation parameters.

    Examples:
        # Generate and serve (default behavior)
        luxar demo

        # Test with 3G network conditions
        luxar demo --profile 3g

        # Just generate without serving
        luxar demo --no-serve --output my_demo.luxar.zarr

        # Generate with specific parameters and simulate slow network
        luxar demo --points 100000 --bandwidth 500kbps --latency 200ms

    Args:
        output (Path, optional): Output path. Required when --no-serve.
        n_points (int, optional): Number of points. Defaults to 10000.
        demo_type (str, optional): Demo type. Defaults to "lorenz".
        seed (int, optional): Random seed.
        serve (bool, optional): Serve with viewer. Defaults to True.
        open_browser (bool, optional): Open browser. Defaults to True.
        port (int, optional): Data server port. Defaults to 8000.
        viewer_port (int, optional): Viewer port. Defaults to 5173.
        profile (str, optional): Network profile name.
        bandwidth (str, optional): Bandwidth limit.
        latency (str, optional): Network latency.
        jitter (str, optional): Latency jitter percentage.
        packet_loss (str, optional): Packet loss rate.
        cors_origin (str, optional): Allowed CORS origin for both viewer and
            data servers. Defaults to "local" (loopback only).
    """
    # Validate inputs early
    if n_points <= 0:
        aprint(f"❌ --points must be positive, got {n_points}")
        raise typer.Exit(1)

    if not serve and output is None:
        aprint("❌ --output is required when using --no-serve")
        raise typer.Exit(1)

    _temp_dir_ctx = None
    try:
        with asection("Demo Configuration and Generation"):
            # Determine output path
            if output is None:
                # serve=True and output=None: use a temp directory
                _temp_dir_ctx = tempfile.TemporaryDirectory(prefix="luxar_demo_")
                temp_dir = Path(_temp_dir_ctx.__enter__())
                output = temp_dir / f"{demo_type}_demo.luxar.zarr"
                aprint(f"📂 Using temporary directory: {temp_dir}")
            else:
                # Enforce the canonical scene extension so the served/reported
                # path matches what the compiler actually writes.
                from luxar.utils.paths import normalize_zarr_path

                output = normalize_zarr_path(output, ".luxar.zarr")

            # Generate demo
            aprint(f"🎲 Generating {demo_type} demo with {n_points:,} points...")

            if demo_type == "lorenz":
                from luxar.utils.demos import create_lorenz_attractor

                create_lorenz_attractor(output, n_points=n_points, seed=seed)
            else:
                aprint(f"❌ Unknown demo type: {demo_type}")
                aprint("💡 Available types: lorenz")
                raise typer.Exit(1)

            aprint(f"✅ Generated {n_points:,} points → {output}")

        if not serve:
            return

        # Parse network simulation parameters (if serving)
        try:
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate = (
                parse_network_options(profile, bandwidth, latency, jitter, packet_loss)
            )
        except ValueError as e:
            aprint(f"❌ [Luxar] {e}")
            raise typer.Exit(code=1)

        if has_network_simulation(
            bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate
        ):
            print_network_params(
                bandwidth_mbps, latency_ms, jitter_percent, packet_loss_rate
            )

        with asection("Viewer Setup and Port Management"):
            # Check viewer is built
            if not check_viewer_built():
                aprint("🔨 Building viewer...")
                if not build_viewer():
                    aprint("❌ Failed to build viewer")
                    raise typer.Exit(1)

            # Find available ports
            actual_port = find_available_port(port)
            actual_viewer_port = find_available_port(viewer_port)

            if actual_port is None or actual_viewer_port is None:
                aprint("❌ Could not find available ports")
                raise typer.Exit(1)

        with asection("Server Startup"):
            # Start data server in background
            _start_data_server_thread(
                output,
                "127.0.0.1",
                actual_port,
                bandwidth_mbps,
                latency_ms,
                jitter_percent,
                packet_loss_rate,
                False,  # allow_sensitive_path
                cors_origin,
            )

            # Construct data URL — the store is mounted at the server root,
            # so no store-name suffix (and no trailing slash).
            data_url = f"http://127.0.0.1:{actual_port}"

            # Serve viewer (this blocks)
            aprint("\n🎉 Demo ready! Starting viewer...")
            _serve_viewer(
                "127.0.0.1",
                actual_viewer_port,
                data_url,
                open_browser,
                cors_origin,
            )

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        aprint("\n🛑 Shutting down demo...")
    except Exception as e:
        aprint(f"❌ Error: {e}")
        raise typer.Exit(1)
    finally:
        if _temp_dir_ctx is not None:
            _temp_dir_ctx.__exit__(None, None, None)


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
    port: int = typer.Option(
        8000, "--port", "-p", help="Port for local server (with --open)"
    ),
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
        aprint(f"❌ Error exporting scene: {e}")
        raise typer.Exit(1)


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

    from .native_app import (
        SUPPORTED_PLATFORMS,
        bundle_linux_folder,
        bundle_macos_app,
        get_launcher_path,
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

    # Pre-validate every requested launcher binary before touching the
    # output directory. This prevents the rmtree-then-fail-on-second-
    # platform footgun where --overwrite would wipe the user's old
    # export only to discover a missing binary mid-flight.
    for plat in requested:
        get_launcher_path(plat)  # raises LauncherNotBuiltError on miss

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Strip the full (possibly compound) zarr suffix for the bundle name —
    # Path.stem only drops the last suffix, so ``foo.luxar.zarr`` would yield
    # ``foo.luxar`` rather than ``foo``.
    _bundle_stem = source.name
    for _suf in (".luxar.zarr", ".gsplats.zarr", ".zarr"):
        if _bundle_stem.endswith(_suf):
            _bundle_stem = _bundle_stem[: -len(_suf)]
            break
    bundle_name = name or _bundle_stem or "LuxarScene"
    viewer_dist = get_viewer_dist_path()

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
    aprint("       luxar demo --profile 3g")
    aprint("       luxar viewer --data data.luxar.zarr --profile satellite")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app()
