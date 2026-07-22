"""HTTP serving + path-safety helpers for the luxar CLI.

These are the Typer-free building blocks the ``serve`` / ``viewer`` / ``demo``
commands (and the ``gsplat view`` command) share: CORS configuration, the
loopback / sensitive-path guards, the directory-listing static handler, and the
foreground/background server runners. They live here — not in ``cli/main.py`` —
so subcommand modules can reach ``_serve_data`` / ``_serve_viewer`` without
importing ``main`` (which would form a cycle through the gsplat subcommand
mount at the bottom of ``main.py``).

``cli/main.py`` re-exports the public names so existing
``luxar.cli.main.create_server_app`` / ``._serve_data`` / ``._serve_viewer``
imports and ``unittest.mock.patch`` targets keep resolving unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, MutableMapping, Optional, cast

import uvicorn
from arbol import aprint
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp

from .network_simulation import (
    NetworkSimulationMiddleware,
    has_network_simulation,
)
from .utils import (
    _DEFAULT_CORS_ORIGIN,
    _LOCAL_CORS_ORIGIN_REGEX,
    get_viewer_dist_path,
)
from .utils import (
    open_browser as open_browser_func,
)


def _add_cors(api: FastAPI, cors_origin: str = _DEFAULT_CORS_ORIGIN) -> None:
    """Add CORS middleware to a FastAPI application.

    ``cors_origin="local"`` is the safe development default: browser clients
    on localhost/127.0.0.1/::1 may read served data from any port. Pass
    ``"*"`` explicitly to allow any origin; credentials are disabled for that
    mode because wildcard origins and credentials are an unsafe combination.

    Args:
        api: FastAPI app to extend.
        cors_origin: Origin to allow. ``"local"`` allows loopback origins.
            ``"*"`` allows any origin without credentials. Comma-separated
            explicit origins are also accepted.
    """
    origin = cors_origin.strip() or _DEFAULT_CORS_ORIGIN
    allow_origins: list[str]
    allow_origin_regex: str | None = None
    allow_credentials = True

    if origin == _DEFAULT_CORS_ORIGIN:
        allow_origins = []
        allow_origin_regex = _LOCAL_CORS_ORIGIN_REGEX
    elif origin == "*":
        allow_origins = ["*"]
        allow_credentials = False
    else:
        allow_origins = [item.strip() for item in origin.split(",") if item.strip()]

    api.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Genuine loopback addresses only. The all-interfaces sentinel (0.0.0.0 / ::)
# is deliberately NOT here: binding it exposes the server on every network
# interface, which is exactly the case the LAN-exposure warning must fire on.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _warn_if_lan_exposed(host: str, cors_origin: str) -> None:
    """Warn when the user is binding a non-loopback address AND opening CORS to all.

    ``host`` is the bind address. Anything that is not a genuine loopback
    address — an all-interfaces sentinel, a routable LAN address, or a
    hostname — reaches the network and triggers the warning.
    """
    if cors_origin.strip() != "*":
        return
    bind_host = host.strip().lower()
    if bind_host in _LOOPBACK_HOSTS:
        return
    aprint(
        f"⚠️  Serving on host={host} with --cors-origin '*'. "
        "This exposes the data to anything that can reach this machine on "
        "the network. Pass --cors-origin local (or an explicit origin) "
        "if that was not intended."
    )


def _path_is_within(path: Path, base: Path) -> bool:
    """Return True if ``path`` resolves inside ``base``."""
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _is_sensitive_serve_path(path: Path) -> bool:
    """Return True for obvious system locations that should not be served.

    This intentionally does not block ordinary project directories under a
    user's home or temporary directory. It only catches filesystem roots and
    well-known sensitive system roots.
    """
    resolved = path.resolve()
    if resolved == Path(resolved.anchor):
        return True

    # System roots that are never legitimate to serve over a dev HTTP server.
    # /home and /Users are intentionally NOT on this list — users routinely
    # store project data there. /var and /private/var are also omitted because
    # macOS' TMPDIR resolves under /private/var/folders/... and the test
    # suite (and many user workflows) legitimately serves from tmpdirs.
    sensitive_roots = [
        Path("/etc"),
        Path("/private/etc"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev"),
        Path("/root"),
        Path("/usr"),
        Path("/boot"),
    ]
    for root in sensitive_roots:
        try:
            root_resolved = root.resolve(strict=False)
            if resolved == root_resolved or resolved.is_relative_to(root_resolved):
                return True
        except OSError:
            continue
    return False


def _validate_serve_path(path: Path, *, allow_sensitive_path: bool = False) -> None:
    """Validate that a path is safe enough for the local development server."""
    if _is_sensitive_serve_path(path) and not allow_sensitive_path:
        raise ValueError(
            f"Refusing to serve sensitive system path: {path.resolve()}. "
            "Pass --allow-sensitive-path if you really intend to expose it."
        )


class DirectoryListingStaticFiles(StaticFiles):
    """Static files handler with JSON directory listing support."""

    async def get_response(self, path: str, scope: MutableMapping[str, Any]) -> Any:
        """Override to provide directory listing."""
        from starlette.responses import Response

        # Handle OPTIONS requests for CORS
        if scope.get("method") == "OPTIONS":
            return Response(status_code=204)

        if self.directory is None:
            raise ValueError("Directory not set")

        base_path = Path(self.directory).resolve()
        full_path = (base_path / path).resolve() if path else base_path
        if not _path_is_within(full_path, base_path):
            return Response("Forbidden", status_code=403)

        # If it's a directory, provide listing
        if full_path.exists() and full_path.is_dir():
            # Check Accept header
            headers = dict(scope.get("headers", []))
            accept = headers.get(b"accept", b"").decode("utf-8")

            # Zarr dot-files that should appear in directory listings
            _ZARR_DOT_FILES = {".zgroup", ".zattrs", ".zarray", ".zmetadata"}

            # Generate directory listing
            entries = []
            try:
                for item in sorted(full_path.iterdir()):
                    # Skip hidden files except zarr metadata files
                    if item.name.startswith(".") and item.name not in _ZARR_DOT_FILES:
                        continue

                    item_type = "directory" if item.is_dir() else "file"
                    # Check if it's a zarr directory
                    if item.is_dir() and item.name.endswith(".zarr"):
                        item_type = "zarr"
                    elif item.is_dir() and (item / ".zgroup").exists():
                        item_type = "zarr"

                    entries.append(
                        {
                            "name": item.name,
                            "type": item_type,
                            "size": item.stat().st_size if item.is_file() else None,
                        }
                    )
            except PermissionError:
                return Response("Permission denied", status_code=403)

            # Return JSON for API requests
            if "application/json" in accept:
                from starlette.responses import JSONResponse

                return JSONResponse({"entries": entries})

            # Return HTML for browser requests
            import html
            from urllib.parse import quote

            from starlette.responses import HTMLResponse

            html_content = "<html><body><h1>Directory Listing</h1><ul>"
            if path:
                html_content += '<li><a href="../">../</a></li>'
            for entry in entries:
                name = str(entry["name"])
                if entry["type"] in ("directory", "zarr"):
                    name += "/"
                safe_name = html.escape(name)
                safe_href = quote(name, safe="/")
                html_content += f'<li><a href="{safe_href}">{safe_name}</a></li>'
            html_content += "</ul></body></html>"
            return HTMLResponse(content=html_content)

        # Fall back to default static file serving
        return await super().get_response(path, scope)


def create_server_app(
    path: str,
    serve_viewer: bool = False,
    *,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
    allow_sensitive_path: bool = False,
) -> FastAPI:
    """Create a FastAPI server application for serving Zarr data.

    This function is used by both the CLI and integration tests to create
    a configured server instance.

    Args:
        path: Path to directory or Zarr dataset to serve
        serve_viewer: Whether to include viewer static files (not used in basic tests)
        cors_origin: Allowed CORS origin. ``"local"`` allows loopback origins.
        allow_sensitive_path: If True, permit serving system directories.

    Returns:
        FastAPI application instance
    """
    serve_path = Path(path)
    _validate_serve_path(serve_path, allow_sensitive_path=allow_sensitive_path)

    api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)
    _add_cors(api, cors_origin)

    # Add health check endpoint
    @api.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Mount the static files handler with directory listing
    api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

    return api


def _serve_viewer(
    host: str,
    port: int,
    data_url: Optional[str] = None,
    open_browser_flag: bool = True,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
) -> None:
    """Internal function to serve the viewer.

    Args:
        host: Host interface for the viewer HTTP server.
        port: Port for the viewer HTTP server.
        data_url: Optional data URL appended as ``?src=<data_url>`` to the
            opened viewer URL (trailing slash stripped to avoid double-slash
            in viewer fetches).
        open_browser_flag: If True, open the viewer URL in the system browser
            shortly after the server starts.
        cors_origin: Allowed CORS origin (see :func:`_add_cors`).
    """
    viewer_dist = get_viewer_dist_path()

    api = FastAPI(title="Luxar Viewer", docs_url=None, redoc_url=None)
    _add_cors(api, cors_origin)

    # Mount viewer static files
    api.mount("/", StaticFiles(directory=str(viewer_dist), html=True))

    # Construct viewer URL - ensure data_url has no trailing slash
    if data_url:
        # Strip trailing slash from data_url to prevent double-slash in viewer requests
        data_url_clean = data_url.rstrip("/")
        viewer_url = f"http://{host}:{port}/?src={data_url_clean}"
    else:
        viewer_url = f"http://{host}:{port}/"

    aprint(f"🌐 Viewer available at: {viewer_url}")

    if open_browser_flag:
        time.sleep(1)
        open_browser_func(viewer_url)

    uvicorn.run(api, host=host, port=port, reload=False, log_level="warning")


def _resolve_mount_root(path: Path) -> Path:
    """Root directory actually mounted by the data server for ``path``.

    A directory (zarr store or plain folder) is mounted itself; a single file
    is served from its parent directory. Validation must run against this
    root, not the argument, so the guard covers what is actually exposed.
    """
    return path if path.is_dir() else path.parent


def _serve_data(
    path: Path,
    host: str,
    port: int,
    bandwidth_mbps: Optional[float] = None,
    latency_ms: Optional[float] = None,
    jitter_percent: float = 0.0,
    packet_loss_rate: float = 0.0,
    allow_sensitive_path: bool = False,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
) -> None:
    """Internal function to serve data in background.

    The mount root is the dataset itself (a ``.zarr`` store is served AT the
    server root, so the data URL is ``http://host:port`` with no store-name
    suffix). Serving the parent directory would expose every sibling file of
    the dataset over HTTP.

    Args:
        path: Path to data directory or zarr file
        host: Host address
        port: Port number
        bandwidth_mbps: Bandwidth limit in Mbps (optional)
        latency_ms: Latency in milliseconds (optional)
        jitter_percent: Jitter as percentage (0.0-1.0)
        packet_loss_rate: Packet loss rate (0.0-1.0)
        allow_sensitive_path: Permit serving system paths.
        cors_origin: Allowed CORS origin (see :func:`_add_cors`).
    """
    asgi_app = _build_data_app(
        path,
        bandwidth_mbps=bandwidth_mbps,
        latency_ms=latency_ms,
        jitter_percent=jitter_percent,
        packet_loss_rate=packet_loss_rate,
        allow_sensitive_path=allow_sensitive_path,
        cors_origin=cors_origin,
    )

    aprint(f"💾 Data server running at http://{host}:{port}")

    uvicorn.run(asgi_app, host=host, port=port, reload=False, log_level="warning")


def _build_data_app(
    path: Path,
    *,
    bandwidth_mbps: Optional[float] = None,
    latency_ms: Optional[float] = None,
    jitter_percent: float = 0.0,
    packet_loss_rate: float = 0.0,
    allow_sensitive_path: bool = False,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
) -> ASGIApp:
    """Build the data-server ASGI app for ``path`` (see :func:`_serve_data`).

    Split out so tests can exercise the real mount/validation logic with a
    ``TestClient`` instead of a live uvicorn server.
    """
    serve_path = _resolve_mount_root(path)
    _validate_serve_path(serve_path, allow_sensitive_path=allow_sensitive_path)

    api = FastAPI(title="Luxar Data Server", docs_url=None, redoc_url=None)
    _add_cors(api, cors_origin)

    api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

    # Wrap with network simulation if enabled
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

    return asgi_app
