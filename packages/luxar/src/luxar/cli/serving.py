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

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterator,
    MutableMapping,
    Optional,
    Union,
    cast,
)

import uvicorn
from arbol import aprint
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from .control_hub import ROLE_CONTROLLER, ControlHub
from .network_simulation import (
    NetworkSimulationMiddleware,
    has_network_simulation,
)
from .utils import (
    _DEFAULT_CORS_ORIGIN,
    _LOCAL_CORS_ORIGIN_REGEX,
    advertised_host,
    append_title_param,
    authority_hostname,
    get_viewer_dist_path,
    is_loopback_host,
    origin_hostname,
)
from .utils import (
    open_browser as open_browser_func,
)


class _NoCacheMiddleware:
    """Force browser revalidation unless a response declares its own policy.

    Starlette's ``StaticFiles`` sends ``ETag``/``Last-Modified`` but no
    ``Cache-Control``, so browsers fall back to HEURISTIC freshness (a
    fraction of the file's age) and serve stale bytes WITHOUT revalidating.
    Dataset directories are mutable — regenerating a demo/scene in place
    keeps every chunk URL identical — so a heuristically-cached chunk from
    the previous dataset silently poisons the next session (stale geometry
    that no viewer-side cache clearing can fix). ``no-cache`` does NOT
    disable caching: the browser keeps the entry but revalidates with the
    ETag, so unchanged chunks still come back as cheap 304s.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        should_revalidate: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.app = app
        self.should_revalidate = should_revalidate

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if self.should_revalidate is not None and not self.should_revalidate(
            scope.get("path", "")
        ):
            await self.app(scope, receive, send)
            return

        async def send_with_no_cache(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                headers = MutableHeaders(scope=message)
                if "cache-control" not in headers:
                    headers["cache-control"] = "no-cache"
            await send(message)

        await self.app(scope, receive, send_with_no_cache)


def _is_immutable_viewer_asset(path: str) -> bool:
    """True for Vite's content-hashed viewer chunks (the ``assets/`` subtree).

    Their filenames embed a build hash, so the URL changes on every rebuild and
    the bytes at a given URL never change — safe to cache indefinitely. The
    unhashed ``index.html`` shell and the fixed-name ``wasm/`` payloads are
    replaced in place on rebuild, so they must revalidate exactly like mutable
    dataset chunks.
    """
    return "/assets/" in path


class _ViewerStaticFiles(StaticFiles):
    """Revalidate the unhashed viewer shell; leave content-hashed assets cacheable."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve the viewer mount, revalidating everything but the assets subtree."""
        await _NoCacheMiddleware(
            super().__call__,
            should_revalidate=lambda p: not _is_immutable_viewer_asset(p),
        )(scope, receive, send)


class _SameHostCORSMiddleware(CORSMiddleware):
    """CORS that also allows origins on the host the request arrived on.

    Why this exists: `luxar serve --viewer` runs the data server and the
    viewer on two ports of ONE host, and derives the viewer's ``?src=`` from
    the address it bound. So on a non-loopback bind the browser fetches
    ``http://<lan>:8000`` from a page on ``http://<lan>:5173`` — a
    cross-origin pair that the loopback-only default refuses. The symptom is
    a blank viewer and a console full of CORS errors while the data sits
    there perfectly readable by ``curl``, which is the whole LAN kiosk
    story broken.

    Why the check is per-request rather than a computed allow-list: a machine
    can hold several addresses on the same network, and a wildcard bind
    answers on all of them. Baking one guessed address into
    ``allow_origin_regex`` rejects a client that dialled a sibling address —
    verified on a development Mac with Wi-Fi at ``10.0.0.55`` and Ethernet at
    ``10.0.0.146``, where the route probe names only the second. Comparing
    against the ``Host`` the request actually carried needs no guess at all,
    and keeps working for an address added after startup.

    Why it is not a widening: CORS restrains browsers only. Anything that can
    reach a non-loopback bind can already read the data without a browser, so
    this grants a page on the kiosk's own host exactly what ``curl`` has. To
    forge a match an attacker would have to serve their page FROM the kiosk's
    address, which means already being on the machine. ``--cors-origin`` stays
    available to name origins exactly.

    The ``Host`` is carried in a :class:`~contextvars.ContextVar` because
    Starlette's ``is_allowed_origin`` hook receives only the origin, while
    both of its call sites (preflight and response) run inside this
    middleware's ``__call__`` on the same asyncio task — so a ContextVar is
    per-request and safe under concurrency, where an instance attribute
    would not be.
    """

    _request_host: ClassVar[ContextVar[Optional[str]]] = ContextVar(
        "luxar_cors_request_host", default=None
    )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Record this request's ``Host`` for the duration of the request."""
        host = Headers(scope=scope).get("host") if scope["type"] == "http" else None
        token = self._request_host.set(host)
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._request_host.reset(token)

    def is_allowed_origin(self, origin: str) -> bool:
        """Allow the configured origins, plus any port on the request's host."""
        if super().is_allowed_origin(origin):
            return True
        host_header = self._request_host.get()
        if not host_header:
            return False
        # Port excluded on purpose: the two servers are different ports of the
        # same host. `origin_hostname` documents the contrast with the control
        # hub, which compares WITH the port. Both sides fail closed on a
        # malformed header, so neither can widen the allowance.
        request_host = authority_hostname(host_header)
        return bool(request_host) and origin_hostname(origin) == request_host


def _add_cors(
    api: FastAPI,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
    *,
    bind_host: Optional[str] = None,
) -> None:
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
        bind_host: The address this server is bound to. Under
            ``cors_origin="local"`` a non-loopback bind also allows that
            address as an origin on any port — see
            :func:`~luxar.cli.utils.local_cors_origin_regex` for why that is
            required rather than generous. Omitted (or loopback) keeps the
            loopback-only default.
    """
    origin = cors_origin.strip() or _DEFAULT_CORS_ORIGIN
    allow_origins: list[str]
    allow_origin_regex: str | None = None
    allow_credentials = True
    middleware_class: type[CORSMiddleware] = CORSMiddleware

    if origin == _DEFAULT_CORS_ORIGIN:
        allow_origins = []
        allow_origin_regex = _LOCAL_CORS_ORIGIN_REGEX
        # A loopback bind keeps the plain middleware: its Host is already in
        # the regex, so the same-host rule would add nothing.
        if not is_loopback_host(bind_host):
            middleware_class = _SameHostCORSMiddleware
    elif origin == "*":
        allow_origins = ["*"]
        allow_credentials = False
    else:
        allow_origins = [item.strip() for item in origin.split(",") if item.strip()]

    api.add_middleware(
        middleware_class,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges", "ETag"],
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


def _build_viewer_url(
    host: str,
    port: int,
    data_url: Optional[str] = None,
    *,
    title: Optional[str] = None,
    control: bool = False,
    control_token: Optional[str] = None,
) -> str:
    """Build the browser URL shared by viewer serving and ``serve --open``.

    A wildcard bind is resolved to a concrete address first: ``0.0.0.0`` is a
    bind address, not a destination, so a URL containing it is not reachable
    from the tablet the kiosk panel is meant to run on.
    """
    host = advertised_host(host)
    if data_url:
        viewer_url = append_title_param(
            f"http://{host}:{port}/?src={data_url.rstrip('/')}", title
        )
    else:
        viewer_url = f"http://{host}:{port}/"

    if control:
        from urllib.parse import quote

        viewer_url += "&control" if "?" in viewer_url else "?control"
        if control_token:
            viewer_url += f"&controlToken={quote(control_token, safe='')}"
    return viewer_url


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


def _listing_item_type(item: Path) -> str:
    """Classify one directory-listing entry as zarr, directory, or file."""
    if item.is_dir():
        # Check if it's a zarr directory
        if item.name.endswith(".zarr"):
            return "zarr"
        # Fallback for a zarr store whose directory is not named `*.zarr`.
        # BOTH root-group documents count: format 2 writes `.zgroup`, format 3
        # writes `zarr.json`, and Luxar now emits 3 while existing stores stay 2.
        if (item / ".zgroup").exists() or (item / "zarr.json").exists():
            return "zarr"
        return "directory"

    # A zipped store is a FILE, and the viewer reads it in place over range
    # requests (no unpacking) — so it is a dataset, not an archive to download.
    # Listing it as `file` made a perfectly loadable scene invisible in the
    # dataset browser.
    if item.is_file() and item.name.lower().endswith(".zarr.zip"):
        return "zarr"
    return "file"


class DirectoryListingStaticFiles(StaticFiles):
    """Mutable data files with JSON listings and forced revalidation."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve this data mount through the revalidation middleware."""
        await _NoCacheMiddleware(super().__call__)(scope, receive, send)

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

                    entries.append(
                        {
                            "name": item.name,
                            "type": _listing_item_type(item),
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
                if entry["size"] is None:
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
    *,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
    allow_sensitive_path: bool = False,
    bind_host: Optional[str] = None,
) -> FastAPI:
    """Create a FastAPI server application for serving Zarr data.

    This function is used by both the CLI and integration tests to create
    a configured server instance.

    Args:
        path: Path to directory or Zarr dataset to serve
        cors_origin: Allowed CORS origin. ``"local"`` allows loopback origins.
        allow_sensitive_path: If True, permit serving system directories.
        bind_host: Bind address, so a LAN bind also allows its own origin
            under ``cors_origin="local"`` (see :func:`_add_cors`).

    Returns:
        FastAPI application instance
    """
    serve_path = Path(path)
    _validate_serve_path(serve_path, allow_sensitive_path=allow_sensitive_path)

    api = FastAPI(title="Luxar static server", docs_url=None, redoc_url=None)
    _add_cors(api, cors_origin, bind_host=bind_host)

    # Add health check endpoint
    @api.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Mount the static files handler with directory listing
    api.mount("/", DirectoryListingStaticFiles(directory=serve_path, html=True))

    return api


def _add_control_hub(api: FastAPI, *, token: Optional[str] = None) -> ControlHub:
    """Mount the remote-control WebSocket endpoint at ``/control``.

    The hub rides on the VIEWER app because that app serves both the viewer and
    the control panel, so the socket is same-origin with both pages and neither
    needs a URL to find it. Returns the hub (also stashed on ``api.state``) so
    a test can inspect its attachment counts.
    """
    hub = ControlHub(token=token)

    @api.websocket("/control")
    async def control_socket(
        websocket: WebSocket,
        role: str = ROLE_CONTROLLER,
        token: Optional[str] = None,
    ) -> None:
        """Attach one party to the hub. Role and token arrive as query params."""
        await hub.serve(websocket, role, token)

    api.state.control_hub = hub
    return hub


def _build_viewer_app(
    viewer_dist: Path,
    *,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
    control: bool = False,
    control_token: Optional[str] = None,
    bind_host: Optional[str] = None,
) -> FastAPI:
    """Build the viewer-server FastAPI app for ``viewer_dist``.

    Split out so tests can exercise the mount (and its cache policy) with a
    ``TestClient`` instead of a live uvicorn server.
    """
    api = FastAPI(title="Luxar Viewer", docs_url=None, redoc_url=None)
    _add_cors(api, cors_origin, bind_host=bind_host)
    if control:
        _add_control_hub(api, token=control_token)
    # AFTER the control route, and that order is load-bearing: starlette matches
    # routes in declaration order and a ``Mount`` at "/" swallows every path
    # below it, WebSocket paths included. Registered the other way round,
    # /control would 404 through the static handler — which
    # TestViewerAppWiring pins, and that pin is verified to fire.
    api.mount("/", _ViewerStaticFiles(directory=str(viewer_dist), html=True))
    return api


def _serve_viewer(
    host: str,
    port: int,
    data_url: Optional[str] = None,
    open_browser_flag: bool = True,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
    title: Optional[str] = None,
    control: bool = False,
    control_token: Optional[str] = None,
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
        title: Optional browser-tab title appended as ``&title=<title>``
            (URL-encoded) when a ``data_url`` is present. Serve-family
            commands derive it from the dataset file name
            (:func:`luxar.cli.utils.dataset_title`); a scene's authored
            ``viewer_config.title`` overrides it in the viewer.
        control: If True, expose the remote-control hub at ``/control`` and
            hand the viewer ``&control`` so the display attaches to it.
        control_token: Shared secret every control socket must present as
            ``?token=``. ``None`` leaves the hub open to the bound interface.
    """
    viewer_dist = get_viewer_dist_path()

    api = _build_viewer_app(
        viewer_dist,
        cors_origin=cors_origin,
        control=control,
        control_token=control_token,
        bind_host=host,
    )

    viewer_url = _build_viewer_url(
        host,
        port,
        data_url,
        title=title,
        control=control,
        control_token=control_token,
    )

    aprint(f"🌐 Viewer available at: {viewer_url}")
    if control:
        # Same resolution as the viewer URL: a tablet cannot dial 0.0.0.0.
        reachable = advertised_host(host)
        aprint(f"🎛️  Control hub listening at: ws://{reachable}:{port}/control")
        from urllib.parse import quote

        # The touch panel is a second page in the same bundle, so it shares this
        # origin and needs no address of its own — `?control` is a bare flag.
        panel_url = f"http://{reachable}:{port}/control.html?control"
        if control_token:
            panel_url += f"&controlToken={quote(control_token, safe='')}"
        aprint(f"📱 Control panel for a tablet: {panel_url}")
        if host.strip().lower() not in _LOOPBACK_HOSTS and not control_token:
            aprint(
                "⚠️  The control hub is reachable from the network without a token. "
                "Pass --control-token if remote control should be restricted."
            )

    if open_browser_flag:
        # uvicorn.run blocks THIS thread, so a same-thread open must fire
        # before the server binds — the old fixed sleep opened the browser
        # onto an unbound port when startup took >1s. Poll readiness from a
        # helper thread instead and open only once the server answers
        # (mirrors `serve --viewer`'s polled open in main.py).
        def _open_when_ready() -> None:
            from .utils import wait_for_server

            if wait_for_server(host, port, timeout=15.0):
                open_browser_func(viewer_url)
            else:
                aprint(
                    "⚠️  Viewer server did not become ready within 15s — "
                    f"not opening a browser. Try {viewer_url} manually."
                )

        threading.Thread(target=_open_when_ready, daemon=True).start()

    uvicorn.run(api, host=host, port=port, reload=False, log_level="warning")


def _resolve_mount_root(path: Path) -> Path:
    """Resolve and revalidate the directory mounted by the data server.

    Public CLI paths reject files before starting the server, but the mount is
    constructed on a background thread. Re-checking here closes the
    check-then-use window: if the path has become a file, fail instead of
    silently mounting its parent directory and exposing siblings.
    """
    mount_root = path.resolve()
    if not mount_root.is_dir():
        raise ValueError(f"Data mount root must be a directory: {path}")
    return mount_root


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

    The validated mount root is the dataset directory itself (a ``.zarr`` store
    is served AT the server root, so the data URL is ``http://host:port`` with no
    store-name suffix). A file path is rejected again when the server constructs
    the mount; it never falls back to the parent directory, which would expose
    every sibling file of the dataset over HTTP.

    Args:
        path: Path to the data directory or directory-backed Zarr store
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
        bind_host=host,
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
    bind_host: Optional[str] = None,
) -> ASGIApp:
    """Build the data-server ASGI app for ``path`` (see :func:`_serve_data`).

    Split out so tests can exercise the real mount/validation logic with a
    ``TestClient`` instead of a live uvicorn server.
    """
    serve_path = _resolve_mount_root(path)
    _validate_serve_path(serve_path, allow_sensitive_path=allow_sensitive_path)

    api = FastAPI(title="Luxar Data Server", docs_url=None, redoc_url=None)
    _add_cors(api, cors_origin, bind_host=bind_host)

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


# --------------------------------------------------------------------------
# A stoppable server for programmatic callers
# --------------------------------------------------------------------------


class ServedStore:
    """URLs of a store + viewer served by :func:`served_store`."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        #: The data root — what goes into the viewer's ``?src=`` (no trailing slash).
        self.data_url = f"http://{host}:{port}"
        #: The viewer's ``index.html`` (trailing slash, so a relative ``?src`` resolves).
        self.viewer_url = f"http://{host}:{port}/viewer/"


@contextmanager
def served_store(
    path: Union[str, Path],
    *,
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    cors_origin: str = _DEFAULT_CORS_ORIGIN,
    ready_timeout: float = 30.0,
) -> Iterator[ServedStore]:
    """Serve ``path`` at ``/`` and the built viewer under ``/viewer`` — and STOP.

    The serve-family commands all end in a blocking ``uvicorn.run`` on a daemon
    thread, which is right for a terminal and useless for a caller that needs the
    server for the duration of one job (``luxar env bake``). This is the same
    single-process layout ``luxar export`` ships, run through a ``uvicorn.Server``
    handle so the ``with`` block's exit shuts it down, and readiness is a polled
    ``/health`` rather than a sleep. Needs a built viewer
    (:func:`~luxar.cli.utils.ensure_viewer_built`).
    """
    import asyncio
    import time
    import urllib.error
    import urllib.request

    from .utils import pick_port

    viewer_dist = get_viewer_dist_path()
    if not (viewer_dist / "index.html").exists():
        raise RuntimeError(
            f"The viewer is not built ({viewer_dist}); run "
            "`cd packages/luxar-viewer && pnpm build` first."
        )
    chosen = pick_port(port if port is not None else 8900, host, label="server")
    if chosen is None:
        raise RuntimeError(f"No free port near {port or 8900} on {host}.")
    api = create_server_app(str(path), cors_origin=cors_origin, bind_host=host)
    # Mounted BEFORE the data root already on `/`? No — Starlette matches mounts in
    # order, and `/` would swallow `/viewer`, so the viewer goes on first by
    # rebuilding the router order: FastAPI appends routes, hence insert at 0.
    viewer_app = _build_viewer_app(viewer_dist, cors_origin=cors_origin, bind_host=host)
    api.router.routes.insert(0, Mount("/viewer", app=viewer_app))

    config = uvicorn.Config(api, host=host, port=chosen, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()),
        daemon=True,
        name="luxar-served-store",
    )
    thread.start()
    try:
        deadline = time.monotonic() + ready_timeout
        while True:
            try:
                # The URL's scheme is fixed here; only host and selected port vary.
                with urllib.request.urlopen(  # nosec B310
                    f"http://{host}:{chosen}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if not thread.is_alive():
                raise RuntimeError("The server thread exited before becoming healthy.")
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"The server did not become healthy within {ready_timeout:.0f}s."
                )
            time.sleep(0.05)
        yield ServedStore(host, chosen)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
