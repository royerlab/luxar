"""luxar.cli_utils – Helper utilities for the Luxar CLI."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import zarr
from arbol import aprint

from .._zarr_compat import open_group as zarr_open_group

# CORS configuration shared across the CLI. Lives here (not in main.py)
# so subcommand modules like gsplat_commands.py can import it without
# creating a cycle through main.py — main.py also imports gsplat_commands
# to attach the subcommand tree, so the constant must live below both.
#
# 0.0.0.0 is a bind address, not a routable origin — browsers never send it
# as Origin — so it is intentionally absent from this regex.
_LOCAL_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_DEFAULT_CORS_ORIGIN = "local"


def open_browser(url: str, suppress_errors: bool = False) -> bool:
    """Open a URL in the default web browser.

    Args:
        url: URL to open.
        suppress_errors: If True, don't print error messages.

    Returns:
        True if successful, False otherwise.
    """
    try:
        webbrowser.open(url)
        return True
    except Exception as e:
        if not suppress_errors:
            aprint(f"⚠️ Could not open browser: {e}")
        return False


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available for binding.

    The probe sets ``SO_REUSEADDR`` so it matches the bind it predicts: the real
    server is uvicorn's ``loop.create_server``, and asyncio passes
    ``reuse_address=True`` on POSIX. A plain probe is *stricter* than that — a
    lingering non-listening socket left by a previous run (TIME_WAIT /
    FIN_WAIT2) makes it report "busy" on a port the server would have bound
    fine, so a serve-family command shifts to the next port with only a warning
    — which a harness that discards the server's stdout never sees, leaving it to
    wait out its timeout on a port nothing came up on.
    On Linux a LISTENING socket on the same address still conflicts regardless
    of ``SO_REUSEADDR``, so a genuinely running server is still detected.

    The option is applied under asyncio's own condition rather than
    unconditionally: on Windows asyncio deliberately omits it, because there
    ``SO_REUSEADDR`` permits binding over an *active* listener — the probe would
    call an occupied port free and the real bind would then fail hard, trading a
    warned port shift for a crash.

    The address family follows the host: an IPv6 literal (``::1``, ``::``) needs
    an ``AF_INET6`` socket, and probing it on ``AF_INET`` fails for every port —
    which ``pick_port`` reports as "No available ports found", so
    ``luxar serve --host ::1`` died on a completely free port. Only literals are
    switched; names (``localhost``) stay on ``AF_INET`` as before rather than
    inheriting whatever order the resolver happens to return.

    Args:
        port: Port number to check.
        host: Host address to check.

    Returns:
        True if port is available, False if in use.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        if os.name == "posix" and sys.platform != "cygwin":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_available_port(
    start_port: int = 8000,
    max_attempts: int = 100,
    *,
    end_port: Optional[int] = None,
    host: str = "127.0.0.1",
) -> Optional[int]:
    """Find an available port starting from the given port.

    Args:
        start_port: Port to start searching from.
        max_attempts: Maximum number of ports to try.
        end_port: Inclusive upper bound for the search; overrides
            ``max_attempts`` when given (capped at 65535).
        host: Host address the port must be bindable on — pass the same
            host the server will bind so availability is checked on the
            interface actually used.

    Returns:
        Available port number, or None if none found.
    """
    if end_port is not None:
        end_port = min(end_port, 65535)
        if end_port < start_port:
            return None
        max_attempts = end_port - start_port + 1

    for i in range(max_attempts):
        port = start_port + i
        if port > 65535:
            return None
        if check_port_available(port, host):
            return port
    return None


def pick_port(
    requested: int, host: str = "127.0.0.1", label: str = ""
) -> Optional[int]:
    """Resolve a usable port near ``requested``, warning when it shifts.

    Wraps :func:`find_available_port` with the uniform "port busy" warning
    every serve-family command should print, so callers can't silently bind
    a different port than the user asked for.

    Returns:
        The chosen port, or None if no port is available.
    """
    actual = find_available_port(requested, host=host)
    if actual is None:
        aprint(f"❌ Error: No available ports found near {requested}")
        return None
    if actual != requested:
        prefix = f"{label} port".strip().capitalize()
        aprint(f"⚠️  {prefix} {requested} busy, using {actual} instead")
        # Name the squatter when it is one of ours: a forgotten `demo run` in
        # another terminal is the usual culprit, and without this line the
        # user's browser quietly shows the OLD scene on the original port.
        # Pure decoration — a failure here must never break serving.
        try:
            from .demo_runs import describe_port_holder

            holder = describe_port_holder(requested)
        except Exception:
            holder = None
        if holder:
            aprint(f"   {holder}")
    return actual


def dataset_title(path: "Path | str | None") -> Optional[str]:
    """A human-recognizable title for a dataset path, or None.

    Peels an archive extension (``.zip``, ``.tar.gz``, ``.tgz``) and then a
    dataset suffix (``.luxar.zarr``, ``.gsplats.zarr``, ``.zarr``) off the file
    name — ``global_rivers_earth`` from ``global_rivers_earth.luxar.zarr``, and
    equally from ``global_rivers_earth.luxar.zarr.zip``. Both layers matter:
    ``gsplat view`` takes ``.gsplats.zarr.zip`` / ``.gsplats.zarr.tar.gz``
    archives, and a shipped demo scene is a ``.luxar.zarr.zip``.

    Serve-family commands pass the result as the viewer's ``?title=``
    parameter so every browser tab names the scene it shows (several demo/dev
    tabs are otherwise indistinguishable). A scene's authored
    ``viewer_config.title`` overrides it in the viewer.
    """
    if path is None:
        return None
    name = Path(path).name
    for archive in (".tar.gz", ".tgz", ".zip"):
        if name.lower().endswith(archive):
            name = name[: -len(archive)]
            break
    for suffix in (".luxar.zarr", ".gsplats.zarr", ".zarr"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = name.strip()
    return name or None


def append_title_param(viewer_url: str, title: Optional[str]) -> str:
    """Append ``&title=<url-encoded>`` to a viewer URL, or return it unchanged.

    The viewer URL always already carries ``?src=``, so ``&`` is the right
    separator. Shared by every serve-family command so the tab title is
    spelled identically wherever a viewer URL is printed or opened.
    """
    if not title:
        return viewer_url
    return f"{viewer_url}&title={quote(title)}"


def wait_for_server(
    host: str,
    port: int,
    thread: Optional[threading.Thread] = None,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """Poll until ``(host, port)`` accepts a TCP connection.

    Replaces the old fixed ``time.sleep(1)`` startup delays: returns as soon
    as the server is actually reachable, and fails fast when the optional
    ``thread`` running the server has died (e.g. lost a bind race) instead
    of letting the caller open a browser onto a dead server.

    Args:
        host: Host the server binds; all-interfaces sentinels (``0.0.0.0`` /
            ``::``) are probed via loopback.
        port: Port the server binds.
        thread: Server thread to watch; a dead thread returns False early.
        timeout: Total seconds to wait before giving up.
        poll_interval: Delay between connection attempts.

    Returns:
        True once the server accepts a connection, False on timeout or
        thread death.
    """
    # Not a bind: detect the all-interfaces host the user passed and redirect
    # the readiness *probe* to loopback (you cannot connect() to 0.0.0.0 on
    # macOS). The real bind is uvicorn's, with the user's explicit host,
    # guarded by _warn_if_lan_exposed. The nosec waives bandit's B104
    # (hardcoded_bind_all_interfaces) false positive on the compared "0.0.0.0" literal.
    connect_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host  # nosec
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if thread is not None and not thread.is_alive():
            return False
        try:
            with socket.create_connection((connect_host, port), timeout=poll_interval):
                return True
        except OSError:
            time.sleep(poll_interval)
    return False


def check_viewer_built() -> bool:
    """Check if the Luxar viewer is built.

    Returns:
        True if viewer dist directory exists, False otherwise.
    """
    viewer_dist = get_viewer_dist_path()
    return viewer_dist.exists() and (viewer_dist / "index.html").exists()


def _find_dev_repo_root() -> Optional[Path]:
    """Walk up from this file to the dev repo root (a pyproject.toml ancestor).

    Returns None when luxar runs from an installed wheel rather than the
    source tree.
    """
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return None


def get_viewer_dist_path() -> Path:
    """Get the path to the viewer distribution directory.

    Checks two locations in order:
    1. Bundled viewer inside the installed package (``luxar/_viewer_dist/``)
    2. Development source tree (``packages/luxar-viewer/dist/``)

    Returns:
        Path to the viewer dist directory.
    """
    # 1. Check for viewer bundled inside the installed package
    #    cli/utils.py → cli/ → luxar/ → _viewer_dist/
    bundled = Path(__file__).resolve().parent.parent / "_viewer_dist"
    if bundled.is_dir() and (bundled / "index.html").exists():
        return bundled

    # 2. Development: use the source tree layout. Returned even when dist/
    #    doesn't exist yet — a fresh clone has no build, and callers like
    #    ensure_viewer_built()/build_viewer() need the REAL location to
    #    auto-build into (gating on exists() here used to divert fresh
    #    clones to the broken fallback below, so auto-build ran pnpm in a
    #    nonexistent directory and misreported "pnpm not found").
    repo_root = _find_dev_repo_root()
    if repo_root is not None:
        return repo_root / "packages" / "luxar-viewer" / "dist"

    # 3. Last-resort fallback when no pyproject.toml ancestor exists:
    #    hop from .../packages/luxar/src/luxar/cli/utils.py to the repo root
    #    (parents[5], not parents[4] — that pointed at packages/packages/…).
    return Path(__file__).resolve().parents[5] / "packages" / "luxar-viewer" / "dist"


def exit_code_from(returncode: int) -> int:
    """Map a ``subprocess`` returncode to a shell-conventional exit code.

    POSIX ``subprocess.run`` reports a signal-killed child as ``-N`` (signal
    number). Passing that straight to ``typer.Exit``/``sys.exit`` truncates to
    ``256 - N`` (SIGKILL → 247), which no tooling recognizes; the shell
    convention is ``128 + N`` (SIGKILL → 137). Non-negative codes pass through.
    """
    return 128 - returncode if returncode < 0 else returncode


def _dist_is_stale(dist: Path, src_dir: Path) -> bool:
    """True when any viewer source is newer than the built ``dist/index.html``.

    A dist that exists but predates the newest source edit silently serves an
    outdated viewer (the "stale server" pitfall) — an existence check alone
    can't see it. Single ``os.walk`` + ``stat`` over ``src/`` (~1600 files,
    tens of ms); mirrors the viewer test harness's fixture-staleness check
    (``newestPySourceMtime`` in ``global-setup.ts``). Fail-open: unreadable
    entries are skipped, and a missing ``src/`` reads as not-stale.
    """
    index = dist / "index.html"
    if not index.exists() or not src_dir.is_dir():
        return False
    built_at = index.stat().st_mtime
    for dirpath, _dirnames, filenames in os.walk(src_dir):
        for name in filenames:
            try:
                if os.stat(os.path.join(dirpath, name)).st_mtime > built_at:
                    return True
            except OSError:
                continue
    return False


def ensure_viewer_built(auto_build: bool = True) -> bool:
    """Return True when a CURRENT viewer dist exists, building it if possible.

    The serve-family commands previously disagreed on what to do when the
    viewer wasn't built (warn-and-skip vs auto-build vs error). This is the
    single policy: auto-build via pnpm only when running from a dev source
    tree; from an installed wheel the bundled ``_viewer_dist`` should already
    exist, so a missing viewer is a packaging problem, not a build step.

    In a dev tree "built" also means "not stale": a dist older than the newest
    viewer source triggers a rebuild too. A FAILED rebuild of a merely-stale
    dist degrades to serving the stale build — with a loud warning — instead
    of taking serving down entirely.
    """
    built = check_viewer_built()
    repo_root = _find_dev_repo_root()
    stale = (
        built
        and repo_root is not None
        and _dist_is_stale(
            get_viewer_dist_path(), repo_root / "packages" / "luxar-viewer" / "src"
        )
    )
    if built and not stale:
        return True
    if auto_build and repo_root is not None:
        aprint(
            "🔨 Viewer dist is stale (sources changed since the last build). Rebuilding..."
            if stale
            else "🔨 Viewer not built. Building now..."
        )
        if build_viewer():
            return True
        if stale:
            aprint(
                "⚠️  Rebuild FAILED — serving the previous (stale) viewer build. "
                "Fix the build (cd packages/luxar-viewer && pnpm build) to pick "
                "up your source changes."
            )
            return True
        return False
    if built:
        return True
    aprint(
        "❌ Viewer not available: the installed package is missing its bundled "
        "viewer (_viewer_dist). Reinstall luxar, or in a dev tree run: "
        "cd packages/luxar-viewer && pnpm build"
    )
    return False


def build_viewer() -> bool:
    """Build the Luxar viewer using pnpm.

    Returns:
        True if build successful, False otherwise.
    """
    viewer_path = get_viewer_dist_path().parent
    try:
        subprocess.run(
            ["pnpm", "build"],
            cwd=viewer_path,
            capture_output=True,
            text=True,
            check=True,
        )
        aprint("✅ Viewer built successfully")
        return True
    except subprocess.CalledProcessError as e:
        aprint(f"❌ Failed to build viewer: {e.stderr}")
        return False
    except FileNotFoundError:
        aprint("❌ pnpm not found. Please install pnpm first: npm install -g pnpm")
        return False


#: Type indicator per node type in `luxar info`'s tree (`sound` is heard, not drawn).
_TREE_TYPE_ICONS: Dict[str, str] = {
    "scene": "🌐",
    "group": "📁",
    "points": "⚫",
    "lines": "📏",
    "gsplats": "💠",
    "mesh": "🔺",
    "sound": "🔈",
}


def format_tree_node(
    name: str,
    depth: int,
    is_last: bool,
    prefix: str = "",
    node_type: Optional[str] = None,
    attrs: Optional[dict[str, Any]] = None,
) -> str:
    """Format a tree node for display.

    Args:
        name: Node name.
        depth: Current depth in tree.
        is_last: Whether this is the last child.
        prefix: Prefix for the current line.
        node_type: Type of node ("scene", "group", "points", "lines",
            "gsplats", or "mesh"). Unknown values are rendered without a type
            icon.
        attrs: Node attributes to display.

    Returns:
        Formatted tree node string.
    """
    if depth == 0:
        line = f"📊 {name}"
    else:
        connector = "└─" if is_last else "├─"
        line = f"{prefix}{connector} {name}"

    # Add type indicator (unknown types get none)
    icon = _TREE_TYPE_ICONS.get(node_type) if node_type else None
    if icon:
        line += f" {icon}"

    # Add selected attributes
    if attrs:
        important_attrs = []
        if "n_points" in attrs:
            important_attrs.append(f"n={attrs['n_points']:,}")
        if "n_vertices" in attrs:
            important_attrs.append(f"n={attrs['n_vertices']:,}")
        if "n_splats" in attrs:
            important_attrs.append(f"n={attrs['n_splats']:,}")
        # Faces get their own label rather than sharing the `n=` slot: a mesh's
        # vertex count says little about its size on its own (a coarse surface and
        # a dense one can share a vertex budget), and the render cost tracks
        # triangles. Both are shown, so `n=` keeps meaning "primary elements"
        # across every geometry type.
        if "n_faces" in attrs:
            important_attrs.append(f"faces={attrs['n_faces']:,}")
        if "shape" in attrs:
            important_attrs.append(f"shape={attrs['shape']}")
        if "dtype" in attrs:
            important_attrs.append(f"dtype={attrs['dtype']}")
        if important_attrs:
            line += f" [{', '.join(important_attrs)}]"

    return line


def format_memory_size(bytes_size: float) -> str:
    """Format bytes size to human-readable string.

    Args:
        bytes_size: Size in bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def get_zarr_info(store_path: Path, detailed: bool = False) -> dict[str, Any]:
    """Get detailed information about a Zarr store.

    Args:
        store_path: Path to the Zarr store.
        detailed: If True, include extra per-object details (shapes, dtypes).

    Returns:
        Dictionary with store information.
    """
    info: dict[str, Any] = {
        "path": str(store_path),
        "exists": store_path.exists(),
        "size": 0,
        "n_groups": 0,
        "n_arrays": 0,
        "n_points_total": 0,
        "points_objects": [],
        "n_lines_vertices_total": 0,
        "lines_objects": [],
        "n_gsplats_total": 0,
        "gsplats_objects": [],
    }

    if not info["exists"]:
        return info

    try:
        # Calculate total size
        if store_path.is_dir():
            info["size"] = sum(
                f.stat().st_size for f in store_path.rglob("*") if f.is_file()
            )

        # Open and analyze store. Through the compat facade, not `zarr.open_group`
        # directly: zarr 3 answers membership and shapes from `.zmetadata` by
        # default, so an interrupted write whose array directory never landed
        # would still be reported here with its full element count. This command
        # exists to tell the truth about what is on disk.
        root = zarr_open_group(store_path, mode="r")

        def analyze_group(group: zarr.Group, path: str = "") -> None:
            """Recursively analyze a Zarr group."""
            info["n_groups"] += 1

            # Detect geometry type from attrs (set by compiler)
            node_type = group.attrs.get("type", "")

            if node_type == "points" and "positions" in group:
                positions = group["positions"]
                point_info: dict[str, Any] = {
                    "path": path or "/",
                    "n_points": positions.shape[0],
                    "n_dims": positions.shape[1] if len(positions.shape) > 1 else 1,
                    "has_colors": "colors" in group,
                    "has_radii": "radii" in group,
                    "has_sharpness": "sharpnesses" in group,
                }
                if detailed:
                    point_info["shape"] = list(positions.shape)
                    point_info["dtype"] = str(positions.dtype)
                info["points_objects"].append(point_info)
                info["n_points_total"] += point_info["n_points"]

            elif node_type == "lines" and "vertices" in group:
                vertices = group["vertices"]
                line_info: dict[str, Any] = {
                    "path": path or "/",
                    "n_vertices": vertices.shape[0],
                    "n_dims": vertices.shape[1] if len(vertices.shape) > 1 else 1,
                    "has_colors": "colors" in group,
                    "has_widths": "widths" in group,
                }
                if detailed:
                    line_info["shape"] = list(vertices.shape)
                    line_info["dtype"] = str(vertices.dtype)
                info["lines_objects"].append(line_info)
                info["n_lines_vertices_total"] += line_info["n_vertices"]

            elif node_type == "gsplats" and "centers" in group:
                centers = group["centers"]
                gsplat_info: dict[str, Any] = {
                    "path": path or "/",
                    "n_splats": centers.shape[0],
                    "n_dims": centers.shape[1] if len(centers.shape) > 1 else 1,
                    "has_colors": "colors" in group,
                    "has_label_ids": "label_ids" in group,
                }
                if detailed:
                    gsplat_info["shape"] = list(centers.shape)
                    gsplat_info["dtype"] = str(centers.dtype)
                info["gsplats_objects"].append(gsplat_info)
                info["n_gsplats_total"] += gsplat_info["n_splats"]

            # Count arrays
            for _array_name in group.array_keys():
                info["n_arrays"] += 1

            # Recurse into subgroups
            for subgroup_name in group.group_keys():
                analyze_group(group[subgroup_name], f"{path}/{subgroup_name}")

        analyze_group(root)

    except Exception as e:
        info["error"] = str(e)

    return info


def validate_zarr_store(store_path: Path) -> tuple[bool, Optional[str]]:
    """Validate that a path is a valid Zarr store.

    Args:
        store_path: Path to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not store_path.exists():
        return False, "Path does not exist"

    if not store_path.is_dir():
        return False, "Path is not a directory"

    try:
        zarr_open_group(store_path, mode="r")
        return True, None
    except Exception as e:
        return False, f"Not a valid Zarr store: {e}"
