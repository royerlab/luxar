#!/usr/bin/env python3
"""Serve this exported Luxar scene locally.

Usage:
    python3 serve.py [--port PORT] [--host HOST] [--no-open]
    python3 serve.py --control [--control-token SECRET]

Requirements:
    Python 3.9+ (stdlib only -- no pip install needed)

With ``--control`` this also hosts the remote-control relay, so the exported
folder can run a kiosk: the big display opens the viewer, a tablet opens
``viewer/control.html``, and a tap on the tablet moves the display. Off by
default, because a folder you double-click should not start listening for
anything that wants to drive it.
"""

# Maintainer note (this file is shipped verbatim, so the note travels):
# the template lives at `luxar/cli/_export_serve_template.py` and is copied by
# `luxar export`, with DATA_DIR_NAME and TITLE_QUERY substituted. It is a real
# module rather than a string inside the exporter so that ruff, mypy and the
# test suite see it -- a 400-line WebSocket relay hidden in an f-string is
# unreviewable. It must stay STDLIB-ONLY: this folder gets zipped and emailed
# to people who have never installed Luxar, and `pip install` is not a step
# they agreed to.

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.server
import json
import socket
import struct
import sys
import threading
import webbrowser
from collections.abc import Iterator
from functools import partial
from io import BufferedIOBase
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

# `luxar_qr` only exists in an EXPORTED folder, where the exporter copies
# `luxar/cli/_qr.py` beside this script. mypy cannot see it from the source
# tree, hence the ignore; the try/except is what makes a folder missing the
# file still serve, printing the panel URL without a QR under it.
try:
    from luxar_qr import (  # type: ignore[import-not-found]
        qr_ascii,
        qr_matrix,
        qr_png_bytes,
    )
except ImportError:  # pragma: no cover - exercised by deleting the file
    qr_matrix = None

SCRIPT_DIR = Path(__file__).resolve().parent

# Substituted by `luxar export` (luxar/cli/export.py). The defaults are what an
# export with no options produces, so this script runs as-is from a checkout.
DATA_DIR_NAME = "data"
#: Whether the exported scene declares a control panel (a `control_panel` in
#: its viewer config). Substituted at export time. It decides the DEFAULTS
#: only: a scene authored to be driven from a tablet starts its relay and
#: binds the network without being asked, because the alternative is an
#: operator reading a README at an exhibit. A scene with no panel behaves as
#: before -- loopback, no relay -- since a folder you double-click should not
#: start listening for anything that wants to drive it. `--no-control` and
#: `--host 127.0.0.1` override either way.
HAS_CONTROL_PANEL = False
TITLE_QUERY = ""

# ── wire contract ───────────────────────────────────────────────────────────
# These mirror `control-contract/contract.yaml`, the single source the Python
# hub, the viewer's client and the Go launcher's relay are all generated from.
# This script cannot import that projection -- it must run with nothing but the
# stdlib -- so the values are written out, and
# `cli/tests/test_export_control_relay.py::test_contract_values_match_the_contract`
# fails if they ever disagree with it. That test is the fourth party's drift
# gate; do not "fix" a mismatch by editing this block alone.

#: Query value attaching a socket as the party being driven.
ROLE_VIEWER = "viewer"
#: Query value attaching a socket as the party doing the driving.
ROLE_CONTROLLER = "controller"
#: Query-parameter names on the socket URL.
PARAM_ROLE = "role"
PARAM_TOKEN = "token"
#: RFC 6455 "policy violation" -- an unknown role, a bad token, a foreign
#: origin. Sent AFTER the handshake completes, so the client sees a code it can
#: report rather than a bare HTTP error it cannot tell from "no relay here".
CLOSE_POLICY_VIOLATION = 1008
#: JSON-RPC.
JSONRPC_VERSION = "2.0"
EVENT_METHOD = "event"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
NO_VIEWER = -32001
#: Requests in flight per controller before the oldest is evicted.
MAX_PENDING_PER_CONTROLLER = 64
#: Largest frame accepted from a peer.
MAX_FRAME_BYTES = 16777216

# ── WebSocket plumbing ──────────────────────────────────────────────────────
# RFC 6455 close codes this script decides with. They are NOT in the shared
# contract: no peer branches on them, they are what a broken or hostile socket
# gets told on its way out, and putting them in the contract would imply the
# other implementations handle them.
_CLOSE_PROTOCOL_ERROR = 1002
_CLOSE_UNSUPPORTED_DATA = 1003
_CLOSE_INVALID_PAYLOAD = 1007
_CLOSE_TOO_BIG = 1009

#: The magic string every RFC 6455 handshake is keyed with.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_OP_CONTINUATION = 0x0
_OP_TEXT = 0x1
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA

#: Bind addresses that mean "every interface" rather than a destination.
_ALL_INTERFACES = frozenset({"0.0.0.0", "::", ""})  # nosec B104

#: The path the viewer's `?control` flag derives. Keep in step with
#: `CONTROL_SOCKET_PATH` in the viewer's `config/url-params.ts`.
CONTROL_PATH = "/control"


class _SocketClosed(Exception):
    """The peer went away. Not an error: every socket ends this way."""


class _ProtocolError(Exception):
    """The peer sent something RFC 6455 does not allow."""

    def __init__(self, code: int, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


def _read_exact(stream: BufferedIOBase, count: int) -> bytes:
    """Read exactly ``count`` bytes, or raise :class:`_SocketClosed`.

    ``count`` of zero returns early rather than reading: a zero-length frame is
    legal and ordinary (an empty ping, a close with no code), and ``read(0)``
    returns ``b""`` — which is indistinguishable from end-of-stream.
    """
    if count == 0:
        return b""
    data = stream.read(count)
    if not data or len(data) != count:
        raise _SocketClosed()
    return data


def _unmask(payload: bytes, key: bytes) -> bytes:
    """XOR ``payload`` with the repeating 4-byte ``key``.

    Done as one big-integer XOR rather than a per-byte loop: the per-byte form
    is a 16-million-iteration Python loop at the frame cap, which is long
    enough to stall a kiosk on a single large frame.
    """
    length = len(payload)
    if length == 0:
        return b""
    mask = (key * (length // 4 + 1))[:length]
    xored = int.from_bytes(payload, "big") ^ int.from_bytes(mask, "big")
    return xored.to_bytes(length, "big")


def _read_frame(stream: BufferedIOBase) -> tuple[bool, int, bytes]:
    """Read one WebSocket frame as ``(fin, opcode, payload)``."""
    first, second = _read_exact(stream, 2)
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F
    if length == 126:
        length = int(struct.unpack("!H", _read_exact(stream, 2))[0])
    elif length == 127:
        length = int(struct.unpack("!Q", _read_exact(stream, 8))[0])
    if length > MAX_FRAME_BYTES:
        raise _ProtocolError(_CLOSE_TOO_BIG, "frame too large")
    if not masked:
        # RFC 6455 s5.1: a client MUST mask. An unmasked frame is a broken
        # client, or an attempt to smuggle something a proxy will mistake for
        # an HTTP request.
        raise _ProtocolError(_CLOSE_PROTOCOL_ERROR, "unmasked client frame")
    key = _read_exact(stream, 4)
    return fin, opcode, _unmask(_read_exact(stream, length), key)


def _frame_bytes(opcode: int, payload: bytes) -> bytes:
    """Encode one unfragmented server-to-client frame (never masked)."""
    header = bytearray([0x80 | opcode])
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += struct.pack("!H", length)
    else:
        header.append(127)
        header += struct.pack("!Q", length)
    return bytes(header) + payload


def _start_or_continue(opcode: int, received: int) -> int:
    """Track which data opcode a fragmented message started with."""
    if received == _OP_CONTINUATION:
        if opcode == 0:
            raise _ProtocolError(_CLOSE_PROTOCOL_ERROR, "continuation without start")
        return opcode
    if received not in (_OP_TEXT, 0x2):
        raise _ProtocolError(_CLOSE_PROTOCOL_ERROR, f"opcode {received}")
    if opcode != 0:
        raise _ProtocolError(_CLOSE_PROTOCOL_ERROR, "interleaved data frame")
    return received


class _Peer:
    """One attached socket, plus the lock that serialises writes to it.

    The lock is load-bearing: a reply to this controller and a broadcast event
    can be written from two threads at once, and two threads writing one
    WebSocket interleave their frames into garbage.
    """

    def __init__(self, reader: BufferedIOBase, writer: BufferedIOBase) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = threading.Lock()

    def _write(self, opcode: int, payload: bytes) -> bool:
        with self._write_lock:
            try:
                self._writer.write(_frame_bytes(opcode, payload))
                self._writer.flush()
            except OSError:
                return False
        return True

    def send(self, text: str) -> bool:
        """Send one text frame. ``False`` if the peer has gone."""
        return self._write(_OP_TEXT, text.encode("utf-8"))

    def close(self, code: int, reason: str) -> None:
        """Send a close frame. Best effort -- the peer may already be gone."""
        payload = struct.pack("!H", code) + reason.encode("utf-8")[:123]
        self._write(_OP_CLOSE, payload)

    def _read_message(self) -> bytes:
        """Read one complete message, answering pings and joining fragments."""
        parts: list[bytes] = []
        opcode = 0
        total = 0
        while True:
            fin, received, payload = _read_frame(self._reader)
            if received == _OP_CLOSE:
                raise _SocketClosed()
            if received == _OP_PING:
                self._write(_OP_PONG, payload)
                continue
            if received == _OP_PONG:
                continue
            opcode = _start_or_continue(opcode, received)
            total += len(payload)
            if total > MAX_FRAME_BYTES:
                # A fragmented message can exceed the per-frame cap in pieces.
                raise _ProtocolError(_CLOSE_TOO_BIG, "message too large")
            parts.append(payload)
            if fin:
                if opcode != _OP_TEXT:
                    raise _ProtocolError(_CLOSE_UNSUPPORTED_DATA, "text frames only")
                return b"".join(parts)

    def messages(self) -> Iterator[str]:
        """Yield text messages until the peer closes or misbehaves."""
        while True:
            try:
                payload = self._read_message()
            except _SocketClosed:
                return
            except _ProtocolError as error:
                self.close(error.code, error.reason)
                return
            try:
                yield payload.decode("utf-8")
            except UnicodeDecodeError:
                self.close(_CLOSE_INVALID_PAYLOAD, "invalid utf-8")
                return


def origin_allowed(origin: str | None, host: str | None) -> bool:
    """Whether a handshake's ``Origin`` may attach.

    **Cross-Site WebSocket Hijacking.** A WebSocket handshake is not subject to
    the same-origin policy and carries no CORS preflight, so without this check
    any page a visitor happens to open could connect to this relay and drive
    the display -- and binding loopback does not help, because the visitor's own
    browser is inside the trust boundary.

    A MISSING ``Origin`` is allowed: no non-browser client sends one, and
    refusing them would break every script. This check exists to stop a
    *browser* being used as the attacker's proxy.
    """
    if origin is None:
        return True
    try:
        parsed = urlsplit(origin.strip())
    except ValueError:
        # `urlsplit` RAISES on an unterminated bracket like `http://[::1`, so
        # without this an unauthenticated client could crash the handshake with
        # one header, before the token is ever checked.
        return False
    if not parsed.netloc or parsed.scheme not in ("http", "https"):
        # Unparseable, or `null` from a sandboxed iframe. Never matches.
        return False
    if host is None:
        return False
    return parsed.netloc.lower() == host.strip().lower()


def _error_frame(request_id: Any, code: int, message: str) -> str:
    """A JSON-RPC error response. ``request_id`` may be ``None``."""
    return json.dumps(
        {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


class Relay:
    """Forwards JSON-RPC frames between one display and many controllers.

    A dumb relay, like its two siblings (`luxar.cli.control_hub` in Python and
    `hub.go` in the native launcher): it understands no methods, keeps no scene
    state, and only remaps request ids so two controllers numbering from 1
    cannot receive each other's replies.

    Unlike the Python hub -- which runs on one asyncio loop and needs no
    locking -- every socket here owns a thread, so the registries are guarded.
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._lock = threading.Lock()
        # The display. Only one is useful; a second replaces it, because a
        # reloaded page must be able to take over from its own dead socket.
        self._viewer: _Peer | None = None
        self._controllers: dict[int, _Peer] = {}
        self._next_peer = 0
        # hub id -> (controller key, the id that controller used).
        self._pending: dict[int, tuple[int, Any]] = {}
        self._next_request = 0

    # ── attachment ──────────────────────────────────────────────────────────

    def authorized(self, token: str | None) -> bool:
        """Whether a socket presenting ``token`` may attach.

        Constant-time compare over UTF-8 bytes: a relay on a LAN kiosk is
        reachable by anything on that LAN, ``==`` on a secret leaks its prefix
        through timing, and ``hmac.compare_digest`` raises on a non-ASCII
        ``str`` -- which would turn a wrong password into a traceback.
        """
        if self._token is None:
            return True
        if token is None:
            return False
        return hmac.compare_digest(token.encode("utf-8"), self._token.encode("utf-8"))

    def refusal(
        self,
        role: str,
        token: str | None,
        origin: str | None,
        host: str | None,
    ) -> str | None:
        """Why this handshake may not attach, or ``None`` if it may.

        Called from :meth:`serve` rather than from the route, so a route cannot
        forget it. A check a caller can forget is a check that eventually is
        forgotten, and its failure mode here is an open socket.
        """
        if role not in (ROLE_VIEWER, ROLE_CONTROLLER):
            # Refused rather than defaulted: a typo'd `?role=viewr` silently
            # attaching as a CONTROLLER leaves a display that never receives a
            # call and reports no error.
            return f"unknown role {role!r}"
        if not self.authorized(token):
            return "bad token"
        if self._token is None and not origin_allowed(origin, host):
            # A configured token supersedes the origin check, so an
            # authenticated split-origin exhibit works.
            return "cross-origin handshake"
        return None

    def serve(
        self,
        peer: _Peer,
        role: str,
        token: str | None = None,
        origin: str | None = None,
        host: str | None = None,
    ) -> None:
        """Attach ``peer`` as ``role`` and relay until it disconnects."""
        refusal = self.refusal(role, token, origin, host)
        if refusal is not None:
            peer.close(CLOSE_POLICY_VIOLATION, refusal)
            return
        if role == ROLE_VIEWER:
            self._serve_viewer(peer)
        else:
            self._serve_controller(peer)

    @property
    def viewer_attached(self) -> bool:
        """Whether a display is attached. Exposed for tests and status lines."""
        with self._lock:
            return self._viewer is not None

    @property
    def controller_count(self) -> int:
        """Attached controllers. Exposed for tests and status lines."""
        with self._lock:
            return len(self._controllers)

    @property
    def pending_count(self) -> int:
        """Requests awaiting a reply. Exposed so a test can pin the cap."""
        with self._lock:
            return len(self._pending)

    # ── viewer side ─────────────────────────────────────────────────────────

    def _serve_viewer(self, peer: _Peer) -> None:
        with self._lock:
            self._viewer = peer
        try:
            for text in peer.messages():
                self._from_viewer(text)
        finally:
            with self._lock:
                if self._viewer is peer:
                    self._viewer = None

    def _from_viewer(self, text: str) -> None:
        """A reply headed for one controller, or an event headed for all."""
        frame = _decode(text)
        if frame is None:
            return  # Malformed, from our own display: nothing to answer.
        if "result" in frame or "error" in frame:
            self._route_reply(frame)
        elif frame.get("method") == EVENT_METHOD and frame.get("id") is None:
            self._broadcast(text)

    def _route_reply(self, frame: dict[str, Any]) -> None:
        """Restore the asking controller's own request id and answer it."""
        hub_id = _as_int(frame.get("id"))
        if not isinstance(hub_id, int) or isinstance(hub_id, bool):
            return
        with self._lock:
            pending = self._pending.pop(hub_id, None)
            if pending is None:
                return  # A duplicate, or the asker has gone.
            key, original_id = pending
            target = self._controllers.get(key)
        if target is not None:
            target.send(json.dumps({**frame, "id": original_id}))

    def _broadcast(self, text: str) -> None:
        with self._lock:
            targets = list(self._controllers.values())
        for target in targets:
            target.send(text)

    # ── controller side ─────────────────────────────────────────────────────

    def _serve_controller(self, peer: _Peer) -> None:
        with self._lock:
            self._next_peer += 1
            key = self._next_peer
            self._controllers[key] = peer
        try:
            for text in peer.messages():
                self._from_controller(key, peer, text)
        finally:
            self._drop_controller(key)

    def _drop_controller(self, key: int) -> None:
        """Forget a controller AND the replies it will never read."""
        with self._lock:
            self._controllers.pop(key, None)
            for hub_id in [k for k, (c, _) in self._pending.items() if c == key]:
                self._pending.pop(hub_id, None)

    def _from_controller(self, key: int, peer: _Peer, text: str) -> None:
        """A request or notification headed for the display."""
        try:
            frame = json.loads(text)
        except ValueError:
            peer.send(_error_frame(None, PARSE_ERROR, "parse error"))
            return
        if not isinstance(frame, dict):
            peer.send(_error_frame(None, INVALID_REQUEST, "frame must be an object"))
            return
        if "method" not in frame:
            # A controller has nothing to answer; a stray response is noise.
            return
        request_id = frame.get("id")
        with self._lock:
            viewer = self._viewer
            hub_id = 0
            if viewer is not None and request_id is not None:
                hub_id = self._remember(key, request_id)
        if viewer is None:
            # Not an error to abort on: a kiosk panel opened before the display
            # waits for it. Only a request gets an answer.
            if request_id is not None:
                peer.send(_error_frame(request_id, NO_VIEWER, "no viewer attached"))
            return
        payload = text if request_id is None else json.dumps({**frame, "id": hub_id})
        if not viewer.send(payload) and request_id is not None:
            peer.send(_error_frame(request_id, NO_VIEWER, "viewer unreachable"))

    def _remember(self, key: int, original_id: Any) -> int:
        """Record one outstanding request, evicting this controller's oldest.

        Caller holds the lock. A wedged display never replies, and without a
        cap one dict entry leaks per tap for the lifetime of the process.
        Eviction drops a reply the controller was never going to receive.

        The count is derived from ``_pending`` rather than tracked alongside
        it. `hub.go` keeps a separate per-controller tally because Go has no
        comprehension to recount with; carrying one here too meant four
        maintenance sites for a number nothing read.
        """
        outstanding = sorted(k for k, (c, _) in self._pending.items() if c == key)
        while len(outstanding) >= MAX_PENDING_PER_CONTROLLER:
            self._pending.pop(outstanding.pop(0), None)
        self._next_request += 1
        self._pending[self._next_request] = (key, original_id)
        return self._next_request


def _decode(text: str) -> dict[str, Any] | None:
    """Parse one frame, or ``None`` if it is not a JSON object."""
    try:
        frame = json.loads(text)
    except ValueError:
        return None
    return frame if isinstance(frame, dict) else None


def _as_int(value: Any) -> Any:
    """Hub ids go out as JSON numbers; a client may echo one back as a string."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


# ── HTTP ────────────────────────────────────────────────────────────────────


class ControlServer(http.server.ThreadingHTTPServer):
    """A threaded server, optionally hosting the control relay.

    Threaded for two reasons. A WebSocket occupies its thread for as long as
    the kiosk runs, so on the single-threaded server this replaced the first
    attached panel would have blocked every subsequent request -- including the
    scene's own chunks. And a zarr scene is thousands of small files, which the
    browser fetches several at a time.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: Any, relay: Relay | None):
        # The family follows the bind host, because `ThreadingHTTPServer`
        # hardcodes AF_INET and `find_port` already probes AF_INET6 for a
        # colon-bearing host: without this, `--host ::1` passed the port probe
        # and then died in this constructor with a bare `gaierror` traceback,
        # AFTER the script looked like it was starting.
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, handler)
        self.relay = relay

    #: Client disconnects, not faults. A browser that has changed its mind
    #: closes the socket mid-response and the write raises one of these.
    QUIET_ERRORS = (
        BrokenPipeError,
        ConnectionResetError,
        ConnectionAbortedError,
    )

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Swallow client disconnects; report anything else as usual.

        The viewer CANCELS in-flight chunk fetches constantly -- every camera
        move and every level-of-detail decision abandons requests it no longer
        wants -- and each cancellation aborts a response the server is still
        writing. `socketserver` treats that as a handler crash and prints a
        full traceback, which on a single scene load buries the two URLs and
        the QR under dozens of `BrokenPipeError`s. It is not an error: the
        client asked for less, not the server failing to give it.

        Deliberately narrow. Any other exception still reaches the base
        implementation, so a real handler fault stays loud; wrapping `do_GET`
        in a bare `except` would hide those too.
        """
        if isinstance(sys.exc_info()[1], self.QUIET_ERRORS):
            return
        super().handle_error(request, client_address)


class LuxarHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the viewer and its zarr data from one origin, plus the relay.

    No cross-origin headers: the viewer (/viewer) and its data are served from
    the same host:port, differing only by path, so same-origin fetches need
    none. A wildcard cross-origin policy here would only widen exposure --
    letting any web page read the locally-served scene if it discovered the
    port.
    """

    #: HTTP/1.1 rather than the stdlib default of 1.0, for two reasons. A zarr
    #: scene is thousands of small files, and 1.0 closes the connection after
    #: every one -- so the browser pays a fresh TCP handshake per chunk. And a
    #: strict WebSocket client cannot parse an HTTP/1.0 refusal at all: with
    #: 1.0, a `/control` 404 reached `websockets` as "did not receive a valid
    #: HTTP response" instead of a status code.
    protocol_version = "HTTP/1.1"

    #: Reap an idle keep-alive connection, which 1.1 would otherwise hold a
    #: thread for indefinitely. NOT applied to the relay: see `_handshake`.
    timeout = 60

    def end_headers(self) -> None:
        # SimpleHTTPRequestHandler sends Last-Modified but no Cache-Control, so
        # browsers fall back to HEURISTIC freshness and can serve STALE files
        # after this folder is re-exported in place (same URLs, new bytes).
        # no-cache forces revalidation on mutable scene data AND the unhashed
        # viewer shell (index.html, wasm); unchanged files still return as cheap
        # 304s. Content-hashed viewer chunks under /viewer/assets/ embed a build
        # hash in their URL and are immutable, so they stay cacheable.
        #
        # self.path is unset when a malformed request line trips send_error()
        # before parse_request() assigns it, so read it defensively -- a bad
        # request must still get a clean 400, not an AttributeError traceback.
        path = getattr(self, "path", "").split("?", 1)[0]
        if not path.startswith("/viewer/assets/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress request logging for cleaner output."""

    def do_GET(self) -> None:  # noqa: N802  (http.server's spelling)
        """Serve a file, or take over the socket for the control relay."""
        if _request_path(self.path) == CONTROL_PATH:
            self._serve_control()
            return
        super().do_GET()

    def _relay(self) -> Relay | None:
        server = self.server
        return server.relay if isinstance(server, ControlServer) else None

    def _is_upgrade(self) -> bool:
        upgrade = self.headers.get("Upgrade", "")
        connection = self.headers.get("Connection", "")
        return (
            upgrade.strip().lower() == "websocket"
            and "upgrade" in connection.lower()
            and self.headers.get("Sec-WebSocket-Version") == "13"
        )

    def _handshake(self) -> bool:
        """Complete the RFC 6455 handshake, writing the 101 by hand."""
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "missing Sec-WebSocket-Key")
            return False
        # SHA-1 is not a security choice here: RFC 6455 specifies this exact
        # digest of a public nonce, whose only job is proving the server
        # understood the handshake.
        digest = hashlib.sha1(
            (key + _WS_GUID).encode("utf-8"), usedforsecurity=False
        ).digest()
        self.wfile.write(
            (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {base64.b64encode(digest).decode('ascii')}\r\n"
                "\r\n"
            ).encode("ascii")
        )
        self.wfile.flush()
        self.close_connection = True
        # Clear the keep-alive read timeout for THIS socket. A control socket is
        # idle most of the time by design -- a panel nobody is touching sends
        # nothing -- so leaving `timeout` in force would drop the panel after a
        # minute of quiet and the exhibit would stop working in front of an
        # audience. The timeout exists to reap idle HTTP connections, which this
        # no longer is.
        self.connection.settimeout(None)
        return True

    def _serve_control(self) -> None:
        relay = self._relay()
        if relay is None:
            self.send_error(404, "control relay not enabled (restart with --control)")
            return
        if not self._is_upgrade():
            # Somebody navigated here, or a health check hit it. Say so rather
            # than upgrading nothing and leaving the socket hanging.
            self.send_error(400, "expected a WebSocket upgrade")
            return
        query = parse_qs(_request_query(self.path))
        role = _first(query, PARAM_ROLE) or ROLE_CONTROLLER
        if not self._handshake():
            return
        relay.serve(
            _Peer(self.rfile, self.wfile),
            role,
            _first(query, PARAM_TOKEN),
            self.headers.get("Origin"),
            self.headers.get("Host"),
        )


def _first(query: dict[str, list[str]], name: str) -> str | None:
    """The first value of ``name``, or ``None``."""
    values = query.get(name)
    return values[0] if values else None


def _request_path(target: str) -> str:
    """The path of a request target, split rather than parsed.

    Deliberately NOT ``urlsplit``: ``GET //[::1`` makes it raise
    ``ValueError("Invalid IPv6 URL")``, and a request line is something any
    client can choose. Comparing against one known path needs no parser.
    """
    return target.split("?", 1)[0].split("#", 1)[0]


def _request_query(target: str) -> str:
    """The query string of a request target, split rather than parsed."""
    _, separator, query = target.partition("?")
    return query.split("#", 1)[0] if separator else ""


# ── addresses ───────────────────────────────────────────────────────────────


def primary_lan_address() -> str | None:
    """The local address that would carry traffic off this machine.

    A UDP ``connect()`` sends nothing -- it only fixes the route, which is what
    names the outbound interface. 192.0.2.1 is RFC 5737 documentation space, so
    it is never a real destination.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            address = str(probe.getsockname()[0])
    except OSError:
        return None
    return address or None


def advertised_host(bind_host: str) -> str:
    """Resolve a wildcard bind to an address a browser can actually dial.

    ``http://0.0.0.0:8000`` is not a destination, so printing the bind address
    verbatim hands the user a URL that cannot work from the tablet the panel is
    meant to run on.
    """
    if bind_host not in _ALL_INTERFACES:
        return bind_host
    return primary_lan_address() or "127.0.0.1"


def _url_host(host: str) -> str:
    """Bracket a bare IPv6 address so it is legal inside a URL."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` only ever accepts connections from this machine."""
    return host.strip().lower() in ("127.0.0.1", "::1", "localhost", "[::1]")


def find_port(host: str, start: int = 8000, attempts: int = 100) -> int | None:
    """Find an available port on ``host``, starting from ``start``."""
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    for offset in range(attempts):
        port = start + offset
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    return None


# ── entry point ─────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve exported Luxar scene")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    # Binding every interface is the POINT of a panel scene: the tablet is a
    # different device, so loopback cannot serve it. A scene without a panel
    # still defaults to loopback, and `--host 127.0.0.1` overrides either way.
    # The exposure is documented in the generated README and TESTING notes,
    # with `--control-token` for an untrusted network.
    default_host = (
        "0.0.0.0"  # noqa: S104 # nosec B104 - a tablet cannot reach loopback
        if HAS_CONTROL_PANEL
        else "127.0.0.1"
    )
    parser.add_argument(
        "--host",
        default=default_host,
        help=f"Bind address (default: {default_host}; 127.0.0.1 is local-only)",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Don't open browser automatically"
    )
    parser.add_argument(
        "--control",
        action=argparse.BooleanOptionalAction,
        default=HAS_CONTROL_PANEL,
        help=(
            "Host the remote-control relay for a touch panel "
            f"(default: {'on, this scene has a panel' if HAS_CONTROL_PANEL else 'off'})"
        ),
    )
    parser.add_argument(
        "--control-token",
        default=None,
        help="Secret every control socket must present (advise with --host)",
    )
    return parser.parse_args(argv)


def urls(host: str, port: int, control: bool, token: str | None) -> list[str]:
    """The URLs to print, and the first of them is the one to open."""
    reachable = _url_host(advertised_host(host))
    data_url = f"http://{reachable}:{port}/{DATA_DIR_NAME}"
    viewer_url = f"http://{reachable}:{port}/viewer/?src={data_url}{TITLE_QUERY}"
    if not control:
        return [viewer_url]
    suffix = f"&controlToken={quote(token, safe='')}" if token else ""
    return [
        viewer_url + "&control" + suffix,
        f"http://{reachable}:{port}/viewer/control.html?control{suffix}",
    ]


def print_control_qr(url: str) -> None:
    """Print a scannable QR for ``url`` and leave a PNG copy beside the script.

    The point of kiosk mode is a tablet, and the URL carries a host, a port, a
    path and possibly a token -- too much to read off a screen and type in
    correctly at an exhibit. Half-block glyphs are used so the symbol comes out
    roughly square in a terminal; one module per character is twice as tall as
    wide and many phone cameras will not lock onto it.

    Missing encoder (someone deleted luxar_qr.py) or an unwritable folder are
    both non-fatal: the URL above it is still the real instruction.
    """
    if qr_matrix is None:
        return
    try:
        matrix = qr_matrix(url)
    except Exception:  # noqa: BLE001 - a QR is a convenience, never the gate
        return
    try:
        print()
        print(qr_ascii(matrix, border=2))
    except (OSError, UnicodeError):
        pass
    try:
        target = SCRIPT_DIR / "control-qr.png"
        target.write_bytes(qr_png_bytes(matrix, scale=8, border=4))
        print(f"  (also written to {target.name}, for printing or a second screen)")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    """Serve this folder until interrupted."""
    args = _parse_args(argv)
    if args.control_token and not args.control:
        print("Warning: --control-token requires --control. Ignoring it.")

    port = find_port(args.host, args.port)
    if port is None:
        print(f"Error: No available port found near {args.port}")
        return 1

    relay = Relay(args.control_token) if args.control else None
    handler = partial(LuxarHandler, directory=str(SCRIPT_DIR))
    server = ControlServer((args.host, port), handler, relay)

    addresses = urls(args.host, port, args.control, args.control_token)
    print()
    print("  Display (open this on the big screen)")
    print(f"    {addresses[0]}")
    if args.control:
        print()
        print("  Control panel (open this on the tablet)")
        print(f"    {addresses[1]}")
        print_control_qr(addresses[1])
        if not is_loopback_host(args.host) and not args.control_token:
            print()
            print(
                "  Note: anyone who can reach this machine on the network can "
                "drive the display.\n"
                "  Foreign web pages cannot (the relay checks the request "
                "origin), but a person on\n"
                "  this network who opens the panel URL can. Pass "
                "--control-token SECRET on an\n"
                "  untrusted network, or --host 127.0.0.1 to keep it local."
            )
    print()
    print("Press Ctrl+C to stop")
    # Flushed explicitly: stdout is block-buffered when it is a pipe rather
    # than a terminal, so `python serve.py | tee log` would hold the one thing
    # the operator has to read -- the URLs -- until the buffer filled or the
    # process exited.
    sys.stdout.flush()

    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=[addresses[0]]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
