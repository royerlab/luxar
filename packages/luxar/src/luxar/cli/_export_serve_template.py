#!/usr/bin/env python3
"""Serve this exported Luxar scene locally.

Usage:
    python3 serve.py [--port PORT] [--host HOST] [--no-open]
    python3 serve.py --control [--advertise-host NAME]
    python3 serve.py --control [--control-token SECRET | --no-control-token]

Requirements:
    Python 3.9+ (stdlib only -- no pip install needed)

With ``--control`` this also hosts the remote-control relay, so the exported
folder can run a kiosk: the big display opens the viewer, a tablet opens
``viewer/control.html``, and a tap on the tablet moves the display. Off by
default, because a folder you double-click should not start listening for
anything that wants to drive it.

The panel URL is meant to be printed as a QR, so it stays the same from one
start to the next: off loopback the relay requires a token, generated once
and saved in ``control-token.txt``; with the relay on, a busy port is an error
rather than a move to another port; and ``--advertise-host`` puts a fixed name
in the URL while the server listens on every interface.
"""

# Maintainer note (this file is shipped verbatim, so the note travels):
# the template lives at `luxar/cli/_export_serve_template.py` and is copied by
# `luxar export`, with DATA_DIR_NAME, TITLE_QUERY, LAUNCH_QUERY,
# HAS_CONTROL_PANEL and AUTO_ADVANCE substituted. It is a real
# module rather than a string inside the exporter so that ruff, mypy and the
# test suite see it -- a 400-line WebSocket relay hidden in an f-string is
# unreviewable. It must stay STDLIB-ONLY: this folder gets zipped and emailed
# to people who have never installed Luxar, and `pip install` is not a step
# they agreed to.

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import hmac
import http.server
import ipaddress
import json
import os
import re
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import threading
import webbrowser
from collections.abc import Iterator
from functools import partial
from io import BufferedIOBase
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

# `luxar_qr` only exists in an EXPORTED folder, where the exporter copies
# `luxar/cli/_qr.py` beside this script. mypy cannot see it from the source
# tree, hence the ignore; the try/except is what makes a folder missing the
# file still serve, printing the panel URL without a QR under it.
try:
    from luxar_qr import (  # type: ignore[import-not-found]
        qr_ascii,
        qr_matrix,
        qr_pdf_bytes,
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
#: The scene's `viewer_config.launch` machine settings (renderer, workers,
#: prefetch) as a ready-to-append query fragment. The viewer resolves them
#: before it reads the scene, so they travel on the URL instead.
LAUNCH_QUERY = ""
#: Seconds per stop for `--auto-advance` when it is not given, set by
#: `luxar export --auto-advance`; None keeps the scene's own setting. An
#: explicit `--auto-advance` still wins (0 turns it off).
AUTO_ADVANCE = None
#: Range the viewer accepts for `?textScale=`; mirrored for a clear error.
TEXT_SCALE_RANGE = (0.25, 4.0)
#: Where the relay's default control token is kept, beside this script. A
#: panel's URL -- and the QR printed from it -- carries the token, so the token
#: must outlive a restart for a printed QR to keep working: it is generated on
#: the first start that needs one and reused after that. Deleting the file
#: issues a new token on the next start, which retires every printed QR.
CONTROL_TOKEN_FILE = "control-token.txt"  # nosec B105 - a file name, not a secret
#: Where the display opened by `--chrome` reports what it renders with.
DISPLAY_REPORT_PATH = "/display-report"
_MAX_REPORT_BYTES = 65536
#: Seconds `--chrome` waits for that report before warning that none came.
_DISPLAY_REPORT_WAIT_S = 90.0
#: The browser profile `--chrome` uses, beside this script: its own, so the
#: flags below apply even when Chrome is already running with another one.
CHROME_PROFILE_DIR = ".chrome-display"
#: The panel URL's QR, written beside this script for printing.
QR_FILE = "control-qr.png"
#: The same QR on an A4 sheet with what it is and how to use it, to print and
#: put up beside the display.
SHEET_FILE = "control-panel.pdf"
#: The tokens this script writes and accepts from that file: URL-safe as they
#: stand, so a QR carries the token verbatim.
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,128}")
#: Bytes of randomness in a generated token (16 URL-safe characters).
_TOKEN_BYTES = 12
#: Socket defaults differ on Windows (see `ControlServer.allow_reuse_address`).
_IS_WINDOWS = os.name == "nt"
#: One DNS label; `--advertise-host` is dot-joined labels or an IP literal.
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")

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
VIEWER_ATTACHED_EVENT = "viewer-attached"
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
        self._broadcast(
            json.dumps(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "method": EVENT_METHOD,
                    "params": [VIEWER_ATTACHED_EVENT, None],
                }
            )
        )
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

#: `parse_byte_range`'s answer for a range that starts past the end of the file.
UNSATISFIABLE = (-1, -1)


def parse_byte_range(spec: str, size: int) -> tuple[int, int] | None:
    """Resolve one `Range` header against a file of `size` bytes.

    Returns the inclusive `(start, end)` to serve, :data:`UNSATISFIABLE` for a
    range that begins past the end (a 416), or ``None`` for anything to answer
    with the whole file: a header that does not parse, a unit other than bytes,
    or several ranges at once -- all of which RFC 9110 lets a server ignore.
    """
    unit, _, ranges = spec.strip().partition("=")
    if unit.strip().lower() != "bytes" or "," in ranges:
        return None
    first, dash, last = ranges.strip().partition("-")
    if not dash:
        return None
    try:
        if not first:  # "-N": the final N bytes
            suffix = int(last)
            if suffix <= 0:
                return UNSATISFIABLE
            return (max(size - suffix, 0), size - 1) if size else UNSATISFIABLE
        start = int(first)
        end = int(last) if last else size - 1
    except ValueError:
        return None
    if start < 0:
        return None
    # Past the end first: an open "N-" defaults its end to the last byte, which
    # for N >= size is BEFORE N and would otherwise read as a malformed range.
    if start >= size:
        return UNSATISFIABLE
    if end < start:
        return None
    return (start, min(end, size - 1))


class DisplayReportSlot:
    """The one display report ``--chrome`` expects, keyed by a nonce.

    The page ``serve.py`` opened carries the nonce in its URL and echoes it in
    the report, so a report from anyone else on the network -- or from a web
    page firing a cross-site POST -- is refused, and the first good one claims
    the slot under a lock, so two at once cannot both be printed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nonce: str | None = None

    def expect(self) -> str:
        """Open the slot; returns the nonce the page must echo."""
        with self._lock:
            self._nonce = secrets.token_urlsafe(16)
            return self._nonce

    def pending(self) -> bool:
        """Whether a report is still awaited."""
        with self._lock:
            return self._nonce is not None

    def claim(self, nonce: object) -> bool:
        """Close the slot if ``nonce`` is the awaited one; True exactly once."""
        with self._lock:
            if self._nonce is None or not isinstance(nonce, str):
                return False
            if not hmac.compare_digest(
                nonce.encode("utf-8"), self._nonce.encode("utf-8")
            ):
                return False
            self._nonce = None
            return True


class ControlServer(http.server.ThreadingHTTPServer):
    """A threaded server, optionally hosting the control relay.

    Threaded for two reasons. A WebSocket occupies its thread for as long as
    the kiosk runs, so on the single-threaded server this replaced the first
    attached panel would have blocked every subsequent request -- including the
    scene's own chunks. And a zarr scene is thousands of small files, which the
    browser fetches several at a time.
    """

    daemon_threads = True
    #: `HTTPServer` sets SO_REUSEADDR, so a restart right after Ctrl+C can
    #: rebind a port whose old connections are still draining (TIME_WAIT). On
    #: Windows that option instead lets a SECOND process take a port in use,
    #: so there the bind keeps its default: Windows refuses a second default
    #: bind of the same port with WSAEADDRINUSE, refuses an SO_REUSEADDR
    #: takeover of a wildcard bind with WSAEACCES, and does not hold a port
    #: back for TIME_WAIT. (SO_EXCLUSIVEADDRUSE would, until the display's and
    #: the tablets' old connections close -- the wrong trade for a restart.)
    allow_reuse_address = not _IS_WINDOWS

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
        #: ``--text-scale``, carried into the ``/`` redirect to the viewer.
        self.text_scale: float | None = None
        #: ``--auto-advance``, carried into the ``/`` redirect too.
        self.playback_period: float | None = None
        #: The report ``--chrome`` waits for (see ``LuxarHandler.do_POST``).
        self.display_report = DisplayReportSlot()
        #: ``--advertise-host``: the ``/`` redirect's fallback host when the
        #: request's own ``Host`` header is unusable.
        self.advertise: str | None = None

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
        if getattr(self, "_serving_file", False):
            self.send_header("Accept-Ranges", "bytes")
            # Per response: a keep-alive socket's next reply may not be a file.
            self._serving_file = False
        super().end_headers()

    # Byte ranges (RFC 9110 §14) are what a media element needs to let go of a
    # connection. Without them a <video> cannot drop a paused download and
    # resume it later, so the browser keeps the socket open and stops reading
    # -- and an HTTP/1.1 origin gets six. A tour stepped through quickly left a
    # paused turntable on each of them, and the scene's own chunk fetches then
    # timed out waiting for a socket ("data server unreachable"). One range per
    # request is all a media element asks for; a multi-range request gets the
    # whole file, which RFC 9110 allows.

    def send_head(self) -> Any:
        """Serve a requested byte range as a 206; everything else as before.

        Only the viewer and the scene's data are files this server hands out
        (:func:`servable_path`): the folder also holds the control token, the
        QR and the sheet that carry it, Chrome's display profile (whose history
        holds the token too) and this script, none of which a visitor on the
        network should be able to download.
        """
        self._range_remaining: int | None = None
        if not servable_path(_request_path(self.path)):
            self.send_error(404)
            return None
        file_path = self.translate_path(self.path)
        self._serving_file = os.path.isfile(file_path)
        spec = self.headers.get("Range")
        if (
            spec is None
            or not self._serving_file
            or self.command not in ("GET", "HEAD")
        ):
            return super().send_head()
        stat = Path(file_path).stat()
        last_modified = self.date_time_string(int(stat.st_mtime))
        if_range = self.headers.get("If-Range")
        if if_range is not None and if_range.strip() != last_modified:
            # The client's copy is stale: a range of the new bytes would splice
            # two versions of the file together, so send the whole new one.
            return super().send_head()
        span = parse_byte_range(spec, stat.st_size)
        if span is None:
            return super().send_head()
        if span == UNSATISFIABLE:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{stat.st_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, end = span
        try:
            f = open(file_path, "rb")  # noqa: SIM115 -- returned to copyfile(), closed by do_GET
        except OSError:
            self.send_error(404, "File not found")
            return None
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-type", self.guess_type(file_path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", last_modified)
        self.end_headers()
        self._range_remaining = end - start + 1
        return f

    def copyfile(self, source: Any, outputfile: Any) -> None:
        """Copy only the requested range when one is being served."""
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            super().copyfile(source, outputfile)
            return
        while remaining > 0:
            block = source.read(min(64 * 1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress request logging for cleaner output."""

    def do_GET(self) -> None:  # noqa: N802  (http.server's spelling)
        """Serve a file, or take over the socket for the control relay."""
        if _request_path(self.path) == CONTROL_PATH:
            self._serve_control()
            return
        if _request_path(self.path) == "/":
            self._redirect_to_viewer()
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802  (http.server's spelling)
        """Take the display's report (``?displayReport=<nonce>``), once.

        The page ``serve.py --chrome`` opened POSTs what it renders with; it is
        checked and printed for the operator (:func:`assess_display`). Only
        that page's report is taken: it must echo the nonce, and the first one
        closes the slot. Anything else is a 404 (or 400 / 413 for a report
        that cannot be read), and never reaches the terminal.
        """
        server = self.server
        slot = server.display_report if isinstance(server, ControlServer) else None
        if (
            _request_path(self.path) != DISPLAY_REPORT_PATH
            or slot is None
            or not slot.pending()
        ):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or "")
        except ValueError:
            self.send_error(400, "a display report needs a Content-Length")
            return
        if not 0 < length <= _MAX_REPORT_BYTES:
            self.send_error(413 if length > 0 else 400)
            return
        try:
            report = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeError):
            self.send_error(400, "not a JSON display report")
            return
        if not isinstance(report, dict) or not slot.claim(report.get("nonce")):
            self.send_error(404)
            return
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()
        print_display_check(report)

    def do_HEAD(self) -> None:  # noqa: N802  (http.server's spelling)
        """As ``do_GET``: the folder root redirects to the viewer."""
        if _request_path(self.path) == "/":
            self._redirect_to_viewer()
            return
        super().do_HEAD()

    def _redirect_to_viewer(self) -> None:
        """Send ``/`` to the viewer with the scene loaded, not a file listing.

        The data URL is built from the request's own ``Host``, so the redirect
        is right however the folder was reached -- the printed LAN address, a
        port forward on another port, or a hostname. The header is echoed only
        when it matches ``_HOST_HEADER`` (host or bracketed IPv6, optional port:
        no scheme, path, query or whitespace can ride along), and the Location
        itself is path-relative, so a forged Host can at worst point this one
        client's ``src`` at another server -- never redirect it off this origin.
        Deliberately WITHOUT the control flag and token: anyone who can reach
        ``/`` would otherwise be handed the token that is meant to keep them out
        (``main`` says so at startup when the relay is on).
        """
        server = self.server
        host = self.headers.get("Host", "")
        if not _HOST_HEADER.fullmatch(host):
            address: Any = server.server_address
            fixed = server.advertise if isinstance(server, ControlServer) else None
            reachable = fixed or advertised_host(str(address[0]))
            host = f"{_url_host(reachable)}:{address[1]}"
        ours = server if isinstance(server, ControlServer) else None
        target = viewer_target(
            f"http://{host}/{DATA_DIR_NAME}",
            ours.text_scale if ours else None,
            playback_period=ours.playback_period if ours else None,
        )
        self.send_response(302)
        self.send_header("Location", target)
        self.send_header("Content-Length", "0")
        self.end_headers()

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


#: A ``Host`` header safe to echo into a redirect: a hostname, IPv4 or bracketed
#: IPv6 address, and an optional port. Anything else falls back to
#: ``--advertise-host``, else the bind.
_HOST_HEADER = re.compile(r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9.-]+)(?::[0-9]{1,5})?")


def servable_path(path: str) -> bool:
    """Whether a request path names a file this server may hand out.

    An allowlist, not a denylist: the viewer bundle and the scene's data
    directory, and nothing else -- not the control token, the printable QR and
    sheet, the browser profile (``.chrome-display``), serve.py or anything a
    later version adds beside them. Percent-escapes are decoded first, and a
    ``.`` or ``..`` segment is refused anywhere. Dotfiles INSIDE the data stay
    servable: a zarr format 2 store's metadata is ``.zattrs`` / ``.zgroup`` /
    ``.zarray`` / ``.zmetadata``.
    """
    segments = [part for part in unquote(path).replace("\\", "/").split("/") if part]
    if not segments or any(part in (".", "..") for part in segments):
        return False
    return segments[0] in ("viewer", DATA_DIR_NAME)


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


def local_address(bind_host: str) -> str | None:
    """An address on THIS machine that reaches a server bound to ``bind_host``.

    Loopback for a wildcard or loopback bind; ``None`` for a bind to one
    specific interface, which loopback does not reach.
    """
    if bind_host in _ALL_INTERFACES:
        return "::1" if bind_host == "::" else "127.0.0.1"
    return bind_host if is_loopback_host(bind_host) else None


def _advertise_host(value: str) -> str:
    """Parse ``--advertise-host``: a hostname or an IP literal, nothing else.

    It is written into every printed URL and the QR, so a scheme, a port, a
    path or whitespace riding along would make all of them wrong. An IPv6
    literal may come bracketed or bare; it is stored bare and bracketed where a
    URL is built (``_url_host``).
    """
    text = value.strip()
    bare = text[1:-1] if text.startswith("[") and text.endswith("]") else text
    try:
        address = ipaddress.ip_address(bare)
    except ValueError:
        address = None
    if address is not None:
        # 0.0.0.0 / :: name no machine, a multicast address no server, and a
        # zoned IPv6 literal (fe80::1%en0) is not valid inside a URL at all.
        if address.is_unspecified or address.is_multicast or "%" in bare:
            raise argparse.ArgumentTypeError(
                f"{value!r} is not an address another device can open"
            )
        return str(address)
    name = text.rstrip(".")
    labels = name.split(".")
    if (
        name
        and len(name) <= 253
        and all(_HOST_LABEL.fullmatch(label) for label in labels)
        and not labels[-1].isdigit()  # 1.2.3 or 999.1.1.1: a mistyped address
    ):
        return name
    raise argparse.ArgumentTypeError(
        f"not a hostname or IP address: {value!r} (no scheme, port or path)"
    )


def unresolved_name_warning(name: str) -> str | None:
    """A warning when ``name`` does not resolve on this machine, else ``None``.

    Best effort and never fatal: the tablets resolve the name on THEIR network,
    which may know it when this machine does not (or the reverse), but a name
    nothing here can find is usually a typo worth flagging before it is printed.
    """
    try:
        ipaddress.ip_address(name)
        return None
    except ValueError:
        pass
    found: list[bool] = []

    def look_up() -> None:
        try:
            socket.getaddrinfo(name, None)
            found.append(True)
        except OSError:
            found.append(False)

    # In a thread with a time limit: a failing mDNS (`.local`) or DNS lookup
    # can take seconds, and this must not hold up the banner. Undecided in
    # time counts as resolving -- the warning is for a name nothing can find.
    lookup = threading.Thread(target=look_up, daemon=True)
    lookup.start()
    lookup.join(_NAME_LOOKUP_SECONDS)
    if found == [False]:
        return (
            f"  Note: {name} does not resolve on this machine. The QR names it, so "
            "make sure\n  it resolves to this machine on the tablets' network."
        )
    return None


#: How long the startup name check waits before giving up on it.
_NAME_LOOKUP_SECONDS = 1.5


def someone_listening(bind_host: str, port: int, also: tuple[str, ...] = ()) -> bool:
    """Whether another server already accepts connections on ``port``.

    A connect, not a bind: a bind probe without SO_REUSEADDR also fails on a
    port whose old connections are still draining after a restart (TIME_WAIT),
    which nothing is listening on and the real server binds fine. And on macOS
    SO_REUSEADDR lets a wildcard bind succeed beside another server's bind to
    one address, so the real bind alone would not notice it. For a wildcard
    bind both loopbacks are tried (``::`` may or may not take IPv4 too) plus
    ``also`` -- the addresses the printed URL will name, where a server bound
    to that one address would answer the tablets instead of this one.
    """
    if bind_host in _ALL_INTERFACES:
        targets = ["127.0.0.1", "::1", *also]
    else:
        targets = [bind_host]
    for target in dict.fromkeys(targets):  # de-duplicated, in order
        try:
            with socket.create_connection((target, port), timeout=0.5):
                return True
        except OSError:
            continue
    return False


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
    parser.add_argument(
        "--port",
        type=_port,
        default=8000,
        help=(
            "Port (default: 8000; 0 = any free port). With the control relay on, "
            "exactly this port: the panel URL and its QR name it, so a busy port "
            "is an error. Without it, the next free port is used"
        ),
    )
    # Binding every interface is the POINT of a panel scene: the tablet is a
    # different device, so loopback cannot serve it. A scene without a panel
    # still defaults to loopback, and `--host 127.0.0.1` overrides either way.
    # Off loopback the relay requires a token by default (`resolve_control_token`).
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
        "--chrome",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Open the display in Google Chrome, full screen, with the flags WebGPU "
            "needs, and print a check of what it renders with (WebGPU, the GPU, "
            "full screen). PATH picks a Chrome; quit it with Cmd+Q / Alt+F4"
        ),
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
        "--advertise-host",
        type=_advertise_host,
        default=None,
        metavar="NAME",
        help=(
            "Hostname or IP address to put in the printed URLs and the QR "
            "(default: this machine's network address). Use a name or a reserved "
            "address that stays the same, so a printed QR survives a restart"
        ),
    )
    tokens = parser.add_mutually_exclusive_group()
    tokens.add_argument(
        "--control-token",
        type=_control_token,
        default=None,
        help=(
            "Secret every control socket must present, for this run only. "
            f"Default off loopback: the token saved in {CONTROL_TOKEN_FILE}"
        ),
    )
    tokens.add_argument(
        "--no-control-token",
        action="store_true",
        help="Run the relay without a token: anyone who can reach it can drive the display",
    )
    parser.add_argument(
        "--auto-advance",
        type=_auto_advance,
        default=AUTO_ADVANCE,
        metavar="SECONDS",
        help=(
            "Seconds the display rests at each stop before its tour moves on by "
            "itself, overriding the scene's own setting (0 = no auto-advance). "
            + (
                f"Default: {AUTO_ADVANCE:g} s, set when this folder was exported"
                if AUTO_ADVANCE is not None
                else "Default: the scene's own setting"
            )
        ),
    )
    parser.add_argument(
        "--text-scale",
        type=_text_scale,
        default=None,
        metavar="X",
        help=(
            "Scale the on-screen text by X (e.g. 0.8 for a screen read up close); "
            "positions and widths stay put. Default: the scene's own setting"
        ),
    )
    return parser.parse_args(argv)


def _port(value: str) -> int:
    """Parse ``--port``: a TCP port number."""
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {value!r}") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"must be between 0 and 65535, got {port}")
    return port


def _control_token(value: str) -> str:
    """Parse ``--control-token``: a non-empty secret with no surrounding space.

    An empty one is refused rather than read as "no token" (that is
    ``--no-control-token``), and padding is refused because the viewer trims
    the URL's token, so a padded secret would never match.
    """
    if not value or value != value.strip():
        raise argparse.ArgumentTypeError(
            "must be non-empty, without leading or trailing spaces "
            "(use --no-control-token for none)"
        )
    return value


def _auto_advance(value: str) -> float:
    """Parse ``--auto-advance``: seconds per step, or 0 for no auto-advance.

    Only the sign is checked here; the viewer clamps the period to the range
    it plays (it is the party that knows it), so this script need not mirror it.
    """
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"not a number of seconds: {value!r}"
        ) from None
    if not (seconds >= 0 and seconds != float("inf")):  # NaN fails too
        raise argparse.ArgumentTypeError(
            f"must be 0 (off) or a period in seconds, got {value!r}"
        )
    return seconds


def _text_scale(value: str) -> float:
    """Parse ``--text-scale``: a number inside the range the viewer accepts."""
    try:
        scale = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    low, high = TEXT_SCALE_RANGE
    if not low <= scale <= high:
        raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
    return scale


def viewer_target(
    data_url: str,
    text_scale: float | None = None,
    *,
    playback_period: float | None = None,
) -> str:
    """The viewer's path and query for the scene served at ``data_url``."""
    target = f"/viewer/?src={data_url}{TITLE_QUERY}{LAUNCH_QUERY}"
    if text_scale is not None:
        target += f"&textScale={text_scale:g}"
    if playback_period is not None:
        target += (
            f"&playbackPeriod={quote(str(playback_period).removesuffix('.0'), safe='')}"
        )
    return target


def urls(
    host: str,
    port: int,
    control: bool,
    token: str | None,
    text_scale: float | None = None,
    *,
    advertise: str | None = None,
    playback_period: float | None = None,
) -> list[str]:
    """The display URL, then (with the relay) the panel URL.

    ``advertise`` replaces the address derived from the bind everywhere --
    the page, its ``?src=`` data URL and the panel URL -- so the page and its
    data stay same-origin (this server sends no cross-origin headers).
    ``playback_period`` (``--auto-advance``) retimes the scene's authored
    playback on the display; the panel URL does not carry it.
    """
    reachable = _url_host(advertise or advertised_host(host))
    data_url = f"http://{reachable}:{port}/{DATA_DIR_NAME}"
    viewer_url = f"http://{reachable}:{port}" + viewer_target(
        data_url, text_scale, playback_period=playback_period
    )
    if not control:
        return [viewer_url]
    suffix = f"&controlToken={quote(token, safe='')}" if token else ""
    return [
        viewer_url + "&control" + suffix,
        f"http://{reachable}:{port}/viewer/control.html?control{suffix}",
    ]


def panel_sheet(matrix: Any, url: str, *, has_token: bool) -> bytes:
    """The A4 sheet for the panel URL: the QR half the page wide, the URL, what
    it is and how to use it. Worded here, laid out by ``luxar_qr``."""
    title = unquote(TITLE_QUERY.partition("title=")[2]) or "this Luxar scene"
    host = urlsplit(url).netloc
    notes = ["Everyone who scans the code can drive the display; the last tap wins."]
    if has_token:
        notes.append(
            "The code carries an access key. Keep this sheet in the room, and do "
            "not post a photo of it online: anyone with the code can drive the display."
        )
    notes.append(
        "If the code stops working, the display's address, port or key has "
        f"changed: print a new sheet ({SHEET_FILE}, written beside serve.py "
        "each time it starts)."
    )
    sheet: bytes = qr_pdf_bytes(
        matrix,
        url=url,
        heading="Drive the display from your phone",
        subheading=title,
        intro=(
            "This code opens a touch panel for the big screen in this room. Pick "
            "what it shows from your phone or tablet: no app to install."
        ),
        steps=(
            "Join the same Wi-Fi network as the display.",
            "Point your phone's camera at the code and open the link it offers "
            "(or type the address under the code).",
            "Tap a tile on the panel: the big screen goes there.",
        ),
        notes=tuple(notes),
        footer=f"Luxar  |  {host}",
    )
    return sheet


def print_control_qr(
    url: str, *, printable: bool = True, has_token: bool = False
) -> None:
    """Print a scannable QR for ``url`` and leave a PNG copy beside the script.

    The point of kiosk mode is a tablet, and the URL carries a host, a port, a
    path and a token -- too much to read off a screen and type in correctly at
    an exhibit. Half-block glyphs are used so the symbol comes out roughly
    square in a terminal; one module per character is twice as tall as wide
    and many phone cameras will not lock onto it.

    ``control-qr.png`` and ``control-panel.pdf`` (an A4 sheet, see
    :func:`panel_sheet`) are the copies meant for PRINTING, so they are only
    written when ``url`` will still be valid on the next start (``printable``):
    a one-off ``--control-token``, or a token that could not be saved, gets the
    terminal QR only, never a file someone might print.

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
    if not printable:
        print(
            f"  (not written to {QR_FILE} / {SHEET_FILE}: this QR is for this run only)"
        )
        return
    renders = (
        (QR_FILE, lambda: qr_png_bytes(matrix, scale=8, border=4)),
        (SHEET_FILE, lambda: panel_sheet(matrix, url, has_token=has_token)),
    )
    written = []
    for name, render in renders:
        try:
            data = render()
        except Exception as error:  # noqa: BLE001 - a printable copy is never the gate
            print(f"  (could not draw {name}: {_printable(error)})")
            continue
        if write_private(SCRIPT_DIR / name, data):
            written.append(name)
    if written:
        print(f"  (also written to {' and '.join(written)}, for printing)")


class StartupError(Exception):
    """A refusal to start, printed as ``Error: <message>`` with exit status 1."""


class ControlToken:
    """The relay's token for one run: its value, a note, whether to print it."""

    def __init__(
        self, value: str | None, note: str | None = None, printable: bool = True
    ) -> None:
        self.value = value
        #: Printed under the QR: where the token came from, how to rotate it.
        self.note = note
        #: Whether a QR of this run's URL is still valid on the next start.
        self.printable = printable


def write_private(path: Path, data: bytes) -> bool:
    """Replace ``path`` with ``data`` atomically, created readable by its owner.

    Both files written this way carry the control token -- the token itself,
    and the QR encoding it -- so neither is left readable by every account on
    the machine, not even for the moment between creating and chmod-ing it.
    False when the folder is not writable.
    """
    staging = path.with_name(f".{path.name}.tmp")
    try:
        # A fresh file every time, never one left (or planted) at that name:
        # an old one would keep its mode, and a symlink would be followed.
        staging.unlink(missing_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(staging, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        os.replace(staging, path)
    except OSError:
        try:
            staging.unlink()
        except OSError:
            pass
        return False
    return True


def create_token_file(path: Path, token: str) -> str | None:
    """Create ``path`` holding ``token``; return the token it ends up holding.

    Exclusive create, owner-readable from the start: when two first starts race,
    exactly one creates the file and the other reads what it wrote, so both
    serve the token every later start will read. ``None`` when the folder is
    not writable.
    """
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_saved_token(path)
    except OSError:
        return None
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(token + "\n")
    return token


def read_saved_token(path: Path) -> str | None:
    """The token saved at ``path``, or ``None`` when there is no file yet.

    A file that is there but unusable is an error rather than something to
    overwrite: a printed QR may carry the token it was meant to hold, and
    deleting the file is how a new token is asked for. Read as ``utf-8-sig`` so
    a file re-saved by an editor that adds a byte-order mark still reads.
    """
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        raise StartupError(f"cannot read {path.name}: {error}") from None
    if not _TOKEN_PATTERN.fullmatch(text):
        raise StartupError(
            f"{path.name} does not hold a usable token (8-128 of A-Z a-z 0-9 _ -). "
            "Fix it, or delete it to issue a new token (retiring printed QRs)."
        )
    return text


def resolve_control_token(args: argparse.Namespace, folder: Path) -> ControlToken:
    """The relay's token for this run.

    In order: ``--control-token`` (this run only, the saved one untouched);
    ``--no-control-token`` (none); a loopback bind (none: nothing off this
    machine can connect); else the token saved in :data:`CONTROL_TOKEN_FILE`,
    generated and saved on the first start, so the panel URL and its QR are the
    same every start.
    """
    if not args.control:
        return ControlToken(None)
    if args.control_token is not None:
        return ControlToken(
            args.control_token,
            "  Control token from --control-token: for this run only.",
            printable=False,
        )
    if args.no_control_token or is_loopback_host(args.host):
        return ControlToken(None)
    path = folder / CONTROL_TOKEN_FILE
    saved = read_saved_token(path)
    if saved is not None:
        return ControlToken(
            saved,
            f"  Control token from {path.name}: the same every start, so a printed\n"
            "  QR keeps working. Delete the file to issue a new one (retires printed QRs).",
        )
    token = create_token_file(path, secrets.token_urlsafe(_TOKEN_BYTES))
    if token is not None:
        return ControlToken(
            token,
            f"  New control token saved in {path.name}; every start reuses it, so a\n"
            "  printed QR keeps working. Delete the file to issue a new one.",
        )
    return ControlToken(
        secrets.token_urlsafe(_TOKEN_BYTES),
        f"  Note: could not save the control token in {path.name} (read-only folder?).\n"
        "  It is new on every start, so do not print this QR.",
        printable=False,
    )


#: Bind errors meaning "another socket holds this port". WSAEACCES is Windows'
#: answer when the holder bound exclusively or as another user account.
_ADDRESS_IN_USE = frozenset(
    code
    for code in (
        errno.EADDRINUSE,
        getattr(errno, "WSAEADDRINUSE", None),
        getattr(errno, "WSAEACCES", None),
    )
    if code is not None
)


def _port_in_use(port: int) -> str:
    return (
        f"port {port} is already in use on this machine (another serve.py?).\n"
        "The control panel's URL and its printed QR name this port, so serve.py will\n"
        "not move to another one. Stop the other server, or pass --port N and\n"
        "reprint control-qr.png."
    )


def _probe_targets(args: argparse.Namespace) -> tuple[str, ...]:
    """Addresses the printed URL names, for the busy-port probe of a wildcard bind."""
    named = args.advertise_host or advertised_host(args.host)
    try:
        ipaddress.ip_address(named)
    except ValueError:
        return ()  # a name: resolving it is the tablets' business, not a probe's
    return (named,)


def open_server(args: argparse.Namespace, relay: Relay | None) -> ControlServer:
    """Bind the server, on exactly ``--port`` when the relay is on.

    With the relay the panel URL and its printed QR name the port, so another
    server holding it is an error rather than a reason to move. Without it
    (nothing printed to keep valid) the next free port is used, as a folder
    double-clicked twice expects.
    """
    handler = partial(LuxarHandler, directory=str(SCRIPT_DIR))
    port = int(args.port)
    if args.control and port == 0:
        raise StartupError(
            "--port 0 picks a random port, which the panel URL and its printed QR\n"
            "cannot name; with the control relay, give a fixed --port."
        )
    if args.control:
        if someone_listening(args.host, port, _probe_targets(args)):
            raise StartupError(_port_in_use(port))
    else:
        free = find_port(args.host, port)
        if free is None:
            raise StartupError(f"No available port found near {port}")
        port = free
    try:
        return ControlServer((args.host, port), handler, relay)
    except OSError as error:
        if error.errno not in _ADDRESS_IN_USE:
            raise StartupError(
                f"cannot listen on {args.host}:{port}: {error}"
            ) from None
        if args.control:
            raise StartupError(_port_in_use(port)) from None
        raise StartupError(f"port {port} was taken while starting; try again") from None


def _local_display_url(
    args: argparse.Namespace, port: int, token: str | None
) -> str | None:
    """The display URL on loopback, for the browser this script opens itself.

    The printed address is for OTHER devices: a name or LAN address chosen for
    the tablets need not resolve on this machine. Same token, same query. (Its
    origin differs from the printed one, so this browser keeps its own cache
    and saved settings.)
    """
    local = local_address(args.host)
    if local is None:
        return None
    return urls(
        local,
        port,
        args.control,
        token,
        args.text_scale,
        playback_period=args.auto_advance,
    )[0]


def _control_notes(args: argparse.Namespace, token: ControlToken) -> list[str]:
    """What to say under the QR, in order."""
    notes = [token.note] if token.note else []
    if args.advertise_host:
        if is_loopback_host(args.host):
            notes.append(
                f"  Note: --host {args.host} listens on this machine only, so no other\n"
                f"  device can open the {args.advertise_host} URLs above."
            )
        else:
            unresolved = unresolved_name_warning(args.advertise_host)
            if unresolved:
                notes.append(unresolved)
    notes.append(
        "  Note: the bare address (http://host:port/) opens a VIEW-ONLY viewer\n"
        "  that the panel cannot drive. Open the Display URL above instead."
    )
    if token.value is None and not is_loopback_host(args.host):
        notes.append(
            "  Note: no control token (--no-control-token), so anyone who can reach\n"
            "  this machine on the network can drive the display. Foreign web pages\n"
            "  cannot (the relay checks the request origin), but a person on this\n"
            "  network who opens the panel URL can."
        )
    return notes


# ── --chrome: the display, opened in Chrome with the right flags ────────────

#: Every platform: full screen, no first-run or crash bubbles, sound without a
#: click (narration), and no throttling of a window nobody is looking at.
_CHROME_FLAGS = (
    "--kiosk",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-session-crashed-bubble",
    "--noerrdialogs",
    "--disable-infobars",
    "--autoplay-policy=no-user-gesture-required",
    "--disable-background-timer-throttling",
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--ignore-gpu-blocklist",
)
#: Linux needs WebGPU switched on, through Vulkan; macOS and Windows have it.
_CHROME_LINUX_FLAGS = ("--enable-unsafe-webgpu", "--enable-features=Vulkan")


def find_chrome(platform: str = sys.platform) -> str | None:
    """Google Chrome (or Chromium) on this machine, or None."""
    if platform == "darwin":
        apps = [Path("/Applications"), Path.home() / "Applications"]
        names = ("Google Chrome.app/Contents/MacOS/Google Chrome",
                 "Chromium.app/Contents/MacOS/Chromium")  # fmt: skip
        found = [app / name for app in apps for name in names]
        return next((str(path) for path in found if path.exists()), None)
    if platform.startswith("win"):
        roots = [
            os.environ.get(k, "")
            for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
        ]
        found = [Path(r) / "Google/Chrome/Application/chrome.exe" for r in roots if r]
        return next((str(path) for path in found if path.exists()), None)
    for name in (
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
    ):
        path = shutil.which(name)
        if path:
            return path
    return None


def chrome_command(
    chrome: str, url: str, profile: Path, platform: str = sys.platform
) -> list[str]:
    """The command line that opens ``url`` full screen, WebGPU on."""
    flags = [*_CHROME_FLAGS, f"--user-data-dir={profile}"]
    if platform.startswith("linux"):
        flags += _CHROME_LINUX_FLAGS
    return [chrome, *flags, url]


def _printable(value: object, limit: int = 200) -> str:
    """``value`` as one line for the terminal: no control characters (an escape
    sequence in a report must not reach the operator's terminal), capped."""
    text = "".join(c if c.isprintable() else "?" for c in str(value))
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _number(value: object) -> float:
    """A JSON number, or 0 for anything else (a report is untrusted input)."""
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def assess_display(report: dict[str, Any]) -> list[tuple[str, str]]:
    """``(level, message)`` lines about a display report: OK, WARNING, NOTE or ERROR."""
    if report.get("error"):
        nogpu = "" if report.get("webgpuAvailable") else " (this browser has no WebGPU)"
        return [("ERROR", f"the display failed to start{nogpu}: {_printable(report['error'])}. "
                 "Look at the screen, its browser console and chrome://gpu")]  # fmt: skip
    lines: list[tuple[str, str]] = []
    if report.get("datasetError"):
        lines.append(("ERROR", f"the scene failed to load: {_printable(report['datasetError'])}. "
                      f"Check the {DATA_DIR_NAME}/ folder beside serve.py"))  # fmt: skip
    backend = report.get("backend")
    adapter = report.get("adapter") if isinstance(report.get("adapter"), dict) else None
    gpu = _printable(" ".join(
        str(adapter.get(k, "")) for k in ("vendor", "architecture", "description")
    ).strip()) if adapter else ""  # fmt: skip
    if backend == "webgpu-native":
        lines.append(("OK", f"WebGPU on the GPU{f' ({gpu})' if gpu else ''}"))
    elif backend == "webgpu-gl":
        lines.append(("WARNING", "the WebGPU renderer fell back to its WebGL2 backend: "
                      "no WebGPU adapter. Check chrome://gpu (WebGPU: Hardware accelerated)"))  # fmt: skip
    else:
        why = ("this browser has no WebGPU" if not report.get("webgpuAvailable")
               else "WebGPU did not come up")  # fmt: skip
        lines.append(("WARNING", f"rendering with WebGL, not WebGPU: {why}. "
                      "Check chrome://gpu, or the browser's version"))  # fmt: skip
    software = re.compile(r"swiftshader|llvmpipe|software|basic render", re.IGNORECASE)
    if adapter and (adapter.get("isFallbackAdapter") or software.search(gpu)):
        lines.append(("WARNING", f"a SOFTWARE GPU adapter ({gpu or 'fallback'}): "
                      "slow; check the GPU driver and chrome://gpu"))  # fmt: skip
    screen, window = report.get("screen") or {}, report.get("window") or {}
    if isinstance(screen, dict) and isinstance(window, dict):
        sw, sh = _number(screen.get("width")), _number(screen.get("height"))
        ww, wh = _number(window.get("width")), _number(window.get("height"))
        if sw and sh and (ww < 0.98 * sw or wh < 0.98 * sh):
            lines.append(("NOTE", f"the window is {ww:.0f}x{wh:.0f} on a {sw:.0f}x{sh:.0f} screen: "
                          "not full screen"))  # fmt: skip
    if "Chrome/" not in str(report.get("userAgent", "")):
        lines.append(("NOTE", "the display is not Chrome; WebGPU is tested there"))
    return lines


def print_display_check(report: dict[str, Any]) -> None:
    """Print :func:`assess_display` under a heading the operator can find."""
    try:
        lines = assess_display(report)
    except Exception as error:  # noqa: BLE001 - a check must never take the server down
        lines = [
            ("WARNING", f"could not read the display's report ({_printable(error)})")
        ]
    print()
    print("  Display check (the browser serve.py --chrome opened):")
    for level, message in lines:
        print(f"    {level:<8} {message}")
    sys.stdout.flush()


def resolve_chrome(choice: str) -> str | None:
    """The Chrome executable for ``--chrome [PATH]``, or None.

    A macOS ``.app`` bundle is accepted for the executable inside it; any other
    path must be an executable file, so a directory or a typo is caught here
    rather than as a crash of the launch.
    """
    path = find_chrome() if choice == "auto" else choice
    if path and path.endswith(".app") and Path(path).is_dir():
        inner = Path(path) / "Contents" / "MacOS"
        found = sorted(inner.iterdir()) if inner.is_dir() else []
        path = str(found[0]) if found else None
    if path and os.path.isfile(path) and os.access(path, os.X_OK):
        return path
    return None


def _no_chrome(choice: str, fallback: bool) -> None:
    where = "" if choice == "auto" else f" at {_printable(choice)}"
    then = (
        "opening the default browser instead" if fallback else "not opening a browser"
    )
    print(f"\n  WARNING  Chrome not found{where}: {then}.")
    print("           Install Google Chrome, or pass --chrome /path/to/chrome.")


def open_display_in_chrome(
    args: argparse.Namespace, server: ControlServer, url: str
) -> bool:
    """Launch Chrome on ``url`` and await the page's report; False if it did not launch."""
    chrome = resolve_chrome(args.chrome)
    if chrome is None:
        _no_chrome(args.chrome, fallback=not args.no_open)
        return False
    nonce = server.display_report.expect()
    target = url + ("&" if "?" in url else "?") + f"displayReport={nonce}"
    try:
        process = subprocess.Popen(  # noqa: S603 # nosec B603 - our own flags and URL
            chrome_command(chrome, target, SCRIPT_DIR / CHROME_PROFILE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as error:
        server.display_report.claim(nonce)  # nothing will report
        print(f"\n  WARNING  could not start Chrome ({_printable(error)}).")
        return False
    print(f"\n  Opening the display in Chrome ({chrome}), full screen; quit it with")
    print("  Cmd+Q / Alt+F4. A display check follows once it has loaded.")
    _watch_display(server, process)
    return True


def _watch_display(server: ControlServer, process: Any) -> None:
    """Warn when Chrome quits at once, or when no report comes in time."""

    def early_exit() -> None:
        code = process.poll()
        if code is not None and server.display_report.pending():
            print(
                f"\n  WARNING  Chrome exited at once (status {code}): no display. Is there"
            )
            print(
                "           a screen session? Run it with --chrome from that session."
            )
            sys.stdout.flush()

    def no_report() -> None:
        if server.display_report.pending() and process.poll() is None:
            print(
                f"\n  WARNING  the display sent no report within {_DISPLAY_REPORT_WAIT_S:.0f} s:"
            )
            print(
                "           it may not have loaded. Look at the screen and at chrome://gpu;"
            )
            print("           the check still prints if the report comes later.")
            sys.stdout.flush()

    for delay, check in ((3.0, early_exit), (_DISPLAY_REPORT_WAIT_S, no_report)):
        timer = threading.Timer(delay, check)
        timer.daemon = True
        timer.start()


def main(argv: list[str] | None = None) -> int:
    """Serve this folder until interrupted."""
    args = _parse_args(argv)
    if (args.control_token is not None or args.no_control_token) and not args.control:
        print("Warning: --control-token / --no-control-token need --control. Ignoring.")
    try:
        token = resolve_control_token(args, SCRIPT_DIR)
        relay = Relay(token.value) if args.control else None
        server = open_server(args, relay)
    except StartupError as error:
        print(f"Error: {error}")
        return 1
    server.text_scale = args.text_scale
    server.playback_period = args.auto_advance
    server.advertise = args.advertise_host
    port = int(server.server_address[1])

    addresses = urls(
        args.host,
        port,
        args.control,
        token.value,
        args.text_scale,
        advertise=args.advertise_host,
        playback_period=args.auto_advance,
    )
    print()
    print("  Display (open this on the big screen)")
    print(f"    {addresses[0]}")
    if args.control:
        print()
        print("  Control panel (open this on the tablet)")
        print(f"    {addresses[1]}")
        print_control_qr(
            addresses[1], printable=token.printable, has_token=token.value is not None
        )
        for note in _control_notes(args, token):
            print()
            print(note)
    print()
    print("Press Ctrl+C to stop")
    # Flushed explicitly: stdout is block-buffered when it is a pipe rather
    # than a terminal, so `python serve.py | tee log` would hold the one thing
    # the operator has to read -- the URLs -- until the buffer filled or the
    # process exited.
    sys.stdout.flush()

    local = _local_display_url(args, port, token.value) or addresses[0]
    if args.chrome and open_display_in_chrome(args, server, local):
        pass
    elif not args.no_open:
        threading.Timer(0.5, webbrowser.open, args=[local]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
