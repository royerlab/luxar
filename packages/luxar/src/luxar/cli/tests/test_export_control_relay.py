"""Tests for the exported folder's control relay (`_export_serve_template`).

The template is the FOURTH implementation of one relay — after the Python hub,
the viewer's client and the Go launcher's — and the only one that ships as a
copied file into folders nobody rebuilds. So two things are tested here that a
normal module would not need: that its wire constants still agree with the
generated contract, and that the *generated* script (not just the template)
really speaks RFC 6455 to an independent client.

The client is `websockets`, a declared dependency, and deliberately NOT this
module's own codec: a framing bug that existed on both sides of a test would
cancel itself out and the test would pass on a relay no browser could talk to.
"""

from __future__ import annotations

import io
import json
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Optional

import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import ClientConnection, connect

from luxar.cli import _control_contract as contract
from luxar.cli import _export_serve_template as template
from luxar.cli.export import _get_serve_script_content

# ── the drift gate ──────────────────────────────────────────────────────────


def test_contract_values_match_the_contract() -> None:
    """The template's copy of the wire vocabulary must match the projection.

    The exported script cannot import `_control_contract` (stdlib only, in a
    folder that may have been emailed to someone who has never installed
    Luxar), so it carries its own copy. This is that copy's drift gate: without
    it, a contract change would leave exported kiosks quietly speaking last
    month's protocol.
    """
    shared = {
        "ROLE_VIEWER",
        "ROLE_CONTROLLER",
        "PARAM_ROLE",
        "PARAM_TOKEN",
        "CLOSE_POLICY_VIOLATION",
        "JSONRPC_VERSION",
        "EVENT_METHOD",
        "PARSE_ERROR",
        "INVALID_REQUEST",
        "NO_VIEWER",
        "MAX_PENDING_PER_CONTROLLER",
        "MAX_FRAME_BYTES",
    }
    assert {name: getattr(template, name) for name in shared} == {
        name: getattr(contract, name) for name in shared
    }


def test_contract_covers_every_shared_name() -> None:
    """Every name the contract projects is either mirrored or knowingly not.

    Guards the gate above against being narrowed by omission: adding a value to
    the contract and forgetting to mirror it would otherwise pass, because the
    set of names to compare is written by hand.
    """
    projected = {
        name for name in vars(contract) if name.isupper() and not name.startswith("_")
    }
    # Four values a dumb relay has no use for, each for its own reason:
    # `ROLES` is a frozenset of the two roles it already checks directly;
    # `MAX_BUFFERED_EVENTS` bounds a CLIENT's event buffer, and this relay
    # buffers nothing; `METHOD_NOT_FOUND` and `INTERNAL_ERROR` are the VIEWER's
    # answers to a controller (the method policy lives there), and a relay that
    # emitted either would be claiming to know the method vocabulary.
    unmirrored = {"ROLES", "MAX_BUFFERED_EVENTS", "METHOD_NOT_FOUND", "INTERNAL_ERROR"}
    assert projected - unmirrored <= set(vars(template))


# ── framing ─────────────────────────────────────────────────────────────────


def test_unmask_matches_a_handwritten_xor() -> None:
    """`_unmask` is checked against the obvious loop, once.

    The rest of these tests MASK with `_unmask` (XOR is its own inverse), so
    this is what stops a bug in it from hiding behind its own reuse.
    """
    payload = bytes(range(37))
    key = b"\x11\x22\x33\x44"
    expected = bytes(byte ^ key[index % 4] for index, byte in enumerate(payload))
    assert template._unmask(payload, key) == expected
    assert template._unmask(b"", key) == b""


def client_frame(
    opcode: int, payload: bytes, *, fin: bool = True, mask: bytes = b"\x01\x02\x03\x04"
) -> bytes:
    """Encode one MASKED frame, the way a browser sends them."""
    header = bytearray([(0x80 if fin else 0x00) | opcode])
    length = len(payload)
    if length < 126:
        header.append(0x80 | length)
    elif length < 65536:
        header.append(0x80 | 126)
        header += struct.pack("!H", length)
    else:
        header.append(0x80 | 127)
        header += struct.pack("!Q", length)
    header += mask
    return bytes(header) + template._unmask(payload, mask)


def server_frames(data: bytes) -> list[tuple[int, bytes]]:
    """Decode the unmasked frames a server wrote, as ``(opcode, payload)``."""
    frames: list[tuple[int, bytes]] = []
    offset = 0
    while offset < len(data):
        opcode = data[offset] & 0x0F
        length = data[offset + 1] & 0x7F
        offset += 2
        if length == 126:
            length = int(struct.unpack("!H", data[offset : offset + 2])[0])
            offset += 2
        elif length == 127:
            length = int(struct.unpack("!Q", data[offset : offset + 8])[0])
            offset += 8
        frames.append((opcode, data[offset : offset + length]))
        offset += length
    return frames


def peer_reading(*frames: bytes) -> tuple[template._Peer, io.BytesIO]:
    """A peer whose inbound frames are preloaded and whose writes are captured."""
    written = io.BytesIO()
    return template._Peer(io.BytesIO(b"".join(frames)), written), written


@pytest.mark.parametrize("size", [0, 5, 125, 126, 200, 65535, 65536, 70000])
def test_frame_lengths_round_trip(size: int) -> None:
    """Every length encoding (7-bit, 16-bit, 64-bit) survives a round trip."""
    payload = b"x" * size
    peer, _ = peer_reading(client_frame(template._OP_TEXT, payload))
    assert list(peer.messages()) == [payload.decode()]
    assert server_frames(template._frame_bytes(template._OP_TEXT, payload)) == [
        (template._OP_TEXT, payload)
    ]


def test_fragments_are_joined() -> None:
    """A message split across frames arrives whole."""
    peer, _ = peer_reading(
        client_frame(template._OP_TEXT, b'{"a":', fin=False),
        client_frame(template._OP_CONTINUATION, b"1}"),
    )
    assert list(peer.messages()) == ['{"a":1}']


def test_ping_is_answered_mid_stream() -> None:
    """A ping gets a pong carrying the same payload, and does not end the read."""
    peer, written = peer_reading(
        client_frame(template._OP_PING, b"beat"),
        client_frame(template._OP_TEXT, b"hello"),
    )
    assert list(peer.messages()) == ["hello"]
    assert server_frames(written.getvalue()) == [(template._OP_PONG, b"beat")]


def test_close_frame_ends_the_stream() -> None:
    """A close frame stops iteration without an error."""
    peer, _ = peer_reading(
        client_frame(template._OP_TEXT, b"one"),
        client_frame(template._OP_CLOSE, struct.pack("!H", 1000)),
        client_frame(template._OP_TEXT, b"never read"),
    )
    assert list(peer.messages()) == ["one"]


def test_unmasked_client_frame_is_refused() -> None:
    """RFC 6455 requires client frames to be masked; an unmasked one is refused.

    Not pedantry: an unmasked frame is how a crafted payload gets a proxy in the
    middle to read part of the stream as a fresh HTTP request.
    """
    unmasked = bytes([0x80 | template._OP_TEXT, 3]) + b"abc"
    peer, written = peer_reading(unmasked)
    assert list(peer.messages()) == []
    (opcode, payload), *_ = server_frames(written.getvalue())
    assert opcode == template._OP_CLOSE
    assert struct.unpack("!H", payload[:2])[0] == template._CLOSE_PROTOCOL_ERROR


def test_binary_frames_are_refused() -> None:
    """This relay carries JSON text; a binary frame is closed, not guessed at."""
    peer, written = peer_reading(client_frame(0x2, b"\x00\x01"))
    assert list(peer.messages()) == []
    (_, payload), *_ = server_frames(written.getvalue())
    assert struct.unpack("!H", payload[:2])[0] == template._CLOSE_UNSUPPORTED_DATA


def test_oversized_frame_is_refused_before_reading_it() -> None:
    """A declared length past the cap is refused without allocating it."""
    header = bytes([0x80 | template._OP_TEXT, 0x80 | 127]) + struct.pack(
        "!Q", template.MAX_FRAME_BYTES + 1
    )
    peer, written = peer_reading(header)
    assert list(peer.messages()) == []
    (_, payload), *_ = server_frames(written.getvalue())
    assert struct.unpack("!H", payload[:2])[0] == template._CLOSE_TOO_BIG


def test_fragmented_message_cannot_exceed_the_cap_in_pieces() -> None:
    """The cap is per MESSAGE too, or fragments would walk around it."""
    chunk = b"x" * 1024
    frames = [
        client_frame(template._OP_TEXT, chunk, fin=False),
        *[
            client_frame(template._OP_CONTINUATION, chunk, fin=False)
            for _ in range(template.MAX_FRAME_BYTES // 1024)
        ],
    ]
    peer, written = peer_reading(*frames)
    assert list(peer.messages()) == []
    (_, payload), *_ = server_frames(written.getvalue())
    assert struct.unpack("!H", payload[:2])[0] == template._CLOSE_TOO_BIG


def test_continuation_without_a_start_is_refused() -> None:
    """A continuation frame with nothing to continue is a protocol error."""
    peer, written = peer_reading(client_frame(template._OP_CONTINUATION, b"orphan"))
    assert list(peer.messages()) == []
    (_, payload), *_ = server_frames(written.getvalue())
    assert struct.unpack("!H", payload[:2])[0] == template._CLOSE_PROTOCOL_ERROR


def test_invalid_utf8_is_refused() -> None:
    """A text frame that is not UTF-8 is refused rather than lossily decoded."""
    peer, written = peer_reading(client_frame(template._OP_TEXT, b"\xff\xfe"))
    assert list(peer.messages()) == []
    (_, payload), *_ = server_frames(written.getvalue())
    assert struct.unpack("!H", payload[:2])[0] == template._CLOSE_INVALID_PAYLOAD


# ── policy ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("origin", "host", "allowed"),
    [
        (None, "kiosk.local:8000", True),  # every non-browser client
        ("http://kiosk.local:8000", "kiosk.local:8000", True),
        ("https://kiosk.local:8000", "kiosk.local:8000", True),  # scheme differs
        ("http://KIOSK.local:8000", "kiosk.local:8000", True),  # case
        ("http://kiosk.local:8000", "kiosk.local:9000", False),  # other port
        ("http://evil.example", "kiosk.local:8000", False),
        ("null", "kiosk.local:8000", False),  # sandboxed iframe
        ("", "kiosk.local:8000", False),
        ("http://[::1]:8000", "[::1]:8000", True),
        ("http://[::1", "kiosk.local:8000", False),  # would crash a naive parse
        ("file://", "kiosk.local:8000", False),
        ("http://kiosk.local:8000", None, False),
    ],
)
def test_origin_allowed(
    origin: Optional[str], host: Optional[str], allowed: bool
) -> None:
    """Cross-Site WebSocket Hijacking: the Origin must match the Host."""
    assert template.origin_allowed(origin, host) is allowed


@pytest.mark.parametrize(
    ("target", "path", "query"),
    [
        ("/control", "/control", ""),
        ("/control?role=viewer", "/control", "role=viewer"),
        ("/control#frag", "/control", ""),
        ("/control?token=a#frag", "/control", "token=a"),
        # `urlsplit` RAISES ValueError on this one, and a request line is
        # something any client picks. Splitting cannot raise.
        ("//[::1", "//[::1", ""),
        ("/data/zarr.json", "/data/zarr.json", ""),
    ],
)
def test_request_targets_are_split_not_parsed(
    target: str, path: str, query: str
) -> None:
    """A request target must never be able to raise on its way to a comparison."""
    assert template._request_path(target) == path
    assert template._request_query(target) == query


def test_unknown_role_is_refused_rather_than_defaulted() -> None:
    """`?role=viewr` must not quietly attach with a controller's authority."""
    relay = template.Relay()
    assert relay.refusal("viewr", None, None, "h:1") == "unknown role 'viewr'"
    assert relay.refusal("", None, None, "h:1") is not None


def test_token_is_required_when_configured() -> None:
    relay = template.Relay("s3cret")
    assert relay.refusal("controller", "s3cret", None, "h:1") is None
    assert relay.refusal("controller", "wrong", None, "h:1") == "bad token"
    assert relay.refusal("controller", None, None, "h:1") == "bad token"


def test_a_configured_token_supersedes_the_origin_check() -> None:
    """An authenticated split-origin exhibit is a real deployment, so it works."""
    assert (
        template.Relay().refusal("controller", None, "http://other", "h:1") is not None
    )
    assert (
        template.Relay("s3cret").refusal("controller", "s3cret", "http://other", "h:1")
        is None
    )


def test_non_ascii_token_is_compared_not_crashed_on() -> None:
    """`hmac.compare_digest` raises on a non-ASCII str; we compare bytes."""
    relay = template.Relay("clé-privée")
    assert relay.authorized("clé-privée") is True
    assert relay.authorized("clé-publique") is False


# ── relay, in process over BytesIO peers ────────────────────────────────────


def attached_viewer(relay: template.Relay) -> tuple[template._Peer, io.BytesIO]:
    """Attach a viewer that never speaks, so it stays attached for the test."""
    peer, written = peer_reading()
    relay._viewer = peer
    return peer, written


def test_a_controllers_request_reaches_the_viewer_with_a_hub_id() -> None:
    relay = template.Relay()
    _, viewer_out = attached_viewer(relay)
    controller, controller_out = peer_reading()
    relay._controllers = {1: controller}
    # 99, not 1: hub ids are their own sequence starting at 1, so a controller
    # numbering from 1 would agree with the first hub id by coincidence and
    # prove nothing about the remapping.
    relay._from_controller(1, controller, json.dumps({"id": 99, "method": "m"}))

    (_, payload), *_ = server_frames(viewer_out.getvalue())
    forwarded = json.loads(payload)
    assert forwarded["method"] == "m"
    assert forwarded["id"] != 99, "the controller's own id must not reach the viewer"
    assert controller_out.getvalue() == b""

    relay._from_viewer(json.dumps({"id": forwarded["id"], "result": 42}))
    (_, reply), *_ = server_frames(controller_out.getvalue())
    assert json.loads(reply) == {"id": 99, "result": 42}
    assert relay.pending_count == 0


def test_two_controllers_numbering_from_one_do_not_collide() -> None:
    """The bug this id remapping exists to prevent, stated as a test."""
    relay = template.Relay()
    _, viewer_out = attached_viewer(relay)
    first, first_out = peer_reading()
    second, second_out = peer_reading()
    relay._controllers = {1: first, 2: second}

    relay._from_controller(1, first, json.dumps({"id": 1, "method": "a"}))
    relay._from_controller(2, second, json.dumps({"id": 1, "method": "b"}))
    forwarded = [json.loads(p) for _, p in server_frames(viewer_out.getvalue())]
    assert forwarded[0]["id"] != forwarded[1]["id"]

    # Answer the SECOND one first: the reply must reach the second controller.
    relay._from_viewer(json.dumps({"id": forwarded[1]["id"], "result": "b"}))
    assert first_out.getvalue() == b""
    (_, reply), *_ = server_frames(second_out.getvalue())
    assert json.loads(reply) == {"id": 1, "result": "b"}


def test_notifications_are_forwarded_verbatim_and_remembered_not_at_all() -> None:
    relay = template.Relay()
    _, viewer_out = attached_viewer(relay)
    controller, _ = peer_reading()
    raw = json.dumps({"method": "setDimensionValue", "params": [3, 7]})
    relay._from_controller(1, controller, raw)
    assert [p for _, p in server_frames(viewer_out.getvalue())] == [raw.encode()]
    assert relay.pending_count == 0


def test_no_viewer_answers_requests_and_drops_notifications() -> None:
    relay = template.Relay()
    controller, out = peer_reading()
    relay._from_controller(1, controller, json.dumps({"method": "m"}))
    assert out.getvalue() == b"", "a notification with no display needs no answer"

    relay._from_controller(1, controller, json.dumps({"id": 9, "method": "m"}))
    (_, payload), *_ = server_frames(out.getvalue())
    assert json.loads(payload)["error"]["code"] == template.NO_VIEWER
    assert json.loads(payload)["id"] == 9


def test_malformed_controller_frames_get_named_errors() -> None:
    relay = template.Relay()
    controller, out = peer_reading()
    relay._from_controller(1, controller, "{not json")
    relay._from_controller(1, controller, "[1, 2]")
    codes = [json.loads(p)["error"]["code"] for _, p in server_frames(out.getvalue())]
    assert codes == [template.PARSE_ERROR, template.INVALID_REQUEST]


def test_a_controllers_response_frame_is_ignored() -> None:
    """A controller has nothing to answer, so a `result` from one is noise."""
    relay = template.Relay()
    _, viewer_out = attached_viewer(relay)
    controller, out = peer_reading()
    relay._from_controller(1, controller, json.dumps({"id": 1, "result": "nonsense"}))
    assert out.getvalue() == b""
    assert viewer_out.getvalue() == b""


def test_viewer_events_fan_out_to_every_controller() -> None:
    relay = template.Relay()
    first, first_out = peer_reading()
    second, second_out = peer_reading()
    relay._controllers = {1: first, 2: second}
    raw = json.dumps({"method": template.EVENT_METHOD, "params": ["arrived", {}]})
    relay._from_viewer(raw)
    assert [p for _, p in server_frames(first_out.getvalue())] == [raw.encode()]
    assert [p for _, p in server_frames(second_out.getvalue())] == [raw.encode()]


def test_only_event_notifications_fan_out() -> None:
    """The contract names one notification method; anything else is not ours."""
    relay = template.Relay()
    controller, out = peer_reading()
    relay._controllers = {1: controller}
    relay._from_viewer(json.dumps({"method": "somethingElse", "params": []}))
    assert out.getvalue() == b""


def test_a_reply_to_an_unknown_id_is_dropped() -> None:
    """A duplicate from a second display, or one whose asker has gone."""
    relay = template.Relay()
    controller, out = peer_reading()
    relay._controllers = {1: controller}
    relay._from_viewer(json.dumps({"id": 999, "result": 1}))
    relay._from_viewer(json.dumps({"id": "not-a-number", "result": 1}))
    relay._from_viewer("{not json")
    assert out.getvalue() == b""


def test_pending_is_bounded_per_controller() -> None:
    """A wedged display must not leak one dict entry per tap, forever."""
    relay = template.Relay()
    attached_viewer(relay)
    controller, _ = peer_reading()
    for index in range(template.MAX_PENDING_PER_CONTROLLER * 2):
        relay._from_controller(1, controller, json.dumps({"id": index, "method": "m"}))
    assert relay.pending_count == template.MAX_PENDING_PER_CONTROLLER


def test_dropping_a_controller_purges_its_pending_replies() -> None:
    relay = template.Relay()
    attached_viewer(relay)
    controller, _ = peer_reading()
    relay._controllers = {7: controller}
    relay._from_controller(7, controller, json.dumps({"id": 1, "method": "m"}))
    assert relay.pending_count == 1
    relay._drop_controller(7)
    assert relay.pending_count == 0
    assert relay.controller_count == 0


def test_a_viewer_that_has_gone_is_reported_to_the_asker() -> None:
    """Write failures surface as an error, not as a reply that never comes."""

    class _Gone(template._Peer):
        def send(self, text: str) -> bool:
            return False

    relay = template.Relay()
    relay._viewer = _Gone(io.BytesIO(), io.BytesIO())
    controller, out = peer_reading()
    relay._from_controller(1, controller, json.dumps({"id": 4, "method": "m"}))
    (_, payload), *_ = server_frames(out.getvalue())
    assert json.loads(payload)["error"]["code"] == template.NO_VIEWER


# ── addresses ───────────────────────────────────────────────────────────────


def test_advertised_host_resolves_a_wildcard_bind() -> None:
    """`http://0.0.0.0:8000` is not a destination, so it is never printed."""
    assert template.advertised_host("127.0.0.1") == "127.0.0.1"
    assert template.advertised_host("kiosk.local") == "kiosk.local"
    for wildcard in ("0.0.0.0", "::", ""):
        assert template.advertised_host(wildcard) not in template._ALL_INTERFACES


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("10.0.0.55", False),
    ],
)
def test_is_loopback_host(host: str, loopback: bool) -> None:
    assert template.is_loopback_host(host) is loopback


def test_urls_are_dialable() -> None:
    """The printed URLs carry the data source, the flag and the token."""
    (viewer,) = template.urls("127.0.0.1", 8000, control=False, token=None)
    assert viewer == "http://127.0.0.1:8000/viewer/?src=http://127.0.0.1:8000/data"

    viewer, panel = template.urls("127.0.0.1", 8000, control=True, token="a b")
    assert viewer.endswith("&control&controlToken=a%20b")
    assert (
        panel == "http://127.0.0.1:8000/viewer/control.html?control&controlToken=a%20b"
    )


def test_an_ipv6_bind_actually_serves(tmp_path: Path) -> None:
    """`--host ::1` must bind, not just format a bracketed URL.

    The bracketing test below only exercises string formatting, which is how
    this got through the first time: `find_port` probes AF_INET6 for a
    colon-bearing host and succeeds, then `ThreadingHTTPServer`'s hardcoded
    AF_INET made the bind raise `gaierror` — a bare traceback AFTER the script
    had printed nothing and looked like it was starting.
    """
    from urllib.request import urlopen

    port = template.find_port("::1", start=free_port())
    assert port is not None, "no IPv6 loopback port available"
    (tmp_path / "marker.txt").write_text("served over v6")

    handler = partial(template.LuxarHandler, directory=str(tmp_path))
    server = template.ControlServer(("::1", port), handler, None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = urlopen(f"http://[::1]:{port}/marker.txt", timeout=5)
        assert response.status == 200
        assert b"served over v6" in response.read()
    finally:
        server.shutdown()
        server.server_close()


def test_ipv6_hosts_are_bracketed() -> None:
    """An unbracketed IPv6 address in a URL is not a URL."""
    (viewer,) = template.urls("::1", 8000, control=False, token=None)
    assert viewer.startswith("http://[::1]:8000/viewer/")


# ── end to end, against the GENERATED script ────────────────────────────────


def free_port() -> int:
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def in_process_server(
    tmp_path: Path, token: Optional[str] = None, control: bool = True
) -> Iterator[int]:
    """The template's own server, on a real port. Fast; no subprocess."""
    port = free_port()
    handler = partial(template.LuxarHandler, directory=str(tmp_path))
    relay = template.Relay(token) if control else None
    server = template.ControlServer(("127.0.0.1", port), handler, relay)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


def dial(
    port: int, role: str = "controller", token: Optional[str] = None, **kwargs: Any
) -> ClientConnection:
    query = f"?role={role}" + (f"&token={token}" if token else "")
    return connect(f"ws://127.0.0.1:{port}/control{query}", open_timeout=5, **kwargs)


def closed_code(client: ClientConnection) -> int:
    """The close code the relay refused this handshake with."""
    with pytest.raises(ConnectionClosed) as refusal:
        client.recv(timeout=5)
    received = refusal.value.rcvd
    assert received is not None
    return int(received.code)


def test_a_tap_travels_from_panel_to_display(tmp_path: Path) -> None:
    """The whole point, over real sockets: a controller drives a viewer."""
    with in_process_server(tmp_path) as port:
        with dial(port, "viewer") as viewer, dial(port, "controller") as panel:
            panel.send(json.dumps({"id": 1, "method": "setDimensionValue"}))
            forwarded = json.loads(viewer.recv(timeout=5))
            assert forwarded["method"] == "setDimensionValue"

            viewer.send(json.dumps({"id": forwarded["id"], "result": None}))
            assert json.loads(panel.recv(timeout=5)) == {"id": 1, "result": None}

            viewer.send(json.dumps({"method": "event", "params": ["arrived"]}))
            assert json.loads(panel.recv(timeout=5))["params"] == ["arrived"]


def test_a_bad_token_is_refused_with_a_policy_violation(tmp_path: Path) -> None:
    with in_process_server(tmp_path, token="s3cret") as port:
        with dial(port, "controller", token="wrong") as client:
            assert closed_code(client) == template.CLOSE_POLICY_VIOLATION
        with dial(port, "controller", token="s3cret") as client:
            client.send(json.dumps({"id": 1, "method": "m"}))
            assert json.loads(client.recv(timeout=5))["error"]["code"] == (
                template.NO_VIEWER
            )


def test_a_cross_origin_browser_handshake_is_refused(tmp_path: Path) -> None:
    """The CSWSH case, end to end: a page on another origin cannot attach."""
    with in_process_server(tmp_path) as port:
        with dial(
            port, "controller", additional_headers={"Origin": "http://evil.example"}
        ) as client:
            assert closed_code(client) == template.CLOSE_POLICY_VIOLATION
        with dial(
            port,
            "controller",
            additional_headers={"Origin": f"http://127.0.0.1:{port}"},
        ) as client:
            client.send(json.dumps({"id": 1, "method": "m"}))
            assert json.loads(client.recv(timeout=5))["error"]["code"] == (
                template.NO_VIEWER
            )


def test_an_unknown_role_is_refused_over_the_wire(tmp_path: Path) -> None:
    with in_process_server(tmp_path) as port:
        with dial(port, "viewr") as client:
            assert closed_code(client) == template.CLOSE_POLICY_VIOLATION


def test_the_relay_is_absent_unless_asked_for(tmp_path: Path) -> None:
    """Off by default: `/control` 404s, and the socket cannot be opened."""
    from urllib.error import HTTPError
    from urllib.request import urlopen

    with in_process_server(tmp_path, control=False) as port:
        with pytest.raises(HTTPError) as failure:
            urlopen(f"http://127.0.0.1:{port}/control", timeout=5)
        assert failure.value.code == 404
        # The handshake itself is refused with the same 404, named exactly:
        # `pytest.raises(Exception)` here would also pass if the client could
        # not resolve the host, which would assert nothing about the relay.
        with pytest.raises(InvalidStatus) as refused:
            dial(port).close()
        assert refused.value.response.status_code == 404


def test_a_plain_get_on_the_relay_path_is_a_clear_error(tmp_path: Path) -> None:
    """Navigating to /control must say what is wrong, not hang the socket."""
    from urllib.error import HTTPError
    from urllib.request import urlopen

    with in_process_server(tmp_path) as port:
        with pytest.raises(HTTPError) as failure:
            urlopen(f"http://127.0.0.1:{port}/control", timeout=5)
        assert failure.value.code == 400


def test_files_are_still_served_while_a_socket_is_open(tmp_path: Path) -> None:
    """Why the server is threaded.

    The single-threaded server this replaced would have been held by the first
    attached panel forever — including for the scene's own chunks, so the
    display would never finish loading.
    """
    from urllib.request import urlopen

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "zarr.json").write_text('{"node_type": "group"}')
    with in_process_server(tmp_path) as port:
        with dial(port, "viewer"):
            response = urlopen(f"http://127.0.0.1:{port}/data/zarr.json", timeout=5)
            assert response.status == 200
            assert b"node_type" in response.read()


def test_an_idle_control_socket_outlives_the_keep_alive_timeout(
    tmp_path: Path,
) -> None:
    """A panel nobody is touching must not be dropped as an idle connection.

    The server reaps idle HTTP keep-alive connections, and a control socket
    starts life as one of those — so without clearing the timeout at handshake
    time, a quiet panel would die after `LuxarHandler.timeout` seconds and the
    exhibit would stop working in front of an audience. Driven with a tiny
    timeout so the test costs a second rather than a minute.
    """

    class _Impatient(template.LuxarHandler):
        timeout = 0.3

    port = free_port()
    handler = partial(_Impatient, directory=str(tmp_path))
    server = template.ControlServer(("127.0.0.1", port), handler, template.Relay())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with dial(port, "viewer") as viewer, dial(port) as panel:
            time.sleep(1.0)  # > 3x the timeout, in silence
            panel.send(json.dumps({"id": 1, "method": "stillThere"}))
            assert json.loads(viewer.recv(timeout=5))["method"] == "stillThere"
    finally:
        server.shutdown()
        server.server_close()


def test_the_generated_script_hosts_the_relay(tmp_path: Path) -> None:
    """The script `luxar export` actually writes, run as a subprocess.

    The in-process tests above drive the template's classes; this one proves
    the generated file — substitutions and all — starts, prints a panel URL,
    and relays a real tap.
    """
    (tmp_path / "viewer").mkdir()
    (tmp_path / "viewer" / "index.html").write_text("<html></html>")
    script = tmp_path / "serve.py"
    script.write_text(_get_serve_script_content("scene_data", "My Scene"))
    port = free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "--port",
            str(port),
            "--no-open",
            "--control",
            "--control-token",
            "s3cret",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                with dial(port, "viewer", token="s3cret") as viewer:
                    with dial(port, "controller", token="s3cret") as panel:
                        panel.send(json.dumps({"id": 1, "method": "m"}))
                        assert json.loads(viewer.recv(timeout=5))["method"] == "m"
                break
            except (OSError, ConnectionClosed):
                time.sleep(0.2)
        else:
            pytest.fail("the generated script never served the relay")
    finally:
        process.terminate()
        stdout, _ = process.communicate(timeout=20)

    assert "control.html?control&controlToken=s3cret" in stdout
    assert "src=http://127.0.0.1:%d/scene_data&title=My%%20Scene" % port in stdout
