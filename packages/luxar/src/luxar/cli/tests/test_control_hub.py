"""Tests for the remote-control WebSocket hub.

Everything runs in-process through ``TestClient.websocket_connect`` — no
uvicorn, no real socket, no port to race on. The hub is a relay, so every test
here is about ROUTING: which party receives a frame, and with which id.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from luxar.cli.control_hub import (
    CLOSE_POLICY_VIOLATION,
    MAX_PENDING_PER_CONTROLLER,
    ControlHub,
)
from luxar.cli.serving import _add_control_hub


def _app(token: str | None = None) -> tuple[FastAPI, ControlHub]:
    """A bare app carrying only the control endpoint."""
    api = FastAPI()
    hub = _add_control_hub(api, token=token)
    return api, hub


def _request(method: str, request_id: Any, *params: Any) -> str:
    """A JSON-RPC request frame with positional params."""
    return json.dumps(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": list(params)}
    )


def _reply(request_id: Any, result: Any) -> str:
    """A JSON-RPC success response frame."""
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})


class TestRouting:
    """A request reaches the viewer; its reply reaches the asker alone."""

    def test_request_is_remapped_outbound_and_restored_inbound(self) -> None:
        # A STRING id, so the remap is unambiguously observable: the hub's own
        # ids are integers from its own counter, and an integer id of the
        # controller's could legitimately coincide with one.
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            controller.send_text(_request("setDimensionValue", "abc", 3, 7))

            seen: Dict[str, Any] = viewer.receive_json()
            assert seen["method"] == "setDimensionValue"
            assert seen["params"] == [3, 7]
            # The hub numbers requests itself so two controllers cannot
            # collide, so the id the viewer sees is NOT the controller's.
            assert seen["id"] != "abc"
            assert isinstance(seen["id"], int)

            viewer.send_text(_reply(seen["id"], None))
            answer = controller.receive_json()
            # ...and the controller gets its own id back.
            assert answer["id"] == "abc"
            assert answer["result"] is None

    def test_two_controllers_numbering_from_one_do_not_collide(self) -> None:
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as first,
            client.websocket_connect("/control?role=controller") as second,
        ):
            first.send_text(_request("getCameraPose", 1))
            first_seen = viewer.receive_json()
            second.send_text(_request("getLayers", 1))
            second_seen = viewer.receive_json()

            assert first_seen["id"] != second_seen["id"]

            # Answer the SECOND one first: routing must follow the id,
            # not arrival order.
            viewer.send_text(_reply(second_seen["id"], ["layer"]))
            answer = second.receive_json()
            assert answer == {
                "jsonrpc": "2.0",
                "id": 1,
                "result": ["layer"],
            }

            viewer.send_text(_reply(first_seen["id"], {"fov": 60}))
            answer = first.receive_json()
            assert answer["result"] == {"fov": 60}

    def test_a_notification_needs_no_reply_and_is_forwarded_verbatim(self) -> None:
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            controller.send_text(
                json.dumps({"jsonrpc": "2.0", "method": "recenterCamera"})
            )
            seen = viewer.receive_json()
            assert seen == {"jsonrpc": "2.0", "method": "recenterCamera"}

    def test_a_second_viewers_duplicate_reply_is_dropped(self) -> None:
        """Fan-out means N replies for one request; the controller wants one."""
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as first_viewer,
            client.websocket_connect("/control?role=viewer") as second_viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            controller.send_text(_request("getCameraPose", 9))
            seen = first_viewer.receive_json()
            assert second_viewer.receive_json()["id"] == seen["id"]

            first_viewer.send_text(_reply(seen["id"], "first"))
            assert controller.receive_json()["result"] == "first"

            # The loser's reply has no pending entry left to route on.
            second_viewer.send_text(_reply(seen["id"], "second"))
            controller.send_text(_request("getLayers", 10))
            relayed = first_viewer.receive_json()
            first_viewer.send_text(_reply(relayed["id"], []))
            # If the duplicate had been relayed it would arrive here.
            assert controller.receive_json()["id"] == 10


class TestEvents:
    """A viewer's event notifications fan out to controllers only."""

    def test_events_reach_every_controller_and_no_viewer(self) -> None:
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as first,
            client.websocket_connect("/control?role=controller") as second,
        ):
            event = {
                "jsonrpc": "2.0",
                "method": "event",
                "params": ["dimensions-changed", {}],
            }
            viewer.send_text(json.dumps(event))
            assert first.receive_json() == event
            assert second.receive_json() == event
            # And the viewer is not echoed its own event: the next thing
            # it receives is the relayed request below, not the event.
            second.send_text(_request("getLayers", 4))
            assert viewer.receive_json()["method"] == "getLayers"


class TestNoViewer:
    """Nothing to drive is an answer, not a hang."""

    def test_a_request_with_no_viewer_attached_is_refused(self) -> None:
        api, _ = _app()
        client = TestClient(api)
        with client.websocket_connect("/control?role=controller") as controller:
            controller.send_text(_request("setDimensionValue", 2, 0, 1))
            answer = controller.receive_json()
            assert answer["id"] == 2
            assert answer["error"]["code"] == -32001
            assert "no viewer" in answer["error"]["message"]

    def test_a_notification_with_no_viewer_is_dropped_silently(self) -> None:
        """A notification has no id, so there is nothing to answer it on."""
        api, _ = _app()
        client = TestClient(api)
        with client.websocket_connect("/control?role=controller") as controller:
            controller.send_text(json.dumps({"jsonrpc": "2.0", "method": "resize"}))
            # Prove the socket is still live and unconfused by asking something
            # that DOES answer.
            controller.send_text(_request("getLayers", 3))
            assert controller.receive_json()["id"] == 3


class TestMalformedFrames:
    """Garbage in gets a JSON-RPC error out, and never closes the socket."""

    @pytest.mark.parametrize(
        "raw,code",
        [
            ("not json at all", -32700),
            ("[1, 2, 3]", -32600),
            ('"a bare string"', -32600),
        ],
        ids=["unparseable", "array-frame", "scalar-frame"],
    )
    def test_a_controller_is_told_what_was_wrong(self, raw: str, code: int) -> None:
        api, _ = _app()
        client = TestClient(api)
        with client.websocket_connect("/control?role=controller") as controller:
            controller.send_text(raw)
            answer = controller.receive_json()
            assert answer["id"] is None
            assert answer["error"]["code"] == code

    def test_a_stray_response_from_a_controller_is_ignored(self) -> None:
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            controller.send_text(_reply(1, "I am not a viewer"))
            controller.send_text(_request("getLayers", 5))
            assert viewer.receive_json()["method"] == "getLayers"

    def test_binary_frames_do_not_close_the_socket(self) -> None:
        api, _ = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            controller.send_bytes(b"\x01\x02\x03")
            answer = controller.receive_json()
            assert answer["id"] is None
            assert answer["error"]["code"] == -32600

            controller.send_text(_request("getLayers", 5))
            request = viewer.receive_json()
            viewer.send_bytes(b"\x04\x05")
            viewer.send_text(_reply(request["id"], "still attached"))
            assert controller.receive_json()["result"] == "still attached"

    @pytest.mark.parametrize("bad_id", [[1], {"request": 1}], ids=["array", "object"])
    def test_unroutable_reply_ids_do_not_close_the_socket(self, bad_id: object) -> None:
        api, hub = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            controller.send_text(_request("getLayers", 5))
            request = viewer.receive_json()
            viewer.send_json({"jsonrpc": "2.0", "id": bad_id, "result": "not routable"})
            viewer.send_text(_reply(request["id"], "still attached"))
            assert controller.receive_json()["result"] == "still attached"
            assert hub.pending_count == 0


class TestAttachment:
    """Who may attach, and what happens when they leave."""

    def test_an_unknown_role_is_refused_rather_than_defaulted(self) -> None:
        """`?role=viewr` must not silently attach as a CONTROLLER."""
        api, hub = _app()
        client = TestClient(api)
        with client.websocket_connect("/control?role=viewr") as socket:
            message = socket.receive()
            assert message["type"] == "websocket.close"
            assert message["code"] == CLOSE_POLICY_VIOLATION
        assert hub.controller_count == 0
        assert hub.viewer_count == 0

    def test_the_default_role_is_controller(self) -> None:
        api, hub = _app()
        client = TestClient(api)
        with client.websocket_connect("/control"):
            assert hub.controller_count == 1
            assert hub.viewer_count == 0

    def test_a_bad_token_is_closed_with_a_policy_violation(self) -> None:
        api, hub = _app(token="opensesame")
        client = TestClient(api)
        with client.websocket_connect("/control?token=wrong") as socket:
            message = socket.receive()
            assert message["type"] == "websocket.close"
            assert message["code"] == CLOSE_POLICY_VIOLATION
        assert hub.controller_count == 0

    def test_a_missing_token_is_refused_when_one_is_required(self) -> None:
        api, hub = _app(token="opensesame")
        client = TestClient(api)
        with client.websocket_connect("/control") as socket:
            assert socket.receive()["code"] == CLOSE_POLICY_VIOLATION
        assert hub.controller_count == 0

    def test_the_right_token_attaches(self) -> None:
        api, hub = _app(token="opensesame")
        client = TestClient(api)
        with client.websocket_connect("/control?token=opensesame"):
            assert hub.controller_count == 1

    def test_counts_drop_when_a_party_detaches(self) -> None:
        api, hub = _app()
        client = TestClient(api)
        with client.websocket_connect("/control?role=viewer"):
            assert hub.viewer_count == 1
        assert hub.viewer_count == 0


class TestPendingHygiene:
    """The pending table must not grow without bound."""

    def test_a_departed_controllers_requests_are_forgotten(self) -> None:
        api, hub = _app()
        with TestClient(api) as client:
            with client.websocket_connect("/control?role=viewer") as viewer:
                with client.websocket_connect("/control?role=controller") as controller:
                    controller.send_text(_request("getCameraPose", 1))
                    viewer.receive_json()
                    assert hub.pending_count == 1
                # Its replies are unroutable now, so holding them is a leak.
                assert hub.pending_count == 0

    def test_a_wedged_viewer_cannot_grow_the_table_past_the_cap(self) -> None:
        api, hub = _app()
        with (
            TestClient(api) as client,
            client.websocket_connect("/control?role=viewer") as viewer,
            client.websocket_connect("/control?role=controller") as controller,
        ):
            overshoot = MAX_PENDING_PER_CONTROLLER + 10
            for request_id in range(overshoot):
                controller.send_text(_request("getCameraPose", request_id))
            for _ in range(overshoot):
                viewer.receive_json()  # relayed, never answered
            assert hub.pending_count == MAX_PENDING_PER_CONTROLLER

    def test_the_newest_requests_are_the_ones_kept(self) -> None:
        """Eviction drops the oldest, which is the one least likely to matter."""
        hub = ControlHub()
        controller_key = 7
        relayed = [
            hub._remember(controller_key, request_id)
            for request_id in range(MAX_PENDING_PER_CONTROLLER + 1)
        ]

        assert hub.pending_count == MAX_PENDING_PER_CONTROLLER
        assert list(hub._pending) == relayed[1:]
        assert hub._pending[relayed[-1]] == (
            controller_key,
            MAX_PENDING_PER_CONTROLLER,
        )


class TestViewerAppWiring:
    """The hub's place in the real viewer app, mount order included."""

    @staticmethod
    def _viewer_dist(tmp_path: Any) -> Any:
        viewer = tmp_path / "viewer"
        viewer.mkdir()
        (viewer / "index.html").write_text("<html><body>shell</body></html>")
        return viewer

    def test_control_is_off_by_default(self, tmp_path: Any) -> None:
        from luxar.cli.serving import _build_viewer_app

        client = TestClient(_build_viewer_app(self._viewer_dist(tmp_path)))
        # Without --control there is no endpoint, so the static handler answers
        # and the WebSocket handshake cannot complete.
        with pytest.raises(Exception):  # noqa: B017 - starlette's own type varies
            with client.websocket_connect("/control"):
                pass

    def test_control_endpoint_coexists_with_the_static_mount(
        self, tmp_path: Any
    ) -> None:
        """Mount order is load-bearing: a Mount at "/" swallows every path
        registered after it, WebSockets included."""
        from luxar.cli.serving import _build_viewer_app

        api = _build_viewer_app(self._viewer_dist(tmp_path), control=True)
        client = TestClient(api)

        # The socket is reachable...
        with client.websocket_connect("/control?role=viewer"):
            assert api.state.control_hub.viewer_count == 1

        # ...and the viewer shell it shares an origin with still serves.
        assert client.get("/index.html").status_code == 200
        assert "shell" in client.get("/").text

    def test_the_token_reaches_the_hub(self, tmp_path: Any) -> None:
        from luxar.cli.serving import _build_viewer_app

        api = _build_viewer_app(
            self._viewer_dist(tmp_path), control=True, control_token="hunter2"
        )
        client = TestClient(api)
        with client.websocket_connect("/control?token=hunter2"):
            assert api.state.control_hub.controller_count == 1
        with client.websocket_connect("/control") as socket:
            assert socket.receive()["code"] == CLOSE_POLICY_VIOLATION

    def test_network_control_without_a_token_warns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from luxar.cli import serving

        monkeypatch.setattr(serving, "get_viewer_dist_path", lambda: "viewer")
        monkeypatch.setattr(
            serving, "_build_viewer_app", lambda *args, **kwargs: object()
        )
        monkeypatch.setattr(serving.uvicorn, "run", lambda *args, **kwargs: None)

        serving._serve_viewer("0.0.0.0", 5173, control=True, open_browser_flag=False)

        output = capsys.readouterr().out
        assert "reachable from the network without a token" in output

    def test_control_panel_url_quotes_the_token(self) -> None:
        from luxar.cli.serving import _build_control_panel_url

        assert _build_control_panel_url("kiosk.local", 5173) == (
            "http://kiosk.local:5173/control.html?control"
        )
        assert _build_control_panel_url("kiosk.local", 5173, "space & slash/") == (
            "http://kiosk.local:5173/control.html?control&controlToken=space%20%26%20slash%2F"
        )


class TestOriginCheck:
    """Cross-Site WebSocket Hijacking is the attack this closes.

    A WebSocket handshake is not subject to the same-origin policy and has no
    CORS preflight, so without an ``Origin`` check any page a visitor opens
    could drive the kiosk — and read its dataset URL back out of
    ``getViewerState()``.
    """

    def test_a_cross_origin_browser_handshake_is_refused(self) -> None:
        api, hub = _app()
        client = TestClient(api)
        with client.websocket_connect(
            "/control", headers={"origin": "http://evil.example"}
        ) as socket:
            # Asserted BEFORE the receive, and deliberately: if the check were
            # removed, the socket would attach and `receive()` would block
            # forever waiting for a close that never comes. A hang is a much
            # worse failure than an assertion, so the fast check goes first.
            assert hub.controller_count == 0, "a cross-origin handshake attached"
            message = socket.receive()
            assert message["type"] == "websocket.close"
            assert message["code"] == CLOSE_POLICY_VIOLATION

    def test_a_same_origin_handshake_attaches(self) -> None:
        api, hub = _app()
        client = TestClient(api)
        # TestClient sends Host: testserver.
        with client.websocket_connect(
            "/control", headers={"origin": "http://testserver"}
        ):
            assert hub.controller_count == 1

    def test_the_scheme_may_differ_from_the_host_header(self) -> None:
        # An https page dials wss; `Host` carries no scheme, so only host:port
        # can be compared.
        api, hub = _app()
        client = TestClient(api)
        with client.websocket_connect(
            "/control", headers={"origin": "https://testserver"}
        ):
            assert hub.controller_count == 1

    def test_a_handshake_with_no_origin_attaches(self) -> None:
        """Non-browser clients send none, `luxar.control.Viewer` included."""
        api, hub = _app()
        client = TestClient(api)
        with client.websocket_connect("/control"):
            assert hub.controller_count == 1

    @pytest.mark.parametrize(
        "origin,host,expected",
        [
            (None, "kiosk.local:5173", True),
            ("http://kiosk.local:5173", "kiosk.local:5173", True),
            ("https://kiosk.local:5173", "kiosk.local:5173", True),
            ("HTTP://KIOSK.LOCAL:5173", "kiosk.local:5173", True),
            ("http://kiosk.local", "kiosk.local:5173", False),
            ("http://evil.example", "kiosk.local:5173", False),
            # A sandboxed iframe sends the literal string "null"; it must never
            # match a real host.
            ("null", "kiosk.local:5173", False),
            ("http://kiosk.local:5173", None, False),
        ],
    )
    def test_origin_allowed(
        self, origin: str | None, host: str | None, expected: bool
    ) -> None:
        assert ControlHub().origin_allowed(origin, host) is expected

    def test_an_explicit_allowlist_wins_over_the_host_comparison(self) -> None:
        # The reverse-proxy case, where Origin and Host differ honestly.
        hub = ControlHub(allowed_origins=[" HTTPS://Exhibit.Museum/ "])
        assert hub.origin_allowed("https://EXHIBIT.MUSEUM/", "internal:8000") is True
        assert hub.origin_allowed("https://internal:8000", "internal:8000") is False

    def test_a_valid_token_allows_an_explicit_split_origin(self) -> None:
        api, hub = _app(token="secret")
        client = TestClient(api)
        with client.websocket_connect(
            "/control?role=viewer&token=secret",
            headers={"origin": "https://display.example"},
        ):
            assert hub.viewer_count == 1


class TestAuthorized:
    """The predicate itself, including the open-hub case."""

    @pytest.mark.parametrize(
        "configured,presented,expected",
        [
            (None, None, True),
            (None, "anything", True),
            ("secret", "secret", True),
            ("secret", "Secret", False),
            ("secret", "", False),
            ("secret", None, False),
            # Non-ASCII must REFUSE, not raise: `hmac.compare_digest` throws a
            # TypeError on a non-ASCII str, which would turn a wrong password
            # into a 500 and take the socket down with it.
            ("secret", "sécret", False),
            ("sécret", "sécret", True),
            ("sécret", "secret", False),
            ("🔑", "🔑", True),
            ("🔑", "🗝", False),
        ],
    )
    def test_token_comparison(
        self, configured: str | None, presented: str | None, expected: bool
    ) -> None:
        assert ControlHub(token=configured).authorized(presented) is expected

    def test_a_non_ascii_token_attaches_over_a_real_socket(self) -> None:
        """End to end, because the TypeError was raised inside the handler."""
        api, hub = _app(token="sécret")
        client = TestClient(api)
        with client.websocket_connect("/control?token=s%C3%A9cret"):
            assert hub.controller_count == 1
