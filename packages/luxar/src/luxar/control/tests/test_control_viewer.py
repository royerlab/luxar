"""Tests for the Python controller.

The named-method layer and the URL/error plumbing are exercised against a fake
socket — there is no value in a real one for asserting that
``set_dimension_value`` sends ``setDimensionValue``. One test does run a real
socket against a real uvicorn-hosted hub, because the handshake, the query
parameters and the id round-trip are exactly the things a fake cannot vouch for.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import pytest

from luxar.control import ControlError, Viewer
from luxar.control.viewer import MAX_BUFFERED_EVENTS, _connect, _with_query


class FakeSocket:
    """Records what was sent and replies from a scripted queue."""

    def __init__(self, replies: Optional[List[Dict[str, Any]]] = None) -> None:
        self.sent: List[Dict[str, Any]] = []
        self.replies = list(replies or [])
        self.closed = False
        self.recv_timeouts: List[Optional[float]] = []
        #: Echo the request id onto each scripted reply, so a test does not
        #: have to predict the counter.
        self.echo_id = True

    def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    def recv(self, timeout: Optional[float] = None) -> str:
        self.recv_timeouts.append(timeout)
        if not self.replies:
            raise TimeoutError
        reply = self.replies.pop(0)
        if self.echo_id and "id" not in reply:
            reply = {**reply, "id": self.sent[-1]["id"]}
        return json.dumps({"jsonrpc": "2.0", **reply})

    def close(self) -> None:
        self.closed = True

    def last(self) -> Dict[str, Any]:
        return self.sent[-1]


def make_viewer(monkeypatch: pytest.MonkeyPatch, socket: FakeSocket) -> Viewer:
    """A Viewer wired to ``socket`` instead of a real connection."""
    monkeypatch.setattr("luxar.control.viewer._connect", lambda url, timeout: socket)
    return Viewer("ws://host:5173/control")


class TestUrl:
    """The hub reads two query parameters; both are ours to set."""

    def test_role_is_always_controller(self) -> None:
        query = parse_qs(
            urlparse(_with_query("ws://h/control", role="controller", token=None)).query
        )
        assert query == {"role": ["controller"]}

    def test_token_is_carried_when_given(self) -> None:
        url = _with_query("ws://h/control", role="controller", token="hunter2")
        assert parse_qs(urlparse(url).query)["token"] == ["hunter2"]

    def test_an_empty_token_is_omitted(self) -> None:
        url = _with_query("ws://h/control", role="controller", token="")
        assert "token" not in parse_qs(urlparse(url).query)

    def test_an_existing_query_is_replaced_not_appended(self) -> None:
        # Two roles would be ambiguous, and the hub reads only one.
        url = _with_query("ws://h/control?role=viewer", role="controller", token=None)
        assert parse_qs(urlparse(url).query)["role"] == ["controller"]

    def test_the_path_survives(self) -> None:
        assert "/nested/control" in _with_query(
            "ws://h/nested/control", role="controller", token=None
        )


class TestCall:
    """The wire shape of a request, and what comes back."""

    def test_params_are_positional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket([{"result": None}])
        viewer = make_viewer(monkeypatch, socket)
        viewer.call("setDimensionValue", 3, 7)
        assert socket.last()["method"] == "setDimensionValue"
        assert socket.last()["params"] == [3, 7]

    def test_ids_increment_so_replies_are_matchable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket([{"result": 1}, {"result": 2}])
        viewer = make_viewer(monkeypatch, socket)
        viewer.call("getLayers")
        viewer.call("getLayers")
        assert [frame["id"] for frame in socket.sent] == [1, 2]

    def test_a_result_is_returned_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket([{"result": {"fov": 60}}])
        viewer = make_viewer(monkeypatch, socket)
        assert viewer.call("getCameraPose") == {"fov": 60}

    def test_a_falsy_result_is_not_mistaken_for_absence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket([{"result": False}])
        viewer = make_viewer(monkeypatch, socket)
        assert viewer.call("playSound", "x") is False

    def test_frames_that_are_not_the_answer_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Events share the socket with replies and remain available."""
        socket = FakeSocket(
            [
                {"method": "event", "params": ["camera-changed", {}], "id": None},
                {"result": "mine"},
            ]
        )
        socket.echo_id = False
        socket.replies[1]["id"] = 1
        viewer = make_viewer(monkeypatch, socket)
        assert viewer.call("getLayers") == "mine"
        assert viewer.recv_event() == ("camera-changed", {})

    def test_events_do_not_reset_the_reply_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket(
            [
                {"method": "event", "params": ["camera-changed", {}], "id": None},
                {"result": "mine", "id": 1},
            ]
        )
        socket.echo_id = False
        monkeypatch.setattr(
            "luxar.control.viewer._connect", lambda url, timeout: socket
        )
        monotonic = iter([100.0, 100.0, 101.0])
        monkeypatch.setattr(
            "luxar.control.viewer.time.monotonic", lambda: next(monotonic)
        )

        viewer = Viewer("ws://host:5173/control", timeout_s=5.0)

        assert viewer.call("getLayers") == "mine"
        assert socket.recv_timeouts == [5.0, 4.0]

    def test_a_notification_asks_for_no_reply(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket()
        viewer = make_viewer(monkeypatch, socket)
        viewer.notify("recenterCamera")
        assert "id" not in socket.last()
        assert socket.sent[-1]["method"] == "recenterCamera"


class TestErrors:
    """A refusal must arrive as an exception carrying its code."""

    def test_an_error_frame_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket([{"error": {"code": -32601, "message": "not exposed"}}])
        viewer = make_viewer(monkeypatch, socket)
        with pytest.raises(ControlError) as caught:
            viewer.call("dispose")
        assert caught.value.code == -32601
        assert "not exposed" in caught.value.message

    @pytest.mark.parametrize(
        "error",
        ["nope", [], {"code": "abc"}, {"code": {"nested": True}}],
    )
    def test_a_malformed_error_frame_still_raises_control_error(
        self, monkeypatch: pytest.MonkeyPatch, error: Any
    ) -> None:
        socket = FakeSocket([{"error": error}])
        viewer = make_viewer(monkeypatch, socket)

        with pytest.raises(ControlError) as caught:
            viewer.call("getLayers")

        assert caught.value.code == -32603
        assert repr(error) in caught.value.message

    def test_an_integer_string_error_code_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket([{"error": {"code": "-32601", "message": "not exposed"}}])
        viewer = make_viewer(monkeypatch, socket)

        with pytest.raises(ControlError) as caught:
            viewer.call("dispose")

        assert caught.value.code == -32601
        assert caught.value.message == "not exposed"

    def test_no_viewer_attached_is_distinguishable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A display that has not booted yet is something to wait for, not an
        # error to abort a kiosk script on.
        socket = FakeSocket(
            [{"error": {"code": -32001, "message": "no viewer attached"}}]
        )
        viewer = make_viewer(monkeypatch, socket)
        with pytest.raises(ControlError) as caught:
            viewer.call("getLayers")
        assert caught.value.no_viewer_attached is True

    def test_an_ordinary_failure_is_not_flagged_as_no_viewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket([{"error": {"code": -32603, "message": "boom"}}])
        viewer = make_viewer(monkeypatch, socket)
        with pytest.raises(ControlError) as caught:
            viewer.call("getLayers")
        assert caught.value.no_viewer_attached is False


class TestConnection:
    def test_raises_the_frame_cap_for_screenshot_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: Dict[str, Any] = {}

        class Connection:
            def __enter__(self) -> "Connection":
                return self

        def fake_connect(url: str, **kwargs: Any) -> Connection:
            captured.update(url=url, **kwargs)
            return Connection()

        monkeypatch.setattr("websockets.sync.client.connect", fake_connect)

        _connect("ws://host:5173/control", 7.0)

        assert captured == {
            "url": "ws://host:5173/control",
            "open_timeout": 7.0,
            "max_size": 16 * 1024 * 1024,
        }


class TestNamedMethods:
    """Snake_case here, camelCase on the wire."""

    def test_set_dimension_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket([{"result": None}])
        make_viewer(monkeypatch, socket).set_dimension_value(2, 5)
        assert socket.last()["method"] == "setDimensionValue"
        assert socket.last()["params"] == [2, 5]

    def test_recenter_camera(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket([{"result": None}])
        make_viewer(monkeypatch, socket).recenter_camera()
        assert socket.last()["method"] == "recenterCamera"

    def test_fly_to_forwards_its_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket([{"result": {"completed": True}}])
        viewer = make_viewer(monkeypatch, socket)
        assert viewer.fly_to({"position": [0, 0, 1]}, durationMs=2000) == {
            "completed": True
        }
        assert socket.last()["params"] == [
            {"position": [0, 0, 1]},
            {"durationMs": 2000},
        ]

    def test_subscribe_and_unsubscribe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket([{"result": True}, {"result": True}])
        viewer = make_viewer(monkeypatch, socket)
        viewer.subscribe("waypoint-arrived")
        assert socket.last()["params"] == ["waypoint-arrived"]
        viewer.unsubscribe("waypoint-arrived")
        assert socket.last()["method"] == "unsubscribe"

    def test_dimension_index_finds_a_dimension_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The point of the helper: a script must not hard-code a position that
        # moves when the author reorders Dimensions([...]).
        socket = FakeSocket(
            [
                {
                    "result": {
                        "metadata": [{"name": "z"}, {"name": "y"}, {"name": "story"}]
                    }
                }
            ]
        )
        viewer = make_viewer(monkeypatch, socket)
        assert viewer.dimension_index("story") == 2

    def test_dimension_index_names_what_is_available_when_it_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket([{"result": {"metadata": [{"name": "z"}]}}])
        viewer = make_viewer(monkeypatch, socket)
        with pytest.raises(KeyError, match="stroy"):
            viewer.dimension_index("stroy")


class TestEvents:
    def test_recv_event_returns_a_notification(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket(
            [{"method": "event", "params": ["waypoint-arrived", {"index": 3}]}]
        )
        socket.echo_id = False
        viewer = make_viewer(monkeypatch, socket)

        assert viewer.recv_event() == ("waypoint-arrived", {"index": 3})
        assert socket.recv_timeouts == [None]

    def test_recv_event_returns_none_at_its_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket = FakeSocket()
        viewer = make_viewer(monkeypatch, socket)

        assert viewer.recv_event(timeout_s=0.01) is None
        assert socket.recv_timeouts[0] is not None
        assert 0 <= socket.recv_timeouts[0] <= 0.01


class TestEventBufferIsBounded:
    """A kiosk controller runs for hours. The buffer must not grow for hours.

    `call` drains the socket while it waits for its reply, so events land in
    the buffer during exactly the long waits (`flyTo`, `switchDataset`) when a
    20 Hz `camera-changed` stream is arriving — and the hub fans events to
    EVERY attached controller, so a script that subscribed to nothing still
    receives whatever a touch panel asked for. Unbounded, that measured ~91 MB
    after an hour.
    """

    def test_the_buffer_stops_growing_at_the_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        overflow = 50
        events = [
            {"method": "event", "params": ["camera-changed", {"n": i}]}
            for i in range(MAX_BUFFERED_EVENTS + overflow)
        ]
        # One real reply at the end, so `call` returns rather than timing out
        # after draining every event frame ahead of it. Its id is explicit
        # because `echo_id` is off — the event frames must stay id-less to be
        # recognised as notifications, and the id counter starts at 1.
        socket = FakeSocket([*events, {"id": 1, "result": None}])
        socket.echo_id = False
        viewer = make_viewer(monkeypatch, socket)

        viewer.call("recenterCamera")

        assert len(viewer._events) == MAX_BUFFERED_EVENTS

    def test_the_newest_events_are_the_ones_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping the OLDEST is the useful direction.

        A reader that falls behind wants the current camera pose, not the one
        from a thousand frames ago — and a `maxlen` deque that dropped from the
        wrong end would keep exactly the stale ones.
        """
        overflow = 3
        events = [
            {"method": "event", "params": ["camera-changed", {"n": i}]}
            for i in range(MAX_BUFFERED_EVENTS + overflow)
        ]
        socket = FakeSocket([*events, {"id": 1, "result": None}])
        socket.echo_id = False
        viewer = make_viewer(monkeypatch, socket)

        viewer.call("recenterCamera")

        first = viewer.recv_event()
        assert first == ("camera-changed", {"n": overflow})
        # And the very newest survived to the other end of the buffer.
        drained = [first]
        while viewer._events:
            drained.append(viewer.recv_event())
        assert drained[-1] == (
            "camera-changed",
            {"n": MAX_BUFFERED_EVENTS + overflow - 1},
        )


class TestLifecycle:
    def test_close_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket()
        viewer = make_viewer(monkeypatch, socket)
        viewer.close()
        viewer.close()
        assert socket.closed is True

    def test_context_manager_closes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket()
        monkeypatch.setattr(
            "luxar.control.viewer._connect", lambda url, timeout: socket
        )
        with Viewer("ws://host/control"):
            assert socket.closed is False
        assert socket.closed is True


@pytest.mark.slow
def test_end_to_end_against_a_live_hub() -> None:
    """A real socket, a real hub, a real id round-trip.

    This is the one thing the fake cannot vouch for: the handshake, the query
    parameters actually reaching the endpoint, and the hub's private id remap
    coming back as the id this client sent. A stub "viewer" plays the other
    side, answering whatever it is asked.
    """
    import uvicorn
    from fastapi import FastAPI

    from luxar.cli.serving import _add_control_hub
    from luxar.cli.utils import pick_port

    api = FastAPI()
    _add_control_hub(api)
    port = pick_port(8911, "127.0.0.1", label="control-test")
    assert port is not None

    server = uvicorn.Server(
        uvicorn.Config(api, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True, name="control-hub-test")
    thread.start()
    try:
        _wait_until(lambda: server.started, "the hub did not start")

        from websockets.sync.client import connect

        url = f"ws://127.0.0.1:{port}/control"
        with connect(f"{url}?role=viewer", open_timeout=10) as fake_viewer:
            with connect(f"{url}?role=controller", open_timeout=10) as primer:
                primer.send(
                    json.dumps({"jsonrpc": "2.0", "id": "prime", "method": "getLayers"})
                )
                primed = json.loads(fake_viewer.recv(timeout=10))
                fake_viewer.send(
                    json.dumps({"jsonrpc": "2.0", "id": primed["id"], "result": []})
                )
                assert json.loads(primer.recv(timeout=10))["id"] == "prime"

            with Viewer(url, timeout_s=10) as controller:
                answers: List[Any] = []

                def answer_one() -> None:
                    frame = json.loads(fake_viewer.recv(timeout=10))
                    answers.append(frame)
                    fake_viewer.send(
                        json.dumps(
                            {"jsonrpc": "2.0", "id": frame["id"], "result": {"ndim": 4}}
                        )
                    )

                responder = threading.Thread(target=answer_one, daemon=True)
                responder.start()
                assert controller.get_dimensions() == {"ndim": 4}
                responder.join(timeout=10)

                # The viewer saw the hub's id, not the controller's...
                assert answers[0]["method"] == "getDimensions"
                # ...and the controller's own numbering started at 1.
                assert isinstance(answers[0]["id"], int)
                assert answers[0]["id"] != 1
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _wait_until(predicate: Any, message: str, timeout_s: float = 10.0) -> None:
    """Poll ``predicate`` rather than sleeping a fixed guess."""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(message)
