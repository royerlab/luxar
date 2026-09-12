"""A synchronous controller for a running Luxar viewer.

One class, :class:`Viewer`, that connects to the hub as a *controller* and
turns method calls into JSON-RPC requests. Deliberately thin: it owns the
socket, the request ids and the waiting, and knows nothing about what any
method means. That is what keeps the wire surface and the viewer's embedder API
one thing rather than two — a new embedder method is reachable through
:meth:`Viewer.call` the day it lands, with no change here.

Synchronous on purpose. The callers this exists for are scripts, notebooks and
a REPL, where ``viewer.set_dimension_value(3, 1)`` reading like a function call
matters more than concurrency. An async variant can wrap the same wire format
when something needs one.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import deque
from types import TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type

#: How long to wait for a viewer's reply before giving up, in seconds.
#:
#: Generous because the answer may be behind real work: ``flyTo`` resolves when
#: the flight lands, and ``switchDataset`` when the new scene has loaded.
DEFAULT_TIMEOUT_S = 30.0

#: How many unread events a controller buffers before dropping the oldest.
#:
#: There has to be a bound. The hub fans a viewer's events out to EVERY
#: attached controller regardless of what each one subscribed to, so a script
#: that subscribed to nothing still receives whatever a touch panel asked for
#: — and `call` drains the socket while it waits for its reply, so those
#: events pile up during exactly the long waits (`flyTo`, `switchDataset`)
#: when a 20 Hz `camera-changed` stream is landing. Unbounded, that measured
#: ~91 MB after an hour.
#:
#: It also restores a bound that already existed: ``websockets.connect``
#: defaults to ``max_queue=16`` and stops reading past its high-water mark, so
#: the socket itself applied backpressure until the frames were drained into a
#: Python-side buffer. 1024 is generous for any reader that polls at all,
#: while keeping the worst case at kilobytes rather than tens of megabytes.
MAX_BUFFERED_EVENTS = 1024

#: JSON-RPC code the hub answers with when no viewer is attached.
NO_VIEWER_CODE = -32001

# Match uvicorn's control-hub frame ceiling while allowing display-sized PNGs.
_MAX_FRAME_SIZE_BYTES = 16 * 1024 * 1024


class ControlError(RuntimeError):
    """A viewer (or the hub) refused a call.

    ``code`` is the JSON-RPC error code — ``-32601`` for a method the viewer's
    policy does not expose, ``-32001`` for "no viewer attached", ``-32603`` for
    a method that threw inside the viewer.
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

    @property
    def no_viewer_attached(self) -> bool:
        """Whether this means "nothing is listening" rather than "that failed".

        Worth distinguishing in a kiosk script: a display that has not booted
        yet is something to wait for, not an error to abort on.
        """
        return self.code == NO_VIEWER_CODE


class Viewer:
    """A connected controller for one hub.

    Use as a context manager, or call :meth:`close` yourself::

        with Viewer("ws://kiosk.local:5173/control") as viewer:
            viewer.recenter_camera()

    Args:
        url: The hub's WebSocket URL, as ``luxar serve --control`` prints it.
        token: Shared secret, when the hub was started with
            ``--control-token``.
        timeout_s: Per-call reply timeout. See :data:`DEFAULT_TIMEOUT_S`.
    """

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._url = _with_query(url, role="controller", token=token)
        self._timeout_s = timeout_s
        self._ids = itertools.count(1)
        self._events: deque[Tuple[str, Any]] = deque(maxlen=MAX_BUFFERED_EVENTS)
        self._socket = _connect(self._url, timeout_s)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the socket. Idempotent."""
        try:
            self._socket.close()
        except Exception:  # noqa: BLE001 - a closed socket is the desired state
            pass

    def __enter__(self) -> "Viewer":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    # ── the wire ─────────────────────────────────────────────────────────────

    def call(self, method: str, *params: Any) -> Any:
        """Invoke an embedder method and return its result.

        The escape hatch as much as the engine room: every named method below
        is one line of this, and a method with no wrapper yet is reachable as
        ``viewer.call("setLayer", "/points", {"visible": False})``.

        Raises:
            ControlError: the viewer or the hub refused the call.
            TimeoutError: no reply arrived within the configured timeout.
            websockets.exceptions.ConnectionClosed: the hub connection closed.
        """
        request_id = next(self._ids)
        self._socket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": list(params),
                }
            )
        )
        return self._await_reply(request_id)

    def notify(self, method: str, *params: Any) -> None:
        """Invoke a method without waiting for a reply.

        Right for a fire-and-forget nudge (``notify("recenterCamera")``) where
        the round trip is the only cost. An error is not reported back — the
        viewer logs it instead — so prefer :meth:`call` when it matters.
        """
        self._socket.send(
            json.dumps({"jsonrpc": "2.0", "method": method, "params": list(params)})
        )

    def _await_reply(self, request_id: int) -> Any:
        """Read frames until the answer to ``request_id`` arrives.

        Events and replies share one socket, so event notifications are buffered
        for :meth:`recv_event`; unrelated replies are skipped.
        """
        deadline = time.monotonic() + self._timeout_s
        while True:
            remaining_s = max(0.0, deadline - time.monotonic())
            frame = json.loads(self._socket.recv(timeout=remaining_s))
            event = _event_from_frame(frame)
            if event is not None:
                self._events.append(event)
                continue
            if not isinstance(frame, dict) or frame.get("id") != request_id:
                continue
            if "error" in frame:
                raise _control_error(frame["error"])
            return frame.get("result")

    def recv_event(
        self, timeout_s: Optional[float] = None
    ) -> Optional[Tuple[str, Any]]:
        """Return the next event as ``(name, payload)``.

        Not only the events *this* controller subscribed to: the hub fans a
        viewer's events out to every attached controller, so a script that
        subscribed to nothing still sees whatever another controller — a touch
        panel, say — asked for. Match on ``name`` rather than assuming.

        Events received while :meth:`call` waits for its reply are buffered and
        returned first. That buffer holds at most
        :data:`MAX_BUFFERED_EVENTS`; past the cap the OLDEST are dropped, so a
        long-lived controller that never reads events cannot grow without
        limit. ``timeout_s=None`` waits indefinitely; a finite timeout returns
        ``None`` when no event arrives before its deadline.

        Raises:
            ConnectionClosed: If the socket closed while waiting. This reads
                the same socket as :meth:`call` and fails the same way.
        """
        if self._events:
            return self._events.popleft()

        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            remaining_s = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                frame = json.loads(self._socket.recv(timeout=remaining_s))
            except TimeoutError:
                return None
            event = _event_from_frame(frame)
            if event is not None:
                return event

    # ── named methods ────────────────────────────────────────────────────────
    # A thin layer over `call`, for the handful a controller reaches for most.
    # Snake_case names, camelCase on the wire.

    def get_viewer_state(self) -> Dict[str, Any]:
        """Everything at once: src, camera, dimensions, rendering, layers, audio."""
        result: Dict[str, Any] = self.call("getViewerState")
        return result

    def get_dimensions(self) -> Dict[str, Any]:
        """Dimension metadata, including ``categories`` for a labelled axis."""
        result: Dict[str, Any] = self.call("getDimensions")
        return result

    def set_dimension_value(self, index: int, value: float) -> None:
        """Move one dimension — how a chapter jump is expressed.

        The viewer's waypoint driver does the rest: the camera flies, the
        dimension-bound overlays swap, and ``waypoint-arrived`` fires.
        """
        self.call("setDimensionValue", index, value)

    def dimension_index(self, name: str) -> int:
        """The positional index of a dimension, by name.

        ``setDimensionValue`` takes an index, but an index is a property of the
        scene's dimension order and changes if the author reorders it. Looking
        it up by name is what makes a script survive that.

        Raises:
            KeyError: the scene has no dimension with that name.
        """
        metadata: List[Dict[str, Any]] = self.get_dimensions()["metadata"]
        for index, dimension in enumerate(metadata):
            if dimension.get("name") == name:
                return index
        available = [str(d.get("name")) for d in metadata]
        raise KeyError(f"no dimension named {name!r}; scene has {available}")

    def get_camera_pose(self) -> Dict[str, Any]:
        """The current camera pose, in the shape ``fly_to`` accepts."""
        result: Dict[str, Any] = self.call("getCameraPose")
        return result

    def fly_to(self, pose: Dict[str, Any], **opts: Any) -> Dict[str, Any]:
        """Fly the camera to ``pose``; resolves when the flight ends.

        ``opts`` are the viewer's own flight options (``durationMs``,
        ``easing``, ``keepOrientation``).
        """
        result: Dict[str, Any] = self.call("flyTo", pose, opts or None)
        return result

    def recenter_camera(self) -> None:
        """Re-frame the scene, as the keyboard's recenter action does."""
        self.call("recenterCamera")

    def subscribe(self, event: str) -> None:
        """Start receiving ``event`` notifications on this socket."""
        self.call("subscribe", event)

    def unsubscribe(self, event: str) -> None:
        """Stop receiving ``event`` notifications."""
        self.call("unsubscribe", event)


def _with_query(url: str, *, role: str, token: Optional[str]) -> str:
    """Add the two query parameters the hub reads."""
    from urllib.parse import urlencode, urlparse, urlunparse

    parts = urlparse(url)
    query = {"role": role}
    if token:
        query["token"] = token
    return urlunparse(parts._replace(query=urlencode(query)))


def _control_error(error: Any) -> ControlError:
    """Turn even a malformed JSON-RPC error payload into ``ControlError``."""
    if not isinstance(error, dict):
        return ControlError(-32603, f"malformed error response: {error!r}")

    raw_code = error.get("code", 0)
    if type(raw_code) is int:
        code = raw_code
    elif isinstance(raw_code, str):
        try:
            code = int(raw_code)
        except ValueError:
            return ControlError(-32603, f"malformed error response: {error!r}")
    else:
        return ControlError(-32603, f"malformed error response: {error!r}")
    return ControlError(code, str(error.get("message", "refused")))


def _event_from_frame(frame: Any) -> Optional[Tuple[str, Any]]:
    """Decode the viewer client's positional ``event`` notification."""
    if not isinstance(frame, dict) or frame.get("method") != "event":
        return None
    params = frame.get("params")
    if (
        frame.get("id") is not None
        or not isinstance(params, list)
        or len(params) != 2
        or not isinstance(params[0], str)
    ):
        return None
    return params[0], params[1]


def _connect(url: str, timeout_s: float) -> Any:
    """Open the socket, with an import error a reader can act on."""
    try:
        from websockets.sync.client import connect
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            "luxar.control needs the `websockets` package (declared in "
            "pyproject.toml); install it with `pip install websockets`."
        ) from error
    # Screenshot replies routinely exceed websockets' 1 MiB default frame cap.
    connection = connect(url, open_timeout=timeout_s, max_size=_MAX_FRAME_SIZE_BYTES)
    # websockets >= 14 warns on the first send unless the connection has been
    # entered as a context manager. A Viewer deliberately outlives any single
    # `with` block (that is the point of a REPL handle), so it enters the
    # connection here and closes it in `Viewer.close()`. Leaving the warning in
    # place would fail any suite that runs under `-W error`.
    return connection.__enter__()
