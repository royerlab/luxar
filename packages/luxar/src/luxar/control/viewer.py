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
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

#: How long to wait for a viewer's reply before giving up, in seconds.
#:
#: Generous because the answer may be behind real work: ``flyTo`` resolves when
#: the flight lands, and ``switchDataset`` when the new scene has loaded.
DEFAULT_TIMEOUT_S = 30.0

#: JSON-RPC code the hub answers with when no viewer is attached.
NO_VIEWER_CODE = -32001


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

        Events and replies share one socket, so anything else that turns up is
        skipped. A controller that wants events should read them itself rather
        than rely on this method's leftovers.
        """
        while True:
            frame = json.loads(self._socket.recv(timeout=self._timeout_s))
            if not isinstance(frame, dict) or frame.get("id") != request_id:
                continue
            if "error" in frame:
                error = frame["error"]
                raise ControlError(
                    int(error.get("code", 0)), str(error.get("message", "refused"))
                )
            return frame.get("result")

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


def _connect(url: str, timeout_s: float) -> Any:
    """Open the socket, with an import error a reader can act on."""
    try:
        from websockets.sync.client import connect
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            "luxar.control needs the `websockets` package (declared in "
            "pyproject.toml); install it with `pip install websockets`."
        ) from error
    connection = connect(url, open_timeout=timeout_s)
    # websockets >= 14 warns on the first send unless the connection has been
    # entered as a context manager. A Viewer deliberately outlives any single
    # `with` block (that is the point of a REPL handle), so it enters the
    # connection here and closes it in `Viewer.close()`. Leaving the warning in
    # place would fail any suite that runs under `-W error`.
    return connection.__enter__()
