"""WebSocket control hub — the transport between a controller and a viewer.

A browser cannot host a WebSocket server, and a kiosk wants several parties on
one scene: a touch panel offering chapter tiles, a big display rendering them,
maybe a second display or an agent. So the hub lives here, in the process that
already serves the viewer, and every party attaches to it as a client.

The hub is a **dumb relay**. It keeps no scene state of its own — the viewer is
the source of truth, and a controller that needs to know where things stand
asks it (``getViewerState``). All the hub does is:

- route a controller's JSON-RPC request to every attached viewer, and the
  reply back to the controller that asked (matching on a hub-private id, so
  two controllers numbering their requests from 1 cannot collide);
- broadcast a viewer's event notifications to every attached controller.

The method vocabulary is deliberately NOT defined here: it is the ``LuxarApp``
embedder API, checked against an allow-list on the **viewer** side
(``core/app/control/method-policy.ts``). The hub never inspects ``method``, so
a new embedder method needs no change to this file. See
``docs/guides/specs/REMOTE_CONTROL_SPEC.md`` §3.

Fan-out semantics, stated rather than inferred: a request goes to *every*
attached viewer and the controller receives the FIRST reply; later duplicates
are dropped. One viewer is the designed case, and a wall of clones showing the
same chapter is the other; addressing one viewer of several is not supported.

Authorization lives in :meth:`ControlHub.serve`, not in the route that calls
it. A token check a caller can forget is a token check that eventually is
forgotten, and its failure mode is an open socket rather than a loud error.
"""

from __future__ import annotations

import hmac
import itertools
import json
from typing import Any, Dict, Optional, Tuple

from arbol import aprint
from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

#: Query value attaching a socket as the party being driven.
ROLE_VIEWER = "viewer"
#: Query value attaching a socket as the party doing the driving.
ROLE_CONTROLLER = "controller"
#: The only roles that may attach. Anything else is refused rather than
#: defaulted: a typo'd ``?role=viewr`` silently attaching as a CONTROLLER would
#: leave a display that never receives a call and reports no error.
ROLES = (ROLE_VIEWER, ROLE_CONTROLLER)

#: RFC 6455 "policy violation" — a bad token or an unknown role. Sent after
#: ``accept()`` on purpose: closing before the handshake completes surfaces at
#: the client as a bare HTTP 403 with no code, so a controller could not tell
#: "no hub here" from "your token is wrong".
CLOSE_POLICY_VIOLATION = 1008

#: Requests in flight per controller before the oldest is evicted. A wedged
#: viewer — or a scene mid-load, where every embedder method throws — never
#: replies, and without a cap one dict entry leaks per tap for the lifetime of
#: the process. Eviction drops a reply the controller was never going to get.
MAX_PENDING_PER_CONTROLLER = 64

#: JSON-RPC reserved codes, plus one of ours for "nothing to drive".
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_NO_VIEWER = -32001


def _error_frame(request_id: Any, code: int, message: str) -> str:
    """A JSON-RPC error response. ``request_id`` may be ``None`` (parse error)."""
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


class ControlHub:
    """Attachment registry and relay for one control endpoint.

    One instance per served app. Every method runs on the server's event loop
    (uvicorn drives the whole app there), so the registries need no locking.
    """

    def __init__(self, *, token: Optional[str] = None) -> None:
        #: Shared secret required as ``?token=`` when set (``--control-token``).
        self._token = token
        self._viewers: Dict[int, WebSocket] = {}
        self._controllers: Dict[int, WebSocket] = {}
        self._next_socket = itertools.count(1)
        self._next_request = itertools.count(1)
        #: hub request id -> (controller key, the id that controller used).
        #: Insertion-ordered, and hub ids are monotonic, so this controller's
        #: first key is its oldest outstanding request — what eviction drops.
        self._pending: Dict[int, Tuple[int, Any]] = {}

    # ── attachment ───────────────────────────────────────────────────────────

    def authorized(self, token: Optional[str]) -> bool:
        """Whether a socket presenting ``token`` may attach.

        Constant-time compare: a hub on a LAN kiosk is reachable by anything on
        that LAN, and ``==`` on a secret leaks its prefix through timing.
        """
        if self._token is None:
            return True
        if token is None:
            return False
        return hmac.compare_digest(token, self._token)

    @property
    def viewer_count(self) -> int:
        """Attached viewers. Exposed for tests and status lines."""
        return len(self._viewers)

    @property
    def controller_count(self) -> int:
        """Attached controllers. Exposed for tests and status lines."""
        return len(self._controllers)

    @property
    def pending_count(self) -> int:
        """Requests awaiting a viewer reply. Exposed so a test can pin the cap."""
        return len(self._pending)

    async def serve(
        self, websocket: WebSocket, role: str, token: Optional[str] = None
    ) -> None:
        """Accept ``websocket`` as ``role`` and relay until it disconnects.

        Refuses an unknown role or a bad token with
        :data:`CLOSE_POLICY_VIOLATION`, before the socket is registered.
        """
        await websocket.accept()
        if role not in ROLES:
            await websocket.close(
                code=CLOSE_POLICY_VIOLATION, reason=f"unknown role {role!r}"
            )
            return
        if not self.authorized(token):
            await websocket.close(code=CLOSE_POLICY_VIOLATION, reason="bad token")
            return

        key = next(self._next_socket)
        registry = self._viewers if role == ROLE_VIEWER else self._controllers
        registry[key] = websocket
        aprint(
            f"🔌 control: {role} attached "
            f"({self.viewer_count} viewer(s), {self.controller_count} controller(s))"
        )
        try:
            while True:
                raw = await websocket.receive_text()
                await self._relay(key, role, raw, websocket)
        except WebSocketDisconnect:
            pass
        finally:
            registry.pop(key, None)
            if role == ROLE_CONTROLLER:
                self._drop_pending_for(key)
            aprint(f"🔌 control: {role} detached")

    def _drop_pending_for(self, controller_key: int) -> None:
        """Forget requests whose asker has gone; their replies are unroutable."""
        for hub_id in [k for k, (c, _) in self._pending.items() if c == controller_key]:
            self._pending.pop(hub_id, None)

    def _remember(self, controller_key: int, original_id: Any) -> int:
        """Record one outstanding request, evicting this controller's oldest."""
        outstanding = [k for k, (c, _) in self._pending.items() if c == controller_key]
        while len(outstanding) >= MAX_PENDING_PER_CONTROLLER:
            self._pending.pop(outstanding.pop(0), None)
        hub_id = next(self._next_request)
        self._pending[hub_id] = (controller_key, original_id)
        return hub_id

    # ── relay ────────────────────────────────────────────────────────────────

    async def _relay(self, key: int, role: str, raw: str, websocket: WebSocket) -> None:
        """Route one received frame according to the sender's role."""
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            if role == ROLE_CONTROLLER:
                await websocket.send_text(
                    _error_frame(None, _PARSE_ERROR, "parse error")
                )
            return
        if not isinstance(frame, dict):
            if role == ROLE_CONTROLLER:
                await websocket.send_text(
                    _error_frame(None, _INVALID_REQUEST, "frame must be an object")
                )
            return
        if role == ROLE_VIEWER:
            await self._from_viewer(frame)
        else:
            await self._from_controller(key, frame, websocket)

    async def _from_controller(
        self, key: int, frame: Dict[str, Any], websocket: WebSocket
    ) -> None:
        """A request or notification headed for the viewers."""
        if "method" not in frame:
            # A controller has nothing to answer; a stray response is noise.
            return
        if not self._viewers:
            if frame.get("id") is not None:
                await websocket.send_text(
                    _error_frame(frame.get("id"), _NO_VIEWER, "no viewer attached")
                )
            return
        if frame.get("id") is None:
            await self._to_viewers(frame)
            return
        hub_id = self._remember(key, frame["id"])
        await self._to_viewers({**frame, "id": hub_id})

    async def _from_viewer(self, frame: Dict[str, Any]) -> None:
        """A reply headed for one controller, or an event headed for all."""
        if "result" in frame or "error" in frame:
            await self._route_reply(frame)
        elif "method" in frame and frame.get("id") is None:
            await self._to_controllers(frame)

    async def _route_reply(self, frame: Dict[str, Any]) -> None:
        """Restore the asking controller's own request id and answer it."""
        pending = self._pending.pop(_as_int(frame.get("id")), None)
        if pending is None:
            return  # a duplicate from a second viewer, or a vanished controller
        controller_key, original_id = pending
        socket = self._controllers.get(controller_key)
        if socket is not None:
            await _send(socket, {**frame, "id": original_id})

    async def _to_viewers(self, frame: Dict[str, Any]) -> None:
        for socket in list(self._viewers.values()):
            await _send(socket, frame)

    async def _to_controllers(self, frame: Dict[str, Any]) -> None:
        for socket in list(self._controllers.values()):
            await _send(socket, frame)


def _as_int(value: Any) -> Any:
    """Hub ids go out as JSON numbers; a client may echo one back as a string."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


async def _send(socket: WebSocket, frame: Dict[str, Any]) -> None:
    """Send one frame, tolerating a peer that closed between select and write."""
    try:
        await socket.send_text(json.dumps(frame))
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError is what starlette raises writing to a closed socket; the
        # disconnect handler will unregister it, so dropping the frame is right.
        pass
