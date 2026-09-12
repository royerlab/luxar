# `luxar.control` — driving a running viewer from Python

A viewer opened with `?control` attaches to the hub that
`luxar serve --control` exposes. This package is the other end of that hub: a
synchronous controller that turns Python calls into JSON-RPC requests.

## Quick Start

Start the viewer and the hub, on one machine or one LAN:

```bash
# Loopback (one machine). Prints the viewer URL, already carrying ?control,
# plus the hub address.
luxar serve scene.luxar.zarr --viewer --control

# Reachable from a tablet on the same network, with a shared secret.
luxar serve scene.luxar.zarr --viewer --control \
    --control-token hunter2 --host 0.0.0.0
```

Open the printed viewer URL on the display, then drive it from Python:

```python
from luxar.control import Viewer

with Viewer("ws://kiosk.local:5173/control", token="hunter2") as viewer:
    story = viewer.dimension_index("story")   # by NAME, not a hard-coded index
    viewer.subscribe("waypoint-arrived")
    viewer.set_dimension_value(story, 3)      # fly to the fourth chapter
    print(viewer.recv_event(timeout_s=10.0))  # ("waypoint-arrived", payload)

    viewer.recenter_camera()                  # re-frame the scene
    print(viewer.get_viewer_state()["src"])   # what it is showing

    # Anything the viewer's embedder API exposes, whether or not it has a
    # named wrapper here:
    viewer.call("setLayer", "/points", {"visible": False})
```

If nothing is listening yet — the display has not booted, or its browser is
closed — the call raises `ControlError` with `no_viewer_attached` set, which is
a thing to wait for rather than an error to abort on:

```python
import time
from luxar.control import ControlError, Viewer

with Viewer("ws://kiosk.local:5173/control") as viewer:
    while True:
        try:
            viewer.get_viewer_state()
            break
        except ControlError as error:
            if not error.no_viewer_attached:
                raise
            time.sleep(1.0)
```

## Key classes

| Name | Purpose |
|---|---|
| `Viewer` | A connected controller. Context manager; `close()` is idempotent. |
| `ControlError` | A refused call, carrying the JSON-RPC `code`. `no_viewer_attached` distinguishes "nothing is listening" from "that failed". |

`Viewer.call(method, *params)` is both the engine room and the escape hatch:
every named method is one line of it, and a method with no wrapper yet is
reachable as `viewer.call("setLayer", "/points", {"visible": False})`.

## Things worth knowing

**The method names are the viewer's, not ours.** `getViewerState`,
`setDimensionValue`, `flyTo` — the wire surface *is* the browser `LuxarApp`
embedder API, so there is no second vocabulary to learn or to keep in sync. The
authoritative list of what a controller may call lives in
`packages/luxar-viewer/src/core/app/control/method-policy.ts`; `dispose` is the
one member deliberately kept off the wire.

**Look dimensions up by name.** `set_dimension_value` takes a positional index
because the viewer's method does, but an index is a property of the scene's
dimension order and moves if the author reorders it. `dimension_index("story")`
is what makes a script survive that.

**A chapter jump is one call.** Moving the story dimension is the whole
operation: the viewer's waypoint driver flies the camera, swaps the
dimension-bound overlays and fires `waypoint-arrived`, which the sound layer's
narration keys on. Nothing here needs to know about any of it.

**Replies and events share one socket.** `recv_event(timeout_s=None)` returns
the next notification as `(name, payload)`, and events that arrive while `call`
waits for its answer are buffered for it. `camera-changed` is throttled
viewer-side to 20 Hz; it fires at frame rate otherwise, and an auto-rotating
kiosk never stops moving.

Two things to know about that stream. It is **not filtered to your own
subscriptions** — the hub fans a viewer's events out to every attached
controller, so a script that subscribed to nothing still receives whatever a
touch panel asked for; match on `name` rather than assuming. And the buffer is
**bounded** at `MAX_BUFFERED_EVENTS`, dropping the oldest past the cap, because
a controller that makes calls but never reads events would otherwise grow for
the life of the process — measured at roughly 91 MB an hour against a 20 Hz
`camera-changed` stream. A reader that has fallen behind wants the current
camera pose anyway, not one from a thousand frames ago.

**Timeouts are generous because replies can be slow.** `flyTo` resolves when
the flight lands and `switchDataset` when the new scene has loaded, so the
default is 30 s rather than something that looks snappy and then lies.

**The hub is open unless `--control-token` is set.** Without one, browser
sockets must be same-host while non-browser clients remain unauthenticated. A
valid token permits split-origin or reverse-proxied setups. It travels as a
query parameter, so it also lands in the viewer's address bar and history —
fine for a kiosk, not a substitute for avoiding an untrusted network.

## See also

- `docs/guides/specs/REMOTE_CONTROL_SPEC.md` — the three-party contract.
- `packages/luxar/src/luxar/cli/control_hub.py` — the relay itself.
