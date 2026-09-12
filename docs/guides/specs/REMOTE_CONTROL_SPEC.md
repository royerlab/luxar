# Remote Control Spec — driving a Luxar viewer from an external program

**Status:** Phase A (viewer-side API), Phase B (the WebSocket hub, the viewer's
control client and the Python controller) and Phase C §4.1 (authored waypoints)
are implemented. Phase C §4.2 (kiosk permissions), the control-panel page, and
Phase D are design, not code.

## 1. Purpose

A Luxar viewer on a large display, driven by a separate program: a touch table
offering pre-baked "story" prompts that fly the camera to a cluster and show its
overlays, or, later, a conversational agent. The viewer must be controllable
from outside (camera, dimensions, rendering, layers, dataset) while staying
lean and general. This document is the contract between the three parties:

- the **viewer** (`@luxar/viewer`, the `LuxarApp` embedder API),
- the **transport** (a WebSocket hub in `luxar serve`),
- the **scene author** (Python `viewer_config`: waypoints, kiosk permissions).

Design rule throughout: **reuse the viewer's existing machinery; add one
primitive per gap.** Nothing here invents a second control surface. The
programmatic API is the same `LuxarApp` surface a host page already gets, the
overrides ride the same path an authored `viewer_config` takes at load, and
story overlays are the existing dimension-aware overlays.

## 2. Phase A — viewer API (implemented)

All methods live on `LuxarApp` (`packages/luxar-viewer/src/core/app.ts`) next
to the pre-existing embedder API (`switchDataset`, `getCameraPose`,
`setCameraPose`, `captureSnapshot` / `restoreSnapshot`, `getDimensions`,
`setDimensionValue`, `setInputEnabled`, `screenshot`, `on`). Every method
throws a clear `... called before init()` error when used too early.

### 2.1 Camera flight

```ts
flyTo(pose: CameraSnapshot, opts?: { durationMs?: number; easing?: 'linear' | 'ease-in-out' })
  : Promise<{ completed: boolean }>
```

- `pose` is the shape `getCameraPose()` returns (position, target, up,
  near/far, `fov` or `zoom`). Flights therefore compose with the existing
  snapshot machinery: capture a pose interactively, store it, fly back to it.
- Interpolation happens in the **orbit parameterisation**: the focus target
  moves linearly, the camera's offset from it is slerped in direction and
  log-interpolated in distance, `up` is slerped. A raw position lerp between two
  orbit poses cuts through the object; this one arcs around it.
- Speed is per flight: `durationMs` (default 1500) and `easing` (default
  smoothstep). `durationMs: 0` equals `setCameraPose()`.
- **`keepOrientation`** keeps the live viewing direction and up; only the
  target, the distance and the projection parameters travel, and the pose's
  own orientation is ignored. This is how a flight composes with the orbit
  turntable: `controls.update()` runs before the flight's frame callback, so
  the auto-rotation keeps advancing the direction and the flight carries it to
  the new target. The spin never pauses and landing does not swing to the
  author's azimuth. The waypoint driver sets it whenever
  `ControlsManager.isAutoRotateActive()`; a controller may pass it explicitly.
- **Control modes.** Orbit is the designed case (every frame hands off via
  `setTarget` + `reinitialize`). Ortho is the same orbit class with rotation
  disabled; a flight interpolates `zoom` geometrically, and an authored
  `camera.zoom` is how an ortho waypoint frames tighter (distance changes
  nothing under an orthographic projection). Fly has no orbit state:
  `setTarget` there means "look at this point now" and the physics integrates
  zero velocity, so a flight moves and aims correctly; `keepOrientation` has no
  target to keep and is effectively a plain flight. A pose never switches the
  projection: perspective/ortho stays whatever the viewer is in.
- **Interruption.** Any pointer / wheel / touch on the canvas or a keydown
  cancels the flight where it is, as does a newer `flyTo()`, a dataset switch,
  or disposal. The promise resolves `{ completed: false }`. The user's own
  volition always wins; a controller wanting the camera to return home does so
  with an idle timer of its own.
- Under dynamic clipping (the default) the flight and `setCameraPose()` never
  write `near` / `far`: the per-frame updater owns them and runs before the
  flight's frame callback. A pose's planes describe the distance it was
  captured at, so interpolating them clipped geometry mid-flight.
- Every frame ends with the same hand-off `setCameraPose()` performs
  (`controls.setTarget` + `reinitialize` + a controls `change` event), so the
  orbit controls never snap back, LOD / depth sort / picking see each pose, and
  the render loop is kept awake for the flight's duration. The final frame IS
  `setCameraPose(pose)`, so a completed flight lands bit-exactly.

Implementation: `core/app/camera/camera-flight.ts` (`CameraFlight`,
`buildFlightPath`, `easeFlight`).

### 2.2 Rendering settings

```ts
getRenderingSettings(): RenderingSettings            // copy
setRenderingSettings(patch: Partial<RenderingSettings>): void
```

`setRenderingSettings` is `RenderingControls.applyOverrides`, the path a scene's
authored `viewer_config` already took at load (`applyZarrDefaults` now calls it
with the extracted zarr keys). Same validation (NaN / out-of-range clamps to
defaults), same side-effects (camera FOV and planes, navigation feel,
auto-rotate / dolly, DPR ceiling, the whole post-processing chain). Anything an
author can bake, a controller can set live. Nothing is persisted to the user's
stored preferences.

### 2.3 Layers

```ts
getLayers(): LayerSummary[]                          // copies, panel order
setLayer(path: string, patch: LayerPatch): void      // throws on unknown path
```

`LayerPatch` fields: `visible`, `opacity`, `gamma`, `displayRange`, `colormap`
(`null` = direct colour), `blendingMode`, `absorption`, `layerOrder` (`null` =
release explicit order). Each field takes the route the Layers panel's own
control takes (state-manager setter, then apply engine), so row, material and
stored state cannot drift apart.

### 2.4 State mirror and events

```ts
getViewerState(): { src, camera, dimensions, rendering, layers }   // all copies
on('camera-changed', (pose: CameraSnapshot) => void)
```

`camera-changed` fires at frame rate while the camera moves (interactive,
`setCameraPose`, flight frames, auto-rotate). A transport MUST throttle it (see
§3.4). Together with the pre-existing `dataset-loaded`, `dimensions-changed`,
`selection`, `element-click` events, a controller can mirror the viewer without
polling.

## 3. Phase B — WebSocket hub (implemented)

### 3.1 Topology

A browser cannot host a WebSocket server, and the touch table, an agent and
several displays may all want to attach. So the **hub** lives in `luxar serve`
(already FastAPI + uvicorn):

```
luxar serve scene.luxar.zarr --viewer --control          # adds ws://host:port/control
viewer:  http://host:port/?src=...&control                 # bare flag; §3.3
python:  luxar.control.Viewer("ws://host:port/control")
```

The hub is mounted on the **viewer** app (`_build_viewer_app`), which is what
makes the socket same-origin with the pages it drives. Its route is registered
*before* the static mount, and that order is load-bearing: Starlette matches
routes in declaration order and a `Mount` at `/` swallows every path below it,
WebSockets included. A test pins the order, and the test is verified to fail
when the order is wrong.

The hub relays: a request from any controller goes to every attached viewer;
a viewer's replies and events go to the controllers. Addressing one viewer of
several is not supported. The hub keeps no state of its own beyond the
attachment list; the viewer is the source of truth (`getViewerState()` on
connect).

### 3.2 Wire format

JSON text frames, JSON-RPC 2.0 shape. The method set is **literally the
`LuxarApp` embedder API**; no second vocabulary.

```javascript
// controller → viewer
// params are POSITIONAL — the TypeScript signature's argument order, verbatim.
{ "jsonrpc": "2.0", "id": 7, "method": "flyTo",
  "params": [ { "position": [0, 0, 40], "target": [0, 0, 0],
                "up": [0, 1, 0], "isOrtho": false, "fov": 45,
                "near": 0.1, "far": 1000 }, { "durationMs": 2000 } ] }
// viewer → controller
{ "jsonrpc": "2.0", "id": 7, "result": { "completed": true } }
{ "jsonrpc": "2.0", "id": 8, "error": { "code": -32603, "message": "unknown layer '/x'" } }
// viewer → controllers (notification, no id) — positional here too
{ "jsonrpc": "2.0", "method": "event", "params": [ "dimensions-changed", { "ndim": 4 } ] }
```

**`params` are positional.** JSON-RPC permits an object too; we do not use it.
A named form needs a table mapping every method's parameter names, maintained
on both sides of the wire, and that table drifts the first time a signature
changes. Positional params follow mechanically from the TypeScript signature.
The `event` notification obeys the same rule:
`{"method": "event", "params": ["dimensions-changed", payload]}`.
The `flyTo` pose has the complete shape returned by `getCameraPose()`; callers
normally copy that object and change the fields they want to animate.

**The method set, stated exactly.** "The embedder API verbatim" is nearly true,
and the exceptions are what rot if left unwritten, so:

> The wire surface is every public `LuxarApp` method, minus
> `CONTROL_EXCLUDED_METHODS` (each carrying a recorded reason), plus
> `subscribe` / `unsubscribe`.

The lists live in `core/app/control/method-policy.ts`, and a lock test scans
`core/app.ts` and fails if a public member is in neither — so the next embedder
method cannot land outside the policy unnoticed. Today's exclusions:

| Excluded | Why |
|---|---|
| `dispose` | A one-frame remote kill switch with no way back, since `init` is not exposed. Every other verb is undone by sending another. |
| `init` | Takes a canvas and a container element. |
| `on` | Takes a listener function; `subscribe` / `unsubscribe` carry the capability by name. |
| `registerContext`, `unregisterContext`, `registerBinding`, `unregisterBinding` | `KeyBinding.handler` is a required function, so the argument cannot be encoded. |
| `pushContext`, `popContext` | Mutate an input-context stack whose depth a controller cannot observe. |
| `components`, `initialized` | Property accessors; `components` returns live engine objects. |

`screenshot` crosses as `{mime, base64}` rather than a `Blob`. Live `Error`
values in event payloads become `{name, message}` — a blind `JSON.stringify`
turns an `Error` into `{}`, which is the worst possible answer to "what went
wrong".

### 3.3 Viewer side

One module, `core/app/control/control-client.ts`: connects when `?control` is
present, dispatches method calls onto the live `LuxarApp` by name against the
policy above, forwards subscribed events, reconnects with backoff. It has no
knowledge of what the methods do. `core/app/control/README.md` is the local
guide.

**`?control` is a bare flag.** The hub rides on the app that served the page, so
the socket address is derived from `location` — which keeps it working behind an
origin-rooted reverse proxy, under `luxar export` and in the native launcher,
none of which know their own address at authoring time. A path-prefixed proxy
uses an explicit same-origin path such as `?control=/exhibit/control`.
`?control=<url>` remains as a split-origin override and is **same-origin only**
unless `?controlAllowCrossOrigin` is also present: otherwise a crafted
`?src=<real>&control=ws://attacker/` link would hand an attacker both the
display and `getViewerState()`. `normalizeControlSocketUrl` also rejects
non-`ws`/`wss` schemes, protocol-relative addresses, credentials in the URL, an
insecure socket from a secure page, a fragment, and any query key but `token`.

**Two plumbing details that are bugs if missed.** The viewer only provisions
picking when a `selection` / `element-*` listener exists *at dataset-load
time*, so the client attaches to every event eagerly at construction and
`subscribe` gates only *forwarding* — a lazily-attached client would leave
those three events permanently dead with no error. And `camera-changed` is
throttled to 20 Hz, because it fires at frame rate and an auto-rotating kiosk
never stops moving.

### 3.4 Python side

`luxar.control.Viewer` (`packages/luxar/src/luxar/control/`) is a synchronous
controller: `call(method, *params)` is the engine room and the escape hatch,
with snake_case wrappers for the handful most used —
`get_viewer_state`, `get_dimensions`, `set_dimension_value`, `get_camera_pose`,
`fly_to`, `recenter_camera`, `subscribe` / `unsubscribe`, plus `recv_event` for
reading notifications. A refusal arrives as `ControlError` carrying
the JSON-RPC code, and `ControlError.no_viewer_attached`
distinguishes "nothing is listening yet" from "that failed" — a display that has
not booted is something a kiosk script waits for, not an error to abort on.

`dimension_index(name)` resolves a dimension by name, because
`setDimensionValue` takes a positional index and an index moves when the author
reorders `Dimensions([...])`.

An async variant (for the agent) can wrap the same wire format; nothing here
assumes the synchronous one is the only client.

### 3.5 Security

**Cross-Site WebSocket Hijacking is checked, and it has to be.** A WebSocket
handshake is not subject to the same-origin policy and carries no CORS
preflight, so *any* page a visitor happens to open could otherwise connect to
`ws://localhost:<port>/control`, drive the display, and read the dataset URL
back out of `getViewerState()` — on a loopback-only hub, with no LAN exposure at
all. The hub therefore compares the handshake's `Origin` against the `Host` it
arrived on (`ControlHub.origin_allowed`) and closes a mismatch with 1008. That
blocks the direct cross-origin case; it is not a DNS-rebinding defence, so an
untrusted network still requires `--control-token`.

Two deliberate allowances, both stated rather than implied. A handshake with
**no `Origin` at all is allowed**, because non-browser clients send none —
`luxar.control.Viewer` included — and refusing them would break the Python
controller outright; this check defends against a *browser* being used as the
attacker's proxy, which is the actual attack. A valid configured token also
permits an explicit split-origin viewer (`?controlAllowCrossOrigin`) or a
reverse proxy that rewrites `Host`; an embedding application may instead pass
an explicit `allowed_origins` list to `ControlHub`.

Same-host LAN kiosk by default: the hub binds the address `luxar serve` binds,
which is loopback unless `--host 0.0.0.0` is given. `--control-token <t>`
requires `?token=` on the WebSocket URL, compared with `hmac.compare_digest`
over UTF-8 **bytes** (that function raises `TypeError` on a non-ASCII `str`, so
comparing strings would turn a wrong password into a crashed handler);
viewers and controllers present the same token, and a wrong one is closed with
RFC 6455 code 1008 (as is an unrecognised `?role=`). No other auth is planned.

Two things to say plainly rather than imply. **A token in a query string** lands
in browser history and in the address bar of whatever tablet is on the plinth;
it is a LAN convenience, not a secret. **The hub is open by default to
same-host browser pages and non-browser clients**, so an unauthenticated peer
that can reach the socket can drive the display — including `switchDataset`,
which is on the wire deliberately. The dispatcher runs its argument through
the viewer's own `normalizeDataSourceUrl` (the app's method validates nothing),
so a `file:` or `javascript:` URL is refused, but a reachable hub on an
untrusted network is still a display someone else can repoint. Use
`--control-token`, or do not bind past loopback.

## 4. Phase C — authored waypoints (implemented) and kiosk mode (design)

Both blocks live in `viewer_config` (Python `luxar.core.viewer_config`, viewer
`types/zarr.ts` + `config/zarr-bridge`), because the scene author decides what
the display may do and where the stories are.

### 4.1 Story dimension and waypoints (implemented)

A hidden **story dimension** (discrete, one integer per story) already gave the
author everything except the camera: per-story colours are authored data,
per-story overlays use the existing dimension-aware `visible_range`. The one
missing binding was dimension position → camera pose, and it deliberately
reuses the overlay vocabulary:

```python
from luxar import CameraConfig, Waypoint

vc.waypoints = [
    Waypoint(when={"story": 0}, camera=CameraConfig(position=(0, 0, 40))),
    Waypoint(when={"story": 1, "time": (10, 20)},
             camera=CameraConfig(target_node="cluster_7", position=(12, 3, 8)),
             duration_ms=2500, easing="ease-in-out", rendering={"exposure": 0.5}),
    Waypoint(when={"story": 1}, camera=CameraConfig(target_node="cluster_7")),
]
```

- `when` is the overlay rule (`waypointMatches` mirrors `isOverlayVisible`):
  every named dimension must match, exact = within ±0.5 of the current step,
  `(min, max)` inclusive, dimensions the scene lacks are skipped. **First match
  in list order wins**, so a clause on two dimensions goes before a clause on
  one.
- `camera` is the ordinary camera block; fields left out keep the LIVE value at
  flight time (`resolveWaypointPose` starts from `getCameraPose()`), so a
  waypoint may re-aim (`target_node`) without moving. `target_node` beats
  `target`; an unknown node warns and falls back.
- `duration_ms` / `easing` are the `flyTo` options; `0` snaps. `rendering`
  carries snake_case `ViewerConfig` keys (validated in Python against the
  rendering field list) and rides `RenderingControls.applyOverrides` via
  `extractRenderingOverrides` — the same path as authored defaults.
- `reveal` decides WHEN the story's dimension-bound overlays appear.
  `"immediate"` (default) shows them as the dimension changes, while the
  camera is still flying. `"on_arrival"` holds overlays that would newly
  appear until the flight resolves, so the caption and the turntable show up
  when the camera has arrived — the same `waypoint-arrived` event the sound
  layer's `on_arrive` narration keys on (`SOUND_SPEC.md` §4.3), so text and
  voice land together. The rule is a gate, not a timer: a snap, a
  camera-less waypoint and a flight the visitor cancels all count as arrival;
  a flight a newer waypoint supersedes never reveals (scrubbing through five
  stories shows only the one you stop on); departing overlays hide at once
  either way; overlays without a `visible_range` are untouched. Viewer side:
  `WaypointDriver.inTransit` + `OverlayManager.setTransitGate` — the app
  re-runs the visibility pass after the driver evaluates and on arrival, so
  listener order between the two cannot flash a caption.

Viewer rule (`core/app/camera/waypoint-driver.ts`, wired in
`LuxarApp.applyViewerConfigState` after `dimensions.current_step` is applied):
act on a **change of matched waypoint**, never on every slider tick. At load the
matched waypoint is applied as a snap (the opening framing, ahead of the plain
`camera` block). Afterwards a change of match flies (`duration_ms: 0` is a
zero-length flight); a move inside the same waypoint's ranges does nothing;
leaving every waypoint leaves the camera where it is. **While auto-rotate is
active the flight is `keepOrientation`**: the turntable keeps spinning, the
story step only moves the point it spins around (and how far away), and the
authored orientation is ignored. In ortho mode author `camera.zoom` to frame
tighter; position and distance are not what frames an orthographic view. The touch table therefore only ever calls `setDimensionValue`, and the
keyboard (`[` / `]`) drives the same stories with no controller at all. A
controller can still `flyTo` anywhere; waypoints are defaults, not a cage.

Not carried per waypoint (deliberately): layer patches — a story that wants a
different layer look authors it as data or asks the controller to `setLayer`.

**Reference scene:** `luxar demo run esm3_protein_stories` — the ESM C Swiss-Prot
UMAP with a hidden `story` dimension (Overview + ten protein-family clusters),
a dimmed `extend_to_all` backdrop, one highlight layer per story in a higher
`layer_order` band (a highlight shares every position with its backdrop twin, so
in one band the two z-fight and the backdrop hides it), a fact panel per story as
a dimension-aware HTML overlay, one waypoint per story, and auto-rotate on. It is
the scene to open when checking the flight, the turntable rule, or a controller.

### 4.2 Kiosk permissions (design)

Extend the existing `ui` block:

```python
scene.viewer_config.ui.kiosk = KioskConfig(
    enabled=True,
    allow_pointer=False,      # ignore pointer/wheel/touch on the canvas
    allow_keyboard=False,     # ignore the keyboard entirely
    show_panels=False,        # no rail, no panels, no dataset browser
    watchdog_reload=True,     # reload on WebGL context loss
)
```

`?kiosk` on the URL is a hard override for a display whose store predates the
block. Kiosk mode reuses `setInputEnabled(false)` and the existing panel
visibility flags; only the watchdog is new.

**Launching the display.** The big screen runs Chrome in kiosk mode; these flags
are the operator's side of the contract (the scene-side block above cannot set
them):

```bash
google-chrome --kiosk --noerrdialogs --disable-infobars \
  --disable-features=TranslateUI \
  --autoplay-policy=no-user-gesture-required \
  --disable-session-crashed-bubble \
  "http://<host>:5173/?src=http://<host>:8005&kiosk"
```

`--autoplay-policy=no-user-gesture-required` matters twice: it lets the
`overlay_video` clips start without a tap, and it lets the sound layer
(`SOUND_SPEC.md` §4.4) start its `AudioContext` on load instead of showing its
"Tap to enable sound" gate. Muted video autoplays under the default policy too,
so a display without the flag still shows the turntables; only sound needs it.

## 5. Phase D — conversational agent (design)

Out of Luxar's scope except for the tool surface. The agent (OpenAI Realtime
for voice, push-to-talk from the touch table) gets tools that are thin wrappers
over `luxar.control.Viewer`: `fly_to_cluster(name)`, `set_story(value)`,
`color_by(attribute)`, `get_view_state()`, plus knowledge tools
(`uniprot_lookup`, `web_search`). The semantic layer (cluster name → centroid,
radius, description, marker genes, UniProt ids) is a precomputed JSON sidecar
produced with the scene, not a viewer feature.

## 6. Non-goals

- No camera-facing 3D billboards or runtime label injection: authored overlays
  and per-story colours cover it.
- No second control vocabulary: the wire protocol is the embedder API.
- No multi-viewer synchronisation semantics beyond fan-out.
