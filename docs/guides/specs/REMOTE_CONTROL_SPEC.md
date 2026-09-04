# Remote Control Spec — driving a Luxar viewer from an external program

**Status:** Phase A implemented (viewer-side API). Phases B–D are design, not code.

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
- **Interruption.** Any pointer / wheel / touch on the canvas or a keydown
  cancels the flight where it is, as does a newer `flyTo()`, a dataset switch,
  or disposal. The promise resolves `{ completed: false }`. The user's own
  volition always wins; a controller wanting the camera to return home does so
  with an idle timer of its own.
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

## 3. Phase B — WebSocket hub (design)

### 3.1 Topology

A browser cannot host a WebSocket server, and the touch table, an agent and
several displays may all want to attach. So the **hub** lives in `luxar serve`
(already FastAPI + uvicorn):

```
luxar serve scene.luxar.zarr --viewer --control          # adds ws://host:port/control
viewer:  http://host:port/?src=...&control=ws://host:port/control&kiosk
python:  luxar.control.Viewer("ws://host:port/control")
```

The hub relays: a request from any controller goes to every attached viewer
(or to one, when addressed by `viewer` id); a viewer's replies and events go to
the controllers. The hub keeps no state of its own beyond the attachment list;
the viewer is the source of truth (`getViewerState()` on connect).

### 3.2 Wire format

JSON text frames, JSON-RPC 2.0 shape. The method set is **literally the
`LuxarApp` embedder API**; no second vocabulary.

```jsonc
// controller → viewer
{ "jsonrpc": "2.0", "id": 7, "method": "flyTo",
  "params": { "pose": { ... }, "opts": { "durationMs": 2000 } } }
// viewer → controller
{ "jsonrpc": "2.0", "id": 7, "result": { "completed": true } }
{ "jsonrpc": "2.0", "id": 8, "error": { "code": -32602, "message": "unknown layer '/x'" } }
// viewer → controllers (notification, no id)
{ "jsonrpc": "2.0", "method": "event", "params": { "name": "dimensions-changed", "payload": { ... } } }
```

Methods: `getViewerState`, `getCameraPose`, `setCameraPose`, `flyTo`,
`getDimensions`, `setDimensionValue`, `getRenderingSettings`,
`setRenderingSettings`, `getLayers`, `setLayer`, `switchDataset`, `screenshot`
(returns a data URL), `setInputEnabled`, `subscribe` / `unsubscribe` (event
names). Blobs and functions never cross the wire; everything else is the
TypeScript signature verbatim.

### 3.3 Viewer side

One module, `core/app/control/control-client.ts`: connects when `?control=` is
present, dispatches method calls onto the live `LuxarApp` by name against an
allow-list, forwards subscribed events, reconnects with backoff. It has no
knowledge of what the methods do.

### 3.4 Python side

`luxar.control.Viewer` mirrors the method list with snake_case names and
typed dataclasses for `CameraSnapshot` / `LayerPatch`. Synchronous facade over
`websockets` (an async variant for the agent). Events arrive on a callback or
an iterator. `camera-changed` is throttled by the viewer's client to
≤ 20 Hz before it is sent.

### 3.5 Security

Same-host LAN kiosk by default: the hub binds the address `luxar serve` binds.
`--control-token <t>` requires `?token=` on the WebSocket URL for controllers;
viewers attach with the same token. No other auth is planned.

## 4. Phase C — authored waypoints and kiosk mode (design)

Both blocks live in `viewer_config` (Python `luxar.core.viewer_config`, viewer
`config/zarr-bridge`), because the scene author decides what the display may
do and where the stories are.

### 4.1 Story dimension and waypoints

A hidden **story dimension** (discrete, one integer per story) already gives
the author everything except the camera: per-story colours are authored data,
per-story overlays use the existing dimension-aware overlay visibility. The
one missing binding is dimension value → camera pose:

```python
scene.viewer_config.waypoints = [
    Waypoint(dimension=3, value=0, camera=CameraConfig(position=..., target=..., up=...),
             duration_ms=2000, rendering={"exposure": 0.5}),
    Waypoint(dimension=3, value=1, camera=...),
]
```

Viewer rule: when the story dimension changes to a value that has a waypoint,
`flyTo(waypoint.camera, {durationMs})` and apply `rendering` / `layers`
patches. The touch table therefore only ever calls `setDimensionValue`, and the
keyboard (`[` / `]`) drives the same stories with no controller at all. A
controller can still `flyTo` anywhere; waypoints are defaults, not a cage.

### 4.2 Kiosk permissions

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
