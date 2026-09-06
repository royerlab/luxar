# core/app/embedder — Programmatic embedder API

**Public Surface**: Events, value types, and headless screenshot utilities for embedding Luxar programmatically in other applications.

Unlike the rest of `app/` (private orchestrator support), this subpackage is re-exported from the package barrel (`src/index.ts`) — embedders import `LuxarEmbedderEventMap`, `EmbedderDimensions`, `ScreenshotOptions`, etc. from `'@luxar/viewer'` directly.

## Modules

| File            | Status       | Description                                                                                                                                                                          |
| --------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `events.ts`     | **Public**   | Value and event type catalog: `LuxarEmbedderEventMap`, `EmbedderDimensions`, `ScreenshotOptions`, `SelectionPayload`. Re-exported from `src/index.ts`.                               |
| `screenshot.ts` | **Internal** | `captureScreenshot(sceneManager, overlayManager, opts)` — headless frame → Blob encoding. Composes the Recording panel's pure screenshot helpers without state-machine dependencies. |

## Events

`LuxarApp.on(event, listener)` subscribes to these app-scoped events (emitted on a per-app `EventEmitter`, not the cross-layer singleton):

| Event                  | Payload                         | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `'dataset-loaded'`     | `{ src: string }`               | Fires when a dataset finishes loading (initial `init()`, browser selection, or `switchDataset()`).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `'dataset-error'`      | `{ src: string; error: Error }` | Fires when a dataset fails to load.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `'dataset-fault'`      | `DatasetFaultPayload`           | Fires when the loaded dataset becomes unable to update, while preserving its last complete frame. An explicit retry clears the current fault; if the archive fails again, the event fires again. Subscribe before `init()` / `switchDataset()`, or call `getDatasetFault()` to inspect a fault that occurred before subscription.                                                                                                                                                                                                                                                                                                                                                                                         |
| `'dimensions-changed'` | `EmbedderDimensions`            | Fires on any slice-position change (slider, keyboard, or `setDimensionValue()`). Includes current step, displayed dims, metadata, and navigable ranges (all **cloned** copies).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `'camera-changed'`     | `CameraSnapshot`                | Fires whenever the camera pose changes — interactive input, `setCameraPose()`, each `flyTo()` frame, auto-rotate — at frame rate while the camera moves. A transport relaying it over a network must throttle.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `'selection'`          | `SelectionPayload \| null`      | Fires when the hover-pick changes (`nodeName` + `elementIndex` + `hitNodeName` under the cursor, or `null` when cleared). Subscribe **before** `init()` to provision picking for all datasets. What `elementIndex` counts is per-node — see `SelectionPayload`: a node declaring `has_labels` / `has_image_labels` / `has_keys` reports an on-disk index — flat, or, for **points**, an additive-LOD ladder, whose per-level maps are composed into the parent's union CSR space (#1439); such a **lines** node reports the picked segment's start-vertex row. A node declaring none of those channels — or an additive-LOD **lines** ladder, which publishes no map across its levels — reports the visible-buffer slot. |
| `'sound-started'`      | `{ name: string }`              | A sound node started playing (after its `delay_ms`) — a `continuous` bed on its slab's rising edge, a `once` narration, or a `playSound()` call. `name` is the node name.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `'sound-ended'`        | `{ name: string }`              | A sound node stopped: its `once` clip ran out, it faded out on the slab's falling edge, or `stopSound()` was called.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `'waypoint-departed'`  | `{ index: number }`             | The matched story waypoint (`viewer_config.waypoints` order) changed away from `index`. Synchronous with the dimension change.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `'waypoint-arrived'`   | `{ index, completed }`          | The flight to waypoint `index` resolved (or it was snapped to at load / has no camera block). `completed: false` = the visitor cancelled the flight; it still counts as an arrival from wherever the camera stopped. A flight a NEWER waypoint superseded never arrives.                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

## Usage

```typescript
import { LuxarApp } from '@luxar/viewer';
import type { EmbedderDimensions } from '@luxar/viewer';

const app = new LuxarApp();
await app.init({ canvas, container });

// Subscribe to events (returns an unsubscribe callback)
const unsubscribe = app.on('dimensions-changed', (dims: EmbedderDimensions) => {
  console.log('Slice position:', dims.currentStep);
  console.log('Displayed dims:', dims.displayed);
  console.log('Navigable ranges:', dims.ranges);
});

// Get current dimension state (all arrays are cloned — safe to read)
const dims = app.getDimensions();

// Set a slice position programmatically
app.setDimensionValue(3, 10); // Set dimension 3 (e.g., time) to step 10
await app.awaitDimensionUpdate(); // Wait for the requested slice to load

// Capture a screenshot
const blob = await app.screenshot({ format: 'webp', quality: 0.92 });
const url = URL.createObjectURL(blob);
const link = document.createElement('a');
link.href = url;
link.download = 'screenshot.webp';
link.click();

// Clean up
unsubscribe();
```

## Lifecycle Invariants

- Events are emitted **synchronously** when the state change completes (not deferred).
- All dimension arrays (`currentStep`, `displayed`, `ranges`, `metadata`) are **cloned** before the event fires — an embedder may read them freely without mutating viewer internals.
- The `selection` event fires on **hover** changes (the element under the cursor), not click-to-select.
- Picking is provisioned when a `selection` listener exists at dataset load time — subscribe **before** `init()` / `switchDataset()` to enable picking on all datasets.
- `SelectionPayload.nodeName` is the user-facing layer (the outermost `kind=partition` wrapper when the hit sits under one), so it is **not** the node `elementIndex` is local to. Resolve the element against `hitNodeName`, which is the leaf that was hit and equals `nodeName` when the node is not partitioned.

## Remote control (camera flight, settings, layers, state)

The methods a remote controller (touch table, WebSocket bridge, agent) needs,
all on `LuxarApp` — see `docs/guides/specs/REMOTE_CONTROL_SPEC.md` for the
phased design they belong to, and `docs/guides/specs/SOUND_SPEC.md` §4.5 for the audio methods. Every one throws `... called before init()` when
used too early.

| Method                                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `flyTo(pose, { durationMs?, easing?, keepOrientation? })`  | Smooth transition to a `CameraSnapshot` (the shape `getCameraPose()` returns), interpolated in the orbit parameterisation (target lerp, direction slerp, log distance, up slerp) so it arcs around the scene. `keepOrientation` keeps the live direction and up (only target, distance and projection travel) — how a flight composes with the auto-rotate turntable. Any canvas pointer/wheel/touch or keydown cancels it where it is; so does a newer `flyTo()` or a dataset switch. Resolves `{ completed }`. |
| `getRenderingSettings()`                                   | Copy of the live `RenderingSettings`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `setRenderingSettings(patch)`                              | Partial override through `RenderingControls.applyOverrides` — the same path an authored `viewer_config` takes at load (validation, camera FOV/planes, navigation, post-processing). Not persisted to user preferences.                                                                                                                                                                                                                                                                                           |
| `getLayers()`                                              | `LayerSummary[]` in Layers-panel order (copies).                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `setLayer(path, patch)`                                    | `LayerPatch` (visible, opacity, gamma, displayRange, colormap, blendingMode, absorption, layerOrder; `gain` on a `sound` layer) applied through the panel's own state-manager + apply-engine route. On a `sound` layer `visible: false` mutes the node. Throws on an unknown path.                                                                                                                                                                                                                               |
| `getViewerState()`                                         | `{ src, camera, dimensions, rendering, layers, audio }` — one-call mirror for a controller; pair with the events above to avoid polling.                                                                                                                                                                                                                                                                                                                                                                         |
| `getAudioState()`                                          | `AudioState` — `{ state, muted, masterGain, panningModel, buses, playing, hasSoundNodes }`. `state` is the `AudioContext` state (`unavailable` before a scene with sound nodes loads; `suspended` means the display still needs its tap, see the autoplay gate in `src/audio/README.md`).                                                                                                                                                                                                                        |
| `setAudio({ masterGain?, muted?, buses?, panningModel? })` | Live mixer patch: master gain (persisted), rail mute (persisted; a muted scene never shows the autoplay gate), per-bus gains (`ambient` / `voice` / `effects`), panning model (`equalpower` for room speakers, `HRTF` for headphones).                                                                                                                                                                                                                                                                           |
| `playSound(name)` / `stopSound(name)`                      | Start or stop one sound node by name (the last path segment, or the full path) regardless of its slab audibility; `stopSound` uses the node's fade-out. Both return `false` for an unknown name.                                                                                                                                                                                                                                                                                                                 |

```typescript
const home = app.getCameraPose(); // capture interactively, store it
// ... later, from a touch-table handler:
const { completed } = await app.flyTo(home, { durationMs: 2000 });
if (!completed) console.log('visitor took over mid-flight');
app.setRenderingSettings({ exposure: 0.5, toneMapping: 'ACES' });
app.setLayer(app.getLayers()[0].path, { opacity: 0.3, colormap: 'viridis' });
```

`camera-flight.ts` (in `../camera/`) holds the tween; `buildFlightPath` and
`easeFlight` are exported for tests and for anyone who needs the path without
the driver.

## See Also

- [`../README.md`](../README.md) — `app/` package overview: orchestrator support tree, lifecycle, and the `LuxarApp` public API.
- [`../../app.ts`](../../app.ts) — The `LuxarApp` class that calls these helpers.
- [`../snapshot/`](../snapshot/) — Separate JSON capture/restore of camera + slice position (for tests, share-view links, regression harnesses).
- [`../../../index.ts`](../../../index.ts) — Package barrel re-exporting these public types.
