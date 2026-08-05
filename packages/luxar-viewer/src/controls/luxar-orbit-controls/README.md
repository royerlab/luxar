# LuxarOrbitControls helpers

Pure helpers extracted from `controls/luxar-orbit-controls.ts`. The
orchestrator class lives one level up and keeps the public API
(constructor, listener wiring, `update()`, `reinitialize()`, `dispose()`,
event dispatch). Files here own one slice each — the per-frame update
sequencer, the camera-write step, and the thematic `math/` and `input/`
subgroups it delegates to. None of them hold a back-reference to
`LuxarOrbitControls`; everything flows through parameters or a small
ctx interface.

This mirrors the `luxar-fly-controls/` layout next door so the two
orchestrators stay structurally parallel (see `../README.md`'s
"Package Architecture" tree).

## Files

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `camera-application.ts` | Camera transform write step. `applyToCamera(camera, target, orientation, distance)` is the hot path called every frame by `runUpdateStep`; it uses a module-local scratch `Vector3` to match the orchestrator's no-allocation pattern and calls `updateMatrixWorld()` so pan math can read the matrix columns the next frame. `initializeFromCamera(camera, target, outOrientation)` is the cold path (constructor / `reset` / `reinitialize`) — it derives `up` from `camera.quaternion` (not `camera.up`, which goes stale across fly→orbit switches) and inserts a fallback up vector when the view direction is parallel to the up vector, preventing the `lookAt()` singularity. Returns the new orbit distance since `number` is not pass-by-reference.      |
| `update.ts`             | Per-frame update sequencer. `runUpdateStep(ctx, deltaTime?)` runs nine steps in fixed order: auto-rotation around the screen-vertical axis, trackball-rotation damping, view-axis roll damping, pan damping, zoom damping (delegated to `math/zoom.ts::applyZoomScale`), distance clamp, orthographic-zoom clamp, camera write (`applyToCamera`), and `change`-event dispatch. Returns `true` when the view actually changed — position, orientation, or orthographic zoom (which mutates only `camera.zoom`) — so the host can drive render-on-demand. Reads/writes the orchestrator's state through the `OrbitUpdateCtx` interface (orientation quaternion, rotation/pan/zoom/roll deltas, target, distance, clamp bounds, last-pose accumulators, dispatch fn). |

## Subpackages

- `math/` — pure functions for trackball axis-angle derivation, screen-space pan, and zoom-scale application (perspective distance multiply vs orthographic `camera.zoom` scale). Consumed by both `update.ts` and the input handlers. See `math/README.md`.
- `input/` — DOM-event handler bodies (`pointer.ts`, `touch.ts`, `keyboard.ts`) extracted from the orchestrator's bound listeners. Each handler reads its math from `../math/` and writes deltas onto the orchestrator's accumulators (`rotationDelta`, `panDelta`, `zoomDelta`, `rollDelta`) which the next `update()` tick drains with damping. See `input/README.md`.

## ASCII tree

```
luxar-orbit-controls/
├── camera-application.ts   # applyToCamera (hot), initializeFromCamera (cold)
├── update.ts               # runUpdateStep — nine-step per-frame sequencer
├── math/                   # pure math: trackball, pan, zoom
└── input/                  # DOM handlers: pointer, touch, keyboard
```

## Invariants

- **Hot path is allocation-free.** `applyToCamera` and `runUpdateStep`
  use module-local scratch vectors and quaternions (`_v`, `_v2`,
  `_q1`, `_IDENTITY_QUAT`). New `Vector3`/`Quaternion` allocations are
  confined to cold paths (`initializeFromCamera`, constructors).
- **Step order in `runUpdateStep` is load-bearing.** Auto-rotation
  must come before trackball damping (it premultiplies the orientation
  the trackball step then multiplies into), and `applyToCamera` must
  follow all six damping/clamp steps so the camera reflects this
  frame's final state before the change-detection step compares
  against `lastPosition` / `lastQuaternion`.
- **Orientation is authoritative over `camera.up`.**
  `initializeFromCamera` rebuilds `up` from `camera.quaternion` to
  preserve roll across fly→orbit mode switches (fly controls update
  the quaternion but not `up`).
- **No back-reference to the orchestrator.** `runUpdateStep` receives
  every piece of state it needs through `OrbitUpdateCtx`; the
  orchestrator owns the actual fields and exposes getters/setters for
  the scalar accumulators that can't be shared by reference.

## See also

- `../luxar-orbit-controls.ts` — the orchestrator class; constructor
  builds the `OrbitUpdateCtx`, `update()` is a thin shim around
  `runUpdateStep`, and event dispatch (`start` / `change` / `end`)
  stays on the class.
- `../README.md` — package-level overview, control-mode summary, and
  the parallel `luxar-fly-controls/` layout.
- `../../utils/camera-utils.ts` — `LuxarCamera` union (perspective +
  orthographic) consumed by both helpers.
- `../../scene/scene-manager/camera/` — host-side camera framing,
  mode swap, and zarr-viewer-config application that interact with
  these controls through the `ControlsManager` facade.
