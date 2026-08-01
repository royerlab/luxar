# Controls-manager helpers

Focused modules split out from the `ControlsManager` orchestrator class
(one level up at `controls/controls-manager.ts`). The orchestrator owns
the active-control field, the saved-state buffers, and the public event
dispatch; these helpers own the pure logic — factory tables for the
three control modes, camera-state capture/restore across a mode swap,
event-forwarder plumbing, and the scale-derived limit math.

Every helper here is pure over its argument bundle (no `this`
reference). The orchestrator passes in a ctx object and the helpers
either return values or mutate the supplied fields. That split lets
each module be unit-tested without standing up a real camera and keeps
`controls-manager.ts` focused on orchestration.

## Files

| File                  | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `factories.ts`        | Per-mode constructor functions: `createOrbitControls`, `createFlyControls`, `createOrthoControls`. Each consumes a `ControlsCreationCtx` (camera, DOM element, manager config, scene scale, stored distance/zoom limits) and returns a configured instance. Also exports `naturalDragButtonMap(enabled)` — the LEFT/MIDDLE/RIGHT `THREE.MOUSE` mapping that `setNaturalDrag` swaps in live without recreating controls.                                                 |
| `camera-state.ts`     | `saveCameraState(ctx)` snapshots position / rotation / up / target before disposing one control instance; `restoreCameraState(ctx)` re-applies the saved target on the new orbit/ortho instance and calls `reinitialize` + `update` to re-derive orbit orientation from the live camera. Fly controls auto-initialise from the camera, so restore is a no-op for them; the snapshot still matters because it seeds the pivot for the _next_ swap back into orbit/ortho. |
| `event-forwarders.ts` | `attachControlEventForwarders(controls, dispatch, group)` wires `change` / `start` / `end` listeners on the active control instance up to a manager-level `dispatch(type)` callback and registers matching detach callbacks with the supplied `EventGroup`. `THREE.EventDispatcher` is not a DOM `EventTarget`, so `EventGroup.on()` doesn't apply — manual cleanup callbacks are registered instead.                                                                   |
| `scene-scale.ts`      | `deriveScaleLimits(diagonal)` — pure math that multiplies the scene bounding-box diagonal by `config.controls.scaleMultipliers.{minDistanceFactor, maxDistanceFactor, flySpeedFactor}` and returns `{ minDist, maxDist, flySpeed }`. The orchestrator owns the field writes, the active-control mutation, the gating against stored auto-frame limits, and the logging.                                                                                                 |

## How the orchestrator composes them

```
ControlsManager (class)
   │
   ├── setControlType('orbit'|'fly'|'ortho')
   │     │
   │     ├── camera-state.saveCameraState(ctx)            // snapshot from old controls
   │     ├── currentControls?.dispose()
   │     ├── factories.create{Orbit,Fly,Ortho}Controls(ctx)
   │     ├── event-forwarders.attachControlEventForwarders(new, dispatch, group)
   │     └── camera-state.restoreCameraState(ctx)         // re-seed target + reinitialize
   │
   ├── setNaturalDrag(enabled)
   │     └── factories.naturalDragButtonMap(enabled)      // hot-swap orbit mouse map
   │
   └── setSceneScale(diagonal)
         └── scene-scale.deriveScaleLimits(diagonal)
              └── orchestrator writes minDistance / maxDistance / flySpeed
                  on the active control + on the stored config
```

## Key contracts

- **Stateless helpers.** None of these modules holds per-call state or
  imports `ControlsManager`. Each takes a ctx object that exposes
  exactly the fields it needs — `ControlsCreationCtx` for factories,
  `CameraStateCtx` for state save/restore, an `EventGroup` plus a
  `dispatch` callback for the forwarders, a `number` for scale math.
- **Stored limits win over scale-derived.** `createOrbitControls` /
  `createOrthoControls` prefer `ctx.storedDistanceLimits` /
  `ctx.storedZoomLimits` (set by the scene auto-frame pass) over the
  `sceneScale * multiplier` fallback, and over the hardcoded
  `config.controls.orbit.zoom.{minDistance,maxDistance}` of last resort.
  This keeps a user's framed view stable across a mode swap.
- **Mode-specific defaults set at construction.**
  `createOrbitControls` enables damping, screen-space panning, and
  view-axis rotation (shift+scroll roll); when `config.naturalDrag` is
  true it swaps the mouse-button map to LEFT=rotate / RIGHT=pan.
  `createOrthoControls` disables rotation, remaps LEFT=pan and RIGHT=null
  (Napari / Google Maps convention), and forces `minDistance=0` /
  `maxDistance=Infinity` because zoom is governed by `minZoom` / `maxZoom`
  on the orthographic camera. `createFlyControls` sets
  `externalInputManagement: true` because keyboard is routed through the
  `input/` package's `InputContextManager`, not Fly's own listeners.
- **Restore order matters.** `restoreCameraState` does
  `target.copy(savedTarget)` → `reinitialize()` → `update()`. The
  `LuxarOrbitControls` constructor initialises with `target=(0,0,0)`
  which is wrong post-swap; setting the target before `reinitialize`
  is what makes the new instance re-derive orbit radius and spherical
  coordinates from the live camera pose against the _correct_ pivot.
- **Fly target derivation.** `saveCameraState` reads `target` directly
  off `LuxarOrbitControls`, but for fly mode it reuses the PREVIOUS saved
  pivot's depth along the current view ray — but only while the camera still
  faces it (within a ~60° cone, `cos60 = 0.5`). This is scale-free, so with no
  fly movement it returns the old target exactly on both sub-micro-unit and
  huge-unit scenes, and a control-mode round trip (orbit → fly → ortho →
  orbit) preserves the pivot. The reused depth is floored at `sceneScale·1e-3`
  so flying right up to the pivot can't collapse it onto the camera; outside
  the cone it falls back to walking `sceneScale` (or `10` if scale is unset)
  along the forward vector. `ControlsManager.getFocusTarget` uses the same
  derivation.
- **Event dispatch stays on the orchestrator.**
  `attachControlEventForwarders` only owns the listener
  attach/detach. The `dispatch` callback it invokes is the
  orchestrator's own `this.dispatchEvent({ type })`, so external
  subscribers see the manager — not the underlying control instance —
  as the event source. Cleanup is registered with the supplied
  `EventGroup` so swap-time disposal happens uniformly with the rest
  of the manager's DOM listeners.

## See also

- `../controls-manager.ts` — the host class; public methods
  `setControlType`, `saveCameraState`, `setNaturalDrag`,
  `setSceneScale` are thin shims that delegate here.
- `../README.md` — package-level overview of the three control modes,
  their bindings, mode-switching semantics, and the full
  `ControlsManager` API surface.
- `../luxar-orbit-controls.ts` / `../luxar-fly-controls.ts` — the
  instances these factories build and these helpers snapshot.
- `../../utils/cross-layer/event-group.ts` — `EventGroup` registers
  the detach callbacks queued by `attachControlEventForwarders`.
- `../../utils/camera-utils.ts` — `LuxarCamera` union threaded through
  every ctx in this folder.
- `../../config/index.ts` — `config.controls.scaleMultipliers`,
  `config.controls.orbit.zoom.{min,max}Distance` consumed by
  `factories.ts` and `scene-scale.ts`.
