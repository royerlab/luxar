# `luxar-fly-controls/` helpers

Per-class helper cluster for `../luxar-fly-controls.ts`, the free-flight
6DOF controller. Each module owns one slice of the controller's body —
camera ↔ orientation sync, physics integration, or DOM listener wiring
— while the orchestrator at the package root keeps the lifecycle, the
event channel (`change` / `start` / `end`), and the public class
surface. Parallel in spirit to `../luxar-orbit-controls/`.

The fly controller drives the camera directly from a single quaternion
(no separate target / distance pair), so this cluster is narrower than
the orbit cluster: no `math/` subfolder, just orientation sync +
physics + listeners + the `input/` event-handler subfolder.

## Files

```
luxar-fly-controls/
├── camera-application.ts  # initializeFromCamera / updateOrientation / lookAtSmooth
├── physics.ts             # integrateTranslation / integrateRotation (per-frame step)
├── listeners.ts           # attachListeners → disposer (mouse, touch, wheel, optional keyboard)
└── input/                 # DOM-event handler bodies (keyboard, mouse, touch, wheel)
```

| File                    | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `camera-application.ts` | Camera ↔ orientation sync. `initializeFromCamera(camera, orientation)` seeds the quaternion from `camera.quaternion`. `updateOrientation(camera, orientation)` writes the quaternion back to the camera and keeps `camera.up` in sync (so state export and non-screenSpacePanning pan in adjacent controls stay correct). `lookAtSmooth(camera, orientation, target, smoothness)` builds a look-at quaternion via a temp `Matrix4.lookAt` then `slerp`s the orientation toward it by `1 - smoothness` (0 = snap, 1 = no change) and re-applies it through `updateOrientation`.                                                                                                                                                                                                                                                                                                                                                                                           |
| `physics.ts`            | Per-frame integration step. `integrateTranslation(ctx, delta)` derives forward / right / world-up vectors from the orientation, accumulates acceleration from `moveState` (WASD + Alt+W/S) scaled by `movementSpeed` and a 2× Shift `speedBoost`, applies frame-rate-independent damping via `Math.pow(damping, delta * config.controls.fly.physics.dampingPower)`, advances `camera.position`, and zeros velocity below `velocityThreshold`. `integrateRotation(ctx, delta)` does the angular analogue: torque from `lookState` (vertical / horizontal / roll) along the camera's _local_ axes, world-space angular-velocity accumulation, `setFromAxisAngle` → `premultiply` for a world-space delta rotation, then damping and an `angularVelocityThreshold` zero-snap. Each returns whether motion happened so the orchestrator can dispatch `change`. Non-inertial mode forces damping to `0.5` and skips torque accumulation (direct angular-velocity assignment). |
| `listeners.ts`          | DOM event-listener wiring. `attachListeners(ctx)` registers mouse + touch `pointerdown` + wheel + `contextmenu` listeners on `ctx.domElement`, mouse / touch `pointermove` and `pointerup` plus touch `pointercancel` on `window`, and returns a disposer that detaches the same set. Keyboard listeners are skipped when `ctx.externalInputManagement` is true — in that mode the `InputContextManager` routes keys to the orchestrator's public `handleKeyDown` / `handleKeyUp` methods instead. Wheel is attached with `{ passive: false }` so `preventDefault` can suppress page scroll.                                                                                                                                                                                                                                                                                                                                                                             |

## How user input becomes camera motion

```
KeyboardEvent / MouseEvent / PointerEvent / WheelEvent
        │
        ▼  (input/keyboard.ts, input/mouse.ts, input/touch.ts, input/wheel.ts)
   moveState  /  lookState  /  velocity  /  angularVelocity  /  orientation
        │
        ▼  (physics.ts — every frame, called from orchestrator.update())
   integrateTranslation → camera.position
   integrateRotation    → orientation quaternion
        │
        ▼  (camera-application.ts)
   updateOrientation → camera.quaternion + camera.up
```

The orchestrator (`../luxar-fly-controls.ts`) owns the state objects
(`velocity`, `angularVelocity`, `orientation`, `moveState`, `lookState`,
the inertial / damping / speed knobs) and passes them to the helpers
through small `Ctx` interfaces (`FlyPhysicsCtx`, `FlyListenersCtx`).
Helpers do not import the orchestrator class and do not hold per-call
state — the only module-local state is the scratch vectors / quaternion
inside `physics.ts`.

## Public surface (consumed by `../luxar-fly-controls.ts`)

- `initializeFromCamera`, `updateOrientation`, `lookAtSmooth` from `camera-application.ts`
- `integrateTranslation`, `integrateRotation`, `FlyPhysicsCtx` from `physics.ts`
- `attachListeners`, `FlyListenersCtx` from `listeners.ts`
- Re-exported `FlyMoveState`, `FlyLookState` and the input handlers from `input/`

Nothing in this folder is part of the package's external API — outside
callers go through the `LuxarFlyControls` class.

## See also

- [`../luxar-fly-controls.ts`](../luxar-fly-controls.ts) — the host class; the public methods (`update`, `dispose`, `handleKeyDown`/`Up`, mode-config setters) are thin shims that delegate here.
- [`./input/README.md`](./input/README.md) — the DOM-event handler subfolder (keyboard / mouse / touch / wheel) that produces the `moveState` / `lookState` / `velocity` / `orientation` updates these helpers consume.
- [`../README.md`](../README.md) — the controls package overview, the fly-controls input table, and the inertial / non-inertial mode semantics.
- [`../luxar-orbit-controls/`](../luxar-orbit-controls/) — the parallel helper cluster for orbit/ortho controls.
- `../../config/index.ts` — `config.controls.fly.physics` (`dampingPower`, `velocityThreshold`, `angularVelocityThreshold`) consumed by `physics.ts`.
