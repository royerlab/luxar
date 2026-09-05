# core/app/camera — Camera flight

**Internal** (orchestrator support). The tween behind `LuxarApp.flyTo()`: a smooth,
interruptible transition from the live camera pose to a target `CameraSnapshot`.
Part of the remote-control design — see `docs/guides/specs/REMOTE_CONTROL_SPEC.md`
(§2.1) and the public surface in [`../embedder/README.md`](../embedder/README.md).

## Modules

| File                 | Description                                                                                                                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera-flight.ts`   | `CameraFlight` (the driver), `buildFlightPath` (pure interpolant between two snapshots), `easeFlight`, `FlyToOptions` / `FlightResult`.                                                                   |
| `waypoint-driver.ts` | `WaypointDriver` + the pure `waypointMatches` / `matchWaypoint` / `resolveWaypointPose`: binds `viewer_config.waypoints` (authored camera poses keyed on hidden-dimension positions) to the dims manager. |

## Behaviour

- **Orbit-space interpolation.** The focus target moves linearly; the camera's
  offset from it is slerped in direction and log-interpolated in distance; `up` is
  slerped; `fov` lerps, ortho `zoom` log-lerps, near/far lerp. A straight-line
  position lerp between two orbit poses cuts through the object and through the
  target — this keeps the arc. A degenerate zero-length offset borrows the other
  end's direction.
- **Hand-off every frame.** Each intermediate pose is written like `restoreCamera`
  does (`controls.setTarget` + `reinitialize` + a controls `change` event), so the
  orbit controls never snap back, the render loop wakes, and LOD / depth sort /
  picking see the pose. The final frame **is** `restoreCamera(pose)`: a completed
  flight lands bit-exactly.
- **Driver.** Registered as a `continuous` per-frame callback (`camera-flight`)
  and paired with `startAnimation()` — registration alone never starts a stopped
  loop (see `reference_per_frame_callbacks_need_startanimation`).
- **Interruption.** `pointerdown` / `wheel` / `touchstart` on the input element and
  `keydown` on the document (capture phase, passive) cancel the flight where it is;
  so do a newer `flyTo()`, `cancel()`, and `dispose()`. The promise resolves
  `{ completed: false }`. `durationMs <= 0` applies the pose immediately.
- One flight at a time per app; `LuxarApp` creates the instance in
  `setupEmbedderHooks` and disposes it through the app's event group.

## Waypoints

`WaypointDriver` is the story mechanism: a scene with a hidden discrete "story"
dimension authors, per value, its colours (data), its captions (overlays with a
`visible_range`) and — via `viewer_config.waypoints` — its camera. The matcher
is the overlay manager's `visible_range` rule verbatim (every named dimension
must hold; exact = within ±0.5 of the current step; `[min, max]` inclusive;
unknown dimension names skipped; FIRST match in list order wins).

The driver acts on a **change of matched waypoint**, not on every tick:
`evaluate('snap')` at load applies the opening match through `restoreCamera`
(ahead of the plain `camera` block, which ran during `loadSceneData`);
`evaluate('fly')` on each dims-manager notification flies via `CameraFlight`
with the waypoint's `duration_ms` / `easing` (`0` snaps), applies its optional
`rendering` block through `RenderingControls.applyOverrides` (snake_case keys via
`extractRenderingOverrides`, the authored-defaults path), and does nothing while
the match is unchanged or when nothing matches. `resolveWaypointPose` starts
from the LIVE pose so omitted camera fields keep their value (re-aim without
moving); `target_node` resolves through `resolveTargetNodeCenter` and beats
`target`. `LuxarApp.installWaypoints` owns the lifecycle: built in
`applyViewerConfigState` after `dimensions.current_step` is applied, torn down
at the start of the next `loadDataset` and on dispose.

## Tests

`src/tests/unit/core/app/camera/waypoint-driver.test.ts`: matcher parity with the
overlay rule (tolerance, inclusive ranges, AND across clauses, unknown names,
first-match-wins), `resolveWaypointPose` (live-pose defaults, `target_node`
precedence and fallback, fov / preset / planes), and the driver's
change-of-match semantics (snap at load, fly on change, inert inside a range,
inert when nothing matches, `duration_ms: 0`, rendering ride-along, no dims).

## Camera-flight tests

`src/tests/unit/core/app/camera/camera-flight.test.ts` drives the per-frame callback
by hand with an injected clock against a real `THREE.PerspectiveCamera`: path
geometry (arc distance, geometric distance, spherical up, ortho zoom), landing
exactness, registration + start pairing, cancel on pointer and keyboard,
supersession, idempotent cancel/dispose, and the headless (no input element) case.
