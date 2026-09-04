# core/app/camera — Camera flight

**Internal** (orchestrator support). The tween behind `LuxarApp.flyTo()`: a smooth,
interruptible transition from the live camera pose to a target `CameraSnapshot`.
Part of the remote-control design — see `docs/guides/specs/REMOTE_CONTROL_SPEC.md`
(§2.1) and the public surface in [`../embedder/README.md`](../embedder/README.md).

## Modules

| File               | Description                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `camera-flight.ts` | `CameraFlight` (the driver), `buildFlightPath` (pure interpolant between two snapshots), `easeFlight`, `FlyToOptions` / `FlightResult`. |

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

## Tests

`src/tests/unit/core/app/camera/camera-flight.test.ts` drives the per-frame callback
by hand with an injected clock against a real `THREE.PerspectiveCamera`: path
geometry (arc distance, geometric distance, spherical up, ortho zoom), landing
exactness, registration + start pairing, cancel on pointer and keyboard,
supersession, idempotent cancel/dispose, and the headless (no input element) case.
