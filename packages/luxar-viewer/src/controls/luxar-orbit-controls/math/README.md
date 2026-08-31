# LuxarOrbitControls math helpers

Pure math used by `luxar-orbit-controls.ts`. Each module owns one
slice of the orbit-control update step — trackball rotation, pan,
zoom, turntable axis — and reads everything it needs from parameters. None of them
keep their own state or hold a back-reference to the orchestrator.
Allocations are kept to either the explicit return value or a
caller-supplied accumulator; module-local scratch vectors mirror the
orchestrator's existing per-frame allocation pattern.

## Files

```
math/
├── trackball.ts    Shoemake virtual-trackball rotation (sphere + hyperboloid)
├── pan.ts          OrbitControls pan math (perspective + ortho)
├── zoom.ts         OrbitControls zoom math (perspective distance / ortho zoom)
├── auto-rotate.ts  Turntable axis → world-space vector
└── auto-dolly.ts   Sinusoidal distance oscillation (phase → distance factor)
```

### `auto-rotate.ts`

- `autoRotateAxisVector(axis, orientation, out) -> Vector3` writes the
  world-space rotation axis for an `AutoRotateAxis`. Camera-frame tokens
  (`'vertical'` = camera up, `'horizontal'` = camera right, `'view'` =
  the view direction `-Z`) are rotated through the live `orientation`;
  world tokens (`'world-x'`/`'-y'`/`'-z'`) are world-space constants
  returned as-is. Rotation follows the right-hand rule about the named
  direction uniformly across both families, `'view'` matches the vector
  the Shift+scroll roll delta uses so both roll the same way, and
  `'world-y'` agrees with `'vertical'` exactly while the camera is level.
- The direction table is a total `Record` over the union, so adding a
  token to `AutoRotateAxis` without a direction fails to compile.

### `auto-dolly.ts`

- `advanceDollyPhase(phase, deltaTime, period) -> number` walks the
  oscillation phase, wrapped to `[0, 2π)`. Inert on a non-positive or
  non-finite period.
- `dollyScale(fromPhase, toPhase, amplitude) -> number` returns the factor
  to multiply the orbit distance by. The oscillation is defined in LOG
  distance, `d(φ) = d₀·exp(−A·sin φ)` with `A = ln(1 + amplitude)`, so
  `amplitude` is a RATIO (0.15 → `d₀×1.15` out, `d₀÷1.15` in) and means the
  same thing at any scene scale — which is what makes it the sinusoidal
  mousewheel the feature is named for.
- Returning a RATIO between two phases, rather than an absolute distance, is
  what lets the user keep zooming while it runs: distance is only ever
  multiplied, and multiplication commutes, so a wheel click moves the centre
  the camera breathes around instead of fighting the animation.
- Taking the DIFFERENCE OF SINES rather than integrating `−A·cos φ·dφ` keeps
  the amplitude exact at any frame rate and makes a period's factors
  telescope to exactly 1, so the centre cannot drift over a long session.
- Both the interactive dolly (`update.ts`, wall-clock) and the recorded one
  (the capture strategies, frame-indexed) go through `dollyScale`, so an
  exported video cannot breathe unlike its own preview.
- Both the per-frame turntable (`update.ts` step 1) and the
  programmatic one (`LuxarOrbitControls.applyOrbitRotation`, which
  recording drives) read the axis from here, so an exported turntable
  cannot rotate unlike its own preview.
- A world axis parallel to the view direction is benign rather than
  singular: the camera offset lies along it, so the camera stays put and
  the image rolls, exactly as `'view'` does.
- An unrecognized token degrades to `'vertical'` rather than throwing:
  this runs inside the render loop, and a hand-edited scene attribute
  should not kill every subsequent frame.

### `trackball.ts`

Vendored from ArcballControls / Ken Shoemake's virtual-trackball paper.

- `projectOnTrackball(ndcX, ndcY, radius) -> Vector3` projects an NDC
  point onto a hybrid surface: a sphere of `radius` for points inside
  `d² <= r²/2`, and a `z = r²/(2·d)` hyperboloid outside. The split at
  `d² = r²/2` is the standard C¹-smooth Shoemake transition that keeps
  the trackball from "falling off" near the edges.
- `computeArcballRotation(startNDC, endNDC, radius, rotateSpeed) -> Quaternion`
  builds `axis = p1 × p2`, `angle = acos(p1·p2) · rotateSpeed`, and
  returns `setFromAxisAngle(axis, -angle)`. The angle is **negated** so
  the camera orbits opposite to the drag (drag right ⇒ camera moves
  left around target). Degenerate drags (`|axis|² < 1e-10`) return
  identity.

### `pan.ts`

Vendored from THREE.js OrbitControls. Functions **mutate a
caller-supplied accumulator (`out`)** instead of returning fresh
`Vector3`s, preserving the orchestrator's zero-alloc pattern.

- `applyPanLeft(out, distance, objectMatrix)` adds `-distance · matrix.col(0)`.
- `applyPanUp(out, distance, objectMatrix, cameraUp, screenSpacePanning)`
  pans along either the camera Y axis (screen-space) or
  `cameraUp × camera.x` (world-up pan).
- `applyPan(out, deltaX, deltaY, ctx)` converts pointer pixel deltas
  into world-space. Perspective: screen-height at the orbit distance
  is `2·distance·tan(fov/2)`, scaled by `clientHeight`. Orthographic:
  deltas scale off frustum width/height divided by `zoom`.

### `zoom.ts`

- `computeZoomScale(delta, zoomSpeed) -> number` returns
  `0.95 ^ (zoomSpeed · |delta·0.01|)` — geometric per tick, not
  additive.
- `applyZoomScale(camera, currentDistance, scale, minZoom, maxZoom)`
  branches on projection. **Perspective**: returns
  `currentDistance · scale` for the caller to reassign — position is
  derived from `distance × orientation` outside this module.
  **Orthographic**: mutates `camera.zoom = clamp(zoom/scale, …)`,
  refreshes the projection matrix, and returns `currentDistance`
  unchanged.

## Invariants

- **Pure functions.** No module-level state beyond `pan.ts`'s `_v`
  scratch vector, which is overwritten on every call.
- **Caller owns the camera mutation.** Trackball and the perspective
  branch of zoom return values; only `zoom.ts`'s ortho branch mutates
  the camera directly, because `OrthographicCamera.zoom` has no
  equivalent in the orchestrator's spherical-coords state.
- **Drag direction sign lives here.** The `-angle` in
  `computeArcballRotation` is the single source of truth for "drag
  right rotates view right"; the orchestrator passes raw NDC deltas
  without re-negating.

## See also

- `../../luxar-orbit-controls.ts` — orchestrator that imports
  `applyPan` (accumulating into `this.panDelta`).
- `../input/pointer.ts`, `../input/touch.ts` — sibling DOM-event
  handlers that produce the NDC and pointer-delta inputs, and call
  `computeArcballRotation` / `computeZoomScale` directly.
- `../update.ts` — per-frame update step that applies the accumulated
  zoom delta via `applyZoomScale`.
- `../../README.md` — package overview and control-scheme tables.
