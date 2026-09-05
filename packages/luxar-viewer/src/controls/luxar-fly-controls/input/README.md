# `luxar-fly-controls/input/`

Input adapters for `LuxarFlyControls`. Extracted from
`luxar-fly-controls.ts` so the orchestrator stays focused on lifecycle
and the per-frame physics loop while these helpers translate raw DOM
events into intents.

Each adapter is a free function (no class state). State lives on the
orchestrator and is reached through a per-call `Ctx` object: object
refs (`moveState`, `lookState`, `velocity`, `angularVelocity`,
`orientation`, `camera`) are mutated in place; primitives
(`speedBoost`, `mouseX/Y`, `activeMouseAction`) are read and written
via getter/setter callbacks. Event dispatch is routed back through
`ctx.dispatch` so the orchestrator remains the sole dispatch site.

```
input/
├── keyboard.ts  WASD + QE roll + arrows + Shift speed-boost
├── mouse.ts     Left-drag strafe, right-drag rotate (angular impulse)
└── wheel.ts     Scroll forward/back, Shift+scroll roll
```

## `keyboard.ts`

`handleKeyDown` / `handleKeyUp`, plus the shared `FlyMoveState`,
`FlyLookState`, `FlyMouseAction` types and `FlyKeyboardCtx`.

- **WASD** drives `moveState.{forward,back,left,right}`; **Alt/Meta+W**
  and **Alt/Meta+S** redirect to `moveState.{up,down}` instead.
- **Q / E** sets `lookState.roll` to ±1.
- **Arrow keys** set `lookState.{horizontal,vertical}` for continuous
  look (via the private `startLookChange` helper).
- **Shift** flips the `speedBoost` primitive through `setSpeedBoost`.
- `preventDefault` is scoped: always for `Arrow*`, and for WASD/QE only
  when the active element is not an `<input>`, `<textarea>`, or
  `contenteditable` host — so typing in the GUI is never swallowed.
- Every handled key ends with `ctx.dispatch('change')`.

## `mouse.ts`

`handleMouseDown` / `handleMouseUp` / `handleMouseMove` and
`FlyMouseCtx`. Two module-local `THREE.Vector3` scratch refs avoid
per-event allocation.

- **Left button (0)** activates `'strafe'`; **right button (2)**
  activates `'rotate'`. Both stash the cursor position and dispatch
  `'start'`. `handleMouseUp` only clears the action if the released
  button matches the active action, then dispatches `'end'`.
- **rotate** converts mouse deltas into pitch/yaw angular impulses
  (`±delta * lookSpeed * 2.5`) around the camera's local X and Y axes
  derived from `orientation`, accumulated into `angularVelocity`.
- **strafe** converts deltas into screen-aligned translations scaled by
  `movementSpeed * 0.005`. In `inertialMode` the impulse is added to
  `velocity`; otherwise the camera position is moved directly. Drag
  direction matches on-screen motion (consistent with orbit/ortho pan).

## `wheel.ts`

`handleWheel` and `FlyWheelCtx`. Module-local `_v0` and `_q0` scratch.

- **Ctrl/Meta+scroll** is passed through untouched — it belongs to the
  upstream `InputHandler` FOV path.
- Otherwise calls `preventDefault` and normalises `deltaY` via
  `-Math.sign(...)` so line / pixel / page scroll modes behave alike.
- **Shift+scroll** rolls around the viewing axis: inertial-mode adds an
  impulse to `angularVelocity`; non-inertial premultiplies a quaternion
  into `orientation` and renormalises.
- **Plain scroll** translates along the viewing axis using
  `movementSpeed * wheelZoomSensitivity * 0.3` (inertial impulse on
  `velocity`) or the same `* 0.2` (direct camera-position step otherwise).
  `wheelZoomSensitivity` is the global Settings > Input > Zoom Sensitivity
  multiplier (`config.controls.wheelZoomSensitivity`), read per event; the
  Shift+scroll roll is not scaled by it.

## Public API

| Export                                                | Source        |
| ----------------------------------------------------- | ------------- |
| `handleKeyDown`, `handleKeyUp`                        | `keyboard.ts` |
| `handleMouseDown`, `handleMouseUp`, `handleMouseMove` | `mouse.ts`    |
| `handleWheel`                                         | `wheel.ts`    |
| `FlyMoveState`, `FlyLookState`, `FlyMouseAction`      | `keyboard.ts` |
| `FlyKeyboardCtx`, `FlyMouseCtx`, `FlyWheelCtx`        | each file     |
