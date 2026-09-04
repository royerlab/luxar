# `luxar-orbit-controls/input/`

DOM-event handler bodies for `LuxarOrbitControls`. Extracted from the
orchestrator so it can stay focused on the sequenced per-frame update
and the lifecycle of bound listeners. Each file exports pure functions
that operate on an `OrbitInputCtx` the orchestrator builds once; mutable
THREE refs (`Vector2`, `Quaternion`, pointer arrays) are shared, and
primitive state (`state`, `zoomDelta`) flows through accessor callbacks
so the orchestrator remains the source of truth.

Event dispatch (`change` / `start` / `end`) always goes back through
the orchestrator via `ctx.dispatch` — handlers never construct or fire
events themselves. This keeps the listener contract identical to the
pre-extraction class.

```
input/
├── pointer.ts   # Mouse + pen handlers, the `OrbitInputCtx` shape, action mapping
├── touch.ts     # 1-finger rotate/pan, 2-finger pinch-dolly + drag-pan
└── keyboard.ts  # Arrow-key pan attachment (self-contained, no shared ctx)
```

## Per-file purpose

- `pointer.ts` — Defines `OrbitInputCtx` (the context object every
  handler in this folder consumes), the `ControlAction` /
  `ControlEventName` types, and the pointer-event handler bodies:
  - `pointerNDC(event, domElement)` — clientX/Y → normalized device
    coordinates (used by the trackball math).
  - `mouseAction(button, shiftKey, ctx)` — Resolves a button + Shift
    into `'rotate' | 'pan' | 'zoom' | 'none'`, honoring the
    `mouseButtons` mapping and the `enableRotate/Pan/Zoom` gates.
    Shift+left inverts the primary action (pan↔rotate swap).
  - `handlePointerDown` — Captures the pointer, registers
    pointermove/up/cancel on first contact, primes the per-action start
    accumulators (`rotateStart` / `panStart` / `dollyStart`), and
    dispatches `'start'` if an action engaged. Touch pointers are
    forwarded to `ctx.onTouchStart`.
  - `handlePointerMove` — Updates the live pointer position map, then
    runs the active action: arcball-rotate via `computeArcballRotation`,
    pan via `ctx.pan`, or signed zoom via `ctx.addZoomDelta`
    (`computeZoomScale`). Touch pointers forward to `ctx.onTouchMove`.
  - `handlePointerUp` — Filters out the released pointer (via a local
    `remaining` list, since `setPointers` swaps the array reference),
    releases capture, tears down move/up/cancel listeners on last
    release, and dispatches `'end'`.
  - `handleWheel` — Sign-flipped zoom via `computeZoomScale` at
    `zoomSpeed × wheelZoomSensitivity` (the latter is the global
    Settings > Input knob, read from config per event; pointer-drag
    dolly and two-finger touch pinch deliberately ignore it); wakes the
    animation loop with `dispatch('change')` so damping applies inside
    the orchestrator's `update()`.
- `touch.ts` — Touch-gesture bodies operating on the same
  `OrbitInputCtx`. One finger → rotate (or pan if rotation is
  disabled). Two fingers → combined `'zoom'` state that runs a pinch
  dolly (negated so spread = zoom in, matching scroll-up) plus a
  center-of-mass pan. Coordinates come from
  `ctx.pointerPositions` (the live position map maintained by
  `handlePointerMove`).
- `keyboard.ts` — Self-contained `attachKeyboardPan(element, ctx)`.
  Wires a `keydown` listener for the four arrow keys, calls
  `ctx.pan(±speed, 0)` / `ctx.pan(0, ±speed)`, and returns a disposer.
  Uses a narrower `OrbitKeyboardCtx` (just `enabled`, `enablePan`,
  `keyPanSpeed`, `pan`) so the orchestrator can attach it
  independently of the pointer/touch context.

## How the orchestrator wires it

The `LuxarOrbitControls` orchestrator builds a single `OrbitInputCtx`
during construction and binds it to `pointerdown` / `pointermove` /
`pointerup` / `wheel` on the canvas. Pointer entry points dispatch on
`pointerType` and call the touch bodies via the `onTouchStart` /
`onTouchMove` callbacks on the context — so all input modalities share
the same accumulator state. `attachKeyboardPan` is wired separately
(via `listenToKeyEvents`) onto the window or a focusable element and
returns its own disposer.

## See also

- `../math/` — Pure math the handlers call: `computeArcballRotation`
  (`trackball.ts`), `computeZoomScale` (`zoom.ts`).
- `../README.md` (sibling) and `../../README.md` — Orchestrator and
  package overview.
