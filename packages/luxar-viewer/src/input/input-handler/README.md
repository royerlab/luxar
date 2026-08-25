# `input/input-handler/`

Private implementation of the `InputHandler` orchestrator. Nothing in this
folder is part of the package's public API — external callers import
`InputHandler` from `../index.ts`, the package façade. Everything here is
reachable only through that façade.

The orchestrator does not own logic — it owns wiring. The actual work is
split into five thematic subfolders plus two top-level files: the
context-aware key routing core and the UI capability contracts. When
`InputHandler.init()` runs, it pulls dependencies from each subfolder and
hands them a narrow `*Ctx` object — never `this` — so the subfolders stay
unit-testable in isolation.

## Top-level files

```
input-handler/
├── context-manager.ts          # InputContextManager + InputContext + KeyBinding (the routing core)
└── panel-capabilities.ts       # Narrow UI contracts consumed by input
```

- `context-manager.ts` — `InputContextManager` class plus the
  `InputContext` enum (`NAVIGATION` / `FLY_CONTROLS` / `TYPING` /
  `UI_INTERACTION` / `DIMENSION_NAV`), the `KeyBinding` /
  `ContextConfig` interfaces, and the `MAX_KEY_EVENT_DEPTH = 10`
  recursion cap. This is the routing table the orchestrator pushes
  contexts onto and the per-context bindings are registered into.
  Pure-function helpers live one level down in `context-manager/`.
- `panel-capabilities.ts` — structural contracts for UI panels and injected
  factories, keeping concrete `ui/` classes out of the input layer.

## Subpackages

```
input-handler/
├── context-manager/        # Pure routing-rules helpers for context-manager.ts
├── key-bindings/           # The per-context key→command table
├── window-events/          # Window/document-level listeners (resize, wheel, fullscreen)
├── dimension-navigation/   # nD input coordination + slider lifecycle
└── commands/               # Command bodies the orchestrator delegates to
```

- **`context-manager/`** — `isKeyAllowedInContext` (allow/block filter)
  and `sortContextsByPriority` (descending-priority ordering for
  passthrough fallback). Extracted from `context-manager.ts` so they can
  be unit-tested without instantiating a full manager.
- **`key-bindings/`** — keyboard binding table split per context:
  `register-all.ts` (entry point + `KeyBindings{Deps,Commands,PanelGetters}`
  types), `navigation-bindings.ts` (orbit-mode UI shortcuts),
  `fly-bindings.ts` (WASD + arrows + Shift speed-boost on
  `FLY_CONTROLS`), and `animation-shortcuts.ts` (K / Home / End /
  Shift+↑ / Shift+↓ on `NAVIGATION`).
- **`window-events/`** — `WindowEventHandler` class (`resize` + `wheel`
  - `fullscreenchange`) and the `toggleFullscreen` body.
- **`dimension-navigation/`** — `computeDimensionStep` /
  `resolveSelectedDimension` for the `[`/`]` and digit bindings, plus
  the four lifecycle bodies (`initDimensionSliders`, `initAnimationManager`,
  `clearDimensionUI`, `updateAllNDNodes`). Pure step and selection helpers
  live under `scene/dims/`.
- **`commands/`** — command bodies the orchestrator delegates to:
  `PanelCoordinator` (Escape priority flow + recording short-circuit +
  fullscreen-defer rule), `exportViewerState` (Ctrl+Shift+S clipboard
  export), `toggleControlMode` / `toggleInertialMode` / `nextControlType`
  (V and I keys; the cycle helper is re-exported from `controls/types.ts`),
  and `cycleDataMonitor`. Shared focus predicates live under `utils/dom/`.

## Lifecycle through the subfolders

`InputHandler.init()` performs the wiring in this order:

1. Construct `InputContextManager` and push `InputContext.NAVIGATION`.
2. Construct `WindowEventHandler` (`window-events/`) — installs
   `resize`, `wheel`, and `fullscreenchange` listeners.
3. Call `registerAllKeyBindings` (`key-bindings/`) with four records:
   `deps`, `commands`, `panelGetters`, and `animationShortcuts`. This is the
   only place per-context bindings are added to the manager, including animation
   shortcuts whose handlers decline until a selected dimension and animation
   manager exist.
4. Construct `PanelCoordinator` (`commands/`) so Escape and the
   panel-cycle binding have a single drain.

Binding handlers consume an event by default. A synchronous `false` declines
it so passthrough can continue to lower-priority contexts; async handlers are
always treated as handled. `preventDefault` runs only after a handler accepts
the event, so declining leaves browser behavior untouched.

Optional setters (`setRenderingControls`, `setRecordingPanel`, etc.) are
called by `core/app.ts` as panels are constructed; each one forwards
into `PanelCoordinator` and registers a `getX` on
`KeyBindingsPanelGetters` if a shortcut needs to reach it.

`initDimensionSliders()` runs after a scene loads and delegates to
`dimension-navigation/setup.ts`; `clearDimensionUI()` is its inverse and
runs before a new scene loads. `dispose()` tears down all of the above.

## See Also

- `../README.md` — public-API contract for the package, the three-layer
  architecture (viewer vs. controls vs. UI), and the "adding new
  shortcuts / panels" workflows.
- `../input-handler.ts` — the orchestrator class that ties the subfolders
  together. Read this file before adding new top-level wiring.
- `tests/unit/input/input-handler/` — test suites mirror this folder.
