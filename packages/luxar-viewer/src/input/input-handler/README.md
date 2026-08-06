# `input/input-handler/`

Private implementation of the `InputHandler` orchestrator. Nothing in this
folder is part of the package's public API — external callers import
`InputHandler` from `../input-handler.ts`, which is the one file in the
parent folder. Everything here is reachable only through that façade.

The orchestrator does not own logic — it owns wiring. The actual work is
split into five thematic subfolders plus one top-level file: a
context-aware key routing core (`context-manager.ts`). When
`InputHandler.init()` runs, it pulls dependencies from each subfolder and
hands them a narrow `*Ctx` object — never `this` — so the subfolders stay
unit-testable in isolation.

## Top-level files

```
input-handler/
└── context-manager.ts          # InputContextManager + InputContext + KeyBinding (the routing core)
```

- `context-manager.ts` — `InputContextManager` class plus the
  `InputContext` enum (`NAVIGATION` / `FLY_CONTROLS` / `TYPING` /
  `UI_INTERACTION` / `DIMENSION_NAV`), the `KeyBinding` /
  `ContextConfig` interfaces, and the `MAX_KEY_EVENT_DEPTH = 10`
  recursion cap. This is the routing table the orchestrator pushes
  contexts onto and the per-context bindings are registered into.
  Pure-function helpers live one level down in `context-manager/`.

## Subpackages

```
input-handler/
├── context-manager/        # Pure routing-rules helpers for context-manager.ts
├── key-bindings/           # The per-context key→command table
├── window-events/          # Window/document-level listeners (resize, wheel, fullscreen)
├── dimension-navigation/   # nD navigation math + slider lifecycle
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
  the `calculateStepSize` / `calculateNextPosition` math, dim-index
  helpers (`getNonDisplayedDimensions`, `mapKeyToDimension`), and the
  four lifecycle bodies (`initDimensionSliders`, `initAnimationManager`,
  `clearDimensionUI`, `updateAllNDNodes`).
- **`commands/`** — command bodies the orchestrator delegates to:
  `PanelCoordinator` (Escape priority flow + recording short-circuit +
  fullscreen-defer rule), `exportViewerState` (Ctrl+Shift+S clipboard
  export), `toggleControlMode` / `toggleInertialMode` / `nextControlType`
  (V and I keys), `isTypingInInput` / `isFocusOnSceneCanvas` focus
  helpers, and `cycleDataMonitor`.

## Lifecycle through the subfolders

`InputHandler.init()` performs the wiring in this order:

1. Construct `InputContextManager` and push `InputContext.NAVIGATION`.
2. Construct `WindowEventHandler` (`window-events/`) — installs
   `resize`, `wheel`, and `fullscreenchange` listeners.
3. Call `registerAllKeyBindings` (`key-bindings/`) with three records:
   `deps`, `commands`, `panelGetters`. This is the only place per-context
   bindings are added to the manager.
4. Construct `PanelCoordinator` (`commands/`) so Escape and the
   panel-cycle binding have a single drain.

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
