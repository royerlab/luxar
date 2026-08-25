# Luxar Input Package

> Viewer-wide input handling — keyboard + mouse + wheel + fullscreen, routed
> through context-aware keybinding tables.

## Overview

`index.ts` is the package facade. It exports `InputHandler`, the
`DimensionSlidersFactory` type used by its constructor, the `InputContext`
identifier used by shortcut-help metadata, and `KeyAction`/`KeyActionId` — the
stable action identities callers address a binding by (the control rail asks
for an action's current chord rather than hard-coding a letter). The binding
registry, context manager implementation, dimension-navigation lifecycle,
window-event handler, and command bodies remain private under `input-handler/`.

External callers (`core/app.ts`, `types/window.d.ts`) import the
orchestrator class:

```typescript
import { InputHandler, KeyAction, type KeyActionId } from '../input';
```

Dependency-cruiser rejects value imports into `input/input-handler/**` from
production modules outside this package. Type-only imports and tests are
currently exempt, so they must still follow the documented boundary by review.

## Layout

```
input/
├── index.ts                                  # Public package facade
├── input-handler.ts                          # InputHandler orchestrator
└── input-handler/
    ├── context-manager.ts                    # InputContextManager + InputContext + KeyBinding
    ├── panel-capabilities.ts                 # Narrow UI contracts consumed by input
    ├── context-manager/
    │   └── routing-rules.ts                  # isKeyAllowedInContext + sortContextsByPriority
    ├── key-bindings/                         # The whole key→command table
    │   ├── actions.ts                        # Stable action identifiers
    │   ├── register-all.ts                   # Entry point + KeyBindings{Deps,Commands,PanelGetters}
    │   ├── navigation-bindings.ts            # Orbit-mode UI shortcuts (H/P/R/V/F/C/L/M/[/]/digits/…)
    │   ├── fly-bindings.ts                   # WASD + arrows + Shift speed boost (FLY_CONTROLS)
    │   └── animation-shortcuts.ts            # K/Home/End/Shift+↑/Shift+↓ (NAVIGATION)
    ├── dimension-navigation/                 # nD input coordination + UI lifecycle
    │   ├── compute-step.ts                   # computeDimensionStep + resolveSelectedDimension
    │   └── setup.ts                          # initDimensionSliders / clearDimensionUI bodies
    ├── window-events/                        # Window/document-level listeners
    │   ├── window-event-handler.ts           # resize + wheel + fullscreenchange class
    │   └── fullscreen-toggle.ts              # toggleFullscreen body
    └── commands/                             # Command bodies the orchestrator delegates to
        ├── panel-coordinator.ts              # PanelCoordinator (Escape flow)
        ├── viewer-state-export.ts            # exportViewerState body
        ├── control-mode.ts                   # commands + nextControlType re-export from controls/types.ts
        └── data-monitor-cycle.ts             # cycleDataMonitor body
```

The depth encodes audience. `index.ts` is public; everything under
`input-handler/` is private to the package; thematic subfolders
(`key-bindings/`, `dimension-navigation/`, `window-events/`, `commands/`,
`context-manager/`) group cohesive helpers.

## Three-layer architecture

`input-handler.ts` is **not** the place where every input event in Luxar is
handled — its scope is viewer-level. Input handling is deliberately split
across three layers, each owning a different scope. Pick the right layer
when adding a new listener.

### Layer 1 — `input/input-handler.ts` (viewer-wide)

Owns concerns that span the whole viewer surface:

- Window-level events: `resize`, `wheel`, `keydown`/`keyup`, `fullscreenchange`.
- Global keyboard shortcuts: `H` (help), `P` (perf), `R` (rendering controls),
  `N` (sliders), `T` (recording), dimension navigation `[`/`]`/`1–9`,
  control-mode switches `V`/`I`/`F`.
- Wheel-based zoom and FOV adjust (Ctrl+wheel).
- Fullscreen enter/exit canvas styling.

### Layer 2 — control implementations (`controls/`)

Each control class (orbit, fly, ortho) owns its own pointer + key handlers
for camera manipulation. Mouse drags, fly-mode WASD movement integration,
trackball rotations — they all live in `controls/`, not here.

### Layer 3 — UI panels (`ui/`)

Panels own their own buttons + form controls. Slider drags, range-input
typing, modal close buttons — they live with the panel that needs them.

## Orchestrator surface

`InputHandler` is constructed by `core/app.ts` with:

```typescript
new InputHandler(
  sceneManager,
  animationController,
  performanceMonitor,
  debugConsole,
  dimensionSlidersFactory? // optional — embed callers may omit
);
```

Setters wire optional panels as they're created:

- `setRenderingControls(controls)`
- `setScaleBar(scaleBar)`
- `setColormapLegend(legend)`
- `setRecordingPanel(panel)`
- `setLayersPanel(panel)`
- `setOverlayManager(manager)`
- `setDatasetBrowser(handle | undefined)`

Lifecycle:

- `init()` — register window listeners + key bindings (idempotent).
- `initDimensionSliders()` — call after scene load; wires sliders +
  animation manager + scene-dims listener.
- `clearDimensionUI()` — call before loading a new scene.
- `showDimensionSliders()` — applied by `viewer_config`.
- `getAnimationManager()` — used to restore per-dimension playback state.
- `dispose()` — clean up listeners and disposable refs.

## Adding new keyboard shortcuts

1. Add the binding to the appropriate file under
   `input-handler/key-bindings/`:
   - Global orbit-mode shortcut → `navigation-bindings.ts`.
   - Fly-mode movement key → `fly-bindings.ts`.
   - Animation playback shortcut → `animation-shortcuts.ts`.
2. Register its stable id in `key-bindings/actions.ts`, then provide the
   required `actionId`, `description`, and `help` fields on the binding. Use
   `help: { section, group, order }` for a single chord, add `keys` only for a
   grouped multi-chord row, or use `help: false` for an intentional opt-out.
3. Add the command to `KeyBindingsCommands` in `register-all.ts`.
4. Implement the command on `InputHandler` (typically a 1–3 line delegate
   into a helper under `commands/`).

## Adding a new optional panel

1. Add `setX(panel)` on the orchestrator.
2. If Escape should close it, forward into `panelCoordinator.setX(panel)`.
3. If a binding should open/close/toggle it, add a `getX` getter to
   `KeyBindingsPanelGetters` and reference it in the relevant
   `*-bindings.ts` file.

## Tests

Tests mirror the source layout under `tests/unit/input/`. The shape:

```
tests/unit/input/
├── input-handler-class.test.ts                     # Orchestrator class shape
├── input-handler-utilities.test.ts                 # Orchestrator helper behaviour
└── input-handler/
    ├── context-manager-class.test.ts
    ├── context-manager-keyup.test.ts
    ├── context-manager-recursion.test.ts
    ├── context-manager/
    │   └── routing-rules.test.ts
    ├── key-bindings/
    │   ├── register-all.test.ts
    │   └── animation-shortcuts.test.ts
    ├── dimension-navigation/
    │   ├── compute-step.test.ts
    │   └── setup.test.ts
    ├── window-events/
    │   ├── fullscreen-toggle.test.ts
    │   └── window-event-handler.test.ts
    └── commands/
        ├── control-mode.test.ts
        ├── data-monitor-cycle.test.ts
        ├── panel-coordinator.test.ts
        └── viewer-state-export.test.ts
```

Pure dimension math lives with scene dimension state under
`scene/dims/` (and `tests/unit/scene/dims/`). Pure DOM focus predicates live
under `utils/dom/` (and `tests/unit/utils/dom/`).
