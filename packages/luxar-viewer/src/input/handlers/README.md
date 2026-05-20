# Input Handlers

> Per-concern handlers extracted from `input/input-handler.ts` so the
> orchestrator stays small and each concern is independently testable.

## Overview

`InputHandler` (in the parent folder) is a thin coordinator. The actual
work of registering keyboard bindings, gating Ctrl/Meta-held FOV control,
running animation shortcuts, computing dimension-navigation math,
coordinating Escape-key panel closes, and routing window-level events
lives here — one file per concern.

The split is deliberately **mechanical, not behavioral**: every file in
this folder was lifted out of `InputHandler` byte-for-byte (same key
codes, same contexts, same log emojis, same `preventDefault` rules).
Extracting them lets the math be unit-tested in isolation and lets the
host class read as a wiring sketch instead of a 1500-line god object.

Two flavors of file live side-by-side:

- **Pure helpers** (`control-mode-cycle.ts`, `dimension-navigation.ts`,
  `focus-utils.ts`) — stateless functions taking primitives in and
  returning results out. No DOM, no side effects, no managers. Trivial
  to test.
- **Concern classes / wirers** (`animation-shortcuts.ts`,
  `key-bindings.ts`, `panel-coordinator.ts`, `window-event-handler.ts`)
  — own a slice of `InputHandler`'s registration logic plus the
  late-bound references they need. Constructed by `InputHandler`,
  receive command/getter callbacks so they can read live state at
  dispatch time without holding direct refs.

## File Structure

```
handlers/
├── animation-shortcuts.ts   # K, Home, End, Shift+↑/↓ dim-animation bindings
├── control-mode-cycle.ts    # Pure: orbit → fly → ortho → orbit
├── dimension-navigation.ts  # Pure: step math + selectedDimension resolution
├── focus-utils.ts           # Pure: isTypingInInput / isFocusOnSceneCanvas
├── key-bindings.ts          # Full NAVIGATION + FLY_CONTROLS binding table
├── panel-coordinator.ts     # Escape-key + closeAll() priority order
└── window-event-handler.ts  # resize / wheel / fullscreenchange listeners
```

---

## Modules

### `animation-shortcuts.ts` — `AnimationShortcuts`

Registers the five NAVIGATION-context bindings that drive
`DimensionAnimationManager` against the currently selected dimension:

| Key       | Action                                                |
| --------- | ----------------------------------------------------- |
| `K`       | Toggle play/pause                                     |
| `Home`    | Jump to the first frame                               |
| `End`     | Jump to the last frame                                |
| `Shift+↑` | Increase animation FPS                                |
| `Shift+↓` | Decrease animation FPS                                |

Both `selectedDimension` and `animationManager` are read through getter
callbacks (`AnimationShortcutsContext`) because they mutate after the
bindings are registered — the selected dim changes as the user presses
`1–9`, and the animation manager is constructed lazily by
`InputHandler.initAnimationManager()`. Each binding short-circuits when
no dim is selected (`getSelectedDimensionIndex` returns `-1`) or when
the animation manager isn't built yet — both checks live in the handler
bodies so the bindings can be registered before scene load.

### `control-mode-cycle.ts` — `nextControlType`

Pure cycle helper for the `V` key:

```ts
nextControlType('orbit')  // 'fly'
nextControlType('fly')    // 'ortho'
nextControlType('ortho')  // 'orbit'
nextControlType(<other>)  // 'orbit'  (defensive fallback)
```

The full toggle (which also mutates `SceneManager`,
`InputContextManager`, and `RenderingControls`) stays in
`input-handler.ts`; this helper exists so the cycle order is testable
without spinning up SceneManager + OrbitControls.

### `dimension-navigation.ts` — pure step math

Two helpers:

- **`computeDimensionStep(direction, selectedDim, dims, ranges)`** —
  given the navigable-dimension list, the selected slot, and a `+1/-1`
  direction, returns `{targetDim, newValue, changed}`. `changed` is
  `true` only when the new value differs from the current by `> 1e-6`
  (matches the inline original's "movement big enough to requery the
  spatial index" threshold). Handles cyclic wrap-around, discrete
  rounding, and `[min,max]` clamping. Returns `null` when navigation
  can't proceed (no dims, no ranges, no navigable dims).
- **`getSelectedDimensionIndex(selectedDim, dims)`** — translates a
  0-based slot into the actual `targetDim` index, returning `-1` for
  every "no resolvable dim" case (negative slot, no scene, slot past
  the navigable count). Shared between `AnimationShortcuts` and
  `InputHandler.selectDimension` so the "what does `selectedDimension`
  mean" rule lives in one place.
- **`resolveSelectedDimension(keyIndex, dims)`** — used by the `1–9`
  key bindings to translate a 0-based key index into the new
  `selectedDimension` value. Returns `null` plus the navigable count
  when the key falls past the available dimensions so the caller can
  log a "dimension N not available (only M navigable)" hint.

### `focus-utils.ts` — DOM focus classification

Two stateless DOM predicates that take an `Element | null` in (so tests
control the focus context without `element.focus()` calls):

- **`isTypingInInput(activeElement)`** — `true` when focus is on a
  typing surface (text input, textarea, select, contenteditable).
  Returns `false` for `<input type="range|checkbox|radio">` so
  navigation keys still fire after the user clicks a slider.
- **`isFocusOnSceneCanvas(activeElement, canvas)`** — `true` when focus
  is on the WebGL canvas or `document.body`. Used to gate global keys
  like Space (fullscreen toggle) so they don't fire when focus has
  landed on a panel button.

### `key-bindings.ts` — `registerAllKeyBindings(deps)`

The big one — the entire key→command table. Three sub-flows in order:

1. **FOV-hold gate** — `Control` and `Meta` register
   keydown/keyup handlers that increment / decrement a counter and call
   `controls.setEnableZoom(false|true)`. A counter (not a boolean) gates
   the toggle so releasing one modifier while the other is still held
   doesn't re-enable wheel zoom mid-gesture. `window blur` and
   `document visibilitychange` listeners reset the counter when the
   page loses focus — otherwise a key release that happens off-page
   would leave zoom permanently disabled. The cleanup thunks are
   pushed onto the InputHandler's shared `cleanups` array.
2. **NAVIGATION bindings** — every default-mode application shortcut:
   `[/]` and `1–9` for dimension navigation, `H/N/O/P/R/B/T/G/M/F/V/I/C`
   for panels/modes, `Ctrl+L` for the debug console, `Ctrl+Shift+S` to
   export viewer state, `Space` for fullscreen (context-aware), and
   `Escape` for `handleEscape()`. Optional panels (scale bar, colormap
   legend, overlay manager, recording panel, layers panel) are read
   through `KeyBindingsPanelGetters` so the bindings always see the
   live reference — `InputHandler` may wire them in after this
   function runs.
3. **FLY_CONTROLS bindings** — WASD/Q/E movement keys registered with
   all four modifier combinations the fly controls respect (none,
   Shift, Alt, Shift+Alt), arrow keys with optional Shift for look
   direction, and the Shift speed-boost binding. Each movement key has
   both `handler` (keydown) and `keyupHandler` because fly controls
   hold the key state themselves and integrate movement per-frame.

The `KeyBindingsDeps` shape:

```ts
{
  contextManager: InputContextManager,
  sceneManager: SceneManager,
  debugConsole: DebugConsole,
  cleanups: (() => void)[],
  panels:  KeyBindingsPanelGetters,
  commands: KeyBindingsCommands,
}
```

`KeyBindingsCommands` is the surface the InputHandler still owns —
`navigateDimension`, `selectDimension`, `toggleHelp`,
`toggleControlMode`, `toggleFullscreen`, `handleEscape`, etc.

### `panel-coordinator.ts` — `PanelCoordinator`

Owns the priority-ordered "close everything" flow used by `Escape`.
The InputHandler holds a single coordinator instance constructed with
the always-present panels (`debugConsole`, `performanceStats`) and uses
setters for the panels that are wired in lazily
(`setRenderingControls` / `setDimensionSliders` /
`setRecordingPanel` / `setDatasetBrowser` / `setLayersPanel`).

`closeAll()` priority order (topmost first, byte-for-byte from the
inline original):

1. Help overlay (`notifier.hideHelp()`)
2. Error toast (`notifier.clearError()`)
3. Dataset browser (`close()` so the panel's `onClose` clears the
   owner's reference — otherwise the `O` reopen shortcut sees a
   dangling ref)
4. Rendering controls
5. Data loading monitor (`eventBus.emit('panel-hide', ...)`)
6. Dimension sliders
7. Debug console
8. Recording panel
9. Layers panel
10. Performance stats

`handleEscape()` adds two context-aware short-circuits on top of
`closeAll()`:

- If the recording panel is currently recording, stop the recording
  and return — recording takes priority over panel close.
- If we're in fullscreen, do nothing — the browser handles Escape
  natively.

The `CloseableHandle` and `VisiblyHideableHandle` interfaces let
`panel-coordinator.ts` avoid hard-importing the full `DatasetBrowser`
and `LayersPanel` types — it only needs the close / visibility surface
they expose.

### `window-event-handler.ts` — `WindowEventHandler`

Registers the three window/document-level listeners and pushes their
unregistration thunks onto the InputHandler's shared cleanup array via
`attach(cleanups)`:

- **`resize`** (window) — forward to `SceneManager.updateSize()` and
  kick the animation loop so the resized scene renders.
- **`wheel`** (window, `{ passive: false }`) — orbit/ortho controls own
  the actual zoom math via canvas-local listeners; this handler only
  intercepts Ctrl/Meta+wheel for FOV control and switches the
  rendering-controls preset to "Custom" so the slider value matches.
  The `{ passive: false }` is load-bearing — browsers can default
  root-target wheel listeners to passive, in which case
  `preventDefault()` is silently ignored and the page zooms while the
  FOV also changes.
- **`fullscreenchange`** (document) — toggle fullscreen-fitting inline
  styles on the canvas (`100vw`/`100vh`/`fixed`/`top:0`/`left:0` on
  enter, drop the `style` attribute entirely on exit), then run
  `updateSize()` once on the next `requestAnimationFrame`. The single
  rAF is intentional: modern browsers fire `fullscreenchange` *after*
  the viewport transition completes, so one frame is enough to capture
  final dimensions.

`setRenderingControls(rc)` exists for late wiring — the InputHandler
may receive its rendering-controls reference after the window listeners
are already attached.

---

## Wiring contract

Everything in this folder is owned and constructed by
`InputHandler`. The expected lifecycle:

1. `InputHandler` constructs `PanelCoordinator` (always present
   panels), `WindowEventHandler` (sceneManager + animationController),
   and the cleanups array.
2. `InputHandler.init()` calls `windowEventHandler.attach(cleanups)`
   and `registerAllKeyBindings({...})`, both of which push their
   unregistration thunks onto the shared `cleanups`.
3. `InputHandler.initAnimationManager()` constructs `AnimationShortcuts`
   and calls `register()` once the animation manager is built.
4. Lazy panel setters on `InputHandler`
   (`setRenderingControls` / `setRecordingPanel` / `setDatasetBrowser`
   / `setLayersPanel`) forward to the coordinator's setters and to
   `WindowEventHandler.setRenderingControls()`.
5. `InputHandler.dispose()` runs `cleanups`, which detaches every
   listener these handlers registered.

There is no separate `dispose()` on any handler in this folder — the
cleanups array is the single source of truth.

## See Also

- `../README.md` — the parent input package documentation, including
  the three-layer architecture rule (viewer-wide / camera-motion /
  UI-local) that determines whether new input belongs in this folder
  at all.
- `../input-handler.ts` — the orchestrator that owns and wires these
  handlers together.
- `../input-context-manager.ts` — the keyboard dispatcher every
  binding in `key-bindings.ts` registers with.
- `../input-handler-utils.ts` — the pure dimension / FOV /
  shortcut-blocking helpers that `dimension-navigation.ts` builds on.
