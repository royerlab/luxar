# ui/control-rail

> Always-visible discoverability rail — the one visual entry point to Luxar's keyboard-driven UI

## Purpose

The control rail is a slim, vertical activity rail docked to the left edge of the viewer canvas. Luxar's panels are otherwise entirely keyboard-triggered (H, N, R, L, P, etc.), which means a first-time visitor sees a bare canvas with no hint that controls exist. The rail solves that: one recognizable icon per panel, each with a tooltip showing its keyboard shortcut.

**Design principle: no behavioral drift.** Every rail button fires the _exact same_ command its keyboard shortcut triggers (via `InputHandler.getUiActions()`), so the rail never re-implements panel logic. It is purely a **visual discoverability layer** over the existing keyboard-driven architecture.

## File Map

```
ui/
├── control-rail.ts         — ControlRail class: buttons, idle/wake, collapse/fullscreen, hint
└── control-rail/
    ├── rail-overlay.ts     — RailOverlay: flyout + panel-popover lifecycle (one-open-at-a-time)
    ├── types.ts            — ControlRailItem, ControlRailPopover, ControlRailToggle interfaces
    ├── dom-helpers.ts      — isPanelVisible (visibility check), escapeHtml re-export (tooltip safety)
    └── icons.ts            — RAIL_ICONS: inline SVG map (currentColor stroke, 24×24)
```

## Architecture

### ControlRail (control-rail.ts)

The main orchestrator. Owns:

- **Button construction** — one button per `ControlRailItem` (passed at construction)
- **Active-state refresh** — event-driven (click/routed keydown/luxar-layers-changed/luxar-control-mode-changed), rAF-debounced, reads each item's `isActive()` or `openSelector` to highlight open panels
- **Idle-dim behavior** — wakes on pointer movement (expanded) or hover (collapsed/fullscreen); schedules sleep after `IDLE_MS` (2600ms) unless `:hover` or `:focus-within`
- **Collapse/expand** — chevron handle at the bottom; persisted to localStorage
- **Fullscreen sync** — hides the rail (hover-to-reveal) when `document.fullscreenElement` exists
- **First-run hint** — localStorage-gated one-time nudge that says "Hover these controls" when hover is available and "Tap these controls (hold for options)" otherwise; auto-fades after 10s and dismisses on any click or handled routed keypress
- **Docked footer** — optional element (e.g. the performance readout) inserted above the collapse handle
- **Overlay delegation** — opens flyouts/popovers via `RailOverlay` and re-syncs active-state when the overlay changes
- **Routed-keydown reception** — `InputHandler` calls `handleRoutedKeyDown()` after it handles a key; dismisses the hint and schedules a refresh

**Responsibilities**:

- Construct buttons from `ControlRailItem[]`
- Reflect live panel open/closed state (via `refresh()`)
- Manage idle/wake/collapse/fullscreen/hint behaviors
- Blur buttons after pointer clicks so canvas/body shortcuts (e.g. Space = fullscreen) keep working
- Reference-count the global `luxar-has-control-rail` body marker class (left-anchored panels offset to clear the rail)
- Expose `closeOverlay()` for `PanelCoordinator` and `handleRoutedKeyDown()` for post-routing hint dismissal/refresh

**Does NOT**:

- Own panel logic (delegates to each item's `activate()` callback)
- Own flyout/popover DOM (delegated to `RailOverlay`)
- Own Escape routing (`PanelCoordinator` decides when overlays close)

### RailOverlay (rail-overlay.ts)

The flyout + panel-popover lifecycle coordinator. Owns:

- **One-open-at-a-time enforcement** — `this.current` holds the open overlay (flyout OR popover) + its DOM; opening a new one closes the previous
- **Flyout construction** — horizontal row of `ControlRailToggle` chips (e.g. View options: scale bar, legend, overlays, cinematic, and fullscreen when the Fullscreen API is available)
- **Popover construction** — vertical panel hosting arbitrary rich controls built by `item.popover.build(host)` (lazy, rebuilt on each open; teardown callback run on close)
- **Positioning** — flyouts align vertically with their button; popovers anchor near their button and clamp inside the viewport
- **Focus return** — when `PanelCoordinator` closes the overlay for Escape (focus inside it), returns focus to the opener button
- **Outside-click dismissal** — via `maybeCloseOnPointer(e)` (called by ControlRail's document pointerdown listener)
- **Viewport-change sync** — flyout tooltip flip (--up modifier) recalculates on resize/fullscreenchange while open

**Public surface** (used by ControlRail):

- `activeItem` — currently-open item (undefined when none)
- `toggleFlyout(item, btn)` / `togglePopover(item, btn)` — open/close entry points
- `close()` — unconditional close
- `anyToggleActive(item)` — whether any flyout toggle is active (parent button highlight)
- `refreshChips(item)` — re-sync chip active-state for the open flyout
- `maybeCloseOnPointer(e)` — close on outside click

### Supported Item Types (types.ts)

**`ControlRailItem`** — one button in the rail. Can be:

1. **Momentary action** (`momentary: true`) — e.g. Home (fit scene); never shows active state
2. **Panel toggle** (`activate()` opens/closes a panel) — active when `isActive()` returns true or `openSelector` element is visible; e.g. Help (H), Dimensions (N), Rendering (R), Layers (L)
3. **Flyout** (`flyout: ControlRailToggle[]`) — e.g. View button; opens a horizontal row of icon toggles; parent button is active when the flyout is open OR any non-`excludeFromParentActive` toggle is on
4. **Popover** (`popover: ControlRailPopover`) — e.g. Navigation, Settings, Performance; opens a vertical panel with arbitrary controls built by `popover.build(host)`:
   - `trigger: 'click'` — primary click opens the popover; `activate()` is unused
   - `trigger: 'context'` — right-click opens the popover; primary click still fires `activate()` (e.g. Performance: left-click = toggle readout, right-click = open DPR settings)

**`ControlRailToggle`** — a compact icon toggle inside a flyout (e.g. cinematic mode, overlays, and fullscreen when the Fullscreen API is available). Each has its own `activate()`, `isActive()`, and optional `excludeFromParentActive` (e.g. fullscreen is ambient, not a signal).

**`ControlRailPopover`** — a lazily-built rich panel. `build(host: HTMLElement)` populates the host each time the popover opens (fresh state) and may return a teardown callback (e.g. dispose a GUI, clear intervals).

### DOM Helpers (dom-helpers.ts)

- **`isPanelVisible(selector, scope)`** — true when the element exists, has `getClientRects().length > 0`, and has `visibility !== 'hidden'` + `opacity > 0.01` (the active-state check for `openSelector` items)
- **`escapeHtml(s)`** — re-exports the canonical `escapeHtml`, escaping `&<>"'` (the single-quote superset) for safe tooltip `innerHTML` interpolation

### Icons (icons.ts)

`RAIL_ICONS: Record<string, string>` — inline SVG map. Every icon:

- 24×24 viewBox
- `currentColor` stroke (theme-aware)
- `aria-hidden="true"` (decorative)
- No fill, no transform (simplicity)

Icons are keyed by id (help, home, dims, render, layers, perf, data, monitor, recording, logs, view, settings, hidePanels, plus navigation modes: navOrbit, navFly, navOrtho; view-flyout actions: scalebar, legend, overlays, cinematic, fullscreen; and Home-popover actions: fit, origin). The map also carries a `screenshot` icon that no current rail item uses.

## Lifecycle & Invariants

### Construction

```typescript
const rail = new ControlRail(items: ControlRailItem[], footer?: HTMLElement);
```

1. Constructs one `<button>` per item (via `buildButton`), appends to `this.root`
2. Constructs the collapse handle (appended last), loads persisted collapsed state from localStorage
3. Optionally appends the `footer` element (e.g. performance readout) above the collapse handle
4. Constructs the `RailOverlay` (flyout/popover lifecycle)
5. Wires event listeners:
   - `container.pointermove` → wake (expanded only)
   - `root.pointerenter` → wake (always)
   - `root.focusin` → wake (keyboard focus entering)
   - `document.fullscreenchange` / `webkitfullscreenchange` → `syncFullscreen()`
   - `document.pointerdown` (capture) → dismiss hint + `overlay.maybeCloseOnPointer(e)`
   - `document.click` → schedule refresh (active-state may have changed)
   - `window.luxar-layers-changed` → schedule refresh (Layers disabled state)
   - `window.luxar-control-mode-changed` → schedule refresh (Navigation icon/tooltip)
6. Shows the first-run hint (localStorage-gated)
7. Increments `bodyMarkerRefs` and adds `luxar-has-control-rail` to `document.body`

### Refresh Cycle (Event-Driven)

**Triggers**: document click, handled routed keydown, `luxar-layers-changed`, `luxar-control-mode-changed`, overlay open/close, fullscreen change.

**Flow** (rAF-debounced via `scheduleRefresh()`):

1. For each item:
   - Call `item.render(btn)` (if present) to sync dynamic icon/label (e.g. Navigation mode)
   - Evaluate `item.disabled()` (if present); set native `disabled` attribute when true
   - Skip active-state for disabled/momentary items
2. **Flyout items**: active when the flyout is open OR any non-`excludeFromParentActive` toggle is on; also refresh each chip's own active-state
3. **Popover items**: active when `isActive()` is true OR the popover is open
4. **Panel-toggle items**: active when `isItemActive(item)` is true (via `isActive()` or `isPanelVisible(openSelector)`)

### Active-State Sources

1. **`item.isActive()`** — explicit predicate (e.g. Performance readout visibility)
2. **`item.openSelector`** — CSS selector; active when `isPanelVisible(openSelector, container)` is true
3. **Flyout open** — the flyout button is active while its overlay is visible
4. **Popover open** — the popover button is active while its overlay is visible
5. **Flyout toggle state** — a flyout's parent button is active when any of its non-`excludeFromParentActive` toggles are on

### Idle/Wake Behavior

- **Idle-dim** — `.is-awake` class removed after `IDLE_MS` (2600ms) of no pointer movement (expanded) or `COLLAPSED_IDLE_MS` (5000ms, collapsed)
- **Wake** — `.is-awake` class added on:
  - Pointer movement (expanded only; collapsed/fullscreen wake on hover only)
  - Pointer entering the rail (`root.pointerenter`)
  - Keyboard focus entering the rail (`root.focusin`)
  - Overlay open
  - Collapse/expand
  - Fullscreen enter/exit
- **Persistent wake** — the sleep timer re-schedules itself when `:hover` or `:focus-within` is active (so a focused/hovered control never dims)

### Collapse/Expand

- Toggled via the chevron handle at the bottom
- Persisted to localStorage (`COLLAPSED_STORAGE_KEY`)
- Closes any open overlay on collapse
- Dismisses the first-run hint on collapse

### Fullscreen Sync

- `.is-fullscreen` class toggles on `fullscreenchange` / `webkitfullscreenchange`
- When fullscreen, the rail hides (`opacity: 0`) and only reveals on hover
- Exits fullscreen restore normal idle/wake behavior
- Triggers an active-state refresh (View-options fullscreen chip)

### First-Run Hint

- One-time localStorage-gated hint that names hover when available and tap/hold otherwise
- Auto-fades after 10 seconds (`HINT_AUTO_HIDE_MS`)
- Dismisses immediately on any click/keypress
- Never shown when starting collapsed or when already dismissed

### Disposal

1. Cancel all timers (`idleTimer`, `hintAutoHideTimer`, `hintFadeTimer`, `refreshRaf`)
2. Dispose `overlay` (closes, removes listeners)
3. Remove all document/window listeners
4. Decrement `bodyMarkerRefs`; if zero, remove `luxar-has-control-rail` from `document.body`
5. Remove hint + root from DOM
6. Clear `buttons` map

**Disposal is safe to call multiple times** (guarded by `this.disposed`).

## Integration Points

### Construction (wired in core/app/init/pipeline.ts)

```typescript
const railItems = buildRailItems(deps); // builds ControlRailItem[]
const perfReadout = perfMonitor.element; // optional footer
const controlRail = new ControlRail(railItems, perfReadout);
// Appends to viewer container; self-wires events; shows hint
```

### Item Builder (core/app/init/build-rail-items.ts)

`buildRailItems(deps: RailItemsDeps): ControlRailItem[]` constructs the item array from `deps.ui` (the pre-resolved `InputHandler.getUiActions()` command surface the keyboard shortcuts dispatch into). Each button's `activate()` calls the same action method the key binding invokes.

### External State Changes

Fire custom events to trigger a refresh (wired in the relevant modules):

```typescript
// After layer population change (layers panel or scene loader)
window.dispatchEvent(new Event('luxar-layers-changed'));

// After control-mode switch (orbit/fly/ortho; keyboard V, rail Navigation popover, or cycle)
window.dispatchEvent(new Event('luxar-control-mode-changed'));
```

### Panel Visibility Detection

- **Via `openSelector`** — the rail checks `isPanelVisible(item.openSelector, container)` each refresh
- **Via `isActive()`** — the rail calls `item.isActive()` each refresh (e.g. Performance readout)

### Flyout / Popover Builders

- **Flyout** — array of `ControlRailToggle` (e.g. View options: scale bar, legend, overlays, cinematic, fullscreen)
- **Popover** — `ControlRailPopover.build(host)` is called each time the popover opens; return an optional teardown callback. See `ui/rail-panels/` for examples (Settings, Performance, Navigation, Home).

## See Also

- **`ui/rail-panels/README.md`** — Rich control popovers hosted by the rail (Settings, Performance, Navigation, Home)
- **`ui/README.md`** — Parent UI package overview (control rail is section §0)
- **`core/app/init/build-rail-items.ts`** — Builds rail descriptors from the same UI actions used by keyboard shortcuts
- **`styles/components/control-rail.css`** — Rail styles (glass surface, idle-dim, collapse, fullscreen, hint animations)
- **Depth-sorting Phase 3** — The rail's Navigation button icon swaps to reflect the current control mode (orbit/fly/ortho); see `ui/rail-panels/navigation-popover.ts`
