# Cross-Layer Plumbing

Dependency-inversion seams that let lower viewer layers (`data`, `scene`, `input`) share small contracts and surface user-visible messages without importing UI modules directly. Four primitives sit here — a typed pub/sub bus, a single-backend notifier facade, a per-component DOM-listener group, and the shared slider modifier ladder — collectively enforcing the layer order documented in `CONVENTIONS.md` §10.

## Architecture

```
cross-layer/
├── event-bus.ts     # Typed pub/sub: many subscribers per event, payload-typed catalog
├── event-group.ts   # DOM-listener collection with single-call LIFO teardown
├── modifier-tiers.ts # Shift/Ctrl step multipliers shared by scene, input, and UI
└── notifier.ts       # Single-backend method dictionary (toast, error, help, loading)
```

The bus and the notifier are deliberately distinct: the **notifier** has one backend and a fixed method dictionary (the UI bootstrap calls `setNotifierBackend(...)` once); the **bus** has open subscriber sets typed against `LuxarEventMap` and lets panels subscribe late without bootstrap-order coupling. `EventGroup` is unrelated to either — it's a lifecycle helper that any component (UI or otherwise) uses to bundle its DOM listeners.

## Modules

### modifier-tiers.ts — Shared Step Ladder

`applyModifierTier(baseStep, modifiers, options?)` applies the common slider
law: no modifier ×1, Shift ÷10, Control ×10, and Control+Shift ÷100. It accepts
both DOM `shiftKey`/`ctrlKey` flags and the nD-navigation `shift`/`ctrl` shape so
lower layers can share the contract without importing `ui/`.

### event-bus.ts — Typed Pub/Sub

- `LuxarEventMap` — Catalog mapping event names to payload types:
  - Publisher events: `frame-start`, `frame-end`, `loading-progress`
  - Command events: `panel-toggle`, `panel-cycle`, `panel-hide`
- `TypedEventBus<EventMap>` — `on(type, listener, { replayLast? })`, `emit(type, payload)`, `clear(type?)`
- `eventBus` — Singleton typed against `LuxarEventMap`
- `createEventBus<EventMap>()` — Fresh bus for test isolation or per-app-instance use
- `Unsubscribe` — Idempotent thunk returned by `on(...)`

`emit` snapshots the listener set (`[...set]`) before iterating so a callback that subscribes or unsubscribes mid-emit cannot reorder the active loop. The last payload per event is cached on every emit so `on(..., { replayLast: true })` is O(1) — useful for late-binding panels that need the current value before the next emit.

### event-group.ts — Listener Group with One-Call Teardown

`EventGroup` replaces the error-prone "store a bound handler in a field, remember to remove it later" pattern with a registration helper that captures cleanup at add time.

- `on(target, type, listener, options?)` — Overloads for `Window`, `Document`, `HTMLElement`, `EventTarget`. Returns an early-cancel thunk that self-splices out of the cleanup list so long-lived groups don't accumulate no-op closures.
- `add(cleanup)` — Register an arbitrary teardown callback (e.g. `observer.disconnect()`, `cancelAnimationFrame(handle)`).
- `dispose()` — Run every cleanup in LIFO order; idempotent. A throwing cleanup logs via `log.error(Modules.EVENT_GROUP, ...)` and never stops siblings from running.
- `size` — Pending-cleanup count (for tests).

### notifier.ts — Single-Backend UI Facade

- `NotifierBackend` — Interface a concrete backend implements (`showError`, `showToast`, `showHelpOverlay`, `hideHelpOverlay`, `showLoadingIndicator`, `hideLoadingIndicator`, `clearError`).
- `notifier` — Stable call surface: `error`, `toast`, `showHelp`, `hideHelp`, `showLoading`, `hideLoading`, `clearError`. `error(message, { persistent: true })` asks the backend to suppress auto-dismissal. Pre-registration calls drop silently after a single one-time warn so unit tests and early-startup paths don't crash.
- `setNotifierBackend(b)` — Called once by the UI bootstrap with concrete `ui/` helpers; later calls replace the backend (useful for tests).
- `clearNotifierBackend()` — Tear down and reset the one-time missing-backend warning flag.

## Usage Examples

```typescript
import { eventBus } from '../utils/cross-layer/event-bus';

// Publisher (animation loop) — payload type-checked against LuxarEventMap
eventBus.emit('frame-start', {});

// Late-binding subscriber gets the cached payload immediately
const off = eventBus.on('loading-progress', (p) => updateBar(p), { replayLast: true });
off(); // later — idempotent
```

```typescript
import { EventGroup } from '../utils/cross-layer/event-group';

class Panel {
  private events = new EventGroup();
  attach() {
    this.events.on(window, 'focus', this.handleFocus);
    this.events.on(document, 'visibilitychange', this.handleVisibility);
    this.events.add(() => this.observer.disconnect());
  }
  dispose() {
    this.events.dispose(); // removes everything in LIFO order
  }
}
```

```typescript
import { notifier, setNotifierBackend } from '../utils/cross-layer/notifier';

// UI bootstrap (once)
setNotifierBackend({
  showError,
  showToast,
  showHelpOverlay,
  hideHelpOverlay,
  showLoadingIndicator,
  hideLoadingIndicator,
  clearError,
});

// Lower layers — does not import ui/ directly
notifier.toast('Recording saved', 3000);
notifier.error('Failed to load dataset');
```

## See Also

- `../README.md` — Parent `utils/` package overview
- `../log.ts` — `log` / `Modules` used by `EventGroup` and `notifier` for diagnostics
- `CONVENTIONS.md` §10 — Layer order this folder exists to preserve
