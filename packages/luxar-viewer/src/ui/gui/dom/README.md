# GUI DOM Helpers

DOM-side plumbing for the custom GUI library. Currently houses a single
helper — the centralized `EventManager` — that every controller relies
on for guaranteed listener cleanup.

## Files

- `event-manager.ts` — `EventManager` class. Wraps `addEventListener` /
  `removeEventListener` and records every registered listener so they can
  be torn down in bulk on `dispose()`.

## EventManager

Each `Controller<T>` owns an `EventManager` instance. Controllers never
call `addEventListener` directly; they go through `eventManager.add(...)`
so that `Controller.dispose()` can call `eventManager.removeAll()` and
guarantee zero leaked listeners.

### API

| Method                                   | Purpose                                                                                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `add(element, event, handler, options?)` | Register a listener on `HTMLElement`, `Window`, or `Document` and store an `EventListenerRecord` (from `../types.ts`).                    |
| `remove(element, event, handler)`        | Remove a specific listener; looks up the stored record so the original `options` (notably `capture`) are passed to `removeEventListener`. |
| `removeAll()`                            | Remove every tracked listener. **Must** be called on dispose.                                                                             |
| `count()`                                | Number of currently-tracked listeners (debugging / tests).                                                                                |

### Why preserve `options` on removal

`addEventListener`'s `useCapture` flag is part of the listener's
identity. A capture-phase listener registered with `{ capture: true }`
will leak if removed without the same option. `EventManager.remove()`
looks up the stored record to forward the original options to
`removeEventListener`, so callers never have to repeat them.

### Usage

```typescript
import { EventManager } from './dom/event-manager';

const manager = new EventManager();
manager.add(slider, 'input', onInput);
manager.add(slider, 'change', onChange);
// ... later, on dispose:
manager.removeAll();
```

## See Also

- [`../README.md`](../README.md) — Custom GUI library overview.
- [`../controller.ts`](../controller.ts) — Abstract `Controller<T>`
  base; owns the per-controller `EventManager`.
- [`../controllers/`](./../controllers/README.md) — Concrete controllers,
  all of which route DOM events through this manager.
- [`../format/auto-blur.ts`](../format/auto-blur.ts) — Consumes the
  shared `EventManager` to register its blur triggers.
