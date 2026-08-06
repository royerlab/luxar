# luxar-viewer Conventions

Cross-cutting conventions that apply to every subpackage. A
subpackage's `README.md` may override or extend these for its own
scope; the defaults below are what new code should follow unless there
is a documented reason not to.

## Table of contents

1. [File and module naming](#1-file-and-module-naming)
2. [Class and function naming](#2-class-and-function-naming)
3. [CSS class names (BEM)](#3-css-class-names-bem)
4. [Logging](#4-logging)
5. [Error handling](#5-error-handling)
6. [Resource lifecycle](#6-resource-lifecycle)
7. [Event listeners](#7-event-listeners)
8. [Result&lt;T, E&gt; for fallible operations](#8-resultt-e-for-fallible-operations)
9. [Worker safety](#9-worker-safety)
10. [Imports and barrels](#10-imports-and-barrels)
11. [Types](#11-types)
12. [Dependency inversion via ports and factories](#12-dependency-inversion-via-ports-and-factories)
13. [Error handling discipline](#13-error-handling-discipline)
14. [Resource disposal pattern](#14-resource-disposal-pattern)

---

## 1. File and module naming

- **Filenames**: kebab-case for source modules (`scene-loader.ts`,
  `material-manager.ts`). Class files match the class name in
  kebab-case (`PostProcessingManager` → `post-processing-manager.ts`).
- **Test files**: `<source-name>.test.ts` for unit tests, mirrored under
  `src/tests/unit/<area>/`. E2E tests use `.spec.ts` and live under
  `src/tests/e2e/`.
- **Index/barrel files**: only when a subpackage genuinely has a stable
  public surface (`config/index.ts`, `rendering/index.ts`). Internal
  scratch modules import directly from each other, not through a
  barrel, to avoid cyclic imports.
- **Setup-module pattern**: when decomposing a large facade, sibling
  modules export a `setupX(context)` function returning a
  `{ controllers, folders?, cleanup? }` shape. See
  `ui/rendering-controls/types.ts` for the shared shape.

## 2. Class and function naming

- **Classes**: `UpperCamelCase`. Singletons follow the `getInstance()`
  pattern with a private constructor.
- **Functions / methods**: `lowerCamelCase`. Async methods that fetch
  remote data are named for what they return, not the verb
  (`loadScene`, not `fetchAndParseScene`).
- **Constants**: `UPPER_SNAKE_CASE` for module-level immutable values
  (`MAX_NDIM`, `TARGET_CHUNK_BYTES`). `lowerCamelCase` for everything
  else, even when "conceptually constant" (config defaults,
  pre-computed lookup tables instantiated at runtime).
- **Private members**: `private` modifier, no underscore prefix.
- **Boolean predicates**: prefix with `is` / `has` / `should` (`isDisposed`,
  `hasTransform`).

## 3. CSS class names (BEM)

All viewer-owned class names start with `luxar-` to avoid host-page
collisions. Within that namespace, BEM applies:

```
luxar-block               // block
luxar-block__element       // element inside the block
luxar-block--modifier      // block-level modifier
luxar-block__element--state // element-level modifier
```

Examples:

- `luxar-layer-row` (block)
- `luxar-layer-row__eye` (element)
- `luxar-layer-row--selected` (modifier)
- `luxar-dimension-slider__thumb` (element)

Do **not** use Tailwind utility classes in component CSS. Themes are
CSS-variable driven (see `src/themes/README.md`); component
styles read those variables, never hard-coded values.

## 4. Logging

Channel everything through `src/utils/log.ts`:

```typescript
import { log, Modules } from '../utils/log';

log.info(Modules.RENDERER, 'Loaded scene with N points');
log.warning(Modules.CACHE, 'L2 quota exhausted, falling back to L1 only');
log.error(Modules.SCENE_LOADER, 'Failed to parse zarr metadata', err);
```

The `no-console` ESLint rule enforces this for production code. The
two carve-outs (`src/utils/log.ts` and
`src/utils/console-interceptor.ts`) are the legitimate `console.*`
sites. Tests, benchmarks, screenshot drivers, and mocks may use
`console.*` directly — they are tooling, not in-app code.

Output format is fixed: `[emoji] [Module] message`. Custom emojis go
through `log.custom(emoji, module, message)`.

## 5. Error handling

| Mechanism    | Use when                                                | Example                                                |
| ------------ | ------------------------------------------------------- | ------------------------------------------------------ |
| `throw`      | Unrecoverable invariant violation at JS boundary       | `validateNDArrays` rejecting a malformed buffer        |
| `Result<T,E>`| Recoverable with a typed error code                    | Cache miss vs network error vs corrupt vs aborted      |
| `log.warning`| Degraded behaviour, app continues                      | localStorage quota exceeded                            |
| `log.error`  | Unexpected failure, app continues but UX impacted      | WebGL context lost (with rebuild scheduled)           |

Avoid `throw` for "the network was slow" — that is a `Result<…>`.
Avoid `Result<…>` for "the input is structurally invalid" — that is a
`throw`. The boundary is whether the caller can plausibly *recover*.

## 6. Resource lifecycle

Long-lived objects that hold GPU resources, DOM listeners, workers, or
timers follow a uniform singleton + explicit-dispose pattern. Production
teardown is the explicit chain in `LuxarApp.dispose()` (see §14 for the
disposal rules each manager must implement).

```typescript
class FooManager {
  private static instance?: FooManager;

  static getInstance(): FooManager {
    if (!FooManager.instance) {
      FooManager.instance = new FooManager();
    }
    return FooManager.instance;
  }

  static disposeInstance(): void {
    FooManager.instance?.dispose();
    FooManager.instance = undefined;
  }

  dispose(): void { /* ... idempotent ... */ }
}
```

Disposal must be **idempotent**: calling `dispose()` twice is a no-op,
not an error. Use guards (`if (this.disposed) return;`) or check that
the resource still exists before tearing it down. New singletons get
their `disposeInstance()` call wired explicitly into
`LuxarApp.dispose()` so the cascade order stays auditable in one
place.

## 7. Event listeners

- For DOM listeners that should be torn down with the owner, use
  `EventGroup` (`src/utils/cross-layer/event-group.ts`):

  ```typescript
  private events = new EventGroup();

  setup() {
    this.events.on(window, 'resize', this.handleResize);
    this.events.on(document, 'visibilitychange', this.handleVisibility);
    this.events.add(() => this.observer.disconnect());
  }

  dispose() {
    this.events.dispose();
  }
  ```

- For per-frame event handlers that need fast lookup, cache the bound
  function reference at registration time. Don't `bind()` inline at
  `addEventListener` time — that creates a new reference every call
  and `removeEventListener` will silently fail.
- For listeners added inside a `setTimeout`, track the timer id so a
  pre-fire dispose can cancel the deferred install. See
  `ui/rendering-controls/focus-manager.ts` for the pattern.

## 8. Result&lt;T, E&gt; for fallible operations

`src/utils/result.ts` provides a discriminated union:

```typescript
import { ok, err, type Result, isOk, match } from '../utils/result';

async function loadChunk(key: string): Promise<Result<ArrayBuffer, CacheError>> {
  // ...
  if (!found) return err('Missing');
  if (corrupt) return err('Corrupt');
  return ok(buffer);
}

const r = await loadChunk('test');
if (isOk(r)) {
  use(r.value);
} else {
  // r.error is typed as CacheError
}
```

Helpers: `ok()`, `err()`, `isOk()`, `isErr()`, `match()`, `mapOk()`,
`mapErr()`, `unwrap()` (throws on error — use sparingly), `unwrapOr()`,
`tryAsync()` (wraps an async fn into Result).

## 9. Worker safety

Workers have no DOM, no `window`, and run independent of the main
thread. Conventions:

- **Always validate inputs at the JS boundary**: every public entry
  point in `data-worker.ts` calls a `validate*` helper before crossing
  into WASM.
- **Always set `onerror` and `onmessageerror`** on Workers you create
  — silent failures otherwise become "the worker just stopped".
- **Use `WorkerPool.withTimeout()`** for any RPC that could hang.
  Default timeouts come from
  `config.dataLoading.performance.workerVisibilityTimeoutMs` and
  `workerProjectionTimeoutMs`.

## 10. Imports and barrels

- Prefer **direct imports** from sibling modules over a barrel re-export.
  Barrels are reserved for genuine public surfaces (cross-package
  consumers).
- **No circular imports**. The `tsc --noEmit` build catches structural
  cycles; the `dependency-cruiser` rule catches layer-crossing cycles
  (severity `error`).
- **Type-only imports** use `import type` — keeps emission lean and
  makes the intent obvious.
- **Layer order** (see `.dependency-cruiser.cjs`):

      types → config → cache → rendering → data → scene → input → ui → core

  Each layer may import from layers to its **left**. Cross-cutting
  helpers (`utils/`, `themes/`, `wasm/`, `workers/`, `profiling/`,
  `controls/`) may be imported anywhere. Type-only imports are exempt
  — they're erased at compile time.

  Note: `rendering` sits *below* `data` because rendering primitives
  (materials, geometries, GPU buffer pools) are foundational
  building blocks that the data layer assembles into meshes. This
  order matches the actual dependency direction in the codebase.

  Run `pnpm check:layers` to surface violations. **All layer rules
  are at severity `error`** — any new crossing fails the build. The
  `KNOWN_LAYER_EXCEPTIONS` array in `.dependency-cruiser.cjs` should
  stay empty unless a reviewed exception includes an owner, a narrow
  scope, and a removal condition.

## 11. Types

- `any` is allowed only with an inline `// eslint-disable-next-line
  @typescript-eslint/no-explicit-any` and a justification comment.
  See `eslint.config.js` for which directories enforce the rule.
- Prefer `unknown` over `any` when typing callback args / external
  data. Narrow with type guards.
- Mark function parameters `readonly` whenever the function does not
  mutate them. The viewer's nD arrays are typed as `readonly number[]`
  on every entry point.
- Use `Result<T, E>` (see §8) before reaching for `throw` for
  recoverable failures.
- Augment third-party types in `src/types/*-augmentation.d.ts` rather
  than spreading `as any` casts across call sites.

## 12. Dependency inversion via ports and factories

The layer order `types → config → cache → rendering → data → scene →
input → ui → core` (see `.dependency-cruiser.cjs`) is enforced at
`severity: error`. When a lower layer needs behavior that lives in a
higher layer — typically because the higher layer owns DOM / WebGL /
THREE state — invert the dependency:

1. Declare a **port interface** in the lower layer describing the
   shape the lower layer needs.
2. Implement the port structurally in the higher layer (no `implements`
   keyword needed; TypeScript matches by shape).
3. Inject a **factory** from the orchestration layer
   (`src/core/app.ts`).

Established examples in this repo:

- `SceneLoaderMonitorPort` (`src/data/scene-loader-monitor-port.ts`) —
  the `SceneLoader` (data layer) pushes loader telemetry into the
  `DataLoadingMonitor` (ui layer) through this port. The factory is
  passed into `SceneLoaderManager.createLoader` from `core/app.ts`.
- `DimensionSlidersFactory` (`src/input/input-handler.ts`) — the input
  layer needs to mount sliders that live in ui/panels. The factory is
  injected from `core/app.ts`.
- `LabelTooltipFactory` (`src/rendering/picking/picking-system.ts`) —
  the rendering layer needs a ui tooltip element; the factory lives in
  `core/app.ts`.

Do NOT add the higher-layer module to the dependency-cruiser allowlist
to "fix" a layer violation — that defeats the layer discipline. Always
prefer a port + factory pair.

## 13. Error handling discipline

- **Throw directly** in the unexpected case:
  `throw new Error('[Module] context: details');`. The caller's catch
  block is responsible for logging.
- **Log before throw only at async / cross-boundary edges** where the
  error frame would otherwise be lost. Established sites:
  - `src/workers/data-worker.ts` (worker initialization, before
    re-throwing back across the comlink boundary).
  - `src/data/scene-loader.ts` leaf-node loads — `loadLeafNode` catches
    per-node failures, classifies them, logs with the appropriate
    severity, and **returns null** so sibling nodes still render. It
    does not rethrow. It also does **not** notify the user: since
    `loadScene` cannot throw on a failed node, the end-of-load aggregate
    (`scene-loader/loaders/failure-report.ts`) owns that, and it is the
    only surface that can name every failure at once.
- Use `Result<T, E>` (see §8) for recoverable failures where the caller
  is expected to branch on the outcome (cache lookups, optional
  resolves), not for unexpected programmer errors.
- Never swallow an error silently. If a `catch` truly has nothing to
  do, log at `info` level with a one-line justification (e.g. an
  already-disposed terminate that the browser quirks on).

## 14. Resource disposal pattern

Every component that owns external resources (DOM nodes, event
listeners, WebGL buffers, timers, workers, OPFS handles) must implement
`dispose()` and follow these rules:

- **Idempotent**: `dispose()` is safe to call twice. Use a
  `private disposed = false;` field plus an early return:
  ```ts
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    // ... actual teardown ...
  }
  ```
- **Defensive on internal state**: guard each step with `if (this.x)`
  or optional chaining. Disposal often races with construction failures
  where some fields were never set.
- **Top-down order**: dispose children first, then null out parent
  references. The reverse can leave a child holding a stale parent
  pointer and re-entering the disposed parent during its own teardown.
- **Event listeners go through `EventGroup`** (`src/utils/cross-layer/event-group.ts`).
  Established consumers: `LayersPanel`, `InputHandler`, `SceneManager`.
  Manual `addEventListener` / `removeEventListener` pairs are the
  number-one source of leak regressions; the group ties listener
  installation to disposal in one place.
- **Async disposal awaits its dependencies**. `SceneLoader.dispose()`
  awaits `Promise.all` over its per-loader dispose chain so the next
  `loadScene()` cannot see partially-torn-down caches; do the same
  whenever a `dispose` triggers async I/O (e.g. OPFS).
