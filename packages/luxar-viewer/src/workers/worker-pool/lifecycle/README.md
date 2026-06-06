# Worker-pool lifecycle

Per-worker spawn, init-race, and failure-eviction helpers extracted from
`WorkerPool`. These primitives turn a freshly-constructed `Worker` into
a tracked `WorkerInstance` and remove it from the pool when it crashes
— without baking the pool's private state (generation token,
`initPromise`, `workers` array) into helper code.

The pool itself (`../../worker-pool.ts`) still owns the generation
guard, the cached init promise, and the `workers` list. These helpers
are pure functions over their parameters: they take the worker, the
list, the callbacks, and return.

## Files

```
lifecycle/
├── worker-count.ts        # Resolve config'd worker count, cap at hardwareConcurrency - 1
├── spawn-worker.ts        # Per-worker factory + stale-generation terminate helper
├── init-with-guard.ts     # Race api.initialize() vs timeout + worker onerror
└── error-handlers.ts      # Permanent onerror/onmessageerror + evictFailedWorker
```

| File                 | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `worker-count.ts`    | `getConfiguredWorkerCount(configCount)` — `0` means auto (`hardwareConcurrency - 1`); a positive value is capped at the same max. Falls back to 4 cores when `navigator.hardwareConcurrency` is unavailable. Always reserves one core for the main thread.                                                                                                                                                                                                                                                                                                                                                                       |
| `spawn-worker.ts`    | `spawnWorker(opts)` — constructs a `Worker` (using `urlOverride` or the default Vite `?worker` ctor), tracks it in `pendingWorkers` so a concurrent `dispose()` can terminate it, installs permanent error handlers **before** the first message, then runs the init guard. On success returns a `WorkerInstance`; if `isCurrentGeneration()` flips false during init, self-terminates. `terminateAttemptWorkers(attemptWorkers)` terminates the survivors of a stale-generation init attempt (swallows already-terminated throws).                                                                                              |
| `init-with-guard.ts` | `initializeWithGuard(worker, api, workerNumber, timeoutMs, attachPermanentHandlers)` — races `api.initialize()` against a hard timeout **and** the worker's own `onerror`/`onmessageerror`. Short-lived handlers reject the init promise; on settle, `attachPermanentHandlers()` restores the long-lived ones. `timeoutMs ≤ 0` or non-finite disables the timer (mirrors `withTimeout()` semantics). Without this guard, a worker that fails before joining the pool would leave Comlink's `initialize()` promise pending forever — the pool isn't aware of the worker yet, so its `handleWorkerFailure` finds nothing to evict. |
| `error-handlers.ts`  | `attachWorkerErrorHandlers(worker, workerNumber, onFailure)` installs the permanent `onerror`/`onmessageerror` listeners; `evictFailedWorker(workers, worker, reason)` splices a worker out of the pool, terminates it, and returns an `EvictOutcome` (`'idempotent'` / `'evicted'` / `'pool-empty'`) so the caller can decide whether to clear `initPromise`. Both `terminate()` calls are wrapped in try/catch to swallow double-terminate throws.                                                                                                                                                                             |

## Init-race vs permanent handlers

A subtle handler-swap keeps init failures from being swallowed:

```
spawn → attachPermanentHandlers(worker)              ← long-lived
       │
       ├─► initializeWithGuard()
       │      ├─ overrides onerror / onmessageerror  ← short-lived
       │      ├─ races api.initialize() vs timeout
       │      └─ on settle → attachPermanentHandlers() restores long-lived
       │
       └─► worker.activeQueries = 0; push to pool
```

If a worker crashes **during** init, the short-lived handlers reject
the init promise (which is what the caller awaits). If it crashes
**after** init, the restored permanent handlers route through
`onFailure` → `evictFailedWorker`, which removes the worker from the
pool list and may signal `'pool-empty'` so the pool can clear its
cached `initPromise` and force a fresh init on next call.

## Generation guard

`spawn-worker.ts`'s `isCurrentGeneration()` closure is the bridge
between this folder's stateless helpers and the pool's generation
token. If `dispose()` bumps the generation while a `spawnWorker()`
call is awaiting init, the spawn self-terminates and throws instead
of pushing onto `this.workers` — preventing a stale init from
re-populating a disposed pool.

`terminateAttemptWorkers()` is the symmetric cleanup for survivors
of a stale-generation init attempt: workers that finished init
before the generation bumped but whose siblings have not, so the
whole attempt is discarded.

## See also

- `../../worker-pool.ts` — the orchestrator that wires these helpers
  together; owns generation, `initPromise`, and the `workers` list.
- `../errors.ts` — `WorkerTimeoutError` / `WorkerAbortError` raised by
  the runtime path (`runWithTimeout`), separate from the init-time
  failures handled here.
- `../../README.md` — pool-level architecture and public API.
