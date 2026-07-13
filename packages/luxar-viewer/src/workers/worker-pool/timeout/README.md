# Worker-Pool Timeout Helpers

Pure helpers that back `WorkerPool.runWithTimeout`. Split out of the
pool class so each piece — the `Promise.race` timer, the per-call kind
→ ms lookup, and the abort-signal combinator — is independently
testable and free of `WorkerPool` state. Nothing here is exported from
`worker-pool.ts`; consumers only see the behaviour these files
implement.

## File Structure

```
timeout/
├── with-timeout.ts       # Promise.race + setTimeout, evicts on miss
├── pick-timeout-ms.ts    # TimeoutKind → config-knob lookup
└── combine-signals.ts    # AbortSignal.any + manual-wiring fallback
```

## Components

### `with-timeout.ts` — Per-call timeout race

- **`withTimeout(operation, call, timeoutMs, onTimeoutEvict?, worker?)`**
  — Races `call` against a `setTimeout(timeoutMs)` timer.
  - `timeoutMs <= 0` or non-finite is a transparent pass-through —
    returns `call` directly with no timer attached. Keeps the call
    site shape uniform for tests / configs that disable timeouts.
  - On timeout, logs an error via `utils/log` under
    `Modules.WORKER_POOL`, invokes `onTimeoutEvict(worker, reason)`
    if both are supplied so the pool can prune the responsible
    worker, and rejects with `WorkerTimeoutError(operation, timeoutMs)`.
  - The timer is always cleared in the call's `finally` to avoid
    leaking a pending `setTimeout` after a successful resolve.

### `pick-timeout-ms.ts` — Kind → config knob

- **`pickTimeoutMs(kind, perf)`** — Maps a `TimeoutKind` (from
  `../errors.ts`) to the matching ms budget in
  `config.dataLoading.performance`:
  - `'projection'` and `'decode'` → `workerProjectionTimeoutMs`
    (long-running CPU-bound calls; decode shares the projection knob
    until telemetry justifies a dedicated one).

### `combine-signals.ts` — AbortSignal combinator

- **`combineSignals(a, b)`** — Returns a single `AbortSignal` that
  fires when either input does, or `undefined` if both inputs are
  `undefined`. Used by `runWithTimeout` to fold the pool-level
  `setAbortSignal` source together with the per-call signal.
  - Uses native `AbortSignal.any([a, b])` when available (modern
    browsers, Node 20+).
  - Falls back to a `new AbortController()` plus manual
    `addEventListener('abort', ..., { once: true })` wiring on
    each input.
  - Synchronously aborts the controller if either input is already
    aborted on entry, so the caller's "already cancelled" guard
    trips on the first read.

## Invariants

- **`withTimeout` does not cancel the underlying work.** WebAssembly
  has no cancellation primitive, so the rejection only stops the
  caller from awaiting; the worker keeps running and its result is
  discarded. See `WorkerAbortError` in `../errors.ts` for the same
  caveat on the abort path.
- **`combineSignals` returns `undefined` when both inputs are
  `undefined`.** Callers (`worker-pool.ts::runWithTimeout`) rely on
  this to skip the abort-tracking branch entirely when no caller has
  registered a signal.
- **`pickTimeoutMs` is a pure function over `perf`.** It must not
  read `config` directly — the pool reads `config.dataLoading.
performance` once per call and passes it in, so tests can inject
  custom budgets without monkey-patching the config module.

## See Also

- [`../../README.md`](../../README.md) — workers package overview
  (pool architecture, `runWithTimeout` usage, file layout).
- [`../errors.ts`](../errors.ts) — `TimeoutKind`,
  `WorkerTimeoutError`, `WorkerAbortError`.
- [`../../worker-pool.ts`](../../worker-pool.ts) — the consumer that
  wires these helpers into `runWithTimeout`.
