# worker-pool

Internal helpers behind `workers/worker-pool.ts` — broken into focused
modules so the orchestrator file stays a thin façade. External callers
still import `WorkerPool`, `WorkerTimeoutError`, `TimeoutKind`, etc.
from `../worker-pool.ts`; nothing here is part of the public surface.

The orchestrator's three responsibilities — **lifecycle**, **worker
selection**, and **per-call timeout** — each get their own subfolder.
The three top-level files hold the small bits of shared vocabulary the
subfolders need to talk to each other and to the pool.

## Top-level files

| File        | Purpose                                                                                                                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.ts`  | The `WorkerInstance` interface — one live `Worker` + its Comlink-wrapped `DataWorkerAPI` + the `activeQueries` counter used for load balancing.                                               |
| `errors.ts` | `WorkerTimeoutError`, `WorkerAbortError`, `WorkerUnavailableError`, the `isWorkerInfrastructureError` allow-list predicate, and the `TimeoutKind` discriminator (`'projection' \| 'decode'`). |
| `stats.ts`  | Pure functions over a `WorkerInstance[]`: `computeStats` (full snapshot) and `computeQueueDepth` (cheap sum for live debug overlays).                                                         |

`errors.ts` is the one module re-exported verbatim from
`worker-pool.ts` — `WorkerTimeoutError`, `WorkerUnavailableError`,
`isWorkerInfrastructureError`, and `TimeoutKind` are part of
the public API. `types.ts` and `stats.ts` are pool-internal.

## Layout

```
worker-pool/
├── types.ts                    — WorkerInstance interface
├── errors.ts                   — Timeout / abort / unavailable errors + infra predicate + TimeoutKind
├── stats.ts                    — computeStats, computeQueueDepth
├── lifecycle/                  — spawn, init guard, error handlers, count
├── selection/                  — least-busy and round-robin pickers
└── timeout/                    — Promise.race timer + kind→ms + AbortSignal
```

## Subpackages

- **[lifecycle/](./lifecycle/README.md)** — bring workers up and tear
  them down. Computes the pool size from `hardwareConcurrency`, spawns
  each worker, races `initialize()` against a startup timeout +
  `onerror`, and provides the failed-worker eviction path used when a
  call rejects mid-flight.

- **[selection/](./selection/README.md)** — pick which worker handles
  the next call. `least-busy.ts` scans `activeQueries` and is the
  default path used by `runWithTimeout` / `getWorkerWithTracking`.
  `round-robin.ts` backs the simpler `getWorker()` accessor for callers
  that don't want tracking overhead.

- **[timeout/](./timeout/README.md)** — bound every Comlink round-trip.
  `with-timeout.ts` is the `Promise.race` core; `pick-timeout-ms.ts`
  maps a `TimeoutKind` to the right knob in
  `config.dataLoading.performance`; `combine-signals.ts` merges the
  caller's `AbortSignal` with the pool-wide one set via
  `setAbortSignal()` (using `AbortSignal.any` when available, falling
  back to a hand-rolled forwarder).

## Relationship to `worker-pool.ts`

`workers/worker-pool.ts` is the only file outside this folder that
imports from here. It composes the pieces:

```
WorkerPool.initialize()
  ├─ getConfiguredWorkerCount()  ── lifecycle/worker-count
  ├─ spawnWorker()               ── lifecycle/spawn-worker
  └─ initializeWithGuard()       ── lifecycle/init-with-guard
        └─ attachWorkerErrorHandlers / evictFailedWorker  (lifecycle/error-handlers)

WorkerPool.runWithTimeout()
  ├─ selectLeastBusy()           ── selection/least-busy
  ├─ pickTimeoutMs(kind)         ── timeout/pick-timeout-ms
  ├─ combineSignals()            ── timeout/combine-signals
  └─ withTimeout()               ── timeout/with-timeout

WorkerPool.getStats() / getQueueDepth()
  └─ computeStats / computeQueueDepth   (this folder, stats.ts)
```

The split is purely organizational — every helper here is called from
exactly one place in `worker-pool.ts`, so moving them out kept the
orchestrator readable without changing semantics.

## See Also

- [`../README.md`](../README.md) — full worker subsystem overview
  (architecture diagram, data flow, fallback behavior, public API).
- [`../data-worker.ts`](../data-worker.ts) and `../data-worker/` — the
  worker-side counterpart: WASM bootstrap plus the projection and
  decode task implementations the pool routes calls to.
- `../../config/sections/data-loading/performance/data.ts` — the source of
  truth for the timeout values `pick-timeout-ms.ts` reads.
