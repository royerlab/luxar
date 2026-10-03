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

| File                | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `types.ts`          | The `WorkerInstance` interface — one live `Worker` + its Comlink-wrapped `DataWorkerAPI` + the `activeQueries` counter used for load balancing.                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `errors.ts`         | `WorkerTimeoutError`, `WorkerInitTimeoutError` (the init guard's deadline arm — its own type so a caller can retry a slow start instead of writing off the worker: the pool respawns such a slot after a backoff, 3 attempts in all, and the depth-sort coordinator imports it directly to classify a starved init), `WorkerAbortError`, `WorkerUnavailableError`, the `isWorkerInfrastructureError` allow-list predicate (true only for `WorkerUnavailableError` — a timeout is deliberately not fallback-eligible), and the `TimeoutKind` discriminator (`'projection' \| 'decode'`). |
| `stats.ts`          | Pure functions over a `WorkerInstance[]`: `computeStats` (full snapshot) and `computeQueueDepth` (cheap sum for live debug overlays).                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `codec-dispatch.ts` | `BloscDecodeDispatcher` — the decode backend `getWorkerPool()` installs for the zarr blosc codec (`data/codecs/worker-blosc.ts`). Queues chunk decodes per microtask, splits a flush into batches (≤ `MAX_BATCH_CHUNKS` = 8 chunks, ≤ `MAX_BATCH_BYTES` = 8 MB decoded, sized to spread over idle warm workers) and runs each through `WorkerPool.runDecode` (least-busy among WARM workers only — one still downloading its codec gets none; timeout-guarded, not pool-aborted). Declines (main-thread decode) while the pool has no usable worker or no warm worker codec.            |
| `codec-warmup.ts`   | `CodecWarmup` — lazy, one-worker-first warm-up of the workers' blosc codec. Nothing warms at worker-ready (each warm-up downloads the worker's ~600 KB blosc chunk; 15 concurrent ones delayed a hosted first frame 3.6 -> 6.6 s). The first chunk above the offload floor warms ONE worker while it decodes on the main thread; the rest warm once that finished (HTTP cache). Gated by `?mainThreadCodecs`; never counts in `activeQueries`.                                                                                                                                          |

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
├── codec-dispatch.ts           — batched blosc chunk-decode dispatch (codec backend)
├── codec-warmup.ts             — lazy one-worker-first blosc codec warm-up
├── lifecycle/                  — spawn, init guard, error handlers, count
├── selection/                  — least-busy picker
└── timeout/                    — Promise.race timer + kind→ms
```

## Subpackages

- **[lifecycle/](./lifecycle/README.md)** — bring workers up and tear
  them down. Computes the pool size from `hardwareConcurrency`, spawns
  each worker, races `initialize()` against a startup timeout +
  `onerror` (`WorkerPool` respawns a slot that only missed the timeout), and provides the failed-worker eviction path used when a
  call rejects mid-flight.

- **[selection/](./selection/README.md)** — pick which worker handles
  the next call. `least-busy.ts` scans `activeQueries`; every dispatch
  (`runWithTimeout`, `runDecode`) selects through it.

- **[timeout/](./timeout/README.md)** — bound every Comlink round-trip.
  `with-timeout.ts` wraps the shared `raceTimeout` (`utils/race-timeout.ts`)
  with the pool's logging and eviction; `pick-timeout-ms.ts` maps a
  `TimeoutKind` to the right knob in `config.dataLoading.performance`. Only
  the caller's `AbortSignal` races the call: the pool is shared by every host
  on the page, so it holds no dataset signal of its own (each `SceneLoader`
  merges its dataset signal into the calls it makes).

## Relationship to `worker-pool.ts`

`workers/worker-pool.ts` is the only file outside this folder that
imports from here. It composes the pieces:

```
WorkerPool.initialize()
  ├─ getConfiguredWorkerCount()  ── lifecycle/worker-count
  ├─ spawnWorker()               ── lifecycle/spawn-worker
  └─ initializeWithGuard()       ── lifecycle/init-with-guard
        └─ attachWorkerErrorHandlers / evictFailedWorker  (lifecycle/error-handlers)

WorkerPool.runWithTimeout() / runDecode()
  ├─ selectLeastBusy()           ── selection/least-busy
  ├─ pickTimeoutMs(kind)         ── timeout/pick-timeout-ms
  └─ withTimeout()               ── timeout/with-timeout
     (runWithTimeout also races the caller's own AbortSignal, inline)

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
