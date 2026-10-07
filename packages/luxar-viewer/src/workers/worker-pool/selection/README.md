# selection

Worker selection for `WorkerPool`: a **least-busy** load-aware picker used
by every call the pool dispatches (`runWithTimeout`, `runDecode`). Lifted out
of `worker-pool.ts` to keep the pool class small.

## Files

```
selection/
├── least-busy.ts       — selectLeastBusy: scan activeQueries, return tracking handle
└── dispatch-tracker.ts — DispatchTracker: worker-side in-flight perf counters
```

## least-busy.ts — `selectLeastBusy(workers)`

Linear scan over `workers`, picking the entry with the smallest
`activeQueries`. Ties resolve to the first occurrence (the existing
`leastBusyIndex` is replaced only on strict less-than). Returns a
`TrackedWorkerHandle` that bundles:

- `api`, `worker` — the Comlink remote and the raw `Worker` (the latter
  so `runWithTimeout` can evict it on timeout).
- `markQueryStart()` / `markQueryEnd()` — increment / decrement
  `activeQueries`. End clamps at zero to tolerate paired-call drift.

Used by the pool's `acquireTrackedWorker()`, which marks the selected worker
busy before returning. Once a worker's `activeQueries` grows past its peers
(e.g. a slow WASM call holding it), it stops being selected until the counter
rebalances — the head-of-line blocking fix a round-robin rotation lacks.

## See Also

- `../../worker-pool.ts` — `runWithTimeout`, `runDecode` (selection call
  sites).
- `../types.ts` — `WorkerInstance` (`api`, `worker`, `activeQueries`).
- `../../README.md` — pool-level overview and load-balancing rationale.
