# selection

Worker selection strategies for `WorkerPool`. Two policies coexist: a
**round-robin** rotation used by direct `getWorker()` callers, and a
**least-busy** load-aware picker used by every call routed through
`runWithTimeout` / `getWorkerWithTracking`.

Both files were lifted out of `worker-pool.ts` to keep the pool class
small and to make the policy difference explicit at the import site.

## Files

```
selection/
├── round-robin.ts   — nextRoundRobin: cursor → (instance, nextIndex)
└── least-busy.ts    — selectLeastBusy: scan activeQueries, return tracking handle
```

## round-robin.ts — `nextRoundRobin(workers, cursor)`

Pure rotation. Returns `workers[cursor]` and the next cursor
`(cursor + 1) % workers.length`. The caller (`WorkerPool.nextWorkerInstance`)
stores `nextIndex` back on the pool so the rotation continues across calls.

Used by `WorkerPool.getWorker()` — the untracked accessor retained for
tests and low-level unit tests. Has no load awareness: a stalled worker
keeps getting picked every Nth call, which would cause head-of-line
blocking if used on hot paths.

## least-busy.ts — `selectLeastBusy(workers)`

Linear scan over `workers`, picking the entry with the smallest
`activeQueries`. Ties resolve to the first occurrence (the existing
`leastBusyIndex` is replaced only on strict less-than). Returns a
`TrackedWorkerHandle` that bundles:

- `api`, `worker` — the Comlink remote and the raw `Worker` (the latter
  so `runWithTimeout` can evict it on timeout).
- `markQueryStart()` / `markQueryEnd()` — increment / decrement
  `activeQueries`. End clamps at zero to tolerate paired-call drift.

Used by `WorkerPool.getWorkerWithTracking()`, which `runWithTimeout`
calls on every hot-path dispatch. Once a worker's `activeQueries` grows
past its peers (e.g. a slow WASM call holding it), it stops being
selected until the counter rebalances — this is the head-of-line
blocking fix the round-robin path doesn't have.

## See Also

- `../../worker-pool.ts` — `getWorker`, `getWorkerWithTracking`,
  `runWithTimeout` (selection call sites).
- `../types.ts` — `WorkerInstance` (`api`, `worker`, `activeQueries`).
- `../../README.md` — pool-level overview and load-balancing rationale.
