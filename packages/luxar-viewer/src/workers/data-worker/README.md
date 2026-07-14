# data-worker task implementations

Task helpers backing the Comlink-exposed `workerAPI` in
`../data-worker.ts`. The worker entry itself lives one folder up so
Vite's `?worker` URL import resolves cleanly; this folder is the
implementation tree it dispatches into. Every file here runs inside a
DataWorker context — no DOM, no main-thread globals, no THREE.js.

Each Comlink method on `DataWorkerAPI` is a one-line shim in
`../data-worker.ts` that passes the shared `state` (a `WasmCtx`) plus
the call arguments to one of the per-task implementations re-exported
from this folder. Keeping the task bodies here lets each one own its
own validation, WASM-vs-fallback narrowing, and (where applicable)
scratch-buffer growth, while the entry stays a thin Comlink surface.

## Layout

```
data-worker/
├── state.ts         — WasmCtx { wasm, tsFallback }
│                      + the shared `state` instance, the requireWasm() helper
│                      that throws NOT_INITIALIZED_MSG before any task runs,
│                      and pickBackend(ctx, ndim), which routes EVERY
│                      ndim > 16 operation to the TypeScript backend (the
│                      compiled WASM kernels cap at 16 dims — for >16D data
│                      TS is the production path, not just a WASM-missing
│                      fallback).
├── initialize.ts    — initialize(ctx): loads WASM via initWasm()
│                      (or the TypeScript fallback), throws if
│                      WebAssembly itself is unavailable.
├── types.ts         — ProjectionViewState (narrow ViewState subset the
│                      worker actually consumes). Re-exports
│                      EffectiveRadiusConfig from `../../types/points` so
│                      producers and worker can't drift.
├── validation.ts    — Pure JS→WASM boundary guards: validateNDArrays,
│                      validateProjectionInputs, validateDecodeArgs,
│                      validateLineSegmentReferences, plus MAX_WASM_DIMS
│                      (aliased to config/constants MAX_SUPPORTED_DIMS).
│                      Used by every task before the first WASM call —
│                      a short buffer would otherwise let WASM read past
│                      the end of caller-supplied memory.
├── projection/      — Two tasks: projectLinesTo3D / projectGSplatsTo3D
│                      (nD → 3D). Points project on the main thread
│                      (WASM-accelerated, data/points/projection.ts), so
│                      they are not a worker task; this dir also hosts the
│                      shared in-process dispatcher + getPointsBackend.
└── decode/          — Six tasks: decodeQuantized, decodeLogScalar,
                       decodeGeologScalar, decodePerChannel, decodeLUT,
                       decodeBroadcasted. Per-attribute dequantization
                       paths called from the loaders' chunk-decode stage.
```

## Dispatch shape

The entry file (`../data-worker.ts`) is the only consumer of this
folder. Its `workerAPI` object is what Comlink exposes; each method is
a one-line forward into the matching task here:

```typescript
// ../data-worker.ts (illustrative — see the file for the full surface)
import { state } from './data-worker/state';
import { initialize as initializeImpl } from './data-worker/initialize';
import { projectLinesTo3D as projectLinesTo3DImpl } from './data-worker/projection/lines';
// ...

const workerAPI = {
  initialize: () => initializeImpl(state),
  projectLinesTo3D: (p) => projectLinesTo3DImpl(state, p),
  // ...
};
Comlink.expose(workerAPI);
```

The indirection earns its keep: it keeps each task isolated and
unit-testable without spawning a real worker, it preserves TS narrowing
of `state.wasm` inside each task body (via `requireWasm(ctx)`), and it
lets the entry stay short enough to skim.

## Shared state contract

All tasks accept `ctx: WasmCtx` as their first argument and never read
module-level globals. The `state` singleton from `state.ts` is shared
across calls so:

- `initialize.ts` writes `ctx.wasm` once; every subsequent task reads
  it via `requireWasm(ctx)` and gets a non-null narrowed binding.
- Tasks must not mutate `ctx` beyond the documented fields. Anything
  else belongs in a per-task local.

If a task runs before `initialize()`, `requireWasm` throws
`NOT_INITIALIZED_MSG` — the pool's error handlers in
`../worker-pool/lifecycle/error-handlers.ts` then evict the failed
worker.

## Subpackages

- [projection](./projection/) — Lines/GSplats nD → 3D extraction (worker
  or in-process); Points project on the main thread (WASM-accelerated) in
  `data/points/projection.ts`.
- [decode](./decode/) — quantized / log-scalar / geolog-scalar /
  per-channel / LUT / broadcasted dequantization tasks called from the
  loader chunk-decode stage.

## See also

- `../data-worker.ts` — Comlink entry that re-exports every task here.
- `../README.md` — pool architecture, data flow, and the full
  `DataWorkerAPI` surface.
- `../worker-pool.ts` — pool orchestrator (`runWithTimeout`,
  load-balanced worker selection, error-handler eviction).
- `../../wasm/` — `initWasm()` and the TypeScript fallback consumed by
  `initialize.ts`.
- `../../types/points.ts` — canonical `EffectiveRadiusConfig` re-exported
  from `types.ts`.
