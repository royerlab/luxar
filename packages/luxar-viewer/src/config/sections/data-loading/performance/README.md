# performance

Data-loading performance-optimization slice. Owns the toggles and tuning knobs for the three pipelines that keep nD updates off the main thread and off the allocator: accumulator object pooling, the web-worker pool (with per-call and init timeouts), and the GPU buffer pool — plus a performance-monitoring flag. (WASM acceleration has no config knobs — the module loads automatically when workers are enabled.)

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher. Composed into `dataLoading.performance` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingPerformanceConfig: DataLoadingPerformanceConfig`. Defines the four optimization pipelines:
  - **Accumulators**: `useAccumulators` (true), `initialAccumulatorCapacity` (8192) — multi-type accumulator with in-place projection/filtering to eliminate allocations in `projectTo3D`.
  - **Web Workers**: `useWebWorkers` (true), `workerCount` (0 = auto, `navigator.hardwareConcurrency - 1`), `workerProjectionTimeoutMs` (60000), `workerInitTimeoutMs` (10000). AABB spatial queries always stay on the main thread (faster than the worker roundtrip).
  - **GPU Buffer Pool**: `useGPUBufferPool` (true), `gpuPoolMaxSize` (20), `gpuPoolEvictionFrames` (300), `gpuPoolEvictBatchSize` (5), `gpuPoolMaxBytes` (single GPU-geometry byte budget shared by the pool + LOD retention; `null` = auto-size from 25% of `navigator.deviceMemory` with a 2 GB ceiling and a 512 MB fallback only when no signal exists, `0` = disable, positive = pin; overridable via `?gpuBudgetMB=`, while an explicit `?cacheBudgetMB=` may tighten the auto value).
  - Plus `enablePerformanceMonitoring` (false). (The former `materialCacheMaxSize` knob was removed with the LRU material cache — materials are per-node and disposed by node teardown.)
- `types.ts` — `DataLoadingPerformanceConfig` interface. Each timeout / cap field carries an inline JSDoc block explaining the failure mode it protects against (e.g. why `workerInitTimeoutMs` is needed even though it sits outside the normal `handleWorkerFailure` path; why `gpuPoolEvictBatchSize` caps per-frame eviction to avoid stutter; why `gpuPoolMaxBytes` exists alongside `gpuPoolMaxSize` since a single 10M-element Lines buffer can dwarf a 1K-point buffer in real bytes).
- `validate.ts` — `validateDataLoadingPerformance(config, errors, _warnings)`. Validates the two worker timeouts (`workerProjectionTimeoutMs`, `workerInitTimeoutMs`): each must be finite and `≥ 0`, where `0` disables the timeout. No upper bound is enforced — long-running fits can legitimately exceed any "sane" ceiling. Other knobs (pool sizes, capacities) are not validated here; they're treated as numeric and trusted.

## Public API

- `dataLoadingPerformanceConfig` — composed into `dataLoadingConfig.performance` by `../data.ts`.
- `DataLoadingPerformanceConfig` — re-exported through `../../../types.ts`.
- `validateDataLoadingPerformance` — called from `../validate.ts` (the data-loading dispatcher).
