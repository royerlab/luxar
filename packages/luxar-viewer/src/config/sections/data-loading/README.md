# data-loading

Navigational hub for the five data-loading configuration slices. Each subfolder follows the section-trio pattern (`data.ts` / `types.ts` / `validate.ts`) documented in [../../README.md](../../README.md); this folder composes their literals into a single `DataLoadingConfig` and dispatches validation across them.

Composed into `AppConfig.dataLoading` by `../../index.ts`.

## Contents

- `data.ts` — `dataLoadingConfig: DataLoadingConfig`. Imports the five sub-section literals (`dataLoadingSpatialConfig`, `dataLoadingNetworkConfig`, `dataLoadingMemoryConfig`, `dataLoadingMonitorConfig`, `dataLoadingPerformanceConfig`) and assembles them under the `spatial` / `network` / `memory` / `monitor` / `performance` keys.
- `types.ts` — `DataLoadingConfig` interface, plus a barrel that re-exports every sub-section type (`DataLoadingSpatialConfig`, `DataLoadingNetworkConfig`, `DataLoadingMemoryConfig`, `DataLoadingMonitorConfig` with its `MonitorTimings` / `MonitorThresholds` / `MonitorLimits` companions, and `DataLoadingPerformanceConfig`).
- `validate.ts` — `validateDataLoading(config, errors, warnings)`. Calls the three sub-section validators that exist (`validateDataLoadingNetwork`, `validateDataLoadingMemory`, `validateDataLoadingPerformance`) and inlines the trivial spatial checks (NaN-hardened `defaultTolerance` and `defaultMaxRadius` must be finite and positive). `monitor` has no validator — its values are passive instrumentation knobs.

## Subpackages

- **[spatial/](spatial/README.md)** — default nD slicing tolerance and default maximum splat radius used when a scene or geometry node does not declare its own.
- **[network/](network/README.md)** — fetch-side knobs the zarr `MultiLevelCachingStore` uses to bound request lifetimes and decide how aggressively to retry transient failures.
- **[memory/](memory/README.md)** — heap-usage targets and check cadence that the cache/loader system uses to decide when to evict and how much memory to keep available.
- **[monitor/](monitor/README.md)** — timings, alert thresholds, and ring-buffer limits used by the runtime data-loading monitor that tracks events, cache rates, query timings, and memory pressure.
- **[performance/](performance/README.md)** — toggles and tuning knobs for the four pipelines that keep nD updates off the main thread and off the allocator: accumulator pooling, the web-worker pool, WASM acceleration, and the GPU buffer pool.

## Public API

- `dataLoadingConfig` — composed into `AppConfig.dataLoading` by `../../index.ts`.
- `DataLoadingConfig` (and every sub-section type) — re-exported through `../../types.ts`.
- `validateDataLoading` — called from `../../validation.ts` (the central config dispatcher).

## See Also

- [../../README.md](../../README.md) — config package overview and the section-trio pattern.
