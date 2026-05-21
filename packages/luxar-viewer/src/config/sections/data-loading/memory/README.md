# memory

Data-loading memory budgeting slice. Owns the heap-usage targets and check cadence that the cache/loader system uses to decide when to evict and how much memory to keep available.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher. Composed into `dataLoading.memory` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingMemoryConfig: DataLoadingMemoryConfig`. Defines `targetHeapUsage` (0.8 of available heap), `minCacheMB` (128 MB floor), `checkIntervalMs` (10000 ms poll cadence), and `adjustmentThresholds` (`critical: 0.85`, `high: 0.7`) used to classify heap pressure.
- `types.ts` — `DataLoadingMemoryConfig` interface, including the nested `adjustmentThresholds` shape (`critical`, `high`).
- `validate.ts` — `validateDataLoadingMemory(config, errors, warnings)`. Rejects non-finite `targetHeapUsage` and values outside `(0, 1]`; rejects non-finite or non-positive `minCacheMB`. NaN is explicitly screened because `NaN <= 0` is always false.

## Public API

- `dataLoadingMemoryConfig` — composed into `dataLoadingConfig.memory` by `../data.ts`.
- `DataLoadingMemoryConfig` — re-exported through `../../../types.ts`.
- `validateDataLoadingMemory` — called from `../validate.ts` (the data-loading dispatcher).
