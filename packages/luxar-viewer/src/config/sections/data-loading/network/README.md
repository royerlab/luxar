# network

Data-loading network slice. Owns the fetch-side knobs the zarr loader and `MultiLevelCachingStore` use to bound request lifetimes, cap concurrency, and decide how aggressively to retry transient failures.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher. Composed into `dataLoading.network` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingNetworkConfig: DataLoadingNetworkConfig`. Defines `timeoutMs` (30000 ms general request budget), `validationTimeoutMs` (5000 ms fail-fast budget for the L2 cache-validation HEAD probe in `MultiLevelCachingStore.getRemoteContentHash`), `maxConcurrent` (6 in-flight requests), and `retryAttempts` (3).
- `types.ts` — `DataLoadingNetworkConfig` interface. The `validationTimeoutMs` JSDoc documents the trade-off: lower values fail faster (render from cache while the network is slow); 3G / Edge / high-latency targets should raise to `>=8000` ms because real-world round-trip plus server processing can exceed the 5 s default.
- `validate.ts` — `validateDataLoadingNetwork(config, errors, warnings)`. Rejects non-finite or non-positive `timeoutMs` / `validationTimeoutMs`, non-positive-integer `maxConcurrent`, and negative-integer or non-integer `retryAttempts`. Warns when `validationTimeoutMs < 3000` ms (the "almost certainly broken" floor below which DNS + TLS + server processing rarely complete in one round-trip).

## Public API

- `dataLoadingNetworkConfig` — composed into `dataLoadingConfig.network` by `../data.ts`.
- `DataLoadingNetworkConfig` — re-exported through `../../../types.ts`.
- `validateDataLoadingNetwork` — called from `../validate.ts` (the data-loading dispatcher).
