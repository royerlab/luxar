# network

Data-loading network slice. Owns the fetch-side knobs the zarr `MultiLevelCachingStore` uses to bound request lifetimes and decide how aggressively to retry transient failures.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher. Composed into `dataLoading.network` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingNetworkConfig: DataLoadingNetworkConfig`. Defines `timeoutMs` (30000 ms retry budget split into per-attempt header and body-stall windows), `validationTimeoutMs` (5000 ms fail-fast budget for the L2 cache-validation HEAD probe in `getRemoteContentHash`), `maxConcurrent` (4 — the ChunkPrefetcher's concurrent-fetch cap), and `retryAttempts` (3).
- `types.ts` — `DataLoadingNetworkConfig` interface. The `validationTimeoutMs` JSDoc documents the trade-off: lower values fail faster (render from cache while the network is slow); 3G / Edge / high-latency targets should raise to `>=8000` ms because real-world round-trip plus server processing can exceed the 5 s default.
- `validate.ts` — `validateDataLoadingNetwork(config, errors, warnings)`. Rejects non-finite or non-positive `timeoutMs` / `validationTimeoutMs`, non-positive-integer `maxConcurrent`, and negative-integer or non-integer `retryAttempts`. Warns when `validationTimeoutMs < 3000` ms (the "almost certainly broken" floor below which DNS + TLS + server processing rarely complete in one round-trip).

## Public API

- `dataLoadingNetworkConfig` — composed into `dataLoadingConfig.network` by `../data.ts`.
- `DataLoadingNetworkConfig` — re-exported through `../../../types.ts`.
- `validateDataLoadingNetwork` — called from `../validate.ts` (the data-loading dispatcher).

## Consumers

- `src/cache/multi-level-caching-store/fetch-retry.ts` — `fetchWithRetry` reads `retryAttempts` (becomes `maxAttempts = retryAttempts + 1`) and `timeoutMs` (split into `ceil(timeoutMs / maxAttempts)` per-attempt header and no-progress windows; body lifetime also has an absolute throughput-derived ceiling).
- `src/cache/multi-level-caching-store.ts` — passes `validationTimeoutMs` to `getRemoteContentHash` via `timeoutMsOverride` so the cache-validation HEAD probe runs on a shorter budget than data fetches.
- `maxConcurrent` is defined and validated here but currently has no consumer — `ChunkPrefetcher` and `cache-setup.ts` hard-code their own concurrency (4). Treat the field as reserved for a future fetch-pool wiring.
