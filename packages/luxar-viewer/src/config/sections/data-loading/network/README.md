# network

Data-loading network slice. Owns the fetch-side knobs the zarr `MultiLevelCachingStore` uses to bound request lifetimes and decide how aggressively to retry transient failures.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher. Composed into `dataLoading.network` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingNetworkConfig: DataLoadingNetworkConfig`. Defines `timeoutMs` (30000 ms divided across attempts to set each header and body-stall window; it is not a wall-clock cap for one key), `validationTimeoutMs` (5000 ms fail-fast window input for the L2 cache-validation probe in `getRemoteContentHash`), `maxConcurrent` (4 — the ChunkPrefetcher's concurrent-fetch cap), and `retryAttempts` (3).
- `types.ts` — `DataLoadingNetworkConfig` interface. The `validationTimeoutMs` JSDoc documents the trade-off: lower values fail faster (render from cache while the network is slow); 3G / Edge / high-latency targets should raise to `>=8000` ms because real-world round-trip plus server processing can exceed the 5 s default.
- `validate.ts` — `validateDataLoadingNetwork(config, errors, warnings)`. Rejects non-finite or non-positive `timeoutMs` / `validationTimeoutMs`, non-positive-integer `maxConcurrent`, and negative-integer or non-integer `retryAttempts`. Warns when `validationTimeoutMs < 3000` ms (the "almost certainly broken" floor below which DNS + TLS + server processing rarely complete in one round-trip).

## Public API

- `dataLoadingNetworkConfig` — composed into `dataLoadingConfig.network` by `../data.ts`.
- `DataLoadingNetworkConfig` — re-exported through `../../../types.ts`.
- `validateDataLoadingNetwork` — called from `../validate.ts` (the data-loading dispatcher).

## Consumers

- `src/cache/multi-level-caching-store/fetch-retry.ts` — `fetchWithRetry` reads `retryAttempts` (becomes `maxAttempts = retryAttempts + 1`) and `timeoutMs` (split into `ceil(timeoutMs / maxAttempts)` per-attempt header and aggregate no-progress windows). HTTP/2 queue time before a response's first body byte is free while another live lease is progressing. After its first byte, each attempt may spend up to the longer of eight windows or `Content-Length / 16 KiB/s`, scaled by the active leases sharing that aggregate throughput floor, plus retry backoff. With one active response and the defaults, a body up to 960 KiB can therefore keep one key active for about 270 seconds across four attempts after transfer begins, not 30 seconds; multiplexed waves receive proportionally more time.
- `src/cache/multi-level-caching-store.ts` — passes `validationTimeoutMs` to `getRemoteContentHash` via `timeoutMsOverride` so the cache-validation HEAD probe runs on a shorter budget than data fetches.
- `maxConcurrent` is defined and validated here but currently has no consumer — `ChunkPrefetcher` and `cache-setup.ts` hard-code their own concurrency (4). Treat the field as reserved for a future fetch-pool wiring.
