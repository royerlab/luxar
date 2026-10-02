# network

Data-loading network slice. Owns the fetch-side knobs the zarr `MultiLevelCachingStore` uses to bound request lifetimes and decide how aggressively to retry transient failures, and the widths of the global fetch gate.

Conforms to the section-trio pattern documented in [../../../README.md](../../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher. Composed into `dataLoading.network` by the parent `data-loading` section.

## Contents

- `data.ts` — `dataLoadingNetworkConfig: DataLoadingNetworkConfig`. Defines `timeoutMs` (30000 ms divided across attempts to set each header and body-stall window; it is not a wall-clock cap for one key), `validationTimeoutMs` (5000 ms fail-fast window input for the L2 cache-validation probe in `getRemoteContentHash`), `maxConcurrent` (4 — the ChunkPrefetcher's concurrent-fetch cap), `retryAttempts` (3), and `fetchGate` — the lane widths of `utils/fetch-concurrency.ts`: `maxChunkFetches` (24, the default data lane), `maxMultiplexedChunkFetches` (96, an h2/h3 origin's data lane), `maxMetadataFetches` (4), `http1MaxChunkFetches` / `http1MaxMetadataFetches` (4 + 2 = the six sockets of one plain-`http:` origin) and `speculativeShare` (0.25).
- `types.ts` — `DataLoadingNetworkConfig` and `FetchGateConfig` interfaces (the latter documents why each width is what it is). The `validationTimeoutMs` JSDoc documents the trade-off: lower values fail faster (render from cache while the network is slow); 3G / Edge / high-latency targets should raise to `>=8000` ms because real-world round-trip plus server processing can exceed the 5 s default.
- `validate.ts` — `validateDataLoadingNetwork(config, errors, warnings)`. Rejects non-finite or non-positive `timeoutMs` / `validationTimeoutMs`, non-positive-integer `maxConcurrent`, and negative-integer or non-integer `retryAttempts`; for `fetchGate`, rejects non-positive-integer widths, a multiplexed lane narrower than the default one and a `speculativeShare` outside (0, 1], and warns when the HTTP/1.1 caps do not total six. Warns when `validationTimeoutMs < 3000` ms (the "almost certainly broken" floor below which DNS + TLS + server processing rarely complete in one round-trip).

## Public API

- `dataLoadingNetworkConfig` — composed into `dataLoadingConfig.network` by `../data.ts`.
- `DataLoadingNetworkConfig` — re-exported through `../../../types.ts`.
- `validateDataLoadingNetwork` — called from `../validate.ts` (the data-loading dispatcher).

## Consumers

- `src/cache/multi-level-caching-store/fetch-retry.ts` — `fetchWithRetry` reads `retryAttempts` (becomes `maxAttempts = retryAttempts + 1`) and `timeoutMs` (split into `ceil(timeoutMs / maxAttempts)` per-attempt header and aggregate no-progress windows). After headers arrive and body reading begins, each attempt is bounded by the longer of eight windows or `Content-Length / 16 KiB/s`. Known-length bodies scale for at most eight active leases; unknown-length bodies scale for at most four. With the defaults in TLS-only sessions, an unknown-length attempt is therefore capped at 240 seconds and a representative 500 KiB known-length attempt at 250 seconds, even when all 24 data slots are occupied. Across four attempts, header windows and retry backoff are additional. The separate metadata lane uses 24 data + 4 metadata slots; an origin seen over a plain `http:` URL is held to 4 data + 2 metadata of its own (its six HTTP/1.1 sockets) while other origins keep their width.
- `src/utils/fetch-concurrency.ts` — reads `fetchGate` at every admission (so a test or tuning that changes it applies to the next request).
- `src/cache/multi-level-caching-store.ts` — passes `validationTimeoutMs` to `getRemoteContentHash` via `timeoutMsOverride` so the cache-validation HEAD probe runs on a shorter budget than data fetches.
- `maxConcurrent` is defined and validated here but currently has no consumer — `ChunkPrefetcher` and `cache-setup.ts` hard-code their own concurrency (4). Treat the field as reserved for a future fetch-pool wiring.
