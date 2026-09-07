# cache

OPFS-based zarr cache configuration slice. Owns the three-level cache hierarchy (L0 decompressed chunks, L1 memory LRU, L2 OPFS) size budgets, the OPFS operation timeout, the background L2 write-queue caps, and the external-dataset TTL.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

## Contents

- `data.ts` — `cacheConfig: CacheConfig`. Defaults: caching `enabled`, L0 on with a 200MB decompressed-chunk budget, L1 100MB in-memory LRU, L2 2048MB persistent OPFS, SliceCache ("S-cache") on with a 128MB budget (`sliceCacheEnabled: true`, `sliceCacheMaxSizeMB: 128`), `opfsOperationTimeoutMs: 10_000`, `opfsTimeoutTripThreshold: 3` (consecutive timeouts before the L2 circuit breaker disables the tier for the session — #1645), the background L2 write-queue caps (`opfsWriteConcurrency: 4`, `opfsWriteQueueMax: 16_384`, with retained pending chunk bytes sized from the non-cache heap remainder in `cache/heap-budget.ts` — half of it, capped at 512MB, 256MB with no heap signal — because a pending write's buffer is the same one L1 already holds and bounds, so charging the queue for the L1 budget bound it below what a large scene streams; an overflowing arrival is the entry dropped, so the oldest cap-worth drains in order and leaves a contiguous prefix on disk), `externalDatasetTtlMs: null` (no TTL — external datasets without `content_hash` may be stale indefinitely), `debug: false`.
- `types.ts` — `CacheConfig` interface. Documents each field including the SliceCache pair (per-slice decoded-geometry LRU above L0; disabled via `sliceCacheEnabled: false` or the `?no-slice-cache` URL flag), the rationale for the OPFS timeout (bounds hung-handle stalls into cache misses), and the external-TTL semantics (applies only to datasets without `content_hash`; `null` means no validation, recording `validationMode: 'none'`).
- `validate.ts` — `validateCache(config, errors, warnings)`. Rejects non-finite or non-positive L0/L1/L2/SliceCache sizes; enforces `l1MaxSizeMB ≥ 10` because `SegmentedLRUCache` reserves a 10MB metadata floor below which chunk writes are silently dropped; requires `opfsOperationTimeoutMs`, `opfsWriteConcurrency`, and `opfsWriteQueueMax` to be finite and positive; requires `opfsTimeoutTripThreshold` to be an integer ≥ 1 (it counts consecutive timeouts); allows `externalDatasetTtlMs` to be `null` or a finite positive number (explicitly rejecting `NaN`).

## Public API

- `cacheConfig` — re-exported through `../../index.ts` into `AppConfig.cache`.
- `CacheConfig` — re-exported through `../../types.ts`.
- `validateCache` — called from `../../validation.ts`.
