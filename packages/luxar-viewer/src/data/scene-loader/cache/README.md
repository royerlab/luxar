# Cache stack

Setup of the SceneLoader's three-level cache stack and the read/clear
surface that `__luxarDebug.cache` exposes. Lives here because both
files speak only the cache vocabulary (L0 / L1 / L2, prefetcher,
telemetry) — no zarr, no THREE.js, no monitor.

## Files

| File             | Role                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cache-setup.ts` | `setupCaches(url, flags)` — builds the L0 (`DecompressedChunkCache`) + L1/L2 (`MultiLevelCachingStore`) layers, attaches the `ChunkPrefetcher`, registers L0 invalidation on L1/L2 clear, and resolves the `CacheTelemetryState` for the monitor UI. Returns the raw store the Luxar Zarr facade opens.                                                                                                                                        |
| `cache-api.ts`   | Pure get/clear/list surface for the three cache levels — `getCacheStats`, `listCachedDatasets`, `clearL0Cache`, `clearL1Cache`, `clearL2Cache`, `clearAllCaches`. Each no-ops when its layer is `null` (URL `?noCache` or app-config disable). Defines the `CacheStatsSnapshot` shape `__luxarDebug.cache.getStats()` returns.                                                                                                                 |
| `chunk-packs.ts` | `adoptChunkPacks` hands a packed store's `chunk_packs` index (`luxar optimize --pack`) to the `PackedChunkSource`. A fresh root index proves absence without another request; with a cached index, it probes the source store because packs may have changed without changing `content_hash`. An unpacked store may pay one or two missing-sidecar requests on a warm load. A missing or unreadable sidecar leaves plain chunk reads in place. |

## Consumers

- `../lifecycle/load-scene.ts` calls `setupCaches` once per
  `loadScene` to build the dataset-scoped layers.
- `../../scene-loader.ts` (parent orchestrator) re-exports the
  `cache-api` clear/stats helpers through `__luxarDebug`.
- `core/app/debug/debug-cache-helpers.ts` imports the
  `CacheStatsSnapshot` type for the dev-console formatter.
