/**
 * OPFS-based zarr cache package.
 * Provides multi-level caching (L0: decompressed chunks, L1: memory,
 * L2: OPFS) for zarr chunks with optional prefetching.
 *
 * This barrel is the package's curated public-API surface. OPFSStore and
 * SegmentedLRUCache are internal helpers of MultiLevelCachingStore and are
 * intentionally not re-exported here.
 */

// Core cache classes
export { LRUCache } from './lru-cache';
export { MultiLevelCachingStore } from './multi-level-caching-store';
export { ChunkPrefetcher } from './chunk-prefetcher';

// L0 decompressed chunk cache (caches decoded zarr chunks to avoid Blosc decompression)
export { DecompressedChunkCache } from './decompressed-chunk-cache';
export { wrapWithCache, isCachedArray, unwrapCachedArray } from './decompressed-chunk-cache/cached-zarr-array';

// Types
export type { MultiLevelCachingStoreOptions } from './multi-level-caching-store';
export type { ChunkPrefetcherOptions } from './chunk-prefetcher';
export type { CacheStats, MultiLevelCacheStats, OPFSMetadata } from './types';
export type { CachedDatasetSummary } from './multi-level-caching-store/opfs-store';
export type {
  DecompressedChunk,
  DecompressedChunkCacheOptions,
  DecompressedChunkCacheStats,
} from './decompressed-chunk-cache';
