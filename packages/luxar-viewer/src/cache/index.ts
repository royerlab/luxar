/**
 * OPFS-based zarr cache package.
 * Provides two-level caching (L1: memory, L2: OPFS) for zarr chunks with optional prefetching.
 */

// Core cache classes
export { LRUCache } from './lru-cache';
export { SegmentedLRUCache } from './segmented-lru-cache';
export { OPFSStore } from './opfs-store';
export { TwoLevelCachingStore } from './two-level-caching-store';
export { ChunkPrefetcher } from './chunk-prefetcher';

// L0 decompressed chunk cache (caches decoded zarr chunks to avoid Blosc decompression)
export { DecompressedChunkCache } from './decompressed-chunk-cache';
export { wrapWithCache, isCachedArray, unwrapCachedArray } from './cached-zarr-array';

// Types
export type { TwoLevelCachingStoreOptions } from './two-level-caching-store';
export type { ChunkPrefetcherOptions } from './chunk-prefetcher';
export type { CacheStats, ExtendedCacheStats, OPFSMetadata } from './types';
export type {
  DecompressedChunk,
  DecompressedChunkCacheOptions,
  DecompressedChunkCacheStats,
} from './decompressed-chunk-cache';
