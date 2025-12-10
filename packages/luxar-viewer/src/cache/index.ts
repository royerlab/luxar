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

// Types
export type { TwoLevelCachingStoreOptions } from './two-level-caching-store';
export type { ChunkPrefetcherOptions } from './chunk-prefetcher';
export type { CacheStats, ExtendedCacheStats, OPFSMetadata } from './types';
