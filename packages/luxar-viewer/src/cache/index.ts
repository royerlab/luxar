/**
 * OPFS-based zarr cache package.
 * Provides two-level caching (L1: memory, L2: OPFS) for zarr chunks.
 */

// Core cache classes
export { LRUCache } from './lru-cache';
export { SegmentedLRUCache } from './segmented-lru-cache';
export { OPFSStore } from './opfs-store';
export { TwoLevelCachingStore } from './two-level-caching-store';

// Types
export type { TwoLevelCachingStoreOptions } from './two-level-caching-store';
export type { CacheStats, OPFSMetadata } from './types';
