/**
 * Pool-stats types — common across Points/Lines/GSplats buffer pools.
 *
 * Extracted from gpu-buffer-pool.ts so per-type adapters (added in a
 * later refactor step) can depend on a single source of truth for the
 * stats and ref shapes without circular imports.
 */

import * as THREE from 'three';

/**
 * Attribute type information for Points geometry.
 * Tracks the TypedArray type for each attribute so the pool only reuses
 * geometries whose attribute dtypes match the requested upload.
 */
export interface PointsAttributeTypes {
  position: 'Float32Array';
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
  radius: 'Float32Array' | 'Uint8Array';
  sharpness: 'Float32Array' | 'Uint8Array';
  /**
   * scalar attribute dtype. Omitted (undefined) when the dataset has
   * no scalars — `===` comparison handles undefined === undefined, so
   * `attributeTypesMatch` works without a sentinel.
   */
  scalar?: 'Float32Array' | 'Float16Array' | 'Uint8Array';
}

/**
 * A pooled THREE geometry, plus the metadata the pool needs to decide
 * reuse, eviction, and frame accounting.
 */
export interface PooledBuffer {
  geometry: THREE.BufferGeometry | THREE.InstancedBufferGeometry;
  capacity: number;
  type: 'points' | 'lines' | 'gsplats';
  inUse: boolean;
  lastUsedFrame: number;
  attributeTypes?: PointsAttributeTypes;
}

/** Per-type buffer pool statistics. */
export interface TypePoolStats {
  allocations: number;
  reuses: number;
  evictions: number;
  activeBuffers: number;
  pooledBuffers: number;
  /** per-type byte totals (sum of attribute byteLengths). */
  activeBytes: number;
  pooledBytes: number;
}

/** Overall pool statistics with per-type breakdown. */
export interface PoolStats {
  allocations: number;
  reuses: number;
  evictions: number;
  capacityGrowths: number;
  activeBuffers: number;
  pooledBuffers: number;
  /** cumulative byte counters across all types. */
  activeBytes: number;
  pooledBytes: number;
  totalBytes: number;
  largestPooledBytes: number;
  /**
   * Number of pooled buffers whose eviction was deferred past the
   * current `evictUnused()` call because the per-call batch cap
   * (`evictBatchSize`, default 5) was hit. Diagnostic only — these
   * buffers will be picked up on the next frame's eviction sweep.
   * Useful for spotting "user paused for 5 min then resumed and the
   * eviction queue is stretching across many frames" scenarios.
   */
  deferredEvictions: number;
  byType: {
    points: TypePoolStats;
    lines: TypePoolStats;
    gsplats: TypePoolStats;
  };
}

/**
 * Pure-function-friendly reference to a pooled buffer. The pool itself
 * collects `Ref` structs that include splice metadata; the public ref
 * is the byte-cost + opaque payload that the eviction selector needs
 * — see `selectBuffersToEvict` in `./eviction-policy`.
 */
export interface PooledBufferRef<T = unknown> {
  bytes: number;
  /**
   * Caller-provided opaque payload used to splice the buffer out of
   * its containing pool after the selection returns.
   */
  payload?: T;
}
