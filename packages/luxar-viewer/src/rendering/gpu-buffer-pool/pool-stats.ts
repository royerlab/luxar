/**
 * Pool-stats types — common across Points/Lines/GSplats buffer pools.
 *
 * Shared by per-type adapters so stats and ref shapes have a single
 * source of truth without circular imports.
 */

import * as THREE from 'three';
import type { PooledGeometryType } from '../../types/data-monitor-types';

/**
 * A pooled THREE geometry, plus the metadata the pool needs to decide
 * reuse, eviction, and frame accounting.
 *
 * (The former `attributeTypes?: PointsAttributeTypes` dtype snapshot is
 * gone: points moved to the fixed 3-texel texture layout, so any pooled
 * points geometry fits any points node — dtype normalization happens at
 * upload time, not in the buffer layout.)
 */
export interface PooledBuffer {
  geometry: THREE.BufferGeometry | THREE.InstancedBufferGeometry;
  capacity: number;
  /**
   * `PooledGeometryType`, NOT `GeometryTypeName`: only POOLED geometry types have
   * buffers here (see `pooled` in `types/geometry-capabilities`). A type rendered from
   * a plain `BufferGeometry` — `mesh` — never reaches the pool, so **do not widen this
   * to the geometry vocabulary when a geometry type is added.** Keyed off the shared
   * union rather than spelled out, so adding a POOLED type is a compile error while
   * adding a non-pooled one correctly requires no change here.
   */
  type: PooledGeometryType;
  inUse: boolean;
  lastUsedFrame: number;
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
  /**
   * Per-pooled-type breakdown. `Record<PooledGeometryType, …>` for the same reason as
   * `type` above, and matching its already-keyed sibling
   * `GPUPoolStats.byType` in `types/data-monitor-types.ts` — the two were a keyed
   * record and a hand-written triple describing the same thing.
   */
  byType: Record<PooledGeometryType, TypePoolStats>;
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
