/**
 * GPU Buffer Pool - Geometry reuse for Points, Lines, and GSplats
 *
 * Eliminates GPU buffer allocations by reusing BufferGeometry objects
 * across view updates. Uses LRU eviction to prevent unbounded memory growth.
 *
 * Key optimizations:
 * - Reuses geometries when size AND type match (0ms GPU allocation)
 * - In-place attribute updates via TypedArray.set()
 * - Size-based bucketing for efficient matching
 * - LRU eviction after 300 frames of non-use
 * - Multi-type support for all TypedArray formats
 *
 * TYPE SUPPORT (COMPLETE):
 * - Points: FULL multi-type support!
 *   - Positions: Float32Array
 *   - Colors: Float32Array | Uint8Array | Uint16Array (with normalization)
 *   - Radii: Float32Array | Uint8Array (with normalization)
 *   - Sharpness: Float32Array | Uint8Array (with normalization)
 * - Lines: Float32Array (per ProcessedLinesData interface)
 * - GSplats: Float32Array (per PackedGSplatsData interface)
 *
 * The pool tracks attribute types per geometry and only reuses geometries
 * with matching types, ensuring type safety and optimal memory efficiency.
 *
 * Based on Performance Optimization Specification v3.6.0
 */

import * as THREE from 'three';
import { log, Modules } from '../utils/log';
import { estimateGeometryBytes, invalidateCachedByteSize } from '../utils/geometry-utils';
import type { LoadedPointsData } from '../data/data-loader-types';
import type { ProcessedLinesData } from '../types/lines';
import {
  packInterleavedAttributes,
  widenToFloat32,
  writeInterleavedAttribute,
  type InterleavedAttributeSpec,
} from './interleaved-attributes';

/**
 * Canonical per-segment attribute layout for pooled line geometries.
 * The pool pre-allocates a single `InstancedInterleavedBuffer` over
 * these specs (Float32 throughout — Uint8 clipped flags get widened
 * at upload time). Optional scalar attributes (aStartScalar /
 * aEndScalar) are added via a spec-set rebuild when colormap data
 * first arrives, mirroring the line-geometry.ts pattern.
 *
 * Declaration order matters only for stride bookkeeping; the shader
 * reads attributes by name through the views.
 */
const LINES_BASE_ATTRIBUTE_SPECS: ReadonlyArray<{
  name: string;
  itemSize: 1 | 2 | 3 | 4;
}> = [
  { name: 'aStartPos', itemSize: 3 },
  { name: 'aEndPos', itemSize: 3 },
  { name: 'aStartColor', itemSize: 3 },
  { name: 'aEndColor', itemSize: 3 },
  { name: 'aStartWidth', itemSize: 1 },
  { name: 'aEndWidth', itemSize: 1 },
  { name: 'aStartSharpness', itemSize: 1 },
  { name: 'aEndSharpness', itemSize: 1 },
  { name: 'aSegmentLength', itemSize: 1 },
  { name: 'aStartClipped', itemSize: 1 },
  { name: 'aEndClipped', itemSize: 1 },
];

const LINES_SCALAR_ATTRIBUTE_SPECS: ReadonlyArray<{
  name: string;
  itemSize: 1 | 2 | 3 | 4;
}> = [
  { name: 'aStartScalar', itemSize: 1 },
  { name: 'aEndScalar', itemSize: 1 },
];

/** Canonical per-splat attribute layout for pooled gsplat geometries. */
const GSPLATS_ATTRIBUTE_SPECS: ReadonlyArray<{
  name: string;
  itemSize: 1 | 2 | 3 | 4;
}> = [
  { name: 'aCenter', itemSize: 3 },
  { name: 'aCholesky01', itemSize: 2 },
  { name: 'aCholesky23', itemSize: 2 },
  { name: 'aCholesky45', itemSize: 2 },
  { name: 'aAmplitude', itemSize: 1 },
  { name: 'aColor', itemSize: 3 },
];

/** Base per-instance attribute layout for pooled points geometries. */
const POINTS_BASE_ATTRIBUTE_SPECS: ReadonlyArray<{
  name: string;
  itemSize: 1 | 2 | 3 | 4;
}> = [
  { name: 'aCenter', itemSize: 3 },
  { name: 'aColor', itemSize: 3 },
  { name: 'aRadius', itemSize: 1 },
  { name: 'aSharpness', itemSize: 1 },
];

const POINTS_SCALAR_ATTRIBUTE_SPEC: { name: string; itemSize: 1 | 2 | 3 | 4 } = {
  name: 'aScalar',
  itemSize: 1,
};

/**
 * Resolve the per-instance attribute layout for a points geometry,
 * including the optional `aScalar` slot iff the type snapshot has it.
 */
function pointAttributeSpecs(
  types: PointsAttributeTypes
): Array<{ name: string; itemSize: 1 | 2 | 3 | 4 }> {
  const specs: Array<{ name: string; itemSize: 1 | 2 | 3 | 4 }> = [
    ...POINTS_BASE_ATTRIBUTE_SPECS,
  ];
  if (types.scalar) {
    specs.push(POINTS_SCALAR_ATTRIBUTE_SPEC);
  }
  return specs;
}

/**
 * Pick a normalization divisor for widening Uint8 / Uint16 source
 * data to Float32 while preserving the GPU-shader-visible [0, 1]
 * range that the previous per-attribute `normalized: true` flag
 * produced. See `interleaved-attributes.ts` for context.
 */
function pointsNormalizationDivisor(
  source: ArrayLike<number> | undefined,
  normalized: boolean
): number | undefined {
  if (!source || !normalized) return undefined;
  if (source instanceof Uint8Array) return 255;
  if (source instanceof Uint16Array) return 65535;
  return undefined;
}

/**
 * Rebuild the interleaved buffer on a geometry with new capacity
 * and/or a new spec-set (e.g. lazily adding scalar attributes).
 * Copies as much of the old buffer as fits into the new layout.
 *
 * Returns the new buffer so the caller can stash it / wire usage.
 */
function rebuildInterleavedBuffer(
  geometry: THREE.InstancedBufferGeometry,
  newCapacity: number,
  newSpecs: ReadonlyArray<{ name: string; itemSize: 1 | 2 | 3 | 4 }>
): THREE.InstancedInterleavedBuffer {
  // Snapshot the old buffer + per-attribute float offsets *before*
  // we replace anything. Used to copy still-present attribute data
  // across.
  const oldByName = new Map<
    string,
    { buffer: THREE.InstancedInterleavedBuffer; offset: number; itemSize: number }
  >();
  for (const spec of newSpecs) {
    const oldView = geometry.getAttribute(spec.name) as THREE.InterleavedBufferAttribute | undefined;
    if (oldView && oldView.data) {
      oldByName.set(spec.name, {
        buffer: oldView.data as THREE.InstancedInterleavedBuffer,
        offset: oldView.offset,
        itemSize: oldView.itemSize,
      });
    }
  }

  const specsWithData: InterleavedAttributeSpec[] = newSpecs.map((spec) => ({
    name: spec.name,
    itemSize: spec.itemSize,
    data: new Float32Array(newCapacity * spec.itemSize),
  }));
  const { buffer: newBuffer, views: newViews } = packInterleavedAttributes(
    specsWithData,
    newCapacity
  );
  newBuffer.setUsage(THREE.DynamicDrawUsage);

  // Carry forward each old attribute's data into the new strided
  // layout. We deinterlace from the old buffer and reinterlace into
  // the new — the new offsets are determined by `newSpecs` order.
  for (const spec of newSpecs) {
    const old = oldByName.get(spec.name);
    if (!old) continue;
    const oldArray = old.buffer.array as Float32Array;
    const oldStride = old.buffer.stride;
    const oldCapacity = Math.floor(oldArray.length / oldStride);
    const carry = Math.min(oldCapacity, newCapacity);
    const newView = newViews[spec.name];
    const newOffset = newView.offset;
    const newStride = newBuffer.stride;
    const newArray = newBuffer.array as Float32Array;
    for (let i = 0; i < carry; i++) {
      const oldStart = i * oldStride + old.offset;
      const newStart = i * newStride + newOffset;
      for (let k = 0; k < spec.itemSize; k++) {
        newArray[newStart + k] = oldArray[oldStart + k];
      }
    }
    geometry.setAttribute(spec.name, newView);
  }
  // Attributes new to the spec-set (e.g. aStartScalar on a colormap
  // toggle) still need to be bound — the loop above only handles
  // names that already existed. Bind any that didn't carry forward.
  for (const spec of newSpecs) {
    if (!oldByName.has(spec.name)) {
      geometry.setAttribute(spec.name, newViews[spec.name]);
    }
  }
  // If the old buffer had attributes the new spec-set drops, remove
  // them so the geometry doesn't dangle stale views.
  for (const name of Object.keys(geometry.attributes)) {
    if (name === 'aQuadCorner') continue;
    if (!newSpecs.find((s) => s.name === name)) {
      geometry.deleteAttribute(name);
    }
  }

  // CRITICAL: r184 caches `_maxInstanceCount` on the geometry; replacing
  // the buffer doesn't invalidate it. Mirrors the standalone-geometry
  // workaround in line-geometry.ts / gsplat-geometry.ts.
  delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

  return newBuffer;
}

/**
 * Write a packed per-attribute source array into the geometry's
 * interleaved buffer at the right strided offset. Internal helper —
 * the pool uses this from updateXxxGeometry instead of poking
 * `attr.set(...)` per-attribute.
 */
function writePooledAttribute(
  geometry: THREE.InstancedBufferGeometry,
  name: string,
  src: Float32Array,
  count: number
): void {
  const view = geometry.getAttribute(name) as THREE.InterleavedBufferAttribute;
  const buffer = view.data as THREE.InstancedInterleavedBuffer;
  writeInterleavedAttribute(buffer, view.offset, view.itemSize, src.subarray(0, count * view.itemSize) as Float32Array, count);
}

// D.4: re-export so existing consumers that import these from
// `rendering/gpu-buffer-pool` keep working.
export { estimateGeometryBytes, invalidateCachedByteSize };

/**
 * Packed GSplats data ready for GPU upload (from gsplats/projection.ts)
 */
export interface PackedGSplatsData {
  centers3D: Float32Array; // M * 3
  amplitudes: Float32Array; // M
  cholesky01: Float32Array; // M * 2 [L00, L10]
  cholesky23: Float32Array; // M * 2 [L11, L20]
  cholesky45: Float32Array; // M * 2 [L21, L22]
  colors: Float32Array; // M * 3 (RGB)
  splatCount: number;
}

/**
 * Attribute type information for Points geometry
 * Tracks the TypedArray type for each attribute to enable proper reuse
 */
export interface PointsAttributeTypes {
  position: 'Float32Array'; // Always Float32Array for positions
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
  radius: 'Float32Array' | 'Uint8Array';
  sharpness: 'Float32Array' | 'Uint8Array';
  /**
   * scalar attribute dtype. Omitted (undefined) when the dataset
   * has no scalars — `===` comparison handles undefined === undefined,
   * so `attributeTypesMatch` works without a sentinel.
   */
  scalar?: 'Float32Array' | 'Float16Array' | 'Uint8Array';
}

export interface PooledBuffer {
  geometry: THREE.BufferGeometry | THREE.InstancedBufferGeometry;
  capacity: number;
  type: 'points' | 'lines' | 'gsplats';
  inUse: boolean;
  lastUsedFrame: number;
  // Attribute types (only for Points geometries)
  attributeTypes?: PointsAttributeTypes;
}

/**
 * Per-type buffer pool statistics
 */
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

/**
 * Overall pool statistics with per-type breakdown
 */
export interface PoolStats {
  // Global totals
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
  // Per-type breakdown
  byType: {
    points: TypePoolStats;
    lines: TypePoolStats;
    gsplats: TypePoolStats;
  };
}

/**
 * Pure selector for byte-budget eviction.
 *
 * Given an array of pooled-buffer refs and a target budget, returns
 * the subset that should be evicted to bring total bytes ≤ maxBytes.
 * Strategy: sort largest-first and walk until the running total drops
 * under budget. Exported for unit testing — keeps the policy isolated
 * from the side-effecting eviction logic in the pool.
 */
export interface PooledBufferRef<T = unknown> {
  bytes: number;
  // Caller-provided opaque payload used to splice the buffer out of
  // its containing pool after the selection returns.
  payload?: T;
}

export function selectBuffersToEvict<R extends PooledBufferRef>(
  refs: R[],
  maxBytes: number,
  precomputedTotal?: number
): R[] {
  const total =
    precomputedTotal !== undefined ? precomputedTotal : refs.reduce((sum, r) => sum + r.bytes, 0);
  if (total <= maxBytes) return [];
  // Stable largest-first ordering. JS sort is stable in modern engines
  // (V8, JSC, SpiderMonkey since 2019) so equal-size buffers retain
  // their input order — important for deterministic test output.
  const sorted = refs.slice().sort((a, b) => b.bytes - a.bytes);
  const targets: R[] = [];
  let running = total;
  for (const ref of sorted) {
    if (running <= maxBytes) break;
    targets.push(ref);
    running -= ref.bytes;
  }
  return targets;
}

/**
 * GPU buffer pool for reusing THREE.BufferGeometry objects.
 *
 * Manages separate pools for Points, Lines, and GSplats geometries,
 * each with different attribute layouts and update patterns.
 */
export class GPUBufferPool {
  private pointBuffers = new Map<number, PooledBuffer[]>(); // Bucket by capacity
  private lineBuffers = new Map<number, PooledBuffer[]>();
  private gsplatBuffers = new Map<number, PooledBuffer[]>();

  private activeBuffers = new Map<string, PooledBuffer>(); // nodeId → active geometry
  private frameCount = 0;

  private maxPoolSize: number;
  private evictionFrames: number;
  /**
   * Maximum geometries the pool will dispose in a single
   * `evictUnused()` call when not over the hard limit. Without this
   * cap, a single eviction sweep can dispose dozens of buffers
   * synchronously — each `geometry.dispose()` is 5–20 ms on slow
   * GPUs, so a burst stutters visibly. Remaining evictable buffers
   * are deferred to the next frame's eviction sweep. The
   * `mustEvict` (pool-over-limit) path ignores this cap so the pool
   * never grows unbounded.
   */
  private evictBatchSize: number;
  /**
   * pooled-byte budget. When `pooledBytes` exceeds this value,
   * `evictUnused()` evicts pooled buffers (largest first) until under
   * budget — independent of the count-based cap above. `0` disables
   * the byte budget, restoring count-only behavior.
   */
  private maxPoolBytes: number;

  private stats = {
    allocations: 0,
    reuses: 0,
    evictions: 0,
    capacityGrowths: 0,
    /** Pooled buffers skipped this `evictUnused` call due to batch cap. */
    deferredEvictions: 0,
  };

  /**
   * One-shot guard: have we already logged the >100MB pooled-buffer
   * warning? Re-checked per `evictUnused` so the noise stays bounded.
   */
  private largePoolWarningEmitted = false;

  // Per-type stats tracking
  private typeStats = {
    points: { allocations: 0, reuses: 0, evictions: 0 },
    lines: { allocations: 0, reuses: 0, evictions: 0 },
    gsplats: { allocations: 0, reuses: 0, evictions: 0 },
  };

  constructor(
    maxPoolSize: number = 20,
    evictionFrames: number = 300,
    evictBatchSize: number = 5,
    maxPoolBytes: number = 512_000_000
  ) {
    this.maxPoolSize = maxPoolSize;
    this.evictionFrames = evictionFrames;
    this.evictBatchSize = Math.max(1, evictBatchSize);
    this.maxPoolBytes = Math.max(0, maxPoolBytes);
  }

  /**
   * Advance the frame counter. Call once per frame before any acquire calls.
   * This ensures eviction timing is based on rendered frames, not acquire calls.
   */
  beginFrame(): void {
    this.frameCount++;
  }

  // =========================================================================
  // Points Geometry Management
  // =========================================================================

  /**
   * Helper to check if two attribute type sets match
   */
  private attributeTypesMatch(a: PointsAttributeTypes, b: PointsAttributeTypes): boolean {
    return (
      a.position === b.position &&
      a.color === b.color &&
      a.radius === b.radius &&
      a.sharpness === b.sharpness &&
      a.scalar === b.scalar
    );
  }

  /**
   * Acquire Points geometry from pool (type-aware, capacity-aware)
   *
   * Returns existing geometry if node already has one with matching types and sufficient capacity.
   * Otherwise searches pool for reusable geometry, or creates new with correct types.
   *
   * @param nodeId - Unique identifier for this geometry (typically scene node path)
   * @param data - Point data to detect attribute types from
   * @param pointCount - Number of points needed
   * @returns BufferGeometry with typed attributes matching data
   *
   * @example
   * ```typescript
   * // First acquisition: Creates new geometry with Uint8 colors
   * const geom1 = pool.acquirePointsGeometry('/node1', {
   *   positions: new Float32Array(...),
   *   colors: new Uint8Array(...)  // Detects Uint8
   * }, 1000);
   *
   * // Later: Reuses same geometry (types match, capacity ok)
   * const geom2 = pool.acquirePointsGeometry('/node1', sameTypeData, 900);
   * // geom2 === geom1 (reused!)
   *
   * // Different type: Creates new geometry
   * const geom3 = pool.acquirePointsGeometry('/node2', {
   *   colors: new Float32Array(...)  // Different type
   * }, 1000);
   * // geom3 !== geom1 (different types, can't reuse)
   * ```
   */
  acquirePointsGeometry(
    nodeId: string,
    data: LoadedPointsData,
    pointCount: number
  ): THREE.BufferGeometry {
    // Detect attribute types from data
    const types = this.detectAttributeTypes(data);

    // Check if this node already has an active geometry
    const active = this.activeBuffers.get(nodeId);
    if (active && active.type === 'points' && active.attributeTypes) {
      // Check if types match
      if (this.attributeTypesMatch(active.attributeTypes, types)) {
        if (active.capacity >= pointCount) {
          // Perfect! Reuse existing (same types, sufficient capacity)
          active.lastUsedFrame = this.frameCount;
          this.stats.reuses++;
          this.typeStats.points.reuses++;
          return this.preparePointsGeometryForDraw(active.geometry, pointCount);
        } else {
          // Need to grow - reallocate attributes with same types
          this.growPointsGeometry(
            active.geometry as THREE.BufferGeometry,
            pointCount,
            active.attributeTypes
          );
          active.capacity = Math.ceil(pointCount * 1.5);
          active.lastUsedFrame = this.frameCount;
          this.stats.capacityGrowths++;
          return this.preparePointsGeometryForDraw(active.geometry, pointCount);
        }
      } else {
        // Types changed! Release old geometry and create new
        this.releasePointsGeometry(nodeId);
      }
    }

    // Try to find in pool (search across all buckets for suitable capacity AND matching types)
    for (const pooled of this.pointBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (
          candidate.attributeTypes &&
          candidate.capacity >= pointCount &&
          this.attributeTypesMatch(candidate.attributeTypes, types)
        ) {
          // Found suitable geometry with matching types!
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          this.typeStats.points.reuses++;
          return this.preparePointsGeometryForDraw(candidate.geometry, pointCount);
        }
      }
    }

    // Allocate new geometry with correct types
    const capacity = Math.ceil(pointCount * 1.5);
    const geometry = this.createPointsGeometry(capacity, types);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'points',
      inUse: true,
      lastUsedFrame: this.frameCount,
      attributeTypes: types,
    };

    this.activeBuffers.set(nodeId, newBuffer);
    this.stats.allocations++;
    this.typeStats.points.allocations++;

    return this.preparePointsGeometryForDraw(geometry, pointCount);
  }

  /**
   * Release Points geometry back to pool.
   */
  releasePointsGeometry(nodeId: string): void {
    const buffer = this.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'points') return;

    this.activeBuffers.delete(nodeId);
    buffer.inUse = false;

    // Add to pool (size-bucketed)
    const bucket = this.getBucket(buffer.capacity);
    if (!this.pointBuffers.has(bucket)) {
      this.pointBuffers.set(bucket, []);
    }
    this.pointBuffers.get(bucket)!.push(buffer);

    // Evict old buffers if pool too large
    this.evictUnused();
  }

  /**
   * Prepare pooled point geometry for the visible point count.
   *
   * Points render as instanced unit quads, so the indexed draw range is
   * always the 2-triangle base quad (6 indices) while `instanceCount`
   * carries the number of point sprites. Using drawRange for point count
   * would still render only one non-instanced quad on r184 if the geometry
   * were not explicitly instanced.
   */
  private preparePointsGeometryForDraw(
    geometry: THREE.BufferGeometry,
    pointCount: number
  ): THREE.InstancedBufferGeometry {
    const instanced = geometry as THREE.InstancedBufferGeometry;
    instanced.instanceCount = pointCount;
    instanced.setDrawRange(0, 6);
    return instanced;
  }

  /**
   * Detect attribute types from LoadedPointsData
   */
  private detectAttributeTypes(data: LoadedPointsData): PointsAttributeTypes {
    const types: PointsAttributeTypes = {
      position: 'Float32Array', // Always Float32Array
      color:
        data.colors instanceof Uint8Array
          ? 'Uint8Array'
          : data.colors instanceof Uint16Array
            ? 'Uint16Array'
            : 'Float32Array',
      radius: data.radii instanceof Uint8Array ? 'Uint8Array' : 'Float32Array',
      sharpness: data.sharpness instanceof Uint8Array ? 'Uint8Array' : 'Float32Array',
    };
    // scalar dtype — omitted when the dataset has no scalars (the
    // common case). When present, we honor the source dtype so Uint8
    // normalised LUT lookups work alongside Float32/Float16 raw values.
    if (data.scalars) {
      if (data.scalars instanceof Uint8Array) {
        types.scalar = 'Uint8Array';
      } else if (
        typeof globalThis.Float16Array !== 'undefined' &&
        data.scalars instanceof globalThis.Float16Array
      ) {
        types.scalar = 'Float16Array';
      } else {
        types.scalar = 'Float32Array';
      }
    }
    return types;
  }

  /**
   * Create Points geometry with type-specific instanced attributes.
   *
   * Layout matches the line + gsplat pattern: a shared unit-quad
   * base geometry (4 vertices + 2-triangle index) plus per-instance
   * `InstancedBufferAttribute`s for centre/colour/radius/sharpness/
   * scalar. The base is allocated unconditionally; per-instance
   * attributes are sized to `capacity`.
   */
  private createPointsGeometry(
    capacity: number,
    types: PointsAttributeTypes
  ): THREE.InstancedBufferGeometry {
    // Start from the shared unit-quad base — same shape that
    // `point-geometry.ts::createPointQuadGeometry` produces, but
    // inlined here to keep the pool self-contained.
    const geometry = new THREE.InstancedBufferGeometry();
    const quadCorners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);
    geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    // Safe no-draw state until acquire/update sets the visible count via
    // preparePointsGeometryForDraw(). The draw range always stays on the
    // 2-triangle base quad.
    geometry.instanceCount = 0;
    geometry.setDrawRange(0, 6);

    // Pre-allocate the per-instance interleaved buffer at `capacity`.
    // All attributes are Float32 in the interleaved storage; Uint8 /
    // Uint16 source data is widened (and divided by the appropriate
    // normalization divisor) at upload time in `updatePointsGeometry`.
    // The `types` snapshot is still tracked on `PooledBuffer` for
    // reuse-matching, but no longer drives the buffer layout.
    const specs = pointAttributeSpecs(types).map((spec) => ({
      ...spec,
      data: new Float32Array(capacity * spec.itemSize),
    }));
    const { buffer, views } = packInterleavedAttributes(specs, capacity);
    buffer.setUsage(THREE.DynamicDrawUsage);
    for (const spec of specs) {
      geometry.setAttribute(spec.name, views[spec.name]);
    }

    return geometry;
  }

  /**
   * Grow Points geometry to new capacity (reallocates attributes
   * with same types). Preserves the unit-quad base attribute and
   * index — only the per-instance `InstancedBufferAttribute`s
   * reallocate.
   */
  private growPointsGeometry(
    geometry: THREE.BufferGeometry,
    neededCount: number,
    types: PointsAttributeTypes
  ): void {
    const newCapacity = Math.ceil(neededCount * 1.5);

    // D.3: any attribute we're about to replace invalidates the cached
    // byte estimate. Clear it once up front rather than after each
    // attribute swap.
    invalidateCachedByteSize(geometry);

    // Reallocate the interleaved buffer at the new capacity. Spec-set
    // (with/without `aScalar`) follows `types.scalar`.
    rebuildInterleavedBuffer(
      geometry as THREE.InstancedBufferGeometry,
      newCapacity,
      pointAttributeSpecs(types)
    );
  }

  /**
   * Update Points geometry attributes in-place (zero GPU allocations)
   *
   * Updates all attributes using TypedArray.set() for efficient copying.
   * Sets needsUpdate flags to trigger GPU upload. Preserves native types.
   *
   * @param geometry - Geometry to update (from acquirePointsGeometry)
   * @param data - Point data with new attribute values
   * @param count - Number of points in data
   *
   * @example
   * ```typescript
   * const geometry = pool.acquirePointsGeometry('/node1', data, 1000);
   * pool.updatePointsGeometry(geometry, newData, 1000);  // In-place update
   * mesh.geometry = geometry;  // Assign to mesh (might be same geometry, reused!)
   * ```
   */
  updatePointsGeometry(
    geometry: THREE.BufferGeometry,
    data: LoadedPointsData,
    count: number
  ): void {
    const instanced = geometry as THREE.InstancedBufferGeometry;

    // Positions: typically Float32 but `PositionArray` permits
    // Float16. Widen if needed; no normalization divisor (positions
    // are world-space, not normalized).
    const positionsF32 =
      data.positions instanceof Float32Array
        ? data.positions
        : widenToFloat32(data.positions as ArrayLike<number>);
    writePooledAttribute(instanced, 'aCenter', positionsF32, count);

    // Colors: widen + normalize per the source dtype + the historical
    // "normalized: true" semantics for Uint8 / Uint16 source. Missing
    // colors → write a `1.0` fill so points render white instead of
    // black (which would be discarded by the shader's near-zero check).
    const colorView = instanced.getAttribute('aColor') as THREE.InterleavedBufferAttribute;
    const colorBuffer = colorView.data as THREE.InstancedInterleavedBuffer;
    if (data.colors) {
      const widened = widenToFloat32(
        data.colors.subarray(0, count * 3) as ArrayLike<number>,
        pointsNormalizationDivisor(data.colors, /*normalized=*/ true)
      );
      writeInterleavedAttribute(colorBuffer, colorView.offset, 3, widened, count);
    } else {
      const fill = new Float32Array(count * 3);
      fill.fill(1.0);
      writeInterleavedAttribute(colorBuffer, colorView.offset, 3, fill, count);
    }

    // Radii: widen + normalize. Missing → 0.5 default (matches
    // NodeFactory.createPointsGeometry).
    const radView = instanced.getAttribute('aRadius') as THREE.InterleavedBufferAttribute;
    const radBuffer = radView.data as THREE.InstancedInterleavedBuffer;
    if (data.radii) {
      const widened = widenToFloat32(
        data.radii.subarray(0, count) as ArrayLike<number>,
        pointsNormalizationDivisor(data.radii, /*normalized=*/ true)
      );
      writeInterleavedAttribute(radBuffer, radView.offset, 1, widened, count);
    } else {
      const fill = new Float32Array(count);
      fill.fill(0.5);
      writeInterleavedAttribute(radBuffer, radView.offset, 1, fill, count);
    }

    // Sharpness: widen + normalize. Missing → 2.0 default.
    const sharpView = instanced.getAttribute('aSharpness') as THREE.InterleavedBufferAttribute;
    const sharpBuffer = sharpView.data as THREE.InstancedInterleavedBuffer;
    if (data.sharpness) {
      const widened = widenToFloat32(
        data.sharpness.subarray(0, count) as ArrayLike<number>,
        pointsNormalizationDivisor(data.sharpness, /*normalized=*/ true)
      );
      writeInterleavedAttribute(sharpBuffer, sharpView.offset, 1, widened, count);
    } else {
      const fill = new Float32Array(count);
      fill.fill(2.0);
      writeInterleavedAttribute(sharpBuffer, sharpView.offset, 1, fill, count);
    }

    // Optional scalar — only present when `pointAttributeSpecs` was
    // built with `types.scalar` set. Widen Uint8 / Float16 sources.
    const scalarView = instanced.getAttribute('aScalar') as THREE.InterleavedBufferAttribute | undefined;
    if (scalarView) {
      const scalarBuffer = scalarView.data as THREE.InstancedInterleavedBuffer;
      if (data.scalars) {
        const widened = widenToFloat32(
          data.scalars.subarray(0, count) as ArrayLike<number>,
          pointsNormalizationDivisor(data.scalars, /*normalized=*/ true)
        );
        writeInterleavedAttribute(scalarBuffer, scalarView.offset, 1, widened, count);
      } else {
        // No source scalars but the buffer exists — fill zero so a
        // colormap LUT lookup at scalar=0 returns the LUT's first
        // entry (equivalent to disabling colormap visually).
        const fill = new Float32Array(count);
        writeInterleavedAttribute(scalarBuffer, scalarView.offset, 1, fill, count);
      }
    }

    // Points render as instanced unit quads: keep the indexed draw range
    // on the 2-triangle base quad and put the visible point count in
    // instanceCount.
    this.preparePointsGeometryForDraw(geometry, count);
    if (data.metadata.bounds) {
      instanced.boundingBox = data.metadata.bounds.clone();
    } else {
      const box = new THREE.Box3();
      const v = new THREE.Vector3();
      const positions = data.positions;
      for (let i = 0; i < count; i++) {
        v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        box.expandByPoint(v);
      }
      instanced.boundingBox = box;
    }
    instanced.boundingSphere = new THREE.Sphere();
    instanced.boundingBox.getBoundingSphere(instanced.boundingSphere);
  }

  // =========================================================================
  // Lines Geometry Management
  // =========================================================================

  /**
   * Acquire geometry for Lines (instanced per-segment attributes).
   */
  acquireLinesGeometry(nodeId: string, segmentCount: number): THREE.InstancedBufferGeometry {
    const active = this.activeBuffers.get(nodeId);
    if (active && active.type === 'lines') {
      if (active.capacity >= segmentCount) {
        active.lastUsedFrame = this.frameCount;
        this.stats.reuses++;
        this.typeStats.lines.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      } else {
        this.growLinesGeometry(active.geometry as THREE.InstancedBufferGeometry, segmentCount);
        active.capacity = Math.ceil(segmentCount * 1.5);
        active.lastUsedFrame = this.frameCount;
        this.stats.capacityGrowths++;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
    }

    // Try to find in pool (search across all buckets)
    for (const pooled of this.lineBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= segmentCount) {
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          this.typeStats.lines.reuses++;
          return candidate.geometry as THREE.InstancedBufferGeometry;
        }
      }
    }

    // Allocate new
    const capacity = Math.ceil(segmentCount * 1.5);
    const geometry = this.createLinesGeometry(capacity);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'lines',
      inUse: true,
      lastUsedFrame: this.frameCount,
    };

    this.activeBuffers.set(nodeId, newBuffer);
    this.stats.allocations++;
    this.typeStats.lines.allocations++;

    return geometry;
  }

  /**
   * Release Lines geometry back to pool.
   */
  releaseLinesGeometry(nodeId: string): void {
    const buffer = this.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'lines') return;

    this.activeBuffers.delete(nodeId);
    buffer.inUse = false;

    const bucket = this.getBucket(buffer.capacity);
    if (!this.lineBuffers.has(bucket)) {
      this.lineBuffers.set(bucket, []);
    }
    this.lineBuffers.get(bucket)!.push(buffer);

    this.evictUnused();
  }

  /**
   * Create Lines geometry (InstancedBufferGeometry with per-segment attributes).
   */
  private createLinesGeometry(segmentCapacity: number): THREE.InstancedBufferGeometry {
    const geometry = new THREE.InstancedBufferGeometry();

    // Base quad geometry (shared across all instances)
    const quadPositions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadPositions, 2));
    geometry.setIndex([0, 1, 2, 2, 1, 3]);

    // Pre-allocate the per-segment interleaved buffer at `segmentCapacity`.
    // Scalar attributes (aStartScalar / aEndScalar) are NOT included in
    // the initial stride — they're added on first scalar commit via a
    // spec-set rebuild (see `updateLinesGeometry`). All attributes are
    // Float32 in the interleaved storage (clipped flags widened at
    // upload time in the geometry layer).
    const baseSpecs = LINES_BASE_ATTRIBUTE_SPECS.map((spec) => ({
      ...spec,
      data: new Float32Array(segmentCapacity * spec.itemSize),
    }));
    const { buffer, views } = packInterleavedAttributes(baseSpecs, segmentCapacity);
    buffer.setUsage(THREE.DynamicDrawUsage);
    for (const spec of baseSpecs) {
      geometry.setAttribute(spec.name, views[spec.name]);
    }

    return geometry;
  }

  /**
   * Grow Lines geometry to new capacity.
   *
   * Also grows optional `aStartScalar` / `aEndScalar` instanced
   * attributes when present. Line geometries allocate scalar attributes
   * lazily on first scalar upload, so resizing must carry them forward
   * alongside the always-present per-segment attributes.
   */
  private growLinesGeometry(geometry: THREE.InstancedBufferGeometry, neededCount: number): void {
    const newCapacity = Math.ceil(neededCount * 1.5);

    // D.3: invalidate cached byte estimate before re-allocating any attribute.
    invalidateCachedByteSize(geometry);

    // Reallocate the interleaved buffer at the new capacity. The
    // spec-set is the same one currently bound on the geometry —
    // carry scalar attributes forward iff they were already present.
    const hasScalars = geometry.getAttribute('aStartScalar') !== undefined;
    const specs = hasScalars
      ? [...LINES_BASE_ATTRIBUTE_SPECS, ...LINES_SCALAR_ATTRIBUTE_SPECS]
      : LINES_BASE_ATTRIBUTE_SPECS;
    rebuildInterleavedBuffer(geometry, newCapacity, specs);
  }

  /**
   * Update Lines geometry in place.
   */
  updateLinesGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: ProcessedLinesData,
    count: number
  ): void {
    // Lazy scalar-spec promotion: if data carries scalars but the
    // interleaved buffer wasn't allocated with the scalar slots,
    // rebuild with the larger stride. Carries existing data across.
    const hasScalarsInData = !!(data.startScalars && data.endScalars);
    const hasScalarsInBuffer = geometry.getAttribute('aStartScalar') !== undefined;
    if (hasScalarsInData && !hasScalarsInBuffer) {
      const startView = geometry.getAttribute('aStartPos') as THREE.InterleavedBufferAttribute;
      const capacity = Math.floor(
        (startView.data.array as Float32Array).length / startView.data.stride
      );
      rebuildInterleavedBuffer(geometry, capacity, [
        ...LINES_BASE_ATTRIBUTE_SPECS,
        ...LINES_SCALAR_ATTRIBUTE_SPECS,
      ]);
    }

    // Clipped flags arrive as Uint8 (per ProcessedLinesData); widen
    // to Float32 for the interleaved buffer.
    const startClippedF32 = widenToFloat32(data.startClipped);
    const endClippedF32 = widenToFloat32(data.endClipped);

    // Write each base attribute into its strided slot.
    const baseUpdates: Array<[string, Float32Array]> = [
      ['aStartPos', data.startPositions],
      ['aEndPos', data.endPositions],
      ['aStartColor', data.startColors],
      ['aEndColor', data.endColors],
      ['aStartWidth', data.startWidths],
      ['aEndWidth', data.endWidths],
      ['aStartSharpness', data.startSharpness],
      ['aEndSharpness', data.endSharpness],
      ['aSegmentLength', data.segmentLengths],
      ['aStartClipped', startClippedF32],
      ['aEndClipped', endClippedF32],
    ];
    for (const [name, source] of baseUpdates) {
      writePooledAttribute(geometry, name, source, count);
    }

    if (hasScalarsInData) {
      writePooledAttribute(geometry, 'aStartScalar', data.startScalars as Float32Array, count);
      writePooledAttribute(geometry, 'aEndScalar', data.endScalars as Float32Array, count);
    }

    // Update instance count
    geometry.instanceCount = count;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    // CRITICAL: Recompute bounding box after position updates
    // Without this, frustum culling uses stale bounds from previous frame/time slice
    // This causes geometry to disappear when zooming close (small frustum excludes stale box)
    // Performance: O(n) in segment count, but only runs when geometry updates (not every frame)
    // also track max width to expand bounds by rendered footprint
    // (mirrors `computeLineBounds` in line-geometry.ts).
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    let maxWidth = 0;
    for (let i = 0; i < count; i++) {
      v.set(
        data.startPositions[i * 3],
        data.startPositions[i * 3 + 1],
        data.startPositions[i * 3 + 2]
      );
      box.expandByPoint(v);
      v.set(data.endPositions[i * 3], data.endPositions[i * 3 + 1], data.endPositions[i * 3 + 2]);
      box.expandByPoint(v);

      const sw = data.startWidths[i];
      const ew = data.endWidths[i];
      if (Number.isFinite(sw) && sw > maxWidth) maxWidth = sw;
      if (Number.isFinite(ew) && ew > maxWidth) maxWidth = ew;
    }
    if (count > 0 && maxWidth > 0) {
      box.expandByScalar(maxWidth);
    }

    geometry.boundingBox = box;
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    geometry.boundingSphere = sphere;
  }

  // =========================================================================
  // GSplats Geometry Management
  // =========================================================================

  /**
   * Acquire geometry for GSplats (instanced per-splat attributes).
   */
  acquireGSplatsGeometry(nodeId: string, splatCount: number): THREE.InstancedBufferGeometry {
    const active = this.activeBuffers.get(nodeId);
    if (active && active.type === 'gsplats') {
      if (active.capacity >= splatCount) {
        active.lastUsedFrame = this.frameCount;
        this.stats.reuses++;
        this.typeStats.gsplats.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      } else {
        this.growGSplatsGeometry(active.geometry as THREE.InstancedBufferGeometry, splatCount);
        active.capacity = Math.ceil(splatCount * 1.5);
        active.lastUsedFrame = this.frameCount;
        this.stats.capacityGrowths++;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
    }

    // Try to find in pool (search across all buckets)
    for (const pooled of this.gsplatBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= splatCount) {
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          this.typeStats.gsplats.reuses++;
          return candidate.geometry as THREE.InstancedBufferGeometry;
        }
      }
    }

    // Allocate new
    const capacity = Math.ceil(splatCount * 1.5);
    const geometry = this.createGSplatsGeometry(capacity);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'gsplats',
      inUse: true,
      lastUsedFrame: this.frameCount,
    };

    this.activeBuffers.set(nodeId, newBuffer);
    this.stats.allocations++;
    this.typeStats.gsplats.allocations++;

    return geometry;
  }

  /**
   * Release GSplats geometry back to pool.
   */
  releaseGSplatsGeometry(nodeId: string): void {
    const buffer = this.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'gsplats') return;

    this.activeBuffers.delete(nodeId);
    buffer.inUse = false;

    const bucket = this.getBucket(buffer.capacity);
    if (!this.gsplatBuffers.has(bucket)) {
      this.gsplatBuffers.set(bucket, []);
    }
    this.gsplatBuffers.get(bucket)!.push(buffer);

    this.evictUnused();
  }

  /**
   * Create GSplats geometry (InstancedBufferGeometry with per-splat attributes).
   */
  private createGSplatsGeometry(splatCapacity: number): THREE.InstancedBufferGeometry {
    const geometry = new THREE.InstancedBufferGeometry();

    // Base quad
    const quadPositions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadPositions, 2));
    geometry.setIndex([0, 1, 2, 2, 1, 3]);

    // Pre-allocate the per-splat interleaved buffer at `splatCapacity`.
    const specsWithData: InterleavedAttributeSpec[] = GSPLATS_ATTRIBUTE_SPECS.map((spec) => ({
      ...spec,
      data: new Float32Array(splatCapacity * spec.itemSize),
    }));
    const { buffer, views } = packInterleavedAttributes(specsWithData, splatCapacity);
    buffer.setUsage(THREE.DynamicDrawUsage);
    for (const spec of specsWithData) {
      geometry.setAttribute(spec.name, views[spec.name]);
    }

    return geometry;
  }

  /**
   * Grow GSplats geometry to new capacity.
   */
  private growGSplatsGeometry(geometry: THREE.InstancedBufferGeometry, neededCount: number): void {
    const newCapacity = Math.ceil(neededCount * 1.5);

    // D.3: invalidate cached byte estimate before re-allocating any attribute.
    invalidateCachedByteSize(geometry);

    rebuildInterleavedBuffer(geometry, newCapacity, GSPLATS_ATTRIBUTE_SPECS);
  }

  /**
   * Update GSplats geometry in place.
   *
   * @param truncationRadius - Truncation radius in sigmas (default 3.0).
   *   Must match the material's truncationRadius for correct frustum culling.
   */
  updateGSplatsGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: PackedGSplatsData,
    count: number,
    truncationRadius: number = 3.0
  ): void {
    const updates: Array<[string, Float32Array]> = [
      ['aCenter', data.centers3D],
      ['aCholesky01', data.cholesky01],
      ['aCholesky23', data.cholesky23],
      ['aCholesky45', data.cholesky45],
      ['aAmplitude', data.amplitudes],
      ['aColor', data.colors],
    ];
    for (const [name, source] of updates) {
      writePooledAttribute(geometry, name, source, count);
    }

    geometry.instanceCount = count;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    // Without this, geometries initially created with 0 instances cache _maxInstanceCount=0,
    // causing the renderer to draw min(instanceCount, 0) = 0 instances even after updating.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    // CRITICAL: Recompute bounding box from updated center positions
    // GSplats use aCenter attribute for positions in frustum culling
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      v.set(data.centers3D[i * 3], data.centers3D[i * 3 + 1], data.centers3D[i * 3 + 2]);
      box.expandByPoint(v);
    }

    // Expand bounding box by max splat extent for correct frustum culling.
    // Without this, large splats whose center is outside the frustum but whose
    // visible body extends into view would cause the entire mesh to be culled.
    //
    // For each splat, the per-axis extent is truncationRadius × σ_d, where
    // σ_d = ||L[d,:]|| (the row norm of the Cholesky factor). We use the max
    // row norm across all splats and axes as a conservative expansion.
    //
    // Cholesky layout (packed as attribute pairs):
    //   cholesky01 = [L00, L10], cholesky23 = [L11, L20], cholesky45 = [L21, L22]
    // Row norms: ||row0|| = |L00|, ||row1|| = sqrt(L10² + L11²),
    //            ||row2|| = sqrt(L20² + L21² + L22²)
    let maxRowNorm = 0;
    for (let i = 0; i < count; i++) {
      const L00 = data.cholesky01[i * 2];
      const L10 = data.cholesky01[i * 2 + 1];
      const L11 = data.cholesky23[i * 2];
      const L20 = data.cholesky23[i * 2 + 1];
      const L21 = data.cholesky45[i * 2];
      const L22 = data.cholesky45[i * 2 + 1];

      const row0 = Math.abs(L00);
      const row1 = Math.sqrt(L10 * L10 + L11 * L11);
      const row2 = Math.sqrt(L20 * L20 + L21 * L21 + L22 * L22);
      maxRowNorm = Math.max(maxRowNorm, row0, row1, row2);
    }
    const expansion = maxRowNorm * truncationRadius;
    box.expandByScalar(expansion);

    geometry.boundingBox = box;
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    geometry.boundingSphere = sphere;
  }

  // =========================================================================
  // Pool Management
  // =========================================================================

  /**
   * Get size bucket for capacity-based pooling.
   * Buckets: 1K, 5K, 10K, 50K, 100K, 500K, 1M
   */
  private getBucket(count: number): number {
    if (count <= 1000) return 1000;
    if (count <= 5000) return 5000;
    if (count <= 10000) return 10000;
    if (count <= 50000) return 50000;
    if (count <= 100000) return 100000;
    if (count <= 500000) return 500000;
    return 1000000;
  }

  /**
   * Evict unused geometries using LRU policy.
   * Returns number of geometries evicted.
   *
   * Public for testing and manual pool management.
   */
  evictUnused(): number {
    let evicted = 0;
    const currentFrame = this.frameCount;

    // Check total pool size
    const totalPooled =
      Array.from(this.pointBuffers.values()).reduce((sum, arr) => sum + arr.length, 0) +
      Array.from(this.lineBuffers.values()).reduce((sum, arr) => sum + arr.length, 0) +
      Array.from(this.gsplatBuffers.values()).reduce((sum, arr) => sum + arr.length, 0);

    // If pool is over limit, evict aggressively
    const mustEvict = totalPooled > this.maxPoolSize;

    // Per-call eviction batch cap. Without this, a frame in which many
    // buckets simultaneously cross the eviction threshold (common after
    // a long pause + a viewport change) would burst-dispose every
    // qualifying buffer in a single frame. Each `geometry.dispose()`
    // can take 5–20 ms on slow GPUs; a 50-buffer burst stutters
    // visibly. By capping the per-call batch, the remaining evictable
    // buffers are deferred to the next frame's `acquire*` call. The
    // `mustEvict` over-limit path bypasses the cap so we never let the
    // pool drift unbounded above its limit.
    const batchCap = mustEvict ? Number.POSITIVE_INFINITY : this.evictBatchSize;

    // Helper to evict from a pool, returns count evicted. Stops early
    // when the per-call budget is exhausted. Buffers that WERE
    // eviction-eligible but couldn't run this call (batch cap exhausted)
    // are counted as `deferredEvictions` on the global stats so callers
    // can observe the eviction queue stretching across frames.
    const evictFromPool = (pool: Map<number, PooledBuffer[]>, budget: number): number => {
      let poolEvicted = 0;
      if (budget <= 0) return 0;
      for (const [bucket, buffers] of pool.entries()) {
        const kept: PooledBuffer[] = [];

        for (const buffer of buffers) {
          const framesSinceUse = currentFrame - buffer.lastUsedFrame;
          const evictable =
            framesSinceUse > this.evictionFrames || (mustEvict && framesSinceUse > 60);

          // Evict if: unused for >evictionFrames OR pool over limit,
          // AND we're under the per-call batch cap.
          if (evictable && poolEvicted < budget) {
            buffer.geometry.dispose();
            poolEvicted++;
          } else {
            if (evictable) this.stats.deferredEvictions++;
            kept.push(buffer);
          }
        }

        if (kept.length > 0) {
          pool.set(bucket, kept);
        } else {
          pool.delete(bucket);
        }

        if (poolEvicted >= budget) break;
      }
      return poolEvicted;
    };

    const pointsEvicted = evictFromPool(this.pointBuffers, batchCap);
    const remaining1 = batchCap === Number.POSITIVE_INFINITY ? batchCap : batchCap - pointsEvicted;
    const linesEvicted = evictFromPool(this.lineBuffers, remaining1);
    const remaining2 = batchCap === Number.POSITIVE_INFINITY ? batchCap : remaining1 - linesEvicted;
    const gsplatsEvicted = evictFromPool(this.gsplatBuffers, remaining2);

    evicted = pointsEvicted + linesEvicted + gsplatsEvicted;
    this.stats.evictions += evicted;
    this.typeStats.points.evictions += pointsEvicted;
    this.typeStats.lines.evictions += linesEvicted;
    this.typeStats.gsplats.evictions += gsplatsEvicted;

    // byte-budget pass. Independent of the count-based budget above.
    // Disposes pooled buffers (largest-first) until `pooledBytes` is
    // under `maxPoolBytes`. Without this, a single 760 MB Lines buffer
    // would sit in the pool indefinitely so long as the buffer count
    // stayed under `gpuPoolMaxSize`.
    if (this.maxPoolBytes > 0) {
      const byteEvicted = this._evictUntilUnderByteBudget();
      evicted += byteEvicted;
      this.stats.evictions += byteEvicted;
    }

    if (evicted > 0) {
      log.info(Modules.GPU_BUFFER_POOL, `Evicted ${evicted} unused geometries (LRU + byte-budget)`);
    }

    return evicted;
  }

  /**
   * dispose pooled buffers (largest-first across all type pools)
   * until `getPooledBytes()` is under `maxPoolBytes`. Returns the
   * number disposed. Does NOT touch active buffers.
   */
  private _evictUntilUnderByteBudget(): number {
    // Single-pass collect + sort, then walk. We collect every pooled
    // buffer with its byte size once, sort largest-first, and walk until
    // under budget. Disposal happens in place and we splice from the
    // original pool maps after the walk.
    type Ref = {
      pool: Map<number, PooledBuffer[]>;
      bucket: number;
      index: number;
      buffer: PooledBuffer;
      bytes: number;
    };
    const refs: Ref[] = [];
    const collect = (pool: Map<number, PooledBuffer[]>): void => {
      for (const [bucket, arr] of pool.entries()) {
        for (let i = 0; i < arr.length; i++) {
          const buffer = arr[i];
          refs.push({
            pool,
            bucket,
            index: i,
            buffer,
            bytes: estimateGeometryBytes(buffer.geometry),
          });
        }
      }
    };
    collect(this.pointBuffers);
    collect(this.lineBuffers);
    collect(this.gsplatBuffers);

    // One-shot warning when a pooled buffer crosses 100 MB. Such
    // buffers are usually correct (huge Lines/GSplats datasets), but
    // the size class makes a single eviction pause a frame visibly,
    // and silent gigabyte-class accumulation is the failure mode
    // worth flagging. Bound the noise: emit at most once per pool
    // instance.
    if (!this.largePoolWarningEmitted) {
      const LARGE_POOLED_BYTES_THRESHOLD = 100_000_000; // 100 MB
      const largest = refs.reduce((max, r) => (r.bytes > max ? r.bytes : max), 0);
      if (largest > LARGE_POOLED_BYTES_THRESHOLD) {
        this.largePoolWarningEmitted = true;
        log.warning(
          Modules.GPU_BUFFER_POOL,
          `Pooled buffer of ${(largest / 1024 / 1024).toFixed(1)} MB exceeds the ` +
            '100 MB diagnostic threshold. Eviction of this buffer will pause a frame ' +
            '(geometry.dispose can take 5-20 ms on slow GPUs).'
        );
      }
    }

    const totalBytes = refs.reduce((sum, r) => sum + r.bytes, 0);
    if (totalBytes <= this.maxPoolBytes) return 0;

    const targets = selectBuffersToEvict(refs, this.maxPoolBytes, totalBytes);

    // Bound the eviction count even though selectBuffersToEvict is
    // finite — defensive guard against malformed ref shapes that don't
    // reduce bytes when disposed.
    const maxIterations = Math.max(this.maxPoolSize * 3, 16);
    const evictCount = Math.min(targets.length, maxIterations);
    if (targets.length > maxIterations) {
      log.warning(
        Modules.GPU_BUFFER_POOL,
        `Byte-budget eviction capped at ${maxIterations} of ${targets.length} ` +
          'selected buffers. Pool may still be over budget after this pass.'
      );
    }

    // Splice from the lowest-index entries first WITHIN each (pool,bucket)
    // so later splices don't invalidate earlier indices. We sort the
    // evictions by descending index within their bucket arrays.
    const evicted = targets.slice(0, evictCount);
    // Group by (pool, bucket) then sort descending by index.
    const grouped = new Map<Map<number, PooledBuffer[]>, Map<number, Ref[]>>();
    for (const ref of evicted) {
      let byBucket = grouped.get(ref.pool);
      if (!byBucket) {
        byBucket = new Map();
        grouped.set(ref.pool, byBucket);
      }
      let bucketRefs = byBucket.get(ref.bucket);
      if (!bucketRefs) {
        bucketRefs = [];
        byBucket.set(ref.bucket, bucketRefs);
      }
      bucketRefs.push(ref);
    }
    for (const [pool, byBucket] of grouped) {
      for (const [bucket, bucketRefs] of byBucket) {
        bucketRefs.sort((a, b) => b.index - a.index);
        const arr = pool.get(bucket);
        if (!arr) continue;
        for (const ref of bucketRefs) {
          ref.buffer.geometry.dispose();
          arr.splice(ref.index, 1);
          this.typeStats[ref.buffer.type].evictions++;
        }
        if (arr.length === 0) pool.delete(bucket);
      }
    }
    return evicted.length;
  }

  /**
   * Get pool statistics with per-type breakdown.
   */
  getStats(): PoolStats {
    // Calculate per-type pooled buffers
    const pointsPooled = Array.from(this.pointBuffers.values()).reduce(
      (sum, arr) => sum + arr.length,
      0
    );
    const linesPooled = Array.from(this.lineBuffers.values()).reduce(
      (sum, arr) => sum + arr.length,
      0
    );
    const gsplatsPooled = Array.from(this.gsplatBuffers.values()).reduce(
      (sum, arr) => sum + arr.length,
      0
    );

    // Calculate per-type active buffers and byte totals.
    let pointsActive = 0;
    let linesActive = 0;
    let gsplatsActive = 0;
    let pointsActiveBytes = 0;
    let linesActiveBytes = 0;
    let gsplatsActiveBytes = 0;
    for (const buffer of this.activeBuffers.values()) {
      const bytes = estimateGeometryBytes(buffer.geometry);
      if (buffer.type === 'points') {
        pointsActive++;
        pointsActiveBytes += bytes;
      } else if (buffer.type === 'lines') {
        linesActive++;
        linesActiveBytes += bytes;
      } else if (buffer.type === 'gsplats') {
        gsplatsActive++;
        gsplatsActiveBytes += bytes;
      }
    }

    // per-type pooled bytes + largest pooled buffer.
    let pointsPooledBytes = 0;
    let linesPooledBytes = 0;
    let gsplatsPooledBytes = 0;
    let largestPooledBytes = 0;
    for (const arr of this.pointBuffers.values()) {
      for (const b of arr) {
        const bytes = estimateGeometryBytes(b.geometry);
        pointsPooledBytes += bytes;
        if (bytes > largestPooledBytes) largestPooledBytes = bytes;
      }
    }
    for (const arr of this.lineBuffers.values()) {
      for (const b of arr) {
        const bytes = estimateGeometryBytes(b.geometry);
        linesPooledBytes += bytes;
        if (bytes > largestPooledBytes) largestPooledBytes = bytes;
      }
    }
    for (const arr of this.gsplatBuffers.values()) {
      for (const b of arr) {
        const bytes = estimateGeometryBytes(b.geometry);
        gsplatsPooledBytes += bytes;
        if (bytes > largestPooledBytes) largestPooledBytes = bytes;
      }
    }

    const activeBytes = pointsActiveBytes + linesActiveBytes + gsplatsActiveBytes;
    const pooledBytes = pointsPooledBytes + linesPooledBytes + gsplatsPooledBytes;

    return {
      ...this.stats,
      activeBuffers: this.activeBuffers.size,
      pooledBuffers: pointsPooled + linesPooled + gsplatsPooled,
      activeBytes,
      pooledBytes,
      totalBytes: activeBytes + pooledBytes,
      largestPooledBytes,
      byType: {
        points: {
          allocations: this.typeStats.points.allocations,
          reuses: this.typeStats.points.reuses,
          evictions: this.typeStats.points.evictions,
          activeBuffers: pointsActive,
          pooledBuffers: pointsPooled,
          activeBytes: pointsActiveBytes,
          pooledBytes: pointsPooledBytes,
        },
        lines: {
          allocations: this.typeStats.lines.allocations,
          reuses: this.typeStats.lines.reuses,
          evictions: this.typeStats.lines.evictions,
          activeBuffers: linesActive,
          pooledBuffers: linesPooled,
          activeBytes: linesActiveBytes,
          pooledBytes: linesPooledBytes,
        },
        gsplats: {
          allocations: this.typeStats.gsplats.allocations,
          reuses: this.typeStats.gsplats.reuses,
          evictions: this.typeStats.gsplats.evictions,
          activeBuffers: gsplatsActive,
          pooledBuffers: gsplatsPooled,
          activeBytes: gsplatsActiveBytes,
          pooledBytes: gsplatsPooledBytes,
        },
      },
    };
  }

  /**
   * Dispose all pooled geometries (for cleanup or context loss).
   */
  dispose(): void {
    // Dispose all active geometries
    for (const buffer of this.activeBuffers.values()) {
      buffer.geometry.dispose();
    }
    this.activeBuffers.clear();

    // Dispose all pooled geometries
    const disposePool = (pool: Map<number, PooledBuffer[]>) => {
      for (const buffers of pool.values()) {
        for (const buffer of buffers) {
          buffer.geometry.dispose();
        }
      }
      pool.clear();
    };

    disposePool(this.pointBuffers);
    disposePool(this.lineBuffers);
    disposePool(this.gsplatBuffers);

    log.info(Modules.GPU_BUFFER_POOL, 'All pooled geometries disposed');
  }
}
