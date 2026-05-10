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
import type { LoadedPointsData } from '../data/data-loader-types';
import type { ProcessedLinesData } from '../types/lines';

/**
 * Packed GSplats data ready for GPU upload (from gsplats-processor.ts)
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
  // Per-type breakdown
  byType: {
    points: TypePoolStats;
    lines: TypePoolStats;
    gsplats: TypePoolStats;
  };
}

/**
 * estimate the GPU-resident byte footprint of a geometry by
 * summing the underlying typed-array byte lengths of every attribute
 * (and the index, if present). Mirrors what THREE.js will actually
 * upload — it slightly overstates because we count the full backing
 * array even if `count < array.length / itemSize`, but that's the
 * footprint that matters for pool memory pressure.
 */
export function estimateGeometryBytes(geometry: THREE.BufferGeometry): number {
  let total = 0;
  for (const name in geometry.attributes) {
    const attr = geometry.attributes[name] as THREE.BufferAttribute;
    const arr = attr.array as ArrayBufferView | undefined;
    if (arr && typeof arr.byteLength === 'number') {
      total += arr.byteLength;
    }
  }
  // InstancedBufferGeometry indices are shared with the base geometry
  // (a single quad), so they're a fixed overhead — small, but include
  // them for correctness.
  if (geometry.index) {
    const idxArr = geometry.index.array as ArrayBufferView | undefined;
    if (idxArr && typeof idxArr.byteLength === 'number') {
      total += idxArr.byteLength;
    }
  }
  return total;
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
  };

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
          return active.geometry as THREE.BufferGeometry;
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
          return active.geometry as THREE.BufferGeometry;
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
          return candidate.geometry as THREE.BufferGeometry;
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

    return geometry;
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
   * Create Points geometry with type-specific attributes
   */
  private createPointsGeometry(
    capacity: number,
    types: PointsAttributeTypes
  ): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();

    // Position: Always Float32Array
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(capacity * 3, 3));

    // Color: Type-specific with normalization for Uint8/Uint16
    const colorNormalized = types.color !== 'Float32Array';
    if (types.color === 'Uint8Array') {
      const attr = new THREE.BufferAttribute(new Uint8Array(capacity * 3), 3, colorNormalized);
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('color', attr);
    } else if (types.color === 'Uint16Array') {
      const attr = new THREE.BufferAttribute(new Uint16Array(capacity * 3), 3, colorNormalized);
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('color', attr);
    } else {
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(capacity * 3, 3));
    }

    // Radius: Type-specific with normalization for Uint8
    const radiusNormalized = types.radius === 'Uint8Array';
    if (types.radius === 'Uint8Array') {
      const attr = new THREE.BufferAttribute(new Uint8Array(capacity), 1, radiusNormalized);
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('radius', attr);
    } else {
      geometry.setAttribute('radius', new THREE.Float32BufferAttribute(capacity, 1));
    }

    // Sharpness: Type-specific with normalization for Uint8
    const sharpnessNormalized = types.sharpness === 'Uint8Array';
    if (types.sharpness === 'Uint8Array') {
      const attr = new THREE.BufferAttribute(new Uint8Array(capacity), 1, sharpnessNormalized);
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('sharpness', attr);
    } else {
      geometry.setAttribute('sharpness', new THREE.Float32BufferAttribute(capacity, 1));
    }

    // scalar attribute — only created when scalars are present.
    // The shader reads `scalar` only under USE_COLORMAP, so omitting the
    // attribute when types.scalar is undefined avoids carrying a 4 B/point
    // empty buffer for every non-colormap dataset.
    if (types.scalar === 'Uint8Array') {
      const attr = new THREE.BufferAttribute(new Uint8Array(capacity), 1, /*normalized*/ true);
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('scalar', attr);
    } else if (types.scalar === 'Float16Array') {
      // Float16Array isn't an accepted THREE.js BufferAttribute storage,
      // so the shader receives Float32 — we widen at upload time. The
      // distinction is preserved in types for accurate reuse matching.
      geometry.setAttribute('scalar', new THREE.Float32BufferAttribute(capacity, 1));
    } else if (types.scalar === 'Float32Array') {
      geometry.setAttribute('scalar', new THREE.Float32BufferAttribute(capacity, 1));
    }

    // Set dynamic usage for Float32 attributes
    for (const key in geometry.attributes) {
      const attr = geometry.attributes[key];
      if (attr instanceof THREE.Float32BufferAttribute) {
        attr.setUsage(THREE.DynamicDrawUsage);
      }
    }

    return geometry;
  }

  /**
   * Grow Points geometry to new capacity (reallocates attributes with same types).
   */
  private growPointsGeometry(
    geometry: THREE.BufferGeometry,
    neededCount: number,
    types: PointsAttributeTypes
  ): void {
    const newCapacity = Math.ceil(neededCount * 1.5);

    // Position: Always Float32Array
    const oldPos = geometry.getAttribute('position') as THREE.BufferAttribute;
    const newPos = new THREE.Float32BufferAttribute(newCapacity * 3, 3);
    (newPos.array as Float32Array).set(oldPos.array as Float32Array);
    newPos.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', newPos);

    // Color: Type-preserving growth
    const oldCol = geometry.getAttribute('color') as THREE.BufferAttribute;
    const colorNormalized = types.color !== 'Float32Array';
    if (types.color === 'Uint8Array') {
      const newCol = new THREE.BufferAttribute(new Uint8Array(newCapacity * 3), 3, colorNormalized);
      (newCol.array as Uint8Array).set(oldCol.array as Uint8Array);
      newCol.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('color', newCol);
    } else if (types.color === 'Uint16Array') {
      const newCol = new THREE.BufferAttribute(
        new Uint16Array(newCapacity * 3),
        3,
        colorNormalized
      );
      (newCol.array as Uint16Array).set(oldCol.array as Uint16Array);
      newCol.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('color', newCol);
    } else {
      const newCol = new THREE.Float32BufferAttribute(newCapacity * 3, 3);
      (newCol.array as Float32Array).set(oldCol.array as Float32Array);
      newCol.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('color', newCol);
    }

    // Radius: Type-preserving growth
    const oldRad = geometry.getAttribute('radius') as THREE.BufferAttribute;
    const radiusNormalized = types.radius === 'Uint8Array';
    if (types.radius === 'Uint8Array') {
      const newRad = new THREE.BufferAttribute(new Uint8Array(newCapacity), 1, radiusNormalized);
      (newRad.array as Uint8Array).set(oldRad.array as Uint8Array);
      newRad.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('radius', newRad);
    } else {
      const newRad = new THREE.Float32BufferAttribute(newCapacity, 1);
      (newRad.array as Float32Array).set(oldRad.array as Float32Array);
      newRad.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('radius', newRad);
    }

    // Sharpness: Type-preserving growth
    const oldSharp = geometry.getAttribute('sharpness') as THREE.BufferAttribute;
    const sharpnessNormalized = types.sharpness === 'Uint8Array';
    if (types.sharpness === 'Uint8Array') {
      const newSharp = new THREE.BufferAttribute(
        new Uint8Array(newCapacity),
        1,
        sharpnessNormalized
      );
      (newSharp.array as Uint8Array).set(oldSharp.array as Uint8Array);
      newSharp.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('sharpness', newSharp);
    } else {
      const newSharp = new THREE.Float32BufferAttribute(newCapacity, 1);
      (newSharp.array as Float32Array).set(oldSharp.array as Float32Array);
      newSharp.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute('sharpness', newSharp);
    }

    // Scalar — type-preserving growth, only when present.
    if (types.scalar) {
      const oldScalar = geometry.getAttribute('scalar') as THREE.BufferAttribute | undefined;
      if (types.scalar === 'Uint8Array') {
        const newScalar = new THREE.BufferAttribute(
          new Uint8Array(newCapacity),
          1,
          /*normalized*/ true
        );
        if (oldScalar) (newScalar.array as Uint8Array).set(oldScalar.array as Uint8Array);
        newScalar.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('scalar', newScalar);
      } else {
        // Float16Array and Float32Array both stage into a Float32 GPU
        // attribute — the type tag preserves dtype for reuse matching.
        const newScalar = new THREE.Float32BufferAttribute(newCapacity, 1);
        if (oldScalar) (newScalar.array as Float32Array).set(oldScalar.array as Float32Array);
        newScalar.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('scalar', newScalar);
      }
    }
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
    // Update positions (always Float32Array)
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    (posAttr.array as Float32Array).set(data.positions.subarray(0, count * 3) as Float32Array);
    posAttr.needsUpdate = true;

    // Update colors (type-matched: Uint8Array, Uint16Array, or Float32Array).
    //
    // fill with white defaults when `data.colors` is absent. Without
    // this, the buffer's initial zeros render as black points (the shader
    // discards near-zero adjusted color, so positions-only Points become
    // invisible). When the type is Float32 (the default when `data.colors`
    // is undefined — see `detectAttributeTypes`), fill 1.0; for typed
    // integer buffers we still write 0xFF to be defensive against type
    // changes during reuse.
    const colAttr = geometry.getAttribute('color') as THREE.BufferAttribute;
    if (data.colors) {
      // TypedArray.set() works correctly when source and destination have same type
      // The geometry was created with matching type, so this is safe
      if (data.colors instanceof Uint8Array) {
        (colAttr.array as Uint8Array).set(data.colors.subarray(0, count * 3) as Uint8Array);
      } else if (data.colors instanceof Uint16Array) {
        (colAttr.array as Uint16Array).set(data.colors.subarray(0, count * 3) as Uint16Array);
      } else {
        (colAttr.array as Float32Array).set(data.colors.subarray(0, count * 3) as Float32Array);
      }
    } else {
      const colArr = colAttr.array;
      const fill =
        colArr instanceof Uint8Array ? 0xff : colArr instanceof Uint16Array ? 0xffff : 1.0;
      for (let i = 0; i < count * 3; i++) {
        colArr[i] = fill;
      }
    }
    colAttr.needsUpdate = true;

    // Update radii (type-matched: Uint8Array or Float32Array).
    //
    // fill default 0.5 when `data.radii` is absent (matches
    // NodeFactory.createPointsGeometry). When the type is Uint8
    // (normalized via radiusScale = max_radius), 0.5 maps to byte 128;
    // when Float32, write 0.5 directly.
    const radAttr = geometry.getAttribute('radius') as THREE.BufferAttribute;
    if (data.radii) {
      if (data.radii instanceof Uint8Array) {
        (radAttr.array as Uint8Array).set(data.radii.subarray(0, count) as Uint8Array);
      } else {
        (radAttr.array as Float32Array).set(data.radii.subarray(0, count) as Float32Array);
      }
    } else {
      const radArr = radAttr.array;
      const fill = radArr instanceof Uint8Array ? 128 : 0.5;
      for (let i = 0; i < count; i++) {
        radArr[i] = fill;
      }
    }
    radAttr.needsUpdate = true;

    // Update sharpness (type-matched: Uint8Array or Float32Array).
    //
    // fill default 2.0 when `data.sharpness` is absent. Float32 path
    // writes 2.0 directly; Uint8 path uses 64 (≈2.0/8 * 255 — assumes
    // typical max_sharpness ~31 means scaled value falls in usable range).
    const sharpAttr = geometry.getAttribute('sharpness') as THREE.BufferAttribute;
    if (data.sharpness) {
      if (data.sharpness instanceof Uint8Array) {
        (sharpAttr.array as Uint8Array).set(data.sharpness.subarray(0, count) as Uint8Array);
      } else {
        (sharpAttr.array as Float32Array).set(data.sharpness.subarray(0, count) as Float32Array);
      }
    } else {
      const sharpArr = sharpAttr.array;
      const fill = sharpArr instanceof Uint8Array ? 64 : 2.0;
      for (let i = 0; i < count; i++) {
        sharpArr[i] = fill;
      }
    }
    sharpAttr.needsUpdate = true;

    // scalar attribute. The geometry only carries `scalar` when
    // detectAttributeTypes saw scalars at acquire time. When the data
    // dropped scalars on a later commit (rare — types would mismatch
    // and the pool would re-allocate), nothing to do here.
    const scalarAttr = geometry.getAttribute('scalar') as THREE.BufferAttribute | undefined;
    if (scalarAttr) {
      if (data.scalars) {
        if (data.scalars instanceof Uint8Array) {
          (scalarAttr.array as Uint8Array).set(data.scalars.subarray(0, count) as Uint8Array);
        } else if (
          typeof globalThis.Float16Array !== 'undefined' &&
          data.scalars instanceof globalThis.Float16Array
        ) {
          // widen Float16 → Float32 elementwise since TypedArray.set
          // doesn't accept Float16Array as a source for Float32 attribute
          // storage in current JS engines.
          const src = data.scalars as Float16Array;
          const dst = scalarAttr.array as Float32Array;
          const n = Math.min(count, src.length);
          for (let i = 0; i < n; i++) dst[i] = src[i];
        } else {
          (scalarAttr.array as Float32Array).set(
            data.scalars.subarray(0, count) as Float32Array
          );
        }
      } else {
        // No source scalars but the buffer exists — fill zero so a
        // colormap LUT lookup at scalar=0 returns the LUT's first
        // entry (equivalent to disabling colormap visually).
        const arr = scalarAttr.array;
        for (let i = 0; i < count; i++) arr[i] = 0;
      }
      scalarAttr.needsUpdate = true;
    }

    // Update draw range
    geometry.setDrawRange(0, count);

    // CRITICAL: Recompute bounding box after position updates
    // Same issue as Lines/GSplats - positions change per time slice, bounding box must update
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
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

    // Per-segment instanced attributes (ProcessedLinesData format)
    const attrs = [
      ['aStartPos', 3],
      ['aEndPos', 3],
      ['aStartColor', 3],
      ['aEndColor', 3],
      ['aStartWidth', 1],
      ['aEndWidth', 1],
      ['aStartSharpness', 1],
      ['aEndSharpness', 1],
      ['aSegmentLength', 1],
      ['aStartClipped', 1],
      ['aEndClipped', 1],
    ] as const;

    for (const [name, size] of attrs) {
      const attr = new THREE.InstancedBufferAttribute(
        new Float32Array(segmentCapacity * size),
        size
      );
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attr);
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

    const attrNames = [
      'aStartPos',
      'aEndPos',
      'aStartColor',
      'aEndColor',
      'aStartWidth',
      'aEndWidth',
      'aStartSharpness',
      'aEndSharpness',
      'aSegmentLength',
      'aStartClipped',
      'aEndClipped',
    ];

    for (const name of attrNames) {
      const oldAttr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      const size = oldAttr.itemSize;
      const newAttr = new THREE.InstancedBufferAttribute(
        new Float32Array(newCapacity * size),
        size
      );
      (newAttr.array as Float32Array).set(oldAttr.array as Float32Array);
      newAttr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, newAttr);
    }

    // Scalar attributes are optional (allocated lazily on first scalar
    // commit). When they exist, grow them to match the rest of the
    // per-segment attributes.
    for (const name of ['aStartScalar', 'aEndScalar']) {
      const oldAttr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute | undefined;
      if (!oldAttr) continue;
      const newAttr = new THREE.InstancedBufferAttribute(new Float32Array(newCapacity), 1);
      (newAttr.array as Float32Array).set(oldAttr.array as Float32Array);
      newAttr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, newAttr);
    }
  }

  /**
   * Update Lines geometry in place.
   */
  updateLinesGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: ProcessedLinesData,
    count: number
  ): void {
    const attrs = [
      ['aStartPos', data.startPositions, 3],
      ['aEndPos', data.endPositions, 3],
      ['aStartColor', data.startColors, 3],
      ['aEndColor', data.endColors, 3],
      ['aStartWidth', data.startWidths, 1],
      ['aEndWidth', data.endWidths, 1],
      ['aStartSharpness', data.startSharpness, 1],
      ['aEndSharpness', data.endSharpness, 1],
      ['aSegmentLength', data.segmentLengths, 1],
      ['aStartClipped', data.startClipped, 1],
      ['aEndClipped', data.endClipped, 1],
    ] as const;

    for (const [name, sourceData, size] of attrs) {
      const attr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      (attr.array as Float32Array).set(sourceData.subarray(0, count * size));
      attr.needsUpdate = true;
    }

    // Lazily allocate aStartScalar/aEndScalar when the source has
    // scalars. Grow attributes on first commit and update in place
    // afterward (capacity tracked from aStartPos, which is always there).
    if (data.startScalars && data.endScalars) {
      const startPosAttr = geometry.getAttribute('aStartPos') as THREE.InstancedBufferAttribute;
      const capacity = (startPosAttr.array as Float32Array).length / 3;
      let startScalarAttr = geometry.getAttribute('aStartScalar') as
        | THREE.InstancedBufferAttribute
        | undefined;
      let endScalarAttr = geometry.getAttribute('aEndScalar') as
        | THREE.InstancedBufferAttribute
        | undefined;
      if (!startScalarAttr) {
        startScalarAttr = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
        startScalarAttr.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('aStartScalar', startScalarAttr);
      }
      if (!endScalarAttr) {
        endScalarAttr = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
        endScalarAttr.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('aEndScalar', endScalarAttr);
      }
      // Grow lazily if the pre-existing attribute is too small (rare —
      // happens when a scalar dataset is committed after a non-scalar
      // commit grew aStartPos beyond the scalar buffer).
      if ((startScalarAttr.array as Float32Array).length < capacity) {
        const grown = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
        grown.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('aStartScalar', grown);
        startScalarAttr = grown;
      }
      if ((endScalarAttr.array as Float32Array).length < capacity) {
        const grown = new THREE.InstancedBufferAttribute(new Float32Array(capacity), 1);
        grown.setUsage(THREE.DynamicDrawUsage);
        geometry.setAttribute('aEndScalar', grown);
        endScalarAttr = grown;
      }
      (startScalarAttr.array as Float32Array).set(data.startScalars.subarray(0, count));
      startScalarAttr.needsUpdate = true;
      (endScalarAttr.array as Float32Array).set(data.endScalars.subarray(0, count));
      endScalarAttr.needsUpdate = true;
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

    // Per-splat instance attributes
    const attrs = [
      ['aCenter', 3],
      ['aCholesky01', 2],
      ['aCholesky23', 2],
      ['aCholesky45', 2],
      ['aAmplitude', 1],
      ['aColor', 3],
    ] as const;

    for (const [name, size] of attrs) {
      const attr = new THREE.InstancedBufferAttribute(new Float32Array(splatCapacity * size), size);
      attr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, attr);
    }

    return geometry;
  }

  /**
   * Grow GSplats geometry to new capacity.
   */
  private growGSplatsGeometry(geometry: THREE.InstancedBufferGeometry, neededCount: number): void {
    const newCapacity = Math.ceil(neededCount * 1.5);

    const attrNames = [
      'aCenter',
      'aCholesky01',
      'aCholesky23',
      'aCholesky45',
      'aAmplitude',
      'aColor',
    ];

    for (const name of attrNames) {
      const oldAttr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      const size = oldAttr.itemSize;
      const newAttr = new THREE.InstancedBufferAttribute(
        new Float32Array(newCapacity * size),
        size
      );
      (newAttr.array as Float32Array).set(oldAttr.array as Float32Array);
      newAttr.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute(name, newAttr);
    }
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
    const attrs = [
      ['aCenter', data.centers3D, 3],
      ['aCholesky01', data.cholesky01, 2],
      ['aCholesky23', data.cholesky23, 2],
      ['aCholesky45', data.cholesky45, 2],
      ['aAmplitude', data.amplitudes, 1],
      ['aColor', data.colors, 3],
    ] as const;

    for (const [name, sourceData, size] of attrs) {
      const attr = geometry.getAttribute(name) as THREE.InstancedBufferAttribute;
      (attr.array as Float32Array).set(sourceData.subarray(0, count * size));
      attr.needsUpdate = true;
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
    // when the per-call budget is exhausted.
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
    let disposed = 0;
    while (this._getPooledBytes() > this.maxPoolBytes) {
      const largest = this._findLargestPooledBuffer();
      if (!largest) break;
      const { pool, bucket, index, buffer } = largest;
      buffer.geometry.dispose();
      const arr = pool.get(bucket);
      if (arr) {
        arr.splice(index, 1);
        if (arr.length === 0) pool.delete(bucket);
      }
      this.typeStats[buffer.type].evictions++;
      disposed++;
    }
    return disposed;
  }

  /**
   * total bytes across all pooled (not active) buffers. Recomputed
   * on each call — pool sizes are small (≤ `maxPoolSize` ≈ 20), so
   * this is trivially cheap.
   */
  private _getPooledBytes(): number {
    let total = 0;
    for (const arr of this.pointBuffers.values()) {
      for (const b of arr) total += estimateGeometryBytes(b.geometry);
    }
    for (const arr of this.lineBuffers.values()) {
      for (const b of arr) total += estimateGeometryBytes(b.geometry);
    }
    for (const arr of this.gsplatBuffers.values()) {
      for (const b of arr) total += estimateGeometryBytes(b.geometry);
    }
    return total;
  }

  /**
   * locate the largest pooled buffer across all type pools, with
   * its containing pool / bucket / array index so the caller can
   * splice it out. Returns null when every pool is empty.
   */
  private _findLargestPooledBuffer(): {
    pool: Map<number, PooledBuffer[]>;
    bucket: number;
    index: number;
    buffer: PooledBuffer;
    bytes: number;
  } | null {
    let best: {
      pool: Map<number, PooledBuffer[]>;
      bucket: number;
      index: number;
      buffer: PooledBuffer;
      bytes: number;
    } | null = null;
    const consider = (pool: Map<number, PooledBuffer[]>) => {
      for (const [bucket, arr] of pool.entries()) {
        for (let i = 0; i < arr.length; i++) {
          const buffer = arr[i];
          const bytes = estimateGeometryBytes(buffer.geometry);
          if (!best || bytes > best.bytes) {
            best = { pool, bucket, index: i, buffer, bytes };
          }
        }
      }
    };
    consider(this.pointBuffers);
    consider(this.lineBuffers);
    consider(this.gsplatBuffers);
    return best;
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
