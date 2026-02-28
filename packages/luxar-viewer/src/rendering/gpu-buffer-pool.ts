/**
 * GPU Buffer Pool - Geometry reuse for Points, Lines, and GSplats
 *
 * Phase 4: Eliminates GPU buffer allocations by reusing BufferGeometry objects
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
  sharpness: Float32Array; // M
  splatCount: number;
}

/**
 * Attribute type information for Points geometry
 * Tracks the TypedArray type for each attribute to enable proper reuse
 */
interface PointsAttributeTypes {
  position: 'Float32Array'; // Always Float32Array for positions
  color: 'Float32Array' | 'Uint8Array' | 'Uint16Array';
  radius: 'Float32Array' | 'Uint8Array';
  sharpness: 'Float32Array' | 'Uint8Array';
}

interface PooledBuffer {
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
  // Per-type breakdown
  byType: {
    points: TypePoolStats;
    lines: TypePoolStats;
    gsplats: TypePoolStats;
  };
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

  constructor(maxPoolSize: number = 20, evictionFrames: number = 300) {
    this.maxPoolSize = maxPoolSize;
    this.evictionFrames = evictionFrames;
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
      a.sharpness === b.sharpness
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
    this.frameCount++;

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
    return {
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

    // Update colors (type-matched: Uint8Array, Uint16Array, or Float32Array)
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
    }
    colAttr.needsUpdate = true;

    // Update radii (type-matched: Uint8Array or Float32Array)
    const radAttr = geometry.getAttribute('radius') as THREE.BufferAttribute;
    if (data.radii) {
      if (data.radii instanceof Uint8Array) {
        (radAttr.array as Uint8Array).set(data.radii.subarray(0, count) as Uint8Array);
      } else {
        (radAttr.array as Float32Array).set(data.radii.subarray(0, count) as Float32Array);
      }
    }
    radAttr.needsUpdate = true;

    // Update sharpness (type-matched: Uint8Array or Float32Array)
    const sharpAttr = geometry.getAttribute('sharpness') as THREE.BufferAttribute;
    if (data.sharpness) {
      if (data.sharpness instanceof Uint8Array) {
        (sharpAttr.array as Uint8Array).set(data.sharpness.subarray(0, count) as Uint8Array);
      } else {
        (sharpAttr.array as Float32Array).set(data.sharpness.subarray(0, count) as Float32Array);
      }
    }
    sharpAttr.needsUpdate = true;

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
    this.frameCount++;

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

    // Update instance count
    geometry.instanceCount = count;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (geometry as any)._maxInstanceCount;

    // CRITICAL: Recompute bounding box after position updates
    // Without this, frustum culling uses stale bounds from previous frame/time slice
    // This causes geometry to disappear when zooming close (small frustum excludes stale box)
    // Performance: O(n) in segment count, but only runs when geometry updates (not every frame)
    const positions = new Float32Array(count * 6);
    for (let i = 0; i < count; i++) {
      positions[i * 6 + 0] = data.startPositions[i * 3 + 0];
      positions[i * 6 + 1] = data.startPositions[i * 3 + 1];
      positions[i * 6 + 2] = data.startPositions[i * 3 + 2];
      positions[i * 6 + 3] = data.endPositions[i * 3 + 0];
      positions[i * 6 + 4] = data.endPositions[i * 3 + 1];
      positions[i * 6 + 5] = data.endPositions[i * 3 + 2];
    }

    const tempGeometry = new THREE.BufferGeometry();
    tempGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    tempGeometry.computeBoundingBox();
    tempGeometry.computeBoundingSphere();

    if (tempGeometry.boundingBox) {
      geometry.boundingBox = tempGeometry.boundingBox.clone();
    }
    if (tempGeometry.boundingSphere) {
      geometry.boundingSphere = tempGeometry.boundingSphere.clone();
    }

    tempGeometry.dispose();
  }

  // =========================================================================
  // GSplats Geometry Management
  // =========================================================================

  /**
   * Acquire geometry for GSplats (instanced per-splat attributes).
   */
  acquireGSplatsGeometry(nodeId: string, splatCount: number): THREE.InstancedBufferGeometry {
    this.frameCount++;

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
      ['aSharpness', 1],
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
      'aSharpness',
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
   */
  updateGSplatsGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: PackedGSplatsData,
    count: number
  ): void {
    const attrs = [
      ['aCenter', data.centers3D, 3],
      ['aCholesky01', data.cholesky01, 2],
      ['aCholesky23', data.cholesky23, 2],
      ['aCholesky45', data.cholesky45, 2],
      ['aAmplitude', data.amplitudes, 1],
      ['aColor', data.colors, 3],
      ['aSharpness', data.sharpness, 1],
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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (geometry as any)._maxInstanceCount;

    // CRITICAL: Recompute bounding box from updated center positions
    // GSplats use aCenter attribute for positions in frustum culling
    const tempGeometry = new THREE.BufferGeometry();
    tempGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(data.centers3D.subarray(0, count * 3), 3)
    );
    tempGeometry.computeBoundingBox();
    tempGeometry.computeBoundingSphere();

    if (tempGeometry.boundingBox) {
      geometry.boundingBox = tempGeometry.boundingBox.clone();
    }
    if (tempGeometry.boundingSphere) {
      geometry.boundingSphere = tempGeometry.boundingSphere.clone();
    }

    tempGeometry.dispose();
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

    // Helper to evict from a pool, returns count evicted
    const evictFromPool = (pool: Map<number, PooledBuffer[]>): number => {
      let poolEvicted = 0;
      for (const [bucket, buffers] of pool.entries()) {
        const kept: PooledBuffer[] = [];

        for (const buffer of buffers) {
          const framesSinceUse = currentFrame - buffer.lastUsedFrame;

          // Evict if: unused for >evictionFrames OR pool over limit
          if (framesSinceUse > this.evictionFrames || (mustEvict && framesSinceUse > 60)) {
            // Dispose geometry
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
      }
      return poolEvicted;
    };

    const pointsEvicted = evictFromPool(this.pointBuffers);
    const linesEvicted = evictFromPool(this.lineBuffers);
    const gsplatsEvicted = evictFromPool(this.gsplatBuffers);

    evicted = pointsEvicted + linesEvicted + gsplatsEvicted;
    this.stats.evictions += evicted;
    this.typeStats.points.evictions += pointsEvicted;
    this.typeStats.lines.evictions += linesEvicted;
    this.typeStats.gsplats.evictions += gsplatsEvicted;

    if (evicted > 0) {
      log.info(Modules.GPU_BUFFER_POOL, `Evicted ${evicted} unused geometries (LRU policy)`);
    }

    return evicted;
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

    // Calculate per-type active buffers
    let pointsActive = 0;
    let linesActive = 0;
    let gsplatsActive = 0;
    for (const buffer of this.activeBuffers.values()) {
      if (buffer.type === 'points') pointsActive++;
      else if (buffer.type === 'lines') linesActive++;
      else if (buffer.type === 'gsplats') gsplatsActive++;
    }

    return {
      ...this.stats,
      activeBuffers: this.activeBuffers.size,
      pooledBuffers: pointsPooled + linesPooled + gsplatsPooled,
      byType: {
        points: {
          allocations: this.typeStats.points.allocations,
          reuses: this.typeStats.points.reuses,
          evictions: this.typeStats.points.evictions,
          activeBuffers: pointsActive,
          pooledBuffers: pointsPooled,
        },
        lines: {
          allocations: this.typeStats.lines.allocations,
          reuses: this.typeStats.lines.reuses,
          evictions: this.typeStats.lines.evictions,
          activeBuffers: linesActive,
          pooledBuffers: linesPooled,
        },
        gsplats: {
          allocations: this.typeStats.gsplats.allocations,
          reuses: this.typeStats.gsplats.reuses,
          evictions: this.typeStats.gsplats.evictions,
          activeBuffers: gsplatsActive,
          pooledBuffers: gsplatsPooled,
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
