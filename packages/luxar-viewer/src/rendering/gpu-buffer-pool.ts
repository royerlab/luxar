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
  type InterleavedAttributeSpec,
} from './interleaved-attributes';
import type { PooledBuffer, PoolStats } from './gpu-buffer-pool/pool-stats';
import { selectBuffersToEvict } from './gpu-buffer-pool/eviction-policy';
import {
  rebuildInterleavedBuffer,
  writePooledAttribute,
} from './gpu-buffer-pool/attribute-codec';
import { PointsBufferAdapter } from './gpu-buffer-pool/points-adapter';

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

// Points-specific spec arrays + helpers moved to ./gpu-buffer-pool/points-adapter.

// rebuildInterleavedBuffer + writePooledAttribute moved to
// ./gpu-buffer-pool/attribute-codec — geometry-agnostic helpers that
// the per-type adapters (carved out in step 3) will share.

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

// Pool-stats types moved to ./gpu-buffer-pool/pool-stats. Re-exported so
// existing consumers (importing from `rendering/gpu-buffer-pool`) keep
// working unchanged.
export type {
  PointsAttributeTypes,
  PooledBuffer,
  TypePoolStats,
  PoolStats,
  PooledBufferRef,
} from './gpu-buffer-pool/pool-stats';

// selectBuffersToEvict moved to ./gpu-buffer-pool/eviction-policy.
// Re-exported here so existing consumers keep working unchanged.
export { selectBuffersToEvict } from './gpu-buffer-pool/eviction-policy';

/**
 * GPU buffer pool for reusing THREE.BufferGeometry objects.
 *
 * Manages separate pools for Points, Lines, and GSplats geometries,
 * each with different attribute layouts and update patterns.
 */
export class GPUBufferPool {
  /** @internal — points-specific pool state and methods. */
  readonly points: PointsBufferAdapter;
  private lineBuffers = new Map<number, PooledBuffer[]>();
  private gsplatBuffers = new Map<number, PooledBuffer[]>();

  /** @internal — shared with the per-type adapters. */
  activeBuffers = new Map<string, PooledBuffer>(); // nodeId → active geometry
  /** @internal — shared with the per-type adapters; increments via beginFrame(). */
  frameCount = 0;

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

  /** @internal — shared mutable stats; adapters bump fields here. */
  stats = {
    allocations: 0,
    reuses: 0,
    evictions: 0,
    capacityGrowths: 0,
    /** Pooled buffers skipped this `evictUnused` call due to batch cap. */
    deferredEvictions: 0,
  };

  /**
   * Set by acquire methods when the returned geometry's underlying
   * `InstancedInterleavedBuffer` was re-allocated (grow path) or when
   * the returned geometry is a fresh allocation / different pool
   * candidate from the previous active one. Callers query this via
   * {@link didLastAcquireRebuildAttributes} immediately after acquire
   * and dispatch `invalidateRenderObjectFor(mesh)` when true so Three's
   * WebGPURenderer drops its stale `RenderObject.vertexBuffers` cache.
   *
   * Reset to `false` on every entry into an acquire method, so this
   * field reflects exclusively the *most recent* acquire result. The
   * read-then-act pattern (acquire → didLastAcquireRebuildAttributes →
   * invalidate) is synchronous in all production call paths.
   */
  /** @internal — written by adapters from acquire paths. */
  _lastAcquireRebuilt = false;

  /**
   * One-shot guard: have we already logged the >100MB pooled-buffer
   * warning? Re-checked per `evictUnused` so the noise stays bounded.
   */
  private largePoolWarningEmitted = false;

  // Per-type stats tracking. @internal — shared with adapters.
  typeStats = {
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
    this.points = new PointsBufferAdapter(this);
  }

  /**
   * Advance the frame counter. Call once per frame before any acquire calls.
   * This ensures eviction timing is based on rendered frames, not acquire calls.
   */
  beginFrame(): void {
    this.frameCount++;
  }

  /**
   * Whether the most recent `acquire*Geometry` call returned a
   * geometry whose underlying `InstancedInterleavedBuffer` references
   * differ from what was previously in use for that node — either
   * because the buffer pool grew the existing geometry's capacity, or
   * because the returned geometry is a fresh allocation / different
   * pool candidate.
   *
   * Callers that hold a `THREE.Mesh` pointing at the previously-active
   * geometry should call `invalidateRenderObjectFor(mesh)` (in
   * `data/scene-loader/invalidate-render-object.ts`) when this returns
   * `true`. That forces Three's WebGPURenderer to discard the cached
   * `RenderObject.vertexBuffers` set; without it, WebGPU binds the old
   * GPU buffer next draw and validation fails with "Instance range …
   * requires a larger buffer than the bound buffer size".
   *
   * The flag is overwritten on every acquire call, so consume it
   * immediately after `acquirePointsGeometry` / `acquireLinesGeometry`
   * / `acquireGSplatsGeometry` returns.
   */
  didLastAcquireRebuildAttributes(): boolean {
    return this._lastAcquireRebuilt;
  }

  // =========================================================================
  // Points Geometry Management
  // =========================================================================

  /**
   * Acquire Points geometry from pool (type-aware, capacity-aware).
   * See `PointsBufferAdapter.acquireGeometry` for implementation.
   */
  acquirePointsGeometry(
    nodeId: string,
    data: LoadedPointsData,
    pointCount: number
  ): THREE.BufferGeometry {
    return this.points.acquireGeometry(nodeId, data, pointCount);
  }

  /** Release Points geometry back to pool. */
  releasePointsGeometry(nodeId: string): void {
    this.points.releaseGeometry(nodeId);
  }

  /**
   * Update Points geometry attributes in-place (zero GPU allocations).
   */
  updatePointsGeometry(
    geometry: THREE.BufferGeometry,
    data: LoadedPointsData,
    count: number
  ): void {
    this.points.updateGeometry(geometry, data, count);
  }

  // =========================================================================
  // Lines Geometry Management
  // =========================================================================

  /**
   * Acquire geometry for Lines (instanced per-segment attributes).
   */
  acquireLinesGeometry(nodeId: string, segmentCount: number): THREE.InstancedBufferGeometry {
    this._lastAcquireRebuilt = false;
    const active = this.activeBuffers.get(nodeId);
    if (active && active.type === 'lines') {
      if (active.capacity >= segmentCount) {
        active.lastUsedFrame = this.frameCount;
        this.stats.reuses++;
        this.typeStats.lines.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      } else {
        // Grow rebuilds the InstancedInterleavedBuffer.
        this.growLinesGeometry(active.geometry as THREE.InstancedBufferGeometry, segmentCount);
        active.capacity = Math.ceil(segmentCount * 1.5);
        active.lastUsedFrame = this.frameCount;
        this.stats.capacityGrowths++;
        this._lastAcquireRebuilt = true;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
    }

    // Try to find in pool (search across all buckets)
    for (const pooled of this.lineBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= segmentCount) {
          // Different geometry than what was active for this node.
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          this.typeStats.lines.reuses++;
          this._lastAcquireRebuilt = true;
          return candidate.geometry as THREE.InstancedBufferGeometry;
        }
      }
    }

    // Allocate new — fresh attribute identities.
    this._lastAcquireRebuilt = true;
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
      // Attribute identities changed; downstream callers must
      // invalidate the renderer's cached RenderObject.
      this._lastAcquireRebuilt = true;
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
    this._lastAcquireRebuilt = false;
    const active = this.activeBuffers.get(nodeId);
    if (active && active.type === 'gsplats') {
      if (active.capacity >= splatCount) {
        active.lastUsedFrame = this.frameCount;
        this.stats.reuses++;
        this.typeStats.gsplats.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      } else {
        // Grow rebuilds the InstancedInterleavedBuffer.
        this.growGSplatsGeometry(active.geometry as THREE.InstancedBufferGeometry, splatCount);
        active.capacity = Math.ceil(splatCount * 1.5);
        active.lastUsedFrame = this.frameCount;
        this.stats.capacityGrowths++;
        this._lastAcquireRebuilt = true;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
    }

    // Try to find in pool (search across all buckets)
    for (const pooled of this.gsplatBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= splatCount) {
          // Different geometry than what was active for this node.
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = this.frameCount;
          this.activeBuffers.set(nodeId, candidate);
          this.stats.reuses++;
          this.typeStats.gsplats.reuses++;
          this._lastAcquireRebuilt = true;
          return candidate.geometry as THREE.InstancedBufferGeometry;
        }
      }
    }

    // Allocate new — fresh attribute identities.
    this._lastAcquireRebuilt = true;
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
   * Buckets: 1K, 5K, 10K, 50K, 100K, 500K, 1M.
   * @internal — called by adapters; public to satisfy PointsAdapterHost.
   */
  getBucket(count: number): number {
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
      Array.from(this.points.pointBuffers.values()).reduce((sum, arr) => sum + arr.length, 0) +
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

    const pointsEvicted = evictFromPool(this.points.pointBuffers, batchCap);
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
    collect(this.points.pointBuffers);
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
    const pointsPooled = Array.from(this.points.pointBuffers.values()).reduce(
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
    for (const arr of this.points.pointBuffers.values()) {
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

    disposePool(this.points.pointBuffers);
    disposePool(this.lineBuffers);
    disposePool(this.gsplatBuffers);

    log.info(Modules.GPU_BUFFER_POOL, 'All pooled geometries disposed');
  }
}
