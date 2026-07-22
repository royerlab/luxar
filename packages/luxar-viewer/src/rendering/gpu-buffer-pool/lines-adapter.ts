/**
 * Lines buffer-pool adapter.
 *
 * Owns the lines-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Line geometries. Shared state
 * (activeBuffers, stats, frame counter) is read from the GPUBufferPool
 * reference passed at construction.
 *
 * The top-level GPUBufferPool owns only shared coordination logic
 * (eviction, frame counter, dispose); this adapter owns Lines-specific
 * buffer layout and update behavior.
 */

import * as THREE from 'three';
import { packInterleavedAttributes, widenToFloat32 } from '../interleaved-attributes';
import type { ProcessedLinesData } from '../../types/lines';
import { writePooledAttribute } from './attribute-codec';
import type { PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';

/**
 * Canonical per-segment attribute layout for pooled line geometries.
 * The pool pre-allocates a single `InstancedInterleavedBuffer` over
 * these specs (Float32 throughout — Uint8 clipped flags get widened
 * at upload time). Optional scalar attributes (aStartScalar /
 * aEndScalar) are included at CREATION time when the acquire call
 * declares colormap data (hasScalars) — never via an in-place rebuild,
 * which would strand the old GPU buffer in the renderer caches.
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

function createLinesGeometry(
  segmentCapacity: number,
  hasScalars: boolean
): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  const quadPositions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadPositions, 2));
  geometry.setIndex([0, 1, 2, 2, 1, 3]);

  const specTemplates = hasScalars
    ? [...LINES_BASE_ATTRIBUTE_SPECS, ...LINES_SCALAR_ATTRIBUTE_SPECS]
    : LINES_BASE_ATTRIBUTE_SPECS;
  const specs = specTemplates.map((spec) => ({
    ...spec,
    data: new Float32Array(segmentCapacity * spec.itemSize),
  }));
  const { buffer, views } = packInterleavedAttributes(specs, segmentCapacity);
  buffer.setUsage(THREE.DynamicDrawUsage);
  for (const spec of specs) {
    geometry.setAttribute(spec.name, views[spec.name]);
  }
  // Draw nothing until the first successful write sets the real count
  // (mirrors the points/gsplats adapters — THREE's default is Infinity).
  geometry.instanceCount = 0;
  geometry.setDrawRange(0, 6);
  // Ownership marker — see the points adapter's twin comment.
  if (!geometry.userData) geometry.userData = {};
  geometry.userData.luxarPooled = true;
  return geometry;
}

/** Whether a pooled lines geometry carries the colormap scalar columns. */
function linesGeometryHasScalars(geometry: THREE.BufferGeometry): boolean {
  return geometry.getAttribute('aStartScalar') !== undefined;
}

export interface LinesAdapterHost {
  activeBuffers: Map<string, PooledBuffer>;
  readonly frameCount: number;
  stats: {
    allocations: number;
    reuses: number;
    evictions: number;
    capacityGrowths: number;
    deferredEvictions: number;
  };
  typeStats: {
    points: { allocations: number; reuses: number; evictions: number };
    lines: { allocations: number; reuses: number; evictions: number };
    gsplats: { allocations: number; reuses: number; evictions: number };
  };
  _lastAcquireRebuilt: boolean;
  getBucket(count: number): number;
  evictUnused(fromAcquire?: boolean): number;
}

export class LinesBufferAdapter {
  /** @internal — pool-bucketed reusable line geometries. */
  readonly lineBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: LinesAdapterHost) {}

  acquireGeometry(
    nodeId: string,
    segmentCount: number,
    hasScalars: boolean
  ): THREE.InstancedBufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    const active = host.activeBuffers.get(nodeId);
    if (active && active.type === 'lines') {
      const scalarsMatch = linesGeometryHasScalars(active.geometry) === hasScalars;
      if (scalarsMatch && active.capacity >= segmentCount) {
        active.lastUsedFrame = host.frameCount;
        host.stats.reuses++;
        host.typeStats.lines.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
      // Undersized OR spec-set change (colormap scalars appearing/
      // disappearing): RELEASE + REACQUIRE, never an in-place
      // interleaved-buffer rebuild — replacing a rendered geometry's
      // attributes strands the old GL/GPU buffer in the renderer
      // caches (hard leak under the WebGPU renderer via the strong
      // Info.memoryMap). See the points adapter for the full
      // rationale. The scalar spec set is decided HERE, at acquire
      // time, so updateGeometry never needs to rebuild.
      if (scalarsMatch) host.stats.capacityGrowths++;
      // OOM RE-CLAIM WINDOW — see the points adapter's twin comment.
      // The released buffer can never be picked by the best-fit scan
      // for this call: either its capacity < segmentCount (grow) or its
      // scalar spec set mismatches (the scan filters on both).
      const released = active;
      try {
        this.releaseGeometry(nodeId);
        return this.adoptOrAllocate(nodeId, segmentCount, hasScalars);
      } catch (error) {
        this.reclaimAfterFailedGrow(nodeId, released);
        throw error;
      }
    }

    return this.adoptOrAllocate(nodeId, segmentCount, hasScalars);
  }

  /**
   * Best-fit adoption from the free buckets, else a fresh allocation —
   * see the points adapter's twin comment.
   */
  private adoptOrAllocate(
    nodeId: string,
    segmentCount: number,
    hasScalars: boolean
  ): THREE.InstancedBufferGeometry {
    const host = this.host;

    // BEST-fit, not first-fit — see the gsplats adapter for rationale.
    // Candidates must carry the SAME scalar spec set (a base-only
    // buffer cannot serve colormap data, and a scalar buffer serving
    // base-only data would render stale scalar columns).
    let bestList: PooledBuffer[] | null = null;
    let bestIndex = -1;
    let bestCapacity = Infinity;
    for (const pooled of this.lineBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (
          candidate.capacity >= segmentCount &&
          candidate.capacity < bestCapacity &&
          linesGeometryHasScalars(candidate.geometry) === hasScalars
        ) {
          bestList = pooled;
          bestIndex = i;
          bestCapacity = candidate.capacity;
        }
      }
    }
    if (bestList) {
      const candidate = bestList[bestIndex];
      bestList.splice(bestIndex, 1);
      // See the points adapter's twin comment.
      (candidate.geometry as THREE.InstancedBufferGeometry).instanceCount = 0;
      candidate.inUse = true;
      candidate.lastUsedFrame = host.frameCount;
      host.activeBuffers.set(nodeId, candidate);
      host.stats.reuses++;
      host.typeStats.lines.reuses++;
      host._lastAcquireRebuilt = true;
      return candidate.geometry as THREE.InstancedBufferGeometry;
    }

    host._lastAcquireRebuilt = true;
    const capacity = chooseCapacity(segmentCount);
    const geometry = createLinesGeometry(capacity, hasScalars);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'lines',
      inUse: true,
      lastUsedFrame: host.frameCount,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    host.stats.allocations++;
    // Fresh allocations count against the byte budget too — sweep idle
    // pooled buffers (see growth-path note above).
    host.evictUnused(true);
    host.typeStats.lines.allocations++;
    return geometry;
  }

  /**
   * Undo a failed grow — see the points adapter's twin comment for the
   * full sub-case breakdown (replacement-entry disposal + free-bucket
   * re-claim; a buffer the release-time sweep disposed stays gone).
   */
  private reclaimAfterFailedGrow(nodeId: string, released: PooledBuffer): void {
    const host = this.host;

    const current = host.activeBuffers.get(nodeId);
    if (current && current !== released) {
      host.activeBuffers.delete(nodeId);
      current.geometry.dispose();
    }

    for (const pooled of this.lineBuffers.values()) {
      const index = pooled.indexOf(released);
      if (index !== -1) {
        pooled.splice(index, 1);
        released.inUse = true;
        released.lastUsedFrame = host.frameCount;
        host.activeBuffers.set(nodeId, released);
        return;
      }
    }
  }

  releaseGeometry(nodeId: string): void {
    const host = this.host;
    const buffer = host.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'lines') return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;
    // Stamp the release frame so acquire-triggered byte sweeps later in
    // this same frame grace the buffer (see EvictorCtx.graceFrame) — a
    // released buffer otherwise carries the frame of its last ACQUIRE
    // and the dataset-switch grace never matches. Also makes the
    // just-released buffer the freshest LRU reuse candidate.
    buffer.lastUsedFrame = host.frameCount;

    const bucket = host.getBucket(buffer.capacity);
    if (!this.lineBuffers.has(bucket)) {
      this.lineBuffers.set(bucket, []);
    }
    this.lineBuffers.get(bucket)!.push(buffer);

    host.evictUnused();
  }

  updateGeometry(
    geometry: THREE.InstancedBufferGeometry,
    data: ProcessedLinesData,
    count: number,
    options?: { fromInstance?: number }
  ): void {
    const hasScalarsInData = !!(data.startScalars && data.endScalars);
    if (hasScalarsInData && !linesGeometryHasScalars(geometry)) {
      // The scalar spec set is decided at acquire time (acquireGeometry's
      // hasScalars parameter) — an in-place rebuild here would strand the
      // old GPU buffer in the renderer caches. Reaching this means the
      // caller passed hasScalars=false and then supplied scalar data.
      throw new Error(
        'LinesBufferAdapter.updateGeometry: geometry has no scalar columns but data ' +
          'carries scalars — acquireLinesGeometry must be called with hasScalars=true.'
      );
    }

    // PRE-FLIGHT torn-write guard: unlike points/gsplats (a single fused
    // writer that guards before ANY store), lines perform 11–13
    // sequential per-attribute writes below, each with its own length
    // guard. A short array in the MIDDLE of that sequence would throw
    // after earlier attributes were already stored and range-registered,
    // leaving a torn mix (new positions + old colors) on the GPU.
    // Validate every array this update will write — the same lengths the
    // individual writePooledAttribute guards check (short throws there;
    // longer sources are trimmed by its subarray, hence `<`) — and throw
    // ONE aggregate error before the first store. The per-write guards
    // stay as belt and braces.
    const requiredLengths: Array<[field: string, actual: number, expected: number]> = [
      ['startPositions', data.startPositions.length, count * 3],
      ['endPositions', data.endPositions.length, count * 3],
      ['startColors', data.startColors.length, count * 3],
      ['endColors', data.endColors.length, count * 3],
      ['startWidths', data.startWidths.length, count],
      ['endWidths', data.endWidths.length, count],
      ['startSharpness', data.startSharpness.length, count],
      ['endSharpness', data.endSharpness.length, count],
      ['segmentLengths', data.segmentLengths.length, count],
      ['startClipped', data.startClipped.length, count],
      ['endClipped', data.endClipped.length, count],
    ];
    if (hasScalarsInData) {
      requiredLengths.push(['startScalars', data.startScalars!.length, count]);
      requiredLengths.push(['endScalars', data.endScalars!.length, count]);
    }
    const shortFields = requiredLengths.filter(([, actual, expected]) => actual < expected);
    if (shortFields.length > 0) {
      throw new Error(
        'LinesBufferAdapter.updateGeometry: refusing a torn multi-attribute write for ' +
          `count=${count} — short source arrays: ` +
          shortFields
            .map(([field, actual, expected]) => `${field} (length ${actual}, need ${expected})`)
            .join(', ')
      );
    }

    // Append fast path (depth-sorting Phase 4 Stage 2): the commit layer
    // proved the buffer's first `fromInstance` segments already hold this
    // data's prefix, so every attribute write below skips them — only the
    // `[fromInstance, count)` suffix is copied and dirtied for upload.
    const opts = { fromInstance: options?.fromInstance ?? 0 };

    const startClippedF32 = widenToFloat32(data.startClipped);
    const endClippedF32 = widenToFloat32(data.endClipped);

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
      writePooledAttribute(geometry, name, source, count, opts);
    }

    if (hasScalarsInData) {
      writePooledAttribute(
        geometry,
        'aStartScalar',
        data.startScalars as Float32Array,
        count,
        opts
      );
      writePooledAttribute(geometry, 'aEndScalar', data.endScalars as Float32Array, count, opts);
    }

    geometry.instanceCount = count;

    // CRITICAL: Force THREE.js to recalculate _maxInstanceCount.
    delete (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount;

    // CRITICAL: Recompute bounding box after position updates (mirrors
    // computeLineBounds in line-geometry.ts). Also expand by max width
    // so the rendered footprint is covered by frustum culling.
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

  dispose(): void {
    for (const buffers of this.lineBuffers.values()) {
      for (const buffer of buffers) {
        buffer.geometry.dispose();
      }
    }
    this.lineBuffers.clear();
  }
}
