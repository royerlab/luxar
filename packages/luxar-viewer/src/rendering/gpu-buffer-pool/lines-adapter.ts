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
import { invalidateCachedByteSize } from './geometry-bytes';
import type { ProcessedLinesData } from '../../types/lines';
import { rebuildInterleavedBuffer, writePooledAttribute } from './attribute-codec';
import type { PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';

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

function createLinesGeometry(segmentCapacity: number): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();

  const quadPositions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadPositions, 2));
  geometry.setIndex([0, 1, 2, 2, 1, 3]);

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

function growLinesGeometry(geometry: THREE.InstancedBufferGeometry, neededCount: number): void {
  const newCapacity = chooseCapacity(neededCount);
  invalidateCachedByteSize(geometry);

  const hasScalars = geometry.getAttribute('aStartScalar') !== undefined;
  const specs = hasScalars
    ? [...LINES_BASE_ATTRIBUTE_SPECS, ...LINES_SCALAR_ATTRIBUTE_SPECS]
    : LINES_BASE_ATTRIBUTE_SPECS;
  rebuildInterleavedBuffer(geometry, newCapacity, specs);
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
  evictUnused(): number;
}

export class LinesBufferAdapter {
  /** @internal — pool-bucketed reusable line geometries. */
  readonly lineBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: LinesAdapterHost) {}

  acquireGeometry(nodeId: string, segmentCount: number): THREE.InstancedBufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    const active = host.activeBuffers.get(nodeId);
    if (active && active.type === 'lines') {
      if (active.capacity >= segmentCount) {
        active.lastUsedFrame = host.frameCount;
        host.stats.reuses++;
        host.typeStats.lines.reuses++;
        return active.geometry as THREE.InstancedBufferGeometry;
      } else {
        growLinesGeometry(active.geometry as THREE.InstancedBufferGeometry, segmentCount);
        active.capacity = chooseCapacity(segmentCount);
        active.lastUsedFrame = host.frameCount;
        host.stats.capacityGrowths++;
        // Growth can push total pool bytes past the budget without any
        // release happening (a streaming session that only grows).
        // Sweep idle pooled buffers now instead of waiting for the next
        // releaseGeometry (historically the ONLY byte-budget trigger).
        host.evictUnused();
        host._lastAcquireRebuilt = true;
        return active.geometry as THREE.InstancedBufferGeometry;
      }
    }

    for (const pooled of this.lineBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (candidate.capacity >= segmentCount) {
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = host.frameCount;
          host.activeBuffers.set(nodeId, candidate);
          host.stats.reuses++;
          host.typeStats.lines.reuses++;
          host._lastAcquireRebuilt = true;
          return candidate.geometry as THREE.InstancedBufferGeometry;
        }
      }
    }

    host._lastAcquireRebuilt = true;
    const capacity = chooseCapacity(segmentCount);
    const geometry = createLinesGeometry(capacity);

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
    host.evictUnused();
    host.typeStats.lines.allocations++;
    return geometry;
  }

  releaseGeometry(nodeId: string): void {
    const host = this.host;
    const buffer = host.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'lines') return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;

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
    count: number
  ): void {
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
      this.host._lastAcquireRebuilt = true;
    }

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
      writePooledAttribute(geometry, name, source, count);
    }

    if (hasScalarsInData) {
      writePooledAttribute(geometry, 'aStartScalar', data.startScalars as Float32Array, count);
      writePooledAttribute(geometry, 'aEndScalar', data.endScalars as Float32Array, count);
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
