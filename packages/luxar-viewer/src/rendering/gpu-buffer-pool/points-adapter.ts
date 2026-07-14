/**
 * Points buffer-pool adapter.
 *
 * Owns the points-specific pool state (per-capacity buckets) and
 * implements acquire/release/update for Points geometries. Shared
 * state (activeBuffers, stats, frame counter) is read from the
 * GPUBufferPool reference passed at construction.
 *
 * The top-level GPUBufferPool owns only shared coordination logic
 * (eviction, frame counter, dispose); this adapter owns Points-specific
 * buffer layout and update behavior.
 */

import * as THREE from 'three';
import {
  packInterleavedAttributes,
  widenToFloat32,
  writeInterleavedAttribute,
} from '../interleaved-attributes';
import { invalidateCachedByteSize } from './geometry-bytes';
import type { LoadedPointsData } from '../../data/data-loader-types';
import { rebuildInterleavedBuffer, writePooledAttribute } from './attribute-codec';
import type { PointsAttributeTypes, PooledBuffer } from './pool-stats';
import { chooseCapacity } from './capacity';

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
  const specs: Array<{ name: string; itemSize: 1 | 2 | 3 | 4 }> = [...POINTS_BASE_ATTRIBUTE_SPECS];
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

function attributeTypesMatch(a: PointsAttributeTypes, b: PointsAttributeTypes): boolean {
  return (
    a.position === b.position &&
    a.color === b.color &&
    a.radius === b.radius &&
    a.sharpness === b.sharpness &&
    a.scalar === b.scalar
  );
}

function detectAttributeTypes(data: LoadedPointsData): PointsAttributeTypes {
  const types: PointsAttributeTypes = {
    position: 'Float32Array',
    color:
      data.colors instanceof Uint8Array
        ? 'Uint8Array'
        : data.colors instanceof Uint16Array
          ? 'Uint16Array'
          : 'Float32Array',
    radius: data.radii instanceof Uint8Array ? 'Uint8Array' : 'Float32Array',
    sharpness: data.sharpness instanceof Uint8Array ? 'Uint8Array' : 'Float32Array',
  };
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
 * Points render as instanced unit quads, so the indexed draw range is
 * always the 2-triangle base quad (6 indices) while `instanceCount`
 * carries the number of point sprites.
 */
function preparePointsGeometryForDraw(
  geometry: THREE.BufferGeometry,
  pointCount: number
): THREE.InstancedBufferGeometry {
  const instanced = geometry as THREE.InstancedBufferGeometry;
  instanced.instanceCount = pointCount;
  instanced.setDrawRange(0, 6);
  return instanced;
}

function createPointsGeometry(
  capacity: number,
  types: PointsAttributeTypes
): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  const quadCorners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);
  geometry.setAttribute('aQuadCorner', new THREE.BufferAttribute(quadCorners, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.instanceCount = 0;
  geometry.setDrawRange(0, 6);

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

function growPointsGeometry(
  geometry: THREE.BufferGeometry,
  neededCount: number,
  types: PointsAttributeTypes
): void {
  const newCapacity = chooseCapacity(neededCount);
  invalidateCachedByteSize(geometry);
  rebuildInterleavedBuffer(
    geometry as THREE.InstancedBufferGeometry,
    newCapacity,
    pointAttributeSpecs(types)
  );
}

/**
 * Shared-state surface the points adapter reads/writes on the parent
 * GPUBufferPool. Kept narrow so the adapter can be unit-tested with a
 * minimal stub instead of a full pool instance.
 */
export interface PointsAdapterHost {
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

export class PointsBufferAdapter {
  /** @internal — pool-bucketed reusable point geometries. */
  readonly pointBuffers = new Map<number, PooledBuffer[]>();

  constructor(private readonly host: PointsAdapterHost) {}

  acquireGeometry(
    nodeId: string,
    data: LoadedPointsData,
    pointCount: number
  ): THREE.BufferGeometry {
    const host = this.host;
    host._lastAcquireRebuilt = false;

    const types = detectAttributeTypes(data);

    const active = host.activeBuffers.get(nodeId);
    if (active && active.type === 'points' && active.attributeTypes) {
      if (attributeTypesMatch(active.attributeTypes, types)) {
        if (active.capacity >= pointCount) {
          active.lastUsedFrame = host.frameCount;
          host.stats.reuses++;
          host.typeStats.points.reuses++;
          return preparePointsGeometryForDraw(active.geometry, pointCount);
        } else {
          growPointsGeometry(
            active.geometry as THREE.BufferGeometry,
            pointCount,
            active.attributeTypes
          );
          active.capacity = chooseCapacity(pointCount);
          active.lastUsedFrame = host.frameCount;
          host.stats.capacityGrowths++;
        // Growth can push total pool bytes past the budget without any
        // release happening (a streaming session that only grows).
        // Sweep idle pooled buffers now instead of waiting for the next
        // releaseGeometry (historically the ONLY byte-budget trigger).
        host.evictUnused();
          // growPointsGeometry reallocates the interleaved buffer (a real
          // GPU buffer creation), so bump the per-type allocation counter
          // to keep `typeStats.points.allocations` in sync with actual
          // GPU buffer churn. Without this, the data-loading-monitor
          // reuse-rate metric underreports grows as "free" reuses.
          host.typeStats.points.allocations++;
          host._lastAcquireRebuilt = true;
          return preparePointsGeometryForDraw(active.geometry, pointCount);
        }
      } else {
        this.releaseGeometry(nodeId);
      }
    }

    for (const pooled of this.pointBuffers.values()) {
      for (let i = pooled.length - 1; i >= 0; i--) {
        const candidate = pooled[i];
        if (
          candidate.attributeTypes &&
          candidate.capacity >= pointCount &&
          attributeTypesMatch(candidate.attributeTypes, types)
        ) {
          pooled.splice(i, 1);
          candidate.inUse = true;
          candidate.lastUsedFrame = host.frameCount;
          host.activeBuffers.set(nodeId, candidate);
          host.stats.reuses++;
          host.typeStats.points.reuses++;
          host._lastAcquireRebuilt = true;
          return preparePointsGeometryForDraw(candidate.geometry, pointCount);
        }
      }
    }

    host._lastAcquireRebuilt = true;
    const capacity = chooseCapacity(pointCount);
    const geometry = createPointsGeometry(capacity, types);

    const newBuffer: PooledBuffer = {
      geometry,
      capacity,
      type: 'points',
      inUse: true,
      lastUsedFrame: host.frameCount,
      attributeTypes: types,
    };

    host.activeBuffers.set(nodeId, newBuffer);
    host.stats.allocations++;
    // Fresh allocations count against the byte budget too — sweep idle
    // pooled buffers (see growth-path note above).
    host.evictUnused();
    host.typeStats.points.allocations++;

    return preparePointsGeometryForDraw(geometry, pointCount);
  }

  releaseGeometry(nodeId: string): void {
    const host = this.host;
    const buffer = host.activeBuffers.get(nodeId);
    if (!buffer || buffer.type !== 'points') return;

    host.activeBuffers.delete(nodeId);
    buffer.inUse = false;

    const bucket = host.getBucket(buffer.capacity);
    if (!this.pointBuffers.has(bucket)) {
      this.pointBuffers.set(bucket, []);
    }
    this.pointBuffers.get(bucket)!.push(buffer);

    host.evictUnused();
  }

  updateGeometry(geometry: THREE.BufferGeometry, data: LoadedPointsData, count: number): void {
    const instanced = geometry as THREE.InstancedBufferGeometry;

    const positionsF32 =
      data.positions instanceof Float32Array
        ? data.positions
        : widenToFloat32(data.positions as ArrayLike<number>);
    writePooledAttribute(instanced, 'aCenter', positionsF32, count);

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
      fill.fill(0.5); // default sharpness knob -> beta=2 (Gaussian)
      writeInterleavedAttribute(sharpBuffer, sharpView.offset, 1, fill, count);
    }

    const scalarView = instanced.getAttribute('aScalar') as
      | THREE.InterleavedBufferAttribute
      | undefined;
    if (scalarView) {
      const scalarBuffer = scalarView.data as THREE.InstancedInterleavedBuffer;
      if (data.scalars) {
        const widened = widenToFloat32(
          data.scalars.subarray(0, count) as ArrayLike<number>,
          pointsNormalizationDivisor(data.scalars, /*normalized=*/ true)
        );
        writeInterleavedAttribute(scalarBuffer, scalarView.offset, 1, widened, count);
      } else {
        const fill = new Float32Array(count);
        writeInterleavedAttribute(scalarBuffer, scalarView.offset, 1, fill, count);
      }
    }

    preparePointsGeometryForDraw(geometry, count);
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

  dispose(): void {
    for (const buffers of this.pointBuffers.values()) {
      for (const buffer of buffers) {
        buffer.geometry.dispose();
      }
    }
    this.pointBuffers.clear();
  }
}
