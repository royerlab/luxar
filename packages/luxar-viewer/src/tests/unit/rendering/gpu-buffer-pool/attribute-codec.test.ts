/**
 * Direct tests for the geometry-agnostic helpers in
 * `rendering/gpu-buffer-pool/attribute-codec`.
 *
 * The end-to-end behaviour of the pool's grow/update paths is already
 * covered by gpu-buffer-pool.test.ts. These tests pin the helpers in
 * isolation so load-bearing invariants stay intact:
 *   - carry-forward of old attribute data when capacity grows
 *   - aQuadCorner is preserved across rebuilds (it's the indexed quad
 *     attribute used by every instanced renderer)
 *   - _maxInstanceCount is invalidated after rebuild (r184 caches it
 *     on the geometry and won't refresh when the buffer is replaced)
 *   - the new buffer is marked DynamicDrawUsage
 *   - writePooledAttribute honours the strided offset
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  rebuildInterleavedBuffer,
  writePooledAttribute,
} from '../../../../rendering/gpu-buffer-pool/attribute-codec';

const SPECS_BASIC = [
  { name: 'aCenter', itemSize: 3 as const },
  { name: 'aRadius', itemSize: 1 as const },
];

const SPECS_WITH_COLOR = [
  { name: 'aCenter', itemSize: 3 as const },
  { name: 'aRadius', itemSize: 1 as const },
  { name: 'aColor', itemSize: 3 as const },
];

function makeEmptyInstancedGeometry(): THREE.InstancedBufferGeometry {
  return new THREE.InstancedBufferGeometry();
}

describe('rebuildInterleavedBuffer', () => {
  it('allocates a buffer sized for capacity * stride', () => {
    const geometry = makeEmptyInstancedGeometry();
    const buffer = rebuildInterleavedBuffer(geometry, 4, SPECS_BASIC);
    // stride = 3 + 1 = 4 floats per instance, capacity 4 → 16 floats.
    expect(buffer.stride).toBe(4);
    expect(buffer.array.length).toBe(16);
  });

  it('marks the new buffer DynamicDrawUsage', () => {
    const geometry = makeEmptyInstancedGeometry();
    const buffer = rebuildInterleavedBuffer(geometry, 4, SPECS_BASIC);
    expect(buffer.usage).toBe(THREE.DynamicDrawUsage);
  });

  it('binds every spec attribute on the geometry', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 4, SPECS_WITH_COLOR);
    expect(geometry.getAttribute('aCenter')).toBeDefined();
    expect(geometry.getAttribute('aRadius')).toBeDefined();
    expect(geometry.getAttribute('aColor')).toBeDefined();
  });

  it('carries old per-instance data forward when capacity grows', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);
    writePooledAttribute(geometry, 'aCenter', new Float32Array([1, 2, 3, 4, 5, 6]), 2);
    writePooledAttribute(geometry, 'aRadius', new Float32Array([7, 8]), 2);

    rebuildInterleavedBuffer(geometry, 4, SPECS_BASIC);

    const centerView = geometry.getAttribute('aCenter') as THREE.InterleavedBufferAttribute;
    expect(centerView.getX(0)).toBe(1);
    expect(centerView.getY(0)).toBe(2);
    expect(centerView.getZ(0)).toBe(3);
    expect(centerView.getX(1)).toBe(4);
    expect(centerView.getZ(1)).toBe(6);
    const radiusView = geometry.getAttribute('aRadius') as THREE.InterleavedBufferAttribute;
    expect(radiusView.getX(0)).toBe(7);
    expect(radiusView.getX(1)).toBe(8);
  });

  it('binds an added attribute on spec-set growth without zeroing existing data', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);
    writePooledAttribute(geometry, 'aCenter', new Float32Array([1, 2, 3, 4, 5, 6]), 2);

    rebuildInterleavedBuffer(geometry, 2, SPECS_WITH_COLOR);

    const centerView = geometry.getAttribute('aCenter') as THREE.InterleavedBufferAttribute;
    expect(centerView.getX(0)).toBe(1);
    expect(centerView.getZ(1)).toBe(6);
    const colorView = geometry.getAttribute('aColor') as THREE.InterleavedBufferAttribute;
    // Newly added attribute: zero-initialised.
    expect(colorView.getX(0)).toBe(0);
    expect(colorView.getY(0)).toBe(0);
  });

  it('drops attributes no longer in the spec set', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 2, SPECS_WITH_COLOR);
    expect(geometry.getAttribute('aColor')).toBeDefined();

    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);
    expect(geometry.getAttribute('aColor')).toBeUndefined();
  });

  it('preserves aQuadCorner across rebuild even though it is not in any spec', () => {
    const geometry = makeEmptyInstancedGeometry();
    const quadCorners = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute(quadCorners, 2));

    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);
    rebuildInterleavedBuffer(geometry, 4, SPECS_BASIC);

    const quad = geometry.getAttribute('aQuadCorner');
    expect(quad).toBeDefined();
    expect(quad.itemSize).toBe(2);
  });

  it('invalidates the r184 _maxInstanceCount cache after rebuild', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);
    // Simulate r184 caching _maxInstanceCount on the geometry.
    (geometry as unknown as { _maxInstanceCount: number })._maxInstanceCount = 2;

    rebuildInterleavedBuffer(geometry, 4, SPECS_BASIC);

    expect(
      (geometry as unknown as { _maxInstanceCount?: number })._maxInstanceCount
    ).toBeUndefined();
  });
});

describe('writePooledAttribute', () => {
  it('writes a vec3 attribute at the correct strided offset', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);

    writePooledAttribute(geometry, 'aCenter', new Float32Array([10, 20, 30, 40, 50, 60]), 2);

    const view = geometry.getAttribute('aCenter') as THREE.InterleavedBufferAttribute;
    expect(view.getX(0)).toBe(10);
    expect(view.getY(0)).toBe(20);
    expect(view.getZ(0)).toBe(30);
    expect(view.getX(1)).toBe(40);
    expect(view.getZ(1)).toBe(60);
  });

  it('writes a scalar attribute without touching neighbours in the interleaved buffer', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 2, SPECS_BASIC);
    writePooledAttribute(geometry, 'aCenter', new Float32Array([1, 2, 3, 4, 5, 6]), 2);

    writePooledAttribute(geometry, 'aRadius', new Float32Array([99, 88]), 2);

    const centerView = geometry.getAttribute('aCenter') as THREE.InterleavedBufferAttribute;
    expect(centerView.getX(0)).toBe(1);
    expect(centerView.getZ(1)).toBe(6);
    const radiusView = geometry.getAttribute('aRadius') as THREE.InterleavedBufferAttribute;
    expect(radiusView.getX(0)).toBe(99);
    expect(radiusView.getX(1)).toBe(88);
  });

  it('only writes the requested instance count, leaving remainder untouched', () => {
    const geometry = makeEmptyInstancedGeometry();
    rebuildInterleavedBuffer(geometry, 4, SPECS_BASIC);

    // Pre-fill so we can detect leftover writes.
    writePooledAttribute(geometry, 'aRadius', new Float32Array([7, 7, 7, 7]), 4);

    // Source has 4 values but we only ask for 2 instances.
    writePooledAttribute(geometry, 'aRadius', new Float32Array([1, 2, 3, 4]), 2);

    const radiusView = geometry.getAttribute('aRadius') as THREE.InterleavedBufferAttribute;
    expect(radiusView.getX(0)).toBe(1);
    expect(radiusView.getX(1)).toBe(2);
    expect(radiusView.getX(2)).toBe(7);
    expect(radiusView.getX(3)).toBe(7);
  });
});
