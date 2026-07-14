/**
 * Direct tests for the geometry-agnostic helper in
 * `rendering/gpu-buffer-pool/attribute-codec`.
 *
 * The pool's grow/update behaviour is covered by gpu-buffer-pool.test.ts
 * and lines-scalar-end-to-end.test.ts (growth is release + reacquire —
 * there is no in-place rebuild helper anymore). These tests pin
 * writePooledAttribute in isolation: it must honour the strided offset
 * and the requested instance count.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { writePooledAttribute } from '../../../../rendering/gpu-buffer-pool/attribute-codec';
import { packInterleavedAttributes } from '../../../../rendering/interleaved-attributes';

const SPECS_BASIC = [
  { name: 'aCenter', itemSize: 3 as const },
  { name: 'aRadius', itemSize: 1 as const },
];

function makeEmptyInstancedGeometry(): THREE.InstancedBufferGeometry {
  return new THREE.InstancedBufferGeometry();
}

/**
 * Bind an interleaved buffer over `specs` at `capacity` — the same
 * creation-time packing the adapters use. (There is deliberately no
 * in-place rebuild anymore: growth and spec-set changes go through
 * release + reacquire; see attribute-codec.ts module doc.)
 */
function bindSpecs(
  geometry: THREE.InstancedBufferGeometry,
  capacity: number,
  specs: ReadonlyArray<{ name: string; itemSize: 1 | 2 | 3 | 4 }>
): void {
  const withData = specs.map((spec) => ({
    ...spec,
    data: new Float32Array(capacity * spec.itemSize),
  }));
  const { views } = packInterleavedAttributes(withData, capacity);
  for (const spec of withData) {
    geometry.setAttribute(spec.name, views[spec.name]);
  }
}

describe('writePooledAttribute', () => {
  it('writes a vec3 attribute at the correct strided offset', () => {
    const geometry = makeEmptyInstancedGeometry();
    bindSpecs(geometry, 2, SPECS_BASIC);

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
    bindSpecs(geometry, 2, SPECS_BASIC);
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
    bindSpecs(geometry, 4, SPECS_BASIC);

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
