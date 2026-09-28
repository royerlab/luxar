/**
 * Order-independent (commutative) nodes write the identity ordering on every
 * commit. When the active buffer already holds identity over the drawn prefix,
 * that write must not bump the attribute version (which would re-upload the
 * whole prefix), and a grown population must upload only its new suffix.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import {
  attachElementStorage,
  getActiveSortedIndexAttribute,
  repairSortedIndexForCount,
  writeSortedIndexIdentity,
  writeSortedIndexIdentityRange,
  writeSortedIndexOrderingLive,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';

function makeGeometry(capacity = 64): THREE.InstancedBufferGeometry {
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, capacity, POINT_TEXTURE_LAYOUT);
  return geometry;
}

function active(geometry: THREE.InstancedBufferGeometry): THREE.InstancedBufferAttribute {
  const attr = getActiveSortedIndexAttribute(geometry);
  if (!attr) throw new Error('no ordering attribute');
  return attr;
}

function isIdentity(attr: THREE.InstancedBufferAttribute, n: number): boolean {
  const arr = attr.array as Uint32Array;
  for (let i = 0; i < n; i++) if (arr[i] !== i) return false;
  return true;
}

describe('identity ordering writes skip what the buffer already holds', () => {
  it('a repeated identity write of the same count does not re-upload', () => {
    const geometry = makeGeometry();
    writeSortedIndexIdentity(geometry, 40);
    const attr = active(geometry);
    const version = attr.version;
    writeSortedIndexIdentity(geometry, 40);
    writeSortedIndexIdentity(geometry, 25);
    expect(attr.version).toBe(version);
    expect(isIdentity(attr, 40)).toBe(true);
  });

  it('a grown identity write uploads only the new suffix', () => {
    const geometry = makeGeometry();
    writeSortedIndexIdentity(geometry, 20);
    const attr = active(geometry);
    attr.clearUpdateRanges(); // as the backend does after uploading
    writeSortedIndexIdentity(geometry, 50);
    expect(isIdentity(attr, 50)).toBe(true);
    expect(attr.updateRanges).toEqual([{ start: 20, count: 30 }]);
  });

  it('a live permutation invalidates the identity knowledge', () => {
    const geometry = makeGeometry();
    writeSortedIndexIdentity(geometry, 4);
    writeSortedIndexOrderingLive(geometry, new Uint32Array([3, 2, 1, 0]), 4);
    const attr = active(geometry);
    const version = attr.version;
    writeSortedIndexIdentity(geometry, 4);
    expect(attr.version).toBeGreaterThan(version);
    expect(isIdentity(attr, 4)).toBe(true);
  });

  it('an append after a permutation keeps the permutation and stays unknown', () => {
    const geometry = makeGeometry();
    writeSortedIndexOrderingLive(geometry, new Uint32Array([1, 0]), 2);
    writeSortedIndexIdentityRange(geometry, 2, 4);
    const attr = active(geometry);
    expect(Array.from((attr.array as Uint32Array).subarray(0, 4))).toEqual([1, 0, 2, 3]);
    const version = attr.version;
    writeSortedIndexIdentity(geometry, 4);
    expect(attr.version).toBeGreaterThan(version);
    expect(isIdentity(attr, 4)).toBe(true);
  });

  it('repairing an identity prefix stays identity and needs no rewrite', () => {
    const geometry = makeGeometry();
    writeSortedIndexIdentity(geometry, 10);
    repairSortedIndexForCount(geometry, 10, 14);
    const attr = active(geometry);
    expect(isIdentity(attr, 14)).toBe(true);
    const version = attr.version;
    writeSortedIndexIdentity(geometry, 14);
    expect(attr.version).toBe(version);
  });
});
