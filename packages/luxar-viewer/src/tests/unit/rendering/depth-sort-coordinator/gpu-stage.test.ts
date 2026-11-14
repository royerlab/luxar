import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
  attachElementStorage,
  pumpSortedIndexOrderingApply,
} from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';
import { stageOrdering } from '../../../../rendering/depth-sort-coordinator/scheduler';
import type { SortResult } from '../../../../workers/sort-worker';

const result: SortResult = {
  generation: 1,
  ordering: new Uint32Array([2, 0, 3, 1]),
  kernelMs: 1,
  workerMs: 2,
};

describe('worker depth-order handoff', () => {
  it('stages the complete worker permutation when compute is unavailable', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;
    expect(stageOrdering({ geometry, result, callbacks: {} })).toBe(4);
    pumpSortedIndexOrderingApply(geometry);
    expect(Array.from(inactive.array).slice(0, 4)).toEqual([2, 0, 3, 1]);
    expect(inactive.version).toBeGreaterThan(version);
    geometry.dispose();
  });

  it('writes indexed mesh triangles from the worker permutation', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(6), 1));
    geometry.setDrawRange(0, 6);
    const meshResult = { ...result, ordering: new Uint32Array([1, 0]) };
    expect(
      stageOrdering({
        geometry,
        result: meshResult,
        callbacks: {},
        triangleSource: new Uint32Array([0, 1, 2, 3, 4, 5]),
      })
    ).toBe(6);
    expect(Array.from(geometry.index!.array)).toEqual([3, 4, 5, 0, 1, 2]);
    geometry.dispose();
  });
});
