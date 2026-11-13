import * as THREE from 'three';
import { afterEach, describe, expect, it } from 'vitest';
import {
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  attachElementStorage,
  configureSortedIndexGpuCompute,
  selectComputedSortedIndexOrdering,
  writeSortedIndexIdentity,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';

describe('compute-built ordering handoff', () => {
  afterEach(() => configureSortedIndexGpuCompute(false));

  it('selects the complete GPU order without asking three to upload its CPU mirror', () => {
    configureSortedIndexGpuCompute(true);
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    writeSortedIndexIdentity(geometry, 4);
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;
    let applied = 0;
    expect(
      selectComputedSortedIndexOrdering(geometry, 4, new Uint32Array([2, 0, 3, 1]), {
        onApplied: () => {
          applied++;
        },
      })
    ).toBe(true);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(Array.from(inactive.array)).toEqual([2, 0, 3, 1]);
    expect(inactive.version).toBe(version);
    expect(applied).toBe(0);
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(applied).toBe(1);

    writeSortedIndexIdentity(geometry, 4);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(Array.from(geometry.getAttribute('aSortedIndex').array)).toEqual([0, 1, 2, 3]);
    geometry.dispose();
  });

  it('refuses a truncated permutation before changing the selected slot', () => {
    configureSortedIndexGpuCompute(true);
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    expect(selectComputedSortedIndexOrdering(geometry, 4, new Uint32Array([3, 2]))).toBe(false);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    geometry.dispose();
  });
});
