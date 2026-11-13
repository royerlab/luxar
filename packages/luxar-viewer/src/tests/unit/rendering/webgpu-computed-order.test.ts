import * as THREE from 'three';
import { afterEach, describe, expect, it } from 'vitest';
import {
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  attachElementStorage,
  repairSortedIndexForCount,
  selectComputedSortedIndexOrdering,
  writeSortedIndexIdentity,
  writeSortedIndexOrderingLive,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';
import { configureRendererBackendPolicy } from '../../../rendering/renderer-backend-policy';

describe('compute-built ordering handoff', () => {
  afterEach(() => configureRendererBackendPolicy('webgl'));

  it('selects the GPU buffer without uploading and keeps a valid CPU repair seed', () => {
    configureRendererBackendPolicy('webgpu-native');
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    expect(writeSortedIndexOrderingLive(geometry, new Uint32Array([2, 0, 3, 1]), 4)).toBe(4);
    const active = geometry.getAttribute('aSortedIndex');
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    expect(inactive).toBeInstanceOf(THREE.InstancedBufferAttribute);
    expect('isStorageInstancedBufferAttribute' in inactive).toBe(true);
    const version = inactive.version;
    let applied = 0;
    expect(
      selectComputedSortedIndexOrdering(geometry, 4, {
        onApplied: () => applied++,
      })
    ).toBe(true);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(inactive.version).toBe(version);
    expect(inactive.array).toBe(active.array);
    expect(applied).toBe(0);
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(applied).toBe(1);

    repairSortedIndexForCount(geometry, 4, 3);
    expect(Array.from(inactive.array).slice(0, 3)).toEqual([2, 0, 1]);
    geometry.dispose();
  });

  it('refuses a count beyond the ordering buffer before changing the selected slot', () => {
    configureRendererBackendPolicy('webgpu-native');
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    expect(selectComputedSortedIndexOrdering(geometry, 5)).toBe(false);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    geometry.dispose();
  });

  it('invalidates identity knowledge when the GPU mirror aliases the prior slot', () => {
    configureRendererBackendPolicy('webgpu-native');
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    writeSortedIndexIdentity(geometry, 4);
    const original = geometry.getAttribute('aSortedIndex') as THREE.BufferAttribute;
    expect(selectComputedSortedIndexOrdering(geometry, 4)).toBe(true);
    expect(writeSortedIndexOrderingLive(geometry, new Uint32Array([2, 0, 3, 1]), 4)).toBe(4);
    expect(Array.from(original.array)).toEqual([2, 0, 3, 1]);

    writeSortedIndexIdentity(geometry, 4);

    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(Array.from(original.array)).toEqual([0, 1, 2, 3]);
    geometry.dispose();
  });
});
