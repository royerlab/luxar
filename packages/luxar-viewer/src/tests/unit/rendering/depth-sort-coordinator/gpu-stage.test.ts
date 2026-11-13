import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  activeSortedIndexSlot,
  attachElementStorage,
  configureSortedIndexGpuCompute,
  pumpSortedIndexOrderingApply,
} from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';
import {
  eligibleGpuBuilder,
  stageOrdering,
} from '../../../../rendering/depth-sort-coordinator/scheduler';
import type {
  CoordinatorState,
  NodeSortState,
} from '../../../../rendering/depth-sort-coordinator/state';
import type { WebGpuOrderBuilder } from '../../../../rendering/depth-sort-coordinator/webgpu-order';
import type { SortResult } from '../../../../workers/sort-worker';

const result: SortResult = {
  generation: 1,
  ordering: new Uint32Array([2, 0, 3, 1]),
  keys: new Uint16Array([100, 200, 0, 300]),
  kernelMs: 1,
  workerMs: 2,
};

function scene() {
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  const coordinator = { drawnStateChanged: false } as CoordinatorState;
  const apply = vi.fn();
  const disable = vi.fn();
  const builder = { canSort: () => true, apply, disable } as unknown as WebGpuOrderBuilder;
  const args = {
    geometry,
    mesh,
    coordinator,
    builder,
    compactKeys: true,
    count: 4,
    result,
    callbacks: {},
  };
  return { geometry, args, apply, disable };
}

describe('GPU depth-order handoff', () => {
  afterEach(() => configureSortedIndexGpuCompute(false));

  it('flips to compute output without raising the attribute version', () => {
    configureSortedIndexGpuCompute(true);
    const { geometry, args, apply, disable } = scene();
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;
    expect(stageOrdering(args)).toBe(4);
    expect(apply).toHaveBeenCalledWith(geometry, result.keys, 4);
    expect(disable).not.toHaveBeenCalled();
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(inactive.version).toBe(version);
    geometry.dispose();
  });

  it('disables a throwing builder and stages the complete worker permutation', () => {
    configureSortedIndexGpuCompute(true);
    const { geometry, args, apply, disable } = scene();
    apply.mockImplementation(() => {
      throw new Error('device lost');
    });
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;
    expect(stageOrdering(args)).toBe(4);
    expect(disable).toHaveBeenCalledOnce();
    pumpSortedIndexOrderingApply(geometry);
    expect(Array.from(inactive.array).slice(0, 4)).toEqual([2, 0, 3, 1]);
    expect(inactive.version).toBeGreaterThan(version);
    geometry.dispose();
  });

  it('falls back when compute output cannot be selected', () => {
    configureSortedIndexGpuCompute(true);
    const { geometry, args, disable } = scene();
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    expect(stageOrdering({ ...args, count: 5 })).toBe(4);
    expect(disable).not.toHaveBeenCalled();
    pumpSortedIndexOrderingApply(geometry);
    expect(Array.from(inactive.array).slice(0, 4)).toEqual([2, 0, 3, 1]);
    expect(inactive.version).toBeGreaterThan(0);
    geometry.dispose();
  });

  it('writes mesh triangle order without invoking the GPU builder', () => {
    const { geometry, args, apply } = scene();
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(6), 1));
    geometry.setDrawRange(0, 6);
    const meshResult = { ...result, ordering: new Uint32Array([1, 0]) };
    expect(
      stageOrdering({
        ...args,
        result: meshResult,
        count: 2,
        triangleSource: new Uint32Array([0, 1, 2, 3, 4, 5]),
      })
    ).toBe(6);
    expect(apply).not.toHaveBeenCalled();
    expect(Array.from(geometry.index!.array)).toEqual([3, 4, 5, 0, 1, 2]);
    geometry.dispose();
  });

  it('never requests compact keys for an indexed mesh', () => {
    const { args } = scene();
    const canSort = vi.fn(() => true);
    const builder = { canSort } as unknown as WebGpuOrderBuilder;
    const state = { triangleSource: new Uint32Array([0, 1, 2]), count: 1 } as NodeSortState;
    expect(eligibleGpuBuilder(args.mesh, state, builder)).toBeNull();
    expect(canSort).not.toHaveBeenCalled();
    expect(eligibleGpuBuilder(args.mesh, { ...state, triangleSource: undefined }, builder)).toBe(
      builder
    );
    expect(canSort).toHaveBeenCalledOnce();
    args.geometry.dispose();
  });
});
