import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  acknowledgeSortedIndexOrderingDraw,
  activeSortedIndexSlot,
  attachElementStorage,
  configureSortedIndexGpuCompute,
  selectComputedSortedIndexOrdering,
  writeSortedIndexIdentity,
} from '../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../rendering/element-texture-layout';
import {
  WebGpuOrderBuilder,
  type GpuBackend,
} from '../../../rendering/depth-sort-coordinator/webgpu-order';
import { estimateGeometryBytes } from '../../../rendering/gpu-buffer-pool/geometry-bytes';
import { log } from '../../../utils/log';

describe('compute-built ordering handoff', () => {
  afterEach(() => configureSortedIndexGpuCompute(false));

  it('selects the complete GPU order without asking three to upload its CPU mirror', () => {
    configureSortedIndexGpuCompute(true);
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    writeSortedIndexIdentity(geometry, 4);
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    expect(inactive).toBeInstanceOf(THREE.InstancedBufferAttribute);
    expect(
      (inactive as THREE.InstancedBufferAttribute & { isStorageInstancedBufferAttribute: boolean })
        .isStorageInstancedBufferAttribute
    ).toBe(true);
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

  it('charges every compute scratch buffer and removes the charge on dispose', () => {
    configureSortedIndexGpuCompute(true);
    vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 });
    const sizes: number[] = [];
    const destroyed = vi.fn();
    const backend = {
      device: {
        createShaderModule: () => ({}),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
        createBuffer: ({ size }: { size: number }) => {
          sizes.push(size);
          return { destroy: destroyed };
        },
      },
    } as unknown as GpuBackend;
    const builder = new WebGpuOrderBuilder(backend);
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    const base = estimateGeometryBytes(geometry);
    (
      builder as unknown as { resourcesFor: (g: THREE.InstancedBufferGeometry) => unknown }
    ).resourcesFor(geometry);
    expect(estimateGeometryBytes(geometry)).toBe(base + sizes.reduce((a, b) => a + b, 0));
    geometry.dispose();
    expect(destroyed).toHaveBeenCalledTimes(sizes.length);
    expect(estimateGeometryBytes(geometry)).toBe(base);
    builder.dispose();
    vi.unstubAllGlobals();
  });

  it('destroys partial scratch allocations when the device rejects a buffer', () => {
    configureSortedIndexGpuCompute(true);
    vi.stubGlobal('GPUBufferUsage', { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 });
    const destroyed = vi.fn();
    let created = 0;
    const backend = {
      device: {
        createShaderModule: () => ({}),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
        createBuffer: () => {
          if (++created === 3) throw new Error('out of memory');
          return { destroy: destroyed };
        },
      },
    } as unknown as GpuBackend;
    const builder = new WebGpuOrderBuilder(backend);
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    const base = estimateGeometryBytes(geometry);
    expect(() =>
      (
        builder as unknown as { resourcesFor: (g: THREE.InstancedBufferGeometry) => unknown }
      ).resourcesFor(geometry)
    ).toThrow('out of memory');
    expect(destroyed).toHaveBeenCalledTimes(2);
    expect(estimateGeometryBytes(geometry)).toBe(base);
    geometry.dispose();
    builder.dispose();
    vi.unstubAllGlobals();
  });

  it('warns once when the device dispatch limit forces the worker path', async () => {
    configureSortedIndexGpuCompute(true);
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => undefined);
    const backend = {
      device: {
        limits: { maxStorageBufferBindingSize: 1_000_000, maxComputeWorkgroupsPerDimension: 1 },
        createShaderModule: () => ({}),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
      },
    } as unknown as GpuBackend;
    const builder = new WebGpuOrderBuilder(backend);
    await Promise.resolve();
    await Promise.resolve();
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 300, POINT_TEXTURE_LAYOUT);
    expect(builder.canSort(geometry, NaN)).toBe(false);
    expect(builder.canSort(geometry, 257, 'node-a')).toBe(false);
    expect(builder.canSort(geometry, 257, 'node-a')).toBe(false);
    expect(warning).toHaveBeenCalledTimes(1);
    expect(builder.canSort(geometry, 257, 'node-b')).toBe(false);
    expect(warning).toHaveBeenCalledTimes(2);
    geometry.dispose();
    builder.dispose();
    warning.mockRestore();
  });
});
