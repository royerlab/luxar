import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  acknowledgeSortedIndexOrderingDraw,
  attachElementStorage,
  activeSortedIndexSlot,
} from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';
import { configureRendererBackendPolicy } from '../../../../rendering/renderer-backend-policy';
import {
  canGpuSort,
  configureGpuOrderProducer,
} from '../../../../rendering/depth-sort-coordinator/gpu-order-producer';
import { scheduleSort } from '../../../../rendering/depth-sort-coordinator/scheduler';
import {
  workerHost,
  type CoordinatorState,
  type NodeSortState,
} from '../../../../rendering/depth-sort-coordinator/state';
import {
  BufferUsage,
  type GpuBuffer,
  type GpuDevice,
} from '../../../../rendering/gpu-depth-sort/gpu-types';
import { estimateGeometryBytes } from '../../../../rendering/gpu-buffer-pool/geometry-bytes';
import type { Renderer } from '../../../../rendering/renderer-capabilities';
import { UpdateProfiler } from '../../../../profiling/update-profiler';
import { setCommittedData } from '../../../../types/committed-data';
import { resortForCapture } from '../../../../rendering/depth-sort-coordinator/capture';

function fakeBackend(targetUsage = BufferUsage.VERTEX | BufferUsage.COPY_DST) {
  const copies: { target: GpuBuffer; bytes: number }[] = [];
  const allocations: { size: number; destroyed: boolean }[] = [];
  const buffers = new WeakMap<THREE.BufferAttribute, GpuBuffer>();
  const texture = { createView: () => ({}) };
  const device: GpuDevice = {
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
      minUniformBufferOffsetAlignment: 256,
    },
    queue: { writeBuffer: () => {}, submit: () => {} },
    createBuffer: ({ size, usage }) => {
      const allocation = { size, destroyed: false };
      allocations.push(allocation);
      return { size, usage, destroy: () => (allocation.destroyed = true) };
    },
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipelineAsync: async () => ({}),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => ({
        setPipeline: () => {},
        setBindGroup: () => {},
        dispatchWorkgroups: () => {},
        end: () => {},
      }),
      copyBufferToBuffer: (_source, _sourceOffset, target, _targetOffset, bytes) => {
        copies.push({ target, bytes });
      },
      finish: () => ({}),
    }),
  };
  const backend = {
    device,
    get: (resource: THREE.Texture | THREE.BufferAttribute) =>
      resource instanceof THREE.Texture
        ? { texture }
        : { buffer: buffers.get(resource as THREE.BufferAttribute) },
    createStorageAttribute: (attribute: THREE.BufferAttribute) => {
      buffers.set(attribute, {
        size: attribute.array.byteLength,
        usage: targetUsage,
        destroy: () => {},
      });
    },
  };
  return { backend, buffers, copies, allocations };
}

function scene() {
  const geometry = new THREE.InstancedBufferGeometry();
  const texture = attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
  const material = new THREE.ShaderMaterial({ uniforms: { uSortedIndexSlot: { value: 0 } } });
  material.userData.blendingMode = 'normal';
  const mesh = new THREE.Mesh(geometry, material);
  setCommittedData(mesh, {});
  const camera = new THREE.PerspectiveCamera();
  camera.updateMatrixWorld();
  const state = {
    mesh,
    generation: 1,
    count: 4,
    inFlight: false,
    resortQueued: false,
    lastSortAxis: null,
    lastSortOffset: 0,
    registered: true,
  } as NodeSortState;
  const requestRender = vi.fn();
  const profiler = new UpdateProfiler();
  const coordinator = {
    nodeStates: new Map([[mesh.uuid, state]]),
    getCamera: () => camera,
    requestRender,
    getProfiler: () => profiler,
    captureSuppressDepth: 0,
    depthSortEnabled: true,
    drawnStateChanged: false,
  } as unknown as CoordinatorState;
  return { geometry, texture, mesh, material, state, coordinator, requestRender, profiler };
}

describe('native WebGPU ordering producer', () => {
  const priorApi = workerHost.api;
  const priorInitState = workerHost.workerInitState;
  afterEach(() => {
    workerHost.api = priorApi;
    workerHost.workerInitState = priorInitState;
    configureRendererBackendPolicy('webgl');
    configureGpuOrderProducer({} as Renderer);
  });

  it('sorts from the uploaded element texture into the inactive draw buffer without the worker', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, material, state, coordinator, requestRender, profiler } =
      scene();
    workerHost.api = null;
    expect(canGpuSort(geometry, 4)).toBe(false);
    texture.onUpdate?.(texture);
    expect(canGpuSort(geometry, 4)).toBe(true);
    const baseBytes = estimateGeometryBytes(geometry);
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;
    scheduleSort(coordinator, mesh, mesh.uuid, true);
    expect(fake.copies).toEqual([{ target: fake.buffers.get(inactive), bytes: 16 }]);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(inactive.version).toBe(version);
    expect(material.uniforms.uSortedIndexSlot.value).toBe(1);
    expect(state.inFlight).toBe(false);
    expect(state.lastSortAxis).not.toBeNull();
    expect(requestRender).toHaveBeenCalledOnce();
    expect(profiler.getDepthSortCompletions().total).toBe(0);
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(profiler.getDepthSortCompletions().events).toEqual([
      {
        seq: 1,
        lastMs: expect.any(Number),
        kernelMs: null,
        boundaryMs: null,
        queueMs: null,
        elements: 4,
      },
    ]);
    expect(estimateGeometryBytes(geometry)).toBe(
      baseBytes + fake.allocations.reduce((sum, allocation) => sum + allocation.size, 0)
    );
    geometry.dispose();
    expect(estimateGeometryBytes(geometry)).toBe(baseBytes);
    expect(fake.allocations.every((allocation) => allocation.destroyed)).toBe(true);
  });

  it('declines a texture version that has not reached the GPU', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture } = scene();
    texture.onUpdate?.(texture);
    expect(canGpuSort(geometry, 4)).toBe(true);
    texture.needsUpdate = true;
    expect(canGpuSort(geometry, 4)).toBe(false);
    geometry.dispose();
  });

  it('falls back to the worker when the target lacks COPY_DST', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend(BufferUsage.VERTEX);
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, coordinator } = scene();
    texture.onUpdate?.(texture);
    const sort = vi.fn().mockResolvedValue(null);
    workerHost.api = { sort } as unknown as NonNullable<typeof workerHost.api>;
    workerHost.workerInitState = 'ready';

    scheduleSort(coordinator, mesh, mesh.uuid, true);

    expect(sort).toHaveBeenCalledOnce();
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(canGpuSort(geometry, 4)).toBe(false);
    geometry.dispose();
  });

  it('refreshes a capture on the GPU even when no worker is available', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, coordinator } = scene();
    texture.onUpdate?.(texture);
    workerHost.api = null;

    await resortForCapture(coordinator);

    expect(fake.copies).toHaveLength(1);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    geometry.dispose();
  });
});
