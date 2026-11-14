import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  acknowledgeSortedIndexOrderingDraw,
  attachElementStorage,
  activeSortedIndexSlot,
  getActiveSortedIndexAttribute,
  isSortedIndexHostArrayCurrent,
  pumpSortedIndexOrderingApply,
  writeSortedIndexIdentity,
} from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';
import { configureRendererBackendPolicy } from '../../../../rendering/renderer-backend-policy';
import {
  canGpuSort,
  configureGpuOrderProducer,
  gpuSortProducer,
} from '../../../../rendering/depth-sort-coordinator/gpu-order-producer';
import { workerSortProducer } from '../../../../rendering/depth-sort-coordinator/sort-producers';
import {
  evaluateDepthSortPerFrame,
  scheduleSort,
} from '../../../../rendering/depth-sort-coordinator/scheduler';
import {
  workerHost,
  createCoordinatorState,
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
  const contents = new WeakMap<GpuBuffer, Uint32Array>();
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
        const order = contents.get(target)!;
        for (let i = 0; i < bytes / 4; i++) order[i] = bytes / 4 - 1 - i;
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
      const buffer = {
        size: attribute.array.byteLength,
        usage: targetUsage,
        destroy: () => {},
      };
      buffers.set(attribute, buffer);
      contents.set(buffer, new Uint32Array(attribute.array as Uint32Array));
    },
  };
  return { backend, buffers, contents, copies, allocations };
}

function scene(capacity = 4) {
  const geometry = new THREE.InstancedBufferGeometry();
  const texture = attachElementStorage(geometry, capacity, POINT_TEXTURE_LAYOUT);
  const material = new THREE.ShaderMaterial({ uniforms: { uSortedIndexSlot: { value: 0 } } });
  material.userData.blendingMode = 'normal';
  const mesh = new THREE.Mesh(geometry, material);
  setCommittedData(mesh, {});
  const camera = new THREE.PerspectiveCamera();
  camera.updateMatrixWorld();
  const state = {
    mesh,
    generation: 1,
    count: capacity,
    inFlight: false,
    resortQueued: false,
    lastSortAxis: null,
    lastSortOffset: 0,
    registered: true,
  } as NodeSortState;
  const requestRender = vi.fn();
  const profiler = new UpdateProfiler();
  const coordinator: CoordinatorState = createCoordinatorState([
    gpuSortProducer,
    workerSortProducer,
  ]);
  coordinator.nodeStates.set(mesh.uuid, state);
  coordinator.getCamera = () => camera;
  coordinator.requestRender = requestRender;
  coordinator.getProfiler = () => profiler;
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
    expect(fake.copies).toHaveLength(0);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    evaluateDepthSortPerFrame(coordinator);
    expect(fake.copies).toEqual([{ target: fake.buffers.get(inactive), bytes: 16 }]);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(inactive.version).toBe(version);
    expect(inactive.array).not.toBe(geometry.getAttribute('aSortedIndex').array);
    expect(isSortedIndexHostArrayCurrent(geometry)).toBe(false);
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
    const original = geometry.getAttribute('aSortedIndex') as THREE.BufferAttribute;
    const originalVersion = original.version;
    scheduleSort(coordinator, mesh, mesh.uuid, true);
    evaluateDepthSortPerFrame(coordinator);
    expect(fake.copies.at(-1)).toEqual({ target: fake.buffers.get(original), bytes: 16 });
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(original.version).toBe(originalVersion);
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(profiler.getDepthSortCompletions().total).toBe(2);
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

  it('keeps an undrawn identity tail from overwriting a later GPU ordering', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, state, coordinator } = scene(64);
    workerHost.api = null;
    texture.onUpdate?.(texture);
    writeSortedIndexIdentity(geometry, 32);
    geometry.instanceCount = 32;
    const slotA = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
    fake.backend.createStorageAttribute(slotA);
    slotA.clearUpdateRanges(); // The initial 32-element identity was drawn.
    writeSortedIndexIdentity(geometry, 48);
    state.count = 48;
    geometry.instanceCount = 48;
    expect(slotA.updateRanges).toEqual([{ start: 32, count: 16 }]);

    scheduleSort(coordinator, mesh, mesh.uuid, true);
    pumpSortedIndexOrderingApply(geometry);
    scheduleSort(coordinator, mesh, mesh.uuid, true);
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ stalled: true, flipped: false });
    expect(fake.copies).toHaveLength(1);

    // The next draw flushes the old host tail on A while B remains selected.
    const slotB = geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute;
    for (const attr of [slotA, slotB]) {
      const buffer = fake.buffers.get(attr);
      if (buffer) {
        const data = fake.contents.get(buffer)!;
        for (const range of attr.updateRanges) {
          data.set(
            (attr.array as Uint32Array).subarray(range.start, range.start + range.count),
            range.start
          );
        }
        attr.clearUpdateRanges();
      }
    }
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ stalled: false, flipped: true });
    const drawn = getActiveSortedIndexAttribute(geometry)!;
    const drawnBuffer = fake.buffers.get(drawn)!;
    expect(fake.copies).toHaveLength(2);
    expect(fake.copies[1]).toEqual({ target: drawnBuffer, bytes: 48 * 4 });
    expect(Array.from(fake.contents.get(drawnBuffer)!.subarray(0, 48))).toEqual(
      Array.from({ length: 48 }, (_, i) => 47 - i)
    );
    expect(
      Array.from(fake.contents.get(drawnBuffer)!.subarray(0, 48)).sort((a, b) => a - b)
    ).toEqual(Array.from({ length: 48 }, (_, i) => i));
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
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ flipped: false });

    expect(sort).toHaveBeenCalledOnce();
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(canGpuSort(geometry, 4)).toBe(false);
    geometry.dispose();
  });

  it('uses the worker for a commit sort even when GPU ordering is ready', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, coordinator } = scene();
    texture.onUpdate?.(texture);
    const sort = vi.fn().mockResolvedValue(null);
    workerHost.api = { sort } as unknown as NonNullable<typeof workerHost.api>;
    workerHost.workerInitState = 'ready';

    scheduleSort(coordinator, mesh, mesh.uuid);
    expect(sort).toHaveBeenCalledOnce();
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(false);
    expect(fake.copies).toHaveLength(0);
    geometry.dispose();
  });

  it('falls back once when a staged GPU write finds a stale element texture', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, state, coordinator } = scene();
    texture.onUpdate?.(texture);
    const sort = vi.fn().mockResolvedValue(null);
    workerHost.api = { sort } as unknown as NonNullable<typeof workerHost.api>;
    workerHost.workerInitState = 'ready';

    scheduleSort(coordinator, mesh, mesh.uuid, true);
    texture.needsUpdate = true;
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ flipped: false });

    expect(sort).toHaveBeenCalledOnce();
    expect(fake.copies).toHaveLength(0);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    for (let i = 0; i < 3; i++) evaluateDepthSortPerFrame(coordinator);
    expect(fake.copies).toHaveLength(0);
    expect(state.lastSortAxis).not.toBeNull(); // The worker retry was dispatched.
    geometry.dispose();
  });

  it('disables a throwing GPU writer and retries once through the worker', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, coordinator } = scene();
    texture.onUpdate?.(texture);
    const sort = vi.fn().mockResolvedValue(null);
    workerHost.api = { sort } as unknown as NonNullable<typeof workerHost.api>;
    workerHost.workerInitState = 'ready';
    vi.spyOn(fake.backend.device.queue, 'submit').mockImplementation(() => {
      throw new Error('device lost');
    });

    scheduleSort(coordinator, mesh, mesh.uuid, true);
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(false);
    expect(sort).toHaveBeenCalledOnce();
    expect(canGpuSort(geometry, 4)).toBe(false);
    geometry.dispose();
  });

  it('disables a GPU producer that throws during dispatch and uses the worker thereafter', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, state, coordinator } = scene();
    texture.onUpdate?.(texture);
    const sort = vi.fn().mockResolvedValue(null);
    workerHost.api = { sort } as unknown as NonNullable<typeof workerHost.api>;
    workerHost.workerInitState = 'ready';

    // The resource was valid when accepts() ran, but a geometry lookup fails
    // while run() checks it again.
    const getAttribute = geometry.getAttribute.bind(geometry);
    let lookups = 0;
    vi.spyOn(geometry, 'getAttribute').mockImplementation((name) => {
      if (name === 'aSortedIndex' && ++lookups === 2) throw new Error('device lost');
      return getAttribute(name);
    });

    scheduleSort(coordinator, mesh, mesh.uuid, true);
    expect(sort).toHaveBeenCalledOnce();
    expect(gpuSortProducer.ready(coordinator)).toBe(false);
    await vi.waitFor(() => expect(state.inFlight).toBe(false));

    scheduleSort(coordinator, mesh, mesh.uuid, true);
    expect(sort).toHaveBeenCalledTimes(2);
    geometry.dispose();
  });

  it('clears a staged sort pose when the GPU write fails without a ready worker', async () => {
    configureRendererBackendPolicy('webgpu-native');
    const fake = fakeBackend();
    configureGpuOrderProducer({ backend: fake.backend } as unknown as Renderer);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const { geometry, texture, mesh, state, coordinator } = scene();
    texture.onUpdate?.(texture);
    workerHost.api = null;

    scheduleSort(coordinator, mesh, mesh.uuid, true);
    expect(state.lastSortAxis).not.toBeNull();
    texture.needsUpdate = true;
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ flipped: false });

    expect(fake.copies).toHaveLength(0);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(state.lastSortAxis).toBeNull();
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
