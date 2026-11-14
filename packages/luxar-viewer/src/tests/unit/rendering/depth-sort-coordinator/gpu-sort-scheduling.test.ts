/**
 * The GPU depth sort through the whole scheduling path: GpuSortProducer
 * selected by the scheduler, its device writer run by the element-storage pump
 * against a fake WebGPU backend, the slot flip, the fallbacks to the
 * SortWorker, and capture.
 */
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
import { GpuSortProducer } from '../../../../rendering/depth-sort-coordinator/gpu-sort-producer';
import {
  workerSortProducer,
  type SortProducer,
} from '../../../../rendering/depth-sort-coordinator/sort-producers';
import {
  evaluateDepthSortPerFrame,
  scheduleSort,
} from '../../../../rendering/depth-sort-coordinator/scheduler';
import {
  createCoordinatorState,
  workerHost,
  type NodeSortState,
} from '../../../../rendering/depth-sort-coordinator/state';
import { resortForCapture } from '../../../../rendering/depth-sort-coordinator/capture';
import {
  BufferUsage,
  type GpuBuffer,
  type GpuDevice,
} from '../../../../rendering/gpu-depth-sort/gpu-types';
import { createGpuSortContext } from '../../../../rendering/gpu-depth-sort/three-webgpu-bridge';
import { getGpuByteBudget } from '../../../../rendering/gpu-byte-budget';
import { UpdateProfiler } from '../../../../profiling/update-profiler';
import { setCommittedData } from '../../../../types/committed-data';

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
    isWebGPUBackend: true,
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

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/** A committed points node in a coordinator whose producers are the GPU sort, then the worker. */
async function scene(capacity = 4, targetUsage?: number) {
  configureRendererBackendPolicy('webgpu-native');
  const fake = fakeBackend(targetUsage);
  const context = createGpuSortContext({ backend: fake.backend })!;
  const gpu = new GpuSortProducer(() => context);
  const geometry = new THREE.InstancedBufferGeometry();
  const texture = attachElementStorage(geometry, capacity, POINT_TEXTURE_LAYOUT);
  const material = new THREE.ShaderMaterial({ uniforms: { uSortedIndexSlot: { value: 0 } } });
  material.userData.blendingMode = 'normal';
  const mesh = new THREE.Mesh(geometry, material);
  setCommittedData(mesh, {});
  const camera = new THREE.PerspectiveCamera();
  camera.updateMatrixWorld();
  const state: NodeSortState = {
    mesh,
    generation: 1,
    count: capacity,
    inFlight: false,
    resortQueued: false,
    lastSortAxis: null,
    lastSortOffset: 0,
    registered: true,
  };
  const coordinator = createCoordinatorState([gpu, workerSortProducer]);
  coordinator.nodeStates.set(mesh.uuid, state);
  coordinator.getCamera = () => camera;
  const requestRender = vi.fn();
  coordinator.requestRender = requestRender;
  const profiler = new UpdateProfiler();
  coordinator.getProfiler = () => profiler;
  gpu.ready(); // starts compiling
  await flush();
  return {
    fake,
    gpu,
    geometry,
    texture,
    mesh,
    material,
    state,
    coordinator,
    requestRender,
    profiler,
  };
}

function ready(texture: THREE.DataTexture): void {
  texture.onUpdate?.(texture); // three uploaded the current version
}

function workerReturning(result: unknown) {
  const sort = vi.fn().mockResolvedValue(result);
  workerHost.api = { sort } as unknown as NonNullable<typeof workerHost.api>;
  workerHost.workerInitState = 'ready';
  return sort;
}

describe('GPU depth sort scheduling', () => {
  const priorApi = workerHost.api;
  const priorInitState = workerHost.workerInitState;
  afterEach(() => {
    workerHost.api = priorApi;
    workerHost.workerInitState = priorInitState;
    configureRendererBackendPolicy('webgl');
  });

  it('re-sorts from the uploaded element texture into the inactive buffer, in the dispatching call', async () => {
    const {
      fake,
      gpu,
      geometry,
      texture,
      mesh,
      material,
      state,
      coordinator,
      requestRender,
      profiler,
    } = await scene();
    workerHost.api = null;
    // Not before three has uploaded the texture's current version.
    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(fake.copies).toHaveLength(0);
    ready(texture);
    const budget = getGpuByteBudget();
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(fake.copies).toEqual([{ target: fake.buffers.get(inactive), bytes: 16 }]);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(material.uniforms.uSortedIndexSlot.value).toBe(1);
    expect(inactive.version).toBe(version); // no host upload scheduled
    expect(isSortedIndexHostArrayCurrent(geometry)).toBe(false);
    expect(state.inFlight).toBe(false);
    expect(state.lastSortAxis).not.toBeNull();
    expect(requestRender).toHaveBeenCalled();
    if (budget > 0) expect(getGpuByteBudget()).toBeLessThan(budget);

    expect(profiler.getDepthSortCompletions().total).toBe(0);
    acknowledgeSortedIndexOrderingDraw(geometry);
    expect(profiler.getDepthSortCompletions().events).toEqual([
      expect.objectContaining({ seq: 1, route: 'gpu', elements: 4, queueMs: 0 }),
    ]);

    // The next re-sort lands in the other slot, again without a host upload.
    const original = geometry.getAttribute('aSortedIndex') as THREE.BufferAttribute;
    const originalVersion = original.version;
    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(fake.copies.at(-1)).toEqual({ target: fake.buffers.get(original), bytes: 16 });
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(original.version).toBe(originalVersion);
    gpu.dispose();
    expect(getGpuByteBudget()).toBe(budget);
    expect(fake.allocations.every((allocation) => allocation.destroyed)).toBe(true);
    geometry.dispose();
  });

  it('leaves a commit sort to the worker, whose host array stays current', async () => {
    const { fake, geometry, texture, mesh, coordinator } = await scene();
    ready(texture);
    const sort = workerReturning(null);
    scheduleSort(coordinator, mesh, mesh.uuid);
    expect(sort).toHaveBeenCalledOnce();
    expect(fake.copies).toHaveLength(0);
    geometry.dispose();
  });

  it('declines a texture version that has not reached the GPU', async () => {
    const { fake, geometry, texture, mesh, coordinator } = await scene();
    ready(texture);
    texture.needsUpdate = true;
    const sort = workerReturning(null);
    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(sort).toHaveBeenCalledOnce();
    expect(fake.copies).toHaveLength(0);
    geometry.dispose();
  });

  it('keeps an undrawn identity tail from overwriting a later GPU ordering', async () => {
    const { fake, geometry, texture, mesh, state, coordinator } = await scene(64);
    workerHost.api = null;
    ready(texture);
    writeSortedIndexIdentity(geometry, 32);
    geometry.instanceCount = 32;
    const slotA = geometry.getAttribute('aSortedIndex') as THREE.InstancedBufferAttribute;
    fake.backend.createStorageAttribute(slotA);
    slotA.clearUpdateRanges(); // The initial 32-element identity was drawn.
    writeSortedIndexIdentity(geometry, 48);
    state.count = 48;
    geometry.instanceCount = 48;
    expect(slotA.updateRanges).toEqual([{ start: 32, count: 16 }]);

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion'); // writes B, flips to B
    scheduleSort(coordinator, mesh, mesh.uuid, 'motion'); // A has an undrawn host tail: waits
    expect(fake.copies).toHaveLength(1);
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ stalled: true, flipped: false });

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
    geometry.dispose();
  });

  it('stops using a device whose target it cannot write, re-sorting on the worker at once', async () => {
    const { fake, geometry, texture, mesh, gpu, coordinator } = await scene(4, BufferUsage.VERTEX);
    ready(texture);
    const sort = workerReturning(null);
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const compile = vi.spyOn(fake.backend.device, 'createComputePipelineAsync');

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');

    expect(sort).toHaveBeenCalledOnce();
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(gpu.ready()).toBe(false);
    // The device is not merely released but given up on: no recompile on it.
    await flush();
    expect(gpu.ready()).toBe(false);
    expect(compile).not.toHaveBeenCalled();
    vi.restoreAllMocks();
    geometry.dispose();
  });

  it('gives up on a device whose backend throws while resolving the sort resources', async () => {
    const { fake, geometry, texture, mesh, gpu, coordinator } = await scene();
    ready(texture);
    const sort = workerReturning(null);
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    fake.backend.createStorageAttribute = () => {
      throw new Error('cannot create the buffer');
    };

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');

    expect(sort).toHaveBeenCalledOnce(); // retried on the worker at once
    expect(fake.copies).toHaveLength(0);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    await flush();
    expect(gpu.ready()).toBe(false);
    // The next camera-motion re-sort goes to the worker without trying the GPU.
    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(sort).toHaveBeenCalledTimes(2);
    vi.restoreAllMocks();
    geometry.dispose();
  });

  it('re-sorts on the worker when a device writer throws instead of returning false', async () => {
    const { geometry, texture, mesh, coordinator } = await scene();
    ready(texture);
    const sort = workerReturning(null);
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const throwing: SortProducer = {
      route: 'gpu',
      ready: () => true,
      accepts: (job) => job.reason !== 'commit',
      run: (job) => ({
        kind: 'device',
        generation: job.generation,
        write: {
          count: 4,
          write: () => {
            throw new Error('writer threw');
          },
        },
        timing: { route: 'gpu', kernelMs: 0, boundaryMs: 0, queueMs: 0 },
      }),
    };
    coordinator.sortProducers = [throwing, workerSortProducer];

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');

    expect(sort).toHaveBeenCalledOnce();
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    vi.restoreAllMocks();
    geometry.dispose();
  });

  it('frees the scratch on releaseScratch and allocates it again for the next sort', async () => {
    const { fake, geometry, texture, mesh, gpu, coordinator } = await scene();
    ready(texture);
    workerHost.api = null;
    const budget = getGpuByteBudget();
    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(fake.copies).toHaveLength(1);
    const allocated = fake.allocations.length;
    expect(fake.allocations.some((allocation) => !allocation.destroyed)).toBe(true);

    gpu.releaseScratch();
    expect(fake.allocations.every((allocation) => allocation.destroyed)).toBe(true);
    expect(getGpuByteBudget()).toBe(budget);
    expect(gpu.ready()).toBe(true); // the pipelines stay

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(fake.copies).toHaveLength(2);
    expect(fake.allocations.length).toBeGreaterThan(allocated);
    gpu.dispose();
    geometry.dispose();
  });

  it('re-sorts once on the worker when a staged GPU write finds a stale texture', async () => {
    const { fake, geometry, texture, mesh, state, coordinator } = await scene();
    ready(texture);
    // A host stream holds the device write behind it until the next pump.
    writeSortedIndexIdentity(geometry, 4);
    (geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute).addUpdateRange(0, 4);
    const sort = workerReturning(null);

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(fake.copies).toHaveLength(0);
    texture.needsUpdate = true;
    (geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute).clearUpdateRanges();
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ flipped: false });

    expect(sort).toHaveBeenCalledOnce();
    for (let i = 0; i < 3; i++) evaluateDepthSortPerFrame(coordinator);
    expect(fake.copies).toHaveLength(0);
    expect(state.lastSortAxis).not.toBeNull(); // the worker retry was dispatched
    geometry.dispose();
  });

  it('forgets the pose when a GPU write fails and no worker is ready', async () => {
    const { fake, geometry, texture, mesh, state, coordinator } = await scene();
    ready(texture);
    workerHost.api = null;
    writeSortedIndexIdentity(geometry, 4);
    (geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute).addUpdateRange(0, 4);

    scheduleSort(coordinator, mesh, mesh.uuid, 'motion');
    expect(state.lastSortAxis).not.toBeNull();
    texture.needsUpdate = true;
    (geometry.getAttribute('aSortedIndexB') as THREE.InstancedBufferAttribute).clearUpdateRanges();
    expect(pumpSortedIndexOrderingApply(geometry)).toMatchObject({ flipped: false });

    expect(fake.copies).toHaveLength(0);
    expect(activeSortedIndexSlot(geometry)).toBe(0);
    expect(state.lastSortAxis).toBeNull();
    geometry.dispose();
  });

  it('refreshes a capture on the GPU even when no worker is available', async () => {
    const { fake, geometry, texture, coordinator } = await scene();
    ready(texture);
    workerHost.api = null;

    await resortForCapture(coordinator);

    expect(fake.copies).toHaveLength(1);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    geometry.dispose();
  });
});
