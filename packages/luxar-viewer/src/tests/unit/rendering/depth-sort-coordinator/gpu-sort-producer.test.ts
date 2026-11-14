/**
 * The GPU depth sort as a sort producer: when it is ready, which nodes it
 * takes, what its device writer asks of the sorter, and how it follows the
 * host's device through compile, loss and replacement.
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GpuSortProducer } from '../../../../rendering/depth-sort-coordinator/gpu-sort-producer';
import { GpuDepthSorter } from '../../../../rendering/gpu-depth-sort/gpu-depth-sorter';
import type {
  SortJob,
  SortReason,
} from '../../../../rendering/depth-sort-coordinator/sort-producers';
import { configureRendererBackendPolicy } from '../../../../rendering/renderer-backend-policy';
import { BufferUsage } from '../../../../rendering/gpu-depth-sort/gpu-types';
import type { NodeSortState } from '../../../../rendering/depth-sort-coordinator/state';
import { attachElementStorage } from '../../../../rendering/element-storage';
import {
  LINE_TEXTURE_LAYOUT,
  SPLAT_TEXTURE_LAYOUT,
} from '../../../../rendering/element-texture-layout';
import { getGpuByteBudget } from '../../../../rendering/gpu-byte-budget';
import type { GpuSortContext } from '../../../../rendering/gpu-depth-sort/three-webgpu-bridge';
import { setCommittedData } from '../../../../types/committed-data';
import { fakeDevice } from '../gpu-depth-sort/fake-device';

const COUNT = 64;
const flush = () => new Promise((r) => setTimeout(r, 0));

function context(device = fakeDevice().device): GpuSortContext & {
  texture: ReturnType<typeof vi.fn>;
  buffer: ReturnType<typeof vi.fn>;
} {
  return {
    device,
    texture: vi.fn(() => ({ createView: () => ({}) })),
    buffer: vi.fn(() => ({ size: COUNT * 4, usage: BufferUsage.COPY_DST, destroy: () => {} })),
    readUint32: () => Promise.resolve(new Uint32Array()),
  };
}

function job(
  layout = SPLAT_TEXTURE_LAYOUT,
  extra: Partial<NodeSortState> = {},
  reason: SortReason = 'motion'
): SortJob {
  // Native WebGPU allocates storage-capable ordering pairs.
  configureRendererBackendPolicy('webgpu-native');
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, COUNT, layout);
  configureRendererBackendPolicy('webgl');
  const mesh = new THREE.Mesh(geometry);
  setCommittedData(mesh, {});
  const state: NodeSortState = {
    mesh,
    generation: 3,
    count: COUNT,
    inFlight: true,
    resortQueued: false,
    lastSortAxis: null,
    lastSortOffset: 0,
    registered: true,
    ...extra,
  };
  return { nodeId: mesh.uuid, reason, mesh, state, generation: 3, modelView: new THREE.Matrix4() };
}

async function readyProducer(ctx: GpuSortContext | null = context()) {
  let current = ctx;
  const producer = new GpuSortProducer(() => current);
  expect(producer.ready()).toBe(false); // compiling
  await flush();
  return { producer, setContext: (c: GpuSortContext | null) => (current = c) };
}

const target = () => new THREE.InstancedBufferAttribute(new Uint32Array(COUNT), 1);

afterEach(() => vi.restoreAllMocks());

describe('GpuSortProducer', () => {
  it('is ready once the pipelines compile, and never without a context', async () => {
    const { producer } = await readyProducer();
    expect(producer.ready()).toBe(true);
    expect(new GpuSortProducer(() => null).ready()).toBe(false);
  });

  it('leaves commit sorts and textures three has not uploaded to the worker', async () => {
    const ctx = context();
    const { producer } = await readyProducer(ctx);
    expect(producer.accepts(job(SPLAT_TEXTURE_LAYOUT, {}, 'commit'))).toBe(false);
    expect(producer.accepts(job(SPLAT_TEXTURE_LAYOUT, {}, 'capture'))).toBe(true);
    ctx.texture.mockReturnValueOnce(null); // its current version is not on the GPU yet
    expect(producer.accepts(job())).toBe(false);
  });

  it('takes texture-backed nodes it can hold, never a mesh', async () => {
    const { producer } = await readyProducer();
    expect(producer.accepts(job())).toBe(true);
    expect(producer.accepts(job(SPLAT_TEXTURE_LAYOUT, { triangleSource: {} as never }))).toBe(
      false
    );
    const bare = job();
    bare.mesh.geometry = new THREE.InstancedBufferGeometry();
    expect(producer.accepts(bare)).toBe(false);
  });

  it('returns a device ordering whose writer sorts the node into the target', async () => {
    const ctx = context();
    const { producer } = await readyProducer(ctx);
    const j = job(LINE_TEXTURE_LAYOUT);
    const outcome = producer.run(j);
    expect(outcome).toMatchObject({ kind: 'device', generation: 3 });
    if (outcome.kind !== 'device') throw new Error('expected a device outcome');
    expect(outcome.write.count).toBe(COUNT);

    const into = target();
    expect(outcome.write.write(into)).toBe(true);
    expect(ctx.texture).toHaveBeenCalledWith(
      (j.mesh.geometry as THREE.InstancedBufferGeometry).userData.elementTexture
    );
    expect(ctx.buffer).toHaveBeenCalledWith(into);
    expect(outcome.timing.kernelMs).toBeGreaterThanOrEqual(0);
  });

  it('declines to write an ordering whose population was superseded', async () => {
    const { producer } = await readyProducer();
    const j = job();
    const outcome = producer.run(j);
    if (outcome.kind !== 'device') throw new Error('expected a device outcome');
    j.state.generation = 4;
    expect(outcome.write.write(target())).toBe(false);
  });

  it('declines a node it failed to write until the node commits again', async () => {
    const ctx = context();
    const { producer } = await readyProducer(ctx);
    const j = job();
    ctx.buffer.mockReturnValueOnce(null); // a resource it cannot reach
    const outcome = producer.run(j);
    if (outcome.kind !== 'device') throw new Error('expected a device outcome');
    expect(outcome.write.write(target())).toBe(false);
    expect(producer.accepts(j)).toBe(false);
    j.state.generation = 4;
    expect(producer.accepts(j)).toBe(true);
  });

  it('reserves its scratch against the geometry budget and releases it', async () => {
    const before = getGpuByteBudget();
    const { producer } = await readyProducer();
    const outcome = producer.run(job());
    if (outcome.kind !== 'device') throw new Error('expected a device outcome');
    outcome.write.write(target());
    if (before > 0) expect(getGpuByteBudget()).toBeLessThanOrEqual(before);
    producer.dispose();
    expect(getGpuByteBudget()).toBe(before);
  });

  it('recompiles for a new device and falls back while it compiles', async () => {
    const { producer, setContext } = await readyProducer();
    setContext(context());
    expect(producer.ready()).toBe(false);
    await flush();
    expect(producer.ready()).toBe(true);
  });

  it('installs only the newest compile when the context flickers mid-compile', async () => {
    const dispose = vi.spyOn(GpuDepthSorter.prototype, 'dispose');
    const ctx = context();
    let current: GpuSortContext | null = ctx;
    const producer = new GpuSortProducer(() => current);
    expect(producer.ready()).toBe(false); // compile 1 starts
    current = null;
    expect(producer.ready()).toBe(false); // released
    current = ctx;
    expect(producer.ready()).toBe(false); // compile 2 starts
    await flush();
    expect(producer.ready()).toBe(true);
    // Compile 1 resolved into a superseded epoch and was disposed, not leaked.
    expect(dispose).toHaveBeenCalledTimes(1);
  });

  it('stops using a lost device', async () => {
    let lose!: () => void;
    const device = {
      ...fakeDevice().device,
      lost: new Promise<void>((r) => (lose = r)),
    };
    const { producer } = await readyProducer(context(device));
    expect(producer.ready()).toBe(true);
    lose();
    await flush();
    expect(producer.ready()).toBe(false);
    await flush();
    expect(producer.ready()).toBe(false);
  });

  it('falls back for good when the device cannot build the pipelines', async () => {
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { producer } = await readyProducer(context(fakeDevice({ failCompile: true }).device));
    expect(producer.ready()).toBe(false);
    await flush();
    expect(producer.ready()).toBe(false);
  });
});
