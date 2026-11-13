/**
 * The GPU depth sort as a sort producer: when it is ready, which nodes it
 * takes, what its device writer asks of the sorter, and how it follows the
 * host's device through compile, loss and replacement.
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GpuSortProducer } from '../../../../rendering/depth-sort-coordinator/gpu-sort-producer';
import type { SortJob } from '../../../../rendering/depth-sort-coordinator/sort-producers';
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
    buffer: vi.fn(() => ({ size: COUNT * 4, destroy: () => {} })),
    readUint32: () => Promise.resolve(new Uint32Array()),
  };
}

function job(layout = SPLAT_TEXTURE_LAYOUT, extra: Partial<NodeSortState> = {}): SortJob {
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, COUNT, layout);
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
  return { nodeId: mesh.uuid, mesh, state, generation: 3, modelView: new THREE.Matrix4() };
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
