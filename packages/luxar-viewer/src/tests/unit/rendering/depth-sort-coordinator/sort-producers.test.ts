/**
 * The scheduler's producer seam: a sort goes to the first ready producer that
 * accepts the node, and every outcome — synchronous or not, a host array or a
 * device writer — settles through the same stale-drop, staging, profiler and
 * re-sort rules.
 */
import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  activeSortedIndexSlot,
  attachElementStorage,
  cancelAllSortedIndexOrderingApplies,
  getActiveSortedIndexAttribute,
  pumpSortedIndexOrderingApply,
  writeSortedIndexIdentity,
  type DeviceOrderingWrite,
} from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';
import { configureRendererBackendPolicy } from '../../../../rendering/renderer-backend-policy';
import { scheduleSort } from '../../../../rendering/depth-sort-coordinator/scheduler';
import {
  anySortProducerReady,
  selectSortProducer,
  type SortJob,
  type SortOutcome,
  type SortProducer,
} from '../../../../rendering/depth-sort-coordinator/sort-producers';
import {
  createCoordinatorState,
  type CoordinatorState,
  type NodeSortState,
} from '../../../../rendering/depth-sort-coordinator/state';
import { setCommittedData } from '../../../../types/committed-data';

const COUNT = 16;
const TIMING = { route: 'gpu', kernelMs: 0.5, boundaryMs: 0, queueMs: 0 } as const;

function producer(
  route: 'worker' | 'gpu',
  run: SortProducer['run'],
  options: { ready?: boolean; accepts?: boolean } = {}
): SortProducer & { run: ReturnType<typeof vi.fn> } {
  return {
    route,
    ready: () => options.ready ?? true,
    accepts: () => options.accepts ?? true,
    run: vi.fn(run),
  };
}

/** A committed, sorting points node in a coordinator using `producers`. */
function setup(producers: SortProducer[]): {
  c: CoordinatorState;
  mesh: THREE.Mesh;
  state: NodeSortState;
} {
  configureRendererBackendPolicy('webgpu-native');
  const geometry = new THREE.InstancedBufferGeometry();
  attachElementStorage(geometry, COUNT, POINT_TEXTURE_LAYOUT);
  writeSortedIndexIdentity(geometry, COUNT);
  geometry.instanceCount = COUNT;
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  setCommittedData(mesh, {});
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(0, 0, 5);
  camera.updateMatrixWorld(true);
  const c = createCoordinatorState(producers);
  c.getCamera = () => camera;
  c.requestRender = vi.fn();
  const state: NodeSortState = {
    mesh,
    generation: 7,
    count: COUNT,
    inFlight: false,
    resortQueued: false,
    lastSortAxis: null,
    lastSortOffset: 0,
    registered: true,
  };
  c.nodeStates.set(mesh.uuid, state);
  return { c, mesh, state };
}

const reversed = () => new Uint32Array(COUNT).map((_, i) => COUNT - 1 - i);

function hostOutcome(generation: number): SortOutcome {
  return { kind: 'host', generation, ordering: reversed(), timing: TIMING };
}

afterEach(() => {
  cancelAllSortedIndexOrderingApplies();
  configureRendererBackendPolicy('webgl');
});

describe('producer selection', () => {
  it('takes the first producer that is ready and accepts the node', () => {
    const notReady = producer('gpu', () => null, { ready: false });
    const refuses = producer('gpu', () => null, { accepts: false });
    const worker = producer('worker', () => null);
    const { c, mesh, state } = setup([notReady, refuses, worker]);
    const job = { nodeId: mesh.uuid, mesh, state, generation: 7, modelView: new THREE.Matrix4() };
    expect(selectSortProducer(c, job as SortJob)).toBe(worker);
    expect(anySortProducerReady(c)).toBe(true);
    expect(anySortProducerReady(createCoordinatorState([notReady]))).toBe(false);
  });

  it('dispatches nothing, and releases a held draw, when no producer can sort', () => {
    const idle = producer('worker', () => null, { ready: false });
    const { c, mesh, state } = setup([idle]);
    scheduleSort(c, mesh, mesh.uuid);
    expect(idle.run).not.toHaveBeenCalled();
    expect(state.inFlight).toBe(false);
    expect(state.lastSortAxis).toBeNull();
  });
});

describe('settling an outcome', () => {
  it('stages a synchronous host ordering in the same call', () => {
    const sync = producer('gpu', (job) => hostOutcome(job.generation));
    const { c, mesh, state } = setup([sync]);
    scheduleSort(c, mesh, mesh.uuid);
    expect(state.inFlight).toBe(false);
    expect(state.lastSortAxis).not.toBeNull();
    expect(c.requestRender).toHaveBeenCalled();
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(true);
    expect(Array.from(getActiveSortedIndexAttribute(geometry)!.array as Uint32Array)).toEqual(
      Array.from(reversed())
    );
  });

  it('writes and flips a device ordering in the dispatching call', () => {
    const write = vi.fn<DeviceOrderingWrite['write']>(() => true);
    const device = producer('gpu', (job) => ({
      kind: 'device',
      generation: job.generation,
      write: { count: COUNT, write },
      timing: TIMING,
    }));
    const { c, mesh } = setup([device]);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    const inactive = geometry.getAttribute('aSortedIndexB');
    scheduleSort(c, mesh, mesh.uuid);
    // Before the render this per-frame pass precedes, not at the next pump.
    expect(write).toHaveBeenCalledWith(inactive);
    expect(activeSortedIndexSlot(geometry)).toBe(1);
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(false);
  });

  it('drops an outcome for another generation, or for a population it does not cover', () => {
    const stale = producer('gpu', () => hostOutcome(6));
    const { c, mesh } = setup([stale]);
    scheduleSort(c, mesh, mesh.uuid);
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(pumpSortedIndexOrderingApply(geometry).flipped).toBe(false);

    const short = producer('gpu', (job) => ({
      kind: 'device',
      generation: job.generation,
      write: { count: COUNT - 1, write: () => true },
      timing: TIMING,
    }));
    const second = setup([short]);
    scheduleSort(second.c, second.mesh, second.mesh.uuid);
    const g2 = second.mesh.geometry as THREE.InstancedBufferGeometry;
    expect(pumpSortedIndexOrderingApply(g2).flipped).toBe(false);
  });

  it('runs a re-sort queued during an asynchronous sort once it settles', async () => {
    let resolve!: (o: SortOutcome) => void;
    const slow = producer(
      'worker',
      () =>
        new Promise<SortOutcome>((r) => {
          resolve = r;
        })
    );
    const { c, mesh, state } = setup([slow]);
    scheduleSort(c, mesh, mesh.uuid);
    scheduleSort(c, mesh, mesh.uuid);
    expect(state.resortQueued).toBe(true);
    expect(slow.run).toHaveBeenCalledTimes(1);
    resolve(hostOutcome(7));
    await Promise.resolve();
    await Promise.resolve();
    expect(slow.run).toHaveBeenCalledTimes(2);
    expect(state.inFlight).toBe(true);
  });

  it('clears in-flight and drains the queue when a producer throws', () => {
    let calls = 0;
    const throwing = producer('gpu', () => {
      calls++;
      if (calls === 1) {
        // A commit lands mid-sort and queues a re-sort.
        throwingState.resortQueued = true;
        throw new Error('device lost');
      }
      return null;
    });
    const { c, mesh, state: throwingState } = setup([throwing]);
    scheduleSort(c, mesh, mesh.uuid);
    expect(throwingState.inFlight).toBe(false);
    expect(throwing.run).toHaveBeenCalledTimes(2);
  });
});
