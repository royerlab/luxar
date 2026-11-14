import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import {
  attachElementStorage,
  pumpSortedIndexOrderingApply,
} from '../../../../rendering/element-storage';
import { POINT_TEXTURE_LAYOUT } from '../../../../rendering/element-texture-layout';
import { scheduleSort } from '../../../../rendering/depth-sort-coordinator/scheduler';
import {
  createCoordinatorState,
  type NodeSortState,
} from '../../../../rendering/depth-sort-coordinator/state';
import type { SortProducer } from '../../../../rendering/depth-sort-coordinator/sort-producers';
import { setCommittedData } from '../../../../types/committed-data';

function stageWorkerOrdering(
  geometry: THREE.InstancedBufferGeometry,
  ordering: Uint32Array,
  triangleSource?: Uint32Array
): void {
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial());
  setCommittedData(mesh, {});
  const producer: SortProducer = {
    route: 'worker',
    ready: () => true,
    accepts: () => true,
    run: (job) => ({
      kind: 'host',
      generation: job.generation,
      ordering,
      timing: { route: 'worker', kernelMs: 1, boundaryMs: 1, queueMs: 0 },
    }),
  };
  const c = createCoordinatorState([producer]);
  const camera = new THREE.PerspectiveCamera();
  camera.updateMatrixWorld();
  c.getCamera = () => camera;
  c.nodeStates.set(mesh.uuid, {
    mesh,
    generation: 1,
    count: ordering.length,
    inFlight: false,
    resortQueued: false,
    lastSortAxis: null,
    lastSortOffset: 0,
    registered: true,
    triangleSource,
  } satisfies NodeSortState);
  scheduleSort(c, mesh, mesh.uuid);
}

describe('worker depth-order handoff', () => {
  it('stages the complete worker permutation when compute is unavailable', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    attachElementStorage(geometry, 4, POINT_TEXTURE_LAYOUT);
    const inactive = geometry.getAttribute('aSortedIndexB') as THREE.BufferAttribute;
    const version = inactive.version;
    stageWorkerOrdering(geometry, new Uint32Array([2, 0, 3, 1]));
    pumpSortedIndexOrderingApply(geometry);
    expect(Array.from(inactive.array).slice(0, 4)).toEqual([2, 0, 3, 1]);
    expect(inactive.version).toBeGreaterThan(version);
    geometry.dispose();
  });

  it('writes indexed mesh triangles from the worker permutation', () => {
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(6), 1));
    geometry.setDrawRange(0, 6);
    stageWorkerOrdering(geometry, new Uint32Array([1, 0]), new Uint32Array([0, 1, 2, 3, 4, 5]));
    expect(Array.from(geometry.index!.array)).toEqual([3, 4, 5, 0, 1, 2]);
    geometry.dispose();
  });
});
