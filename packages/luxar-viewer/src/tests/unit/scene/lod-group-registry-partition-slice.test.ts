/**
 * Partition slice gating (B4): a `kind=partition` part whose bounds miss the
 * committed hidden-dimension slice draws nothing, so the registry treats it
 * exactly like a part outside the frustum — hidden, and ineligible for the
 * refinement / background loading that reads `isObjectLoadEligible`.
 *
 * The committed view reaches the registry through the optional
 * `getCommittedViewState` dependency (the scene loader's last committed view).
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { LODGroupRegistry, type LODGroupRegistryDeps } from '../../../scene/lod-group-registry';
import { isObjectLoadEligible } from '../../../data/scene-loader/loaders/run-loader-updates';
import type { ViewState } from '../../../data/data-loader-types';

const DIMENSIONS = [
  { name: 'X', unit: 'um', scale: 1 },
  { name: 'Y', unit: 'um', scale: 1 },
  { name: 'Z', unit: 'um', scale: 1 },
  { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1 },
];

function viewAt(t: number): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, t],
    tolerance: [0, 0, 0, 0],
    dimensions: DIMENSIONS,
  };
}

/** A registry whose identity camera frames NDC [-1, 1]^3, at committed view `view()`. */
function makeRegistry(view: () => ViewState): LODGroupRegistry {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  const deps: LODGroupRegistryDeps & { getCommittedViewState: () => ViewState } = {
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
    getCommittedViewState: view,
  };
  return new LODGroupRegistry(deps);
}

/** Two parts sharing one in-frustum XYZ cell, one at t=0 and one at t=3. */
function registerTimeParts(reg: LODGroupRegistry) {
  const groupObject = new THREE.Group();
  const now = new THREE.Group();
  const later = new THREE.Group();
  groupObject.add(now, later);
  reg.registerPartition({
    path: '/partition',
    groupObject,
    children: [
      {
        path: '/partition/part_0',
        objects: [now],
        positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
      },
      {
        path: '/partition/part_1',
        objects: [later],
        positionBounds: { min: [-0.5, -0.5, -0.5, 3], max: [0.5, 0.5, 0.5, 3] },
      },
    ],
  });
  return { now, later };
}

describe('LODGroupRegistry — partition slice gating (B4)', () => {
  it('an out-of-slice part is not frustum-visible', () => {
    const reg = makeRegistry(() => viewAt(0));
    const { now, later } = registerTimeParts(reg);

    reg.evaluatePerFrame();

    expect(now.visible).toBe(true);
    expect(now.userData.partitionFrustumVisible).toBe(true);
    expect(later.visible).toBe(false);
    expect(later.userData.partitionFrustumVisible).toBe(false);
    // Refinement and background loading read this predicate: an out-of-slice
    // part must not climb its additive ladder.
    expect(isObjectLoadEligible(now)).toBe(true);
    expect(isObjectLoadEligible(later)).toBe(false);
  });

  it('a part entering the committed slice becomes visible again', () => {
    let t = 0;
    const reg = makeRegistry(() => viewAt(t));
    const { now, later } = registerTimeParts(reg);
    reg.evaluatePerFrame();
    expect(later.visible).toBe(false);

    t = 3;
    reg.evaluatePerFrame();
    expect(now.visible).toBe(false);
    expect(later.visible).toBe(true);
    expect(later.userData.partitionFrustumVisible).toBe(true);
  });

  it('a pass skips loaders under a part outside ITS slice, whatever is committed', () => {
    const reg = makeRegistry(() => viewAt(0));
    registerTimeParts(reg);
    expect(reg.isPathInPartitionSlice('/partition/part_1/leaf', viewAt(0))).toBe(false);
    expect(reg.isPathInPartitionSlice('/partition/part_1/leaf', viewAt(3))).toBe(true);
    expect(reg.isPathInPartitionSlice('/partition/part_0', viewAt(0))).toBe(true);
    expect(reg.isPathInPartitionSlice('/elsewhere', viewAt(0))).toBe(true);
  });

  it('applyCommittedSlice re-gates parts at once, without waiting for a frame', () => {
    let t = 0;
    const reg = makeRegistry(() => viewAt(t));
    const { now, later } = registerTimeParts(reg);
    reg.evaluatePerFrame();
    t = 3;
    reg.applyCommittedSlice();
    expect(now.visible).toBe(false);
    expect(later.visible).toBe(true);
    expect(isObjectLoadEligible(later)).toBe(true);
  });

  it('activates a deferred part once, for a pass whose slice holds it', async () => {
    let committed = 0;
    const reg = makeRegistry(() => viewAt(committed));
    const groupObject = new THREE.Group();
    const slot = new THREE.Group();
    groupObject.add(slot);
    let activations = 0;
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [slot],
          positionBounds: { min: [-0.5, -0.5, -0.5, 2], max: [0.5, 0.5, 0.5, 2] },
          activate: async () => {
            activations++;
            slot.add(new THREE.Group());
          },
        },
      ],
    });
    reg.evaluatePerFrame();

    await reg.activatePartitionParts(viewAt(0));
    expect(activations).toBe(0);

    await reg.activatePartitionParts(viewAt(2));
    await reg.activatePartitionParts(viewAt(2));
    expect(activations).toBe(1);
    // Hidden until the pass that needed it commits its view.
    expect(slot.visible).toBe(false);
    committed = 2;
    reg.applyCommittedSlice();
    expect(slot.visible).toBe(true);
  });

  it('an activation changes nothing drawn: the commit of its pass does', async () => {
    const reg = makeRegistry(() => viewAt(0));
    const groupObject = new THREE.Group();
    const slot = new THREE.Group();
    groupObject.add(slot);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [slot],
          positionBounds: { min: [-0.5, -0.5, -0.5, 2], max: [0.5, 0.5, 0.5, 2] },
          // Attaches the part's (still empty) placeholder, as a registration does.
          activate: async () => {
            slot.add(new THREE.Group());
          },
        },
      ],
    });
    reg.evaluatePerFrame();
    reg.takeDrawnStateChanged();

    await reg.activatePartitionParts(viewAt(2));

    // A frame requested here would redraw the committed scene unchanged, and
    // the pass's commit then requests its own: two renders for one step.
    expect(reg.takeDrawnStateChanged()).toBe(false);
  });

  it('asks for a resync pass for a deferred part in the frustum and committed slice', () => {
    const requestReprocess = vi.fn();
    const camera = new THREE.Camera();
    const deps: LODGroupRegistryDeps & { getCommittedViewState: () => ViewState } = {
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getCommittedViewState: () => viewAt(0),
      requestReprocess,
      isUpdateInProgress: () => false,
    };
    const reg = new LODGroupRegistry(deps);
    const groupObject = new THREE.Group();
    const slot = new THREE.Group();
    groupObject.add(slot);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [slot],
          positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
          activate: () => Promise.resolve(),
        },
      ],
    });

    reg.evaluatePerFrame();
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledOnce();
    expect(requestReprocess).toHaveBeenCalledWith(['/partition/part_0']);
  });

  it('keeps a part whose bounds lie within half a step of the slice (renderer membership)', () => {
    const reg = makeRegistry(() => viewAt(1));
    const groupObject = new THREE.Group();
    const straddle = new THREE.Group();
    groupObject.add(straddle);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [straddle],
          // No element is closer than 0.5 to t=1, which the gsplats/lines
          // half-step membership gate still admits: never gate it out.
          positionBounds: { min: [-0.5, -0.5, -0.5, 1.5], max: [0.5, 0.5, 0.5, 2] },
        },
      ],
    });
    reg.evaluatePerFrame();
    expect(straddle.visible).toBe(true);
  });
});
