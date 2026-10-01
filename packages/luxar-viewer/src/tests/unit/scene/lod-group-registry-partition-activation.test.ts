/**
 * Deferred partition part activation (B4): what happens to an activation whose
 * claiming pass does not survive it, and to one that fails.
 *
 * A pass that activates a part "claims" it: it awaits the activation and sweeps
 * the part's new loaders itself. A claim is only as good as the pass holding
 * it — a superseded (aborted) pass commits nothing — so an activation settling
 * under a dead claim must resync the part when the committed view shows it,
 * exactly like an unclaimed (prefetch-started) one. A rejected activation must
 * stay re-armable through Retry instead of leaving an empty placeholder for the
 * rest of the session.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { LODGroupRegistry, type LODGroupRegistryDeps } from '../../../scene/lod-group-registry';
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

type RequestReprocess = (paths: readonly string[]) => void;

function makeRegistry(requestReprocess: RequestReprocess): LODGroupRegistry {
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  const deps: LODGroupRegistryDeps & { getCommittedViewState: () => ViewState } = {
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
    getCommittedViewState: () => viewAt(0),
    requestReprocess,
    isUpdateInProgress: () => false,
  };
  return new LODGroupRegistry(deps);
}

function registerLazyPart(reg: LODGroupRegistry, activate: () => Promise<void>) {
  const groupObject = new THREE.Group();
  const slot = new THREE.Group();
  groupObject.add(slot);
  reg.registerPartition({
    path: '/p',
    groupObject,
    children: [
      {
        path: '/p/part_0',
        objects: [slot],
        positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
        activate,
      },
    ],
  });
  groupObject.updateMatrixWorld(true);
  return { groupObject, slot };
}

describe('LODGroupRegistry — partition part activation (B4)', () => {
  it('resyncs a part whose claiming pass was aborted while its activation ran', async () => {
    const requestReprocess = vi.fn<RequestReprocess>();
    const reg = makeRegistry(requestReprocess);
    let resolveActivation!: () => void;
    const { groupObject, slot } = registerLazyPart(
      reg,
      () =>
        new Promise<void>((resolve) => {
          resolveActivation = resolve;
        })
    );
    reg.evaluatePerFrame(); // wanted -> resync requested
    expect(requestReprocess).toHaveBeenCalledOnce();
    requestReprocess.mockClear();

    // The resync pass claims the activation, then is superseded.
    const pass = new AbortController();
    const claimed = reg.activatePartitionParts(viewAt(0), new Set(['/p/part_0']), pass.signal);
    pass.abort();
    // The part leaves the frustum and re-enters it while the activation runs.
    groupObject.position.x = 1000;
    groupObject.updateMatrixWorld(true);
    reg.evaluatePerFrame();
    expect(slot.visible).toBe(false);
    groupObject.position.x = 0;
    groupObject.updateMatrixWorld(true);
    reg.evaluatePerFrame();
    expect(slot.visible).toBe(true);

    resolveActivation();
    await claimed;
    for (let i = 0; i < 3; i++) reg.evaluatePerFrame();

    // Nothing will sweep the new loaders for the committed view otherwise.
    expect(requestReprocess).toHaveBeenCalledWith(['/p/part_0']);
  });

  it('asks for no resync when the claiming pass is alive: it sweeps the part itself', async () => {
    const requestReprocess = vi.fn<RequestReprocess>();
    const reg = makeRegistry(requestReprocess);
    registerLazyPart(reg, () => Promise.resolve());
    reg.evaluatePerFrame();
    requestReprocess.mockClear();

    const pass = new AbortController();
    await reg.activatePartitionParts(viewAt(0), new Set(['/p/part_0']), pass.signal);
    for (let i = 0; i < 3; i++) reg.evaluatePerFrame();

    expect(requestReprocess).not.toHaveBeenCalled();
  });

  it('a failed activation is not retried by itself, and Retry re-arms it', async () => {
    const requestReprocess = vi.fn<RequestReprocess>();
    const reg = makeRegistry(requestReprocess);
    const activate = vi
      .fn<() => Promise<void>>()
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValue(undefined);
    registerLazyPart(reg, activate);
    reg.evaluatePerFrame();
    requestReprocess.mockClear();

    await reg.activatePartitionParts(viewAt(0), new Set(['/p/part_0']));
    expect(activate).toHaveBeenCalledOnce();
    // No per-frame retry loop against a failing store.
    for (let i = 0; i < 3; i++) reg.evaluatePerFrame();
    expect(requestReprocess).not.toHaveBeenCalled();

    // The user's Retry (retryFailedLoader -> retryLazyChildByNodePath).
    expect(reg.retryLazyChildByNodePath('/p/part_0')).toBe(true);
    reg.evaluatePerFrame();
    expect(requestReprocess).toHaveBeenCalledWith(['/p/part_0']);
    await reg.activatePartitionParts(viewAt(0), new Set(['/p/part_0']));
    expect(activate).toHaveBeenCalledTimes(2);
  });

  it('an activation settling after the registry was cleared asks for nothing', async () => {
    const requestReprocess = vi.fn<RequestReprocess>();
    const reg = makeRegistry(requestReprocess);
    let resolveActivation!: () => void;
    registerLazyPart(
      reg,
      () =>
        new Promise<void>((resolve) => {
          resolveActivation = resolve;
        })
    );
    reg.evaluatePerFrame();
    // Started ahead of any pass (prefetchSlice), then the dataset is switched.
    const run = reg.activatePartitionParts(viewAt(0), undefined, false);
    reg.clear();
    // The next dataset registers a partition at the same path, off screen.
    const next = registerLazyPart(reg, () => Promise.resolve());
    next.groupObject.position.x = 1000;
    next.groupObject.updateMatrixWorld(true);
    reg.evaluatePerFrame();
    requestReprocess.mockClear();

    resolveActivation();
    await run;
    reg.evaluatePerFrame();

    // The old dataset's part must not resync a path of the new one.
    expect(requestReprocess).not.toHaveBeenCalled();
  });

  // #2944 review B: the `requested` flag was cleared only when a pass reached
  // the part, so a resync that ended without activating it (rejected, or
  // superseded by a pass targeting other parts) left the part asking for
  // nothing for the rest of the session.
  it('asks again for a part whose requested resync never activated it', () => {
    const requestReprocess = vi.fn<RequestReprocess>(); // the pass never activates
    let t = 0;
    let tickRequested = true;
    const camera = new THREE.Camera();
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getCommittedViewState: () => viewAt(0),
      requestReprocess,
      isUpdateInProgress: () => false,
      now: () => t,
      requestTick: () => {
        tickRequested = true;
      },
    } as LODGroupRegistryDeps);
    registerLazyPart(reg, () => Promise.resolve());
    // An on-demand 60 Hz loop over a parked camera: a frame runs only when
    // something asked for a tick.
    for (let frame = 0; frame < 600; frame++) {
      t += 1000 / 60;
      if (!tickRequested) continue;
      tickRequested = false;
      reg.evaluatePerFrame();
    }
    expect(requestReprocess.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect(requestReprocess).toHaveBeenLastCalledWith(['/p/part_0']);
  });

  it('does not keep the loop ticking for a resync nobody can run (no requestReprocess)', () => {
    const requestTick = vi.fn();
    const camera = new THREE.Camera();
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getCommittedViewState: () => viewAt(0),
      isUpdateInProgress: () => false,
      requestTick,
    } as LODGroupRegistryDeps);
    registerLazyPart(reg, () => Promise.resolve());
    for (let frame = 0; frame < 5; frame++) reg.evaluatePerFrame();
    requestTick.mockClear();
    for (let frame = 0; frame < 5; frame++) reg.evaluatePerFrame();
    expect(requestTick).not.toHaveBeenCalled();
  });
});
