/**
 * #3332: a per-timepoint `kind=partition` part that LEAVES the committed slice
 * gives its GPU geometry back (`LODGroupRegistryDeps.releasePartitionPart`).
 *
 * Before the fix nothing released a hidden part: its pool buffer stayed
 * ACTIVE, which the byte-budget evictor never touches, so stepping through a
 * 51-part time-lapse grew resident VRAM by one part per step, forever.
 *
 * The harness mirrors what the scene loader does around a pass, against a REAL
 * `GPUBufferPool`: the pass for view `t` commits the part at `t` (acquire its
 * buffer, `invalidatePartitionFootprint`), then moves the committed view and
 * calls `applyCommittedSlice`. The release dep returns the buffer to the pool,
 * as `SceneLoader.releasePartitionPart` does via `releaseNodeGeometry`.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';

import { LODGroupRegistry, type LODGroupRegistryDeps } from '../../../scene/lod-group-registry';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import type { ViewState } from '../../../data/data-loader-types';

const DIMENSIONS = [
  { name: 'X', unit: 'um', scale: 1 },
  { name: 'Y', unit: 'um', scale: 1 },
  { name: 'Z', unit: 'um', scale: 1 },
  { name: 'Time', unit: 'frame', scale: 1, discrete: true, step: 1 },
];

const N_PARTS = 40;
const SPLATS_PER_PART = 1000;

function viewAt(t: number): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, t],
    tolerance: [0, 0, 0, 0],
    dimensions: DIMENSIONS,
  };
}

const partPath = (t: number) => `/partition/part_${t}`;

interface Harness {
  reg: LODGroupRegistry;
  pool: GPUBufferPool;
  camera: THREE.Camera;
  meshes: THREE.Mesh[];
  release: ReturnType<typeof vi.fn<(path: string) => readonly string[]>>;
  /** One view pass for `t`: commit the part at `t`, then commit the view. */
  stepTo(t: number, opts?: { commit?: boolean }): void;
}

function makeHarness(options: { wireRelease: boolean; budgetBytes?: number }): Harness {
  let committed = 0;
  const camera = new THREE.Camera();
  camera.matrixWorldInverse.identity();
  camera.projectionMatrix.identity();
  const pool = new GPUBufferPool(64, 300, 5, () => options.budgetBytes ?? 0);
  const release = vi.fn((path: string): readonly string[] => {
    pool.releaseGSplatsGeometry(path);
    return [path];
  });
  const deps: LODGroupRegistryDeps = {
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
    getCommittedViewState: () => viewAt(committed),
    ...(options.wireRelease ? { releasePartitionPart: release } : {}),
  };
  const reg = new LODGroupRegistry(deps);
  const groupObject = new THREE.Group();
  const meshes = Array.from({ length: N_PARTS }, (_, t) => {
    const mesh = new THREE.Mesh(new THREE.BufferGeometry());
    mesh.name = partPath(t);
    groupObject.add(mesh);
    return mesh;
  });
  reg.registerPartition({
    path: '/partition',
    groupObject,
    children: meshes.map((mesh, t) => ({
      path: partPath(t),
      objects: [mesh],
      positionBounds: { min: [-0.5, -0.5, -0.5, t], max: [0.5, 0.5, 0.5, t] },
    })),
  });
  const commitPart = (t: number) => {
    reg.invalidatePartitionFootprint(partPath(t));
    meshes[t].geometry = pool.acquireGSplatsGeometry(partPath(t), SPLATS_PER_PART);
  };
  const stepTo = (t: number, opts: { commit?: boolean } = {}) => {
    if (opts.commit !== false) commitPart(t);
    committed = t;
    reg.applyCommittedSlice();
    reg.evaluatePerFrame();
  };
  // The initial load commits t=0. The first stamp also releases the parts
  // outside that slice: loaded eagerly here (as on the scene loader's no-view
  // fallback, which defers nothing), they hold data that cannot draw.
  stepTo(0);
  expect(release).toHaveBeenCalledTimes(options.wireRelease ? N_PARTS - 1 : 0);
  release.mockClear();
  return { reg, pool, camera, meshes, release, stepTo };
}

describe('partition slice exit releases the part (#3332)', () => {
  it('control: without the release dep, every visited part stays an active buffer', () => {
    const h = makeHarness({ wireRelease: false });
    for (let t = 1; t < N_PARTS; t++) h.stepTo(t);
    // The leak: one active (never-evictable) buffer per visited timepoint.
    expect(h.pool.getStats().activeBuffers).toBe(N_PARTS);
    h.pool.dispose();
  });

  it('stepping through every timepoint keeps ONE active buffer and bounded bytes', () => {
    const h = makeHarness({ wireRelease: true });
    const onePart = h.pool.getResidentBytes();
    expect(onePart).toBeGreaterThan(0);
    // Budget: a couple of parts' worth, so released buffers become evictable.
    const budgeted = makeHarness({ wireRelease: true, budgetBytes: 2 * onePart });
    for (let t = 1; t < N_PARTS; t++) {
      h.stepTo(t);
      budgeted.stepTo(t);
      expect(h.pool.getStats().activeBuffers).toBe(1);
      expect(budgeted.pool.getStats().activeBuffers).toBe(1);
      // Under a byte budget the released buffers are actually reclaimed.
      expect(budgeted.pool.getResidentBytes()).toBeLessThanOrEqual(2 * onePart);
    }
    // Each part was released exactly once, on its own slice exit.
    expect(h.release).toHaveBeenCalledTimes(N_PARTS - 1);
    for (let t = 0; t < N_PARTS - 1; t++) expect(h.release).toHaveBeenCalledWith(partPath(t));
    // Only the current timepoint draws.
    expect(h.meshes.filter((m) => m.visible).map((m) => m.name)).toEqual([partPath(N_PARTS - 1)]);
    h.pool.dispose();
    budgeted.pool.dispose();
  });

  it('stepping BACK to a released timepoint reloads and shows it', () => {
    const h = makeHarness({ wireRelease: true, budgetBytes: 1 });
    for (let t = 1; t < 10; t++) h.stepTo(t);
    expect(h.meshes[3].visible).toBe(false);

    h.stepTo(3);
    expect(h.meshes[3].visible).toBe(true);
    expect(h.meshes[3].userData.partitionFrustumVisible).toBe(true);
    // It draws its OWN freshly acquired buffer, sized for its own data.
    const stats = h.pool.getStats();
    expect(stats.activeBuffers).toBe(1);
    expect(h.meshes[3].geometry.userData.luxarPooled).toBe(true);
    expect(h.meshes[3].geometry).not.toBe(h.meshes[9].geometry);
    // And the part it left (t=9) is released in turn: no accumulation.
    expect(h.release).toHaveBeenLastCalledWith(partPath(9));
    h.stepTo(4);
    expect(h.meshes[3].visible).toBe(false);
    expect(h.pool.getStats().activeBuffers).toBe(1);
    h.pool.dispose();
  });

  it('a released part back in slice stays hidden until a commit lands under it', () => {
    const h = makeHarness({ wireRelease: true });
    h.stepTo(1);
    // The committed view returns to t=0, but no commit reached part 0 (its
    // load failed, or the pass skipped it as frustum-culled): its mesh may still
    // reference a pool buffer another node adopted, so it must not draw.
    h.stepTo(0, { commit: false });
    expect(h.meshes[0].visible).toBe(false);
    expect(h.meshes[0].userData.partitionFrustumVisible).toBe(false);
    // The view pass still sweeps it: only the frustum half gates a pass.
    expect(h.meshes[0].userData.partitionInFrustum).toBe(true);

    h.stepTo(0);
    expect(h.meshes[0].visible).toBe(true);
  });

  it('a frustum cull does not release (a camera pan must not thrash reloads)', () => {
    const h = makeHarness({ wireRelease: true });
    h.camera.position.set(100, 0, 0);
    h.camera.updateMatrixWorld();
    h.reg.evaluatePerFrame();
    expect(h.meshes[0].visible).toBe(false);
    expect(h.release).not.toHaveBeenCalled();
    h.camera.position.set(0, 0, 0);
    h.camera.updateMatrixWorld();
    h.reg.evaluatePerFrame();
    expect(h.meshes[0].visible).toBe(true);
    expect(h.release).not.toHaveBeenCalled();
  });

  it('a part that leaves the slice while frustum-culled is still released', () => {
    const h = makeHarness({ wireRelease: true });
    h.camera.position.set(100, 0, 0);
    h.camera.updateMatrixWorld();
    h.reg.evaluatePerFrame();
    // Precondition: part 0 really is off-screen (frustum-culled) before the step.
    expect(h.meshes[0].visible).toBe(false);
    expect(h.meshes[0].userData.partitionInFrustum).toBe(false);
    h.stepTo(1, { commit: false });
    expect(h.release).toHaveBeenCalledWith(partPath(0));
  });

  it('a deferred (not yet activated) or pathless part is never released', () => {
    let committed = 0;
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    const release = vi.fn((path: string): readonly string[] => [path]);
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getCommittedViewState: () => viewAt(committed),
      releasePartitionPart: release,
    });
    const groupObject = new THREE.Group();
    const deferred = new THREE.Group();
    const pathless = new THREE.Group();
    groupObject.add(deferred, pathless);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/deferred',
          objects: [deferred],
          positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
          activate: () => Promise.resolve(),
        },
        {
          path: '',
          objects: [pathless],
          positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
        },
      ],
    });
    reg.applyCommittedSlice();
    committed = 1;
    reg.applyCommittedSlice();
    expect(release).not.toHaveBeenCalled();
    expect(pathless.visible).toBe(false);
  });

  it('a release that frees nothing does not hold the part hidden', () => {
    let committed = 0;
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getCommittedViewState: () => viewAt(committed),
      releasePartitionPart: () => [],
    });
    const groupObject = new THREE.Group();
    const part = new THREE.Group();
    groupObject.add(part);
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
        },
      ],
    });
    reg.applyCommittedSlice();
    committed = 1;
    reg.applyCommittedSlice();
    expect(part.visible).toBe(false);
    committed = 0;
    reg.applyCommittedSlice();
    expect(part.visible).toBe(true);
  });
  describe('a part with several leaves', () => {
    function multiLeafHarness() {
      let committed = 0;
      const camera = new THREE.Camera();
      camera.matrixWorldInverse.identity();
      camera.projectionMatrix.identity();
      const leaves = ['/partition/part_0/pts', '/partition/part_0/lines'];
      const release = vi.fn((): readonly string[] => leaves);
      const reg = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        getCommittedViewState: () => viewAt(committed),
        releasePartitionPart: release,
      });
      const groupObject = new THREE.Group();
      const part = new THREE.Group();
      part.name = '/partition/part_0';
      const [pts, lines] = leaves.map((name) => {
        const leaf = new THREE.Mesh(new THREE.BufferGeometry());
        leaf.name = name;
        part.add(leaf);
        return leaf;
      });
      groupObject.add(part);
      reg.registerPartition({
        path: '/partition',
        groupObject,
        children: [
          {
            path: '/partition/part_0',
            objects: [part],
            positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
          },
        ],
      });
      const setView = (t: number) => {
        committed = t;
        reg.applyCommittedSlice();
        reg.evaluatePerFrame();
      };
      setView(0);
      setView(1);
      expect(release).toHaveBeenCalledTimes(1);
      return { reg, part, pts, lines, leaves, setView };
    }

    it('stays hidden until EVERY released leaf has recommitted', () => {
      const h = multiLeafHarness();
      // Back in slice; only `pts` commits (the `lines` load failed): `lines`
      // may still point at a pool buffer another node adopted.
      h.reg.invalidatePartitionFootprint(h.leaves[0]);
      h.setView(0);
      expect(h.part.visible).toBe(false);
      h.reg.invalidatePartitionFootprint(h.leaves[1]);
      h.reg.applyCommittedSlice();
      expect(h.part.visible).toBe(true);
    });

    it('a layer-hidden released leaf does not blank its siblings, until re-shown', () => {
      const h = multiLeafHarness();
      // The sweep skips a layer-hidden leaf, so only `pts` recommits.
      h.lines.userData.layerVisible = false;
      h.lines.visible = false;
      h.reg.invalidatePartitionFootprint(h.leaves[0]);
      h.setView(0);
      expect(h.part.visible).toBe(true);
      // Re-showing the layer: its stale geometry must not draw before its resync commits.
      h.lines.userData.layerVisible = true;
      h.lines.visible = true;
      h.reg.evaluatePerFrame();
      expect(h.part.visible).toBe(false);
      h.reg.invalidatePartitionFootprint(h.leaves[1]);
      h.reg.evaluatePerFrame();
      expect(h.part.visible).toBe(true);
    });
  });

  it('the per-frame frustum gate never releases, even on a slice exit it sees first', () => {
    let committed = 0;
    const camera = new THREE.Camera();
    camera.matrixWorldInverse.identity();
    camera.projectionMatrix.identity();
    const release = vi.fn((path: string): readonly string[] => [path]);
    const reg = new LODGroupRegistry({
      getCamera: () => camera,
      getViewportSize: () => ({ width: 800, height: 600 }),
      getDisplayDims: () => [0, 1, 2],
      getCommittedViewState: () => viewAt(committed),
      releasePartitionPart: release,
    });
    const groupObject = new THREE.Group();
    const part = new THREE.Group();
    groupObject.add(part);
    // Registered mid-pass, outside the committed slice: the frame gate hides
    // it, but a rung load for its loader may be in flight — no release here.
    committed = 1;
    reg.registerPartition({
      path: '/partition',
      groupObject,
      children: [
        {
          path: '/partition/part_0',
          objects: [part],
          positionBounds: { min: [-0.5, -0.5, -0.5, 0], max: [0.5, 0.5, 0.5, 0] },
        },
      ],
    });
    reg.evaluatePerFrame();
    expect(part.visible).toBe(false);
    expect(release).not.toHaveBeenCalled();
    // The pass's own committed-slice stamp, under its lock, releases it.
    reg.applyCommittedSlice();
    expect(release).toHaveBeenCalledWith('/partition/part_0');
  });
});
