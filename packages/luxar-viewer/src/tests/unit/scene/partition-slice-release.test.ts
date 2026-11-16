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
  // As `SceneLoader`: dirty the footprint, write, then settle on success.
  const commitPart = (t: number) => {
    reg.invalidatePartitionFootprint(partPath(t));
    meshes[t].geometry = pool.acquireGSplatsGeometry(partPath(t), SPLATS_PER_PART);
    reg.settleReleasedPartitionLeaf(partPath(t));
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
    /**
     * Part 0 at t=0 with pooled leaves `pts` / `lines` (and optionally a mesh
     * `surface`), against a real pool. The release dep mirrors
     * `SceneLoader.releasePartitionPart`: release EVERY leaf (idempotently),
     * report only those whose object draws a pooled geometry.
     */
    function multiLeafHarness(options: { withMesh?: boolean } = {}) {
      let committed = 0;
      const camera = new THREE.Camera();
      camera.matrixWorldInverse.identity();
      camera.projectionMatrix.identity();
      const pool = new GPUBufferPool(64, 300, 5, () => 0);
      const leafPaths = ['/partition/part_0/pts', '/partition/part_0/lines'];
      if (options.withMesh) leafPaths.push('/partition/part_0/surface');
      const part = new THREE.Group();
      part.name = '/partition/part_0';
      const leaves = new Map(
        leafPaths.map((name) => {
          const leaf = new THREE.Mesh(new THREE.BufferGeometry());
          leaf.name = name;
          part.add(leaf);
          return [name, leaf] as const;
        })
      );
      const release = vi.fn((): readonly string[] => {
        const pooled: string[] = [];
        for (const [path, leaf] of leaves) {
          if (leaf.geometry.userData.luxarPooled === true) pooled.push(path);
          pool.releaseGSplatsGeometry(path);
        }
        return pooled;
      });
      const reg = new LODGroupRegistry({
        getCamera: () => camera,
        getViewportSize: () => ({ width: 800, height: 600 }),
        getDisplayDims: () => [0, 1, 2],
        getCommittedViewState: () => viewAt(committed),
        releasePartitionPart: release,
      });
      const groupObject = new THREE.Group();
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
      const leaf = (name: string) => leaves.get(`/partition/part_0/${name}`)!;
      /** A successful commit of a pooled leaf, as the scene loader funnels it. */
      const commit = (name: string) => {
        const path = `/partition/part_0/${name}`;
        reg.invalidatePartitionFootprint(path);
        leaves.get(path)!.geometry = pool.acquireGSplatsGeometry(path, SPLATS_PER_PART);
        reg.settleReleasedPartitionLeaf(path);
      };
      const setView = (t: number) => {
        committed = t;
        reg.applyCommittedSlice();
        reg.evaluatePerFrame();
      };
      commit('pts');
      commit('lines');
      setView(0);
      setView(1);
      expect(release).toHaveBeenCalledTimes(1);
      expect(pool.getStats().activeBuffers).toBe(0);
      return { reg, pool, part, leaf, commit, release, setView };
    }

    it('stays hidden until EVERY released leaf has recommitted', () => {
      const h = multiLeafHarness();
      // Back in slice; only `pts` commits (the `lines` load failed): `lines`
      // may still point at a pool buffer another node adopted.
      h.commit('pts');
      h.setView(0);
      expect(h.part.visible).toBe(false);
      h.commit('lines');
      h.reg.applyCommittedSlice();
      expect(h.part.visible).toBe(true);
    });

    it('a commit that is only invalidated (it threw before writing) settles nothing', () => {
      const h = multiLeafHarness();
      h.commit('pts');
      // The footprint is dirtied BEFORE the write; the write then throws.
      h.reg.invalidatePartitionFootprint('/partition/part_0/lines');
      h.setView(0);
      expect(h.part.visible).toBe(false);
    });

    it('a layer-hidden released leaf does not blank its siblings, until re-shown', () => {
      const h = multiLeafHarness();
      // The sweep skips a layer-hidden leaf, so only `pts` recommits.
      h.leaf('lines').userData.layerVisible = false;
      h.leaf('lines').visible = false;
      h.commit('pts');
      h.setView(0);
      expect(h.part.visible).toBe(true);
      // Re-showing the layer: its stale geometry must not draw before its resync commits.
      h.leaf('lines').userData.layerVisible = true;
      h.leaf('lines').visible = true;
      h.reg.evaluatePerFrame();
      expect(h.part.visible).toBe(false);
      h.commit('lines');
      h.reg.evaluatePerFrame();
      expect(h.part.visible).toBe(true);
    });

    it('keeps releasing on EVERY exit while a sibling stays pending (looped playback)', () => {
      const h = multiLeafHarness();
      // `lines` sits in a hidden layer for the whole loop: it never recommits,
      // so the part always has a pending leaf when it exits again.
      h.leaf('lines').userData.layerVisible = false;
      h.leaf('lines').visible = false;
      for (let loop = 0; loop < 3; loop++) {
        h.commit('pts');
        h.setView(0);
        expect(h.part.visible).toBe(true);
        expect(h.pool.getStats().activeBuffers).toBe(1);
        h.setView(1);
        // `pts`'s fresh buffer is given back on THIS exit too: no leak. (Each
        // loop exits twice: here, and after the uncommitted revisit below.)
        expect(h.release).toHaveBeenCalledTimes(2 + 2 * loop);
        expect(h.pool.getStats().activeBuffers).toBe(0);
        // ...and it is pending again: back in slice without its commit, the
        // part must not draw the buffer it just released.
        h.setView(0);
        expect(h.part.visible).toBe(false);
        h.setView(1);
      }
    });

    it('a leaf holding no pool buffer that never recommits does not blank its siblings', () => {
      // A mesh's release swaps in an empty placeholder (never pooled); one
      // whose budget refusal is latched never recommits again.
      const h = multiLeafHarness({ withMesh: true });
      h.commit('pts');
      h.commit('lines');
      h.setView(0);
      expect(h.part.visible).toBe(true);
      // Repeated cycles keep it that way.
      h.setView(1);
      h.commit('pts');
      h.commit('lines');
      h.setView(0);
      expect(h.part.visible).toBe(true);
    });
  });

  it('a layer-hidden released leaf that IS the part object still blocks it', () => {
    // The stamp writes a part object's `visible` regardless of its layer, so a
    // released single-object leaf would draw its released buffer.
    const h = makeHarness({ wireRelease: true });
    h.stepTo(1);
    h.meshes[0].userData.layerVisible = false;
    h.stepTo(0, { commit: false });
    expect(h.meshes[0].visible).toBe(false);
    h.stepTo(0);
    expect(h.meshes[0].visible).toBe(true);
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
