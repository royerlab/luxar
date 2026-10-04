/**
 * Regression tests for the render-loop wake-up on geometry commits.
 *
 * The viewer's rAF loop idle-pauses after `config.animation.idleTimeoutMs`
 * (2 s). Progressive-refinement passes, failed-load retries, and the
 * online auto-retry all commit geometry AFTER the sweep that started
 * them — if the loop has idled meanwhile, nothing painted the committed
 * data until the next user input (stale frame shown while LOD chunks
 * upload invisibly).
 *
 * The fix funnels a `requestRender` callback (wired by the app pipeline
 * to `AnimationController.startAnimation`, which is idempotent) through
 * the SceneLoader commit methods that every late-commit path converges
 * on. The per-type funnel is the geometry-behaviour matrix row
 * `commitRequestsRender`, probed below for all four types; the rest pin its
 * optionality (bare loaders must not throw), the footprint invalidation on a
 * throwing commit, and the SceneLoaderManager wire-through.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { SceneLoader } from '../../../../data/scene-loader';
import { SceneLoaderManager } from '../../../../data/scene-loader-manager';
import type { StagedMeshCommit } from '../../../../data/scene-loader/process/data-processor-mesh';
import {
  commitInternals as internals,
  GEOMETRY_COMMITS,
  makeCommitScene,
  makePointsData,
  makeStagedGSplats,
  makeStagedLines,
} from '../../../helpers/geometry-commits';
import { defineBehaviourConformance } from '../../../_conformance/define-behaviour-conformance';

function makeMesh(name: string, nodeType: 'points' | 'lines' | 'gsplats'): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = { nodeType, attrs: {} };
  return mesh;
}

function makeLoaderWithScene(): {
  loader: SceneLoader;
  spy: ReturnType<typeof vi.fn>;
  invalidatePartitionFootprint: ReturnType<typeof vi.fn>;
} {
  const loader = new SceneLoader({ enableMonitor: false });
  const spy = vi.fn();
  const invalidatePartitionFootprint = vi.fn();
  loader.setRequestRender(spy);
  internals(loader).lodGroupRegistry = { invalidatePartitionFootprint };
  makeCommitScene(loader);
  return { loader, spy, invalidatePartitionFootprint };
}

describe('SceneLoader commit → requestRender funnel', () => {
  it('commitMeshGeometry invalidates the partition footprint', () => {
    const { loader, invalidatePartitionFootprint } = makeLoaderWithScene();
    internals(loader).rootGroup = null;
    internals(loader).commitMeshGeometry({ path: '/mesh' } as StagedMeshCommit);
    expect(invalidatePartitionFootprint).toHaveBeenCalledWith('/mesh');
  });

  it('invalidates the partition footprint when a pooled points commit throws', () => {
    const { loader, invalidatePartitionFootprint } = makeLoaderWithScene();
    const geometry = new THREE.BufferGeometry();
    internals(loader)._gpuBufferPool = {
      acquirePointsGeometry: vi.fn(() => geometry),
      updatePointsGeometry: vi.fn(() => {
        throw new Error('upload failed');
      }),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };

    expect(() => internals(loader).updatePointsGeometry('/p', makePointsData(2))).toThrow(
      'upload failed'
    );
    expect(invalidatePartitionFootprint).toHaveBeenCalledWith('/p');
  });

  it('invalidates the partition footprint when a pooled lines commit throws', () => {
    const { loader, invalidatePartitionFootprint } = makeLoaderWithScene();
    const geometry = new THREE.BufferGeometry();
    internals(loader)._gpuBufferPool = {
      acquireLinesGeometry: vi.fn(() => geometry),
      updateLinesGeometry: vi.fn(() => {
        throw new Error('upload failed');
      }),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };

    expect(() => internals(loader).commitLinesGeometry(makeStagedLines(2))).toThrow(
      'upload failed'
    );
    expect(invalidatePartitionFootprint).toHaveBeenCalledWith('/lines');
  });

  it('invalidates the partition footprint when a pooled gsplats commit throws', () => {
    const { loader, invalidatePartitionFootprint } = makeLoaderWithScene();
    const geometry = new THREE.BufferGeometry();
    internals(loader)._gpuBufferPool = {
      acquireGSplatsGeometry: vi.fn(() => geometry),
      updateGSplatsGeometry: vi.fn(() => {
        throw new Error('upload failed');
      }),
      didLastAcquireRebuildAttributes: vi.fn(() => true),
    };

    expect(() => internals(loader).commitGSplatsGeometry(makeStagedGSplats(2))).toThrow(
      'upload failed'
    );
    expect(invalidatePartitionFootprint).toHaveBeenCalledWith('/g');
  });

  it('bare loaders without a callback do not throw', () => {
    const loader = new SceneLoader({ enableMonitor: false });
    const root = new THREE.Group();
    root.add(makeMesh('/p', 'points'));
    internals(loader).rootGroup = root;
    expect(() => internals(loader).updatePointsGeometry('/p', makePointsData(2))).not.toThrow();
  });
});

// A commit to a node the frame does not draw (an LOD level the registry keeps
// hidden, a node under a hidden layer) cannot change the picture: the loop must
// still WAKE (the registry polls per frame and may decide to show the level it
// just committed), but it must not redraw. During timelapse playback with a
// held fine level, every timepoint also re-commits the hidden eager coarse
// level; redrawing for it cost one wasted render per tick (#2944 C3).
describe('SceneLoader commit → requestRender says whether the frame changed', () => {
  // The geometry-behaviour matrix row `commitRequestsRender`
  // (tests/_conformance/geometry-behaviours.ts): every type's commit wakes the
  // loop, and asks for a redraw only when its node is drawn.
  defineBehaviourConformance('commitRequestsRender', {
    async holds(type) {
      const adapter = GEOMETRY_COMMITS[type];

      const drawn = makeLoaderWithScene();
      await adapter.commit(drawn.loader, 2);
      expect(drawn.spy).toHaveBeenCalledWith(true);
      expect(drawn.spy).not.toHaveBeenCalledWith(false);
      expect(drawn.invalidatePartitionFootprint).toHaveBeenCalledWith(adapter.path);

      const hidden = makeLoaderWithScene();
      internals(hidden.loader).rootGroup!.getObjectByName(adapter.path)!.visible = false;
      await adapter.commit(hidden.loader, 2);
      expect(hidden.spy).toHaveBeenCalledWith(false);
      expect(hidden.spy).not.toHaveBeenCalledWith(true);
    },
  });

  it('a commit under a hidden ancestor wakes without a redraw', () => {
    const { loader, spy } = makeLoaderWithScene();
    const root = internals(loader).rootGroup!;
    const layer = new THREE.Group();
    layer.visible = false;
    layer.add(root.getObjectByName('/p')!);
    root.add(layer);
    internals(loader).updatePointsGeometry('/p', makePointsData(2));
    expect(spy).toHaveBeenCalledWith(false);
    expect(spy).not.toHaveBeenCalledWith(true);
  });
});

describe('SceneLoaderManager requestRender wire-through', () => {
  afterEach(() => {
    SceneLoaderManager.getInstance().setRequestRender(null);
    SceneLoaderManager.getInstance().destroyLoader('rr-test');
  });

  it('a loader created after setRequestRender fires the callback on commit', () => {
    const manager = SceneLoaderManager.getInstance();
    const spy = vi.fn();
    manager.setRequestRender(spy);

    const loader = manager.createLoader('rr-test', { enableMonitor: false }, false);
    const root = new THREE.Group();
    root.add(makeMesh('/p', 'points'));
    internals(loader).rootGroup = root;

    internals(loader).updatePointsGeometry('/p', makePointsData(2));
    expect(spy).toHaveBeenCalled();
  });
});
