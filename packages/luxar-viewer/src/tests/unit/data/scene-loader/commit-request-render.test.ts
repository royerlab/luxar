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
 * the three SceneLoader commit methods that every late-commit path
 * converges on. These tests pin the funnel for all three geometry types
 * (three-geometry symmetry), its optionality (bare loaders must not
 * throw), and the SceneLoaderManager wire-through.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import * as THREE from 'three';
import { SceneLoader } from '../../../../data/scene-loader';
import { SceneLoaderManager } from '../../../../data/scene-loader-manager';
import type { LoadedPointsData } from '../../../../data/data-loader-types';
import type { StagedLinesCommit } from '../../../../data/scene-loader/process/data-processor-lines';
import type { StagedGSplatsCommit } from '../../../../data/scene-loader/process/data-processor-gsplats';

/** Reach-in surface for the private commit methods under test. */
interface SceneLoaderInternals {
  rootGroup: THREE.Group;
  updatePointsGeometry(path: string, data: LoadedPointsData): void;
  commitLinesGeometry(staged: StagedLinesCommit): void;
  commitGSplatsGeometry(staged: StagedGSplatsCommit): void;
}

const internals = (loader: SceneLoader): SceneLoaderInternals =>
  loader as unknown as SceneLoaderInternals;

function makePointsData(pointCount: number): LoadedPointsData {
  return {
    positions: new Float32Array(pointCount * 3),
    colors: new Uint8Array(pointCount * 3),
    radii: undefined,
    sharpness: undefined,
    pointCount,
    metadata: {
      bounds: new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)),
    },
  } as unknown as LoadedPointsData;
}

function makeMesh(name: string, nodeType: 'points' | 'lines' | 'gsplats'): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = {
    nodeType,
    attrs: {},
    visiblePointCount: 0,
    visibleSegmentCount: 0,
    visibleSplatCount: 0,
  };
  return mesh;
}

function makeStagedLines(segmentCount = 2): StagedLinesCommit {
  return {
    path: '/lines',
    sourceData: {
      positions: new Float32Array(segmentCount * 2 * 3),
      segments: new Uint32Array(segmentCount * 2),
      widths: new Float32Array(segmentCount * 2),
      colors: null,
      sharpness: null,
      segmentCount,
      vertexCount: segmentCount * 2,
      ndim: 3,
    },
    processed: {
      startPositions: new Float32Array(segmentCount * 3),
      endPositions: new Float32Array(segmentCount * 3),
      startColors: new Float32Array(segmentCount * 3),
      endColors: new Float32Array(segmentCount * 3),
      startWidths: new Float32Array(segmentCount),
      endWidths: new Float32Array(segmentCount),
      startSharpness: new Float32Array(segmentCount),
      endSharpness: new Float32Array(segmentCount),
      segmentLengths: new Float32Array(segmentCount),
      startClipped: new Uint8Array(segmentCount),
      endClipped: new Uint8Array(segmentCount),
      segmentCount,
    },
  } as StagedLinesCommit;
}

function makeStagedGSplats(splatCount = 2): StagedGSplatsCommit {
  return {
    path: '/g',
    sourceData: {
      positions: new Float32Array(splatCount * 3),
      amplitudes: new Float32Array(splatCount),
      choleskyFactors: new Float32Array(splatCount * 6),
      colors: null,
      splatCount,
      ndim: 3,
    },
    processed: {
      centers3D: new Float32Array(splatCount * 3),
      choleskyFactors3D: new Float32Array(splatCount * 6),
      amplitudes: new Float32Array(splatCount),
      colors: new Float32Array(splatCount * 3),
      splatCount,
    },
    cholesky01: new Float32Array(splatCount * 2),
    cholesky23: new Float32Array(splatCount * 2),
    cholesky45: new Float32Array(splatCount * 2),
  } as StagedGSplatsCommit;
}

function makeLoaderWithScene(): { loader: SceneLoader; spy: ReturnType<typeof vi.fn> } {
  const loader = new SceneLoader({ enableMonitor: false });
  const spy = vi.fn();
  loader.setRequestRender(spy);
  const root = new THREE.Group();
  root.add(makeMesh('/p', 'points'));
  root.add(makeMesh('/lines', 'lines'));
  root.add(makeMesh('/g', 'gsplats'));
  internals(loader).rootGroup = root;
  return { loader, spy };
}

describe('SceneLoader commit → requestRender funnel', () => {
  it('updatePointsGeometry invokes the render request', () => {
    const { loader, spy } = makeLoaderWithScene();
    internals(loader).updatePointsGeometry('/p', makePointsData(2));
    expect(spy).toHaveBeenCalled();
  });

  it('commitLinesGeometry invokes the render request', () => {
    const { loader, spy } = makeLoaderWithScene();
    internals(loader).commitLinesGeometry(makeStagedLines(2));
    expect(spy).toHaveBeenCalled();
  });

  it('commitGSplatsGeometry invokes the render request', () => {
    const { loader, spy } = makeLoaderWithScene();
    internals(loader).commitGSplatsGeometry(makeStagedGSplats(2));
    expect(spy).toHaveBeenCalled();
  });

  it('bare loaders without a callback do not throw', () => {
    const loader = new SceneLoader({ enableMonitor: false });
    const root = new THREE.Group();
    root.add(makeMesh('/p', 'points'));
    internals(loader).rootGroup = root;
    expect(() => internals(loader).updatePointsGeometry('/p', makePointsData(2))).not.toThrow();
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
