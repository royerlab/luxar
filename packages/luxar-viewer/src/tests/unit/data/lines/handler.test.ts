/**
 * Smoke tests for the Lines handler — mirrors points/handler.test.ts
 * and gsplats/handler.test.ts so all geometry handlers share the same
 * basic contract.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { kind, label, loadAndStage } from '../../../../data/lines/handler';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { LinesDataLoader } from '../../../../types/lines';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

describe('lines handler', () => {
  it('discriminates as kind="lines" with label="Lines"', () => {
    expect(kind).toBe('lines');
    expect(label).toBe('Lines');
  });

  it('returns null on derived.skip without calling the loader', async () => {
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const result = await loadAndStage('/l', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
    });
    expect(result).toBeNull();
    expect(loader.updateView).not.toHaveBeenCalled();
  });

  // data.md G2 fix: parallel coverage to points/handler.test.ts. The
  // three-geometry symmetry rule requires the forget-on-skip and
  // successful-load paths to be pinned for Lines too (Points had them;
  // Lines did not).
  it('forgets the path on skip so the next non-skip update re-baselines', async () => {
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const queue = new ViewStateQueue();
    const forgetPathSpy = vi.spyOn(queue, 'forgetPath');
    await loadAndStage('/l', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
    });
    expect(forgetPathSpy).toHaveBeenCalledTimes(1);
    expect(forgetPathSpy).toHaveBeenCalledWith('/l');
    expect(loader.updateView).not.toHaveBeenCalled();
  });

  it('passes the derived viewState through to loader.updateView on the non-skip path', async () => {
    // The successful-load shape differs from Points (Lines passes through
    // processLinesData), so we pin the input-to-loader contract rather than
    // the staged-output shape. A regression that bypassed the loader or
    // passed the wrong viewState would still fail this test.
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1],
    };
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null),
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    await loadAndStage('/l', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });
    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(loader.updateView).toHaveBeenCalledWith(viewState, expect.anything());
  });
});
