/**
 * Smoke tests for the Lines handler — mirrors points/handler.test.ts
 * and gsplats/handler.test.ts so all geometry handlers share the same
 * basic contract.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { kind, label, loadAndStage } from '../../../../data/lines/handler';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { LinesDataLoader, LinesMetadata, LoadedLinesData } from '../../../../types/lines';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

// A fully-extended node's derived view state: extend-to-all sentinel on the
// (single) non-displayed dim + that dim's slicePosition pinned to 0. Present
// even though lines opt out of the PARTIAL override — `deriveNodeViewState`
// computes it unconditionally for the full-extend case, so a fully-extended
// lines node loads a slice-invariant query as a normal node (#1157).
const extendedViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1e10],
};

/** Root group with a lines mesh stamped as already-committed. */
function rootWithLinesMesh(path: string): THREE.Group {
  const root = new THREE.Group();
  const mesh = new THREE.Mesh();
  mesh.name = path;
  mesh.userData = { nodeType: 'lines', attrs: {}, loadedViewVersion: 1 };
  root.add(mesh);
  return root;
}

describe('lines handler', () => {
  it('discriminates as kind="lines" with label="Lines"', () => {
    expect(kind).toBe('lines');
    expect(label).toBe('Lines');
  });

  it('loads a fully-extended lines node with the derived extended+pinned view state on FIRST paint (#1157)', async () => {
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null), // empty slice: exits before process
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const session = makeSession();
    await loadAndStage('/l', loader, session, {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      // A fully-extended node is a normal node — no skip shortcut.
      deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
    });
    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(loader.updateView).toHaveBeenCalledWith(extendedViewState, expect.anything(), undefined);
    expect(session.markSkipped).not.toHaveBeenCalled();
  });

  it('still queries the loader on a later sweep even with a committed mesh (no skip shortcut)', async () => {
    // Regression guard for the reverted "skip once loaded" design.
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null),
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const session = makeSession();
    await loadAndStage('/l', loader, session, {
      rootGroup: rootWithLinesMesh('/l'),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 2,
      updateVersion: 2,
      deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
    });
    expect(loader.updateView).toHaveBeenCalledWith(extendedViewState, expect.anything(), undefined);
    expect(session.markSkipped).not.toHaveBeenCalled();
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
    // 3rd arg is the per-update signal (undefined when none is supplied).
    expect(loader.updateView).toHaveBeenCalledWith(viewState, expect.anything(), undefined);
  });

  it('forwards the per-update abort signal to loader.updateView', async () => {
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1],
    };
    const updateView = vi.fn().mockResolvedValue(null);
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView,
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const ac = new AbortController();
    await loadAndStage('/l', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      signal: ac.signal,
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });
    expect(updateView).toHaveBeenCalledWith(viewState, expect.anything(), ac.signal);
  });

  it('skips predictive prefetch when the update was superseded (signal aborted)', async () => {
    // P8 symmetry with the Points handler's gating test: warming chunks for
    // an abandoned view-state wastes bandwidth and pollutes the per-path
    // prefetch baseline.
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1],
    };
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null), // no data -> stage step exits early
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const queue = new ViewStateQueue();
    const prefetchSpy = vi.spyOn(queue, 'dispatchPrefetch');
    const controller = new AbortController();
    controller.abort();

    await loadAndStage('/l', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      signal: controller.signal,
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });

    expect(prefetchSpy).not.toHaveBeenCalled();
  });

  // data.md G2 symmetry [P8]: Points has a "returns the staged commit on a
  // successful load" test; Lines lacked one. Add the parallel coverage.
  // Unlike Points (which returns `{ path, data }` verbatim from the loader),
  // the Lines handler runs the loaded data through `processLinesData`, so the
  // staged commit shape is `{ path, processed: ProcessedLinesData }`. We feed
  // a real lines mesh + a one-segment LoadedLinesData and pin that shape.
  it('returns the staged commit (path + processed lines) on a successful load', async () => {
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [0, 0, 0],
    };

    // One fully-visible 3D segment (both endpoints, displayDims = all dims,
    // so nothing is clipped away → the projector emits exactly 1 segment).
    const loadedData: LoadedLinesData = {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([1, 1]),
      colors: null,
      sharpness: null,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };

    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn().mockResolvedValue(loadedData),
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;

    // The handler resolves the node by name off the rootGroup and reads its
    // lines userData; processLinesData bails to null without a valid mesh.
    const attrs: LinesMetadata = {
      type: 'lines',
      n_vertices: 2,
      n_segments: 1,
      ndim: 3,
      original_line_type: 'segments',
      max_width: 1,
      has_colors: false,
      has_sharpness: false,
      ordering: 'none',
    };
    const mesh = new THREE.Mesh();
    mesh.name = '/l';
    mesh.userData = { nodeType: 'lines', loader, attrs, maxWidth: 1 };
    const rootGroup = new THREE.Group();
    rootGroup.add(mesh);

    const staged = await loadAndStage('/l', loader, makeSession(), {
      rootGroup,
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 5,
      updateVersion: 5,
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });

    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(staged).not.toBeNull();
    expect(staged?.path).toBe('/l');
    // Staged shape is { path, sourceData, processed } (NOT { path, data } like Points).
    if (!staged || staged.noop) throw new Error('expected a geometry staged commit');
    expect(staged.processed).toBeDefined();
    expect(staged.processed.segmentCount).toBe(1);
    expect(staged.processed.startPositions).toBeInstanceOf(Float32Array);
    expect(staged.processed.endPositions).toBeInstanceOf(Float32Array);
    expect(staged.processed.startPositions.length).toBe(3); // 1 segment × xyz
    expect(staged.processed.endPositions.length).toBe(3);
  });
});
