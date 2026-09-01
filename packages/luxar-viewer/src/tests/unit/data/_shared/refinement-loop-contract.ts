/**
 * Shared contract for the per-geometry progressive LOD refinement loops
 * (`runGSplatsRefinement` / `runPointsRefinement` / `runLinesRefinement` /
 * `runMeshRefinement`).
 *
 * All four wrap the same generic `runProgressiveRefinement` loop (tested in
 * isolation in `scene-loader/progressive/refinement.test.ts`), so the loop
 * *semantics* — completion, lock release, cancellation hand-off, error
 * isolation, and skip-path handling — are identical. This helper defines those
 * four cases once; each type's test file invokes it with a thin adapter that
 * wires the geometry-specific ctx fields (e.g. `gsplatLoaders` + `processGSplats`),
 * and keeps only its own type-specific tests inline.
 */

import { describe, it, expect, vi } from 'vitest';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { ViewState } from '../../../../data/data-loader-types';
import { log, Modules } from '../../../../utils/log';

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

/** Wiring passed to a type-specific runX adapter for one contract case.
 *
 * Field types match the shared shape of all four `*RefinementCtx` interfaces
 * so they assign directly; the adapter only casts the loader-map value type,
 * `deriveNodeViewState`'s return, and the `processSpy` return (which differ per
 * geometry). */
export interface RefinementRunWiring {
  /** Loader map keyed by node path. Adapter casts to its loader type. */
  loaders: Map<string, unknown>;
  viewStateQueue: ViewStateQueue;
  deriveNodeViewState: () => unknown;
  updateVisibleCountsInMonitor: () => void;
  releaseLock: () => void;
  retriggerUpdate: (pendingState: Partial<ViewState>) => void;
  /**
   * The geometry-specific "process" callback (processGSplats / processLines /
   * updatePointsGeometry). The adapter casts + wires this into the ctx so the
   * skip-path case can assert it was never invoked.
   */
  processSpy: (...args: unknown[]) => unknown;
  /** Per-run abort signal, threaded into `loader.updateView` by the wrapper. */
  signal?: AbortSignal;
}

/** A loader stub that advances through `hasMoreLODs` stages as updateView runs. */
function makeStagedLoader(stages: Array<{ hasMoreLODs: boolean }>) {
  let i = 0;
  return {
    get hasMoreLODs() {
      return stages[Math.min(i, stages.length - 1)].hasMoreLODs;
    },
    get loadedLODCount() {
      return i;
    },
    get totalLODCount() {
      return stages.length;
    },
    updateView: vi.fn().mockImplementation(async () => {
      i++;
      return null;
    }),
  };
}

/**
 * Register the shared loop-contract tests for one refinement entry point.
 *
 * @param name  Display name (e.g. `'runGSplatsRefinement'`).
 * @param label Geometry label used by the no-progress warning.
 * @param run   Adapter that invokes the type-specific runX with the given
 *              wiring and returns its promise.
 */
export function defineRefinementLoopContract(
  name: string,
  label: 'Points' | 'Lines' | 'GSplats' | 'Mesh',
  run: (wiring: RefinementRunWiring) => Promise<void>
): void {
  describe(`${name} — shared loop contract`, () => {
    it('returns immediately and releases lock when no loaders have more LODs', async () => {
      const loader = makeStagedLoader([{ hasMoreLODs: false }]);
      const releaseLock: () => void = vi.fn();
      const updateVisibleCountsInMonitor: () => void = vi.fn();

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor,
        releaseLock,
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(),
      });

      expect(releaseLock).toHaveBeenCalledTimes(1);
      expect(updateVisibleCountsInMonitor).toHaveBeenCalledTimes(1);
      expect(loader.updateView).not.toHaveBeenCalled();
    });

    it('stops after one successful pass that does not advance the pending loader', async () => {
      const loader = {
        hasMoreLODs: true,
        loadedLODCount: 2,
        totalLODCount: 4,
        updateView: vi.fn().mockResolvedValue(null),
      };
      const releaseLock: () => void = vi.fn();
      const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(),
      });

      expect(loader.updateView).toHaveBeenCalledTimes(1);
      expect(releaseLock).toHaveBeenCalledTimes(1);
      expect(warning).toHaveBeenCalledWith(
        Modules.SCENE_LOADER,
        `${label} refinement stopped for /n: no progress at LOD 2/4`
      );
      warning.mockRestore();
    });

    it('hands off lock to retriggerUpdate when pending view-state is observed', async () => {
      const queue = new ViewStateQueue();
      queue.setPending({ slicePosition: [1, 2, 3, 4] });
      const releaseLock: () => void = vi.fn();
      const retriggerUpdate: (pendingState: Partial<ViewState>) => void = vi.fn();

      await run({
        loaders: new Map([['/n', makeStagedLoader([{ hasMoreLODs: true }])]]),
        viewStateQueue: queue,
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate,
        processSpy: vi.fn(),
      });

      expect(retriggerUpdate).toHaveBeenCalledTimes(1);
      expect(retriggerUpdate).toHaveBeenCalledWith({ slicePosition: [1, 2, 3, 4] });
      // Lock NOT released — handed off to retriggerUpdate's rAF callback.
      expect(releaseLock).not.toHaveBeenCalled();
    });

    it('catches per-loader errors so other loaders continue refining', async () => {
      let attempted = false;
      const loaderA: { updateView: ReturnType<typeof vi.fn>; hasMoreLODs?: boolean } = {
        updateView: vi.fn().mockImplementation(async () => {
          attempted = true;
          throw new Error('synthetic');
        }),
      };
      Object.defineProperty(loaderA, 'hasMoreLODs', {
        get: () => !attempted,
      });

      await expect(
        run({
          loaders: new Map([['/a', loaderA]]),
          viewStateQueue: new ViewStateQueue(),
          deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
          updateVisibleCountsInMonitor: vi.fn(),
          releaseLock: vi.fn(),
          retriggerUpdate: vi.fn(),
          processSpy: vi.fn(),
        })
      ).resolves.not.toThrow();
      expect(loaderA.updateView).toHaveBeenCalledTimes(1);
    });

    it('treats an AbortError as cancellation — signal threaded, no failure backoff, clean hand-off', async () => {
      // A superseding updateView aborts the per-run controller mid-pass; the
      // in-flight read rejects with AbortError. That is cancellation, NOT
      // failure: it must not count toward the 3-strike backoff, and the
      // loop's next pass hands off to the pending state.
      const controller = new AbortController();
      const queue = new ViewStateQueue();
      let capturedSignal: AbortSignal | undefined;
      const loader = {
        hasMoreLODs: true,
        updateView: vi
          .fn()
          .mockImplementation(async (_vs: unknown, _s: unknown, signal?: AbortSignal) => {
            capturedSignal = signal;
            // Simulate the supersede that caused the abort: the new state is
            // queued and the read rejects with the abort classification.
            queue.setPending({ slicePosition: [5, 5, 5, 5] });
            throw new DOMException('aborted', 'AbortError');
          }),
      };
      const releaseLock: () => void = vi.fn();
      const retriggerUpdate: (pendingState: Partial<ViewState>) => void = vi.fn();

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: queue,
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate,
        processSpy: vi.fn(),
        signal: controller.signal,
      });

      expect(capturedSignal).toBe(controller.signal); // signal reaches the loader
      expect(loader.updateView).toHaveBeenCalledTimes(1); // no 3-strike retries
      expect(retriggerUpdate).toHaveBeenCalledWith({ slicePosition: [5, 5, 5, 5] });
      expect(releaseLock).not.toHaveBeenCalled(); // lock handed off, not released
    });

    it('repeated AbortErrors do NOT count toward the 3-strike backoff', async () => {
      // The load-bearing counterpart of the case above: 4 consecutive aborted
      // passes must NOT exhaust the loader (aborts are cancellation, not
      // failure). Mutation this pins: deleting the `isAbortError` guard in
      // the wrapper catch makes aborts hit the failure tracker — the loader
      // would be excluded after 3 strikes and updateView would be called
      // exactly 3 times instead of running the full 5-step ladder.
      let calls = 0;
      const loader = {
        hasMoreLODs: true,
        rollbackToPassStart: vi.fn(),
        updateView: vi.fn().mockImplementation(async () => {
          calls += 1;
          if (calls <= 4) throw new DOMException('aborted', 'AbortError');
          loader.hasMoreLODs = false; // 5th step completes the ladder
          return null;
        }),
      };
      const releaseLock: () => void = vi.fn();

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(), // never pending — loop runs to completion
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(),
      });

      // All 5 passes ran — the 4 aborts were not counted as strikes.
      expect(loader.updateView).toHaveBeenCalledTimes(5);
      expect(loader.rollbackToPassStart).not.toHaveBeenCalled();
      expect(releaseLock).toHaveBeenCalledTimes(1);
    });

    it('unwinds each failed processed pass before retrying the same prefix', async () => {
      let loadedLODCount = 1;
      const loader = {
        get hasMoreLODs() {
          return loadedLODCount < 3;
        },
        get loadedLODCount() {
          return loadedLODCount;
        },
        totalLODCount: 3,
        rollbackToPassStart: vi.fn().mockImplementation(() => {
          loadedLODCount--;
          return 1;
        }),
        updateView: vi.fn().mockImplementation(async () => {
          loadedLODCount++;
          return { loaded: true };
        }),
      };

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn().mockImplementation(() => {
          throw new Error('commit preparation failed');
        }),
      });

      expect(loader.updateView).toHaveBeenCalledTimes(3);
      expect(loader.rollbackToPassStart).toHaveBeenCalledTimes(3);
      expect(loader.hasMoreLODs).toBe(true);
    });

    it('still reaches the failure cap when rollback itself throws', async () => {
      const loader = {
        hasMoreLODs: true,
        rollbackToPassStart: vi.fn(() => {
          throw new Error('rollback failed');
        }),
        updateView: vi.fn().mockResolvedValue({ loaded: true }),
      };
      const releaseLock: () => void = vi.fn();

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(() => {
          throw new Error('commit preparation failed');
        }),
      });

      expect(loader.updateView).toHaveBeenCalledTimes(3);
      expect(loader.rollbackToPassStart).toHaveBeenCalledTimes(3);
      expect(releaseLock).toHaveBeenCalledOnce();
    });

    it('gives up on a persistently failing loader after 3 consecutive failures (lock released)', async () => {
      // Regression: without the per-run failure cap, a loader whose level
      // fetch always throws kept hasMoreLODs=true forever and the loop
      // retried at frame rate indefinitely, holding the update lock — a
      // network retry storm. After MAX_CONSECUTIVE_REFINEMENT_FAILURES (3)
      // the loader is excluded and the loop terminates + releases the lock.
      const failing = {
        hasMoreLODs: true, // never progresses
        updateView: vi.fn().mockRejectedValue(new Error('persistent failure')),
      };
      const releaseLock: () => void = vi.fn();

      await run({
        loaders: new Map([['/bad', failing]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(),
      });

      // Exactly 3 attempts (the cap), then the loop exits and releases.
      expect(failing.updateView).toHaveBeenCalledTimes(3);
      expect(releaseLock).toHaveBeenCalledTimes(1);
    });

    it('a failing loader does not stop a healthy loader from finishing its ladder', async () => {
      const failing = {
        hasMoreLODs: true,
        updateView: vi.fn().mockRejectedValue(new Error('persistent failure')),
      };
      // Healthy loader: 5 levels to stream, one per pass — more passes than
      // the failing loader's 3-strike budget, so it must keep advancing after
      // the failing one is excluded.
      const healthy = makeStagedLoader([
        { hasMoreLODs: true },
        { hasMoreLODs: true },
        { hasMoreLODs: true },
        { hasMoreLODs: true },
        { hasMoreLODs: true },
        { hasMoreLODs: false },
      ]);
      const releaseLock: () => void = vi.fn();

      await run({
        loaders: new Map<string, unknown>([
          ['/bad', failing],
          ['/good', healthy],
        ]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock,
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(),
      });

      expect(failing.updateView).toHaveBeenCalledTimes(3); // capped
      expect(healthy.updateView).toHaveBeenCalledTimes(5); // ladder drained
      expect(releaseLock).toHaveBeenCalledTimes(1);
    });

    it('refines a fully-extended node with the derived extended view-state (#1157)', async () => {
      // A fully-extended node is a normal node with a slice-invariant query
      // (extend-to-all tolerance + pinned slice); while its ladder still has
      // rungs it refines through this loop like any other node (the
      // `hasMoreLODs` gate — not any skip flag — stops a converged one).
      const extendedViewState: ViewState = {
        ...baseViewState,
        tolerance: [1e10, 1e10, 1e10, 1e10],
      };
      const loader = makeStagedLoader([{ hasMoreLODs: true }, { hasMoreLODs: false }]);

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
        processSpy: vi.fn(),
      });

      expect(loader.updateView).toHaveBeenCalled();
      // The refinement pass loads with the derived extended-tolerance state.
      expect(loader.updateView.mock.calls[0][0]).toBe(extendedViewState);
    });
  });
}
