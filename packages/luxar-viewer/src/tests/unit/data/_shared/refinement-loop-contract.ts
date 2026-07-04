/**
 * Shared contract for the per-geometry progressive LOD refinement loops
 * (`runGSplatsRefinement` / `runPointsRefinement` / `runLinesRefinement`).
 *
 * All three wrap the same generic `runProgressiveRefinement` loop (tested in
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

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

/** Wiring passed to a type-specific runX adapter for one contract case.
 *
 * Field types match the shared shape of all three `*RefinementCtx` interfaces
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
}

/** A loader stub that advances through `hasMoreLODs` stages as updateView runs. */
function makeStagedLoader(stages: Array<{ hasMoreLODs: boolean }>) {
  let i = 0;
  return {
    get hasMoreLODs() {
      return stages[Math.min(i, stages.length - 1)].hasMoreLODs;
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
 * @param run   Adapter that invokes the type-specific runX with the given
 *              wiring and returns its promise.
 */
export function defineRefinementLoopContract(
  name: string,
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
      const loaderA: { updateView: ReturnType<typeof vi.fn>; hasMoreLODs?: boolean } = {
        updateView: vi.fn().mockRejectedValue(new Error('synthetic')),
      };
      // hasMoreLODs true on entry (enter try block), false afterwards (loop exits).
      Object.defineProperty(loaderA, 'hasMoreLODs', {
        get: vi.fn().mockReturnValueOnce(true).mockReturnValue(false),
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

    it('skips loaders whose derived view-state is fully-extended', async () => {
      let calls = 0;
      const loader: { updateView: ReturnType<typeof vi.fn>; hasMoreLODs?: boolean } = {
        get hasMoreLODs() {
          calls += 1;
          return calls === 1;
        },
        updateView: vi.fn(),
      };
      const processSpy: (...args: unknown[]) => unknown = vi.fn();

      await run({
        loaders: new Map([['/n', loader]]),
        viewStateQueue: new ViewStateQueue(),
        deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
        processSpy,
      });

      expect(loader.updateView).not.toHaveBeenCalled();
      expect(processSpy).not.toHaveBeenCalled();
    });
  });
}
