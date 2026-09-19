/**
 * Unit tests for scene-owned nD dimension loading orchestration.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: vi.fn(),
    setDimensionValue: vi.fn(),
  },
}));

vi.mock('../../../data', () => ({
  updateSceneForDimensions: vi.fn().mockResolvedValue(undefined),
  prefetchSceneForDimensions: vi.fn(),
  releasePrefetchResources: vi.fn(),
}));

import {
  SCRUB_SETTLE_MS,
  cancelScrubSettleForTests,
  updateAllNDNodes,
  type DimensionLoadingContext,
} from '../../../scene/dimension-loading';
import { sceneDimsManager } from '../../../scene/scene-dims-manager';
import {
  prefetchSceneForDimensions,
  releasePrefetchResources,
  updateSceneForDimensions,
} from '../../../data';

function makeCtx(overrides: Partial<DimensionLoadingContext> = {}): DimensionLoadingContext {
  return {
    sceneManager: { scene: {} } as never,
    animationController: { startAnimation: vi.fn() } as never,
    getAnimationManager: () => undefined,
    ...overrides,
  };
}

beforeEach(() => {
  (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReset();
  (updateSceneForDimensions as ReturnType<typeof vi.fn>).mockReset().mockResolvedValue(undefined);
  (prefetchSceneForDimensions as ReturnType<typeof vi.fn>).mockReset();
  (releasePrefetchResources as ReturnType<typeof vi.fn>).mockReset();
  cancelScrubSettleForTests();
});

describe('updateAllNDNodes', () => {
  it('no-ops when sceneDimsManager.getDims returns null (no scene loaded)', async () => {
    (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue(null);
    const ctx = makeCtx();

    await updateAllNDNodes(ctx);

    expect(updateSceneForDimensions).not.toHaveBeenCalled();
  });

  it('forwards dims + scene to the loader and kicks the animation loop', async () => {
    const dims = { ndim: 4, displayed: [0, 1, 2], currentStep: [0, 0, 0, 5], metadata: undefined };
    (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue(dims);
    const startAnimation = vi.fn();
    const ctx = makeCtx({ animationController: { startAnimation } as never });

    await updateAllNDNodes(ctx);

    expect(updateSceneForDimensions).toHaveBeenCalledWith(dims, expect.anything(), undefined, {
      frameBudgetMs: undefined,
    });
    expect(startAnimation).toHaveBeenCalledTimes(1);
  });

  describe('t+1 shadow prefetch trigger', () => {
    const dims = {
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 5],
      metadata: undefined,
    };

    function makePlayingAnim(overrides: Record<string, unknown> = {}) {
      return {
        dispose: vi.fn(),
        getFrameBudgetMs: vi.fn(() => 60),
        getPlaybackLadderDepth: vi.fn(() => null),
        getScrubLadderDepth: vi.fn(() => null),
        isAnyPlaying: vi.fn(() => true),
        getPlayingDimIndices: vi.fn(() => [3]),
        peekNextValue: vi.fn(() => 6),
        ...overrides,
      } as never;
    }

    beforeEach(() => {
      (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue(dims);
    });

    it('while playing: prefetches the PEEKED next dims with the frame budget', async () => {
      const ctx = makeCtx({ getAnimationManager: () => makePlayingAnim() });

      await updateAllNDNodes(ctx);

      expect(updateSceneForDimensions).toHaveBeenCalledWith(dims, expect.anything(), undefined, {
        frameBudgetMs: 60,
      });
      expect(prefetchSceneForDimensions).toHaveBeenCalledTimes(1);
      const [predictedDims, , loaderId, opts] = (
        prefetchSceneForDimensions as ReturnType<typeof vi.fn>
      ).mock.calls[0];
      expect(predictedDims.currentStep).toEqual([0, 0, 0, 6]);
      expect(dims.currentStep).toEqual([0, 0, 0, 5]);
      expect(loaderId).toBeUndefined();
      expect(opts).toEqual({ budgetMs: 60 });
      expect(releasePrefetchResources).not.toHaveBeenCalled();
    });

    it('threads a pinned playback ladder depth to BOTH the foreground update and the t+1 prefetch', async () => {
      const ctx = makeCtx({
        getAnimationManager: () => makePlayingAnim({ getPlaybackLadderDepth: vi.fn(() => 6) }),
      });

      await updateAllNDNodes(ctx);

      expect(updateSceneForDimensions).toHaveBeenCalledWith(dims, expect.anything(), undefined, {
        frameBudgetMs: 60,
        ladderDepth: 6,
      });
      const [, , , opts] = (prefetchSceneForDimensions as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(opts).toEqual({ budgetMs: 60, ladderDepth: 6 });
    });

    it('not playing: releases prefetch resources and does not prefetch', async () => {
      const animationManager = makePlayingAnim({
        isAnyPlaying: vi.fn(() => false),
        getFrameBudgetMs: vi.fn(() => null),
      });
      const ctx = makeCtx({ getAnimationManager: () => animationManager });

      await updateAllNDNodes(ctx);

      expect(prefetchSceneForDimensions).not.toHaveBeenCalled();
      expect(releasePrefetchResources).toHaveBeenCalledTimes(1);
    });

    it("peek returned null for every playing dim ('once' at boundary): no prefetch fired", async () => {
      const animationManager = makePlayingAnim({ peekNextValue: vi.fn(() => null) });
      const ctx = makeCtx({ getAnimationManager: () => animationManager });

      await updateAllNDNodes(ctx);

      expect(prefetchSceneForDimensions).not.toHaveBeenCalled();
    });

    it('no animation manager at all: silently releases (idempotent path)', async () => {
      const ctx = makeCtx();

      await updateAllNDNodes(ctx);

      expect(prefetchSceneForDimensions).not.toHaveBeenCalled();
      expect(releasePrefetchResources).toHaveBeenCalledTimes(1);
    });
  });
});

describe('scrub pinning (not playing)', () => {
  const dims = { ndim: 4, displayed: [0, 1, 2], currentStep: [0, 0, 0, 5], metadata: undefined };

  function makeIdleAnim(scrubDepth: number | 'auto' | null) {
    return {
      dispose: vi.fn(),
      getFrameBudgetMs: vi.fn(() => null),
      getPlaybackLadderDepth: vi.fn(() => null),
      getScrubLadderDepth: vi.fn(() => scrubDepth),
      isAnyPlaying: vi.fn(() => false),
      getPlayingDimIndices: vi.fn(() => []),
      peekNextValue: vi.fn(() => null),
    } as never;
  }

  beforeEach(() => {
    (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue(dims);
    (sceneDimsManager as unknown as { setDimensionValue?: unknown }).setDimensionValue = vi.fn();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('pins a scrub pass to the scrub detail without a frame budget, then schedules an unpinned settle pass', async () => {
    const ctx = makeCtx({ getAnimationManager: () => makeIdleAnim(6) });

    await updateAllNDNodes(ctx);

    expect(updateSceneForDimensions).toHaveBeenCalledWith(dims, expect.anything(), undefined, {
      frameBudgetMs: undefined,
      ladderDepth: 6,
    });
    expect(prefetchSceneForDimensions).not.toHaveBeenCalled();
    const setValue = sceneDimsManager.setDimensionValue as ReturnType<typeof vi.fn>;
    expect(setValue).not.toHaveBeenCalled();

    // The settle pass re-notifies at the current position once the scrub is quiet,
    // through the first NON-displayed dimension (3 here), never a displayed axis.
    vi.advanceTimersByTime(SCRUB_SETTLE_MS + 1);
    expect(setValue).toHaveBeenCalledWith(3, 5);
  });

  it('a settle pass carries no directive (the listener runs while the settle flag is set)', async () => {
    let listener: (() => Promise<void>) | undefined;
    (sceneDimsManager.setDimensionValue as ReturnType<typeof vi.fn>).mockImplementation(() => {
      void listener?.();
    });
    const ctx = makeCtx({ getAnimationManager: () => makeIdleAnim('auto') });
    listener = () => updateAllNDNodes(ctx);

    await updateAllNDNodes(ctx);
    expect((updateSceneForDimensions as ReturnType<typeof vi.fn>).mock.calls[0][3]).toEqual({
      frameBudgetMs: undefined,
      ladderDepth: 'auto',
    });

    vi.advanceTimersByTime(SCRUB_SETTLE_MS + 1);
    expect((updateSceneForDimensions as ReturnType<typeof vi.fn>).mock.calls[1][3]).toEqual({
      frameBudgetMs: undefined,
      ladderDepth: undefined,
    });
  });

  it('a Fast (null) scrub detail pins nothing and schedules no settle pass', async () => {
    const ctx = makeCtx({ getAnimationManager: () => makeIdleAnim(null) });
    await updateAllNDNodes(ctx);
    expect(updateSceneForDimensions).toHaveBeenCalledWith(dims, expect.anything(), undefined, {
      frameBudgetMs: undefined,
      ladderDepth: undefined,
    });
    vi.advanceTimersByTime(SCRUB_SETTLE_MS + 1);
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
  });

  it('a re-scrub within the quiet period postpones the settle pass', async () => {
    const ctx = makeCtx({ getAnimationManager: () => makeIdleAnim(4) });
    await updateAllNDNodes(ctx);
    vi.advanceTimersByTime(SCRUB_SETTLE_MS - 50);
    await updateAllNDNodes(ctx);
    vi.advanceTimersByTime(SCRUB_SETTLE_MS - 50);
    expect(sceneDimsManager.setDimensionValue).not.toHaveBeenCalled();
    vi.advanceTimersByTime(100);
    expect(sceneDimsManager.setDimensionValue).toHaveBeenCalledTimes(1);
  });
});
