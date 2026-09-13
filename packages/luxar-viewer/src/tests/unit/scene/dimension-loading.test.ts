/**
 * Unit tests for scene-owned nD dimension loading orchestration.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: vi.fn(),
  },
}));

vi.mock('../../../data', () => ({
  updateSceneForDimensions: vi.fn().mockResolvedValue(undefined),
  prefetchSceneForDimensions: vi.fn(),
  releasePrefetchResources: vi.fn(),
}));

import { updateAllNDNodes, type DimensionLoadingContext } from '../../../scene/dimension-loading';
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
