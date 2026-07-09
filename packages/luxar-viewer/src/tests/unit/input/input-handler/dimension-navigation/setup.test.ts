/**
 * Unit tests for the dimension-navigation lifecycle bodies.
 *
 * input.md G5 fix: setup.ts was untested. Critical for the
 * "context-manager tests (context switching, registration/
 * unregistration symmetry)" claim — clearDimensionUI must
 * symmetrically tear down what initDimensionSliders builds.
 *
 * Strategy: mock the two trust-boundary collaborators that take a
 * real THREE.js scene (sceneDimsManager + the loader's
 * updateSceneForDimensions). Drive each setup function with a fake
 * ctx and assert observable state transitions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock sceneDimsManager — the singleton's real implementation walks
// a THREE.Scene tree.
vi.mock('../../../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    initFromScene: vi.fn(),
    getDims: vi.fn(),
    getDimensionRanges: vi.fn(),
    getDimensionNames: vi.fn(),
    getDimensionUnits: vi.fn(),
    hasNonDisplayedDimensions: vi.fn(),
    setDimensionValue: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    reset: vi.fn(),
  },
}));

// Mock the data loader — exercises a full scene graph in real life.
vi.mock('../../../../../data', () => ({
  updateSceneForDimensions: vi.fn().mockResolvedValue(undefined),
  prefetchSceneForDimensions: vi.fn(),
  releasePrefetchResources: vi.fn(),
}));

// Stub the animation manager class so init-animation tests don't
// reach into the real DimensionAnimationManager (which constructs
// timers + listeners).
vi.mock('../../../../../scene/animation/dimension-animation-manager', () => ({
  DimensionAnimationManager: vi.fn().mockImplementation(() => ({
    dispose: vi.fn(),
    // updateAllNDNodes reads the playback frame budget per update; the real
    // manager returns null when nothing is playing. The t+1 prefetch reads
    // the playing set / next-value peek — default: nothing playing.
    getFrameBudgetMs: vi.fn(() => null),
    isAnyPlaying: vi.fn(() => false),
    getPlayingDimIndices: vi.fn(() => []),
    peekNextValue: vi.fn(() => null),
  })),
}));

// AnimationShortcuts.register registers bindings — stub so the test
// observes the lifecycle, not the register-all internals (those are
// covered separately in key-bindings/register-all.test.ts).
vi.mock('../../../../../input/input-handler/key-bindings/animation-shortcuts', () => ({
  AnimationShortcuts: vi.fn().mockImplementation(() => ({
    register: vi.fn(),
  })),
}));

import {
  clearDimensionUI,
  updateAllNDNodes,
  initAnimationManager,
  initDimensionSliders,
  type DimNavSetupCtx,
} from '../../../../../input/input-handler/dimension-navigation/setup';
import { sceneDimsManager } from '../../../../../scene/scene-dims-manager';
import {
  updateSceneForDimensions,
  prefetchSceneForDimensions,
  releasePrefetchResources,
} from '../../../../../data';
import { DimensionAnimationManager } from '../../../../../scene/animation/dimension-animation-manager';
import { AnimationShortcuts } from '../../../../../input/input-handler/key-bindings/animation-shortcuts';
import type { DimensionSliders } from '../../../../../ui/dimension-sliders';

function makeCtx(overrides: Partial<DimNavSetupCtx> = {}): DimNavSetupCtx {
  let selectedDimension = 0;
  let animManager: DimensionAnimationManager | undefined;
  let sliders: DimensionSliders | undefined;
  let listener: (() => Promise<void>) | undefined;

  const panelCoordinator = {
    setDimensionSliders: vi.fn(),
  } as never;

  const sceneManager = {
    scene: {},
    controls: {},
    camera: {},
  } as never;

  return {
    sceneManager,
    animationController: { startAnimation: vi.fn() } as never,
    dimensionSlidersFactory: undefined,
    contextManager: { registerBinding: vi.fn() } as never,
    panelCoordinator,
    recordingPanel: undefined,
    getSelectedDimension: () => selectedDimension,
    setSelectedDimension: (v) => {
      selectedDimension = v;
    },
    getAnimationManager: () => animManager,
    setAnimationManager: (m) => {
      animManager = m;
    },
    getDimensionSliders: () => sliders,
    setDimensionSliders: (s) => {
      sliders = s;
    },
    getSceneDimsListener: () => listener,
    setSceneDimsListener: (l) => {
      listener = l;
    },
    ...overrides,
  };
}

beforeEach(() => {
  (sceneDimsManager.initFromScene as ReturnType<typeof vi.fn>).mockReset();
  (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReset();
  (sceneDimsManager.getDimensionRanges as ReturnType<typeof vi.fn>).mockReset();
  (sceneDimsManager.removeListener as ReturnType<typeof vi.fn>).mockReset();
  (sceneDimsManager.addListener as ReturnType<typeof vi.fn>).mockReset();
  (sceneDimsManager.reset as ReturnType<typeof vi.fn>).mockReset();
  (updateSceneForDimensions as ReturnType<typeof vi.fn>).mockReset().mockResolvedValue(undefined);
  (DimensionAnimationManager as unknown as ReturnType<typeof vi.fn>).mockClear();
  (AnimationShortcuts as unknown as ReturnType<typeof vi.fn>).mockClear();
});

describe('clearDimensionUI', () => {
  it('disposes existing sliders and clears the ref + coordinator', () => {
    const disposeSliders = vi.fn();
    const ctx = makeCtx();
    ctx.setDimensionSliders({ dispose: disposeSliders } as never);

    clearDimensionUI(ctx);

    expect(disposeSliders).toHaveBeenCalledTimes(1);
    expect(ctx.getDimensionSliders()).toBeUndefined();
    expect(ctx.panelCoordinator.setDimensionSliders).toHaveBeenLastCalledWith(undefined);
  });

  it('disposes existing animation manager and clears the ref', () => {
    const disposeAM = vi.fn();
    const ctx = makeCtx();
    ctx.setAnimationManager({ dispose: disposeAM } as never);

    clearDimensionUI(ctx);

    expect(disposeAM).toHaveBeenCalledTimes(1);
    expect(ctx.getAnimationManager()).toBeUndefined();
  });

  it('removes a registered scene-dims listener and clears its slot', () => {
    const ctx = makeCtx();
    const fakeListener = vi.fn(async () => {});
    ctx.setSceneDimsListener(fakeListener);

    clearDimensionUI(ctx);

    expect(sceneDimsManager.removeListener).toHaveBeenCalledWith(fakeListener);
    expect(ctx.getSceneDimsListener()).toBeUndefined();
  });

  it('always resets sceneDimsManager and selectedDimension regardless of prior state', () => {
    const ctx = makeCtx();
    ctx.setSelectedDimension(7);

    clearDimensionUI(ctx);

    expect(sceneDimsManager.reset).toHaveBeenCalledTimes(1);
    expect(ctx.getSelectedDimension()).toBe(0);
  });

  it('removes the listener BEFORE the sceneDimsManager.reset call (no stale-state observation)', () => {
    // The order is asserted by recording the sequence of method calls
    // on the sceneDimsManager singleton.
    const calls: string[] = [];
    (sceneDimsManager.removeListener as ReturnType<typeof vi.fn>).mockImplementation(() => {
      calls.push('removeListener');
    });
    (sceneDimsManager.reset as ReturnType<typeof vi.fn>).mockImplementation(() => {
      calls.push('reset');
    });
    const ctx = makeCtx();
    ctx.setSceneDimsListener(vi.fn(async () => {}));

    clearDimensionUI(ctx);

    expect(calls).toEqual(['removeListener', 'reset']);
  });
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
    const ctx = makeCtx({
      animationController: { startAnimation } as never,
    });
    await updateAllNDNodes(ctx);
    // 3rd arg: loaderId (undefined); 4th: playback frame-budget opts (no
    // budget outside animation playback → { frameBudgetMs: undefined }).
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

    function makePlayingAnim(over: Record<string, unknown> = {}) {
      return {
        dispose: vi.fn(),
        getFrameBudgetMs: vi.fn(() => 60),
        isAnyPlaying: vi.fn(() => true),
        getPlayingDimIndices: vi.fn(() => [3]),
        peekNextValue: vi.fn(() => 6),
        ...over,
      } as never;
    }

    beforeEach(() => {
      (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue(dims);
      (prefetchSceneForDimensions as ReturnType<typeof vi.fn>).mockClear();
      (releasePrefetchResources as ReturnType<typeof vi.fn>).mockClear();
    });

    it('while playing: prefetches the PEEKED next dims with the frame budget', async () => {
      const ctx = makeCtx();
      ctx.setAnimationManager(makePlayingAnim());
      await updateAllNDNodes(ctx);

      expect(prefetchSceneForDimensions).toHaveBeenCalledTimes(1);
      const [predictedDims, _scene, loaderId, opts] = (
        prefetchSceneForDimensions as ReturnType<typeof vi.fn>
      ).mock.calls[0];
      expect(predictedDims.currentStep).toEqual([0, 0, 0, 6]); // peeked t+1
      expect(dims.currentStep).toEqual([0, 0, 0, 5]); // REAL dims untouched
      expect(loaderId).toBeUndefined();
      expect(opts).toEqual({ budgetMs: 60 });
      expect(releasePrefetchResources).not.toHaveBeenCalled();
    });

    it('not playing: releases prefetch resources and does not prefetch', async () => {
      const ctx = makeCtx();
      ctx.setAnimationManager(
        makePlayingAnim({ isAnyPlaying: vi.fn(() => false), getFrameBudgetMs: vi.fn(() => null) })
      );
      await updateAllNDNodes(ctx);
      expect(prefetchSceneForDimensions).not.toHaveBeenCalled();
      expect(releasePrefetchResources).toHaveBeenCalledTimes(1);
    });

    it("peek returned null for every playing dim ('once' at boundary): no prefetch fired", async () => {
      const ctx = makeCtx();
      ctx.setAnimationManager(makePlayingAnim({ peekNextValue: vi.fn(() => null) }));
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

describe('initAnimationManager', () => {
  it('constructs the animation manager exactly once (idempotent)', () => {
    const ctx = makeCtx();
    initAnimationManager(ctx);
    initAnimationManager(ctx); // second call should not re-construct
    expect(DimensionAnimationManager).toHaveBeenCalledTimes(1);
  });

  it('registers AnimationShortcuts after constructing the manager', () => {
    const ctx = makeCtx();
    initAnimationManager(ctx);
    expect(AnimationShortcuts).toHaveBeenCalledTimes(1);
  });

  it('does NOT re-register shortcuts on the second call', () => {
    const ctx = makeCtx();
    initAnimationManager(ctx);
    initAnimationManager(ctx);
    expect(AnimationShortcuts).toHaveBeenCalledTimes(1);
  });
});

describe('initDimensionSliders', () => {
  it('returns early when sceneDimsManager.initFromScene returns false', () => {
    (sceneDimsManager.initFromScene as ReturnType<typeof vi.fn>).mockReturnValue(false);
    const ctx = makeCtx();
    initDimensionSliders(ctx);
    expect(ctx.getDimensionSliders()).toBeUndefined();
    expect(ctx.getAnimationManager()).toBeUndefined();
    expect(sceneDimsManager.addListener).not.toHaveBeenCalled();
  });

  it('returns early when dims is null even after successful init', () => {
    (sceneDimsManager.initFromScene as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue(null);
    (sceneDimsManager.getDimensionRanges as ReturnType<typeof vi.fn>).mockReturnValue([[0, 10]]);
    const ctx = makeCtx();
    initDimensionSliders(ctx);
    expect(sceneDimsManager.addListener).not.toHaveBeenCalled();
  });

  it('registers exactly one sceneDimsManager listener even when invoked twice', () => {
    (sceneDimsManager.initFromScene as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue({
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 0],
    });
    (sceneDimsManager.getDimensionRanges as ReturnType<typeof vi.fn>).mockReturnValue([
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 10],
    ]);
    (sceneDimsManager.hasNonDisplayedDimensions as ReturnType<typeof vi.fn>).mockReturnValue(true);
    const ctx = makeCtx();
    initDimensionSliders(ctx);
    initDimensionSliders(ctx);
    // Two init calls: each replaces the prior listener, so the
    // active listener count should be 1 (not 2). The previous
    // listener should have been removed.
    expect(sceneDimsManager.removeListener).toHaveBeenCalled();
    expect(sceneDimsManager.addListener).toHaveBeenCalledTimes(2);
  });

  // ───────────────────────────────────────────────────────────────────
  // HIGH-15 regression: the initial `updateAllNDNodes(ctx)` call at
  // the bottom of initDimensionSliders is intentionally not awaited
  // (initDimensionSliders is synchronous), but its rejection must be
  // logged via `.catch()` instead of silently disappearing into an
  // unhandled-rejection.
  // ───────────────────────────────────────────────────────────────────
  it('HIGH-15: initial updateAllNDNodes failure is logged, not swallowed', async () => {
    (sceneDimsManager.initFromScene as ReturnType<typeof vi.fn>).mockReturnValue(true);
    (sceneDimsManager.getDims as ReturnType<typeof vi.fn>).mockReturnValue({
      ndim: 4,
      displayed: [0, 1, 2],
      currentStep: [0, 0, 0, 0],
    });
    (sceneDimsManager.getDimensionRanges as ReturnType<typeof vi.fn>).mockReturnValue([
      [0, 1],
      [0, 1],
      [0, 1],
      [0, 10],
    ]);
    (sceneDimsManager.hasNonDisplayedDimensions as ReturnType<typeof vi.fn>).mockReturnValue(true);

    // Force updateSceneForDimensions to reject — this is what
    // updateAllNDNodes awaits inside.
    const boom = new Error('boom');
    (updateSceneForDimensions as ReturnType<typeof vi.fn>).mockRejectedValueOnce(boom);

    // Spy on console.error — log.error funnels through it.
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const ctx = makeCtx();
    // initDimensionSliders is sync; the rejected updateAllNDNodes
    // promise resolves on the next microtask.
    initDimensionSliders(ctx);

    // Flush microtasks so the .catch handler runs.
    await Promise.resolve();
    await Promise.resolve();

    // The .catch handler must have logged the error.
    const errorCall = errorSpy.mock.calls.find((call) =>
      String(call[0] ?? '').includes('updateAllNDNodes failed')
    );
    expect(errorCall).toBeDefined();
    expect(errorCall?.some((arg) => arg === boom)).toBe(true);

    errorSpy.mockRestore();
  });
});
