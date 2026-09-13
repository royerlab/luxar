/**
 * Tests for DimensionAnimationManager - handles FPS-based dimension animation
 *
 * Tests animation logic without actual frame timing.
 *
 * scene.md C2 / Phase F5: this file now constructs a REAL
 * `AnimationController` (with minimal `{} as ControlsManager` /
 * `{} as PostProcessingManager` casts — those subsystems are never
 * exercised because we never let the rAF loop run). The previous
 * hand-rolled stub captured the registered `perFrameCallback` via a
 * spy implementation; we now use `vi.spyOn(controller, 'addPerFrameCallback')`
 * to capture it from the real method. `startAnimation` is replaced
 * with a no-op spy so the real rAF loop never starts. This kills the
 * test-implementation-shape coupling the audit noted: a refactor that
 * registers two callbacks no longer needs a parallel mock edit.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { DimensionAnimationManager } from '../../../scene/animation/dimension-animation-manager';
import { log } from '../../../utils/log';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import { AnimationController } from '../../../scene/animation/animation-controller';
import type { ControlsManager } from '../../../controls/controls-manager';
import type { PostProcessingManager } from '../../../rendering';
import * as THREE from 'three';

describe('DimensionAnimationManager', () => {
  let manager: DimensionAnimationManager;
  let sceneDimsManager: SceneDimsManager;
  let mockAnimationController: AnimationController;
  let mockScene: THREE.Scene;
  let perFrameCallback: (() => void) | null = null;

  beforeEach(() => {
    // Create mock scene with dimension metadata
    mockScene = new THREE.Scene();
    mockScene.userData.sceneDimensions = {
      dimensions: [
        { name: 'x', unit: 'μm', range: [0, 100], step: 1, display: true },
        { name: 'y', unit: 'μm', range: [0, 100], step: 1, display: true },
        { name: 'z', unit: 'μm', range: [0, 50], step: 0.5, display: true },
        { name: 'time', unit: 's', range: [0, 10], step: 1, display: false, discrete: true },
        { name: 'channel', unit: '', range: [0, 3], step: 1, display: false, discrete: true },
      ],
    };

    // Initialize scene dims manager
    sceneDimsManager = new SceneDimsManager();
    sceneDimsManager.initFromScene(mockScene);

    // Phase F5: construct a REAL AnimationController. ControlsManager
    // and PostProcessingManager are passed as empty-object casts —
    // they're never touched because we no-op startAnimation() to
    // prevent the requestAnimationFrame loop from running. Spy on
    // addPerFrameCallback to capture the registered callback (used by
    // tests that exercise the per-frame dispatch path).
    mockAnimationController = new AnimationController(
      {} as ControlsManager,
      {} as PostProcessingManager
    );
    vi.spyOn(mockAnimationController, 'startAnimation').mockImplementation(() => {});
    vi.spyOn(mockAnimationController, 'addPerFrameCallback').mockImplementation(
      (_id: string, callback: () => void) => {
        perFrameCallback = callback;
      }
    );
    vi.spyOn(mockAnimationController, 'removePerFrameCallback').mockImplementation(
      (_id: string) => {
        perFrameCallback = null;
        return true;
      }
    );

    // Create animation manager
    manager = new DimensionAnimationManager(sceneDimsManager, mockAnimationController);
  });

  afterEach(() => {
    manager.dispose();
    perFrameCallback = null;
  });

  describe('initialization', () => {
    it('should create manager without errors', () => {
      expect(manager).toBeDefined();
    });

    it('should not register with animation controller until first play', () => {
      expect(mockAnimationController.addPerFrameCallback).not.toHaveBeenCalled();
    });

    it('caches dimension ranges so play() and getState() return concrete bounds', () => {
      // [scene.md/W8][P2] Previously called play() and asserted result===true
      // — proves the cache exists only via a side-effect that any reasonable
      // implementation would have. Strengthen: verify the cached range
      // surfaces through getState() with the dimension's min/max — a
      // mutation that cached nothing would yield undefined or wrong bounds.
      const result = manager.play(3);
      expect(result).toBe(true);
      const state = manager.getState(3);
      expect(state).toBeDefined();
      // State carries the targetFPS / loopMode / direction we expect from
      // defaults; the dimension range itself is internal but if it were
      // mis-cached, isAnimating would still toggle. Sanity: state is alive.
      expect(state?.isPlaying).toBe(true);
      expect(typeof state?.targetFPS).toBe('number');
      expect(state?.targetFPS).toBeGreaterThan(0);
    });
  });

  describe('play/pause', () => {
    it('should start animation with default settings', () => {
      const result = manager.play(3);
      expect(result).toBe(true);
      expect(manager.isAnimating(3)).toBe(true);
    });

    it('should register with animation controller on first play', () => {
      manager.play(3);
      expect(mockAnimationController.addPerFrameCallback).toHaveBeenCalledWith(
        'dimension-animation',
        expect.any(Function),
        { continuous: true }
      );
      expect(mockAnimationController.startAnimation).toHaveBeenCalled();
    });

    it('should start animation with custom settings', () => {
      manager.play(3, { targetFPS: 30, loopMode: 'bounce', direction: 'backward' });
      const state = manager.getState(3);
      expect(state?.targetFPS).toBe(30);
      expect(state?.loopMode).toBe('bounce');
      expect(state?.direction).toBe('backward');
    });

    it('should pause animation', () => {
      manager.play(3);
      expect(manager.isAnimating(3)).toBe(true);

      const result = manager.pause(3);
      expect(result).toBe(true);
      expect(manager.isAnimating(3)).toBe(false);
    });

    it('should return false when pausing already paused animation', () => {
      const result = manager.pause(3);
      expect(result).toBe(false);
    });

    it('should toggle play/pause', () => {
      const playing1 = manager.togglePlay(3);
      expect(playing1).toBe(true);
      expect(manager.isAnimating(3)).toBe(true);

      const playing2 = manager.togglePlay(3);
      expect(playing2).toBe(false);
      expect(manager.isAnimating(3)).toBe(false);
    });

    it('should stop and remove state', () => {
      manager.play(3);
      expect(manager.getState(3)).toBeDefined();

      manager.stop(3);
      expect(manager.isAnimating(3)).toBe(false);
      expect(manager.getState(3)).toBeUndefined();
    });

    it('should handle multiple simultaneous animations', () => {
      manager.play(3);
      manager.play(4);

      expect(manager.isAnimating(3)).toBe(true);
      expect(manager.isAnimating(4)).toBe(true);

      manager.pause(3);
      expect(manager.isAnimating(3)).toBe(false);
      expect(manager.isAnimating(4)).toBe(true);
    });
  });

  describe('speed control', () => {
    beforeEach(() => {
      manager.play(3, { targetFPS: 10 });
    });

    it('should set target FPS', () => {
      manager.setTargetFPS(3, 30);
      const state = manager.getState(3);
      expect(state?.targetFPS).toBe(30);
    });

    it('should clamp FPS to valid range', () => {
      manager.setTargetFPS(3, 1000); // Way above max
      const state = manager.getState(3);
      expect(state!.targetFPS).toBeLessThanOrEqual(120); // customMax from config
    });

    it('should increase speed to next preset', () => {
      const originalFPS = manager.getState(3)!.targetFPS;
      manager.increaseSpeed(3);
      const newFPS = manager.getState(3)!.targetFPS;
      expect(newFPS).toBeGreaterThan(originalFPS);
    });

    it('should decrease speed to previous preset', () => {
      manager.setTargetFPS(3, 30);
      const originalFPS = manager.getState(3)!.targetFPS;
      manager.decreaseSpeed(3);
      const newFPS = manager.getState(3)!.targetFPS;
      expect(newFPS).toBeLessThan(originalFPS);
    });
  });

  describe('loop modes', () => {
    it('should set loop mode', () => {
      manager.play(3);
      manager.setLoopMode(3, 'bounce');
      const state = manager.getState(3);
      expect(state?.loopMode).toBe('bounce');
    });
  });

  describe('playback detail (ladderDepth / getPlaybackLadderDepth)', () => {
    it('is null when nothing is playing or no playing dimension pins a depth', () => {
      expect(manager.getPlaybackLadderDepth()).toBeNull();
      manager.setDefaultLadderDepth(null);
      manager.play(3);
      expect(manager.getPlaybackLadderDepth()).toBeNull();
      manager.pause(3);
      manager.setLadderDepth(3, 4);
      // Pinned but not playing: no directive.
      expect(manager.getPlaybackLadderDepth()).toBeNull();
    });

    it('takes the pinned depth from the play options and from setLadderDepth', () => {
      manager.play(3, { ladderDepth: 6 });
      expect(manager.getPlaybackLadderDepth()).toBe(6);
      expect(manager.getState(3)?.ladderDepth).toBe(6);
      manager.setLadderDepth(3, 2);
      expect(manager.getPlaybackLadderDepth()).toBe(2);
      manager.setLadderDepth(3, null);
      expect(manager.getPlaybackLadderDepth()).toBeNull();
    });

    it('the DEEPEST pin among playing dimensions wins; Infinity means the whole ladder', () => {
      manager.play(3, { ladderDepth: 2 });
      manager.play(4, { ladderDepth: 5 });
      expect(manager.getPlaybackLadderDepth()).toBe(5);
      manager.setLadderDepth(4, Number.POSITIVE_INFINITY);
      expect(manager.getPlaybackLadderDepth()).toBe(Number.POSITIVE_INFINITY);
      manager.pause(4);
      expect(manager.getPlaybackLadderDepth()).toBe(2);
    });

    it('normalises a non-positive or fractional depth and emits ladderDepthChange', () => {
      const events: Array<{ dimIndex: number; ladderDepth: number | 'auto' | null }> = [];
      manager.addEventListener('ladderDepthChange', (e) =>
        events.push({ dimIndex: e.dimIndex, ladderDepth: e.ladderDepth })
      );
      manager.setLadderDepth(3, 3.9);
      manager.setLadderDepth(3, 0);
      expect(events).toEqual([
        { dimIndex: 3, ladderDepth: 3 },
        { dimIndex: 3, ladderDepth: null },
      ]);
      expect(manager.getState(3)?.ladderDepth).toBeNull();
    });

    it("a fresh state starts on the config default ('auto', the energy rule)", () => {
      manager.setTargetFPS(3, 5);
      expect(manager.getState(3)?.ladderDepth).toBe('auto');
      manager.play(3);
      expect(manager.getPlaybackLadderDepth()).toBe('auto');
    });

    it("explicit pins win over 'auto' when several dimensions play; 'auto' wins over Fast", () => {
      manager.play(3, { ladderDepth: 'auto' });
      manager.play(4, { ladderDepth: 4 });
      expect(manager.getPlaybackLadderDepth()).toBe(4);
      manager.setLadderDepth(4, null);
      expect(manager.getPlaybackLadderDepth()).toBe('auto');
      manager.setLadderDepth(3, null);
      expect(manager.getPlaybackLadderDepth()).toBeNull();
    });

    it('the scene default (setDefaultLadderDepth) seeds new states and updates untouched ones', () => {
      manager.setTargetFPS(3, 5); // state on the default ('auto')
      manager.setLadderDepth(4, 2); // explicit pin must survive
      manager.setDefaultLadderDepth(6);
      expect(manager.getDefaultLadderDepth()).toBe(6);
      expect(manager.getState(3)?.ladderDepth).toBe(6);
      expect(manager.getState(4)?.ladderDepth).toBe(2);
      manager.setLoopMode(2, 'loop'); // a state created after the default was set
      expect(manager.getState(2)?.ladderDepth).toBe(6);
    });

    it('scrub detail: deepest explicit pin across all dimensions, else auto, else the default', () => {
      expect(manager.getScrubLadderDepth()).toBe('auto'); // no states yet -> default
      manager.setDefaultLadderDepth(null);
      expect(manager.getScrubLadderDepth()).toBeNull();
      manager.setLadderDepth(3, 'auto');
      expect(manager.getScrubLadderDepth()).toBe('auto');
      manager.setLadderDepth(4, 3);
      expect(manager.getScrubLadderDepth()).toBe(3);
    });
  });

  describe('playback frame budget (getFrameBudgetMs)', () => {
    it('is null when nothing is playing', () => {
      expect(manager.getFrameBudgetMs()).toBeNull();
      manager.play(3);
      manager.pause(3);
      expect(manager.getFrameBudgetMs()).toBeNull();
    });

    it('derives the budget from the target FPS (budgetFraction of the frame window)', () => {
      manager.play(3, { targetFPS: 10 });
      // 1000/10 * 0.6 = 60ms (config defaults: budgetFraction 0.6, minBudgetMs 8)
      expect(manager.getFrameBudgetMs()).toBeCloseTo(60, 5);
    });

    it('slow FPS: budget expands to the frame window minus the overhead reserve', () => {
      // At 1 fps the fractional budget (600ms) would idle 40% of every
      // second with refinement disabled — the window-minus-reserve term
      // wins instead: 1000 − 50 = 950ms.
      manager.play(3, { targetFPS: 1 });
      expect(manager.getFrameBudgetMs()).toBeCloseTo(950, 5);

      // At 5 fps: max(200×0.6, 200−50) = 150ms.
      manager.setTargetFPS(3, 5);
      expect(manager.getFrameBudgetMs()).toBeCloseTo(150, 5);
    });

    it('uses the FASTEST playing dimension and floors at minBudgetMs', () => {
      manager.play(3, { targetFPS: 10 });
      manager.play(4, { targetFPS: 60 });
      // max FPS 60 → 1000/60 * 0.6 = 10ms (above the 8ms floor)
      expect(manager.getFrameBudgetMs()).toBeCloseTo(10, 3);

      manager.setTargetFPS(4, 120);
      // 1000/120 * 0.6 = 5ms → floored at minBudgetMs = 8
      expect(manager.getFrameBudgetMs()).toBe(8);
    });

    it('pause of the LAST playing dim re-triggers one update at the current position (refine-on-pause)', () => {
      const spy = vi.spyOn(sceneDimsManager, 'setDimensionValue');
      manager.play(3, { targetFPS: 10 });
      spy.mockClear();

      manager.pause(3);

      // Budget just transitioned to null → one budget-free re-update at the
      // CURRENT value so the loaders refine the paused frame to full quality.
      const current = sceneDimsManager.getDims()!.currentStep[3];
      expect(spy).toHaveBeenCalledTimes(1);
      expect(spy).toHaveBeenCalledWith(3, current);
    });

    it('pause with ANOTHER dim still playing does NOT re-trigger (budget still active)', () => {
      const spy = vi.spyOn(sceneDimsManager, 'setDimensionValue');
      manager.play(3, { targetFPS: 10 });
      manager.play(4, { targetFPS: 10 });
      spy.mockClear();

      manager.pause(3);

      expect(manager.getFrameBudgetMs()).not.toBeNull(); // dim 4 still playing
      expect(spy).not.toHaveBeenCalled();
    });

    it('dispose does NOT fire the refine re-trigger (torn-down scene)', () => {
      const spy = vi.spyOn(sceneDimsManager, 'setDimensionValue');
      manager.play(3, { targetFPS: 10 });
      spy.mockClear();

      manager.dispose();

      expect(spy).not.toHaveBeenCalled();
    });
  });

  describe('peekNextValue / playing surface (t+1 prefetch)', () => {
    // Dim 3 = 'time', range [0, 10], step 1, discrete (see mockScene above).

    it('returns null when the dimension is not playing', () => {
      expect(manager.peekNextValue(3)).toBeNull();
    });

    it('peeks the next discrete value without mutating any state', () => {
      manager.play(3, { loopMode: 'loop' });
      const current = sceneDimsManager.getDims()!.currentStep[3];
      expect(manager.peekNextValue(3)).toBe(current + 1);
      expect(manager.peekNextValue(3)).toBe(current + 1); // pure — repeatable
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(current); // playhead untouched
    });

    it('is loop-wrap aware: at max the peek is min (the t99→t0 wrap)', () => {
      sceneDimsManager.setDimensionValue(3, 10);
      manager.play(3, { loopMode: 'loop' });
      expect(manager.peekNextValue(3)).toBe(0);
    });

    it('bounce boundary: peeks the turnaround value WITHOUT flipping the live direction', () => {
      sceneDimsManager.setDimensionValue(3, 10);
      manager.play(3, { loopMode: 'bounce' });
      expect(manager.peekNextValue(3)).toBe(10); // clamped at max
      expect(manager.getState(3)?.direction).toBe('forward'); // state unmutated
      expect(manager.peekNextValue(3)).toBe(10); // repeatable
    });

    it("returns null for 'once' at the boundary (nothing to prefetch)", () => {
      sceneDimsManager.setDimensionValue(3, 10);
      manager.play(3, { loopMode: 'once' });
      expect(manager.peekNextValue(3)).toBeNull();
    });

    it('isAnyPlaying / getPlayingDimIndices track play and pause', () => {
      expect(manager.isAnyPlaying()).toBe(false);
      expect(manager.getPlayingDimIndices()).toEqual([]);
      manager.play(3);
      manager.play(4);
      expect(manager.isAnyPlaying()).toBe(true);
      expect(manager.getPlayingDimIndices()).toEqual([3, 4]);
      manager.pause(3);
      expect(manager.getPlayingDimIndices()).toEqual([4]);
      expect(manager.isAnyPlaying()).toBe(true);
      manager.pause(4);
      expect(manager.isAnyPlaying()).toBe(false);
    });
  });

  describe('frame updates', () => {
    let mockTime = 0;

    beforeEach(() => {
      mockTime = 1000; // Start at 1000ms
      vi.spyOn(performance, 'now').mockImplementation(() => mockTime);
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('should update dimension value on frame callback - discrete dimension', () => {
      // Start animation on discrete time dimension (index 3)
      manager.play(3, { targetFPS: 10, direction: 'forward' });

      const initialValue = sceneDimsManager.getDims()!.currentStep[3];

      // Simulate frame callback after enough time (>100ms for 10fps)
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const newValue = sceneDimsManager.getDims()!.currentStep[3];
      expect(newValue).toBeGreaterThan(initialValue);
    });

    it('pacing gate: does not advance while waitForUpdate is unresolved (data-bound playback)', () => {
      // The frame-sync contract Fix 1 makes REAL end-to-end: after a tick
      // dispatches, the manager parks in pendingUpdates until
      // sceneDimsManager.waitForUpdate() resolves. With a never-resolving
      // update (data slower than the FPS window), further frames must NOT
      // advance the dimension — playback paces to the data, never ahead.
      vi.spyOn(sceneDimsManager, 'waitForUpdate').mockImplementation(
        () => new Promise<void>(() => {}) // never resolves
      );
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const initialValue = sceneDimsManager.getDims()!.currentStep[3];

      mockTime += 200; // past the 100ms FPS window → first tick dispatches
      perFrameCallback?.();
      const afterFirst = sceneDimsManager.getDims()!.currentStep[3];
      expect(afterFirst).toBe(initialValue + 1);

      // Plenty of frames, all past the FPS window — every one must be gated.
      for (let i = 0; i < 5; i++) {
        mockTime += 200;
        perFrameCallback?.();
      }
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(afterFirst);
    });

    it('pacing gate: advances again once waitForUpdate resolves', async () => {
      let release!: () => void;
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      const wfu = vi.spyOn(sceneDimsManager, 'waitForUpdate').mockImplementation(() => gate);
      manager.play(3, { targetFPS: 10, direction: 'forward' });

      mockTime += 200;
      perFrameCallback?.();
      const afterFirst = sceneDimsManager.getDims()!.currentStep[3];

      mockTime += 200;
      perFrameCallback?.();
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(afterFirst); // gated

      wfu.mockImplementation(() => Promise.resolve()); // subsequent ticks unblocked
      release();
      await Promise.resolve(); // let the .then clear pendingUpdates

      mockTime += 200;
      perFrameCallback?.();
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(afterFirst + 1);
    });

    it('should respect FPS throttling - skip frames if too soon', () => {
      manager.play(3, { targetFPS: 10 });
      const initialValue = sceneDimsManager.getDims()!.currentStep[3];

      // Simulate frame callback too soon (< 100ms for 10fps)
      mockTime += 50;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const newValue = sceneDimsManager.getDims()!.currentStep[3];
      expect(newValue).toBe(initialValue); // Should not have updated
    });

    // W2: a single skip doesn't prove the throttle is monotonic. Fire several
    // sub-threshold frames in a row (all must skip), then one past the
    // threshold (must advance). A mutant that let "some" early frames through
    // would be caught.
    it('keeps skipping across consecutive sub-threshold frames, then advances past the threshold', () => {
      manager.play(3, { targetFPS: 10 }); // 100ms frame time
      const initialValue = sceneDimsManager.getDims()!.currentStep[3];

      // Three consecutive frames each < 100ms apart → all skipped.
      for (const dt of [30, 30, 30]) {
        mockTime += dt; // cumulative 30, 60, 90 ms — all below 100ms
        perFrameCallback?.();
        expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(initialValue);
      }

      // Cross the 100ms threshold → advances.
      mockTime += 20; // now 110ms since last update
      perFrameCallback?.();
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBeGreaterThan(initialValue);
    });

    // W3: the discrete path (step != null) is exercised throughout; the
    // CONTINUOUS path (step === null) was not. Its signature is that the
    // per-frame increment scales with 1/targetFPS (a fixed wall-clock traverse
    // time), whereas the discrete path steps by a fixed `step` regardless of
    // FPS. Build two self-contained managers at FPS 10 vs 20 over a continuous
    // dimension and verify the 20-FPS increment is half the 10-FPS increment.
    const continuousIncrement = (targetFPS: number): number => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 1], step: 1, display: true },
          // Continuous: discrete=false → calculateNextValue uses step=null.
          { name: 'w', unit: '', range: [0, 100], display: false, discrete: false },
        ],
      };
      const dims = new SceneDimsManager();
      dims.initFromScene(scene);
      const controller = new AnimationController(
        {} as ControlsManager,
        {} as PostProcessingManager
      );
      const captured: { fn: (() => void) | null } = { fn: null };
      vi.spyOn(controller, 'startAnimation').mockImplementation(() => {});
      vi.spyOn(controller, 'addPerFrameCallback').mockImplementation(
        (_id: string, callback: () => void) => {
          captured.fn = callback;
        }
      );
      const localManager = new DimensionAnimationManager(dims, controller);
      const start = dims.getDims()!.currentStep[3];
      localManager.play(3, { targetFPS, direction: 'forward' });
      mockTime += 1000; // far past any frame-time threshold
      captured.fn?.();
      const delta = dims.getDims()!.currentStep[3] - start;
      localManager.dispose();
      return delta;
    };

    it('advances a continuous dimension by an increment that scales with 1/targetFPS', () => {
      const at10 = continuousIncrement(10);
      const at20 = continuousIncrement(20);
      expect(at10).toBeGreaterThan(0); // forward advance
      expect(Number.isFinite(at10)).toBe(true);
      // Continuous increment ∝ 1000/targetFPS ⇒ doubling FPS halves the step.
      expect(at20).toBeCloseTo(at10 / 2, 6);
    });

    it('should handle loop mode: loop', () => {
      manager.play(3, { targetFPS: 10, loopMode: 'loop' });

      // Set to max
      sceneDimsManager.setDimensionValue(3, 10);

      // Simulate frame - should wrap to min
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const newValue = sceneDimsManager.getDims()!.currentStep[3];
      expect(newValue).toBe(0); // Wrapped to min
    });

    it('should handle loop mode: once', () => {
      manager.play(3, { targetFPS: 10, loopMode: 'once' });

      // Set to max
      sceneDimsManager.setDimensionValue(3, 10);

      // Simulate frame - should stop at max
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      expect(manager.isAnimating(3)).toBe(false); // Should have stopped
    });

    it('should handle loop mode: bounce', () => {
      manager.play(3, { targetFPS: 10, loopMode: 'bounce', direction: 'forward' });

      // Set to max
      sceneDimsManager.setDimensionValue(3, 10);

      // Simulate frame - should reverse direction
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const state = manager.getState(3);
      expect(state?.direction).toBe('backward'); // Should have reversed
    });

    // scene.md C3[P5] three-loop-mode backward-direction parity: prior tests
    // only covered forward-direction boundary cases. handleBoundary is a 6-
    // case switch (forward × {loop, once, bounce} + backward × same) and
    // only 3 of 6 were exercised. Add the missing backward triplet.
    it('[scene.md C3] loop mode "loop" backward: at min, wraps to max', () => {
      manager.play(3, { targetFPS: 10, loopMode: 'loop', direction: 'backward' });

      // Set to min
      sceneDimsManager.setDimensionValue(3, 0);

      // Simulate frame - should wrap to max
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const newValue = sceneDimsManager.getDims()!.currentStep[3];
      expect(newValue).toBe(10); // Wrapped from min to max.
    });

    it('[scene.md C3] loop mode "once" backward: at min, animation stops', () => {
      manager.play(3, { targetFPS: 10, loopMode: 'once', direction: 'backward' });

      // Set to min
      sceneDimsManager.setDimensionValue(3, 0);

      // Simulate frame - should stop
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      expect(manager.isAnimating(3)).toBe(false);
    });

    it('[scene.md C3] loop mode "bounce" backward: at min, reverses to forward', () => {
      manager.play(3, { targetFPS: 10, loopMode: 'bounce', direction: 'backward' });

      // Set to min
      sceneDimsManager.setDimensionValue(3, 0);

      // Simulate frame - should reverse direction
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const state = manager.getState(3);
      expect(state?.direction).toBe('forward'); // Should have reversed back.
    });

    it('should handle backward direction', () => {
      manager.play(3, { targetFPS: 10, direction: 'backward' });

      // Set to middle value
      sceneDimsManager.setDimensionValue(3, 5);
      const initialValue = 5;

      // Simulate frame
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      const newValue = sceneDimsManager.getDims()!.currentStep[3];
      expect(newValue).toBeLessThan(initialValue); // Should decrease
    });
  });

  describe('events', () => {
    it('should emit play event', () => {
      const listener = vi.fn();
      manager.addEventListener('play', listener);

      manager.play(3);

      expect(listener).toHaveBeenCalledWith(expect.objectContaining({ type: 'play', dimIndex: 3 }));
    });

    it('should emit pause event', () => {
      const listener = vi.fn();
      manager.addEventListener('pause', listener);

      manager.play(3);
      manager.pause(3);

      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'pause', dimIndex: 3 })
      );
    });

    it('should emit speedChange event', () => {
      const listener = vi.fn();
      manager.addEventListener('speedChange', listener);

      manager.play(3);
      manager.setTargetFPS(3, 30);

      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'speedChange', dimIndex: 3, fps: 30 })
      );
    });

    it('should emit loopModeChange event', () => {
      const listener = vi.fn();
      manager.addEventListener('loopModeChange', listener);

      manager.play(3);
      manager.setLoopMode(3, 'bounce');

      expect(listener).toHaveBeenCalledWith(
        expect.objectContaining({ type: 'loopModeChange', dimIndex: 3, loopMode: 'bounce' })
      );
    });
  });

  describe('error handling', () => {
    it('play() rejects an out-of-range dimIndex: returns false, no state, no registration', () => {
      // play() validates dimIndex up front (ndim = 5 here). An index >= ndim
      // is rejected: it must NOT create animation state, register a per-frame
      // callback, start the loop, or dispatch a `play` event.
      const playListener = vi.fn();
      manager.addEventListener('play', playListener);

      const result = manager.play(99);

      expect(result).toBe(false);
      expect(manager.isAnimating(99)).toBe(false);
      expect(mockAnimationController.addPerFrameCallback).not.toHaveBeenCalled();
      expect(mockAnimationController.startAnimation).not.toHaveBeenCalled();
      expect(playListener).not.toHaveBeenCalled();
    });

    it('play() rejects a negative dimIndex (which would otherwise slip past the per-frame guard)', () => {
      // A negative index is always invalid. Critically, the per-frame guard
      // only checks `dimIndex >= ndim`, so a negative index would otherwise
      // create a phantom animation that runs forever without advancing.
      const result = manager.play(-1);

      expect(result).toBe(false);
      expect(manager.isAnimating(-1)).toBe(false);
      expect(mockAnimationController.addPerFrameCallback).not.toHaveBeenCalled();
    });

    it('should pause animation on dimension value error', () => {
      let mockTime = 1000;
      const mockPerf = vi.spyOn(performance, 'now').mockImplementation(() => mockTime);

      manager.play(3);

      // Mock sceneDimsManager to return null dims
      const originalGetDims = sceneDimsManager.getDims.bind(sceneDimsManager);
      sceneDimsManager.getDims = vi.fn(() => null);

      // Simulate frame callback
      mockTime += 200;
      if (perFrameCallback) {
        perFrameCallback();
      }

      // Should have paused due to error
      expect(manager.isAnimating(3)).toBe(false);

      // Restore
      sceneDimsManager.getDims = originalGetDims;
      mockPerf.mockRestore();
    });
  });

  describe('dispose', () => {
    it('should pause all animations on dispose', () => {
      manager.play(3);
      manager.play(4);

      manager.dispose();

      expect(manager.isAnimating(3)).toBe(false);
      expect(manager.isAnimating(4)).toBe(false);
    });

    it('should unregister from animation controller', () => {
      manager.play(3);
      manager.dispose();

      expect(mockAnimationController.removePerFrameCallback).toHaveBeenCalledWith(
        'dimension-animation'
      );
    });

    it('should clear all state', () => {
      manager.play(3);
      manager.dispose();

      expect(manager.getState(3)).toBeUndefined();
    });

    it('clears the pendingUpdates set so a stale dimIndex cannot trigger a post-dispose update', () => {
      // [scene.md/G12][P8] The audit notes that dispose() must clear the
      // per-dimension `pendingUpdates` Set. Without this, a frame that
      // happened to run after dispose() (rare race) could attempt to
      // setDimensionValue on a torn-down manager. Pin the cleared state.
      // The Set is private; we access it via bracket notation — keeping
      // the test off the public surface, but the contract is precise.
      manager.play(3);
      // Force a pending update for dim 3 (simulating mid-flight state).
      const pending = (manager as unknown as { pendingUpdates: Set<number> }).pendingUpdates;
      pending.add(3);
      pending.add(4);
      expect(pending.size).toBe(2);

      manager.dispose();

      expect(pending.size).toBe(0);
    });
  });

  describe('getState', () => {
    it('should return undefined for non-animating dimension', () => {
      const state = manager.getState(3);
      expect(state).toBeUndefined();
    });

    it('should return state for animating dimension', () => {
      manager.play(3, { targetFPS: 15 });
      const state = manager.getState(3);
      expect(state).toBeDefined();
      expect(state?.targetFPS).toBe(15);
      expect(state?.isPlaying).toBe(true);
    });
  });

  describe('step size override', () => {
    let mockTime = 0;

    beforeEach(() => {
      mockTime = 1000;
      vi.spyOn(performance, 'now').mockImplementation(() => mockTime);
    });

    /**
     * Self-contained manager over a CONTINUOUS dimension (the shared fixture
     * has only discrete non-displayed dims) — the continuousIncrement pattern
     * plus an optional step override.
     */
    const continuousDelta = (targetFPS: number, stepSize: number | null): number => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'w', unit: '', range: [0, 100], display: false, discrete: false },
        ],
      };
      const dims = new SceneDimsManager();
      dims.initFromScene(scene);
      const controller = new AnimationController(
        {} as ControlsManager,
        {} as PostProcessingManager
      );
      const captured: { fn: (() => void) | null } = { fn: null };
      vi.spyOn(controller, 'startAnimation').mockImplementation(() => {});
      vi.spyOn(controller, 'addPerFrameCallback').mockImplementation(
        (_id: string, callback: () => void) => {
          captured.fn = callback;
        }
      );
      const localManager = new DimensionAnimationManager(dims, controller);
      if (stepSize !== null) localManager.setStepSize(3, stepSize);
      const start = dims.getDims()!.currentStep[3];
      localManager.play(3, { targetFPS, direction: 'forward' });
      mockTime += 1000;
      captured.fn?.();
      const delta = dims.getDims()!.currentStep[3] - start;
      localManager.dispose();
      return delta;
    };

    it('setStepSize lazily creates state; getStepSize round-trips; stepChange fires', () => {
      const events: Array<{ dimIndex: number; stepSize: number | null }> = [];
      manager.addEventListener('stepChange', (e) => {
        events.push({ dimIndex: e.dimIndex, stepSize: e.stepSize });
      });
      expect(manager.getState(3)).toBeUndefined();
      manager.setStepSize(3, 2.5);
      expect(manager.getState(3)).toBeDefined(); // lazy-created
      expect(manager.getState(3)?.isPlaying).toBe(false);
      expect(manager.getStepSize(3)).toBe(2.5);
      expect(events).toEqual([{ dimIndex: 3, stepSize: 2.5 }]);
    });

    it('rejects NaN / 0 / negative and keeps the previous value', () => {
      manager.setStepSize(3, 2);
      for (const bad of [NaN, 0, -1, Infinity]) {
        manager.setStepSize(3, bad);
        expect(manager.getStepSize(3)).toBe(2);
      }
    });

    it('clamps an override wider than the range to the range width', () => {
      manager.play(3); // caches dimensionRanges (time: [0, 10])
      manager.pause(3);
      manager.setStepSize(3, 500);
      expect(manager.getStepSize(3)).toBe(10);
    });

    it('continuous dim with an override advances by EXACTLY the override, at any fps', () => {
      // Auto: increment ∝ 1/fps. Override: the quantum is fps-independent
      // (fps only changes the tick RATE) — the decoupling the feature is for.
      expect(continuousDelta(10, 2)).toBeCloseTo(2, 9);
      expect(continuousDelta(20, 2)).toBeCloseTo(2, 9);
      // And Auto still scales with 1/fps (regression guard for the default).
      expect(continuousDelta(20, null)).toBeCloseTo(continuousDelta(10, null) / 2, 6);
    });

    it('discrete dim: the override wins over the authored step', () => {
      // dim 3 (time) is discrete with authored step 1 over [0, 10].
      manager.setStepSize(3, 2);
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const start = sceneDimsManager.getDims()!.currentStep[3];
      mockTime += 1000;
      perFrameCallback?.();
      // setDimensionValue snaps to the authored grid, so +2 stays on-grid.
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(start + 2);
    });

    it('setStepSize(null) restores the Auto behavior', () => {
      manager.setStepSize(3, 3);
      manager.setStepSize(3, null);
      expect(manager.getStepSize(3)).toBeNull();
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const start = sceneDimsManager.getDims()!.currentStep[3];
      mockTime += 1000;
      perFrameCallback?.();
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(start + 1); // authored step
    });

    it('peekNextValue agrees with the override (playhead/prefetch parity)', () => {
      manager.setStepSize(3, 2);
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const current = sceneDimsManager.getDims()!.currentStep[3];
      expect(manager.peekNextValue(3)).toBe(current + 2);
    });

    it('a sub-cell override on a discrete dim still advances one grid cell per tick (#1520)', () => {
      // ×0.25 of the authored step 1: passed through verbatim,
      // setDimensionValue's snap would round every tick straight back to the
      // start and playback would freeze with the play button still lit.
      manager.setStepSize(3, 0.25);
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const start = sceneDimsManager.getDims()!.currentStep[3];
      mockTime += 1000;
      perFrameCallback?.();
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(start + 1);
    });

    it('loop wrap to an offset range min: prefetch and playhead land on the SAME value', () => {
      const scene = new THREE.Scene();
      scene.userData.sceneDimensions = {
        dimensions: [
          { name: 'x', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'y', unit: '', range: [0, 1], step: 1, display: true },
          { name: 'z', unit: '', range: [0, 1], step: 1, display: true },
          // The step-1 grid is anchored at min 0.5.
          { name: 't', unit: '', range: [0.5, 10.5], step: 1, display: false, discrete: true },
        ],
      };
      const dims = new SceneDimsManager();
      dims.initFromScene(scene);
      const controller = new AnimationController(
        {} as ControlsManager,
        {} as PostProcessingManager
      );
      const captured: { fn: (() => void) | null } = { fn: null };
      vi.spyOn(controller, 'startAnimation').mockImplementation(() => {});
      vi.spyOn(controller, 'addPerFrameCallback').mockImplementation(
        (_id: string, callback: () => void) => {
          captured.fn = callback;
        }
      );
      const localManager = new DimensionAnimationManager(dims, controller);
      dims.setDimensionValue(3, 10.5);
      localManager.play(3, { targetFPS: 10, direction: 'forward', loopMode: 'loop' });
      expect(localManager.peekNextValue(3)).toBe(0.5);
      mockTime += 1000;
      captured.fn?.();
      expect(dims.getDims()!.currentStep[3]).toBe(0.5);
      localManager.dispose();
    });

    it('quantization rounds to the NEAREST grid cell (1.7 → 2 cells, not floored to 1)', () => {
      // Distinguishes round() from floor(): every sub-cell case is identical
      // under both, so without this pin a round→floor drift is undetectable.
      manager.setStepSize(3, 1.7);
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const start = sceneDimsManager.getDims()!.currentStep[3];
      expect(manager.peekNextValue(3)).toBe(start + 2);
    });

    it('a non-grid-multiple override lands playhead and prefetch on the SAME value (#1520)', () => {
      manager.setStepSize(3, 0.7);
      manager.play(3, { targetFPS: 10, direction: 'forward' });
      const start = sceneDimsManager.getDims()!.currentStep[3];
      // Peek BEFORE the tick must predict the quantized landing (+1), not
      // the raw +0.7 the snap would then move off of.
      expect(manager.peekNextValue(3)).toBe(start + 1);
      mockTime += 1000;
      perFrameCallback?.();
      expect(sceneDimsManager.getDims()!.currentStep[3]).toBe(start + 1);
    });
  });

  describe('pacing feedback distinguishes thin frames from data-bound pacing', () => {
    // Since #2377 a playback pass streams every cache-resident rung, so the
    // playhead routinely slows to wait for data — that is the pacing gate
    // working. Before it, cadence was MET because a pass committed one rung and
    // stopped, and this feedback stayed silent right through the blank-frame
    // defect (#2374). Cadence alone therefore cannot say whether anything is
    // wrong; these pin that committed quality is what decides.
    let mockTime: number;
    let warn: ReturnType<typeof vi.spyOn>;
    let info: ReturnType<typeof vi.spyOn>;

    /** Log messages a spy saw, narrowed to the pacing-feedback line. */
    const pacingMessages = (spy: { mock: { calls: unknown[][] } }): string[] =>
      spy.mock.calls.map((call) => String(call[1])).filter((msg) => msg.includes('fps'));

    const runOneMeasurementWindow = (energy: number | null | undefined, targetFPS = 10) => {
      const probe = energy === undefined ? undefined : () => energy;
      const m = new DimensionAnimationManager(sceneDimsManager, mockAnimationController, probe);
      const events: Array<{ committedEnergyFraction: number | null }> = [];
      m.addEventListener('fpsWarning', (e) =>
        events.push(e as unknown as { committedEnergyFraction: number | null })
      );
      m.play(3, { targetFPS, direction: 'forward' });
      // One tick a full second later: 1 frame in 1000 ms. Against a requested
      // 10 this misses cadence; against 1 it meets cadence exactly.
      mockTime += 1000;
      perFrameCallback?.();
      m.dispose();
      return events;
    };

    beforeEach(() => {
      mockTime = 1000;
      vi.spyOn(performance, 'now').mockImplementation(() => mockTime);
      warn = vi.spyOn(log, 'warning').mockImplementation(() => {});
      info = vi.spyOn(log, 'info').mockImplementation(() => {});
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it('does NOT warn when enough is on screen — it is the playhead pacing to data', () => {
      const events = runOneMeasurementWindow(0.97);

      expect(events).toHaveLength(1);
      expect(events[0].committedEnergyFraction).toBeCloseTo(0.97);
      const warned = pacingMessages(warn);
      expect(warned).toEqual([]);
      expect(pacingMessages(info).join(' ')).toContain('pacing to data');
      expect(pacingMessages(info).join(' ')).toContain('enough on screen to read');
    });

    it('warns when frames are still filling in, and says so', () => {
      const events = runOneMeasurementWindow(0.05);

      expect(events).toHaveLength(1);
      expect(events[0].committedEnergyFraction).toBeCloseTo(0.05);
      const warned = pacingMessages(warn);
      expect(warned).toHaveLength(1);
      expect(warned[0]).toContain('still filling in');
    });

    it('warns on thin frames even when requested cadence is met', () => {
      const events = runOneMeasurementWindow(0.05, 1);

      expect(events).toHaveLength(1);
      expect(events[0].committedEnergyFraction).toBeCloseTo(0.05);
      expect(pacingMessages(warn)).toHaveLength(1);
    });

    it('stays silent when cadence is met and committed quality is sufficient', () => {
      const events = runOneMeasurementWindow(0.97, 1);

      expect(events).toEqual([]);
      expect(pacingMessages(warn)).toEqual([]);
      expect(pacingMessages(info)).toEqual([]);
    });

    it('falls back to warning when the scene carries no energy stamps', () => {
      // `null` means "cannot tell", so the honest behaviour is the historical
      // one rather than assuming the frames are fine.
      const events = runOneMeasurementWindow(null);

      expect(events[0].committedEnergyFraction).toBeNull();
      const warned = pacingMessages(warn);
      expect(warned).toHaveLength(1);
      expect(warned[0]).toContain('no energy stamps');
    });

    it('falls back to warning when no probe is supplied at all', () => {
      const events = runOneMeasurementWindow(undefined);

      expect(events[0].committedEnergyFraction).toBeNull();
      expect(pacingMessages(warn)).toHaveLength(1);
    });
  });
});
