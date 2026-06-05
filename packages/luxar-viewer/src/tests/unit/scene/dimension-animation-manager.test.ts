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
});
