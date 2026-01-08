/**
 * Tests for DimensionAnimationManager - handles FPS-based dimension animation
 *
 * Tests animation logic without actual frame timing
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { DimensionAnimationManager } from '../../../scene/dimension-animation-manager';
import { SceneDimsManager } from '../../../scene/scene-dims-manager';
import { AnimationController } from '../../../scene/animation-controller';
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

    // Create mock animation controller
    mockAnimationController = {
      addPerFrameCallback: vi.fn((_id: string, callback: () => void) => {
        perFrameCallback = callback;
      }),
      removePerFrameCallback: vi.fn((_id: string) => {
        perFrameCallback = null;
        return true;
      }),
      startAnimation: vi.fn(),
    } as any;

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

    it('should cache dimension ranges on creation', () => {
      // Test by playing an animation - should not crash
      const result = manager.play(3);
      expect(result).toBe(true);
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
        expect.any(Function)
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
    it('should handle invalid dimension index gracefully', () => {
      const result = manager.play(99); // Invalid index
      // Should not crash, but might not start (depending on implementation)
      expect(result).toBeDefined();
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
