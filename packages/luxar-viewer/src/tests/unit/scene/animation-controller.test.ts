/**
 * Tests for AnimationController - manages the main rendering loop
 *
 * These tests verify callback management, lifecycle control, idle timeout
 * behavior, and proper resource cleanup. External dependencies (ControlsManager,
 * PostProcessingManager, PerformanceMonitor) are mocked while testing real
 * AnimationController logic.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

// Mock external dependencies only - NOT the module under test
vi.mock('../../../controls/controls-manager', () => ({
  ControlsManager: vi.fn(),
}));

vi.mock('../../../rendering/post-processing/post-processing-manager', () => ({
  PostProcessingManager: vi.fn(),
}));

vi.mock('../../../ui/performance-monitor', () => {
  return {
    PerformanceMonitor: class MockPerformanceMonitor {
      begin = vi.fn();
      end = vi.fn();
      dispose = vi.fn();
    },
  };
});

vi.mock('../../../rendering/adaptive-dpr-manager', () => ({
  AdaptiveDPRManager: vi.fn(),
}));

import { AnimationController } from '../../../scene/animation/animation-controller';

describe('AnimationController', () => {
  let controller: AnimationController;
  let mockControls: any;
  let mockPostProcessing: any;
  let mockRAF: ReturnType<typeof vi.fn>;
  let mockCAF: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();

    // Create mock controls manager
    mockControls = {
      update: vi.fn(),
      getAutoRotate: vi.fn().mockReturnValue(false),
    };

    // Create mock post-processing manager
    mockPostProcessing = {
      render: vi.fn(),
      needsContinuousAnimation: vi.fn().mockReturnValue(false),
    };

    // Mock requestAnimationFrame and cancelAnimationFrame
    let rafId = 0;
    mockRAF = vi.fn().mockImplementation(() => ++rafId);
    mockCAF = vi.fn();
    vi.stubGlobal('requestAnimationFrame', mockRAF);
    vi.stubGlobal('cancelAnimationFrame', mockCAF);

    // Mock performance.now for adaptive DPR
    vi.spyOn(performance, 'now').mockReturnValue(1000);

    // Create controller with mock dependencies
    controller = new AnimationController(mockControls, mockPostProcessing);
  });

  afterEach(() => {
    controller.dispose();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe('per-frame callbacks', () => {
    it('should add a per-frame callback', () => {
      const callback = vi.fn();
      controller.addPerFrameCallback('test', callback);

      expect(controller.hasPerFrameCallback('test')).toBe(true);
    });

    it('should remove a per-frame callback', () => {
      const callback = vi.fn();
      controller.addPerFrameCallback('test', callback);

      const removed = controller.removePerFrameCallback('test');

      expect(removed).toBe(true);
      expect(controller.hasPerFrameCallback('test')).toBe(false);
    });

    it('should return false when removing a non-existent callback', () => {
      const removed = controller.removePerFrameCallback('nonexistent');

      expect(removed).toBe(false);
    });

    it('should return false for hasPerFrameCallback with non-existent ID', () => {
      expect(controller.hasPerFrameCallback('nonexistent')).toBe(false);
    });

    it('should execute registered callbacks during animation', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();
      controller.addPerFrameCallback('cb1', callback1);
      controller.addPerFrameCallback('cb2', callback2);

      controller.startAnimation();

      // The animate() is called synchronously from startAnimation
      expect(callback1).toHaveBeenCalledTimes(1);
      expect(callback2).toHaveBeenCalledTimes(1);
    });

    it('should not execute removed callbacks', () => {
      const callback = vi.fn();
      controller.addPerFrameCallback('test', callback);
      controller.removePerFrameCallback('test');

      controller.startAnimation();

      expect(callback).not.toHaveBeenCalled();
    });

    it('should support overwriting a callback with the same ID', () => {
      const callback1 = vi.fn();
      const callback2 = vi.fn();
      controller.addPerFrameCallback('test', callback1);
      controller.addPerFrameCallback('test', callback2);

      controller.startAnimation();

      expect(callback1).not.toHaveBeenCalled();
      expect(callback2).toHaveBeenCalled();
    });
  });

  describe('setAdaptiveDPRManager', () => {
    it('should set the adaptive DPR manager', () => {
      const mockDPRManager = {
        recordFrame: vi.fn(),
      };

      controller.setAdaptiveDPRManager(mockDPRManager as any);

      // Verify it's used during animation
      controller.startAnimation();

      expect(mockDPRManager.recordFrame).toHaveBeenCalledWith(expect.any(Number));
    });

    it('should stop calling recordFrame after setting manager to null', () => {
      const mockDPRManager = {
        recordFrame: vi.fn(),
      };

      // Set manager and verify it's called during animation
      controller.setAdaptiveDPRManager(mockDPRManager as any);
      controller.startAnimation();
      expect(mockDPRManager.recordFrame).toHaveBeenCalled();

      // Clear the call history, set to null, then restart animation
      mockDPRManager.recordFrame.mockClear();
      controller.stopAnimation();
      controller.setAdaptiveDPRManager(null);
      controller.startAnimation();

      // Should not call recordFrame after null
      expect(mockDPRManager.recordFrame).not.toHaveBeenCalled();
    });
  });

  describe('lifecycle', () => {
    it('should set isActive to true when startAnimation is called', () => {
      expect(controller.isActive).toBe(false);

      controller.startAnimation();

      expect(controller.isActive).toBe(true);
    });

    it('should set isActive to false when stopAnimation is called', () => {
      controller.startAnimation();
      controller.stopAnimation();

      expect(controller.isActive).toBe(false);
    });

    it('should be idempotent on double start (does not start duplicate loops)', () => {
      controller.startAnimation();
      controller.startAnimation();

      // animate() is called once in the first startAnimation
      // The second startAnimation should not trigger another animate()
      // since isAnimating is already true
      // It only calls RAF once from the first animate() call
      expect(mockRAF).toHaveBeenCalledTimes(1);
    });

    it('should call controls.update during animation', () => {
      controller.startAnimation();

      expect(mockControls.update).toHaveBeenCalledTimes(1);
    });

    it('should call postProcessing.render during animation', () => {
      controller.startAnimation();

      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
    });

    it('schedules next frame via requestAnimationFrame and the callback advances the loop', () => {
      // [scene.md/W9][P2] Previously only asserted rAF called with a fn,
      // but never invoked the callback to check it advances the loop.
      // A mutation that scheduled rAF with a no-op would survive.
      controller.startAnimation();

      expect(mockRAF).toHaveBeenCalledTimes(1);
      const rafCallback = mockRAF.mock.calls[0][0] as FrameRequestCallback;
      expect(typeof rafCallback).toBe('function');

      // Invoking the scheduled callback should run another animate() pass:
      // - controls.update fires again
      // - postProcessing.render fires again
      // - rAF is re-scheduled
      mockRAF.mockClear();
      mockControls.update.mockClear();
      mockPostProcessing.render.mockClear();
      rafCallback(performance.now());

      expect(mockControls.update).toHaveBeenCalledTimes(1);
      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
      expect(mockRAF).toHaveBeenCalledTimes(1);
    });

    // when WebGL context is lost, animation loop must
    // skip postProcessing.render() to avoid issuing draw calls
    // against a dead context. controls.update() and per-frame
    // callbacks still run.
    it('skips postProcessing.render() while context is lost', () => {
      controller.setContextLostPredicate(() => true);
      controller.startAnimation();

      expect(mockControls.update).toHaveBeenCalledTimes(1);
      expect(mockPostProcessing.render).not.toHaveBeenCalled();
    });

    it('renders normally when context-lost predicate returns false', () => {
      controller.setContextLostPredicate(() => false);
      controller.startAnimation();

      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
    });

    it('renders normally when no context-lost predicate is set (default)', () => {
      // Predicate is null by default — backward-compat for tests/embed
      // contexts that never lose the context.
      controller.startAnimation();
      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
    });

    it('should cancel animation frame on stop', () => {
      controller.startAnimation();
      controller.stopAnimation();

      expect(mockCAF).toHaveBeenCalled();
    });

    it('clears idle timeout on stop — no late idle-fire reactivates renderer or controls', () => {
      // [scene.md/W11][P2] Previously advanced time 10000ms and only checked
      // "no crash". A mutation that left the idle timeout alive (e.g. did
      // not clearTimeout) would still pass. Pin observable state: after
      // stop+advance, controls.update / postProcessing.render must NOT
      // have been called again, and controller must remain inactive.
      controller.startAnimation();
      const updateCallsBefore = mockControls.update.mock.calls.length;
      const renderCallsBefore = mockPostProcessing.render.mock.calls.length;

      controller.stopAnimation();
      expect(controller.isActive).toBe(false);

      // If timeout was alive, an idle-fire (or other timer-driven re-trigger)
      // would re-invoke animate() and bump these counters.
      vi.advanceTimersByTime(10000);
      expect(mockControls.update.mock.calls.length).toBe(updateCallsBefore);
      expect(mockPostProcessing.render.mock.calls.length).toBe(renderCallsBefore);
      expect(controller.isActive).toBe(false);
    });
  });

  describe('pause / idle-restore / resume hooks', () => {
    function makeDPRManagerStub() {
      return {
        recordFrame: vi.fn(),
        notifyPaused: vi.fn(),
        notifyResumed: vi.fn(),
        isActive: vi.fn().mockReturnValue(true),
        prepareIdleFrame: vi.fn().mockReturnValue(true),
      };
    }

    it('stopAnimation notifies the manager of the pause', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);
      controller.startAnimation();
      controller.stopAnimation();
      expect(manager.notifyPaused).toHaveBeenCalled();
    });

    it('stopAnimation stays safe with a bare {recordFrame} manager mock (optional chaining)', () => {
      controller.setAdaptiveDPRManager({ recordFrame: vi.fn() } as never);
      controller.startAnimation();
      expect(() => controller.stopAnimation()).not.toThrow();
    });

    it('idle timeout restores native DPR and renders exactly one resting frame', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);
      controller.startAnimation();
      mockPostProcessing.render.mockClear();

      vi.advanceTimersByTime(2000); // idle-pause fires

      expect(controller.isActive).toBe(false);
      expect(manager.prepareIdleFrame).toHaveBeenCalledTimes(1);
      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
    });

    it('does not render a resting frame when prepareIdleFrame reports no change', () => {
      const manager = makeDPRManagerStub();
      manager.prepareIdleFrame.mockReturnValue(false); // already at native
      controller.setAdaptiveDPRManager(manager as never);
      controller.startAnimation();
      mockPostProcessing.render.mockClear();

      vi.advanceTimersByTime(2000);

      expect(manager.prepareIdleFrame).toHaveBeenCalledTimes(1);
      expect(mockPostProcessing.render).not.toHaveBeenCalled();
    });

    it('skips the idle restore while the idle-restore predicate forbids it (recording)', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);
      controller.setIdleRestorePredicate(() => false);
      controller.startAnimation();

      vi.advanceTimersByTime(2000);

      expect(controller.isActive).toBe(false); // loop still pauses
      expect(manager.prepareIdleFrame).not.toHaveBeenCalled();
    });

    it('skips the idle restore while the context is lost', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);
      controller.setContextLostPredicate(() => true);
      controller.startAnimation();

      vi.advanceTimersByTime(2000);

      expect(manager.prepareIdleFrame).not.toHaveBeenCalled();
    });

    it('idle-pauses cleanly with NO manager set (predicates untouched)', () => {
      controller.startAnimation();
      expect(() => vi.advanceTimersByTime(2000)).not.toThrow();
      expect(controller.isActive).toBe(false);
    });

    it('notifyResumed fires only on the stopped→running edge, not on every interaction poke', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);

      controller.startAnimation(); // edge: stopped → running
      controller.startAnimation(); // already running — just resets the idle timer
      controller.startAnimation();
      expect(manager.notifyResumed).toHaveBeenCalledTimes(1);

      controller.stopAnimation();
      controller.startAnimation(); // new edge
      expect(manager.notifyResumed).toHaveBeenCalledTimes(2);
    });

    it('does not record frames while the context is lost', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);
      controller.setContextLostPredicate(() => true);

      controller.startAnimation();

      expect(manager.recordFrame).not.toHaveBeenCalled();
    });
  });

  describe('idle timeout', () => {
    it('should stop animation after idle timeout when no continuous effects', () => {
      mockControls.getAutoRotate.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();
      expect(controller.isActive).toBe(true);

      // Advance time past idle timeout (config.animation.idleTimeoutMs = 2000)
      vi.advanceTimersByTime(2000);

      expect(controller.isActive).toBe(false);
    });

    it('should continue animation when autoRotate is enabled', () => {
      mockControls.getAutoRotate.mockReturnValue(true);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();

      // Advance time past idle timeout
      vi.advanceTimersByTime(2000);

      // Should still be active due to autoRotate
      expect(controller.isActive).toBe(true);
    });

    it('should continue animation when postProcessing needs continuous', () => {
      mockControls.getAutoRotate.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(true);

      controller.startAnimation();

      // Advance time past idle timeout
      vi.advanceTimersByTime(2000);

      // Should still be active due to continuous effects
      expect(controller.isActive).toBe(true);
    });

    it('should reset idle timeout on subsequent startAnimation calls', () => {
      mockControls.getAutoRotate.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();

      // Advance halfway through idle timeout
      vi.advanceTimersByTime(1000);
      expect(controller.isActive).toBe(true);

      // Trigger startAnimation again (simulating user interaction)
      controller.startAnimation();

      // Advance another 1500ms (total 2500ms since start, 1500ms since reset)
      vi.advanceTimersByTime(1500);

      // Should still be active because timeout was reset
      expect(controller.isActive).toBe(true);

      // Advance remaining time past the reset timeout
      vi.advanceTimersByTime(500);

      // Now should be idle
      expect(controller.isActive).toBe(false);
    });

    it('should reschedule check when continuous effects are active at timeout', () => {
      // Start with continuous effects active
      mockControls.getAutoRotate.mockReturnValue(true);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();

      // First idle timeout fires - continuous effects active, reschedules
      vi.advanceTimersByTime(2000);
      expect(controller.isActive).toBe(true);

      // Now disable continuous effects
      mockControls.getAutoRotate.mockReturnValue(false);

      // Second idle timeout fires - no continuous effects, should stop
      vi.advanceTimersByTime(2000);
      expect(controller.isActive).toBe(false);
    });
  });

  describe('dispose', () => {
    it('should stop animation on dispose', () => {
      controller.startAnimation();
      expect(controller.isActive).toBe(true);

      controller.dispose();

      expect(controller.isActive).toBe(false);
    });

    it('should cancel animation frame on dispose', () => {
      controller.startAnimation();
      controller.dispose();

      expect(mockCAF).toHaveBeenCalled();
    });

    it('should not crash on double dispose', () => {
      controller.startAnimation();
      controller.dispose();

      expect(() => controller.dispose()).not.toThrow();
    });
  });

  describe('getters', () => {
    it('should return animation state via isActive', () => {
      expect(controller.isActive).toBe(false);

      controller.startAnimation();
      expect(controller.isActive).toBe(true);

      controller.stopAnimation();
      expect(controller.isActive).toBe(false);
    });
  });

  // [scene.md/O1][P10] Moved out of `describe('getters', ...)` — frame-start/end
  // event-bus emission is not a getter; it's an integration with the cross-layer
  // event bus that fires from inside the rAF loop body.
  describe('event bus integration', () => {
    it('emits frame-start and frame-end on the event bus per frame', async () => {
      const { eventBus } = await import('../../../utils/cross-layer/event-bus');
      const startListener = vi.fn();
      const endListener = vi.fn();
      const offStart = eventBus.on('frame-start', startListener);
      const offEnd = eventBus.on('frame-end', endListener);

      try {
        controller.startAnimation();
        // The mock requestAnimationFrame should have fired the loop body
        // at least once already (see makeMockRAF in this file's setup).
        expect(startListener).toHaveBeenCalled();
        expect(endListener).toHaveBeenCalled();
      } finally {
        offStart();
        offEnd();
        controller.stopAnimation();
      }
    });
  });
});
