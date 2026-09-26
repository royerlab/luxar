/**
 * Tests for AnimationController - manages the main rendering loop
 *
 * These tests verify callback management, lifecycle control, idle timeout
 * behavior, and proper resource cleanup. External dependencies (ControlsManager,
 * PostProcessingManager, PerformanceMonitor) are mocked while testing real
 * AnimationController logic.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { MockInstance } from 'vitest';

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
import { config } from '../../../config';
import { log } from '../../../utils/log';
import { eventBus } from '../../../utils/cross-layer/event-bus';

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
      isAutoRotateActive: vi.fn().mockReturnValue(false),
      isAutoDollyActive: vi.fn().mockReturnValue(false),
      isGestureActive: vi.fn().mockReturnValue(false),
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
    // Restore BEFORE uninstalling the fake timers: the frame-pacing block
    // spies on globalThis.setTimeout, and restoring after `useRealTimers()`
    // would write the fake-timer implementation back onto a global whose
    // fake-timer machinery has already been torn down.
    vi.restoreAllMocks();
    vi.useRealTimers();
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

  describe('frame phases', () => {
    const order: string[] = [];
    const push = (name: string) => () => {
      order.push(name);
    };

    beforeEach(() => {
      order.length = 0;
    });

    it('runs camera → view → pre-render → ui whatever the registration order', () => {
      controller.addPerFrameCallback('ui', push('ui'), { phase: 'ui' });
      controller.addPerFrameCallback('pre', push('pre-render'), { phase: 'pre-render' });
      controller.addPerFrameCallback('view', push('view'), { phase: 'view' });
      controller.addPerFrameCallback('cam', push('camera'), { phase: 'camera' });

      controller.tick();

      expect(order).toEqual(['camera', 'view', 'pre-render', 'ui']);
    });

    it('defaults to the view phase and keeps registration order within a phase', () => {
      controller.addPerFrameCallback('v1', push('v1'));
      controller.addPerFrameCallback('ui', push('ui'), { phase: 'ui' });
      controller.addPerFrameCallback('v2', push('v2'), { phase: 'view' });
      controller.addPerFrameCallback('v3', push('v3'));

      controller.tick();

      expect(order).toEqual(['v1', 'v2', 'v3', 'ui']);
    });

    it('a camera writer registered after the view callbacks still runs before them', () => {
      // The flight case: clipping/LOD register at init, the flight when it starts.
      controller.addPerFrameCallback('dynamic-clipping', push('clipping'));
      controller.addPerFrameCallback('lod', push('lod'));
      controller.startAnimation();
      order.length = 0;

      controller.addPerFrameCallback('flight', push('flight'), { phase: 'camera' });
      controller.tick();

      expect(order).toEqual(['flight', 'clipping', 'lod']);
    });

    it('re-registering in the same phase keeps the position; a new phase moves it', () => {
      controller.addPerFrameCallback('a', push('a'));
      controller.addPerFrameCallback('b', push('b'));
      controller.addPerFrameCallback('a', push('a2'));
      controller.tick();
      expect(order).toEqual(['a2', 'b']);

      order.length = 0;
      controller.addPerFrameCallback('b', push('b-camera'), { phase: 'camera' });
      controller.tick();
      expect(order).toEqual(['b-camera', 'a2']);
      expect(controller.hasPerFrameCallback('b')).toBe(true);
    });

    it('removes a callback from whichever phase holds it', () => {
      controller.addPerFrameCallback('cam', push('camera'), { phase: 'camera' });
      controller.addPerFrameCallback('ui', push('ui'), { phase: 'ui' });

      expect(controller.removePerFrameCallback('cam')).toBe(true);
      expect(controller.removePerFrameCallback('cam')).toBe(false);
      expect(controller.hasPerFrameCallback('cam')).toBe(false);
      controller.tick();

      expect(order).toEqual(['ui']);
    });

    it('keeps the mid-frame semantics: a later-phase add runs this frame, a pending removal is skipped', () => {
      controller.addPerFrameCallback(
        'cam',
        () => {
          order.push('camera');
          controller.addPerFrameCallback('late-ui', push('late-ui'), { phase: 'ui' });
          controller.removePerFrameCallback('doomed');
        },
        { phase: 'camera' }
      );
      controller.addPerFrameCallback('doomed', push('doomed'));

      controller.tick();

      expect(order).toEqual(['camera', 'late-ui']);
    });

    it('a continuous callback in any phase keeps the loop awake', () => {
      controller.addPerFrameCallback('cam', () => {}, { continuous: true, phase: 'camera' });
      controller.startAnimation();

      vi.advanceTimersByTime(config.animation.idleTimeoutMs + 10);

      expect(controller.isActive).toBe(true);
    });
  });

  describe('per-frame callback isolation', () => {
    let logError: MockInstance;
    const events: string[] = [];
    let offs: Array<() => void> = [];

    beforeEach(() => {
      logError = vi.spyOn(log, 'error').mockImplementation(() => {});
      events.length = 0;
      offs = [
        eventBus.on('frame-start', () => events.push('frame-start')),
        eventBus.on('frame-end', () => events.push('frame-end')),
      ];
    });

    afterEach(() => {
      for (const off of offs) off();
    });

    it('a throwing callback does not skip the callbacks after it, or the render', () => {
      const before = vi.fn();
      const after = vi.fn();
      controller.addPerFrameCallback('before', before);
      controller.addPerFrameCallback('broken', () => {
        throw new Error('broken subsystem');
      });
      controller.addPerFrameCallback('after', after);

      controller.tick();

      expect(before).toHaveBeenCalledTimes(1);
      expect(after).toHaveBeenCalledTimes(1);
      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
      expect(events).toEqual(['frame-start', 'frame-end']);
    });

    it('logs a failing callback once, not once per frame', () => {
      controller.addPerFrameCallback('broken', () => {
        throw new Error('broken subsystem');
      });

      controller.tick();
      controller.tick();
      controller.tick();

      expect(logError).toHaveBeenCalledTimes(1);
      expect(String(logError.mock.calls[0][1])).toContain("'broken'");
    });

    it('a re-registered callback reports its own first failure', () => {
      const broken = (): void => {
        throw new Error('broken subsystem');
      };
      controller.addPerFrameCallback('broken', broken);
      controller.tick();
      controller.removePerFrameCallback('broken');
      controller.addPerFrameCallback('broken', broken);
      controller.tick();

      expect(logError).toHaveBeenCalledTimes(2);
    });

    it('frame-end still pairs frame-start when the render throws', () => {
      mockPostProcessing.render.mockImplementation(() => {
        throw new Error('render failed');
      });

      expect(() => controller.tick()).toThrow('render failed');
      expect(events).toEqual(['frame-start', 'frame-end']);
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

    // The render-skip predicate is set during an offline capture, which
    // renders its own pipeline pass per frame. The loop must keep ticking
    // (per-frame callbacks drive the depth-sort scheduler and the LOD
    // group selector, which have to follow the orbiting camera) while
    // issuing no draw call of its own.
    it('skips postProcessing.render() but still ticks controls + callbacks when render-skip is on', async () => {
      const { eventBus } = await import('../../../utils/cross-layer/event-bus');
      const callback = vi.fn();
      const startListener = vi.fn();
      const endListener = vi.fn();
      const dprManager = { recordFrame: vi.fn() };
      const offStart = eventBus.on('frame-start', startListener);
      const offEnd = eventBus.on('frame-end', endListener);
      controller.addPerFrameCallback('cb', callback);
      controller.setAdaptiveDPRManager(dprManager as never);
      controller.setRenderSkipPredicate(() => true);

      try {
        controller.startAnimation();

        expect(mockControls.update).toHaveBeenCalledTimes(1);
        expect(callback).toHaveBeenCalledTimes(1);
        expect(mockPostProcessing.render).not.toHaveBeenCalled();
        // A render-skipped frame did no GPU work of its own, so feeding
        // its duration to adaptive DPR would drive bogus scale-ups and
        // falsely settle U-shape probes — same reasoning as context-lost.
        expect(dprManager.recordFrame).not.toHaveBeenCalled();
        // A skipped frame still has to CLOSE its measurement: the early
        // return emits frame-end, so the performance monitor never sees an
        // unpaired frame-start (one per skipped frame, for a whole capture).
        expect(startListener).toHaveBeenCalledTimes(1);
        expect(endListener).toHaveBeenCalledTimes(1);
      } finally {
        offStart();
        offEnd();
        controller.stopAnimation();
      }
    });

    it('renders normally when the render-skip predicate returns false', () => {
      controller.setRenderSkipPredicate(() => false);
      controller.startAnimation();

      expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);
    });

    it('renders normally when the render-skip predicate is cleared to null', () => {
      controller.setRenderSkipPredicate(() => true);
      controller.setRenderSkipPredicate(null);
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

  describe('idle pause vs. a live pointer gesture', () => {
    it('does not idle-pause while a gesture is in progress, and pauses once it ends', () => {
      // A button or finger held still for longer than the idle timeout, then
      // dragged: the drag's deltas are applied by update(), which only a live
      // loop calls. Pausing mid-gesture froze the camera for the whole drag.
      mockControls.isGestureActive.mockReturnValue(true);
      controller.startAnimation();
      vi.advanceTimersByTime(2000); // idle timeout elapses with the pointer still down
      expect(controller.isActive).toBe(true);
      vi.advanceTimersByTime(2000); // …and keeps re-arming while it stays down
      expect(controller.isActive).toBe(true);

      mockControls.isGestureActive.mockReturnValue(false);
      vi.advanceTimersByTime(2000);
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

    // The idle restore is the loop's OTHER render call site, so the
    // render-skip predicate has to reach it too — and BEFORE
    // prepareIdleFrame(), which resizes (clearing the canvas) on its way
    // to returning true. Rendering here mid-capture would paint a
    // native-DPR frame through the capture scrim; resizing and then not
    // rendering would leave the canvas blank with nothing to repaint it.
    it('skips the idle restore — resize included — while the render-skip predicate is on', () => {
      const manager = makeDPRManagerStub();
      controller.setAdaptiveDPRManager(manager as never);
      controller.setRenderSkipPredicate(() => true);
      controller.startAnimation();
      mockPostProcessing.render.mockClear();

      vi.advanceTimersByTime(2000);

      expect(controller.isActive).toBe(false); // loop still pauses
      expect(manager.prepareIdleFrame).not.toHaveBeenCalled();
      expect(mockPostProcessing.render).not.toHaveBeenCalled();
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
      mockControls.isAutoRotateActive.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();
      expect(controller.isActive).toBe(true);

      // Advance time past idle timeout (config.animation.idleTimeoutMs = 2000)
      vi.advanceTimersByTime(2000);

      expect(controller.isActive).toBe(false);
    });

    it('should continue animation when autoRotate is enabled', () => {
      mockControls.isAutoRotateActive.mockReturnValue(true);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();

      // Advance time past idle timeout
      vi.advanceTimersByTime(2000);

      // Should still be active due to autoRotate
      expect(controller.isActive).toBe(true);
    });

    it('should continue animation when postProcessing needs continuous', () => {
      mockControls.isAutoRotateActive.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(true);

      controller.startAnimation();

      // Advance time past idle timeout
      vi.advanceTimersByTime(2000);

      // Should still be active due to continuous effects
      expect(controller.isActive).toBe(true);
    });

    it('should reset idle timeout on subsequent startAnimation calls', () => {
      mockControls.isAutoRotateActive.mockReturnValue(false);
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

    // The `hasContinuousCallbacks` clause of shouldContinueAnimating() is
    // the entire reason both recording keep-alives work: startAnimation()
    // arms the idle timer, nothing in a capture loop re-arms it, so a
    // registered `continuous` callback is the only thing keeping the loop
    // alive past the first two seconds of a capture.
    it('should continue animation when a continuous per-frame callback is registered', () => {
      mockControls.isAutoRotateActive.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.addPerFrameCallback('keepalive', vi.fn(), { continuous: true });
      controller.startAnimation();

      vi.advanceTimersByTime(2000);

      expect(controller.isActive).toBe(true);
    });

    // Negative control: an on-demand callback (the default) must NOT hold
    // the loop open, or dynamic-clipping and the scale bar would defeat
    // the whole idle-pause power optimization.
    it('should NOT keep the loop alive for a non-continuous per-frame callback', () => {
      mockControls.isAutoRotateActive.mockReturnValue(false);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.addPerFrameCallback('on-demand', vi.fn());
      controller.startAnimation();

      vi.advanceTimersByTime(2000);

      expect(controller.isActive).toBe(false);
    });

    // Registration alone is inert on a stopped loop — this is exactly why
    // both recording paths must call startAnimation() explicitly.
    it('registering a callback while stopped neither fires it nor starts the loop', () => {
      const callback = vi.fn();

      controller.addPerFrameCallback('keepalive', callback, { continuous: true });

      expect(controller.isActive).toBe(false);
      expect(callback).not.toHaveBeenCalled();
      expect(mockControls.update).not.toHaveBeenCalled();
    });

    it('should reschedule check when continuous effects are active at timeout', () => {
      // Start with continuous effects active
      mockControls.isAutoRotateActive.mockReturnValue(true);
      mockPostProcessing.needsContinuousAnimation.mockReturnValue(false);

      controller.startAnimation();

      // First idle timeout fires - continuous effects active, reschedules
      vi.advanceTimersByTime(2000);
      expect(controller.isActive).toBe(true);

      // Now disable continuous effects
      mockControls.isAutoRotateActive.mockReturnValue(false);

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

  // Frame pacing (#1724): a scene whose frames cost ~1s used to hold the main
  // thread at a 100% duty cycle of long tasks, starving worker-reply delivery
  // and the CDP evaluate channel into a livelock. Pacing inserts a bounded
  // cooldown once a STREAK of consecutive frames has been pathologically slow
  // — the wedge is sustained, and a lone outlier must stay on the fast path.
  // These tests drive performance.now() from a variable — never real time —
  // and read the scheduling decision off the rAF / setTimeout spies.
  describe('frame pacing', () => {
    let now: number;
    let setTimeoutSpy: MockInstance<typeof globalThis.setTimeout>;

    beforeEach(() => {
      now = 1000;
      vi.spyOn(performance, 'now').mockImplementation(() => now);
      setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout');
    });

    /**
     * Cooldown delays passed to setTimeout, excluding the idle-pause timer.
     *
     * A paced frame arms TWO timers — the zero-delay hop and then the real
     * cooldown — so a paced expectation is `[0, cooldown]`. The hop is what
     * makes the gap real: the frame's expensive work runs after the rAF
     * callback returns but inside the same main-thread task, so a timer armed
     * at the frame's start would already be overdue and insert nothing.
     */
    function pacingDelays(): number[] {
      return setTimeoutSpy.mock.calls
        .map((call) => call[1] as number)
        .filter((delay) => delay !== config.animation.idleTimeoutMs);
    }

    /**
     * Run the zero-delay hop and report the WHOLE two-stage delay sequence a
     * paced frame arms: `[]` for a fast frame, `[0, cooldown]` for a paced
     * one. The second stage does not exist until the hop fires — that is the
     * mechanism, so it has to be stepped through rather than assumed.
     */
    function pacedDelaysAfterHop(): number[] {
      vi.advanceTimersByTime(0);
      return pacingDelays();
    }

    /**
     * Run out a paced frame's two-stage timer chain: the zero-delay hop
     * (which arms the real cooldown), then the cooldown itself.
     */
    function advancePacing(cooldownMs: number): void {
      vi.advanceTimersByTime(0);
      vi.advanceTimersByTime(cooldownMs);
    }

    /** The pacing timeout callbacks, in the order they were armed. */
    function pacingCallbacks(): Array<() => void> {
      return setTimeoutSpy.mock.calls
        .filter((call) => (call[1] as number) !== config.animation.idleTimeoutMs)
        .map((call) => call[0] as () => void);
    }

    /**
     * Advance the fake clock by `elapsedMs` and run the frame that the most
     * recent requestAnimationFrame scheduled. Clears the rAF spy first so
     * each assertion sees only the scheduling done by that frame.
     */
    function runNextFrame(elapsedMs: number): void {
      const calls = mockRAF.mock.calls;
      const callback = calls[calls.length - 1][0] as FrameRequestCallback;
      now += elapsedMs;
      mockRAF.mockClear();
      callback(now);
    }

    // Guards every `pacingDelays()` assertion in this block: the helper tells
    // pacing timers from the idle timer by DELAY VALUE alone, so a config
    // change that made a cooldown (or the hop) collide with idleTimeoutMs
    // would silently filter the pacing timers out and make every expectation
    // below vacuously `[]`. Cooldowns are bounded by maxCooldownMs, so one
    // strict inequality plus the hop covers the whole reachable set.
    it('the pacing delays are distinguishable from the idle-timeout delay', () => {
      expect(config.animation.idleTimeoutMs).toBeGreaterThan(config.animation.pacing.maxCooldownMs);
      expect(config.animation.idleTimeoutMs).not.toBe(0); // the hop's delay
    });

    it('leaves the 60fps path alone — a fast frame re-arms rAF and arms no timer', () => {
      controller.startAnimation();
      expect(mockRAF).toHaveBeenCalledTimes(1);

      runNextFrame(16);

      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacingDelays()).toEqual([]);
    });

    it('paces after a STREAK of slow frames: a hop, then the real cooldown, then rAF', () => {
      controller.startAnimation();

      // First slow frame: a streak of one is an isolated hiccup, not the
      // sustained slowness pacing exists for — still the fast path.
      runNextFrame(1000);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacingDelays()).toEqual([]);

      runNextFrame(1000); // second consecutive 1000ms frame → pace the next one

      // No immediate rAF — the next frame waits out the cooldown.
      expect(mockRAF).not.toHaveBeenCalled();
      // Only the hop is armed so far. The cooldown DOES NOT EXIST yet, and
      // that is the mechanism: armed here it would be measured from the
      // frame's start, and the frame's expensive browser-side work runs after
      // this callback returns inside the same task, so it would already be
      // overdue and insert no gap.
      expect(pacingDelays()).toEqual([0]);

      // The hop fires at the first event-loop turn after that work and arms
      // the real cooldown — min(250, round(1000 * 0.25)) — without releasing
      // the frame.
      vi.advanceTimersByTime(0);
      expect(pacingDelays()).toEqual([0, 250]);
      expect(mockRAF).not.toHaveBeenCalled();

      vi.advanceTimersByTime(250);

      expect(mockRAF).toHaveBeenCalledTimes(1);
    });

    it('the cooldown is a quarter of the frame cost, clamped to maxCooldownMs', () => {
      // Below the clamp: 400ms frames → 100ms gap.
      controller.startAnimation();
      runNextFrame(400); // streak of 1, unpaced
      runNextFrame(400);
      expect(pacedDelaysAfterHop()).toEqual([0, 100]);

      // Above the clamp: 2000ms frames → 500ms uncapped, clamped to 250ms.
      controller.stopAnimation();
      setTimeoutSpy.mockClear();
      controller.startAnimation();
      runNextFrame(2000); // streak of 1, unpaced
      runNextFrame(2000);
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);
    });

    // The streak is what keeps an isolated outlier — a GC pause, a shader
    // compile, one synchronous chunk decode, or any foreign main-thread task
    // charged to the loop because the measurement is a PERIOD — off the paced
    // path. This is the test that goes red if the streak gate is removed.
    it('a SINGLE slow frame is never paced — an isolated hiccup keeps the fast path', () => {
      controller.startAnimation();

      runNextFrame(1000);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacedDelaysAfterHop()).toEqual([]);

      // The fast frame that follows drops the streak, so the next isolated
      // hiccup starts counting from zero again rather than from one.
      runNextFrame(16);
      expect(pacedDelaysAfterHop()).toEqual([]);

      runNextFrame(1000);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacedDelaysAfterHop()).toEqual([]);
    });

    // A fast frame is proof the main thread is already getting slots, which is
    // the only thing a cooldown buys — so an alternating cadence is not paced
    // no matter how long it runs.
    it('an alternating slow/fast cadence is never paced', () => {
      controller.startAnimation();

      for (let i = 0; i < 4; i++) {
        runNextFrame(1000);
        expect(mockRAF).toHaveBeenCalledTimes(1);
        runNextFrame(16);
        expect(mockRAF).toHaveBeenCalledTimes(1);
      }

      expect(pacedDelaysAfterHop()).toEqual([]);
    });

    // Exactly at the threshold is NOT slow (`<=`), one millisecond over is —
    // so a whole RUN at the threshold never advances the streak, while a run
    // one millisecond over paces on its second frame. Pins the comparison
    // against a `<` mutation, and the delay against a fraction/rounding one.
    it('the threshold boundary: cost === slowFrameMs is not paced, +1ms is', () => {
      controller.startAnimation();
      runNextFrame(config.animation.pacing.slowFrameMs);
      runNextFrame(config.animation.pacing.slowFrameMs);
      runNextFrame(config.animation.pacing.slowFrameMs);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacedDelaysAfterHop()).toEqual([]);

      controller.stopAnimation();
      setTimeoutSpy.mockClear();
      controller.startAnimation();
      runNextFrame(config.animation.pacing.slowFrameMs + 1); // 251ms, streak of 1
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacedDelaysAfterHop()).toEqual([]);

      runNextFrame(config.animation.pacing.slowFrameMs + 1); // 251ms, streak of 2
      expect(mockRAF).not.toHaveBeenCalled();
      expect(pacedDelaysAfterHop()).toEqual([0, 63]); // round(251 * 0.25)
    });

    it('recovers: once frames are fast again the loop goes back to plain rAF', () => {
      // This is the test that fails if the cost measurement forgets to
      // subtract the cooldown WE inserted — the period would then read
      // cost + cooldown and pacing would latch on forever.
      controller.startAnimation();
      runNextFrame(1000);
      runNextFrame(1000);
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);

      // Wait out the cooldown; it arms the paced frame's rAF.
      advancePacing(250);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      setTimeoutSpy.mockClear();

      // The paced frame arrives 250ms (our gap) + 16ms (a fast frame) later.
      // One fast frame is enough: it drops the streak to 0 immediately.
      runNextFrame(266);

      expect(mockRAF).toHaveBeenCalledTimes(1); // plain rAF, no cooldown
      expect(pacedDelaysAfterHop()).toEqual([]);

      // And the fast path must have CLEARED the applied-delay bookkeeping.
      // A 260ms period with no gap of ours in it is a 260ms frame; if the
      // stale 250ms were still subtracted both of these would read 10ms and
      // neither would advance the streak, so nothing would be paced.
      runNextFrame(260);
      runNextFrame(260);
      expect(pacedDelaysAfterHop()).toEqual([0, 65]); // round(260 * 0.25)
    });

    // The claim that justifies delaying rather than skipping frames: a paced
    // frame is a WHOLE frame, and the cooldown ticks are pure scheduling.
    it('a paced frame runs exactly one body — one frame-start / frame-end / recordFrame, none on the cooldown ticks', async () => {
      const { eventBus } = await import('../../../utils/cross-layer/event-bus');
      const startListener = vi.fn();
      const endListener = vi.fn();
      const dprManager = { recordFrame: vi.fn() };
      const offStart = eventBus.on('frame-start', startListener);
      const offEnd = eventBus.on('frame-end', endListener);
      controller.setAdaptiveDPRManager(dprManager as never);

      try {
        controller.startAnimation(); // frame 1: nothing measured yet, fast path
        runNextFrame(1000); // frame 2: first slow frame, still unpaced
        startListener.mockClear();
        endListener.mockClear();
        dprManager.recordFrame.mockClear();
        mockControls.update.mockClear();
        mockPostProcessing.render.mockClear();

        // Frame 3 measures a second consecutive 1000ms frame and paces the next.
        runNextFrame(1000);
        expect(pacedDelaysAfterHop()).toEqual([0, 250]);
        expect(startListener).toHaveBeenCalledTimes(1);
        expect(endListener).toHaveBeenCalledTimes(1);
        expect(dprManager.recordFrame).toHaveBeenCalledTimes(1);
        expect(mockControls.update).toHaveBeenCalledTimes(1);
        expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);

        // The hop and the cooldown only schedule — no frame body runs on them.
        advancePacing(250);
        expect(startListener).toHaveBeenCalledTimes(1);
        expect(endListener).toHaveBeenCalledTimes(1);
        expect(dprManager.recordFrame).toHaveBeenCalledTimes(1);
        expect(mockPostProcessing.render).toHaveBeenCalledTimes(1);

        // The frame the cooldown released is one whole frame, no more.
        runNextFrame(266);
        expect(startListener).toHaveBeenCalledTimes(2);
        expect(endListener).toHaveBeenCalledTimes(2);
        expect(dprManager.recordFrame).toHaveBeenCalledTimes(2);
        expect(mockControls.update).toHaveBeenCalledTimes(2);
        expect(mockPostProcessing.render).toHaveBeenCalledTimes(2);
      } finally {
        offStart();
        offEnd();
        controller.stopAnimation();
      }
    });

    // Adaptive DPR gets the RAW clock, paced frames included: the achieved
    // frame rate really is lower and the FPS readout must not lie. There is no
    // virtual pacing clock to keep in step with `notifyContentChanged()`, and
    // the interactions that once argued for one cannot arise, because an
    // isolated slow frame is never paced (the streak above).
    it('hands adaptive DPR the raw frame timestamps, paced frames included', () => {
      const dprManager = { recordFrame: vi.fn() };
      controller.setAdaptiveDPRManager(dprManager as never);

      controller.startAnimation(); // t=1000, first frame
      runNextFrame(16); // t=1016, a 16ms frame
      runNextFrame(1000); // t=2016, first slow frame — streak of 1, unpaced
      expect(pacedDelaysAfterHop()).toEqual([]);
      runNextFrame(1000); // t=3016, second slow frame → paced by 250ms
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);
      advancePacing(250);
      runNextFrame(266); // t=3282: our 250ms gap + a 16ms frame

      const stamps = dprManager.recordFrame.mock.calls.map((call) => call[0] as number);
      expect(stamps).toEqual([1000, 1016, 2016, 3016, 3282]);
      // The paced frame's interval INCLUDES our cooldown — 266ms, not 16ms.
      const intervals = stamps.slice(1).map((t, i) => t - stamps[i]);
      expect(intervals).toEqual([16, 1000, 1000, 266]);
    });

    it('the suspend predicate disables pacing (a recording keeps its cadence)', () => {
      controller.setPacingSuspendPredicate(() => true);
      controller.startAnimation();

      runNextFrame(1000);
      runNextFrame(1000);

      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacingDelays()).toEqual([]);
    });

    // The predicate is read when the cooldown is ARMED, and a capture can
    // start during the very frame that armed it. Without the second read in
    // the hop callback, that cooldown plays out INSIDE the capture — up to
    // maxCooldownMs of frozen duplicate frame at the head of a real-time
    // recording of an already-slow scene.
    it('a capture that starts mid-cooldown drops the pending gap', () => {
      let recording = false;
      controller.setPacingSuspendPredicate(() => recording);
      controller.startAnimation();

      runNextFrame(1000);
      runNextFrame(1000);
      expect(pacingDelays()).toEqual([0]); // the hop; the cooldown is not armed yet
      expect(mockRAF).not.toHaveBeenCalled();

      // The capture starts while the hop is in flight.
      recording = true;
      vi.advanceTimersByTime(0);

      // No cooldown was armed and the frame is released straight away; the
      // only timer left is the idle timer.
      expect(pacingDelays()).toEqual([0]);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(vi.getTimerCount()).toBe(1);

      // The gap that never happened must not be subtracted from the next
      // measurement: 300ms with no cooldown of ours in it is a 300ms frame,
      // so the streak survives and pacing resumes once the capture ends.
      recording = false;
      setTimeoutSpy.mockClear();
      runNextFrame(300);
      expect(mockRAF).not.toHaveBeenCalled();
      expect(pacedDelaysAfterHop()).toEqual([0, 75]); // round(300 * 0.25)
    });

    it('config.animation.pacing.enabled = false disables pacing', () => {
      const wasEnabled = config.animation.pacing.enabled;
      config.animation.pacing.enabled = false;
      try {
        controller.startAnimation();

        runNextFrame(1000);
        runNextFrame(1000);

        expect(mockRAF).toHaveBeenCalledTimes(1);
        expect(pacingDelays()).toEqual([]);
      } finally {
        config.animation.pacing.enabled = wasEnabled;
      }
    });

    it('a resting gap is not a slow frame — the first frame after a resume is not paced', () => {
      controller.startAnimation();
      // Genuinely SLOW frames before the rest, so the measurement carried
      // across it is non-zero: this is what makes the test fail if
      // `lastFrameCostMs = 0` were dropped from startAnimation().
      runNextFrame(1000);
      runNextFrame(1000);
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);
      controller.stopAnimation();

      // The clock keeps running across the rest; without the reset in
      // startAnimation this idle time reads as one enormous frame.
      now += 5000;
      setTimeoutSpy.mockClear();
      mockRAF.mockClear();

      controller.startAnimation();

      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacingDelays()).toEqual([]);

      // And the rest must not become the FIRST link of a new streak either:
      // without `lastFrameStartTime = null` the resume frame would measure the
      // 5000ms rest, and this single slow frame would then be the second.
      runNextFrame(1000);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacedDelaysAfterHop()).toEqual([]);
    });

    // A streak is session state, like the frame-cost measurement: a rest must
    // not leave one banked for the frames after the resume.
    it('the slow-frame streak does not survive a stop/resume', () => {
      controller.startAnimation();
      runNextFrame(1000);
      runNextFrame(1000);
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);

      controller.stopAnimation();
      setTimeoutSpy.mockClear();
      mockRAF.mockClear();
      controller.startAnimation();

      // One slow frame after the resume: a fresh streak of one, unpaced.
      runNextFrame(1000);
      expect(mockRAF).toHaveBeenCalledTimes(1);
      expect(pacedDelaysAfterHop()).toEqual([]);

      // The streak is reset, not disabled — a second one paces as usual.
      runNextFrame(1000);
      expect(mockRAF).not.toHaveBeenCalled();
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);
    });

    it('stopAnimation clears a pending cooldown at either stage, and a late fire does not re-arm a frame', () => {
      controller.startAnimation();
      runNextFrame(1000);
      runNextFrame(1000);
      expect(pacingDelays()).toEqual([0]); // the hop, cooldown not armed yet

      controller.stopAnimation();

      // No timers left at all: the pacing hop and the idle timer are both gone.
      expect(vi.getTimerCount()).toBe(0);

      // And even a hop callback that already escaped the clear is inert — it
      // must not arm the cooldown behind a stopped loop.
      mockRAF.mockClear();
      pacingCallbacks()[0]();
      expect(vi.getTimerCount()).toBe(0);
      vi.advanceTimersByTime(1000);
      expect(mockRAF).not.toHaveBeenCalled();

      // Same at the second stage: stop the loop with the real cooldown armed.
      setTimeoutSpy.mockClear();
      controller.startAnimation();
      runNextFrame(1000);
      runNextFrame(1000);
      vi.advanceTimersByTime(0); // the hop fires, arming the cooldown
      expect(mockRAF).not.toHaveBeenCalled();

      controller.stopAnimation();
      expect(vi.getTimerCount()).toBe(0);
      pacingCallbacks()[1]();
      vi.advanceTimersByTime(1000);
      expect(mockRAF).not.toHaveBeenCalled();
    });

    // The loop's only re-arm point is scheduleNextFrame(), so a suspend
    // predicate that throws must not escape it: nothing would be armed while
    // `isAnimating` stayed true, startAnimation() would early-return forever
    // and no requestRender() could recover. A throw reads as "not suspended".
    it('a throwing suspend predicate never loses the re-arm', () => {
      controller.setPacingSuspendPredicate(() => {
        throw new Error('recording panel disposed');
      });

      expect(() => controller.startAnimation()).not.toThrow();
      expect(mockRAF).toHaveBeenCalledTimes(1);

      // Slow frames still re-arm — paced, since a throw is not a suspend.
      expect(() => runNextFrame(1000)).not.toThrow();
      expect(() => runNextFrame(1000)).not.toThrow();
      expect(pacedDelaysAfterHop()).toEqual([0, 250]);
      advancePacing(250);
      expect(mockRAF).toHaveBeenCalledTimes(1);
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
