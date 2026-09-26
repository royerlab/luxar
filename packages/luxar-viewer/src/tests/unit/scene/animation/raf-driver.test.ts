/**
 * Tests for RafDriver (frame scheduling) and the scheduling/work seam it
 * gives AnimationController: `tick()` is the frame's work and runs without
 * the window's requestAnimationFrame, which is what a WebXR session's own
 * animation loop needs.
 *
 * Pacing itself is covered through the controller's public API in
 * `animation-controller.test.ts`, which passes unchanged across the split.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../../controls/controls-manager', () => ({ ControlsManager: vi.fn() }));
vi.mock('../../../../rendering/post-processing/post-processing-manager', () => ({
  PostProcessingManager: vi.fn(),
}));
vi.mock('../../../../rendering/adaptive-dpr-manager', () => ({ AdaptiveDPRManager: vi.fn() }));

import { RafDriver } from '../../../../scene/animation/raf-driver';
import { AnimationController } from '../../../../scene/animation/animation-controller';
import { eventBus } from '../../../../utils/cross-layer/event-bus';

describe('RafDriver', () => {
  let rafCallbacks: Map<number, FrameRequestCallback>;
  let nextId: number;

  beforeEach(() => {
    vi.useFakeTimers();
    rafCallbacks = new Map();
    nextId = 0;
    vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
      rafCallbacks.set(++nextId, cb);
      return nextId;
    });
    vi.stubGlobal('cancelAnimationFrame', (id: number) => rafCallbacks.delete(id));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  /** Run every requestAnimationFrame callback armed so far, once. */
  const flushFrame = (): void => {
    const pending = [...rafCallbacks.entries()];
    rafCallbacks.clear();
    for (const [, cb] of pending) cb(0);
  };

  it('start() runs the first frame synchronously and reports the edge once', () => {
    const onFrame = vi.fn();
    const driver = new RafDriver(onFrame);
    expect(driver.isRunning).toBe(false);

    expect(driver.start()).toBe(true);
    expect(driver.isRunning).toBe(true);
    expect(onFrame).toHaveBeenCalledTimes(1);

    // Already running: no second loop, no extra frame.
    expect(driver.start()).toBe(false);
    expect(onFrame).toHaveBeenCalledTimes(1);
    expect(rafCallbacks.size).toBe(1);
    driver.stop();
  });

  it('arms the next frame before the frame work runs, so a throwing frame keeps the loop', () => {
    let armedWhenWorkRan = -1;
    const driver = new RafDriver(() => {
      armedWhenWorkRan = rafCallbacks.size;
      throw new Error('frame work failed');
    });
    expect(() => driver.start()).toThrow('frame work failed');
    expect(armedWhenWorkRan).toBe(1);
    expect(driver.isRunning).toBe(true);
    // The loop survived: the next frame is armed and runs.
    expect(rafCallbacks.size).toBe(1);
    expect(() => flushFrame()).toThrow('frame work failed');
    expect(rafCallbacks.size).toBe(1);
    driver.stop();
  });

  it('stop() cancels the pending frame, and a frame that already fired does nothing', () => {
    const onFrame = vi.fn();
    const driver = new RafDriver(onFrame);
    driver.start();
    const [pendingId, pendingCb] = [...rafCallbacks.entries()][0];

    driver.stop();
    expect(driver.isRunning).toBe(false);
    expect(rafCallbacks.has(pendingId)).toBe(false);

    // A callback the browser had already dequeued must not resurrect the loop.
    pendingCb(0);
    expect(onFrame).toHaveBeenCalledTimes(1);
    expect(rafCallbacks.size).toBe(0);
  });

  it('each scheduled frame runs the work exactly once', () => {
    const onFrame = vi.fn();
    const driver = new RafDriver(onFrame);
    driver.start();
    flushFrame();
    flushFrame();
    expect(onFrame).toHaveBeenCalledTimes(3);
    driver.stop();
  });
});

describe('AnimationController.tick() (the frame work, without a driver)', () => {
  it('runs controls, callbacks and one render without starting the loop', () => {
    vi.stubGlobal('requestAnimationFrame', vi.fn());
    const order: string[] = [];
    const controls = {
      update: vi.fn(() => order.push('controls')),
      isAutoRotateActive: () => false,
      isAutoDollyActive: () => false,
      isGestureActive: () => false,
    };
    const post = {
      render: vi.fn(() => order.push('render')),
      needsContinuousAnimation: () => false,
    };
    const controller = new AnimationController(controls as never, post as never);
    controller.addPerFrameCallback('cb', () => order.push('callback'));
    const events: string[] = [];
    const offStart = eventBus.on('frame-start', () => events.push('frame-start'));
    const offEnd = eventBus.on('frame-end', () => events.push('frame-end'));
    try {
      controller.tick();
      expect(order).toEqual(['controls', 'callback', 'render']);
      expect(events).toEqual(['frame-start', 'frame-end']);
      // tick() is work only: it neither starts the loop nor schedules a frame.
      expect(controller.isActive).toBe(false);
      expect(requestAnimationFrame).not.toHaveBeenCalled();
    } finally {
      offStart();
      offEnd();
      controller.dispose();
      vi.unstubAllGlobals();
    }
  });
});
