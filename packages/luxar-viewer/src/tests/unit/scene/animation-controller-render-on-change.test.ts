/**
 * Render-on-change contract of AnimationController (#2944 A2).
 *
 * The loop keeps TICKING as before (controls, per-frame callbacks, the idle
 * timer), but a tick RENDERS only when something changed since the last
 * rendered frame: an explicit request (wake / event / once), a per-frame
 * callback that returned `true`, a camera or drawing-buffer change, or a
 * consumer that needs every frame (cinematic detector noise, a real-time
 * recording, `?renderAlways`). Same harness style as
 * `animation-controller.test.ts`: fake timers, a stubbed rAF whose armed
 * callback the test runs, mocked controls / post-processing.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as THREE from 'three';

vi.mock('../../../controls/controls-manager', () => ({
  ControlsManager: vi.fn(),
}));

vi.mock('../../../rendering/post-processing/post-processing-manager', () => ({
  PostProcessingManager: vi.fn(),
}));

vi.mock('../../../rendering/adaptive-dpr-manager', () => ({
  AdaptiveDPRManager: vi.fn(),
}));

import { AnimationController } from '../../../scene/animation/animation-controller';
import { RenderAudit } from '../../../scene/animation/render-audit';
import { config } from '../../../config';
import { eventBus } from '../../../utils/cross-layer/event-bus';
import { log } from '../../../utils/log';
import { perfCounters } from '../../../profiling/perf-counters';

describe('AnimationController render-on-change', () => {
  let controller: AnimationController;
  let mockControls: {
    update: ReturnType<typeof vi.fn>;
    isAutoRotateActive: ReturnType<typeof vi.fn>;
    isAutoDollyActive: ReturnType<typeof vi.fn>;
    isGestureActive: ReturnType<typeof vi.fn>;
  };
  let mockPostProcessing: {
    render: ReturnType<typeof vi.fn>;
    needsContinuousAnimation: ReturnType<typeof vi.fn>;
  };
  let mockRAF: ReturnType<typeof vi.fn>;
  let camera: THREE.PerspectiveCamera;
  let canvas: { width: number; height: number };
  let now: number;

  beforeEach(() => {
    vi.useFakeTimers();
    perfCounters.reset();
    mockControls = {
      update: vi.fn(),
      isAutoRotateActive: vi.fn().mockReturnValue(false),
      isAutoDollyActive: vi.fn().mockReturnValue(false),
      isGestureActive: vi.fn().mockReturnValue(false),
    };
    mockPostProcessing = {
      render: vi.fn(),
      needsContinuousAnimation: vi.fn().mockReturnValue(false),
    };
    let rafId = 0;
    mockRAF = vi.fn().mockImplementation(() => ++rafId);
    vi.stubGlobal('requestAnimationFrame', mockRAF);
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
    now = 1000;
    vi.spyOn(performance, 'now').mockImplementation(() => now);

    camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    camera.position.set(0, 0, 10);
    canvas = { width: 800, height: 600 };
    controller = new AnimationController(mockControls as never, mockPostProcessing as never);
    controller.setViewSignatureSource({
      getCamera: () => camera,
      getDrawingBuffer: () => canvas,
    });
  });

  afterEach(() => {
    controller.dispose();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  /** Run the frame the most recent requestAnimationFrame armed (+16 ms). */
  function runFrame(): void {
    const calls = mockRAF.mock.calls;
    const callback = calls[calls.length - 1][0] as FrameRequestCallback;
    now += 16;
    callback(now);
  }

  function runFrames(n: number): void {
    for (let i = 0; i < n; i++) runFrame();
  }

  const renders = (): number => mockPostProcessing.render.mock.calls.length;

  it('is on by default (config.animation.renderOnChange)', () => {
    expect(config.animation.renderOnChange).toBe(true);
    expect(controller.isRenderOnChange).toBe(true);
  });

  it('renders exactly once per wake, then the clean tail ticks do not render', () => {
    controller.startAnimation();
    runFrames(10);

    expect(renders()).toBe(1);
    // The loop kept ticking: controls ran on every frame.
    expect(mockControls.update).toHaveBeenCalledTimes(10);
    expect(perfCounters.get('render.ticks')).toBe(10);
    expect(perfCounters.get('render.skippedTicks')).toBe(9);
    expect(perfCounters.get('render.byReason.wake')).toBe(1);
  });

  it('every wake gets its render, even mid-tail', () => {
    controller.startAnimation();
    runFrames(3);
    controller.startAnimation();
    runFrames(3);

    expect(renders()).toBe(2);
  });

  it('requestRender marks the next tick dirty (reason `event`)', () => {
    controller.startAnimation();
    runFrames(2);
    controller.requestRender('geometry');
    runFrames(2);

    expect(renders()).toBe(2);
    expect(perfCounters.get('render.byReason.event')).toBe(1);
  });

  it('a request made DURING the render dirties the next tick, not this one', () => {
    let requested = false;
    mockPostProcessing.render.mockImplementation(() => {
      if (!requested) {
        requested = true;
        controller.requestRender('onAfterRender');
      }
    });
    controller.startAnimation();
    runFrames(4);

    expect(renders()).toBe(2);
  });

  it('requestTick keeps the loop ticking without rendering', () => {
    controller.startAnimation();
    runFrame();
    const before = renders();
    controller.requestTick();
    runFrames(5);

    expect(renders()).toBe(before);
    expect(controller.isActive).toBe(true);
  });

  it('a camera matrix change renders without a wake', () => {
    controller.startAnimation();
    runFrames(3);
    expect(renders()).toBe(1);

    // Moved through position alone (matrixWorld is stale until updated): the
    // signature must still see it.
    camera.position.x += 1;
    runFrame();
    expect(renders()).toBe(2);
    expect(perfCounters.get('render.byReason.camera')).toBe(1);

    runFrames(3);
    expect(renders()).toBe(2);
  });

  it('a projection change (near/far, zoom) renders', () => {
    controller.startAnimation();
    runFrames(2);
    camera.near = 0.5;
    camera.updateProjectionMatrix();
    runFrame();

    expect(renders()).toBe(2);
  });

  it('a drawing-buffer resize renders (a DPR step clears the canvas)', () => {
    controller.startAnimation();
    runFrames(2);
    canvas.width = 400;
    runFrame();

    expect(renders()).toBe(2);
  });

  it('a camera object swap (ortho toggle) renders', () => {
    controller.startAnimation();
    runFrames(2);
    const other = camera.clone();
    camera = other;
    runFrame();

    expect(renders()).toBe(2);
  });

  it('a per-frame callback returning true renders; false does not', () => {
    let changed = false;
    controller.addPerFrameCallback('lod-group-selector', () => changed);
    controller.addPerFrameCallback('quiet', () => false);
    controller.startAnimation();
    runFrames(3);
    expect(renders()).toBe(1);

    changed = true;
    runFrame();
    changed = false;
    runFrames(2);

    expect(renders()).toBe(2);
    expect(perfCounters.get('render.byReason.cb:lod-group-selector')).toBe(1);
  });

  it('a callback must state whether it changed what is drawn (compile-time)', () => {
    // The return is required: a callback that mutates drawn state and returns
    // nothing would leave a stale frame. These directives ARE the assertion —
    // if PerFrameCallback ever accepts `void` again, `pnpm typecheck` fails
    // on the unused @ts-expect-error.
    let mutated = 0;
    // @ts-expect-error a callback returning nothing does not say whether it changed the frame
    controller.addPerFrameCallback('forgot', () => {});
    // @ts-expect-error nor does one that mutates state and returns nothing
    controller.addPerFrameCallback('forgot-mutating', () => {
      mutated++;
    });
    controller.addPerFrameCallback('explicit', () => false);
    controller.startAnimation();
    runFrames(2);

    // At runtime a stray `undefined` (an untyped caller) still counts as "no change".
    expect(mutated).toBe(2);
    expect(renders()).toBe(1);
  });

  it('every callback still runs after one reports a change', () => {
    const later = vi.fn();
    controller.addPerFrameCallback('first', () => true);
    controller.addPerFrameCallback('later', later);
    controller.startAnimation();
    runFrames(2);

    expect(later).toHaveBeenCalledTimes(2);
  });

  it('cinematic needsContinuousAnimation renders every tick', () => {
    mockPostProcessing.needsContinuousAnimation.mockReturnValue(true);
    controller.startAnimation();
    runFrames(6);

    expect(renders()).toBe(6);
    expect(perfCounters.get('render.byReason.continuous')).toBe(5);
  });

  it('a renderEveryFrame keep-alive (real-time recording) renders every tick', () => {
    controller.addPerFrameCallback('recording-keep-alive', () => false, {
      continuous: true,
      renderEveryFrame: true,
    });
    controller.startAnimation();
    runFrames(5);
    expect(renders()).toBe(5);

    controller.removePerFrameCallback('recording-keep-alive');
    runFrames(3);
    expect(renders()).toBe(5);
  });

  it('re-registering a renderEveryFrame callback does not double-count it', () => {
    const options = { continuous: true, renderEveryFrame: true };
    controller.addPerFrameCallback('rec', () => false, options);
    controller.addPerFrameCallback('rec', () => false, options);
    controller.removePerFrameCallback('rec');
    controller.startAnimation();
    runFrames(3);

    expect(renders()).toBe(1);
  });

  it('a continuous callback keeps the loop ticking past the idle timeout without rendering', () => {
    const cb = vi.fn();
    controller.addPerFrameCallback('dimension-animation', cb, { continuous: true });
    controller.startAnimation();
    runFrame();
    vi.advanceTimersByTime(config.animation.idleTimeoutMs + 10);
    expect(controller.isActive).toBe(true);
    runFrames(5);

    expect(cb).toHaveBeenCalledTimes(6);
    expect(renders()).toBe(1);
  });

  it('renderAlways restores the legacy loop: every tick renders', () => {
    controller.setRenderOnChange(false);
    expect(controller.isRenderOnChange).toBe(false);
    controller.startAnimation();
    runFrames(8);

    expect(renders()).toBe(8);
    expect(perfCounters.get('render.skippedTicks')).toBe(0);
  });

  it('context loss skips the render but keeps the frame dirty for recovery', () => {
    let lost = true;
    controller.setContextLostPredicate(() => lost);
    controller.startAnimation();
    runFrames(3);
    expect(renders()).toBe(0);

    lost = false;
    runFrame();
    expect(renders()).toBe(1);
  });

  it('renderOnce on a running loop marks the next tick dirty', () => {
    controller.startAnimation();
    runFrames(2);
    controller.renderOnce();
    expect(renders()).toBe(1); // nothing drawn synchronously on a running loop
    runFrame();

    expect(renders()).toBe(2);
    expect(perfCounters.get('render.byReason.once')).toBe(1);
  });

  it('renderOnce on a stopped loop always draws now', () => {
    controller.renderOnce();
    expect(renders()).toBe(1);
    controller.stopAnimation();
    controller.renderOnce();
    expect(renders()).toBe(2);
  });

  it('prepareFrame marks the loop dirty when a callback changed drawn state', () => {
    let changed = false;
    controller.addPerFrameCallback('depth-sort-scheduler', () => changed);
    controller.startAnimation();
    runFrames(2);
    changed = true;
    controller.prepareFrame();
    changed = false;
    expect(renders()).toBe(1); // prepareFrame never draws
    runFrame();

    expect(renders()).toBe(2);
  });

  it('frame-end carries `rendered` so readouts count only drawn frames', () => {
    const rendered: Array<boolean | undefined> = [];
    const off = eventBus.on('frame-end', (frame) => rendered.push(frame.rendered));
    try {
      controller.startAnimation();
      runFrames(3);
    } finally {
      off();
    }
    expect(rendered).toEqual([true, false, false]);
  });

  describe('adaptive DPR sampling', () => {
    let manager: {
      recordFrame: ReturnType<typeof vi.fn>;
      notifyStreamBreak: ReturnType<typeof vi.fn>;
      notifyPaused: ReturnType<typeof vi.fn>;
    };

    beforeEach(() => {
      manager = { recordFrame: vi.fn(), notifyStreamBreak: vi.fn(), notifyPaused: vi.fn() };
      controller.setAdaptiveDPRManager(manager as never);
    });

    it('records only the intervals of rendered ticks', () => {
      let changed = true;
      controller.addPerFrameCallback('motion', () => changed);
      controller.startAnimation();
      runFrames(4); // t = 1016, 1032, 1048, 1064 — all render
      changed = false;
      runFrames(3); // t = 1080 (samples 1064's frame), 1096, 1112 — skipped

      expect(manager.recordFrame.mock.calls.map((c) => c[0])).toEqual([1032, 1048, 1064, 1080]);
      expect(perfCounters.get('adaptiveDpr.samples')).toBe(4);
    });

    it('re-bases on the first render after skipped ticks, without a sample', () => {
      let changed = false;
      controller.addPerFrameCallback('playback', () => changed);
      controller.startAnimation();
      runFrame(); // t=1016: wake render (first of the stream: re-based)
      runFrames(4); // t=1032 samples it; 1048..1080 skipped
      changed = true;
      runFrame(); // t=1096: renders after skips → stream break, no sample
      changed = false;
      runFrame(); // t=1112: samples the 1096 frame

      expect(manager.notifyStreamBreak.mock.calls.map((c) => c[0])).toEqual([1016, 1096]);
      expect(manager.recordFrame.mock.calls.map((c) => c[0])).toEqual([1032, 1112]);
    });

    it('under renderAlways every tick records, as before', () => {
      controller.setRenderOnChange(false);
      controller.startAnimation();
      runFrames(4);

      expect(manager.recordFrame).toHaveBeenCalledTimes(4);
      expect(manager.notifyStreamBreak).not.toHaveBeenCalled();
    });

    it('a stop/resume starts a fresh stream (re-based, not sampled across the rest)', () => {
      controller.addPerFrameCallback('motion', () => true);
      controller.startAnimation();
      runFrames(2);
      controller.stopAnimation();
      now += 5000;
      controller.startAnimation();
      runFrame();

      expect(manager.recordFrame.mock.calls.map((c) => c[0])).toEqual([1032]);
      expect(manager.notifyStreamBreak.mock.calls.map((c) => c[0])).toEqual([1016, 6048]);
    });
  });

  describe('idle timeout', () => {
    it('still stops the loop after idleTimeoutMs of no wakes', () => {
      controller.startAnimation();
      runFrames(3);
      vi.advanceTimersByTime(config.animation.idleTimeoutMs + 10);

      expect(controller.isActive).toBe(false);
    });

    it('runs the idle-restore render exactly as before', () => {
      const manager = {
        recordFrame: vi.fn(),
        notifyPaused: vi.fn(),
        isActive: vi.fn(() => true),
        prepareIdleFrame: vi.fn(() => true),
      };
      controller.setAdaptiveDPRManager(manager as never);
      controller.startAnimation();
      runFrames(3);
      expect(renders()).toBe(1);
      vi.advanceTimersByTime(config.animation.idleTimeoutMs + 10);

      expect(controller.isActive).toBe(false);
      expect(manager.prepareIdleFrame).toHaveBeenCalledTimes(1);
      expect(renders()).toBe(2);
      expect(perfCounters.get('render.byReason.idleRestore')).toBe(1);
    });
  });

  describe('render audit', () => {
    it('renders every tick and counts a skipped tick whose pixels changed as a missed dirty', () => {
      vi.spyOn(log, 'warning').mockImplementation(() => {});
      let pixel = 10;
      const audit = new RenderAudit(() => new Uint8ClampedArray([pixel, 0, 0, 255]));
      controller.setRenderAudit(audit);
      controller.startAnimation();
      runFrames(3);
      // Every tick rendered (1 real + 2 audit frames), all identical.
      expect(renders()).toBe(3);
      expect(perfCounters.get('render.missedDirty')).toBe(0);
      expect(perfCounters.get('render.byReason.audit')).toBe(2);

      // Drawn state changed silently (nobody requested a render).
      pixel = 200;
      runFrames(3);
      expect(perfCounters.get('render.missedDirty')).toBe(1);
    });

    it('a reported change is not a missed dirty', () => {
      let pixel = 10;
      controller.setRenderAudit(new RenderAudit(() => new Uint8ClampedArray([pixel, 0, 0, 255])));
      controller.startAnimation();
      runFrames(2);
      pixel = 200;
      controller.requestRender('material');
      runFrames(3);

      expect(perfCounters.get('render.missedDirty')).toBe(0);
    });
  });
});
