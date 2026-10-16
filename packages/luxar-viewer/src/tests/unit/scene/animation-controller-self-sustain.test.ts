/**
 * AnimationController self-sustain suspension — the render loop must not keep
 * itself alive while the data server is unreachable (cinematic noise,
 * auto-rotate, playback and `requestTick` polls would otherwise render
 * indefinitely against a scene that can no longer stream), while user input
 * still renders and the loop resumes when the server answers again.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

vi.mock('../../../controls/controls-manager', () => ({ ControlsManager: vi.fn() }));
vi.mock('../../../rendering/post-processing/post-processing-manager', () => ({
  PostProcessingManager: vi.fn(),
}));
vi.mock('../../../rendering/adaptive-dpr-manager', () => ({ AdaptiveDPRManager: vi.fn() }));

import { AnimationController } from '../../../scene/animation/animation-controller';
import { config } from '../../../config';

const REASON = 'data-server-unreachable';

describe('AnimationController self-sustain suspension', () => {
  let controller: AnimationController;
  let controls: {
    update: ReturnType<typeof vi.fn>;
    isAutoRotateActive: ReturnType<typeof vi.fn>;
    isAutoDollyActive: ReturnType<typeof vi.fn>;
    isGestureActive: ReturnType<typeof vi.fn>;
  };
  let post: {
    render: ReturnType<typeof vi.fn>;
    needsContinuousAnimation: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    vi.useFakeTimers();
    controls = {
      update: vi.fn(),
      isAutoRotateActive: vi.fn().mockReturnValue(false),
      isAutoDollyActive: vi.fn().mockReturnValue(false),
      isGestureActive: vi.fn().mockReturnValue(false),
    };
    // Cinematic mode: continuous by design.
    post = { render: vi.fn(), needsContinuousAnimation: vi.fn().mockReturnValue(true) };
    let rafId = 0;
    vi.stubGlobal(
      'requestAnimationFrame',
      vi.fn(() => ++rafId)
    );
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
    controller = new AnimationController(controls as never, post as never);
  });

  afterEach(() => {
    controller.dispose();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('keeps a continuous consumer running when not suspended (cinematic by design)', () => {
    controller.startAnimation();
    vi.advanceTimersByTime(config.animation.idleTimeoutMs * 3);
    expect(controller.isActive).toBe(true);
  });

  it('stops the loop at once when suspended', () => {
    controller.startAnimation();
    controller.suspendSelfSustainedAnimation(REASON);
    expect(controller.isActive).toBe(false);
    expect(controller.isSelfSustainSuspended).toBe(true);
  });

  it('lets user input wake the loop, which then idles out despite cinematic', () => {
    controller.suspendSelfSustainedAnimation(REASON);
    controller.startAnimation();
    expect(controller.isActive).toBe(true);
    vi.advanceTimersByTime(config.animation.idleTimeoutMs + 1);
    expect(controller.isActive).toBe(false);
  });

  it('ignores requestTick polls while suspended', () => {
    controller.suspendSelfSustainedAnimation(REASON);
    controller.requestTick();
    expect(controller.isActive).toBe(false);
  });

  it('does not stop a gesture in progress', () => {
    controller.startAnimation();
    controls.isGestureActive.mockReturnValue(true);
    controller.suspendSelfSustainedAnimation(REASON);
    expect(controller.isActive).toBe(true);
  });

  it('resumes when the last reason is released', () => {
    controller.startAnimation();
    controller.suspendSelfSustainedAnimation(REASON);
    controller.suspendSelfSustainedAnimation('other');
    controller.resumeSelfSustainedAnimation(REASON);
    expect(controller.isActive).toBe(false);
    controller.resumeSelfSustainedAnimation('other');
    expect(controller.isActive).toBe(true);
    vi.advanceTimersByTime(config.animation.idleTimeoutMs * 3);
    expect(controller.isActive).toBe(true);
  });

  it('releasing an unknown reason is a no-op', () => {
    controller.resumeSelfSustainedAnimation(REASON);
    expect(controller.isActive).toBe(false);
  });

  it('is never in effect while a capture runs (exempt predicate)', () => {
    let recording = true;
    controller.setSelfSustainExemptPredicate(() => recording);
    controller.startAnimation();
    controller.suspendSelfSustainedAnimation(REASON);
    // Recording: the loop keeps running and continuous consumers keep it alive.
    expect(controller.isActive).toBe(true);
    expect(controller.isSelfSustainSuspended).toBe(false);
    vi.advanceTimersByTime(config.animation.idleTimeoutMs * 3);
    expect(controller.isActive).toBe(true);
    // The capture ends: the hold takes effect again and the loop idles out.
    recording = false;
    expect(controller.isSelfSustainSuspended).toBe(true);
    vi.advanceTimersByTime(config.animation.idleTimeoutMs + 1);
    expect(controller.isActive).toBe(false);
  });

  it('exempts a registered renderEveryFrame consumer (real-time video)', () => {
    controller.addPerFrameCallback('video-keepalive', () => false, {
      continuous: true,
      renderEveryFrame: true,
    });
    controller.startAnimation();
    controller.suspendSelfSustainedAnimation(REASON);
    expect(controller.isActive).toBe(true);
    vi.advanceTimersByTime(config.animation.idleTimeoutMs * 3);
    expect(controller.isActive).toBe(true);
    controller.removePerFrameCallback('video-keepalive');
    expect(controller.isSelfSustainSuspended).toBe(true);
  });

  it('treats a throwing exempt predicate as exempt', () => {
    controller.setSelfSustainExemptPredicate(() => {
      throw new Error('boom');
    });
    controller.suspendSelfSustainedAnimation(REASON);
    expect(controller.isSelfSustainSuspended).toBe(false);
  });

  it('does not wake a hidden page on release', () => {
    controller.suspendSelfSustainedAnimation(REASON);
    vi.stubGlobal('document', { hidden: true });
    controller.resumeSelfSustainedAnimation(REASON);
    expect(controller.isActive).toBe(false);
    expect(controller.isSelfSustainSuspended).toBe(false);
  });
});
