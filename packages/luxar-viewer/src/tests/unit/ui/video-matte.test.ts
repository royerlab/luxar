// @vitest-environment jsdom
/**
 * Stacked-alpha-matte compositor — the parts that hold without a GPU: the
 * frame geometry, the shader's two taps, and the no-WebGL answer.
 */
import { describe, it, expect, vi } from 'vitest';
import {
  MATTE_FRAGMENT_SHADER,
  createVideoMatteCompositor,
  stackedFrameSize,
} from '../../../ui/video-matte';

/** A <video> that already has a decodable frame of a 400x600 stacked clip. */
function readyVideo(): HTMLVideoElement {
  const video = document.createElement('video');
  Object.defineProperty(video, 'readyState', {
    value: HTMLMediaElement.HAVE_CURRENT_DATA,
    configurable: true,
  });
  Object.defineProperty(video, 'videoWidth', { value: 400, configurable: true });
  Object.defineProperty(video, 'videoHeight', { value: 600, configurable: true });
  return video;
}

/**
 * Replace `getContext` with a counting stub handing out fake GL contexts, so
 * the test observes acquisitions and `WEBGL_lose_context` calls rather than
 * depending on whether the environment has a real GPU. Every property of the
 * fake answers truthily and callably, which is all `setupGl` and `draw` ask of
 * it (shader/program/buffer handles, enum tokens, the lose-context extension).
 */
function stubGl(): { getContext: ReturnType<typeof vi.spyOn>; lost: () => number } {
  let lost = 0;
  const makeGl = (): unknown => {
    const any = (): unknown =>
      new Proxy(function stub() {} as object, {
        get: (_t, prop) => (prop === 'loseContext' ? () => void lost++ : any()),
        apply: () => any(),
      });
    return any();
  };
  const getContext = vi
    .spyOn(HTMLCanvasElement.prototype, 'getContext')
    .mockImplementation(() => makeGl() as never);
  return { getContext, lost: () => lost };
}

describe('stackedFrameSize', () => {
  it('is the full width and half the height of the stacked frame, never below 1x1', () => {
    expect(stackedFrameSize(768, 1536)).toEqual([768, 768]);
    expect(stackedFrameSize(640, 961)).toEqual([640, 480]); // odd heights floor
    expect(stackedFrameSize(0, 0)).toEqual([1, 1]); // before metadata
  });
});

describe('MATTE_FRAGMENT_SHADER', () => {
  it('takes colour from the top half and alpha from the bottom half, premultiplied', () => {
    // Colour tap: v in [0, 0.5) — the top half of a top-down texture.
    expect(MATTE_FRAGMENT_SHADER).toContain('vec2(uv.x, uv.y * 0.5)).rgb');
    // Alpha tap: v in [0.5, 1) — the matte below, read from its red channel.
    expect(MATTE_FRAGMENT_SHADER).toContain('vec2(uv.x, 0.5 + uv.y * 0.5)).r');
    // Premultiplied output, the canvas compositing contract.
    expect(MATTE_FRAGMENT_SHADER).toContain('gl_FragColor = vec4(rgb * a, a)');
    // Texture rows are top-down while clip space is bottom-up.
    expect(MATTE_FRAGMENT_SHADER).toContain('1.0 - v_uv.y');
  });
});

describe('createVideoMatteCompositor', () => {
  it('allocates WebGL lazily on first start and reports when it is unavailable', () => {
    const spy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const onFailure = vi.fn();
    try {
      const matte = createVideoMatteCompositor(document.createElement('video'), { onFailure });
      expect(matte).not.toBeNull();
      expect(spy).not.toHaveBeenCalled();
      matte.start();
      expect(spy).toHaveBeenCalledTimes(1);
      expect(onFailure).toHaveBeenCalledWith(
        expect.objectContaining({ message: 'WebGL is unavailable' })
      );
    } finally {
      spy.mockRestore();
    }
  });

  it('with a context, builds an unattached canvas and tolerates start/stop/dispose before any frame', () => {
    const video = document.createElement('video');
    const matte = createVideoMatteCompositor(video);
    expect(matte.canvas.className).toBe('luxar-overlay__matte');
    expect(matte.canvas.parentElement).toBeNull();
    matte.start();
    matte.start(); // idempotent
    matte.stop();
    matte.dispose();
    expect(matte.canvas.dataset.hasFrame).toBeUndefined(); // readyState 0: nothing was drawn
  });

  it('keeps the default canvas size until video metadata is available', () => {
    const video = document.createElement('video');
    const matte = createVideoMatteCompositor(video);

    expect([matte.canvas.width, matte.canvas.height]).toEqual([300, 150]);
    matte.start();
    matte.stop();
    expect([matte.canvas.width, matte.canvas.height]).toEqual([300, 150]);

    Object.defineProperty(video, 'videoWidth', { value: 768, configurable: true });
    Object.defineProperty(video, 'videoHeight', { value: 1536, configurable: true });
    video.dispatchEvent(new Event('loadedmetadata'));
    expect([matte.canvas.width, matte.canvas.height]).toEqual([768, 768]);
    matte.dispose();
  });

  it('release hands the context back and a later start acquires a fresh one', () => {
    // The reason this matters: a browser caps how many WebGL contexts may be
    // live and evicts the OLDEST when the cap is passed — which is the scene's
    // own renderer. A clip that is off screen must hold no context at all, so
    // a tour of any length costs one context, not one per stop.
    const video = readyVideo();
    const { getContext, lost } = stubGl();
    const onFirstFrame = vi.fn();
    const onRelease = vi.fn();
    try {
      const matte = createVideoMatteCompositor(video, { onFirstFrame, onRelease });
      // Count acquisitions as deltas: a frame callback left scheduled by an
      // earlier test in this file can also reach this spy.
      const acquired = (): number => getContext.mock.calls.length;
      const base = acquired();

      matte.start();
      expect(acquired()).toBe(base + 1);
      expect(matte.canvas.dataset.hasFrame).toBe('1');
      const lostBase = lost();

      matte.release();
      expect(onRelease).toHaveBeenCalledTimes(1);
      expect(lost()).toBe(lostBase + 1); // the context really is handed back
      // The canvas is blank again, so the caller can put its poster back and
      // `onFirstFrame` is owed a second call.
      expect(matte.canvas.dataset.hasFrame).toBeUndefined();

      matte.release(); // idempotent: nothing held, nothing to give back
      expect(acquired()).toBe(base + 1);
      expect(lost()).toBe(lostBase + 1);

      matte.start();
      expect(acquired()).toBe(base + 2); // a FRESH context
      expect(matte.canvas.dataset.hasFrame).toBe('1');
      expect(onFirstFrame).toHaveBeenCalledTimes(2);

      matte.dispose();
      expect(lost()).toBe(lostBase + 2);
    } finally {
      getContext.mockRestore();
    }
  });

  it('stop keeps the context so a re-shown clip needs no new one', () => {
    // stop() is still the cheap path (the last frame stays on the canvas); only
    // the manager's visibility change escalates to release().
    const video = readyVideo();
    const { getContext, lost } = stubGl();
    try {
      const matte = createVideoMatteCompositor(video);
      const base = getContext.mock.calls.length;
      const lostBase = lost();
      matte.start();
      matte.stop();
      matte.start();
      expect(getContext.mock.calls.length).toBe(base + 1);
      expect(lost()).toBe(lostBase);
      matte.dispose();
    } finally {
      getContext.mockRestore();
    }
  });

  it('a clip that failed to be read stays abandoned across a release', () => {
    // A tainted cross-origin clip must not be retried at every story step.
    const video = readyVideo();
    const spy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    const onFailure = vi.fn();
    try {
      const matte = createVideoMatteCompositor(video, { onFailure });
      matte.start();
      expect(onFailure).toHaveBeenCalledTimes(1);
      matte.release();
      matte.start();
      // No second acquisition attempt and no second report.
      expect(spy).toHaveBeenCalledTimes(1);
      expect(onFailure).toHaveBeenCalledTimes(1);
      matte.dispose();
    } finally {
      spy.mockRestore();
    }
  });

  it('marks the canvas and reports exactly once after the first frame is drawn', () => {
    const video = document.createElement('video');
    Object.defineProperty(video, 'readyState', {
      value: HTMLMediaElement.HAVE_CURRENT_DATA,
      configurable: true,
    });
    Object.defineProperty(video, 'videoWidth', { value: 400, configurable: true });
    Object.defineProperty(video, 'videoHeight', { value: 600, configurable: true });
    const onFirstFrame = vi.fn();
    const matte = createVideoMatteCompositor(video, { onFirstFrame });

    matte.start();
    matte.stop();
    matte.start();
    matte.stop();

    expect(matte.canvas.dataset.hasFrame).toBe('1');
    expect(onFirstFrame).toHaveBeenCalledTimes(1);
    matte.dispose();
  });
});
