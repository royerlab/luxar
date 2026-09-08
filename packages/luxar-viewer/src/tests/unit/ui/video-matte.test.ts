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
      matte!.start();
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
    if (!matte) return; // a setup without a GL stub: covered by the case above
    expect(matte.canvas.className).toBe('luxar-overlay__matte');
    expect(matte.canvas.parentElement).toBeNull();
    matte.start();
    matte.start(); // idempotent
    matte.stop();
    matte.dispose();
    expect(matte.canvas.dataset.hasFrame).toBeUndefined(); // readyState 0: nothing was drawn
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
    if (!matte) return;

    matte.start();
    matte.stop();
    matte.start();
    matte.stop();

    expect(matte.canvas.dataset.hasFrame).toBe('1');
    expect(onFirstFrame).toHaveBeenCalledTimes(1);
    matte.dispose();
  });
});
