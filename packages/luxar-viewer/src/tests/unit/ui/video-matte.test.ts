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
  it('declines (null) where there is no WebGL, so the manager shows the plain clip', () => {
    // The test setup stubs a GL context on jsdom canvases; take it away here.
    const spy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    try {
      expect(createVideoMatteCompositor(document.createElement('video'))).toBeNull();
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
    expect(matte.hasFrame).toBe(false);
    matte.start();
    matte.start(); // idempotent
    matte.stop();
    matte.dispose();
    expect(matte.hasFrame).toBe(false); // readyState 0: nothing was drawn
  });
});
