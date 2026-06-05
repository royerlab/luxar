/**
 * Unit tests for the screenshot-exporter helpers.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  normalizeScreenshotFormat,
  encodeScreenshotBlob,
  downloadBlob,
  renderFrameToCanvas,
} from '../../../../ui/recording-panel/screenshot-exporter';
import type { OverlayManager } from '../../../../ui/overlay-manager';

describe('normalizeScreenshotFormat', () => {
  it('passes through valid image formats unchanged', () => {
    for (const fmt of ['png', 'webp', 'jpeg'] as const) {
      expect(normalizeScreenshotFormat(fmt, false)).toEqual({ format: fmt, warning: null });
    }
  });

  it('downgrades video container formats to PNG with a warning flag', () => {
    for (const fmt of ['mp4', 'webm', 'mkv']) {
      expect(normalizeScreenshotFormat(fmt, false)).toEqual({
        format: 'png',
        warning: 'video-fallback',
      });
    }
  });

  it('downgrades JPEG → PNG when transparency is required', () => {
    expect(normalizeScreenshotFormat('jpeg', true)).toEqual({
      format: 'png',
      warning: 'jpeg-no-alpha',
    });
  });

  it('keeps JPEG when transparency is not required', () => {
    expect(normalizeScreenshotFormat('jpeg', false).format).toBe('jpeg');
  });

  it('falls back to PNG for unknown formats', () => {
    expect(normalizeScreenshotFormat('bizarro', false)).toEqual({
      format: 'png',
      warning: null,
    });
  });
});

describe('encodeScreenshotBlob', () => {
  function makeFakeCanvas(): HTMLCanvasElement {
    const canvas = {} as HTMLCanvasElement;
    Object.defineProperty(canvas, 'toBlob', {
      value: vi.fn((resolve: (b: Blob | null) => void, mimeType: string, quality?: number) => {
        // Surface mime/quality on the spy so the test can assert.
        (canvas as unknown as { lastCall: unknown }).lastCall = { mimeType, quality };
        resolve(new Blob(['x'], { type: mimeType }));
      }),
      writable: true,
    });
    return canvas;
  }

  it('uses image/png and ignores quality for PNG output', async () => {
    const canvas = makeFakeCanvas();
    const blob = await encodeScreenshotBlob(canvas, 'png', 0.5);
    const call = (canvas as unknown as { lastCall: { mimeType: string; quality?: number } })
      .lastCall;
    expect(call.mimeType).toBe('image/png');
    expect(call.quality).toBeUndefined();
    expect(blob).toBeInstanceOf(Blob);
  });

  it('uses image/jpeg with the supplied quality for JPEG output', async () => {
    const canvas = makeFakeCanvas();
    await encodeScreenshotBlob(canvas, 'jpeg', 0.7);
    const call = (canvas as unknown as { lastCall: { mimeType: string; quality?: number } })
      .lastCall;
    expect(call.mimeType).toBe('image/jpeg');
    expect(call.quality).toBe(0.7);
  });

  it('uses image/webp with quality for WebP output', async () => {
    const canvas = makeFakeCanvas();
    await encodeScreenshotBlob(canvas, 'webp', 0.95);
    const call = (canvas as unknown as { lastCall: { mimeType: string; quality?: number } })
      .lastCall;
    expect(call.mimeType).toBe('image/webp');
    expect(call.quality).toBe(0.95);
  });
});

describe('downloadBlob', () => {
  // Install fake timers BEFORE calling downloadBlob so the test
  // observes the scheduled revoke. (Installing them after lets the
  // real setTimeout schedule first, before fake timers take over.)
  it('appends an anchor, clicks it, and revokes the URL after a 100ms delay', () => {
    vi.useFakeTimers();
    const before = document.body.children.length;
    const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake');
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    try {
      downloadBlob(new Blob(['x']), 'test.png');

      expect(createSpy).toHaveBeenCalledTimes(1);
      expect(document.body.children.length).toBeGreaterThanOrEqual(before);
      // Revoke must be scheduled but NOT fired immediately.
      expect(revokeSpy).not.toHaveBeenCalled();

      // Advance just past the 100 ms revoke delay.
      vi.advanceTimersByTime(150);
      expect(revokeSpy).toHaveBeenCalledTimes(1);
      expect(revokeSpy).toHaveBeenCalledWith('blob:fake');
    } finally {
      vi.useRealTimers();
      createSpy.mockRestore();
      revokeSpy.mockRestore();
    }
  });

  it('revokes the URL synchronously and rethrows when appendChild throws', () => {
    // A synchronous DOM exception (e.g. document.body removed mid-call,
    // some test environments) must not leak the object URL.
    const createSpy = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:throws');
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const appendSpy = vi.spyOn(document.body, 'appendChild').mockImplementation(() => {
      throw new Error('detached document');
    });

    try {
      expect(() => downloadBlob(new Blob(['x']), 'test.png')).toThrow('detached document');
      // Catch path revokes immediately — no setTimeout queued.
      expect(revokeSpy).toHaveBeenCalledTimes(1);
      expect(revokeSpy).toHaveBeenCalledWith('blob:throws');
    } finally {
      createSpy.mockRestore();
      revokeSpy.mockRestore();
      appendSpy.mockRestore();
    }
  });
});

describe('renderFrameToCanvas', () => {
  it('reads an ImageData from postProcessing, sizes the canvas, and returns it', async () => {
    const imgData = { width: 200, height: 100, data: new Uint8ClampedArray(200 * 100 * 4) };
    const postProcessing = {
      renderToImageData: vi.fn(async () => imgData as unknown as ImageData),
    };
    // jsdom doesn't give us a real 2D context; supply a fake on the
    // canvas's getContext.
    const fakePut = vi.fn();
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      putImageData: fakePut,
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;

    try {
      const canvas = await renderFrameToCanvas(
        postProcessing,
        false,
        null,
        document.createElement('canvas')
      );
      expect(canvas.width).toBe(200);
      expect(canvas.height).toBe(100);
      expect(fakePut).toHaveBeenCalledTimes(1);
      expect(postProcessing.renderToImageData).toHaveBeenCalledTimes(1);
    } finally {
      HTMLCanvasElement.prototype.getContext = origGetContext;
    }
  });

  it('skips compositing when includeOverlays is false', async () => {
    const imgData = { width: 10, height: 10, data: new Uint8ClampedArray(10 * 10 * 4) };
    const postProcessing = {
      renderToImageData: vi.fn(async () => imgData as unknown as ImageData),
    };
    const overlayManager = {
      getVisibleOverlays: vi.fn(),
    } as unknown as OverlayManager;
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      putImageData: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;

    try {
      await renderFrameToCanvas(
        postProcessing,
        false,
        overlayManager,
        document.createElement('canvas')
      );
      expect(overlayManager.getVisibleOverlays).not.toHaveBeenCalled();
    } finally {
      HTMLCanvasElement.prototype.getContext = origGetContext;
    }
  });

  it('composites overlays when includeOverlays is true and a manager is present', async () => {
    // [P5/G2] The includeOverlays=true branch (screenshot-exporter.ts:60-62)
    // was never exercised — only the false path had a test.
    const imgData = { width: 10, height: 10, data: new Uint8ClampedArray(10 * 10 * 4) };
    const postProcessing = {
      renderToImageData: vi.fn(async () => imgData as unknown as ImageData),
    };
    const overlayManager = {
      getVisibleOverlays: vi.fn(() => []),
    } as unknown as OverlayManager;
    const origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
      putImageData: vi.fn(),
    })) as unknown as typeof HTMLCanvasElement.prototype.getContext;

    try {
      await renderFrameToCanvas(
        postProcessing,
        true,
        overlayManager,
        document.createElement('canvas')
      );
      expect(overlayManager.getVisibleOverlays).toHaveBeenCalledTimes(1);
    } finally {
      HTMLCanvasElement.prototype.getContext = origGetContext;
    }
  });
});
