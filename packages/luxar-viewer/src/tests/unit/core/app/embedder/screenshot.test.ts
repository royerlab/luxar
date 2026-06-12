/**
 * Unit tests for the embedder screenshot helper. The pure framebuffer-read +
 * encode primitives in `screenshot-exporter.ts` are mocked (jsdom has no real
 * 2D canvas), so these assert OUR composition: option defaults, format
 * normalization passthrough, and the null-encode rejection.
 *
 * @see src/core/app/embedder/screenshot.ts
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';

const renderFrameToCanvas = vi.fn();
const encodeScreenshotBlob = vi.fn();
const normalizeScreenshotFormat = vi.fn();

vi.mock('../../../../../ui/recording-panel/screenshot-exporter', () => ({
  renderFrameToCanvas: (...args: unknown[]) => renderFrameToCanvas(...args),
  encodeScreenshotBlob: (...args: unknown[]) => encodeScreenshotBlob(...args),
  normalizeScreenshotFormat: (...args: unknown[]) => normalizeScreenshotFormat(...args),
}));

import { captureScreenshot } from '../../../../../core/app/embedder/screenshot';

// Minimal SceneManager stand-in — the helper only forwards these refs.
const fakeSceneManager = {
  postProcessing: { id: 'pp' },
  renderer: { domElement: { id: 'gl-canvas' } },
};
// The helper's type wants a SceneManager; cast at the call sites so the
// assertions below can still read the stub's fields.
const sm = fakeSceneManager as never;

describe('captureScreenshot', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    const fakeCanvas = { id: 'frame-canvas' };
    renderFrameToCanvas.mockResolvedValue(fakeCanvas);
    // Echo the requested format through by default.
    normalizeScreenshotFormat.mockImplementation((req: string) => ({ format: req, warning: null }));
    encodeScreenshotBlob.mockResolvedValue(new Blob(['x']));
  });

  it('defaults to png, includeOverlays=true, quality 0.92', async () => {
    const blob = await captureScreenshot(sm, null);

    expect(normalizeScreenshotFormat).toHaveBeenCalledWith('png', false);
    expect(renderFrameToCanvas).toHaveBeenCalledWith(
      fakeSceneManager.postProcessing,
      true, // includeOverlays default
      null,
      fakeSceneManager.renderer.domElement
    );
    expect(encodeScreenshotBlob).toHaveBeenCalledWith({ id: 'frame-canvas' }, 'png', 0.92);
    expect(blob).toBeInstanceOf(Blob);
  });

  it('passes format/quality/includeOverlays through', async () => {
    await captureScreenshot(sm, null, {
      format: 'jpeg',
      quality: 0.5,
      includeOverlays: false,
    });

    expect(normalizeScreenshotFormat).toHaveBeenCalledWith('jpeg', false);
    expect(renderFrameToCanvas).toHaveBeenCalledWith(
      fakeSceneManager.postProcessing,
      false,
      null,
      fakeSceneManager.renderer.domElement
    );
    expect(encodeScreenshotBlob).toHaveBeenCalledWith(expect.anything(), 'jpeg', 0.5);
  });

  it('rejects when encoding fails (toBlob returns null)', async () => {
    encodeScreenshotBlob.mockResolvedValue(null);
    await expect(captureScreenshot(sm, null)).rejects.toThrow(/encoding failed/i);
  });
});
