/**
 * Headless screenshot for the programmatic embedder API.
 *
 * Composes the pure helpers in `ui/recording-panel/screenshot-exporter.ts`
 * (the same ones the Recording panel uses) WITHOUT any panel/state-machine
 * dependency, so `LuxarApp.screenshot()` can produce a Blob from host code.
 *
 * Async because `PostProcessingManager.renderToImageData()` is async (the
 * WebGPU readback path needs `buffer.mapAsync()`); under WebGL2 the wrapped
 * read is synchronous.
 */

import type { SceneManager } from '../../../scene/scene-manager';
import type { OverlayManager } from '../../../ui/overlay-manager';
import {
  renderFrameToCanvas,
  normalizeScreenshotFormat,
  encodeScreenshotBlob,
} from '../../../ui/recording-panel/screenshot-exporter';
import type { ScreenshotOptions } from './events';

/**
 * Render the current frame to an encoded image Blob.
 *
 * @param sceneManager    Provides the post-processed framebuffer and GL canvas.
 * @param overlayManager  Visible DOM overlays to composite, or `null` to skip.
 * @param opts            Format / quality / overlay options.
 * @throws if the browser fails to encode the canvas to a Blob.
 */
export async function captureScreenshot(
  sceneManager: SceneManager,
  overlayManager: OverlayManager | null,
  opts: ScreenshotOptions = {}
): Promise<Blob> {
  const includeOverlays = opts.includeOverlays ?? true;
  const { format } = normalizeScreenshotFormat(opts.format ?? 'png', false);
  const quality = opts.quality ?? 0.92;

  const canvas = await renderFrameToCanvas(
    sceneManager.postProcessing,
    includeOverlays,
    overlayManager,
    sceneManager.renderer.domElement
  );

  const blob = await encodeScreenshotBlob(canvas, format, quality);
  if (!blob) {
    throw new Error(`Screenshot encoding failed (format: ${format}).`);
  }
  return blob;
}
