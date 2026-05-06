/**
 * Screenshot-export concern extracted from `recording-panel.ts`.
 *
 * Three pure-ish helpers that the recording panel composes:
 *   - `renderFrameToCanvas` — read the post-processed framebuffer
 *     into a 2D canvas and (optionally) composite overlays. Used by
 *     both the screenshot path and the offline-capture loops in
 *     video / turntable modes.
 *   - `encodeScreenshotBlob` — pick the right MIME type, fall back
 *     from JPEG to PNG when transparency is required, and produce
 *     the encoded Blob via `canvas.toBlob`.
 *   - `downloadBlob` — trigger the standard "download a blob" flow
 *     by creating an invisible `<a>` element with an object URL.
 *
 * The state-machine parts of captureScreenshot (panel hide/show,
 * DPR override, save/restore recording state) stay in the panel
 * because they're tightly coupled to its mode tracking.
 *
 * @module ui/recording/screenshot-exporter
 */

import type { OverlayManager } from '../overlay-manager';
import { compositeOverlays } from './overlay-compositor';

/** Subset of PostProcessingManager the screenshot path reads. */
export interface ScreenshotPostProcessing {
  renderToImageData(): ImageData;
}

/**
 * Render the current frame into a fresh 2D canvas, then composite
 * visible DOM overlays on top if requested. The returned canvas can
 * be fed straight to `canvas.toBlob()` or used as a VideoSample
 * source.
 *
 * The post-processing manager owns the framebuffer read; this helper
 * never touches WebGL state directly.
 */
export function renderFrameToCanvas(
  postProcessing: ScreenshotPostProcessing,
  includeOverlays: boolean,
  overlayManager: OverlayManager | null,
  glCanvas: HTMLCanvasElement
): HTMLCanvasElement {
  const imgData = postProcessing.renderToImageData();
  const canvas = document.createElement('canvas');
  canvas.width = imgData.width;
  canvas.height = imgData.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(imgData, 0, 0);

  if (includeOverlays && overlayManager) {
    compositeOverlays(canvas, ctx, overlayManager, glCanvas);
  }

  return canvas;
}

/**
 * Output format whose encoding can be done with `canvas.toBlob`.
 * EXR / video formats are excluded because they go through a
 * different pipeline.
 */
export type ScreenshotImageFormat = 'png' | 'webp' | 'jpeg';

/**
 * Validate and normalize a screenshot output format. Falls back to
 * PNG if the requested format is a video container, and downgrades
 * JPEG → PNG when transparency is required (JPEG has no alpha).
 *
 * Returns the effective format to encode and a flag indicating
 * whether any normalization happened (for telemetry / toasts).
 */
export function normalizeScreenshotFormat(
  requested: string,
  transparentBackground: boolean
): { format: ScreenshotImageFormat; warning: 'video-fallback' | 'jpeg-no-alpha' | null } {
  if (requested === 'mp4' || requested === 'webm' || requested === 'mkv') {
    return { format: 'png', warning: 'video-fallback' };
  }
  if (transparentBackground && requested === 'jpeg') {
    return { format: 'png', warning: 'jpeg-no-alpha' };
  }
  if (requested === 'png' || requested === 'webp' || requested === 'jpeg') {
    return { format: requested, warning: null };
  }
  // Unknown / 'exr' shouldn't reach here — caller filters those — but
  // PNG is the safest default if it does.
  return { format: 'png', warning: null };
}

/**
 * Encode a canvas to a Blob with the requested image format. Returns
 * `null` if the browser fails to encode (matches `toBlob`'s API).
 *
 * The `quality` parameter is honored only for lossy formats (jpeg,
 * webp); PNG ignores it per the HTML spec.
 */
export function encodeScreenshotBlob(
  canvas: HTMLCanvasElement,
  format: ScreenshotImageFormat,
  quality: number
): Promise<Blob | null> {
  const mimeType = format === 'jpeg' ? 'image/jpeg' : `image/${format}`;
  const q = format === 'png' ? undefined : quality;
  return new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, mimeType, q));
}

/**
 * Trigger the standard browser "download a blob" flow: create an
 * invisible `<a>` element with an object URL, click it, then revoke
 * the URL after a short delay so the download has time to start.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    a.remove();
    URL.revokeObjectURL(url);
  }, 100);
}
