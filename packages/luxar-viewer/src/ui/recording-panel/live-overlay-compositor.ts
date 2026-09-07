/**
 * Live overlay compositing for the REAL-TIME (MediaRecorder) capture path.
 *
 * The offline loop composites overlays by re-reading the framebuffer per
 * frame (`renderFrameToCanvas`), which the real-time path cannot afford —
 * it films at wall-clock speed. So instead of handing
 * `canvas.captureStream()` the WebGL canvas (which carries no DOM
 * overlays), the recorder captures a MIRROR 2D canvas that this class
 * refreshes once per rendered frame: blit the GL canvas, then run the
 * same `compositeOverlays()` the screenshot path uses.
 *
 * ## Why `frame-end` and not `requestAnimationFrame`
 *
 * The renderer runs with `preserveDrawingBuffer: false` (see
 * `config/sections/webgl/data.ts`), so the drawing buffer is cleared once
 * the browser composites the frame. `drawImage(glCanvas)` therefore only
 * sees pixels while still inside the SAME task that issued the draw calls.
 * `frame-end` is emitted by `AnimationController.animate()` on the line
 * after `postProcessing.render()`, which is exactly that window. A plain
 * rAF callback runs in a later task and blits a fully black frame —
 * measured, not assumed.
 *
 * ## Cost
 *
 * One `drawImage` of the whole canvas plus the canvas-2D overlay draws.
 * Measured at 0.1 ms median / 0.3 ms max on a 2942x1602 (4.7 Mpx) canvas,
 * i.e. below the noise floor of a 60 FPS budget. It is only attached when
 * the capture actually has overlays to draw.
 *
 * ## What it cannot do
 *
 * HTML overlays are rasterized through an async `<svg><foreignObject>`
 * data URL, and `compositeHtmlOverlay` only draws one that already
 * happens to be decoded — the same best-effort behaviour the screenshot
 * and offline paths have. Text, image, and video overlays are exact.
 *
 * @module ui/recording-panel/live-overlay-compositor
 */

import { log, Modules } from '../../utils/log';
import { eventBus } from '../../utils/cross-layer/event-bus';
import type { OverlayManager } from '../overlay-manager';
import { compositeOverlays } from './overlay-compositor';

/**
 * Mirror canvas + per-frame compositing subscription. Construct via
 * {@link createLiveOverlayCompositor}, which returns `null` when there is
 * nothing to composite.
 */
export class LiveOverlayCompositor {
  /** The canvas to hand `captureStream()` — NOT the WebGL canvas. */
  readonly canvas: HTMLCanvasElement;

  private readonly ctx: CanvasRenderingContext2D;
  private unsubscribe: (() => void) | null = null;
  /** Latched so a per-frame failure logs once instead of every frame. */
  private loggedFailure = false;

  constructor(
    private readonly glCanvas: HTMLCanvasElement,
    private readonly overlayManager: OverlayManager
  ) {
    this.canvas = document.createElement('canvas');
    // Frozen for the whole recording: MediaRecorder is fed a fixed frame
    // size, and `saveRecordingState` locks resize for the duration. A GL
    // canvas that changes size anyway is scaled into this one (see
    // composite) rather than resizing the stream mid-flight.
    this.canvas.width = glCanvas.width;
    this.canvas.height = glCanvas.height;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('LiveOverlayCompositor: 2D context unavailable');
    this.ctx = ctx;
  }

  /**
   * Subscribe to the post-render hook. Idempotent — a second call while
   * already attached is a no-op rather than a duplicate blit per frame.
   */
  attach(): void {
    if (this.unsubscribe) return;
    this.unsubscribe = eventBus.on('frame-end', this.composite);
    // Paint once immediately so the stream's first frame is never the
    // blank canvas: captureStream() can grab a frame before the next
    // render lands. This one runs outside the render task, so the GL
    // buffer may already be cleared — `drawImage` then contributes
    // nothing and only the overlays land, which is still better than a
    // fully empty frame and is overwritten a frame later.
    this.composite();
  }

  /** Unsubscribe. Idempotent, and safe to call from a dispose path. */
  detach(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
  }

  /**
   * Blit the GL canvas and draw the overlays over it.
   *
   * Arrow-bound so `eventBus.on`/the matching unsubscribe see one stable
   * reference (the bind-a-new-function listener-leak trap).
   */
  private composite = (): void => {
    try {
      const { width, height } = this.canvas;
      // The GL canvas is opaque in the common case, but a transparent
      // background recording would otherwise accumulate previous frames
      // under the current one.
      this.ctx.clearRect(0, 0, width, height);
      // Explicit destination size: a GL canvas that changed size despite
      // the resize lock is scaled to fit rather than corrupting the
      // stream's frame geometry.
      this.ctx.drawImage(this.glCanvas, 0, 0, width, height);
      compositeOverlays(this.canvas, this.ctx, this.overlayManager, this.glCanvas);
    } catch (err) {
      if (!this.loggedFailure) {
        this.loggedFailure = true;
        log.warning(Modules.RECORDING, `Live overlay compositing failed: ${err}`);
      }
    }
  };
}

/**
 * Build an attached {@link LiveOverlayCompositor}, or `null` when the
 * recording has no overlays to draw — in which case the caller should
 * capture the WebGL canvas directly and pay nothing.
 *
 * Returns `null` rather than throwing if the mirror canvas cannot be
 * created: a recording without overlays beats no recording at all.
 */
export function createLiveOverlayCompositor(
  includeOverlays: boolean,
  overlayManager: OverlayManager | null,
  glCanvas: HTMLCanvasElement
): LiveOverlayCompositor | null {
  if (!includeOverlays || !overlayManager) return null;
  if (overlayManager.getVisibleOverlays().length === 0) return null;
  try {
    const compositor = new LiveOverlayCompositor(glCanvas, overlayManager);
    compositor.attach();
    return compositor;
  } catch (err) {
    log.warning(Modules.RECORDING, `Live overlay compositing unavailable: ${err}`);
    return null;
  }
}
