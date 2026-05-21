/**
 * Overlay-compositing concern extracted from `recording-panel.ts`.
 *
 * The recording pipeline draws its own copy of the visible DOM
 * overlays onto the capture canvas so that screenshot / video output
 * matches what the user sees on screen. The canvas-2D pipeline is
 * pure given:
 *   - the capture canvas + 2D context,
 *   - the list of currently-visible overlays (from OverlayManager),
 *   - the renderer's GL canvas (only needed for the HTML branch's
 *     coordinate mapping).
 *
 * Behavior is identical to the inline original: same blend-mode map,
 * same per-overlay save/restore, same anchor offset rules, same
 * SVG foreignObject rasterization for HTML overlays.
 *
 * @module ui/recording-panel/overlay-compositor
 */

import type { OverlayManager } from '../overlay-manager';
import { FONT_PRESETS } from '../overlay-manager';
import type { OverlayConfig } from '../../data/loaders/overlay-loader';
import { anchorOffset as computeAnchorOffset } from './media-utilities';
import { log, Modules } from '../../utils/log';

/** Maps Luxar blend mode names to Canvas 2D globalCompositeOperation values. */
const BLEND_MODE_TO_COMPOSITE: Record<string, GlobalCompositeOperation> = {
  normal: 'source-over',
  multiply: 'multiply',
  screen: 'screen',
  overlay: 'overlay',
  additive: 'lighter',
  difference: 'difference',
};

/**
 * Composite every visible overlay onto the capture canvas. No-op
 * when the overlay manager has nothing to draw.
 *
 * `glCanvas` is the renderer's DOM element — its bounding rect is the
 * coordinate frame for HTML-overlay positioning. For text and image
 * overlays, only the canvas dimensions matter.
 */
export function compositeOverlays(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  overlayManager: OverlayManager,
  glCanvas: HTMLCanvasElement
): void {
  const overlays = overlayManager.getVisibleOverlays();
  if (overlays.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;

  for (const { el, config } of overlays) {
    ctx.save();

    if (config.blend_mode && config.blend_mode !== 'normal') {
      ctx.globalCompositeOperation = BLEND_MODE_TO_COMPOSITE[config.blend_mode] ?? 'source-over';
    }

    ctx.globalAlpha = parseFloat(el.style.opacity) || config.opacity;

    const [nx, ny] = config.position;
    const x = nx * w;
    const y = ny * h;

    if (el.classList.contains('luxar-overlay--text')) {
      compositeTextOverlay(ctx, el, config, x, y, w, h);
    } else if (el.classList.contains('luxar-overlay--image')) {
      compositeImageOverlay(ctx, el, config, x, y, w, h);
    } else if (el.classList.contains('luxar-overlay--html')) {
      compositeHtmlOverlay(ctx, el, glCanvas);
    }

    ctx.restore();
  }
}

/** Composite a single text overlay onto the capture canvas. */
export function compositeTextOverlay(
  ctx: CanvasRenderingContext2D,
  el: HTMLDivElement,
  config: OverlayConfig,
  xIn: number,
  yIn: number,
  _canvasW: number,
  canvasH: number
): void {
  const text = el.textContent ?? '';
  if (!text) return;

  // Font sizes / paddings / strokes are stored as vh-relative fractions;
  // resolve against canvas height for capture-time pixel values.
  const fontSize = (config.font_size ?? 0.03) * canvasH;
  const fontFamily = FONT_PRESETS[config.font ?? 'sans'] ?? config.font ?? FONT_PRESETS.sans;
  ctx.font = `${fontSize}px ${fontFamily}`;
  ctx.textBaseline = 'top';

  const metrics = ctx.measureText(text);
  const textWidth = metrics.width;
  const textHeight = fontSize * 1.2;

  const [dx, dy] = computeAnchorOffset(config.anchor, textWidth, textHeight);
  const x = xIn + dx;
  const y = yIn + dy;

  if (config.background) {
    const padding = (config.padding ?? 0.005) * canvasH;
    ctx.fillStyle = config.background;
    ctx.fillRect(x - padding, y - padding, textWidth + padding * 2, textHeight + padding * 2);
  }

  if (config.stroke_color) {
    const strokeWidth = (config.stroke_width ?? 0.002) * canvasH;
    ctx.strokeStyle = config.stroke_color;
    ctx.lineWidth = strokeWidth * 2;
    ctx.lineJoin = 'round';
    ctx.strokeText(text, x, y);
  }

  ctx.fillStyle = config.color ?? '#ffffff';
  ctx.fillText(text, x, y);
}

/** Composite a single image overlay onto the capture canvas. */
export function compositeImageOverlay(
  ctx: CanvasRenderingContext2D,
  el: HTMLDivElement,
  config: OverlayConfig,
  xIn: number,
  yIn: number,
  canvasW: number,
  canvasH: number
): void {
  const img = el.querySelector('img');
  if (!img || !img.complete || img.naturalWidth === 0) return;

  let drawW: number;
  let drawH: number;
  if (config.size) {
    drawW = config.size[0] * canvasW;
    drawH = config.size[1] * canvasH;
  } else {
    drawW = img.naturalWidth;
    drawH = img.naturalHeight;
  }

  const [dx, dy] = computeAnchorOffset(config.anchor, drawW, drawH);
  ctx.drawImage(img, xIn + dx, yIn + dy, drawW, drawH);
}

/**
 * Composite an HTML overlay by serializing its DOM into an
 * `<svg><foreignObject>` data URL and drawing the resulting Image.
 * The data URL decodes synchronously in modern browsers; if the
 * decode hasn't happened yet (rare), we log and skip the frame.
 */
export function compositeHtmlOverlay(
  ctx: CanvasRenderingContext2D,
  el: HTMLDivElement,
  glCanvas: HTMLCanvasElement
): void {
  const glRect = glCanvas.getBoundingClientRect();
  const elRect = el.getBoundingClientRect();
  if (glRect.width === 0 || glRect.height === 0) return;

  const scaleX = ctx.canvas.width / glRect.width;
  const scaleY = ctx.canvas.height / glRect.height;

  // For HTML the anchor/position config doesn't apply — we use the
  // element's actual rendered position so it matches what the user sees.
  const x = (elRect.left - glRect.left) * scaleX;
  const y = (elRect.top - glRect.top) * scaleY;
  const drawW = elRect.width * scaleX;
  const drawH = elRect.height * scaleY;

  const clone = el.cloneNode(true) as HTMLDivElement;
  const computed = getComputedStyle(el);
  clone.style.cssText = '';
  for (const prop of [
    'color',
    'font-family',
    'font-size',
    'font-weight',
    'line-height',
    'background-color',
    'padding',
    'border',
    'white-space',
    'word-wrap',
    'text-align',
  ] as const) {
    clone.style.setProperty(prop, computed.getPropertyValue(prop));
  }
  clone.style.setProperty('position', 'static');
  clone.style.setProperty('width', `${elRect.width}px`);
  clone.style.setProperty('height', `${elRect.height}px`);
  clone.style.setProperty('overflow', 'hidden');

  const svgNs = 'http://www.w3.org/2000/svg';
  const xhtmlNs = 'http://www.w3.org/1999/xhtml';
  const pct = '100%';
  const svg = `<svg xmlns="${svgNs}" width="${elRect.width}" height="${elRect.height}"><foreignObject width="${pct}" height="${pct}"><div xmlns="${xhtmlNs}">${clone.outerHTML}</div></foreignObject></svg>`;

  const img = new Image();
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);

  if (img.complete && img.naturalWidth > 0) {
    ctx.drawImage(img, x, y, drawW, drawH);
  } else {
    img
      .decode()
      .then(() => {
        log.info(Modules.RECORDING, '[Overlay] HTML overlay rasterized async (missed frame)');
      })
      .catch(() => {
        log.warning(Modules.RECORDING, '[Overlay] Failed to rasterize HTML overlay');
      });
  }
}
