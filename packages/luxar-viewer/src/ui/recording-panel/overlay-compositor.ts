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
 * Text and image overlays are drawn with canvas-2D primitives (one
 * blend-mode map, one save/restore per overlay, the shared anchor
 * offset rules); HTML overlays are rasterized by wrapping the live node
 * in an `<svg><foreignObject>` data URL. That wrapper is parsed as XML,
 * so the markup is XML-serialized rather than read off `outerHTML` —
 * HTML serialization leaves void elements such as `<br>` unclosed,
 * which is fatal there. Wrapped text follows the DOM's own
 * `white-space: normal` layout, where a `\n` is a space and not a
 * break.
 *
 * @module ui/recording-panel/overlay-compositor
 */

import type { OverlayManager } from '../overlay-manager';
import { FONT_PRESETS } from '../overlay-manager';
import type { OverlayConfig } from '../../data/loaders';
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
 * The two scale factors every overlay branch needs to reproduce what
 * the screen shows.
 *
 * On screen, overlay sizes are authored in VIEWPORT units — the
 * overlay manager writes `font-size: <font_size × 100>vh`,
 * `width: <size[0] × 100>vw`, `padding: …vh` — and an `<img>` with no
 * configured size lays out at its natural CSS-pixel size. None of
 * those are relative to the capture frame, so resolving them against
 * the capture canvas (as this module used to) only agrees with the
 * screen in the one case where the canvas exactly fills the viewport
 * AND the capture is exactly canvas-sized. Recording breaks both
 * halves of that: the offline path renders at the chosen output
 * height (1080p/1440p/4K), and an embedded viewer's canvas is a
 * fraction of the window.
 *
 * So: convert to CSS pixels first (viewport-relative sizes against the
 * real viewport, natural image sizes as-is), then multiply by the
 * capture-pixels-per-CSS-pixel ratio. This is the same mapping the
 * HTML branch has always used for its `getBoundingClientRect` math.
 */
export interface OverlayCaptureMetrics {
  /** Capture pixels per CSS pixel, horizontally. */
  scaleX: number;
  /** Capture pixels per CSS pixel, vertically. */
  scaleY: number;
  /** Capture pixels spanned by a `1.0` (=100vw) width fraction. */
  vw: number;
  /** Capture pixels spanned by a `1.0` (=100vh) height fraction. */
  vh: number;
}

/** Viewport dimensions, injectable so the metrics stay unit-testable. */
export interface ViewportSize {
  width: number;
  height: number;
}

/**
 * Derive {@link OverlayCaptureMetrics} for a capture.
 *
 * Falls back to treating the capture canvas as the viewport when the
 * on-screen canvas has no layout box (detached / display:none), which
 * keeps a headless or offscreen capture rendering something sane
 * instead of dividing by zero.
 */
export function computeOverlayMetrics(
  captureWidth: number,
  captureHeight: number,
  glCanvas: HTMLCanvasElement,
  viewport: ViewportSize
): OverlayCaptureMetrics {
  const rect = glCanvas.getBoundingClientRect();
  const hasLayout = rect.width > 0 && rect.height > 0;
  const scaleX = hasLayout ? captureWidth / rect.width : 1;
  const scaleY = hasLayout ? captureHeight / rect.height : 1;
  return {
    scaleX,
    scaleY,
    vw: hasLayout ? viewport.width * scaleX : captureWidth,
    vh: hasLayout ? viewport.height * scaleY : captureHeight,
  };
}

/**
 * Composite every visible overlay onto the capture canvas. No-op
 * when the overlay manager has nothing to draw.
 *
 * `glCanvas` is the renderer's DOM element — its bounding rect maps
 * CSS pixels to capture pixels (see {@link OverlayCaptureMetrics}) and
 * is the coordinate frame for HTML-overlay positioning.
 */
export function compositeOverlays(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  overlayManager: OverlayManager,
  glCanvas: HTMLCanvasElement,
  viewport: ViewportSize = { width: window.innerWidth, height: window.innerHeight }
): void {
  const overlays = overlayManager.getVisibleOverlays();
  if (overlays.length === 0) return;

  const w = canvas.width;
  const h = canvas.height;
  const metrics = computeOverlayMetrics(w, h, glCanvas, viewport);

  for (const { el, config } of overlays) {
    ctx.save();

    if (config.blend_mode && config.blend_mode !== 'normal') {
      ctx.globalCompositeOperation = BLEND_MODE_TO_COMPOSITE[config.blend_mode] ?? 'source-over';
    }

    // A live fade writes an inline opacity; fall back to the config only
    // when there is no inline value to read (`parseFloat('')` is NaN).
    // `|| config.opacity` would have promoted a legitimate 0 to full.
    const inlineOpacity = parseFloat(el.style.opacity);
    ctx.globalAlpha = Number.isFinite(inlineOpacity) ? inlineOpacity : config.opacity;

    // `position` is placed as a fraction of the frame. On screen the
    // overlay is `position: fixed`, so its `left/top: %` resolve against
    // the viewport — the two agree whenever the canvas fills the window,
    // and for a capture the frame is the box you want to place against
    // (a viewport fraction can land outside an embedded canvas entirely).
    const [nx, ny] = config.position;
    const x = nx * w;
    const y = ny * h;

    if (el.classList.contains('luxar-overlay--text')) {
      compositeTextOverlay(ctx, el, config, x, y, metrics);
    } else if (el.classList.contains('luxar-overlay--image')) {
      compositeImageOverlay(ctx, el, config, x, y, metrics);
    } else if (el.classList.contains('luxar-overlay--video')) {
      compositeVideoOverlay(ctx, el, config, x, y, metrics);
    } else if (el.classList.contains('luxar-overlay--html')) {
      compositeHtmlOverlay(ctx, el, glCanvas);
    }

    ctx.restore();
  }
}

/**
 * Split a single word that cannot fit on a line of its own into
 * character chunks that do. At least one character always goes on a
 * chunk, so a maxWidth narrower than one glyph terminates.
 */
function breakWord(ctx: CanvasRenderingContext2D, word: string, maxWidth: number): string[] {
  const chunks: string[] = [];
  let chunk = '';
  for (const ch of word) {
    if (chunk && ctx.measureText(chunk + ch).width > maxWidth) {
      chunks.push(chunk);
      chunk = ch;
    } else {
      chunk += ch;
    }
  }
  if (chunk) chunks.push(chunk);
  return chunks;
}

/**
 * Greedy word-wrap into lines no wider than `maxWidth`. A word wider
 * than the box is split mid-word: the DOM overlay sets `word-wrap:
 * break-word` alongside its `width`, so on screen an unbreakable token
 * (a long URL or identifier) breaks rather than overflowing, and the
 * capture has to do the same.
 *
 * Newlines are NOT line breaks here. Every overlay this module can see
 * is laid out `white-space: normal` (the overlay manager only sets
 * `pre-line` for hover overlays, and those are excluded from
 * compositing), so on screen a `\n` collapses to a space like any other
 * whitespace. Honouring it would break the line where the DOM does not
 * — and a trailing newline would add an empty line to the block, lifting
 * a bottom-anchored overlay by a whole line height.
 */
function wrapText(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const lines: string[] = [];
  let line = '';
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const pieces = ctx.measureText(word).width > maxWidth ? breakWord(ctx, word, maxWidth) : [word];
    for (const piece of pieces) {
      const candidate = line ? `${line} ${piece}` : piece;
      if (line && ctx.measureText(candidate).width > maxWidth) {
        lines.push(line);
        line = piece;
      } else {
        line = candidate;
      }
    }
  }
  lines.push(line);
  return lines;
}

/** Composite a single text overlay onto the capture canvas. */
export function compositeTextOverlay(
  ctx: CanvasRenderingContext2D,
  el: HTMLDivElement,
  config: OverlayConfig,
  xIn: number,
  yIn: number,
  metrics: OverlayCaptureMetrics
): void {
  const text = el.textContent ?? '';
  if (!text) return;

  // Font sizes / paddings / strokes are authored as vh fractions, and
  // `width` as a vw fraction — the same units the DOM overlay uses.
  const fontSize = (config.font_size ?? 0.03) * metrics.vh;
  const fontFamily = FONT_PRESETS[config.font ?? 'sans'] ?? config.font ?? FONT_PRESETS.sans;
  ctx.font = `${fontSize}px ${fontFamily}`;
  ctx.textBaseline = 'top';

  // An overlay with a configured width wraps on screen (the manager sets
  // `width: …vw` + `word-wrap: break-word`), so the capture has to wrap
  // too — `fillText` alone would run one long line off the frame.
  const lineHeight = fontSize * (config.line_height ?? 1.2);
  const wrapWidth = config.width ? config.width * metrics.vw : null;
  const lines = wrapWidth ? wrapText(ctx, text, wrapWidth) : [text];

  // `reduce`, not `Math.max(...)`: a wrapped overlay can produce an
  // unbounded number of lines and spreading them all as arguments blows
  // the stack at ~100k.
  const longestLine = lines.reduce((w, l) => Math.max(w, ctx.measureText(l).width), 0);
  // The on-screen box is the configured width when there is one, even if
  // the text is shorter — anchoring must use the same box.
  const blockWidth = wrapWidth ?? longestLine;
  // One line box per line, as on screen (the default 1.2 keeps the
  // single-line height this used to hardcode).
  const blockHeight = lineHeight * lines.length;

  const [dx, dy] = computeAnchorOffset(config.anchor, blockWidth, blockHeight);
  const x = xIn + dx;
  const y = yIn + dy;

  if (config.background) {
    const padding = (config.padding ?? 0.005) * metrics.vh;
    ctx.fillStyle = config.background;
    ctx.fillRect(x - padding, y - padding, blockWidth + padding * 2, blockHeight + padding * 2);
  }

  // Horizontal alignment inside the block (only meaningful with a width).
  const align = config.text_align ?? 'left';
  const lineX = (lineWidth: number): number => {
    if (align === 'center') return x + (blockWidth - lineWidth) / 2;
    if (align === 'right') return x + (blockWidth - lineWidth);
    return x;
  };

  const strokeWidth = config.stroke_color ? (config.stroke_width ?? 0.002) * metrics.vh : 0;
  if (config.stroke_color) {
    ctx.strokeStyle = config.stroke_color;
    ctx.lineWidth = strokeWidth * 2;
    ctx.lineJoin = 'round';
  }
  ctx.fillStyle = config.color ?? '#ffffff';

  lines.forEach((line, i) => {
    const lx = lineX(ctx.measureText(line).width);
    const ly = y + i * lineHeight;
    if (config.stroke_color) ctx.strokeText(line, lx, ly);
    ctx.fillText(line, lx, ly);
  });
}

/** Composite a single image overlay onto the capture canvas. */
export function compositeImageOverlay(
  ctx: CanvasRenderingContext2D,
  el: HTMLDivElement,
  config: OverlayConfig,
  xIn: number,
  yIn: number,
  metrics: OverlayCaptureMetrics
): void {
  const img = el.querySelector('img');
  if (!img || !img.complete || img.naturalWidth === 0) return;

  let drawW: number;
  let drawH: number;
  if (config.size) {
    // `size` is [vw, vh] fractions — the units the manager writes onto
    // the <img> style.
    drawW = config.size[0] * metrics.vw;
    // A null height means "keep the media's aspect": derive it from the
    // element's natural dimensions, falling back to square.
    drawH =
      config.size[1] == null
        ? drawW * (img.naturalWidth > 0 ? img.naturalHeight / img.naturalWidth : 1)
        : config.size[1] * metrics.vh;
  } else {
    // No configured size: the <img> lays out at its natural size in CSS
    // pixels, so the capture has to scale those into capture pixels.
    // Drawing raw natural pixels made the image shrink relative to the
    // frame as the recording resolution went up (a 64 px logo stayed
    // 64 px whether the frame was 720 or 2160 tall).
    drawW = img.naturalWidth * metrics.scaleX;
    drawH = img.naturalHeight * metrics.scaleY;
  }

  const [dx, dy] = computeAnchorOffset(config.anchor, drawW, drawH);
  ctx.drawImage(img, xIn + dx, yIn + dy, drawW, drawH);
}

/** Composite the current frame of a single video overlay. */
export function compositeVideoOverlay(
  ctx: CanvasRenderingContext2D,
  el: HTMLDivElement,
  config: OverlayConfig,
  xIn: number,
  yIn: number,
  metrics: OverlayCaptureMetrics
): void {
  const video = el.querySelector('video');
  if (
    !video ||
    video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
    video.videoWidth === 0
  ) {
    return;
  }

  let drawW: number;
  let drawH: number;
  if (config.size) {
    drawW = config.size[0] * metrics.vw;
    drawH =
      config.size[1] == null
        ? drawW * (video.videoHeight / video.videoWidth)
        : config.size[1] * metrics.vh;
  } else {
    drawW = video.videoWidth * metrics.scaleX;
    drawH = video.videoHeight * metrics.scaleY;
  }

  const [dx, dy] = computeAnchorOffset(config.anchor, drawW, drawH);
  ctx.drawImage(video, xIn + dx, yIn + dy, drawW, drawH);
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
  // XMLSerializer, not `outerHTML`: the markup goes into an
  // `image/svg+xml` data URL, which the browser parses as XML, and HTML
  // serialization leaves void elements unclosed. A single `<br>` — which
  // the overlay allowlist permits and several demos use — is then a fatal
  // well-formedness error, so the Image never loads and the overlay is
  // missing from every captured frame rather than merely mis-drawn.
  const markup = new XMLSerializer().serializeToString(clone);
  const svg = `<svg xmlns="${svgNs}" width="${elRect.width}" height="${elRect.height}"><foreignObject width="${pct}" height="${pct}"><div xmlns="${xhtmlNs}">${markup}</div></foreignObject></svg>`;

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
