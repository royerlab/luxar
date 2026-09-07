// @vitest-environment jsdom
/**
 * Unit tests for the overlay-compositor helpers.
 *
 * Strategy: jsdom doesn't provide a real Canvas 2D context, so we
 * construct a hand-rolled fake 2D context with the methods + state
 * the helpers touch. Real DOM elements stand in for overlays so the
 * class-based dispatch (`luxar-overlay--text` etc.) is exercised.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  compositeOverlays,
  compositeTextOverlay,
  compositeImageOverlay,
  compositeHtmlOverlay,
  computeOverlayMetrics,
} from '../../../../ui/recording-panel/overlay-compositor';
import type { OverlayCaptureMetrics } from '../../../../ui/recording-panel/overlay-compositor';
import type { OverlayManager } from '../../../../ui/overlay-manager';
import type { OverlayConfig } from '../../../../data/loaders';

interface FakeCtx {
  canvas: { width: number; height: number };
  globalAlpha: number;
  globalCompositeOperation: string;
  font: string;
  textBaseline: string;
  fillStyle: string;
  strokeStyle: string;
  lineWidth: number;
  lineJoin: string;
  save: ReturnType<typeof vi.fn>;
  restore: ReturnType<typeof vi.fn>;
  fillRect: ReturnType<typeof vi.fn>;
  strokeText: ReturnType<typeof vi.fn>;
  fillText: ReturnType<typeof vi.fn>;
  drawImage: ReturnType<typeof vi.fn>;
  measureText: ReturnType<typeof vi.fn>;
}

function makeFakeCtx(canvasW = 800, canvasH = 600): FakeCtx {
  return {
    canvas: { width: canvasW, height: canvasH },
    globalAlpha: 1,
    globalCompositeOperation: 'source-over',
    font: '',
    textBaseline: 'alphabetic',
    fillStyle: '#000',
    strokeStyle: '#000',
    lineWidth: 1,
    lineJoin: 'miter',
    save: vi.fn(),
    restore: vi.fn(),
    fillRect: vi.fn(),
    strokeText: vi.fn(),
    fillText: vi.fn(),
    drawImage: vi.fn(),
    measureText: vi.fn(() => ({ width: 50 })),
  };
}

function makeCanvas(width = 800, height = 600) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

/**
 * A canvas with a real layout box. jsdom's getBoundingClientRect is all
 * zeros, which is the "no layout" fallback path — tests that care about
 * the CSS-pixel → capture-pixel ratio have to supply one.
 */
function makeLaidOutCanvas(cssW: number, cssH: number, pxW = cssW, pxH = cssH) {
  const canvas = makeCanvas(pxW, pxH);
  canvas.getBoundingClientRect = () =>
    ({ width: cssW, height: cssH, left: 0, top: 0, right: cssW, bottom: cssH }) as DOMRect;
  return canvas;
}

/** 1:1 metrics — capture pixel == CSS pixel == viewport-relative unit. */
function unitMetrics(vw = 800, vh = 600): OverlayCaptureMetrics {
  return { scaleX: 1, scaleY: 1, vw, vh };
}

function makeTextOverlay(text: string, opacity = '1'): HTMLDivElement {
  const el = document.createElement('div');
  el.classList.add('luxar-overlay--text');
  el.textContent = text;
  el.style.opacity = opacity;
  return el;
}

function makeImageOverlay(): HTMLDivElement {
  const el = document.createElement('div');
  el.classList.add('luxar-overlay--image');
  el.style.opacity = '1';
  const img = document.createElement('img');
  Object.defineProperty(img, 'complete', { value: true, configurable: true });
  Object.defineProperty(img, 'naturalWidth', { value: 100, configurable: true });
  Object.defineProperty(img, 'naturalHeight', { value: 50, configurable: true });
  el.appendChild(img);
  return el;
}

function makeVideoOverlay(): HTMLDivElement {
  const el = document.createElement('div');
  el.classList.add('luxar-overlay--video');
  el.style.opacity = '1';
  const video = document.createElement('video');
  Object.defineProperty(video, 'readyState', {
    value: HTMLMediaElement.HAVE_CURRENT_DATA,
    configurable: true,
  });
  Object.defineProperty(video, 'videoWidth', { value: 400, configurable: true });
  Object.defineProperty(video, 'videoHeight', { value: 300, configurable: true });
  el.appendChild(video);
  return el;
}

function makeConfig(overrides: Partial<OverlayConfig> = {}): OverlayConfig {
  return {
    type: 'text',
    position: [0.5, 0.5],
    anchor: 'center',
    opacity: 1,
    ...overrides,
  } as OverlayConfig;
}

function makeManager(overlays: Array<{ el: HTMLDivElement; config: OverlayConfig }>) {
  return {
    getVisibleOverlays: vi.fn(() => overlays),
  } as unknown as OverlayManager;
}

describe('compositeOverlays', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('no-ops when there are no visible overlays', () => {
    const canvas = makeCanvas();
    const ctx = makeFakeCtx() as unknown as CanvasRenderingContext2D;
    const manager = makeManager([]);
    const glCanvas = makeCanvas();
    expect(() => compositeOverlays(canvas, ctx, manager, glCanvas)).not.toThrow();
    expect(manager.getVisibleOverlays).toHaveBeenCalledTimes(1);
  });

  it('iterates over each visible overlay', () => {
    const canvas = makeCanvas();
    const fake = makeFakeCtx();
    const overlays = [
      { el: makeTextOverlay('A'), config: makeConfig({ position: [0.1, 0.1] }) },
      { el: makeTextOverlay('B'), config: makeConfig({ position: [0.5, 0.5] }) },
    ];
    compositeOverlays(
      canvas,
      fake as unknown as CanvasRenderingContext2D,
      makeManager(overlays),
      makeCanvas()
    );
    expect(fake.save).toHaveBeenCalledTimes(2);
    expect(fake.restore).toHaveBeenCalledTimes(2);
  });

  it('applies element opacity preferentially over config opacity', () => {
    const canvas = makeCanvas();
    const fake = makeFakeCtx();
    const overlay = { el: makeTextOverlay('X', '0.3'), config: makeConfig({ opacity: 1 }) };
    compositeOverlays(
      canvas,
      fake as unknown as CanvasRenderingContext2D,
      makeManager([overlay]),
      makeCanvas()
    );
    expect(fake.globalAlpha).toBeCloseTo(0.3, 5);
  });

  it('maps blend_mode through the BLEND_MODE_TO_COMPOSITE table', () => {
    const canvas = makeCanvas();
    const fake = makeFakeCtx();
    const overlay = {
      el: makeTextOverlay('X'),
      config: makeConfig({ blend_mode: 'multiply' }),
    };
    compositeOverlays(
      canvas,
      fake as unknown as CanvasRenderingContext2D,
      makeManager([overlay]),
      makeCanvas()
    );
    expect(fake.globalCompositeOperation).toBe('multiply');
  });

  it('falls back to source-over for an unknown blend_mode', () => {
    // [P11/M3] BLEND_MODE_TO_COMPOSITE[unknown] is undefined → the `?? 'source-over'`
    // fallback applies. Seed a non-default sentinel so we can distinguish
    // "assignment skipped" (stays sentinel) from "fallback dropped" (undefined).
    const canvas = makeCanvas();
    const fake = makeFakeCtx();
    fake.globalCompositeOperation = 'multiply';
    const overlay = {
      el: makeTextOverlay('X'),
      config: makeConfig({ blend_mode: 'unknown-mode' }),
    };
    compositeOverlays(
      canvas,
      fake as unknown as CanvasRenderingContext2D,
      makeManager([overlay]),
      makeCanvas()
    );
    expect(fake.globalCompositeOperation).toBe('source-over');
  });

  it('draws a visible video frame with its intrinsic aspect ratio', () => {
    const canvas = makeCanvas(1000, 500);
    const fake = makeFakeCtx(1000, 500);
    const overlay = {
      el: makeVideoOverlay(),
      config: makeConfig({ type: 'overlay_video', size: [0.2, null] }),
    };

    compositeOverlays(
      canvas,
      fake as unknown as CanvasRenderingContext2D,
      makeManager([overlay]),
      makeCanvas(),
      { width: 1000, height: 500 }
    );

    expect(fake.drawImage).toHaveBeenCalledTimes(1);
    const [video, x, y, width, height] = fake.drawImage.mock.calls[0];
    expect(video).toBe(overlay.el.querySelector('video'));
    expect([x, y, width, height]).toEqual([400, 175, 200, 150]);
  });
});

describe('compositeTextOverlay', () => {
  it('does nothing for empty text', () => {
    const fake = makeFakeCtx();
    const el = makeTextOverlay('');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig(),
      0,
      0,
      unitMetrics()
    );
    expect(fake.fillText).not.toHaveBeenCalled();
  });

  it('draws background, stroke, and fill in order when configured', () => {
    const fake = makeFakeCtx();
    const el = makeTextOverlay('Hi');
    const config = makeConfig({
      background: '#000',
      stroke_color: '#fff',
      color: '#ff0000',
    });

    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      config,
      100,
      100,
      unitMetrics()
    );

    expect(fake.fillRect).toHaveBeenCalledTimes(1);
    expect(fake.strokeText).toHaveBeenCalledTimes(1);
    expect(fake.fillText).toHaveBeenCalledTimes(1);
    expect(fake.strokeText.mock.invocationCallOrder[0]).toBeGreaterThan(
      fake.fillRect.mock.invocationCallOrder[0]
    );
    expect(fake.fillText.mock.invocationCallOrder[0]).toBeGreaterThan(
      fake.strokeText.mock.invocationCallOrder[0]
    );

    // [P2/W5] Pin the text content AND the anchor-adjusted draw position, not
    // just "fillText was called". With measureText→50, fontSize=0.03×600=18,
    // textHeight=21.6, anchor 'center' → dx=-25, dy=-10.8; xIn/yIn=100,100.
    const [text, drawX, drawY] = fake.fillText.mock.calls[0];
    expect(text).toBe('Hi');
    expect(drawX).toBeCloseTo(75, 5);
    expect(drawY).toBeCloseTo(89.2, 5);
  });

  it('sizes the font from the viewport height, not the capture height', () => {
    // font_size is authored as a vh fraction (the manager writes
    // `font-size: <font_size × 100>vh`), so it must resolve against the
    // viewport scaled into capture pixels — resolving it against the
    // capture frame made overlays shrink or grow with the chosen
    // recording resolution.
    const fake = makeFakeCtx(1920, 1080);
    const el = makeTextOverlay('Hi');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig({ font_size: 0.05 }),
      0,
      0,
      { scaleX: 3, scaleY: 3, vw: 3840, vh: 2160 }
    );
    // 0.05 × 2160 = 108 capture px (5% of the 2160-px viewport span).
    expect(fake.font).toBe('108px system-ui, -apple-system, sans-serif');
  });

  it('wraps text to the configured width instead of drawing one long line', () => {
    // With `width` set the DOM overlay wraps (`width: …vw` +
    // `word-wrap: break-word`); fillText does not, so the capture used to
    // run a single line straight off the frame.
    const fake = makeFakeCtx();
    // 10 px per character makes the wrap points predictable.
    fake.measureText = vi.fn((t: string) => ({ width: t.length * 10 }));
    const el = makeTextOverlay('one two three four');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig({ width: 0.1, anchor: 'top-left' }),
      0,
      0,
      unitMetrics(1000, 600)
    );
    // Box is 0.1 × 1000 = 100 px → 10 characters per line.
    const lines = fake.fillText.mock.calls.map((c) => c[0]);
    expect(lines).toEqual(['one two', 'three four']);
  });

  it('splits a word too long for the box, the way break-word does', () => {
    // The manager sets `word-wrap: break-word` next to the width, so on
    // screen an unbreakable token (a long URL or identifier) is cut
    // rather than allowed to overflow. Leaving it whole ran it off the
    // frame in the capture only.
    const fake = makeFakeCtx();
    fake.measureText = vi.fn((t: string) => ({ width: t.length * 10 }));
    const el = makeTextOverlay('a supercalifragilistic b');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig({ width: 0.1, anchor: 'top-left' }),
      0,
      0,
      unitMetrics(1000, 600)
    );
    // Box is 0.1 × 1000 = 100 px → 10 characters per line.
    const lines = fake.fillText.mock.calls.map((c) => c[0]);
    expect(lines).toEqual(['a', 'supercalif', 'ragilistic', 'b']);
    expect(lines.every((l: string) => l.length <= 10)).toBe(true);
  });

  it('terminates when the wrap box is narrower than a single character', () => {
    // 0.0001 × 1000 = 0.1 px, so even one character overflows and
    // `breakWord` cannot satisfy the width. The loop must still finish —
    // one character per line — rather than spin or throw.
    const fake = makeFakeCtx();
    fake.measureText = vi.fn((t: string) => ({ width: t.length * 10 }));
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      makeTextOverlay('one two three'),
      makeConfig({ width: 0.0001, anchor: 'top-left' }),
      0,
      0,
      unitMetrics(1000, 600)
    );
    // 3 + 3 + 5 characters, each on its own line.
    expect(fake.fillText).toHaveBeenCalledTimes(11);
    const lines = fake.fillText.mock.calls.map((c) => c[0] as string);
    expect(lines.every((l) => l.length === 1)).toBe(true);
  });

  it('honours line_height when anchoring a single line', () => {
    // The DOM box of a single line is line-height tall, so a bottom
    // anchor has to lift by that much — not by a hardcoded 1.2.
    const fake = makeFakeCtx();
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      makeTextOverlay('Hi'),
      makeConfig({ font_size: 0.1, line_height: 2, anchor: 'bottom-left' }),
      0,
      100,
      unitMetrics(1000, 600)
    );
    // fontSize = 0.1 × 600 = 60; block = 60 × 2 = 120; bottom anchor
    // → dy = -120, so the baseline lands at y = 100 - 120.
    const [, , drawY] = fake.fillText.mock.calls[0];
    expect(drawY).toBeCloseTo(-20, 5);
  });

  it('collapses a trailing newline instead of adding an empty line', () => {
    // The overlay manager lays every compositable overlay out as
    // `white-space: normal` (only hover overlays get `pre-line`, and
    // those are excluded from compositing), so on screen a `\n` is just
    // whitespace. Splitting on it added an empty last line to the block,
    // inflating its height by a whole line — enough to lift a
    // bottom-anchored overlay 3.5% of the frame above where the screen
    // has it.
    const measure = (text: string): number => {
      const fake = makeFakeCtx();
      fake.measureText = vi.fn((t: string) => ({ width: t.length * 10 }));
      compositeTextOverlay(
        fake as unknown as CanvasRenderingContext2D,
        makeTextOverlay(text),
        makeConfig({ width: 0.5, background: '#000', anchor: 'top-left' }),
        0,
        0,
        unitMetrics(1000, 600)
      );
      return fake.fillRect.mock.calls[0][3] as number;
    };

    expect(measure('Title\n')).toBe(measure('Title'));
  });

  it('wraps a mid-string newline as a space, the way the DOM does', () => {
    const fake = makeFakeCtx();
    fake.measureText = vi.fn((t: string) => ({ width: t.length * 10 }));
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      makeTextOverlay('one\ntwo three'),
      makeConfig({ width: 0.5, anchor: 'top-left' }),
      0,
      0,
      unitMetrics(1000, 600)
    );
    // Box is 0.5 × 1000 = 500 px → the whole run fits on one line.
    expect(fake.fillText.mock.calls.map((c) => c[0])).toEqual(['one two three']);
  });

  it('keeps single-line text on one fillText call when no width is set', () => {
    const fake = makeFakeCtx();
    fake.measureText = vi.fn((t: string) => ({ width: t.length * 10 }));
    const el = makeTextOverlay('one two three four');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig(),
      0,
      0,
      unitMetrics()
    );
    expect(fake.fillText).toHaveBeenCalledTimes(1);
  });

  it('skips background and stroke when not configured', () => {
    const fake = makeFakeCtx();
    const el = makeTextOverlay('Hi');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig(),
      0,
      0,
      unitMetrics()
    );
    expect(fake.fillRect).not.toHaveBeenCalled();
    expect(fake.strokeText).not.toHaveBeenCalled();
    expect(fake.fillText).toHaveBeenCalledTimes(1);
  });
});

describe('compositeImageOverlay', () => {
  it('skips when img is missing or not complete', () => {
    const fake = makeFakeCtx();
    const el = document.createElement('div');
    el.classList.add('luxar-overlay--image');
    compositeImageOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el as HTMLDivElement,
      makeConfig(),
      0,
      0,
      unitMetrics()
    );
    expect(fake.drawImage).not.toHaveBeenCalled();
  });

  it('draws the image at the requested size when size is configured', () => {
    const fake = makeFakeCtx(1000, 500);
    const el = makeImageOverlay();
    compositeImageOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig({ size: [0.2, 0.4] }),
      100,
      100,
      unitMetrics(1000, 500)
    );
    expect(fake.drawImage).toHaveBeenCalledTimes(1);
    // size = [0.2 * 1000, 0.4 * 500] = [200, 200]
    const args = fake.drawImage.mock.calls[0];
    expect(args[3]).toBe(200);
    expect(args[4]).toBe(200);
    // [P2/W6] Also pin the anchor-adjusted position: anchor 'center' →
    // dx=-100, dy=-100 at 200×200; xIn/yIn=100,100 → drawn at (0, 0).
    expect(args[1]).toBe(0);
    expect(args[2]).toBe(0);
  });

  it('falls back to natural image dimensions when size is not configured', () => {
    const fake = makeFakeCtx();
    const el = makeImageOverlay();
    compositeImageOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig(),
      0,
      0,
      unitMetrics()
    );
    const args = fake.drawImage.mock.calls[0];
    expect(args[3]).toBe(100);
    expect(args[4]).toBe(50);
    // [P2/W6] Natural dims 100×50, anchor 'center' → dx=-50, dy=-25;
    // xIn/yIn=0 → drawn at (-50, -25).
    expect(args[1]).toBe(-50);
    expect(args[2]).toBe(-25);
  });

  it('scales a natural-size image by the capture/CSS pixel ratio', () => {
    // The bug: an <img> with no configured `size` lays out at its natural
    // CSS size on screen, but was drawn at raw natural pixels into the
    // capture — so recording a 720-tall canvas at 4K shrank it 3×.
    const fake = makeFakeCtx(2560, 2160);
    const el = makeImageOverlay();
    compositeImageOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig({ anchor: 'top-left' }),
      0,
      0,
      { scaleX: 3, scaleY: 3, vw: 3840, vh: 2160 }
    );
    const args = fake.drawImage.mock.calls[0];
    expect(args[3]).toBe(300); // 100 CSS px × 3
    expect(args[4]).toBe(150); // 50 CSS px × 3
  });
});

describe('computeOverlayMetrics', () => {
  it('maps CSS pixels to capture pixels and viewport units to capture pixels', () => {
    // Canvas laid out at 640×360 CSS inside a 1280×720 viewport, captured
    // at 1920×1080: 3 capture px per CSS px, and a 1.0 vh fraction spans
    // 720 × 3 = 2160 capture px (twice the frame — which is correct: the
    // canvas only shows half the viewport's height).
    const gl = makeLaidOutCanvas(640, 360);
    const m = computeOverlayMetrics(1920, 1080, gl, { width: 1280, height: 720 });
    expect(m.scaleX).toBeCloseTo(3, 6);
    expect(m.scaleY).toBeCloseTo(3, 6);
    expect(m.vw).toBeCloseTo(3840, 6);
    expect(m.vh).toBeCloseTo(2160, 6);
  });

  it('is the identity when the canvas fills the viewport at capture size', () => {
    const gl = makeLaidOutCanvas(1280, 720);
    const m = computeOverlayMetrics(1280, 720, gl, { width: 1280, height: 720 });
    expect(m).toEqual({ scaleX: 1, scaleY: 1, vw: 1280, vh: 720 });
  });

  it('falls back to the capture size when the canvas has no layout box', () => {
    // Detached / display:none canvas → getBoundingClientRect is all zeros.
    // Dividing by that would give Infinity; fall back to treating the
    // capture as the viewport.
    const m = computeOverlayMetrics(1920, 1080, makeCanvas(), { width: 1280, height: 720 });
    expect(m).toEqual({ scaleX: 1, scaleY: 1, vw: 1920, vh: 1080 });
  });
});

describe('compositeHtmlOverlay', () => {
  function rect(overrides: Partial<DOMRect>): DOMRect {
    return {
      left: 0,
      top: 0,
      right: 0,
      bottom: 0,
      width: 0,
      height: 0,
      x: 0,
      y: 0,
      toJSON: () => ({}),
      ...overrides,
    } as DOMRect;
  }

  it('returns early without drawing when the GL canvas has zero width', () => {
    // [P5/G1] Guard at overlay-compositor.ts:167.
    const fake = makeFakeCtx();
    const el = document.createElement('div');
    el.classList.add('luxar-overlay--html');
    const glCanvas = makeCanvas(800, 600);
    vi.spyOn(glCanvas, 'getBoundingClientRect').mockReturnValue(rect({ width: 0, height: 600 }));
    compositeHtmlOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el as HTMLDivElement,
      glCanvas
    );
    expect(fake.drawImage).not.toHaveBeenCalled();
  });

  it('returns early without drawing when the GL canvas has zero height', () => {
    const fake = makeFakeCtx();
    const el = document.createElement('div');
    el.classList.add('luxar-overlay--html');
    const glCanvas = makeCanvas(800, 600);
    vi.spyOn(glCanvas, 'getBoundingClientRect').mockReturnValue(rect({ width: 800, height: 0 }));
    compositeHtmlOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el as HTMLDivElement,
      glCanvas
    );
    expect(fake.drawImage).not.toHaveBeenCalled();
  });

  it('scales the element position + size by the canvas/glRect ratio and draws', () => {
    // [P5/G1] The whole rasterization path was uncovered (lines 160-222).
    // scaleX = ctx.canvas.width/glRect.width = 1600/800 = 2 (same for Y);
    // x = (200-0)*2 = 400, y = (150-0)*2 = 300, drawW = 100*2 = 200, drawH = 200.
    const fake = makeFakeCtx(1600, 1200);
    const el = document.createElement('div');
    el.classList.add('luxar-overlay--html');
    el.textContent = 'overlay';
    const glCanvas = makeCanvas(800, 600);
    vi.spyOn(glCanvas, 'getBoundingClientRect').mockReturnValue(
      rect({ left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600 })
    );
    vi.spyOn(el, 'getBoundingClientRect').mockReturnValue(
      rect({
        left: 200,
        top: 150,
        width: 100,
        height: 100,
        right: 300,
        bottom: 250,
        x: 200,
        y: 150,
      })
    );

    // Force the synchronous draw branch — jsdom's Image never reports
    // complete from a data URL, so stub a decoded Image.
    const ImageOrig = globalThis.Image;
    class FakeImage {
      complete = true;
      naturalWidth = 1;
      set src(_v: string) {}
      decode(): Promise<void> {
        return Promise.resolve();
      }
    }
    (globalThis as unknown as { Image: unknown }).Image = FakeImage;
    try {
      compositeHtmlOverlay(
        fake as unknown as CanvasRenderingContext2D,
        el as HTMLDivElement,
        glCanvas
      );
      expect(fake.drawImage).toHaveBeenCalledWith(expect.anything(), 400, 300, 200, 200);
    } finally {
      (globalThis as unknown as { Image: unknown }).Image = ImageOrig;
    }
  });

  it('builds a well-formed XML document even with a void element inside', () => {
    // The markup is loaded as `data:image/svg+xml`, which the browser
    // parses as XML. HTML serialization leaves `<br>` and `<img>`
    // unclosed, which is a FATAL well-formedness error there: the Image
    // never loads, `drawImage` never runs, and the overlay is missing
    // from every screenshot and every recorded frame. Both tags are in
    // the overlay HTML allowlist and several shipped demos use `<br>`.
    const fake = makeFakeCtx(1600, 1200);
    const el = document.createElement('div');
    el.classList.add('luxar-overlay--html');
    el.innerHTML = 'Line one<br>Line two<img src="data:image/gif;base64,R0lGOD">';
    const glCanvas = makeCanvas(800, 600);
    vi.spyOn(glCanvas, 'getBoundingClientRect').mockReturnValue(
      rect({ left: 0, top: 0, width: 800, height: 600, right: 800, bottom: 600 })
    );
    vi.spyOn(el, 'getBoundingClientRect').mockReturnValue(
      rect({ left: 0, top: 0, width: 100, height: 100, right: 100, bottom: 100 })
    );

    let captured = '';
    const ImageOrig = globalThis.Image;
    class RecordingImage {
      complete = true;
      naturalWidth = 1;
      set src(value: string) {
        captured = value;
      }
      decode(): Promise<void> {
        return Promise.resolve();
      }
    }
    (globalThis as unknown as { Image: unknown }).Image = RecordingImage;
    try {
      compositeHtmlOverlay(
        fake as unknown as CanvasRenderingContext2D,
        el as HTMLDivElement,
        glCanvas
      );
    } finally {
      (globalThis as unknown as { Image: unknown }).Image = ImageOrig;
    }

    expect(captured.startsWith('data:image/svg+xml;charset=utf-8,')).toBe(true);
    const svg = decodeURIComponent(captured.slice('data:image/svg+xml;charset=utf-8,'.length));
    const parsed = new DOMParser().parseFromString(svg, 'image/svg+xml');
    expect(parsed.querySelector('parsererror')).toBeNull();
  });
});
