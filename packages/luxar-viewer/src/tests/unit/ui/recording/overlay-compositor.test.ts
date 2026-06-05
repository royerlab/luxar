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
} from '../../../../ui/recording-panel/overlay-compositor';
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
      800,
      600
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
      800,
      600
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

  it('skips background and stroke when not configured', () => {
    const fake = makeFakeCtx();
    const el = makeTextOverlay('Hi');
    compositeTextOverlay(
      fake as unknown as CanvasRenderingContext2D,
      el,
      makeConfig(),
      0,
      0,
      800,
      600
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
      800,
      600
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
      1000,
      500
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
      800,
      600
    );
    const args = fake.drawImage.mock.calls[0];
    expect(args[3]).toBe(100);
    expect(args[4]).toBe(50);
    // [P2/W6] Natural dims 100×50, anchor 'center' → dx=-50, dy=-25;
    // xIn/yIn=0 → drawn at (-50, -25).
    expect(args[1]).toBe(-50);
    expect(args[2]).toBe(-25);
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
});
