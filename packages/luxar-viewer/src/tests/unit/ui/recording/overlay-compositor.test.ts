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
} from '../../../../ui/recording/overlay-compositor';
import type { OverlayManager } from '../../../../ui/overlay-manager';
import type { OverlayConfig } from '../../../../data/loaders/overlay-loader';

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
    compositeOverlays(canvas, fake as unknown as CanvasRenderingContext2D, makeManager(overlays), makeCanvas());
    expect(fake.save).toHaveBeenCalledTimes(2);
    expect(fake.restore).toHaveBeenCalledTimes(2);
  });

  it('applies element opacity preferentially over config opacity', () => {
    const canvas = makeCanvas();
    const fake = makeFakeCtx();
    const overlay = { el: makeTextOverlay('X', '0.3'), config: makeConfig({ opacity: 1 }) };
    compositeOverlays(canvas, fake as unknown as CanvasRenderingContext2D, makeManager([overlay]), makeCanvas());
    expect(fake.globalAlpha).toBeCloseTo(0.3, 5);
  });

  it('maps blend_mode through the BLEND_MODE_TO_COMPOSITE table', () => {
    const canvas = makeCanvas();
    const fake = makeFakeCtx();
    const overlay = {
      el: makeTextOverlay('X'),
      config: makeConfig({ blend_mode: 'multiply' }),
    };
    compositeOverlays(canvas, fake as unknown as CanvasRenderingContext2D, makeManager([overlay]), makeCanvas());
    expect(fake.globalCompositeOperation).toBe('multiply');
  });
});

describe('compositeTextOverlay', () => {
  it('does nothing for empty text', () => {
    const fake = makeFakeCtx();
    const el = makeTextOverlay('');
    compositeTextOverlay(fake as unknown as CanvasRenderingContext2D, el, makeConfig(), 0, 0, 800, 600);
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
  });

  it('skips background and stroke when not configured', () => {
    const fake = makeFakeCtx();
    const el = makeTextOverlay('Hi');
    compositeTextOverlay(fake as unknown as CanvasRenderingContext2D, el, makeConfig(), 0, 0, 800, 600);
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
  });
});
