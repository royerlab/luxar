/**
 * Unit tests for ui/rail-panels/home-popover.ts.
 *
 * The Home popover is a horizontal row of five square icon chips (the View
 * flyout's aesthetic) with a live caption strip that names + explains the
 * hovered/focused action. These tests verify the rendered chips, that each
 * click fires its context callback plus a render request, the live disabled
 * states (Reset dimensions / Reset layers), the caption's hover/focus/idle
 * behavior, and that the builder adds no listeners outside its host.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { buildHomePopover, type HomePopoverContext } from '../../../../ui/rail-panels/home-popover';

function makeCtx(overrides: Partial<HomePopoverContext> = {}): HomePopoverContext {
  return {
    fitScene: vi.fn(),
    centerOnOrigin: vi.fn(),
    resetDimensions: vi.fn(),
    hasDimensionSliders: vi.fn(() => true),
    resetRendering: vi.fn(),
    resetLayers: vi.fn(),
    hasLayers: vi.fn(() => true),
    triggerAnimation: vi.fn(),
    ...overrides,
  };
}

function chips(host: HTMLElement): HTMLButtonElement[] {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__home-chip'));
}

function caption(host: HTMLElement): { label: string; hint: string } {
  return {
    label: host.querySelector('.luxar-control-rail__home-caption-label')?.textContent ?? '',
    hint: host.querySelector('.luxar-control-rail__home-caption-hint')?.textContent ?? '',
  };
}

const EXPECTED_LABELS = [
  'Fit scene',
  'Center on origin',
  'Reset dimensions',
  'Reset rendering',
  'Reset layers',
];

afterEach(() => {
  vi.restoreAllMocks();
});

describe('buildHomePopover', () => {
  it('renders five icon chips (each with an svg + accessible label) and a caption strip', () => {
    const host = document.createElement('div');
    buildHomePopover(host, makeCtx());

    const actionChips = chips(host);
    expect(actionChips).toHaveLength(5);
    expect(actionChips.map((c) => c.getAttribute('aria-label')?.split(' — ')[0])).toEqual(
      EXPECTED_LABELS
    );
    for (const chip of actionChips) {
      expect(chip.querySelector('svg')).not.toBeNull();
    }
    // Idle caption is present from the start (so the popover never resizes).
    expect(caption(host).label).toBe('Home');
    expect(caption(host).hint).toBeTruthy();
  });

  it.each([
    ['Fit scene', 0, 'fitScene'],
    ['Center on origin', 1, 'centerOnOrigin'],
    ['Reset dimensions', 2, 'resetDimensions'],
    ['Reset rendering', 3, 'resetRendering'],
    ['Reset layers', 4, 'resetLayers'],
  ] as const)('clicking "%s" fires %s once plus a render request', (_label, index, fn) => {
    const ctx = makeCtx();
    const host = document.createElement('div');
    buildHomePopover(host, ctx);

    chips(host)[index].click();

    expect(ctx[fn]).toHaveBeenCalledTimes(1);
    expect(ctx.triggerAnimation).toHaveBeenCalledTimes(1);
  });

  it('clicking one chip does not fire the other actions', () => {
    const ctx = makeCtx();
    const host = document.createElement('div');
    buildHomePopover(host, ctx);

    chips(host)[0].click();

    expect(ctx.fitScene).toHaveBeenCalledTimes(1);
    expect(ctx.centerOnOrigin).not.toHaveBeenCalled();
    expect(ctx.resetDimensions).not.toHaveBeenCalled();
    expect(ctx.resetRendering).not.toHaveBeenCalled();
    expect(ctx.resetLayers).not.toHaveBeenCalled();
  });

  describe('caption behavior', () => {
    it("shows a chip's label + hint on hover and returns to idle on leave", () => {
      const host = document.createElement('div');
      buildHomePopover(host, makeCtx());
      const [fitChip] = chips(host);

      fitChip.dispatchEvent(new MouseEvent('mouseenter'));
      expect(caption(host).label).toBe('Fit scene');
      expect(caption(host).hint).toBe('Frame all visible geometry (F)');

      host
        .querySelector('.luxar-control-rail__home-chips')!
        .dispatchEvent(new MouseEvent('mouseleave'));
      expect(caption(host).label).toBe('Home');
    });

    it("shows a chip's label + hint on keyboard focus", () => {
      const host = document.createElement('div');
      document.body.appendChild(host);
      buildHomePopover(host, makeCtx());

      chips(host)[1].focus();
      chips(host)[1].dispatchEvent(new FocusEvent('focus'));
      expect(caption(host).label).toBe('Center on origin');
      host.remove();
    });
  });

  describe('disabled states', () => {
    it('disables Reset dimensions (native attribute + explanatory title) with no slider dims', () => {
      const ctx = makeCtx({ hasDimensionSliders: vi.fn(() => false) });
      const host = document.createElement('div');
      buildHomePopover(host, ctx);

      const dimsChip = chips(host)[2];
      expect(dimsChip.disabled).toBe(true);
      expect(dimsChip.title).toBe('No dimension sliders in this scene');

      // A disabled chip must not fire even if forced.
      dimsChip.click();
      expect(ctx.resetDimensions).not.toHaveBeenCalled();
      expect(ctx.triggerAnimation).not.toHaveBeenCalled();
    });

    it('disables Reset layers (with explanatory title) when the scene has no layers', () => {
      const ctx = makeCtx({ hasLayers: vi.fn(() => false) });
      const host = document.createElement('div');
      buildHomePopover(host, ctx);

      const layersChip = chips(host)[4];
      expect(layersChip.disabled).toBe(true);
      expect(layersChip.title).toBe('No layers in this scene');
      layersChip.click();
      expect(ctx.resetLayers).not.toHaveBeenCalled();
    });

    it('enables every chip when all predicates are true', () => {
      const host = document.createElement('div');
      buildHomePopover(host, makeCtx());
      expect(chips(host).filter((c) => c.disabled)).toHaveLength(0);
    });
  });

  it('adds no window/document listeners (all state lives in the host DOM)', () => {
    const winSpy = vi.spyOn(window, 'addEventListener');
    const docSpy = vi.spyOn(document, 'addEventListener');

    const host = document.createElement('div');
    const teardown = buildHomePopover(host, makeCtx());

    expect(winSpy).not.toHaveBeenCalled();
    expect(docSpy).not.toHaveBeenCalled();
    expect(() => teardown()).not.toThrow();
  });
});
