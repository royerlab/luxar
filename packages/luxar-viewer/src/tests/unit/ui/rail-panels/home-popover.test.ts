/**
 * Unit tests for ui/rail-panels/home-popover.ts.
 *
 * The Home popover is the one rail-panel builder with NO nested GUI — a plain
 * vertical stack of four action rows (label + hint). These tests verify the
 * rendered rows, that each click fires its context callback plus a render
 * request, the live disabled state of "Reset dimensions", and that the builder
 * adds no listeners outside its host (teardown is trivially safe).
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
    triggerAnimation: vi.fn(),
    ...overrides,
  };
}

function rows(host: HTMLElement): HTMLButtonElement[] {
  return Array.from(host.querySelectorAll<HTMLButtonElement>('.luxar-control-rail__action'));
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe('buildHomePopover', () => {
  it('renders exactly four action rows with the expected labels and hints', () => {
    const host = document.createElement('div');
    buildHomePopover(host, makeCtx());

    const actionRows = rows(host);
    expect(actionRows).toHaveLength(4);
    expect(
      actionRows.map((r) => r.querySelector('.luxar-control-rail__action-label')?.textContent)
    ).toEqual(['Fit scene', 'Center on origin', 'Reset dimensions', 'Reset rendering']);
    // Every row carries a muted hint line.
    for (const row of actionRows) {
      expect(row.querySelector('.luxar-control-rail__action-hint')?.textContent).toBeTruthy();
    }
  });

  it.each([
    ['Fit scene', 0, 'fitScene'],
    ['Center on origin', 1, 'centerOnOrigin'],
    ['Reset dimensions', 2, 'resetDimensions'],
    ['Reset rendering', 3, 'resetRendering'],
  ] as const)('clicking "%s" fires %s once plus a render request', (_label, index, fn) => {
    const ctx = makeCtx();
    const host = document.createElement('div');
    buildHomePopover(host, ctx);

    rows(host)[index].click();

    expect(ctx[fn]).toHaveBeenCalledTimes(1);
    expect(ctx.triggerAnimation).toHaveBeenCalledTimes(1);
  });

  it('clicking one row does not fire the other actions', () => {
    const ctx = makeCtx();
    const host = document.createElement('div');
    buildHomePopover(host, ctx);

    rows(host)[0].click();

    expect(ctx.fitScene).toHaveBeenCalledTimes(1);
    expect(ctx.centerOnOrigin).not.toHaveBeenCalled();
    expect(ctx.resetDimensions).not.toHaveBeenCalled();
    expect(ctx.resetRendering).not.toHaveBeenCalled();
  });

  describe('Reset dimensions disabled state', () => {
    it('is disabled (native attribute + explanatory title) when the scene has no slider dimensions', () => {
      const ctx = makeCtx({ hasDimensionSliders: vi.fn(() => false) });
      const host = document.createElement('div');
      buildHomePopover(host, ctx);

      const dimsRow = rows(host)[2];
      expect(dimsRow.disabled).toBe(true);
      expect(dimsRow.title).toBe('No dimension sliders in this scene');

      // A disabled row must not fire even if forced.
      dimsRow.click();
      expect(ctx.resetDimensions).not.toHaveBeenCalled();
      expect(ctx.triggerAnimation).not.toHaveBeenCalled();
    });

    it('is enabled when the scene has dimensions (other rows unaffected either way)', () => {
      const ctx = makeCtx({ hasDimensionSliders: vi.fn(() => true) });
      const host = document.createElement('div');
      buildHomePopover(host, ctx);

      const actionRows = rows(host);
      expect(actionRows[2].disabled).toBe(false);
      expect(actionRows.filter((r) => r.disabled)).toHaveLength(0);
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
