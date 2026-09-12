// @vitest-environment jsdom
/**
 * Tests for the control panel's DOM.
 *
 * jsdom rather than a real browser: everything asserted here is structure and
 * event wiring, which is exactly what jsdom is good for and what a Playwright
 * run would pay a WebGL context to tell us more slowly.
 */

import { beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import {
  deriveChapters,
  type ChapterSource,
} from '../../../../config/control-panel/derive-chapters';
import {
  CONTROL_COLUMNS_PROPERTY,
  CONTROL_PANEL_CLASS,
  createControlPanel,
  type ControlPanelView,
} from '../../../../ui/control-panel/render-panel';
import type { DimensionMetadata } from '../../../../types/dims';

function dim(name: string, extra: Partial<DimensionMetadata> = {}): DimensionMetadata {
  return { name, unit: '', scale: 1, ...extra };
}

function tourSource(categories: string[]): ChapterSource {
  const source = deriveChapters({
    displayed: [0, 1, 2],
    metadata: [dim('x'), dim('y'), dim('z'), dim('story', { discrete: true, step: 1, categories })],
    ranges: [
      [0, 1],
      [0, 1],
      [0, 1],
      [0, categories.length - 1],
    ],
  });
  if (source === null) throw new Error('fixture failed to derive');
  return source;
}

const STORIES = ['Overview', 'Haemoglobin', 'Hsp70'];

/** The renderer's `call` port, spelled out so a mock matches it exactly. */
type CallPort = (method: string, params: unknown[]) => void;

describe('createControlPanel', () => {
  let root: HTMLElement;
  let call: Mock<CallPort>;
  let panel: ControlPanelView;

  beforeEach(() => {
    document.body.replaceChildren();
    root = document.createElement('div');
    document.body.append(root);
    call = vi.fn<CallPort>();
    panel = createControlPanel({ root, call });
  });

  const tiles = (): HTMLElement[] =>
    Array.from(root.querySelectorAll<HTMLElement>('.luxar-control-tile'));

  it('marks the root so a stylesheet can find it', () => {
    expect(root.classList.contains(CONTROL_PANEL_CLASS)).toBe(true);
    expect(root.dataset.luxarControl).toBe('panel');
  });

  it('draws nothing until render is called', () => {
    // So the page can show a connecting state without flashing an empty grid.
    expect(root.children).toHaveLength(0);
  });

  it('renders one tile per chapter, in order', () => {
    panel.render(tourSource(STORIES));
    // The LABEL, not the tile's whole `textContent`: a tile also carries its
    // position in the tour as a numeral, and asserting on the concatenation
    // would make this test fail for a purely decorative change.
    expect(tiles().map((t) => t.querySelector('.luxar-control-tile-label')?.textContent)).toEqual(
      STORIES
    );
  });

  it('numbers the tiles from one, in order', () => {
    panel.render(tourSource(STORIES));
    expect(tiles().map((t) => t.querySelector('.luxar-control-tile-index')?.textContent)).toEqual([
      '1',
      '2',
      '3',
    ]);
  });

  it('hides the numeral from assistive technology', () => {
    // A screen reader already conveys position from the list; hearing
    // "one, Overview" is worse than hearing "Overview".
    panel.render(tourSource(STORIES));
    for (const tile of tiles()) {
      expect(tile.querySelector('.luxar-control-tile-index')?.getAttribute('aria-hidden')).toBe(
        'true'
      );
    }
  });

  it('renders tiles as real buttons', () => {
    // Focusable, keyboard-reachable, announced — a plinth tablet is still
    // someone's only way in.
    panel.render(tourSource(STORIES));
    for (const tile of tiles()) {
      expect(tile.tagName).toBe('BUTTON');
      expect((tile as HTMLButtonElement).type).toBe('button');
    }
  });

  it('carries the chapter identity on each tile', () => {
    panel.render(tourSource(STORIES));
    expect(tiles().map((t) => t.dataset.chapterIndex)).toEqual(['0', '1', '2']);
    expect(tiles().map((t) => t.dataset.chapterValue)).toEqual(['0', '1', '2']);
  });

  it('says whether a label came from the scene or was generated', () => {
    const source = tourSource(['Overview']);
    source.chapters.push({ index: 1, value: 1, label: 'story 1', authored: false });
    panel.render(source);
    expect(tiles().map((t) => t.dataset.authoredLabel)).toEqual(['true', 'false']);
  });

  it('sets one dimension value per tap, and nothing else', () => {
    panel.render(tourSource(STORIES));
    tiles()[2].click();
    expect(call).toHaveBeenCalledTimes(1);
    expect(call).toHaveBeenCalledWith('setDimensionValue', [3, 2]);
  });

  it('reports the tap so the page can restart its idle timer', () => {
    const onInteraction = vi.fn();
    const withIdle = createControlPanel({ root, call, onInteraction });
    withIdle.render(tourSource(STORIES));
    root.querySelectorAll<HTMLElement>('.luxar-control-tile')[1].click();
    expect(onInteraction).toHaveBeenCalledTimes(1);
  });

  it('renders an optional title and subtitle', () => {
    panel.render(tourSource(STORIES), { title: 'Twelve stories', subtitle: 'Tap one' });
    expect(root.querySelector('.luxar-control-title')?.textContent).toBe('Twelve stories');
    expect(root.querySelector('.luxar-control-subtitle')?.textContent).toBe('Tap one');
  });

  it('omits the heading elements entirely when unset', () => {
    panel.render(tourSource(STORIES));
    expect(root.querySelector('.luxar-control-title')).toBeNull();
    expect(root.querySelector('.luxar-control-subtitle')).toBeNull();
  });

  it('exposes the column count as a custom property', () => {
    panel.render(tourSource(STORIES), { columns: 4 });
    const grid = root.querySelector<HTMLElement>('.luxar-control-grid');
    expect(grid?.style.getPropertyValue(CONTROL_COLUMNS_PROPERTY)).toBe('4');
  });

  it('fits a column count itself when unset or nonsense', () => {
    // The panel must be ONE page with no scrolling, so an absent column count
    // is not "let the stylesheet decide" — `repeat(auto-fill, minmax(...))`
    // cannot fit exactly N cells in a box and overflows once the chapters
    // outgrow the viewport. The renderer measures the grid and commits to a
    // number. Under jsdom every box is 0x0, so this asserts the shape of the
    // answer (a real column count) rather than a specific layout; `fitGrid`'s
    // own tests cover which number comes out for which aspect.
    for (const columns of [undefined, null, 0, -3]) {
      panel.render(tourSource(STORIES), { columns });
      const grid = root.querySelector<HTMLElement>('.luxar-control-grid');
      const fitted = Number(grid?.style.getPropertyValue(CONTROL_COLUMNS_PROPERTY));
      expect(Number.isInteger(fitted)).toBe(true);
      expect(fitted).toBeGreaterThan(0);
      expect(fitted).toBeLessThanOrEqual(STORIES.length);
      expect(grid?.dataset.gridColumns).toBe(String(fitted));
    }
  });

  it('keeps an authored column count instead of re-fitting over it', () => {
    // A pinned number is a deliberate choice; measuring would silently
    // overwrite it on the first resize.
    panel.render(tourSource(STORIES), { columns: 2 });
    const grid = root.querySelector<HTMLElement>('.luxar-control-grid');
    expect(grid?.style.getPropertyValue(CONTROL_COLUMNS_PROPERTY)).toBe('2');
    expect(grid?.dataset.gridColumns).toBeUndefined();
  });

  it('records the chapter count on the grid', () => {
    panel.render(tourSource(STORIES));
    const grid = root.querySelector<HTMLElement>('.luxar-control-grid');
    expect(grid?.dataset.chapterCount).toBe(String(STORIES.length));
  });

  it('renders a per-chapter sublabel when given one', () => {
    panel.render(tourSource(STORIES), { sublabels: { 1: 'the molecule of breath' } });
    expect(tiles()[1].querySelector('.luxar-control-tile-sublabel')?.textContent).toBe(
      'the molecule of breath'
    );
    expect(tiles()[0].querySelector('.luxar-control-tile-sublabel')).toBeNull();
  });

  it('never interprets a label as markup', () => {
    // Chapter labels come from scene attributes, which are untrusted input.
    panel.render(tourSource(['<img src=x onerror=alert(1)>']));
    expect(tiles()[0].querySelector('img')).toBeNull();
    expect(tiles()[0].querySelector('.luxar-control-tile-label')?.textContent).toBe(
      '<img src=x onerror=alert(1)>'
    );
  });

  describe('setActive', () => {
    it('marks exactly one tile', () => {
      panel.render(tourSource(STORIES));
      panel.setActive(1);
      expect(tiles().map((t) => t.dataset.active)).toEqual(['false', 'true', 'false']);
      expect(tiles()[1].getAttribute('aria-current')).toBe('true');
    });

    it('moves the mark rather than accumulating them', () => {
      panel.render(tourSource(STORIES));
      panel.setActive(0);
      panel.setActive(2);
      expect(tiles().filter((t) => t.dataset.active === 'true')).toHaveLength(1);
      expect(tiles()[0].hasAttribute('aria-current')).toBe(false);
    });

    it('clears every mark for -1', () => {
      panel.render(tourSource(STORIES));
      panel.setActive(1);
      panel.setActive(-1);
      expect(tiles().every((t) => t.dataset.active === 'false')).toBe(true);
    });

    it('is a no-op before anything is rendered', () => {
      expect(() => panel.setActive(0)).not.toThrow();
    });
  });

  describe('empty and error states', () => {
    it('says so plainly when the scene has no chapters', () => {
      // Most scenes are not tours; an empty grid would look broken.
      panel.render(null);
      expect(root.querySelector('.luxar-control-message')?.textContent).toContain('no chapters');
      expect(tiles()).toHaveLength(0);
    });

    it('treats a chapter-less source the same way', () => {
      panel.render({ dimensionIndex: 3, dimensionName: 'story', chapters: [] });
      expect(root.querySelector('.luxar-control-message')).not.toBeNull();
    });

    it('shows a message with its own heading', () => {
      panel.showMessage('No control hub', 'Run luxar serve --control');
      expect(root.querySelector('.luxar-control-title')?.textContent).toBe('No control hub');
      expect(root.querySelector('.luxar-control-message')?.textContent).toContain(
        'luxar serve --control'
      );
    });

    it('replaces tiles when a message arrives', () => {
      panel.render(tourSource(STORIES));
      panel.showMessage('Disconnected', 'Reconnecting...');
      expect(tiles()).toHaveLength(0);
    });
  });

  describe('re-render and teardown', () => {
    it('rebuilds rather than appending a second grid', () => {
      panel.render(tourSource(STORIES));
      panel.render(tourSource(STORIES));
      expect(root.querySelectorAll('.luxar-control-grid')).toHaveLength(1);
      expect(tiles()).toHaveLength(3);
    });

    it('drops the old tiles listeners, so a redraw cannot double-fire', () => {
      // A kiosk redraws on every reconnect; leaked listeners would send one
      // extra call per redraw, forever.
      panel.render(tourSource(STORIES));
      const stale = tiles()[0];
      panel.render(tourSource(STORIES));
      stale.click();
      expect(call).not.toHaveBeenCalled();
      tiles()[0].click();
      expect(call).toHaveBeenCalledTimes(1);
    });

    it('leaves the root clean on dispose', () => {
      panel.render(tourSource(STORIES));
      panel.dispose();
      expect(root.children).toHaveLength(0);
      expect(root.classList.contains(CONTROL_PANEL_CLASS)).toBe(false);
      expect(root.dataset.luxarControl).toBeUndefined();
    });

    it('stops responding to taps after dispose', () => {
      panel.render(tourSource(STORIES));
      const tile = tiles()[0];
      panel.dispose();
      tile.click();
      expect(call).not.toHaveBeenCalled();
    });
  });
});
