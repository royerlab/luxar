/**
 * The control panel's DOM: a full-screen grid of chapter tiles.
 *
 * Built for a tablet on a plinth. Every tile is a real `<button>` so it is
 * focusable, keyboard-reachable and announced; the grid is the whole viewport
 * because the panel has one job and no scene to share space with.
 *
 * Port-injected rather than importing the app, and not by preference — the
 * layering contract forbids `ui` from importing `core`, so the only way for
 * this module to drive a viewer is through a `call` port handed in from the
 * page's bootstrap. That constraint is doing useful work: the whole renderer is
 * testable under jsdom with a fake `call`, and it cannot quietly grow a
 * dependency on the viewer's internals.
 *
 * ## Styling contract
 *
 * The *documented* surface an author styles against, and the reason there is no
 * `layout` enum in the authored block: CSS is the general answer, and an enum
 * would accrete `font_size`, `aspect`, `padding` forever.
 *
 * | Hook | Meaning |
 * |---|---|
 * | `.luxar-control-panel` | the root |
 * | `.luxar-control-header` | the one-line header bar |
 * | `.luxar-control-title` / `.luxar-control-subtitle` | heading text |
 * | `.luxar-control-tile-index` | the tile's position in the tour |
 * | `.luxar-control-grid` | the tile container |
 * | `.luxar-control-tile` | one chapter |
 * | `.luxar-control-tile-label` / `-sublabel` | text inside a tile |
 * | `.luxar-control-message` | the no-tiles state |
 * | `[data-active="true"]` | the tile matching the viewer's live position |
 * | `[data-authored-label]` | `"true"` when the label came from the scene |
 * | `[data-chapter-index]` / `[data-chapter-value]` | which stop a tile is |
 * | `--luxar-control-columns` | grid column count |
 * | `[data-chapter-count]` | how many tiles the grid holds |
 * | `[data-grid-columns]` | the fitted column count, mirrored for tests/CSS |
 * | `[data-variant]` | `"full"` or `"short"` on a label or sublabel |
 * | `[data-short-label]` / `[data-short-sublabel]` | `"true"` on a tile with a short text |
 * | `[data-label-text]` / `[data-sublabel-text]` | on the grid: the version shown |
 * | `[data-tile-index]` | on the grid: `"shown"` or `"hidden"` |
 * | `[data-label-wrap]` | on the grid: `"words"`, or `"anywhere"` as a last resort |
 * | `--luxar-control-sublabel-lines` | on the grid: a sublabel's line limit |
 * | `--luxar-control-label-scale` | on the grid: multiplier on the label size |
 *
 * The grid-level hooks are the tile-text fit (`fit-tile-text.ts`): they say
 * how much of each tile's text the screen had room for, so an authored
 * stylesheet can follow the same decision rather than fight it.
 *
 * A lock test pins these names, because a rename would silently break every
 * authored stylesheet in the field.
 */

import type { ChapterSource } from '../../config/control-panel/derive-chapters';
import { rankGrids } from '../../config/control-panel/fit-grid';
import { tileTextLadder, type TileTextStep } from '../../config/control-panel/tile-text-fit';
import { fitTileText } from './fit-tile-text';
import { EventGroup } from '../../utils/cross-layer/event-group';

/** CSS class prefix for everything this module creates. */
export const CONTROL_PANEL_CLASS = 'luxar-control-panel';
/** Custom property carrying the grid's column count. */
export const CONTROL_COLUMNS_PROPERTY = '--luxar-control-columns';
/**
 * How many of the best-shaped grids the text fit may choose between. Past the
 * third the cells are far from the preferred shape, and a tile that keeps one
 * more line of text is not worth a grid that looks wrong.
 */
export const CONTROL_GRID_TEXT_CANDIDATES = 3;

/** What the renderer needs from the page around it. */
export interface ControlPanelPorts {
  /** Element to fill. The page passes `document.body`. */
  root: HTMLElement;
  /**
   * Invoke a viewer method. Fire-and-forget: a tap must feel instant, and the
   * authoritative answer arrives as a `dimensions-changed` event anyway.
   */
  call: (method: string, params: unknown[]) => void;
  /** Called after any tap, so the page can restart its idle timer. */
  onInteraction?: () => void;
}

/** Presentation knobs. Authored ones arrive here in a later stage. */
export interface ControlPanelOptions {
  title?: string;
  subtitle?: string;
  /**
   * Grid columns. Unset means "fit the container": the renderer measures the
   * grid and picks a count so the tiles form a full-page matrix. An authored
   * number pins it and the measurement is skipped.
   */
  columns?: number | null;
  /** Per-chapter extra line, keyed by chapter index. */
  sublabels?: Record<number, string>;
  /**
   * Shorter versions, keyed the same way, shown instead of the full text when
   * a tile is too small for it (`config/control-panel/tile-text-fit.ts`).
   */
  shortLabels?: Record<number, string>;
  shortSublabels?: Record<number, string>;
}

export interface ControlPanelView {
  /** Draw, or redraw, from scratch. Safe to call repeatedly. */
  render(source: ChapterSource | null, options?: ControlPanelOptions): void;
  /** Mark one tile active; `-1` clears. */
  setActive(index: number): void;
  /** Replace the tiles with a message — no hub, or no chapters. */
  showMessage(heading: string, detail: string): void;
  dispose(): void;
}

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  // textContent, never innerHTML: chapter labels come from scene attributes,
  // which are untrusted input by the same reasoning as `overlay_html`.
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * Append a tile's label or sublabel, with its short version beside it.
 *
 * Both are in the DOM and the stylesheet shows one, so switching between them
 * is an attribute on the grid rather than a rewrite of every tile — which is
 * what lets the text fit try a step with one layout. The hidden one is
 * `display: none`, so a screen reader hears only the version on screen.
 */
function appendText(
  tile: HTMLElement,
  role: 'label' | 'sublabel',
  full: string,
  short: string | undefined
): void {
  const node = element('span', `luxar-control-tile-${role}`, full);
  node.dataset.variant = 'full';
  tile.append(node);
  if (!short || short === full) return;
  const shortNode = element('span', `luxar-control-tile-${role}`, short);
  shortNode.dataset.variant = 'short';
  tile.append(shortNode);
  tile.dataset[role === 'label' ? 'shortLabel' : 'shortSublabel'] = 'true';
}

/** The text ladder for what this grid's tiles actually carry. */
function ladderFor(source: ChapterSource, options: ControlPanelOptions): TileTextStep[] {
  const has = (map: Record<number, string> | undefined): boolean =>
    source.chapters.some((chapter) => Boolean(map?.[chapter.index]));
  return tileTextLadder({
    sublabels: has(options.sublabels),
    shortSublabels: has(options.sublabels) && has(options.shortSublabels),
    shortLabels: has(options.shortLabels),
  });
}

/** Build the tile grid for `source`. */
function buildGrid(
  source: ChapterSource,
  options: ControlPanelOptions,
  onPick: (chapterIndex: number) => void,
  events: EventGroup
): HTMLElement {
  const grid = element('div', 'luxar-control-grid');
  if (options.columns != null && options.columns > 0) {
    grid.style.setProperty(CONTROL_COLUMNS_PROPERTY, String(options.columns));
  }
  grid.dataset.chapterCount = String(source.chapters.length);
  const total = source.chapters.length;
  for (const [ordinal, chapter] of source.chapters.entries()) {
    const tile = element('button', 'luxar-control-tile');
    tile.type = 'button';
    tile.dataset.chapterIndex = String(chapter.index);
    tile.dataset.chapterValue = String(chapter.value);
    tile.dataset.authoredLabel = String(chapter.authored);
    tile.dataset.active = 'false';
    // A tour has an order, and a visitor wants to know where in it they are.
    // `aria-hidden` because the numeral is a visual cue only — a screen reader
    // already gets position from the list, and hearing "zero one, Overview"
    // is worse than hearing "Overview".
    const numeral = element(
      'span',
      'luxar-control-tile-index',
      String(ordinal + 1).padStart(total >= 10 ? 2 : 1, '0')
    );
    numeral.setAttribute('aria-hidden', 'true');
    tile.append(numeral);
    appendText(tile, 'label', chapter.label, options.shortLabels?.[chapter.index]);
    const sublabel = options.sublabels?.[chapter.index];
    if (sublabel) {
      appendText(tile, 'sublabel', sublabel, options.shortSublabels?.[chapter.index]);
    }
    events.on(tile, 'click', () => onPick(chapter.index));
    grid.append(tile);
  }
  return grid;
}

/** Commit a column count, and re-centre the last row for it. */
function applyColumns(grid: HTMLElement, count: number, columns: number): void {
  grid.style.setProperty(CONTROL_COLUMNS_PROPERTY, String(columns));
  grid.dataset.gridColumns = String(columns);
  centreFinalRow(grid, count, columns);
}

/**
 * Size the grid to its own box so the tiles form one full page, then fit
 * the text into the tiles.
 *
 * Measured rather than expressed in CSS because `repeat(auto-fill,
 * minmax(...))` cannot say "fit exactly N cells in this box" — it fills from
 * a minimum width and overflows once N outgrows the viewport, which is the
 * scrolling this page exists to avoid.
 *
 * The best-SHAPED grid is tried first and kept whenever its tiles hold all
 * their text, which is every roomy screen. Only when text has to give way
 * are the next shapes tried, and the one that gives up least wins (ties to
 * the better shape). On a small phone that is the difference between four
 * columns of titles broken mid-word and three columns of whole ones.
 */
function fitLayout(grid: HTMLElement, count: number, ladder: readonly TileTextStep[]): void {
  const box = grid.getBoundingClientRect();
  const candidates = rankGrids(count, box.width / box.height).slice(
    0,
    CONTROL_GRID_TEXT_CANDIDATES
  );
  let best = { columns: 0, step: Number.POSITIVE_INFINITY };
  let applied = 0;
  for (const { columns } of candidates) {
    applyColumns(grid, count, columns);
    applied = columns;
    const step = fitTileText(grid, ladder);
    if (step < best.step) best = { columns, step };
    if (step === 0) break;
  }
  if (best.columns !== applied) {
    applyColumns(grid, count, best.columns);
    fitTileText(grid, ladder);
  }
}

/**
 * Nudge the last row so a partial one sits centred under the full ones.
 *
 * Eleven chapters in four columns leave one hole, and a hole at the end of
 * the bottom row reads as a mistake on a screen a visitor is looking at.
 *
 * The grid is laid out in HALF-tile tracks (see the stylesheet) precisely so
 * this is expressible: centring an odd leftover needs a half-tile shift,
 * which whole columns cannot describe. Each tile spans two tracks, so the
 * first tile of the last row starts `leftover` tracks in — one track per
 * half-tile — and the row ends up symmetric.
 */
function centreFinalRow(grid: HTMLElement, count: number, columns: number): void {
  const tiles = Array.from(grid.querySelectorAll<HTMLElement>('.luxar-control-tile'));
  // Always clear first: a re-fit to a different column count changes both
  // which tile begins the last row and how far it should move.
  for (const tile of tiles) tile.style.removeProperty('grid-column');
  const remainder = count % columns;
  if (remainder === 0) return;
  const leftover = columns - remainder;
  const firstOfLastRow = tiles[count - remainder];
  if (firstOfLastRow === undefined) return;
  // `grid-column`, not `grid-column-start`. Writing only the start resets
  // the span to 1 — the stylesheet's `span 2` lives in the same shorthand —
  // so the centred tile came out half the width of every other one. The
  // span has to be restated here.
  firstOfLastRow.style.setProperty('grid-column', `${1 + leftover} / span 2`);
}

/**
 * Create the panel.
 *
 * Nothing is drawn until `render` is called, so the page can show a connecting
 * state first and never flash an empty grid.
 */
export function createControlPanel(ports: ControlPanelPorts): ControlPanelView {
  const root = ports.root;
  root.classList.add(CONTROL_PANEL_CLASS);
  root.dataset.luxarControl = 'panel';

  // One group per render: tiles are rebuilt wholesale, and their listeners must
  // go with them or a long-running kiosk accumulates one set per redraw.
  let events = new EventGroup();
  let source: ChapterSource | null = null;
  let grid: HTMLElement | null = null;

  const reset = (): void => {
    events.dispose();
    events = new EventGroup();
    root.replaceChildren();
    grid = null;
  };

  /**
   * Re-fit whenever the grid's box changes: the columns, then the text.
   *
   * A ResizeObserver rather than a `window.resize` listener: a tablet rotating
   * fires resize, but so does the on-screen keyboard, a browser-chrome change
   * and a safe-area shift — and none of those are guaranteed to have laid the
   * grid out by the time the handler runs. Observing the element itself asks
   * the question that actually matters. Guarded because jsdom has no
   * ResizeObserver, and the renderer's tests run there.
   */
  const observeFit = (grid: HTMLElement, layout: () => void): void => {
    layout();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(layout);
    observer.observe(grid);
    events.add(() => observer.disconnect());
  };

  /**
   * The header, as one element or none.
   *
   * Wrapped rather than returned as loose siblings so the stylesheet can put
   * the scene name and the touch hint on a single line — two children of the
   * panel's column layout would stack, and stacking costs a band of tile area
   * for no gain.
   */
  const header = (options: ControlPanelOptions): HTMLElement[] => {
    if (!options.title && !options.subtitle) return [];
    const bar = element('header', 'luxar-control-header');
    if (options.title) bar.append(element('h1', 'luxar-control-title', options.title));
    if (options.subtitle) {
      bar.append(element('p', 'luxar-control-subtitle', options.subtitle));
    }
    return [bar];
  };

  return {
    render(next, options = {}) {
      reset();
      source = next;
      // Captured before the branch: the fit call below runs after `append`,
      // where TypeScript can no longer narrow `next` for us.
      const chapterCount = next?.chapters.length ?? 0;
      const nodes = header(options);
      if (next === null || next.chapters.length === 0) {
        nodes.push(
          element('div', 'luxar-control-message', 'This scene has no chapters to jump to.')
        );
      } else {
        grid = buildGrid(
          next,
          options,
          (chapterIndex) => {
            const chapter = next.chapters[chapterIndex];
            if (chapter === undefined) return;
            ports.call('setDimensionValue', [next.dimensionIndex, chapter.value]);
            ports.onInteraction?.();
          },
          events
        );
        nodes.push(grid);
      }
      root.append(...nodes);
      // AFTER append, or the grid has no box to measure yet. An author who
      // pinned `columns` keeps it: their number is already on the element and
      // re-fitting would overwrite a deliberate choice. The text is fitted
      // either way, and after the columns, since they decide the tile size.
      if (grid !== null && next !== null && chapterCount > 0) {
        const fitted = grid;
        const pinned = options.columns != null && options.columns > 0;
        const ladder = ladderFor(next, options);
        observeFit(fitted, () => {
          if (pinned) fitTileText(fitted, ladder);
          else fitLayout(fitted, chapterCount, ladder);
        });
      }
    },

    setActive(index) {
      if (source === null) return;
      // `Array.from` rather than `for...of`: this package's lib config has DOM
      // but not DOM.Iterable, so a NodeList is not typed as iterable.
      for (const tile of Array.from(root.querySelectorAll<HTMLElement>('.luxar-control-tile'))) {
        const isActive = tile.dataset.chapterIndex === String(index);
        tile.dataset.active = String(isActive);
        // `aria-current` rather than `aria-selected`: these are navigation
        // targets, not options in a listbox.
        if (isActive) tile.setAttribute('aria-current', 'true');
        else tile.removeAttribute('aria-current');
      }
    },

    showMessage(heading, detail) {
      reset();
      source = null;
      root.append(
        element('h1', 'luxar-control-title', heading),
        element('div', 'luxar-control-message', detail)
      );
    },

    dispose() {
      events.dispose();
      root.replaceChildren();
      root.classList.remove(CONTROL_PANEL_CLASS);
      delete root.dataset.luxarControl;
      source = null;
    },
  };
}
