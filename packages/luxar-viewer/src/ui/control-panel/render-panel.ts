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
 * | `.luxar-control-title` / `.luxar-control-subtitle` | heading text |
 * | `.luxar-control-grid` | the tile container |
 * | `.luxar-control-tile` | one chapter |
 * | `.luxar-control-tile-label` / `-sublabel` | text inside a tile |
 * | `.luxar-control-message` | the no-tiles state |
 * | `[data-active="true"]` | the tile matching the viewer's live position |
 * | `[data-authored-label]` | `"true"` when the label came from the scene |
 * | `[data-chapter-index]` / `[data-chapter-value]` | which stop a tile is |
 * | `--luxar-control-columns` | grid column count |
 *
 * A lock test pins these names, because a rename would silently break every
 * authored stylesheet in the field.
 */

import type { ChapterSource } from '../../config/control-panel/derive-chapters';
import { EventGroup } from '../../utils/cross-layer/event-group';

/** CSS class prefix for everything this module creates. */
export const CONTROL_PANEL_CLASS = 'luxar-control-panel';
/** Custom property carrying the grid's column count. */
export const CONTROL_COLUMNS_PROPERTY = '--luxar-control-columns';

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
  /** Grid columns. Unset lets the stylesheet decide from the viewport. */
  columns?: number | null;
  /** Per-chapter extra line, keyed by chapter index. */
  sublabels?: Record<number, string>;
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
  for (const chapter of source.chapters) {
    const tile = element('button', 'luxar-control-tile');
    tile.type = 'button';
    tile.dataset.chapterIndex = String(chapter.index);
    tile.dataset.chapterValue = String(chapter.value);
    tile.dataset.authoredLabel = String(chapter.authored);
    tile.dataset.active = 'false';
    tile.append(element('span', 'luxar-control-tile-label', chapter.label));
    const sublabel = options.sublabels?.[chapter.index];
    if (sublabel) tile.append(element('span', 'luxar-control-tile-sublabel', sublabel));
    events.on(tile, 'click', () => onPick(chapter.index));
    grid.append(tile);
  }
  return grid;
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

  const reset = (): void => {
    events.dispose();
    events = new EventGroup();
    root.replaceChildren();
  };

  const header = (options: ControlPanelOptions): HTMLElement[] => {
    const nodes: HTMLElement[] = [];
    if (options.title) nodes.push(element('h1', 'luxar-control-title', options.title));
    if (options.subtitle) {
      nodes.push(element('p', 'luxar-control-subtitle', options.subtitle));
    }
    return nodes;
  };

  return {
    render(next, options = {}) {
      reset();
      source = next;
      const nodes = header(options);
      if (next === null || next.chapters.length === 0) {
        nodes.push(
          element('div', 'luxar-control-message', 'This scene has no chapters to jump to.')
        );
      } else {
        nodes.push(
          buildGrid(
            next,
            options,
            (chapterIndex) => {
              const chapter = next.chapters[chapterIndex];
              if (chapter === undefined) return;
              ports.call('setDimensionValue', [next.dimensionIndex, chapter.value]);
              ports.onInteraction?.();
            },
            events
          )
        );
      }
      root.append(...nodes);
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
