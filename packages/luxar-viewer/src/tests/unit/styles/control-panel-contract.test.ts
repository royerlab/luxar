// @vitest-environment jsdom
/**
 * The control panel's styling contract, pinned from both sides.
 *
 * A scene may ship its own stylesheet for the kiosk panel, which makes these
 * class names, data attributes and custom properties a **public API** — one
 * that lives in other people's files, where a rename here breaks them silently
 * and at a distance. So this test asserts the same names twice: that the
 * renderer emits them into the DOM, and that the shipped stylesheet targets
 * them. Renaming either side alone fails the build.
 *
 * It is also the reason the authored block has no `layout` enum: CSS is the
 * general answer, and a presentational enum beside it would only ever accrete
 * (`font_size`, `aspect`, `columns`, `padding`…).
 */

import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import { deriveChapters } from '../../../config/control-panel/derive-chapters';
import {
  CONTROL_COLUMNS_PROPERTY,
  CONTROL_PANEL_CLASS,
  createControlPanel,
} from '../../../ui/control-panel/render-panel';

const STYLESHEET = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../styles/control-panel.css'
);
const CONTROL_HTML = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../control.html');

/** Every class an authored stylesheet is invited to target. */
const CONTRACT_CLASSES = [
  'luxar-control-panel',
  'luxar-control-header',
  'luxar-control-title',
  'luxar-control-subtitle',
  'luxar-control-tile-index',
  'luxar-control-grid',
  'luxar-control-tile',
  'luxar-control-tile-label',
  'luxar-control-tile-sublabel',
  'luxar-control-message',
];

/** Custom properties an author may set. */
// `--luxar-control-tile-min` was deliberately RETIRED, not lost. It set a
// minimum tile WIDTH, and a minimum width is incompatible with the page's one
// hard requirement: a `minmax(<min>, 1fr)` track cannot shrink past its
// floor, so the grid overflows as soon as the chapters outgrow the viewport —
// which is the scrolling a kiosk panel must never do. Tile size now falls out
// of the fitted column count instead (`config/control-panel/fit-grid.ts`).
// Safe to remove because the authoring vocabulary has not shipped yet, so no
// authored stylesheet can be reading it.
const CONTRACT_PROPERTIES = [
  '--luxar-control-columns',
  '--luxar-control-gap',
  '--luxar-control-radius',
];

/** State hooks the renderer maintains. */
const CONTRACT_ATTRIBUTES = [
  'data-active',
  'data-authored-label',
  'data-chapter-index',
  'data-chapter-value',
  'data-chapter-count',
  'data-grid-columns',
];

function renderedPanel(): HTMLElement {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  const panel = createControlPanel({ root, call: () => undefined });
  const source = deriveChapters({
    displayed: [0],
    metadata: [
      { name: 'x', unit: '', scale: 1 },
      { name: 'story', unit: '', scale: 1, discrete: true, step: 1, categories: ['A', 'B'] },
    ],
    ranges: [
      [0, 1],
      [0, 1],
    ],
  });
  panel.render(source, { title: 'Title', subtitle: 'Subtitle', sublabels: { 0: 'Sub' } });
  panel.setActive(0);
  return root;
}

/** A panel showing its no-tiles state, where `-message` lives. */
function messagePanel(): HTMLElement {
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  createControlPanel({ root, call: () => undefined }).showMessage('Heading', 'Detail');
  return root;
}

describe('control panel styling contract', () => {
  const css = readFileSync(STYLESHEET, 'utf8');
  const html = readFileSync(CONTROL_HTML, 'utf8');
  /** Rules only — the file's own prose mentions names it must not IMPORT. */
  const rules = css.replace(/\/\*[\s\S]*?\*\//g, '');

  it('reads a stylesheet that is actually there', () => {
    // Without this, every assertion below would pass against an empty string.
    expect(css.length).toBeGreaterThan(500);
    expect(css).toContain(CONTROL_PANEL_CLASS);
  });

  it('blocks author CSS from fetching remote images, fonts, or stylesheets', () => {
    expect(html).toContain("default-src 'self'");
    expect(html).toContain("img-src 'self' data:");
    expect(html).toContain("font-src 'self' data:");
    expect(html).toContain("style-src 'self' 'unsafe-inline'");
  });

  it.each(CONTRACT_CLASSES)('styles .%s', (className) => {
    expect(css).toContain(`.${className}`);
  });

  it.each(CONTRACT_PROPERTIES)('reads or sets %s', (property) => {
    expect(css).toContain(property);
  });

  it.each(CONTRACT_ATTRIBUTES)('styles or documents [%s]', (attribute) => {
    expect(css).toContain(attribute);
  });

  it('keeps the exported property name in step with the stylesheet', () => {
    expect(CONTRACT_PROPERTIES).toContain(CONTROL_COLUMNS_PROPERTY);
  });

  it('emits every contract class into the DOM', () => {
    // Two renders, because the contract spans two states: `-message` only
    // exists when there are no tiles, and the tile classes only when there are.
    const populated = renderedPanel();
    const empty = messagePanel();
    for (const className of CONTRACT_CLASSES) {
      const found = [populated, empty].some(
        (root) => root.classList.contains(className) || root.querySelector(`.${className}`) !== null
      );
      expect(found, `the renderer emits no .${className} in any state`).toBe(true);
    }
  });

  it('emits every contract state attribute into the DOM', () => {
    const root = renderedPanel();
    for (const attribute of CONTRACT_ATTRIBUTES) {
      expect(
        root.querySelector(`[${attribute}]`),
        `no element carries [${attribute}]`
      ).not.toBeNull();
    }
  });

  it('imports nothing at all', () => {
    // `styles/index.css` pulls twenty component sheets plus the GUI library,
    // and the panel's whole value is being light enough for a plinth tablet.
    // Asserted as "no @import whatsoever" rather than a denylist, so a future
    // import of anything has to be a deliberate change to this test.
    expect(rules).not.toContain('@import');
  });

  it('keeps every colour behind a theme token with a fallback', () => {
    // The page ships without ThemeManager (it persists to localStorage, and the
    // panel shares an origin with the display — a panel theme would re-theme
    // the big screen on its next reload). So each token needs a fallback, or
    // the panel renders unstyled.
    const tokens = [...css.matchAll(/var\((--luxar-[a-z-]+)([^)]*)\)/g)];
    expect(tokens.length).toBeGreaterThan(5);
    const themeTokensWithoutFallback = tokens
      .filter(([, name]) => !name.startsWith('--luxar-control-'))
      .filter(([, , rest]) => !rest.includes(','))
      .map(([, name]) => name);
    expect(themeTokensWithoutFallback).toEqual([]);
  });
});
