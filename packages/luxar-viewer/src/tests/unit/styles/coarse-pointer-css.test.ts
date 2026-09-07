/**
 * Contract test for `styles/components/coarse-pointer.css` — the ONE place
 * touch / coarse-pointer adaptations live.
 *
 * Desktop neutrality is a property of the CSS text, so it is asserted on the
 * text: (1) the file's top level holds nothing but `@media` blocks, (2) every
 * block is keyed on a pointer / hover media feature (never on viewport width
 * alone), (3) no other stylesheet in the library entry mentions those
 * features (centralisation), and (4) the load-bearing clamps are present.
 *
 * jsdom does not evaluate `@media`, so this is the only unit-level guard; the
 * behavioural half is the mobile Playwright suite.
 */

import { describe, expect, it } from 'vitest';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { expandImports, ruleBody, stripComments, stripMediaQueries } from './_helpers/css-text';

const HERE = dirname(fileURLToPath(import.meta.url));
const STYLES_ROOT = resolve(HERE, '../../../styles');
const GUI_STYLES_ROOT = resolve(HERE, '../../../ui/gui/styles');
const FILE = resolve(STYLES_ROOT, 'components/coarse-pointer.css');

const raw = readFileSync(FILE, 'utf8');
const css = stripComments(raw);

/** The pointer/hover features a block may be keyed on. */
const POINTER_FEATURES = /\((pointer:\s*coarse|hover:\s*none|any-hover:\s*hover)\)/;
/** Nested `@supports` is allowed (dvh fallback) — only inside a media block. */
const SUPPORTS_DVH = /@supports\s*\(height:\s*100dvh\)/;

/** Every `@media` prelude in the file, in order. */
function mediaPreludes(text: string): string[] {
  return [...text.matchAll(/@media\s*([^{]+)\{/g)].map((m) => m[1].trim());
}

/** Return the bodies of every top-level `@media` block whose prelude matches. */
function mediaBlocks(text: string, prelude: RegExp): string[] {
  const out: string[] = [];
  let i = 0;
  while (i < text.length) {
    const m = /@media\s*([^{]+)\{/g;
    m.lastIndex = i;
    const at = m.exec(text);
    if (!at) break;
    const open = at.index + at[0].length - 1;
    let depth = 1;
    let j = open + 1;
    while (j < text.length && depth > 0) {
      if (text[j] === '{') depth++;
      else if (text[j] === '}') depth--;
      j++;
    }
    if (prelude.test(at[1])) out.push(text.slice(open + 1, j - 1));
    i = j;
  }
  return out;
}

/** The first matching top-level `@media` block body ('' when absent). */
function mediaBlock(text: string, prelude: RegExp): string {
  return mediaBlocks(text, prelude)[0] ?? '';
}

function cssFilesUnder(dir: string): string[] {
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) out.push(...cssFilesUnder(full));
    else if (name.endsWith('.css')) out.push(full);
  }
  return out;
}

describe('coarse-pointer.css contract', () => {
  it('is imported by the library entry and is the last component import', () => {
    const index = stripComments(readFileSync(resolve(STYLES_ROOT, 'index.css'), 'utf8'));
    const imports = [...index.matchAll(/@import\s+['"]([^'"]+)['"]/g)].map((m) => m[1]);
    expect(imports.at(-1)).toBe('./components/coarse-pointer.css');
    expect(expandImports(resolve(STYLES_ROOT, 'index.css'))).toContain(
      '.luxar-control-rail__items'
    );
  });

  it('has nothing at the top level except @media blocks (desktop CSS is untouched)', () => {
    expect(stripMediaQueries(css).trim()).toBe('');
  });

  it('keys every @media block on a pointer / hover feature, never on width alone', () => {
    const preludes = mediaPreludes(css);
    expect(preludes.length).toBeGreaterThan(0);
    for (const prelude of preludes) {
      expect(prelude, `media prelude "${prelude}"`).toMatch(POINTER_FEATURES);
      expect(prelude, `no width-only gate in "${prelude}"`).not.toMatch(/^\(?\s*(max|min)-width/);
    }
  });

  it('only nests @supports for the dvh fallback', () => {
    const supports = [...css.matchAll(/@supports\s*\([^)]*\)/g)].map((m) => m[0]);
    for (const s of supports) expect(s).toMatch(SUPPORTS_DVH);
    // Every dvh use sits inside such a fallback block.
    const outsideSupports = css.replace(
      /@supports\s*\(height:\s*100dvh\)\s*\{[\s\S]*?\n\s*\}\n\s*\}/g,
      ''
    );
    expect(outsideSupports).not.toMatch(/\d+dvh/);
  });

  it('is the only stylesheet in the tree that mentions the pointer / hover features', () => {
    const files = [...cssFilesUnder(STYLES_ROOT), ...cssFilesUnder(GUI_STYLES_ROOT)].filter(
      (f) => resolve(f) !== FILE
    );
    for (const file of files) {
      const text = stripComments(readFileSync(file, 'utf8'));
      expect(text, `pointer/hover media feature outside coarse-pointer.css: ${file}`).not.toMatch(
        POINTER_FEATURES
      );
    }
  });

  it('clamps the rail, the fixed-width panels and the bottom strip under (pointer: coarse)', () => {
    const coarse = mediaBlock(css, /pointer:\s*coarse/);
    expect(coarse).not.toBe('');

    // Rail: the ITEMS wrapper scrolls (the root must not — it hosts popovers).
    const items = ruleBody(coarse, '.luxar-control-rail__items');
    expect(items).toMatch(/overflow:\s*hidden auto/);
    expect(items).toMatch(/min-height:\s*0/);
    expect(items).toMatch(/overscroll-behavior:\s*contain/);
    expect(ruleBody(coarse, '.luxar-control-rail')).not.toMatch(/overflow/);
    expect(ruleBody(coarse, '.luxar-control-rail')).toMatch(/max-height:\s*calc\(100vh/);

    // The wrapper scrolls, but its children and the root's fixed controls do not shrink.
    expect(coarse).toMatch(
      /\.luxar-control-rail__items > \.luxar-control-rail__btn,[\s\S]*?\.luxar-control-rail > \.luxar-perf\s*\{[^}]*flex:\s*0 0 auto/
    );

    // Labels cannot render outside the clipped scroll box; D2 provides the touch label route.
    expect(ruleBody(coarse, '.luxar-control-rail__tip')).toMatch(/display:\s*none/);

    // The empty wrapper must not add a flex gap to the collapsed horizontal rail.
    expect(ruleBody(coarse, '.luxar-control-rail.is-collapsed .luxar-control-rail__items')).toMatch(
      /display:\s*none/
    );

    // Width clamps use min(<desktop width>, viewport - margins).
    expect(ruleBody(coarse, '.luxar-help-overlay')).toMatch(/width:\s*min\(400px,/);
    expect(ruleBody(coarse, '.luxar-data-monitor--expanded')).toMatch(/width:\s*min\(600px,/);
    expect(ruleBody(coarse, '.luxar-debug-console')).toMatch(/width:\s*min\(600px,/);

    // Safe-area insets on the docked gutter and the bottom strip.
    expect(coarse).toMatch(
      /\.luxar-has-control-rail \.luxar-layers-panel \{[^}]*safe-area-inset-left[^}]*!important/
    );
    expect(ruleBody(coarse, '.luxar-control-rail-hint')).toMatch(/safe-area-inset-left/);
    expect(ruleBody(coarse, '.luxar-debug-console')).toMatch(/safe-area-inset-bottom/);
    for (const sel of [
      '.luxar-dimension-sliders',
      '.luxar-toast',
      '.luxar-scale-bar',
      '.luxar-colormap-legend',
      '.luxar-resolution-indicator',
    ]) {
      expect(ruleBody(coarse, sel), sel).toMatch(/bottom:\s*calc\(.*safe-area-inset-bottom/);
    }

    // Tap targets opt out of the double-tap-zoom delay.
    expect(coarse).toMatch(/\.luxar-control-rail__btn,[\s\S]*?\{[^}]*touch-action:\s*manipulation/);
  });

  it('keeps the rail visible without hover under (hover: none)', () => {
    const noHover = mediaBlock(css, /hover:\s*none/);
    expect(noHover).toMatch(
      /\.luxar-control-rail,\s*\.luxar-control-rail\.is-collapsed\s*\{[^}]*opacity:\s*1/
    );
    // Fullscreen stays findable rather than fully hidden.
    expect(ruleBody(noHover, '.luxar-control-rail.is-fullscreen')).toMatch(/opacity:\s*0\.\d+/);
  });

  it('converts every vh bound the panels use to dvh inside the fallback block', () => {
    // The dvh rules live in the coarse block that wraps them in @supports.
    const dvh = mediaBlocks(css, /pointer:\s*coarse/).find((b) => b.includes('@supports'))!;
    expect(dvh).toBeDefined();
    for (const sel of [
      '.luxar-control-rail',
      '.luxar-help-overlay',
      '.luxar-layers-panel',
      '.luxar-data-monitor--expanded',
      '.luxar-dataset-browser',
      '.luxar-dimension-sliders',
    ]) {
      expect(ruleBody(dvh, sel), sel).toMatch(/dvh/);
    }
  });
});

describe('control-rail items wrapper', () => {
  it('is layout-transparent on fine pointers (display: contents in the component file)', () => {
    const rail = stripComments(
      readFileSync(resolve(STYLES_ROOT, 'components/control-rail.css'), 'utf8')
    );
    expect(ruleBody(rail, '.luxar-control-rail__items')).toMatch(/display:\s*contents/);
  });
});

describe('standalone page', () => {
  it('declares viewport-fit=cover so safe-area insets resolve', () => {
    const html = readFileSync(resolve(HERE, '../../../../index.html'), 'utf8');
    expect(html).toMatch(/name="viewport"[^>]*viewport-fit=cover/);
  });

  it('sizes the page with dvh behind an @supports fallback', () => {
    const layout = stripComments(readFileSync(resolve(STYLES_ROOT, 'base/layout.css'), 'utf8'));
    expect(layout).toMatch(
      /@supports\s*\(height:\s*100dvh\)\s*\{\s*html,\s*body\s*\{\s*height:\s*100dvh/
    );
  });
});
