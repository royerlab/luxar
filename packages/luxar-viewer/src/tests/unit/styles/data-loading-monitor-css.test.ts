/**
 * CSS-shape guards for `data-loading-monitor.css`. These rules are
 * not asserted indirectly via class names elsewhere — they encode the
 * visual contract for the Cache tab layout, so we pin them with a
 * direct text scan against the stylesheet source.
 *
 *   1. The 3-card sections (L0/L1) must share a 4-column grid with
 *      the 4-card section (L2) so column edges align across all
 *      cache sections — the trailing card spans 2 columns to fill
 *      the row.
 *   2. The EFFECTIVE-HIT-RATE footer must be styled (not unstyled
 *      stray text drifting outside the TOTAL card).
 *   3. The cache tab must reserve bottom padding so the TOTAL section
 *      is not flush against the scroll edge.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const CSS_PATH = resolve(
  HERE,
  '../../../styles/components/data-loading-monitor.css'
);

/**
 * Strip every `@media (...) { ... }` block. The cache layout rules
 * we guard must live in the base stylesheet — a duplicate hiding
 * inside a media query (e.g. the 480px responsive fallback) is not
 * enough to satisfy desktop layout, so we exclude media-query bodies
 * from the rule-body lookup.
 */
function stripMediaQueries(css: string): string {
  // Walk the source, dropping `@media` blocks with balanced braces.
  // A regex with `[^{}]*` would mis-handle nested rules inside the
  // media block, so we use an explicit brace counter.
  let out = '';
  let i = 0;
  while (i < css.length) {
    const at = css.indexOf('@media', i);
    if (at === -1) {
      out += css.slice(i);
      break;
    }
    out += css.slice(i, at);
    const openBrace = css.indexOf('{', at);
    if (openBrace === -1) break;
    let depth = 1;
    let j = openBrace + 1;
    while (j < css.length && depth > 0) {
      const ch = css[j];
      if (ch === '{') depth++;
      else if (ch === '}') depth--;
      j++;
    }
    i = j;
  }
  return out;
}

/**
 * Extract the declaration block for a given CSS selector. Matches
 * the literal selector at the start of a rule. Returns an empty
 * string when the selector is not present. Good enough for the
 * focused rule-shape assertions below.
 */
function ruleBody(css: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(^|[^\\w-])${escaped}\\s*\\{([^}]*)\\}`, 'm');
  const m = css.match(re);
  return m ? m[2] : '';
}

describe('data-loading-monitor.css cache layout rules', () => {
  const cssRaw = readFileSync(CSS_PATH, 'utf8');
  // The rules we guard apply at every viewport width; a duplicate
  // hiding inside the @media (max-width: 480px) fallback must not be
  // enough to satisfy these assertions.
  const css = stripMediaQueries(cssRaw);

  it('cols-3 metrics grid uses 4 columns so it aligns with cols-4', () => {
    const body = ruleBody(css, '.luxar-cache-section__metrics--cols-3');
    expect(body).not.toBe('');
    expect(body).toMatch(/grid-template-columns:\s*repeat\(\s*4\s*,/);
  });

  it('cols-3 last child spans 2 columns to fill the row', () => {
    const body = ruleBody(
      css,
      '.luxar-cache-section__metrics--cols-3 > .luxar-metric-card:last-child'
    );
    expect(body).not.toBe('');
    expect(body).toMatch(/grid-column:\s*span\s*2/);
  });

  it('cols-4 metrics grid uses 4 columns', () => {
    const body = ruleBody(css, '.luxar-cache-section__metrics--cols-4');
    expect(body).not.toBe('');
    expect(body).toMatch(/grid-template-columns:\s*repeat\(\s*4\s*,/);
  });

  it('EFFECTIVE HIT RATE footer has explicit typography + spacing', () => {
    const body = ruleBody(css, '.luxar-cache-total__demand');
    expect(body).not.toBe('');
    // Must have separation from the progress bar above and its own
    // text styling — otherwise it reads as drifting unstyled text.
    expect(body).toMatch(/margin-top:/);
    expect(body).toMatch(/font-size:/);
    expect(body).toMatch(/color:/);
  });

  it('cache tab content reserves bottom padding for the TOTAL section', () => {
    const body = ruleBody(css, '.luxar-tab-content--cache');
    expect(body).not.toBe('');
    expect(body).toMatch(/padding-bottom:/);
  });

  it('stripMediaQueries actually removes the responsive fallback block', () => {
    // Belt-and-braces: confirm the helper drops the @media block,
    // since the guards above only mean what they say when this holds.
    expect(cssRaw).toContain('@media (max-width: 480px)');
    expect(css).not.toContain('@media');
  });
});
