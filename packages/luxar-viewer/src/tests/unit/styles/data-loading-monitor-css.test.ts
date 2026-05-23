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
// [styles.md/O2][P10] Shared CSS-text helpers extracted to
// `_helpers/css-text.ts` to remove the bootstrap duplication this file
// previously shared with library-css-scope.test.ts.
import { stripMediaQueries, ruleBody } from './_helpers/css-text';

const HERE = dirname(fileURLToPath(import.meta.url));
const CSS_PATH = resolve(HERE, '../../../styles/components/data-loading-monitor.css');

describe('data-loading-monitor.css cache layout rules', () => {
  const cssRaw = readFileSync(CSS_PATH, 'utf8');
  // The rules we guard apply at every viewport width; a duplicate
  // hiding inside the @media (max-width: 480px) fallback must not be
  // enough to satisfy these assertions.
  const css = stripMediaQueries(cssRaw);

  it('cols-3 metrics grid uses 4 columns with minmax+1fr (not max-content) so it aligns with cols-4', () => {
    // styles.md W3 fix: previous assertion only matched `repeat(4,`. A
    // mutation to `repeat(4, max-content)` (column-count preserved but
    // sizing track different) would have passed silently. Pin the full
    // `minmax(0, 1fr)` track shape that drives the equal-width layout.
    const body = ruleBody(css, '.luxar-cache-section__metrics--cols-3');
    expect(body).not.toBe('');
    expect(body).toMatch(/grid-template-columns:\s*repeat\(\s*4\s*,\s*minmax\(\s*0\s*,\s*1fr\s*\)\s*\)/);
  });

  it('cols-3 last child spans 2 columns to fill the row', () => {
    const body = ruleBody(
      css,
      '.luxar-cache-section__metrics--cols-3 > .luxar-metric-card:last-child'
    );
    expect(body).not.toBe('');
    expect(body).toMatch(/grid-column:\s*span\s*2/);
  });

  it('cols-4 metrics grid uses 4 columns with minmax+1fr (full track shape)', () => {
    // Mirror cols-3: pin the full track to defend against
    // max-content / auto / specific-px regressions.
    const body = ruleBody(css, '.luxar-cache-section__metrics--cols-4');
    expect(body).not.toBe('');
    expect(body).toMatch(/grid-template-columns:\s*repeat\(\s*4\s*,\s*minmax\(\s*0\s*,\s*1fr\s*\)\s*\)/);
  });

  it('EFFECTIVE HIT RATE footer has explicit non-zero typography + spacing', () => {
    // styles.md W2 fix: previous matched only `margin-top:` / `font-size:`
    // / `color:` with NO value constraint, so `margin-top: 0` /
    // `font-size: 0` would pass. Pin that the value is something non-zero
    // (any unit) — a regression that nukes the visual spacing would now
    // fail.
    const body = ruleBody(css, '.luxar-cache-total__demand');
    expect(body).not.toBe('');
    // `margin-top: <number>` where the number is NOT just "0" — accept
    // var(...) tokens or units. Mutation `margin-top: 0` fails this.
    expect(body).toMatch(/margin-top:\s*(?!0(?:;|\s)|0px|0em|0rem|0%)\S+/);
    expect(body).toMatch(/font-size:\s*(?!0(?:;|\s)|0px|0em|0rem)\S+/);
    expect(body).toMatch(/color:\s*\S+/);
  });

  it('cache tab content reserves NON-ZERO bottom padding for the TOTAL section', () => {
    // styles.md W1 fix: previous matched `padding-bottom:` with no value.
    // A mutation `padding-bottom: 0` would have passed. Pin non-zero.
    const body = ruleBody(css, '.luxar-tab-content--cache');
    expect(body).not.toBe('');
    expect(body).toMatch(/padding-bottom:\s*(?!0(?:;|\s)|0px|0em|0rem|0%)\S+/);
  });

  it('stripMediaQueries actually removes the responsive fallback block', () => {
    // Belt-and-braces: confirm the helper drops the @media block,
    // since the guards above only mean what they say when this holds.
    expect(cssRaw).toContain('@media (max-width: 480px)');
    expect(css).not.toContain('@media');
  });
});
