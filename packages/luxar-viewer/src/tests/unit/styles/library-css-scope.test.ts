/**
 * Asserts that the library's published CSS entry point (`styles/index.css`)
 * does NOT contain global selectors that would clobber a host page.
 *
 * Embedders import `@luxar/viewer/styles.css` (= styles/index.css). Any rule
 * targeting `body`, `html`, `*`, `:focus-visible`, list elements, or form
 * elements would override their page styling without warning. Rules of that
 * shape belong in `styles/standalone.css`, which only the standalone-app
 * entry imports.
 *
 * This test resolves @import chains and scans the resulting CSS source for
 * top-level selectors. If a future change adds a new component CSS file
 * that touches `body { … }` etc., this test fails — push it into
 * standalone.css instead.
 *
 * Comments are stripped before scanning: the patterns below are deliberately
 * loose (a bare `body` at the start of a line counts), so a wrapped prose
 * line inside a CSS comment beginning with "body" or "html" would otherwise
 * fail the embed contract over a selector that isn't there.
 */

import { describe, expect, it } from 'vitest';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
// [styles.md/O2][P10] Shared @import-resolution helper extracted to
// `_helpers/css-text.ts` to remove the bootstrap duplication this file
// previously shared with data-loading-monitor-css.test.ts.
import { expandImports, stripComments } from './_helpers/css-text';

const HERE = dirname(fileURLToPath(import.meta.url));
const STYLES_ROOT = resolve(HERE, '../../../styles');

/**
 * Match top-level CSS selectors that would clobber a host page. We do this
 * with a regex against the raw text rather than parsing the CSS — good
 * enough for the small surface here. False-positive guard: only match the
 * selector at the start of a rule (newline + selector + `{`), not inside
 * descendant selectors like `.luxar-foo body`.
 */
const FORBIDDEN_SELECTORS = [
  /^body\s*[,{]/m,
  // styles.md C2 fix: also catch body-with-class (`body.x`),
  // body-with-attr (`body[data-theme]`), body-followed-by-pseudo
  // (`body:hover`), and leading-whitespace-indented selectors that the
  // strict `^body` anchors miss inside nested rules.
  /^\s*body[\s.[:#]/m,
  /^html\s*[,{]/m,
  /^\s*html[\s.[:#]/m,
  /^html\s*,\s*body\s*\{/m,
  /^\*\s*[,{]/m,
  /^::-webkit-scrollbar/m,
  /^:focus-visible\s*\{/m,
  /^ul\s*[,{]/m,
  /^ol\s*[,{]/m,
  /^a\s*\{/m,
  /^button\s*\{/m,
  /^input\s*[,{]/m,
  /^textarea\s*[,{]/m,
  /^select\s*[,{]/m,
  /^h[1-6]\s*[,{]/m,
];

describe('Library CSS scope (embed safety)', () => {
  it('styles/index.css contains no global / host-page selectors', () => {
    // styles.md C3 fix: prior aggregate `expect(offenders).toEqual([])`
    // produced a single message bundling every forbidden pattern's match.
    // Switch to one assertion per pattern so a regression introducing
    // a global selector surfaces with the specific pattern name in the
    // failure header (not buried in a concatenated string).
    const css = stripComments(expandImports(resolve(STYLES_ROOT, 'index.css')));
    expect(css.length).toBeGreaterThan(0);

    for (const pattern of FORBIDDEN_SELECTORS) {
      const match = css.match(pattern);
      // The second arg to expect() supplies a label that appears in the
      // diff header — pattern.source gives "body" / "^html" / "*{" etc.
      expect(match, `forbidden selector ${pattern.source}`).toBeNull();
    }
  });

  it('styles/standalone.css still contains the global rules (sanity check)', () => {
    const css = expandImports(resolve(STYLES_ROOT, 'standalone.css'));
    // standalone.css is allowed to have these — that's the whole point.
    expect(css).toMatch(/^body\s*\{/m);
    expect(css).toMatch(/^html\s*,\s*body/m);
  });
});
