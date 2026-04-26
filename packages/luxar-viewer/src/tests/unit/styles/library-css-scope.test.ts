/**
 * Asserts that the library's published CSS entry point (`styles/index.css`)
 * does NOT contain global selectors that would clobber a host page.
 *
 * Embedders import `luxar-viewer/styles.css` (= styles/index.css). Any rule
 * targeting `body`, `html`, `*`, `:focus-visible`, list elements, or form
 * elements would override their page styling without warning. Rules of that
 * shape belong in `styles/standalone.css`, which only the standalone-app
 * entry imports.
 *
 * This test resolves @import chains and scans the resulting CSS source for
 * top-level selectors. If a future change adds a new component CSS file
 * that touches `body { … }` etc., this test fails — push it into
 * standalone.css instead.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const STYLES_ROOT = resolve(HERE, '../../../styles');

/** Inline @import directives recursively into a single CSS string. */
function expandImports(file: string, seen = new Set<string>()): string {
  const abs = resolve(file);
  if (seen.has(abs)) return '';
  seen.add(abs);
  if (!existsSync(abs)) return '';

  const source = readFileSync(abs, 'utf8');
  return source.replace(/@import\s+['"]([^'"]+)['"]\s*;?/g, (_, importPath: string) => {
    const importedAbs = resolve(dirname(abs), importPath);
    return expandImports(importedAbs, seen);
  });
}

/**
 * Match top-level CSS selectors that would clobber a host page. We do this
 * with a regex against the raw text rather than parsing the CSS — good
 * enough for the small surface here. False-positive guard: only match the
 * selector at the start of a rule (newline + selector + `{`), not inside
 * descendant selectors like `.luxar-foo body`.
 */
const FORBIDDEN_SELECTORS = [
  /^body\s*[,{]/m,
  /^html\s*[,{]/m,
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
    const css = expandImports(resolve(STYLES_ROOT, 'index.css'));
    expect(css.length).toBeGreaterThan(0);

    const offenders: string[] = [];
    for (const pattern of FORBIDDEN_SELECTORS) {
      const match = css.match(pattern);
      if (match) {
        offenders.push(`${pattern.source} matched: "${match[0].slice(0, 80)}"`);
      }
    }

    if (offenders.length > 0) {
      throw new Error(
        'styles/index.css must not include global selectors that affect ' +
          'host pages. Move these to styles/standalone.css:\n' +
          offenders.map((o) => '  - ' + o).join('\n')
      );
    }
  });

  it('styles/standalone.css still contains the global rules (sanity check)', () => {
    const css = expandImports(resolve(STYLES_ROOT, 'standalone.css'));
    // standalone.css is allowed to have these — that's the whole point.
    expect(css).toMatch(/^body\s*\{/m);
    expect(css).toMatch(/^html\s*,\s*body/m);
  });
});
