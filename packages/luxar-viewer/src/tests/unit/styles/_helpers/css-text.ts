/**
 * Shared CSS-text helpers for the `styles/` test suite.
 *
 * [styles.md/O2][P10] Extracted from data-loading-monitor-css.test.ts and
 * library-css-scope.test.ts which independently duplicated the same
 * file-read + @import-resolution + media-query-strip primitives.
 *
 * Pure file-system helpers — no DOM, no vitest globals.
 */

import { readFileSync, existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

/**
 * Inline `@import 'relative/path';` directives recursively into a single CSS
 * string. Visited files are tracked via the optional `seen` set so cycles
 * resolve to the empty string rather than infinite recursion.
 *
 * Supports BOTH the bare-string form (`@import 'foo.css';`) and the
 * `url(...)` form (`@import url('foo.css');` / `@import url(foo.css);`).
 * styles.md C4: the prior regex only matched bare-string imports, so any
 * future `@import url(...)` would silently skip — breaking the embed-safety
 * contract under test in library-css-scope.test.ts.
 *
 * Quietly returns '' for missing files — callers can length-check.
 */
export function expandImports(file: string, seen = new Set<string>()): string {
  const abs = resolve(file);
  if (seen.has(abs)) return '';
  seen.add(abs);
  if (!existsSync(abs)) return '';

  const source = readFileSync(abs, 'utf8');
  // Pattern matches both forms:
  //   @import 'path';          (bare string)
  //   @import "path";          (bare string, double-quote)
  //   @import url('path');     (url-form, quoted)
  //   @import url("path");
  //   @import url(path);       (url-form, unquoted)
  const importRe = /@import\s+(?:url\(\s*['"]?([^'")]+)['"]?\s*\)|['"]([^'"]+)['"])\s*;?/g;
  return source.replace(importRe, (_, urlPath: string | undefined, bare: string | undefined) => {
    const importPath = urlPath ?? bare;
    if (!importPath) return '';
    const importedAbs = resolve(dirname(abs), importPath);
    return expandImports(importedAbs, seen);
  });
}

/**
 * Strip every `@media (...) { ... }` block from a CSS string. Uses an
 * explicit brace counter (a regex with `[^{}]*` mishandles nested rules
 * inside the media block).
 */
export function stripMediaQueries(css: string): string {
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
 * Extract the declaration block for a given CSS selector. Matches the
 * literal selector at the start of a rule. Returns an empty string when
 * the selector is not present.
 */
export function ruleBody(css: string, selector: string): string {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(^|[^\\w-])${escaped}\\s*\\{([^}]*)\\}`, 'm');
  const m = css.match(re);
  return m ? m[2] : '';
}
