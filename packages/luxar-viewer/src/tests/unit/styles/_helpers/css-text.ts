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
 * Strip every `/* … *\/` comment from a CSS string, so a text-level selector
 * scan reads declarations only. Without this, a wrapped prose line inside a
 * comment (e.g. one starting with the word "body") looks exactly like a
 * top-level selector to the line-anchored patterns in
 * library-css-scope.test.ts.
 *
 * Single pass over the source rather than a `\/\*[\s\S]*?\*\/` replace, so
 * comments and quoted strings are recognised in the order they actually
 * appear. A plain regex is wrong in both directions: `content: "/*"` would
 * open a comment that swallows the following (real) rules, and scanning for
 * strings first would trip over the apostrophes this codebase's own prose
 * comments are full of. Strings are copied through verbatim — a CSS string
 * cannot span a raw newline, so an unterminated quote ends at the line break
 * instead of eating the rest of the file.
 */
export function stripComments(css: string): string {
  let out = '';
  let i = 0;
  while (i < css.length) {
    const ch = css[i];
    if (ch === '/' && css[i + 1] === '*') {
      const end = css.indexOf('*/', i + 2);
      i = end === -1 ? css.length : end + 2;
      continue;
    }
    if (ch === '"' || ch === "'") {
      let j = i + 1;
      while (j < css.length && css[j] !== ch && css[j] !== '\n') {
        if (css[j] === '\\') j++;
        j++;
      }
      out += css.slice(i, Math.min(j + 1, css.length));
      i = j + 1;
      continue;
    }
    out += ch;
    i++;
  }
  return out;
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
 * literal selector at the start of a rule after normalizing whitespace and
 * selector punctuation. Returns an empty string when the selector is absent.
 */
export function ruleBody(css: string, selector: string): string {
  const compact = (text: string): string =>
    text
      .replace(/\s+/g, ' ')
      .replace(/\s*([(),>+~])\s*/g, '$1')
      .trim();
  const compactCss = compact(css);
  const compactSelector = compact(selector);
  const escaped = compactSelector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(`(^|[^\\w-])${escaped}\\s*\\{([^}]*)\\}`, 'm');
  const m = compactCss.match(re);
  return m ? m[2] : '';
}
