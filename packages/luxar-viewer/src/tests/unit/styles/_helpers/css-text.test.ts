/**
 * Unit tests for the styles/_helpers/css-text helpers.
 *
 * Targets audit finding styles.md C4: `expandImports` previously matched
 * only the bare-string form `@import 'foo.css'`. Any future `@import url(...)`
 * would silently skip — breaking the embed-safety contract in
 * library-css-scope.test.ts.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { expandImports, ruleBody, stripComments } from './css-text';

describe('expandImports — @import syntax coverage [styles.md C4]', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'luxar-css-text-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('[C4] resolves @import with bare-string form', () => {
    writeFileSync(join(dir, 'child.css'), '.child { color: red; }');
    writeFileSync(join(dir, 'root.css'), "@import 'child.css';\n.root { color: blue; }");
    const css = expandImports(join(dir, 'root.css'));
    expect(css).toContain('.child { color: red; }');
    expect(css).toContain('.root { color: blue; }');
  });

  it('[C4] resolves @import url(...) with single-quoted path', () => {
    // Previously the regex `/@import\s+['"]([^'"]+)['"]\s*;?/g` did NOT
    // match url(...) syntax. A future stylesheet using url() would have
    // silently skipped, hiding a global selector from the scope test.
    writeFileSync(join(dir, 'child.css'), '.url-child { color: green; }');
    writeFileSync(join(dir, 'root.css'), "@import url('child.css');");
    const css = expandImports(join(dir, 'root.css'));
    expect(css).toContain('.url-child { color: green; }');
  });

  it('[C4] resolves @import url(...) with double-quoted path', () => {
    writeFileSync(join(dir, 'child.css'), '.dq-child { color: orange; }');
    writeFileSync(join(dir, 'root.css'), '@import url("child.css");');
    const css = expandImports(join(dir, 'root.css'));
    expect(css).toContain('.dq-child { color: orange; }');
  });

  it('[C4] resolves @import url(...) with UNQUOTED path', () => {
    writeFileSync(join(dir, 'child.css'), '.uq-child { color: purple; }');
    writeFileSync(join(dir, 'root.css'), '@import url(child.css);');
    const css = expandImports(join(dir, 'root.css'));
    expect(css).toContain('.uq-child { color: purple; }');
  });

  it('quietly returns empty string for missing imports', () => {
    writeFileSync(join(dir, 'root.css'), "@import 'does-not-exist.css';\n.root { color: red; }");
    const css = expandImports(join(dir, 'root.css'));
    // Missing import resolves to '' but the surrounding rule is preserved.
    expect(css).toContain('.root { color: red; }');
  });

  it('breaks cycles via the seen-set (no infinite recursion)', () => {
    writeFileSync(join(dir, 'a.css'), "@import 'b.css';\n.a {}");
    writeFileSync(join(dir, 'b.css'), "@import 'a.css';\n.b {}");
    expect(() => expandImports(join(dir, 'a.css'))).not.toThrow();
    const css = expandImports(join(dir, 'a.css'));
    expect(css).toContain('.a {}');
    expect(css).toContain('.b {}');
  });
});

describe('stripComments', () => {
  it('removes a multi-line comment whose wrapped prose looks like a selector', () => {
    // The exact shape that tripped the embed-safety scan: a comment line
    // beginning with the word "body" reads as a top-level `body` selector.
    const css = [
      '.luxar-panel {',
      '  /* header and the scrolling',
      '     body share a bound. */',
      '  color: red;',
      '}',
    ].join('\n');
    expect(stripComments(css)).not.toMatch(/^\s*body[\s.[:#]/m);
    expect(stripComments(css)).toContain('color: red;');
  });

  it('leaves a real top-level selector in place', () => {
    expect(stripComments('/* note */\nbody { margin: 0; }')).toMatch(/^body\s*\{/m);
  });

  it('does not treat comment delimiters inside a quoted value as a comment', () => {
    // A `content` string carrying `/*` must not open a comment and swallow the
    // real rules after it — that would hide a forbidden selector from the
    // embed-safety scan.
    const css = ['.x {', '  content: "/*";', '}', 'body { margin: 0; }'].join('\n');
    expect(stripComments(css)).toMatch(/^body\s*\{/m);
  });

  it('keeps scanning past the apostrophes in prose comments', () => {
    // Real comments in this codebase are full of apostrophes; treating one as
    // a string opener would consume everything up to the next quote.
    const css = ["/* the panel's own padding */", 'body { margin: 0; }'].join('\n');
    expect(stripComments(css)).toMatch(/^body\s*\{/m);
  });

  it('an unterminated quote stops at the line break', () => {
    const css = ['.x { font-family: "Broken;', 'body { margin: 0; }'].join('\n');
    expect(stripComments(css)).toMatch(/^body\s*\{/m);
  });
});

describe('ruleBody', () => {
  const overlaySelector =
    '.luxar-overlay-layer .luxar-overlay--text:not(.luxar-overlay--center-anchored, .luxar-overlay--right-anchored)';

  it('matches a prettier-wrapped selector', () => {
    const css = [
      '.luxar-overlay-layer',
      '  .luxar-overlay--text:not(',
      '    .luxar-overlay--center-anchored,',
      '    .luxar-overlay--right-anchored',
      '  ) {',
      '  width: 18ch;',
      '}',
    ].join('\n');

    expect(ruleBody(css, overlaySelector)).toContain('width: 18ch;');
  });

  it('matches punctuation with tighter stylesheet spacing', () => {
    const css =
      '.luxar-overlay-layer .luxar-overlay--text:not(.luxar-overlay--center-anchored,.luxar-overlay--right-anchored) { width: 18ch; }';

    expect(ruleBody(css, overlaySelector)).toContain('width: 18ch;');
  });

  it('preserves descendant combinators', () => {
    expect(ruleBody('.a.b { color: red; }', '.a .b')).toBe('');
  });
});
