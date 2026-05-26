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
import { expandImports } from './css-text';

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
