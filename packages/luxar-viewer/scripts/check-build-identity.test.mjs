/**
 * Tests for `check-build-identity.mjs` — the gate that proves a built bundle
 * carries its stamp.
 *
 * A gate is a hypothesis until it has been seen to fail, and this one guards a
 * defect whose whole character is silence. So every case below is a way the
 * gate itself could quietly stop working: passing on an empty directory,
 * passing on an unstamped bundle, or failing on a legitimately unstamped source
 * tarball.
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

import { check, findStamp, findMetaStamp, bundleFiles } from './check-build-identity.mjs';

const VERSION = '2026.9.15';
const STAMP = { version: VERSION, commit: 'abc1234', buildTime: '2026-09-15T10:00:00Z' };

/** A fake package root: `package.json` plus a `dist/` to point the gate at. */
function makeRoot() {
  const root = mkdtempSync(join(tmpdir(), 'luxar-build-id-'));
  writeFileSync(join(root, 'package.json'), JSON.stringify({ version: VERSION }));
  mkdirSync(join(root, 'dist'));
  return root;
}

/**
 * Write a bundle with one of the two payload spellings the real builds emit.
 *
 * The minified application build puts unescaped JSON inside a backtick template
 * literal; this fixture uses an equivalent single-quoted literal so the test
 * isolates payload escaping. The unminified library build keeps double quotes
 * and backslash-escapes the inner ones. Both payload spellings were read out of
 * real `dist/` output, not assumed.
 */
function writeBundle(root, stamp, { escaped = false } = {}) {
  const inner = JSON.stringify(stamp);
  const literal = escaped ? JSON.stringify(inner) : `'${inner}'`;
  writeFileSync(join(root, 'dist', 'index-abc.js'), `const s=JSON.parse(${literal});`);
}

function writeIndexHtml(root, stamp) {
  const content = stamp ? `${stamp.version} ${stamp.commit} ${stamp.buildTime}` : null;
  const meta = content ? `<meta name="luxar-build" content="${content}" />` : '';
  writeFileSync(join(root, 'dist', 'index.html'), `<html><head>${meta}</head><body></body></html>`);
}

let root;
beforeEach(() => {
  root = makeRoot();
});
afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const run = (opts = {}) => check(join(root, 'dist'), { root, requireCommit: true, ...opts });

describe('findStamp', () => {
  it('reads an unescaped literal (the minified application payload spelling)', () => {
    expect(findStamp(`x=JSON.parse('${JSON.stringify(STAMP)}')`)).toEqual(STAMP);
  });

  it('reads a backslash-escaped literal (the library build spelling)', () => {
    // The two Vite builds emit the SAME stamp with different quoting. A matcher
    // written against one reports a confident false negative on the other --
    // which is what a hand-written grep did while this stamp was being added,
    // reporting "LIB MISSING STAMP" about a bundle that carried it correctly.
    expect(findStamp(`x=JSON.parse(${JSON.stringify(JSON.stringify(STAMP))})`)).toEqual(STAMP);
  });

  it('returns null rather than a partial object when there is no stamp', () => {
    expect(findStamp('console.log("version");')).toBeNull();
  });
});

describe('findMetaStamp', () => {
  it('parses the three fields out of the meta tag', () => {
    expect(
      findMetaStamp('<meta name="luxar-build" content="1.2.3 abc 2026-01-01T00:00:00Z" />')
    ).toEqual({ version: '1.2.3', commit: 'abc', buildTime: '2026-01-01T00:00:00Z' });
  });

  it('returns null for html with no stamp', () => {
    expect(findMetaStamp('<html><head></head></html>')).toBeNull();
  });
});

describe('check', () => {
  it('passes a correctly stamped application build', () => {
    writeBundle(root, STAMP);
    writeIndexHtml(root, STAMP);
    expect(run()).toEqual([]);
  });

  it('passes a library build, which has no index.html to check', () => {
    writeBundle(root, STAMP, { escaped: true });
    expect(run()).toEqual([]);
  });

  it('FAILS on a directory with no .js files rather than reporting success', () => {
    // The fail-open case that matters most: a gate pointed at the wrong path,
    // or run before the build, must not report that everything is fine.
    const problems = run();
    expect(problems).toHaveLength(1);
    expect(problems[0]).toMatch(/nothing was scanned/);
  });

  it('FAILS when the dist directory does not exist', () => {
    rmSync(join(root, 'dist'), { recursive: true });
    expect(run()[0]).toMatch(/does not exist/);
  });

  it('FAILS on a bundle built without the define', () => {
    writeFileSync(join(root, 'dist', 'index-abc.js'), 'console.log("no stamp here");');
    writeIndexHtml(root, STAMP);
    expect(run()[0]).toMatch(/no build stamp/);
  });

  it('FAILS when only a separately built dist/lib bundle carries the stamp', () => {
    mkdirSync(join(root, 'dist', 'assets'));
    writeFileSync(join(root, 'dist', 'assets', 'index-abc.js'), 'console.log("no stamp here");');
    mkdirSync(join(root, 'dist', 'lib'));
    const inner = JSON.stringify(STAMP);
    writeFileSync(
      join(root, 'dist', 'lib', 'luxar-viewer.js'),
      `const s=JSON.parse(${JSON.stringify(inner)});`
    );
    writeIndexHtml(root, STAMP);

    expect(run()[0]).toMatch(/no build stamp/);
  });

  it('FAILS when the stamped version has drifted from package.json', () => {
    writeBundle(root, { ...STAMP, version: '2026.1.1' });
    expect(run()[0]).toMatch(/says version 2026\.1\.1, package\.json says 2026\.9\.15/);
  });

  it('FAILS when the application build lost its meta tag but kept the bundle stamp', () => {
    // Half a fix is the likely regression: someone drops the html plugin while
    // keeping `define`, and the surface that survives a broken boot goes away.
    writeBundle(root, STAMP);
    writeIndexHtml(root, null);
    expect(run()[0]).toMatch(/carries no <meta name="luxar-build">/);
  });

  it('FAILS when the meta tag and bundle carry different build identities', () => {
    writeBundle(root, STAMP);
    const metaStamp = { ...STAMP, buildTime: '2026-09-15T10:11:13Z' };
    writeIndexHtml(root, metaStamp);

    expect(run()).toEqual([
      `${join(root, 'dist', 'index.html')} stamp ${JSON.stringify(metaStamp)} disagrees with ` +
        `${join(root, 'dist', 'index-abc.js')} stamp ${JSON.stringify(STAMP)}`,
    ]);
  });

  it('FAILS on an unresolved commit when the tree HAS git history', () => {
    writeBundle(root, { ...STAMP, commit: 'unknown' });
    expect(run({ requireCommit: true })[0]).toMatch(/no commit, but this tree has git history/);
  });

  it('ACCEPTS an unresolved commit when there is no git history to read', () => {
    // A source tarball honestly cannot answer. Failing here would break builds
    // that are doing nothing wrong, and a gate that cries wolf gets disabled.
    writeBundle(root, { ...STAMP, commit: 'unknown' });
    expect(run({ requireCommit: false })).toEqual([]);
  });
});

describe('bundleFiles', () => {
  it('finds .js files in nested output directories', () => {
    mkdirSync(join(root, 'dist', 'assets'));
    writeFileSync(join(root, 'dist', 'assets', 'a.js'), '');
    writeFileSync(join(root, 'dist', 'b.js'), '');
    writeFileSync(join(root, 'dist', 'c.css'), '');
    expect(
      bundleFiles(join(root, 'dist'))
        .map((f) => f.split('/').pop())
        .sort()
    ).toEqual(['a.js', 'b.js']);
  });
});
