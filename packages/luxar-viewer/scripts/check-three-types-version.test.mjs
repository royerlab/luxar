import { afterEach, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { checkThreeTypesVersion } from './check-three-types-version.mjs';

const roots = [];

function makeFixture({
  runtime = '~0.185.1',
  types = '~0.185.4',
  indexVersion = '0.185.1',
  readmeVersion = '0.185.1',
} = {}) {
  const viewerRoot = mkdtempSync(join(tmpdir(), 'luxar-three-types-'));
  writeFileSync(
    join(viewerRoot, 'package.json'),
    JSON.stringify({
      peerDependencies: { three: runtime },
      devDependencies: { '@types/three': types },
    })
  );
  const embedRoot = join(viewerRoot, 'examples', 'embed');
  mkdirSync(embedRoot, { recursive: true });
  writeFileSync(
    join(embedRoot, 'index.html'),
    `"three": "https://cdn.jsdelivr.net/npm/three@${indexVersion}/build/three.module.js"`
  );
  writeFileSync(
    join(embedRoot, 'README.md'),
    `"three": "https://cdn.jsdelivr.net/npm/three@${readmeVersion}/build/three.module.js"`
  );
  roots.push(viewerRoot);
  return viewerRoot;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('checkThreeTypesVersion', () => {
  it('passes when declarations and embed pins share a minor with different patches', () => {
    expect(
      checkThreeTypesVersion(makeFixture({ indexVersion: '0.185.2', readmeVersion: '0.185.3' }))
    ).toEqual([]);
  });

  it('rejects a type-only minor bump', () => {
    expect(checkThreeTypesVersion(makeFixture({ types: '~0.186.0' }))).toEqual(
      expect.arrayContaining([expect.stringContaining('@types/three targets r186')])
    );
  });

  it.each([
    { field: 'indexVersion', path: 'examples/embed/index.html' },
    { field: 'readmeVersion', path: 'examples/embed/README.md' },
  ])('rejects a stale $path importmap pin', ({ field, path }) => {
    expect(checkThreeTypesVersion(makeFixture({ [field]: '0.184.1' }))).toEqual(
      expect.arrayContaining([expect.stringContaining(`${path} targets r184`)])
    );
  });

  it('rejects a malformed importmap pin', () => {
    expect(checkThreeTypesVersion(makeFixture({ indexVersion: 'latest' }))).toEqual(
      expect.arrayContaining([
        expect.stringContaining('examples/embed/index.html must pin three@0.MINOR.PATCH'),
      ])
    );
  });

  it.each([
    { runtime: '^0.185.1', types: '~0.185.4' },
    { runtime: '~0.185.1', types: '0.185.4' },
    { runtime: '~0.185', types: '~0.185.4' },
    { runtime: '~1.2.0', types: '~1.2.3' },
  ])('rejects non-plain tilde ranges: %o', ({ runtime, types }) => {
    expect(checkThreeTypesVersion(makeFixture({ runtime, types }))).toEqual(
      expect.arrayContaining([expect.stringContaining('Expected plain ~0.MINOR.PATCH ranges')])
    );
  });

  it('rejects a missing runtime declaration', () => {
    const viewerRoot = makeFixture();
    writeFileSync(
      join(viewerRoot, 'package.json'),
      JSON.stringify({ peerDependencies: {}, devDependencies: { '@types/three': '~0.185.4' } })
    );
    expect(checkThreeTypesVersion(viewerRoot)).toEqual([
      'package.json must declare peerDependencies.three',
    ]);
  });

  it('rejects a missing types declaration', () => {
    const viewerRoot = makeFixture();
    writeFileSync(
      join(viewerRoot, 'package.json'),
      JSON.stringify({ peerDependencies: { three: '~0.185.1' }, devDependencies: {} })
    );
    expect(checkThreeTypesVersion(viewerRoot)).toEqual([
      'package.json must declare devDependencies["@types/three"]',
    ]);
  });
});
