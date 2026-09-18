import { afterEach, describe, expect, it } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { checkThreeTypesVersion } from './check-three-types-version.mjs';

const roots = [];

function makeFixture({ runtime = '~0.185.1', types = '~0.185.4' } = {}) {
  const viewerRoot = mkdtempSync(join(tmpdir(), 'luxar-three-types-'));
  writeFileSync(
    join(viewerRoot, 'package.json'),
    JSON.stringify({
      peerDependencies: { three: runtime },
      devDependencies: { '@types/three': types },
    })
  );
  roots.push(viewerRoot);
  return viewerRoot;
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('checkThreeTypesVersion', () => {
  it('passes when runtime and types share a minor', () => {
    expect(checkThreeTypesVersion(makeFixture())).toEqual([]);
  });

  it('rejects a type-only minor bump', () => {
    expect(checkThreeTypesVersion(makeFixture({ types: '~0.186.0' }))).toEqual(
      expect.arrayContaining([expect.stringContaining('@types/three targets r186')])
    );
  });

  it.each([
    { runtime: '^0.185.1', types: '~0.185.4' },
    { runtime: '~0.185.1', types: '0.185.4' },
    { runtime: '~0.185', types: '~0.185.4' },
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
