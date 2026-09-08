import { afterEach, describe, expect, it } from 'vitest';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { checkNodeTypesVersion } from './check-node-types-version.mjs';

const roots = [];

function makeFixture({ nvm = '22', declared = '^22.12.0' } = {}) {
  const repoRoot = mkdtempSync(join(tmpdir(), 'luxar-node-types-'));
  const viewerRoot = join(repoRoot, 'packages', 'luxar-viewer');
  mkdirSync(viewerRoot, { recursive: true });
  writeFileSync(join(repoRoot, '.nvmrc'), `${nvm}\n`);
  writeFileSync(
    join(viewerRoot, 'package.json'),
    JSON.stringify({ devDependencies: { '@types/node': declared } })
  );
  roots.push(repoRoot);
  return { repoRoot, viewerRoot };
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('checkNodeTypesVersion', () => {
  it('passes when the package and runtime major agree', () => {
    const { viewerRoot, repoRoot } = makeFixture();
    expect(checkNodeTypesVersion(viewerRoot, repoRoot)).toEqual([]);
  });

  it.each(['v22.22.0', '22.22.0'])('accepts a full .nvmrc version: %s', (nvm) => {
    const { viewerRoot, repoRoot } = makeFixture({ nvm });
    expect(checkNodeTypesVersion(viewerRoot, repoRoot)).toEqual([]);
  });

  it('rejects an .nvmrc release alias', () => {
    const { viewerRoot, repoRoot } = makeFixture({ nvm: 'lts/jod' });
    expect(checkNodeTypesVersion(viewerRoot, repoRoot)).toEqual(
      expect.arrayContaining([expect.stringContaining('Expected single-major versions')])
    );
  });

  it('rejects a package range for another Node major', () => {
    const { viewerRoot, repoRoot } = makeFixture({ declared: '^26' });
    expect(checkNodeTypesVersion(viewerRoot, repoRoot)).toEqual(
      expect.arrayContaining([expect.stringContaining('targets Node 26')])
    );
  });

  it('rejects a range spanning multiple Node majors', () => {
    const { viewerRoot, repoRoot } = makeFixture({ declared: '^20.19.0 || >=22.12.0' });
    expect(checkNodeTypesVersion(viewerRoot, repoRoot)).toEqual(
      expect.arrayContaining([expect.stringContaining('Expected single-major versions')])
    );
  });

  it('rejects a missing declaration', () => {
    const { viewerRoot, repoRoot } = makeFixture();
    writeFileSync(join(viewerRoot, 'package.json'), JSON.stringify({ devDependencies: {} }));
    expect(checkNodeTypesVersion(viewerRoot, repoRoot)).toEqual([
      'package.json must declare devDependencies["@types/node"]',
    ]);
  });
});
