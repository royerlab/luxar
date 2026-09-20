import { afterEach, describe, expect, it } from 'vitest';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { checkJsdomVersion } from './check-jsdom-version.mjs';

const roots = [];

function makeFixture({
  jsdom = '~30.0.1',
  ignore = ['version-update:semver-minor', 'version-update:semver-major'],
  includeJsdomIgnore = true,
} = {}) {
  const root = mkdtempSync(join(tmpdir(), 'luxar-jsdom-version-'));
  const viewerRoot = join(root, 'packages', 'luxar-viewer');
  mkdirSync(viewerRoot, { recursive: true });
  writeFileSync(
    join(viewerRoot, 'package.json'),
    JSON.stringify({ devDependencies: jsdom === null ? {} : { jsdom } })
  );
  mkdirSync(join(root, '.github'), { recursive: true });
  const ignoreLines = ignore.map((updateType) => `          - "${updateType}"`).join('\n');
  const jsdomIgnore = includeJsdomIgnore
    ? `      - dependency-name: "jsdom"\n        update-types:\n${ignoreLines}\n`
    : '';
  writeFileSync(
    join(root, '.github', 'dependabot.yml'),
    `version: 2\nupdates:\n  - package-ecosystem: "npm"\n    directory: "/packages/luxar-viewer"\n    ignore:\n      - dependency-name: "other"\n        update-types:\n          - "version-update:semver-minor"\n          - "version-update:semver-major"\n${jsdomIgnore}`
  );
  roots.push(root);
  return { viewerRoot, repoRoot: root };
}

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

describe('checkJsdomVersion', () => {
  it('accepts the jsdom 30.0 patch line with both Dependabot update types ignored', () => {
    const { viewerRoot, repoRoot } = makeFixture();
    expect(checkJsdomVersion(viewerRoot, repoRoot)).toEqual([]);
  });

  it.each([['^30.0.1'], ['~30.1.0'], ['30.0.1']])('rejects the range %s', (jsdom) => {
    const { viewerRoot, repoRoot } = makeFixture({ jsdom });
    expect(checkJsdomVersion(viewerRoot, repoRoot)).toEqual(
      expect.arrayContaining([expect.stringContaining('plain ~30.0.PATCH range')])
    );
  });

  it('rejects a missing jsdom declaration', () => {
    const { viewerRoot, repoRoot } = makeFixture({ jsdom: null });
    expect(checkJsdomVersion(viewerRoot, repoRoot)).toEqual(
      expect.arrayContaining([expect.stringContaining('devDependencies.jsdom')])
    );
  });

  it('rejects a missing jsdom Dependabot entry even when another ignore has both update types', () => {
    const { viewerRoot, repoRoot } = makeFixture({ includeJsdomIgnore: false });
    expect(checkJsdomVersion(viewerRoot, repoRoot)).toEqual(
      expect.arrayContaining([expect.stringContaining('must ignore jsdom updates')])
    );
  });

  it.each(['version-update:semver-minor', 'version-update:semver-major'])(
    'requires the %s ignore on jsdom itself',
    (missing) => {
      const { viewerRoot, repoRoot } = makeFixture({
        ignore: ['version-update:semver-minor', 'version-update:semver-major'].filter(
          (updateType) => updateType !== missing
        ),
      });
      expect(checkJsdomVersion(viewerRoot, repoRoot)).toEqual(
        expect.arrayContaining([expect.stringContaining(missing)])
      );
    }
  );
});
