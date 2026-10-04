import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

import {
  ISSUE_TYPES,
  compareToBaseline,
  configFingerprint,
  issueKeys,
  parseArgs,
  runKnip,
} from './check-knip-ratchet.mjs';

const row = (file, issues) => ({
  file,
  ...Object.fromEntries(ISSUE_TYPES.map((t) => [t, []])),
  ...issues,
});

describe('issueKeys', () => {
  it('keys every issue type by type, file and symbol, sorted and unique', () => {
    const report = {
      issues: [
        row('src/b.ts', { types: [{ name: 'Shape' }], exports: [{ name: 'make' }] }),
        row('src/a.ts', {
          enumMembers: [{ namespace: 'Mode', name: 'Legacy' }],
          duplicates: [[{ name: 'zeta' }, { name: 'alpha' }]],
          exports: [{ name: 'make' }, { name: 'make' }],
        }),
      ],
    };
    expect(issueKeys(report)).toEqual([
      'duplicates:src/a.ts::alpha|zeta',
      'enumMembers:src/a.ts::Mode.Legacy',
      'exports:src/a.ts::make',
      'exports:src/b.ts::make',
      'types:src/b.ts::Shape',
    ]);
  });

  it('refuses a report without an issues array (fail closed)', () => {
    expect(() => issueKeys({})).toThrow(/no `issues` array/);
  });
});

describe('compareToBaseline', () => {
  const baseline = ['exports:src/a.ts::old', 'types:src/a.ts::Kept'];

  it('reports a new entry', () => {
    const result = compareToBaseline(baseline, [...baseline, 'exports:src/c.ts::fresh']);
    expect(result).toEqual({ added: ['exports:src/c.ts::fresh'], removed: [], moved: [] });
  });

  it('reports a paid-down entry', () => {
    const result = compareToBaseline(baseline, ['types:src/a.ts::Kept']);
    expect(result).toEqual({ added: [], removed: ['exports:src/a.ts::old'], moved: [] });
  });

  it('pairs a vanished and a new key of the same symbol as one move', () => {
    const result = compareToBaseline(baseline, [
      'exports:src/moved/a.ts::old',
      'types:src/a.ts::Kept',
    ]);
    expect(result).toEqual({
      added: [],
      removed: [],
      moved: [{ from: 'exports:src/a.ts::old', to: 'exports:src/moved/a.ts::old' }],
    });
  });

  it('does not pair across issue types', () => {
    const result = compareToBaseline(baseline, ['types:src/b.ts::old', 'types:src/a.ts::Kept']);
    expect(result.moved).toEqual([]);
    expect(result.added).toEqual(['types:src/b.ts::old']);
  });

  it('restricts both sides to the scope prefixes', () => {
    const current = ['exports:src/ui/x.ts::fresh', 'exports:src/data/y.ts::elsewhere'];
    expect(compareToBaseline(baseline, current, ['src/ui/'])).toEqual({
      added: ['exports:src/ui/x.ts::fresh'],
      removed: [],
      moved: [],
    });
  });
});

describe('configFingerprint', () => {
  const base = { knipConfig: { entry: ['a.ts'] }, knipVersion: '6.0.0' };

  it('changes with the config, the knip version and the issue types', () => {
    const fp = configFingerprint(base);
    expect(configFingerprint({ ...base, knipConfig: { entry: ['b.ts'] } })).not.toBe(fp);
    expect(configFingerprint({ ...base, knipVersion: '6.0.1' })).not.toBe(fp);
    expect(configFingerprint({ ...base, issueTypes: ['exports'] })).not.toBe(fp);
    expect(configFingerprint({ ...base })).toBe(fp);
  });

  it('matches the committed baseline for the committed config', () => {
    const baseline = JSON.parse(readFileSync(new URL('../knip-baseline.json', import.meta.url)));
    const knipConfig = JSON.parse(readFileSync(new URL('../knip.json', import.meta.url)));
    const knipPackage = new URL('../node_modules/knip/package.json', import.meta.url);
    const knipVersion = JSON.parse(readFileSync(knipPackage)).version;
    expect(baseline.fingerprint).toBe(configFingerprint({ knipConfig, knipVersion }));
  });
});

describe('parseArgs', () => {
  it('collects scope prefixes and the update flag', () => {
    expect(parseArgs(['./src/ui', 'src/data'])).toEqual({
      updateBaseline: false,
      scope: ['src/ui', 'src/data'],
    });
    expect(parseArgs(['--update-baseline'])).toEqual({ updateBaseline: true, scope: [] });
  });

  it('refuses an update restricted to paths, and unknown options', () => {
    expect(() => parseArgs(['--update-baseline', 'src/ui'])).toThrow(/drop every baselined entry/);
    expect(() => parseArgs(['--frobnicate'])).toThrow(/Unknown option/);
  });
});

describe('runKnip', () => {
  it('fails closed when knip exits non-zero or prints non-JSON', () => {
    const failing = () => ({ status: 2, stdout: '', stderr: 'boom' });
    expect(() => runKnip(ISSUE_TYPES, failing)).toThrow(/knip failed \(exit 2\):\nboom/);
    const garbled = () => ({ status: 0, stdout: 'not json', stderr: '' });
    expect(() => runKnip(ISSUE_TYPES, garbled)).toThrow(/not JSON/);
  });

  it('asks knip for exactly the ratcheted issue types as JSON', () => {
    let seen;
    const spy = (_cmd, args) => {
      seen = args;
      return { status: 0, stdout: '{"issues":[]}', stderr: '' };
    };
    expect(runKnip(ISSUE_TYPES, spy)).toEqual({ issues: [] });
    expect(seen).toEqual(expect.arrayContaining(['--reporter', 'json', '--no-exit-code']));
    expect(seen[seen.indexOf('--include') + 1]).toBe(ISSUE_TYPES.join(','));
  });
});
