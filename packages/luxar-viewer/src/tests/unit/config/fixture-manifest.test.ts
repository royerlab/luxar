import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  FIXTURES_REPO_RELATIVE_PATH,
  isGeneratedFixtureComplete,
  parseGeneratedFixtureNames,
} from '../../../../tools/fixture-manifest';

const VIEWER_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const PROJECT_ROOT = path.resolve(VIEWER_ROOT, '../..');

let temporaryRoot: string;

function writeGenerator(source: string): string {
  const file = path.join(temporaryRoot, 'generate_test_data.py');
  writeFileSync(file, source, 'utf-8');
  return file;
}

beforeEach(() => {
  temporaryRoot = mkdtempSync(path.join(tmpdir(), 'luxar-fixture-manifest-'));
});

afterEach(() => {
  rmSync(temporaryRoot, { recursive: true, force: true });
});

describe('parseGeneratedFixtureNames', () => {
  it('returns the FIXTURE_NAMES entries sorted and de-duplicated', () => {
    const file = writeGenerator(
      [
        'FIXTURE_NAMES: list[str] = [',
        '    "test_mixed.luxar.zarr",',
        "    'test_lines.luxar.zarr',",
        '    "test_mixed.luxar.zarr",',
        '    "test_standalone_gsplats.gsplats.zarr",',
        ']',
      ].join('\n')
    );

    expect(parseGeneratedFixtureNames(file)).toEqual([
      'test_lines.luxar.zarr',
      'test_mixed.luxar.zarr',
      'test_standalone_gsplats.gsplats.zarr',
    ]);
  });

  it('ignores .zarr literals outside the declaration', () => {
    // The reason the parser targets the declaration rather than scattered
    // `FIXTURES_DIR / "..."` usages: intermediates and temp paths must not
    // become preflight requirements.
    const file = writeGenerator(
      [
        'FIXTURE_NAMES = [',
        '    "test_lut.luxar.zarr",',
        ']',
        '',
        'def generate_scratch() -> None:',
        '    output = FIXTURES_DIR / "scratch_intermediate.luxar.zarr"',
      ].join('\n')
    );

    expect(parseGeneratedFixtureNames(file)).toEqual(['test_lut.luxar.zarr']);
  });

  it('throws naming the generator when the declaration is absent', () => {
    const file = writeGenerator('FIXTURES = ["test_lut.luxar.zarr"]\n');

    expect(() => parseGeneratedFixtureNames(file)).toThrow(/FIXTURE_NAMES declaration not found/);
    expect(() => parseGeneratedFixtureNames(file)).toThrow(file);
  });

  it('throws when the declaration holds no .zarr literals', () => {
    const file = writeGenerator('FIXTURE_NAMES: list[str] = [\n    "roundtrip.json",\n]\n');

    expect(() => parseGeneratedFixtureNames(file)).toThrow(/empty or unparseable/);
  });

  it('parses the real generator, which lives under FIXTURES_REPO_RELATIVE_PATH', () => {
    const fixturesDir = path.join(PROJECT_ROOT, FIXTURES_REPO_RELATIVE_PATH);
    const generator = path.join(fixturesDir, 'generate_test_data.py');
    expect(existsSync(generator)).toBe(true);

    const names = parseGeneratedFixtureNames(generator);
    expect(names.length).toBeGreaterThan(0);
    expect(names.every((name) => name.endsWith('.zarr'))).toBe(true);
  });
});

describe('isGeneratedFixtureComplete', () => {
  it('rejects a fixture that does not exist at all', () => {
    expect(isGeneratedFixtureComplete(path.join(temporaryRoot, 'absent.luxar.zarr'))).toBe(false);
  });

  it('rejects a directory the generator was interrupted while writing', () => {
    // The whole point: `existsSync` on the directory says yes, but nothing can read it.
    const stump = path.join(temporaryRoot, 'interrupted.luxar.zarr');
    mkdirSync(path.join(stump, 'points'), { recursive: true });
    writeFileSync(path.join(stump, '.zgroup'), '{"zarr_format": 2}', 'utf-8');

    expect(existsSync(stump)).toBe(true);
    expect(isGeneratedFixtureComplete(stump)).toBe(false);
  });

  it('accepts a fixture once consolidated metadata is written', () => {
    const complete = path.join(temporaryRoot, 'complete.luxar.zarr');
    mkdirSync(complete, { recursive: true });
    writeFileSync(path.join(complete, '.zgroup'), '{"zarr_format": 2}', 'utf-8');
    writeFileSync(path.join(complete, '.zmetadata'), '{"metadata": {}}', 'utf-8');

    expect(isGeneratedFixtureComplete(complete)).toBe(true);
  });

  it('accepts every fixture the checked-in generator manifest declares', () => {
    // Guards the predicate against a producer change: if the compiler ever stopped
    // writing `.zmetadata`, both global setups would demand a regeneration that can
    // never satisfy them.
    const fixturesDir = path.join(PROJECT_ROOT, FIXTURES_REPO_RELATIVE_PATH);
    const names = parseGeneratedFixtureNames(path.join(fixturesDir, 'generate_test_data.py'));
    const present = names.filter((name) => existsSync(path.join(fixturesDir, name)));

    // Skipped on a checkout where the fixtures were never generated.
    for (const name of present) {
      expect(isGeneratedFixtureComplete(path.join(fixturesDir, name)), name).toBe(true);
    }
  });
});
