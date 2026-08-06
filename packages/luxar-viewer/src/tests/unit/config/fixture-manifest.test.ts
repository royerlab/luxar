import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  FIXTURES_REPO_RELATIVE_PATH,
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
