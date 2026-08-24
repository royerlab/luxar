import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import { areFixturesStale, fixtureInputsFingerprint } from '../../global-setup';

let temporaryRoot: string | undefined;

afterEach(() => {
  if (temporaryRoot) rmSync(temporaryRoot, { force: true, recursive: true });
  temporaryRoot = undefined;
});

function writeFixtureInputs(): {
  fixturesDir: string;
  generatorPath: string;
  projectRoot: string;
  writerPath: string;
} {
  temporaryRoot = mkdtempSync(join(tmpdir(), 'luxar-fixture-freshness-'));
  const projectRoot = temporaryRoot;
  const fixturesDir = join(projectRoot, 'packages/luxar-viewer/tests/fixtures');
  const generatorPath = join(fixturesDir, 'generate_test_data.py');
  const writerPath = join(projectRoot, 'packages/luxar/src/luxar/_zarr_compat.py');
  mkdirSync(fixturesDir, { recursive: true });
  mkdirSync(join(projectRoot, 'packages/luxar/src/luxar/core/tests'), { recursive: true });
  writeFileSync(generatorPath, 'FIXTURE_NAMES = ["test.luxar.zarr"]\n');
  writeFileSync(writerPath, 'WRITER_VERSION = 1\n');
  writeFileSync(
    join(projectRoot, 'packages/luxar/src/luxar/core/tests/test_writer.py'),
    'def test_writer(): pass\n'
  );
  writeFileSync(
    join(fixturesDir, '.fixture-inputs.sha256'),
    `${fixtureInputsFingerprint(projectRoot, generatorPath)}\n`
  );
  return { fixturesDir, generatorPath, projectRoot, writerPath };
}

describe('generated fixture freshness', () => {
  it('becomes stale when production writer source changes', () => {
    const { fixturesDir, generatorPath, projectRoot, writerPath } = writeFixtureInputs();
    expect(areFixturesStale(projectRoot, fixturesDir, generatorPath)).toBe(false);

    writeFileSync(writerPath, 'WRITER_VERSION = 2\n');

    expect(areFixturesStale(projectRoot, fixturesDir, generatorPath)).toBe(true);
  });

  it('ignores Python test-only changes', () => {
    const { fixturesDir, generatorPath, projectRoot } = writeFixtureInputs();
    writeFileSync(
      join(projectRoot, 'packages/luxar/src/luxar/core/tests/test_writer.py'),
      'def test_writer(): assert False\n'
    );

    expect(areFixturesStale(projectRoot, fixturesDir, generatorPath)).toBe(false);
  });
});
