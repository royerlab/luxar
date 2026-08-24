import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { afterEach, describe, expect, it } from 'vitest';
import {
  areFixturesStale,
  expectationsInputsFingerprint,
  fixtureInputFiles,
  fixtureInputsFingerprint,
  stampFixtureInputs,
} from '../../../../tools/fixture-freshness';

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
  writeFileSync(join(fixturesDir, 'generate_expectations.py'), 'EXPECTATIONS_VERSION = 1\n');
  writeFileSync(join(fixturesDir, 'roundtrip_expectations.json'), '{}\n');
  mkdirSync(join(fixturesDir, 'test.luxar.zarr'));
  writeFileSync(join(fixturesDir, 'test.luxar.zarr/.zmetadata'), '{}\n');
  writeFileSync(writerPath, 'WRITER_VERSION = 1\n');
  writeFileSync(
    join(projectRoot, 'packages/luxar/src/luxar/core/tests/test_writer.py'),
    'def test_writer(): pass\n'
  );
  writeFileSync(
    join(fixturesDir, '.fixture-inputs.sha256'),
    `${fixtureInputsFingerprint(projectRoot, fixturesDir)}\n`
  );
  return { fixturesDir, generatorPath, projectRoot, writerPath };
}

describe('generated fixture freshness', () => {
  it('becomes stale when production writer source changes', () => {
    const { fixturesDir, projectRoot, writerPath } = writeFixtureInputs();
    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(false);

    writeFileSync(writerPath, 'WRITER_VERSION = 2\n');

    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(true);
  });

  it('ignores Python test-only changes', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    writeFileSync(
      join(projectRoot, 'packages/luxar/src/luxar/core/tests/test_writer.py'),
      'def test_writer(): assert False\n'
    );

    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(false);
  });

  it('uses the same production-source exclusions as the Python fingerprint', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    const conftestPath = join(projectRoot, 'packages/luxar/src/luxar/conftest.py');
    writeFileSync(conftestPath, 'PYTEST_ONLY = 1\n');

    const relativeInputs = fixtureInputFiles(projectRoot, fixturesDir).map((path) =>
      path.slice(projectRoot.length + 1)
    );

    expect(relativeInputs).toContain('packages/luxar-viewer/tests/fixtures/generate_test_data.py');
    expect(relativeInputs).toContain('packages/luxar/src/luxar/_zarr_compat.py');
    expect(relativeInputs).not.toContain('packages/luxar/src/luxar/conftest.py');
    expect(relativeInputs).not.toContain('packages/luxar/src/luxar/core/tests/test_writer.py');
  });

  it('produces the same digest through a symlinked checkout path', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    const symlinkRoot = `${projectRoot}-link`;
    symlinkSync(projectRoot, symlinkRoot, 'dir');
    try {
      expect(
        fixtureInputsFingerprint(
          symlinkRoot,
          join(symlinkRoot, 'packages/luxar-viewer/tests/fixtures')
        )
      ).toBe(fixtureInputsFingerprint(projectRoot, fixturesDir));
    } finally {
      rmSync(symlinkRoot, { force: true });
    }
  });

  it('stamps complete fixtures and expectations with the current inputs', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    rmSync(join(fixturesDir, '.fixture-inputs.sha256'));

    stampFixtureInputs(projectRoot, fixturesDir);

    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(false);
    expect(readFileSync(join(fixturesDir, '.fixture-inputs.sha256'), 'utf8').trim()).toBe(
      fixtureInputsFingerprint(projectRoot, fixturesDir)
    );
    expect(readFileSync(join(fixturesDir, '.expectations-inputs.sha256'), 'utf8').trim()).toBe(
      expectationsInputsFingerprint(projectRoot, fixturesDir)
    );
  });

  it('refuses to stamp an incomplete fixture set', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    rmSync(join(fixturesDir, '.fixture-inputs.sha256'));
    rmSync(join(fixturesDir, 'test.luxar.zarr'), { recursive: true });

    expect(() => stampFixtureInputs(projectRoot, fixturesDir)).toThrow(
      'Cannot stamp missing or incomplete generated fixtures: test.luxar.zarr'
    );
    expect(() => readFileSync(join(fixturesDir, '.fixture-inputs.sha256'))).toThrow();
  });
});
