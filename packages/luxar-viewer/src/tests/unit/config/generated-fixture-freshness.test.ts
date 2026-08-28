import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { dirname, join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { afterEach, describe, expect, it } from 'vitest';
import {
  areFixturesStale,
  ensureGeneratedFixtures,
  expectationsInputsFingerprint,
  fixtureInputFiles,
  fixtureInputsFingerprint,
  stampExpectationsInputs,
  stampFixtureInputs,
} from '../../../../tools/fixture-freshness';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../../..');
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
  writeFileSync(
    generatorPath,
    'from luxar._zarr_compat import WRITER_VERSION\nFIXTURE_NAMES = ["test.luxar.zarr"]\n'
  );
  writeFileSync(join(fixturesDir, 'generate_expectations.py'), 'EXPECTATIONS_VERSION = 1\n');
  writeFileSync(join(fixturesDir, 'roundtrip_expectations.json'), '{}\n');
  mkdirSync(join(fixturesDir, 'test.luxar.zarr'));
  writeFileSync(join(fixturesDir, 'test.luxar.zarr/.zmetadata'), '{}\n');
  writeFileSync(join(projectRoot, 'packages/luxar/src/luxar/__init__.py'), 'PACKAGE = 1\n');
  writeFileSync(writerPath, 'WRITER_VERSION = 1\n');
  writeFileSync(join(projectRoot, 'packages/luxar/src/luxar/core/unrelated.py'), 'UNRELATED = 1\n');
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

  it('tracks the producer import closure instead of every production source', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();

    const relativeInputs = fixtureInputFiles(projectRoot, fixturesDir).map((path) =>
      path.slice(projectRoot.length + 1)
    );

    expect(relativeInputs).toContain('packages/luxar-viewer/tests/fixtures/generate_test_data.py');
    expect(relativeInputs).toContain('packages/luxar/src/luxar/__init__.py');
    expect(relativeInputs).toContain('packages/luxar/src/luxar/_zarr_compat.py');
    expect(relativeInputs).not.toContain('packages/luxar/src/luxar/core/unrelated.py');
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

  it('stamps complete fixtures independently of expectations', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    rmSync(join(fixturesDir, '.fixture-inputs.sha256'));
    rmSync(join(fixturesDir, 'roundtrip_expectations.json'));

    stampFixtureInputs(projectRoot, fixturesDir);

    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(false);
    expect(readFileSync(join(fixturesDir, '.fixture-inputs.sha256'), 'utf8').trim()).toBe(
      fixtureInputsFingerprint(projectRoot, fixturesDir)
    );
  });

  it('stamps expectations only when their output exists', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();

    stampExpectationsInputs(projectRoot, fixturesDir);

    expect(readFileSync(join(fixturesDir, '.expectations-inputs.sha256'), 'utf8').trim()).toBe(
      expectationsInputsFingerprint(projectRoot, fixturesDir)
    );

    rmSync(join(fixturesDir, 'roundtrip_expectations.json'));
    expect(() => stampExpectationsInputs(projectRoot, fixturesDir)).toThrow(
      'Cannot stamp missing roundtrip_expectations.json'
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

  it('does not run generators when fixtures and expectations are current', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    stampExpectationsInputs(projectRoot, fixturesDir);
    const generated: string[] = [];

    ensureGeneratedFixtures(projectRoot, fixturesDir, (script) => generated.push(script));

    expect(generated).toEqual([]);
  });

  it('regenerates and stamps stale fixtures before an E2E run', () => {
    const { fixturesDir, projectRoot, writerPath } = writeFixtureInputs();
    stampExpectationsInputs(projectRoot, fixturesDir);
    writeFileSync(writerPath, 'WRITER_VERSION = 2\n');
    const generated: string[] = [];

    ensureGeneratedFixtures(projectRoot, fixturesDir, (script) => generated.push(script));

    expect(generated.map((script) => script.slice(script.lastIndexOf('/') + 1))).toEqual([
      'generate_test_data.py',
      'generate_expectations.py',
    ]);
    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(false);
  });

  it('regenerates an incomplete fixture set before stamping it current', () => {
    const { fixturesDir, projectRoot } = writeFixtureInputs();
    stampExpectationsInputs(projectRoot, fixturesDir);
    rmSync(join(fixturesDir, 'test.luxar.zarr'), { recursive: true });
    const generated: string[] = [];

    ensureGeneratedFixtures(projectRoot, fixturesDir, (script) => {
      generated.push(script);
      if (script.endsWith('generate_test_data.py')) {
        mkdirSync(join(fixturesDir, 'test.luxar.zarr'));
        writeFileSync(join(fixturesDir, 'test.luxar.zarr/.zmetadata'), '{}\n');
      }
    });

    expect(generated.map((script) => script.slice(script.lastIndexOf('/') + 1))).toEqual([
      'generate_test_data.py',
      'generate_expectations.py',
    ]);
    expect(areFixturesStale(projectRoot, fixturesDir)).toBe(false);
  });

  it('wires the generated-fixture ensure step into full E2E', () => {
    const makefile = readFileSync(resolve(REPO_ROOT, 'Makefile'), 'utf8');

    expect(makefile).toContain('test-e2e: run-examples ensure-viewer-fixtures ');
  });
});
