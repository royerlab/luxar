/** Freshness fingerprints for Python-generated viewer fixtures. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  FIXTURES_REPO_RELATIVE_PATH,
  isGeneratedFixtureComplete,
  parseGeneratedFixtureNames,
} from './fixture-manifest';

const VIEWER_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');

type FixtureGenerator = (scriptPath: string) => void;

/**
 * Every Python file whose content can change what the generator writes.
 *
 * The wide production set is deliberate: compiler behavior also depends on
 * root-level compatibility code and `core/`, not only `encoding/` and `io/`.
 * Test trees, `__pycache__`, and `conftest.py` cannot change a fixture byte and
 * are excluded so test-only edits do not trigger a costly regeneration.
 */
export function fixtureInputFiles(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): string[] {
  const generatorPath = resolve(fixturesDir, 'generate_test_data.py');
  const files = [generatorPath];
  const sourceRoot = resolve(projectRoot, 'packages/luxar/src/luxar');
  if (!existsSync(sourceRoot)) return files;
  for (const entry of readdirSync(sourceRoot, { recursive: true }) as string[]) {
    if (!entry.endsWith('.py')) continue;
    const parts = entry.split(/[\\/]/);
    if (parts.includes('tests') || parts.includes('__pycache__')) continue;
    if (parts[parts.length - 1] === 'conftest.py') continue;
    const full = join(sourceRoot, entry);
    if (existsSync(full)) files.push(full);
  }
  return files.sort();
}

/** Content digest of paths, including project-relative names so renames count. */
function hashFiles(projectRoot: string, paths: string[]): string {
  const digest = createHash('sha256');
  for (const path of paths) {
    digest.update(relative(projectRoot, path).split(sep).join('/'));
    digest.update(readFileSync(path));
  }
  return digest.digest('hex');
}

function readStamp(path: string): string | null {
  return existsSync(path) ? readFileSync(path, 'utf-8').trim() : null;
}

export function fixtureInputsFingerprint(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): string {
  return hashFiles(projectRoot, fixtureInputFiles(projectRoot, fixturesDir));
}

export function areFixturesStale(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): boolean {
  return (
    readStamp(resolve(fixturesDir, '.fixture-inputs.sha256')) !==
    fixtureInputsFingerprint(projectRoot, fixturesDir)
  );
}

export function expectationsInputsFingerprint(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): string {
  return hashFiles(projectRoot, [
    ...fixtureInputFiles(projectRoot, fixturesDir),
    resolve(fixturesDir, 'generate_expectations.py'),
  ]);
}

export function areExpectationsStale(
  regeneratedFixtures: boolean,
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): boolean {
  if (regeneratedFixtures) return true;
  if (!existsSync(resolve(fixturesDir, 'roundtrip_expectations.json'))) return true;
  return (
    readStamp(resolve(fixturesDir, '.expectations-inputs.sha256')) !==
    expectationsInputsFingerprint(projectRoot, fixturesDir)
  );
}

/** Verify generated fixtures are complete, then record their input fingerprint. */
export function stampFixtureInputs(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): void {
  const generatorPath = resolve(fixturesDir, 'generate_test_data.py');
  const expected = parseGeneratedFixtureNames(generatorPath);
  const incomplete = expected.filter(
    (name) => !isGeneratedFixtureComplete(resolve(fixturesDir, name))
  );
  if (incomplete.length > 0) {
    throw new Error(
      `Cannot stamp missing or incomplete generated fixtures: ${incomplete.join(', ')}`
    );
  }

  writeFileSync(
    resolve(fixturesDir, '.fixture-inputs.sha256'),
    `${fixtureInputsFingerprint(projectRoot, fixturesDir)}\n`
  );
}

/** Verify round-trip expectations exist, then record their input fingerprint. */
export function stampExpectationsInputs(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): void {
  if (!existsSync(resolve(fixturesDir, 'roundtrip_expectations.json'))) {
    throw new Error('Cannot stamp missing roundtrip_expectations.json');
  }

  writeFileSync(
    resolve(fixturesDir, '.expectations-inputs.sha256'),
    `${expectationsInputsFingerprint(projectRoot, fixturesDir)}\n`
  );
}

function runFixtureGenerator(scriptPath: string, projectRoot: string = PROJECT_ROOT): void {
  execFileSync('hatch', ['run', 'fixtures:python', scriptPath], {
    cwd: projectRoot,
    stdio: 'inherit',
  });
}

/** Regenerate stale or incomplete fixture artifacts, otherwise return immediately. */
export function ensureGeneratedFixtures(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH),
  generate: FixtureGenerator = (scriptPath) => runFixtureGenerator(scriptPath, projectRoot)
): void {
  const fixtureGenerator = resolve(fixturesDir, 'generate_test_data.py');
  const expected = parseGeneratedFixtureNames(fixtureGenerator);
  const incomplete = expected.filter(
    (name) => !isGeneratedFixtureComplete(resolve(fixturesDir, name))
  );
  const regeneratedFixtures = incomplete.length > 0 || areFixturesStale(projectRoot, fixturesDir);

  if (regeneratedFixtures) generate(fixtureGenerator);

  const stillIncomplete = expected.filter(
    (name) => !isGeneratedFixtureComplete(resolve(fixturesDir, name))
  );
  if (stillIncomplete.length > 0) {
    throw new Error(
      `Fixture generation left missing or incomplete outputs: ${stillIncomplete.join(', ')}`
    );
  }
  stampFixtureInputs(projectRoot, fixturesDir);

  if (areExpectationsStale(regeneratedFixtures, projectRoot, fixturesDir)) {
    generate(resolve(fixturesDir, 'generate_expectations.py'));
  }
  stampExpectationsInputs(projectRoot, fixturesDir);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.includes('--ensure')) {
    ensureGeneratedFixtures();
    console.log('[fixture-freshness] Generated fixtures are current.');
  } else {
    stampFixtureInputs();
    stampExpectationsInputs();
    console.log('[fixture-freshness] Generated fixture stamps updated.');
  }
}
