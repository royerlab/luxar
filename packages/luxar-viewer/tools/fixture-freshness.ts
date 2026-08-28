/** Freshness fingerprints for Python-generated viewer fixtures. */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  FIXTURES_REPO_RELATIVE_PATH,
  isGeneratedFixtureComplete,
  parseGeneratedFixtureNames,
} from './fixture-manifest';

const VIEWER_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');

type FixtureGenerator = (scriptPath: string) => void;

export const FIXTURE_GENERATOR_TIMEOUT_MS =
  Number(process.env.LUXAR_FIXTURE_GEN_TIMEOUT_MS) || 1_200_000;

// Hatch expands braces in command arguments; keep this program free of brace syntax.
const IMPORT_CLOSURE_PROGRAM = `
import json
import sys
from pathlib import Path

from luxar.utils.source_fingerprints import imported_source_files

project_root = Path(sys.argv[1]).resolve()
generators = [Path(path) for path in sys.argv[2:4]]
import_roots = tuple(Path(path) for path in sys.argv[4:])
closures = [imported_source_files(generator, import_roots, within=project_root) for generator in generators]
print(json.dumps([[path.relative_to(project_root).as_posix() for path in sources] for sources in closures]))
`;

interface FixtureImportClosures {
  expectations: string[];
  fixtures: string[];
}

const fixtureImportClosures = new Map<string, FixtureImportClosures>();

/**
 * Resolve fixture producer imports through Python so fixtures, demos, and
 * examples share one staleness model instead of reimplementing it in TS.
 * Static resolution cannot see string-built imports or non-Python inputs.
 * `PROJECT_ROOT` owns the Hatch environment; `projectRoot` is the tree being
 * fingerprinted and may be a temporary or symlinked checkout.
 */
function resolveFixtureImportClosures(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): FixtureImportClosures {
  const cacheKey = `${projectRoot}\0${fixturesDir}`;
  const cached = fixtureImportClosures.get(cacheKey);
  if (cached) return cached;

  let output: string;
  try {
    output = execFileSync(
      'hatch',
      [
        'run',
        'fixtures:python',
        '-c',
        IMPORT_CLOSURE_PROGRAM,
        projectRoot,
        resolve(fixturesDir, 'generate_test_data.py'),
        resolve(fixturesDir, 'generate_expectations.py'),
        resolve(projectRoot, 'packages/luxar/src'),
        fixturesDir,
      ],
      { cwd: PROJECT_ROOT, encoding: 'utf8', timeout: FIXTURE_GENERATOR_TIMEOUT_MS }
    );
  } catch (error: unknown) {
    if (error && typeof error === 'object' && (error as { code?: string }).code === 'ETIMEDOUT') {
      throw new Error(
        `Fixture import resolution exceeded the ${FIXTURE_GENERATOR_TIMEOUT_MS} ms budget. ` +
          'Raise it with LUXAR_FIXTURE_GEN_TIMEOUT_MS if this machine is slower.',
        { cause: error }
      );
    }
    throw error;
  }

  const encodedPaths = output.trim().split(/\r?\n/).at(-1);
  const closures: unknown = JSON.parse(encodedPaths ?? '[]');
  if (
    !Array.isArray(closures) ||
    closures.length !== 2 ||
    closures.some(
      (paths) => !Array.isArray(paths) || paths.some((path) => typeof path !== 'string')
    )
  ) {
    throw new Error('Fixture import resolver returned an invalid source list');
  }
  const [fixturePaths, expectationPaths] = closures as string[][];
  const resolved = {
    fixtures: fixturePaths.map((path) => resolve(projectRoot, path)).sort(),
    expectations: [...new Set([...fixturePaths, ...expectationPaths])]
      .map((path) => resolve(projectRoot, path))
      .sort(),
  };
  fixtureImportClosures.set(cacheKey, resolved);
  return resolved;
}

/** Every local Python source reachable from the fixture generator's imports. */
export function fixtureInputFiles(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): string[] {
  return [...resolveFixtureImportClosures(projectRoot, fixturesDir).fixtures];
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

/**
 * Whether the fixtures were produced by the current input content.
 *
 * The generate-if-missing gate alone let #448 through with complete but stale
 * stores. Do not replace this digest with mtimes: `git checkout` and fresh
 * worktrees rewrite mtimes without changing the bytes that determine output.
 */
export function areFixturesStale(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): boolean {
  return (
    readStamp(resolve(fixturesDir, '.fixture-inputs.sha256')) !==
    fixtureInputsFingerprint(projectRoot, fixturesDir)
  );
}

/**
 * Digest the inputs that describe the expectations rather than statting every
 * generated zarr tree. This avoids mtime churn while still tracking fixture
 * producer changes and the expectations generator itself.
 */
export function expectationsInputsFingerprint(
  projectRoot: string = PROJECT_ROOT,
  fixturesDir: string = resolve(projectRoot, FIXTURES_REPO_RELATIVE_PATH)
): string {
  return hashFiles(
    projectRoot,
    resolveFixtureImportClosures(projectRoot, fixturesDir).expectations
  );
}

/**
 * Whether round-trip expectations must be regenerated.
 *
 * `regeneratedFixtures` is load-bearing: `generate_test_data.py` is not
 * byte-reproducible, so rebuilding fixtures from unchanged inputs can still
 * change encoded values. Fixtures rebuilt therefore always means expectations
 * rebuilt, or the round-trip tests compare against values from different stores.
 */
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
