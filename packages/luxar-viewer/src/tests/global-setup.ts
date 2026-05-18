/**
 * Vitest global setup — runs once before the entire test suite.
 *
 * Ensures Python-generated zarr test fixtures exist.
 * If any are missing, runs the generator script automatically.
 */

import { existsSync, readFileSync, statSync } from 'fs';
import { execSync } from 'child_process';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

// Use import.meta.url for reliable path resolution in vitest global setup
const THIS_DIR = resolve(fileURLToPath(import.meta.url), '..');
const VIEWER_ROOT = resolve(THIS_DIR, '../..');
const FIXTURES_DIR = resolve(VIEWER_ROOT, 'tests/fixtures');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');
const GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_test_data.py');
const EXPECTATIONS_GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_expectations.py');
const EXPECTATIONS_PATH = resolve(FIXTURES_DIR, 'roundtrip_expectations.json');

/**
 * Parse fixture names from generate_test_data.py — the single source of truth.
 *
 * Each fixture is written by an `output = FIXTURES_DIR / "<name>.zarr"` line.
 * Deriving the list at runtime instead of duplicating it here means the TS
 * side cannot silently drift when fixtures are added or renamed in Python.
 */
function parseGeneratedFixtureNames(): string[] {
  const source = readFileSync(GENERATOR_PATH, 'utf-8');
  const re = /FIXTURES_DIR\s*\/\s*"([^"]+\.zarr)"/g;
  const names = new Set<string>();
  let m: RegExpExecArray | null;
  while ((m = re.exec(source)) !== null) {
    names.add(m[1]);
  }
  if (names.size === 0) {
    throw new Error(
      `[test-setup] No fixtures parsed from ${GENERATOR_PATH} — ` +
        'expected pattern `FIXTURES_DIR / "test_*.zarr"`. ' +
        'The generator file moved or its structure changed; update this parser.'
    );
  }
  return [...names].sort();
}

const EXPECTED_FIXTURES = parseGeneratedFixtureNames();

function runPythonGenerator(command: string, label: string): void {
  try {
    execSync(command, {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      timeout: 120_000,
    });
    console.log(`[test-setup] ${label} generated successfully.`);
  } catch (err: unknown) {
    // execSync errors include stderr/stdout as Buffers
    const stderr =
      err && typeof err === 'object' && 'stderr' in err && err.stderr ? String(err.stderr) : '';
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[test-setup] Failed to generate ${label}: ${message}`);
    if (stderr) console.error(stderr);
    console.error(`[test-setup] Run manually: ${command}`);
    throw new Error(`${label} generation failed. See above for details.`);
  }
}

function isExpectationsStale(fixtureNames: string[]): boolean {
  if (!existsSync(EXPECTATIONS_PATH)) return true;

  const expectationsMtime = statSync(EXPECTATIONS_PATH).mtimeMs;
  const dependencyPaths = [
    GENERATOR_PATH,
    EXPECTATIONS_GENERATOR_PATH,
    ...fixtureNames.map((name) => resolve(FIXTURES_DIR, name)),
  ];

  return dependencyPaths.some(
    (dependencyPath) =>
      existsSync(dependencyPath) && statSync(dependencyPath).mtimeMs > expectationsMtime
  );
}

export async function setup(): Promise<void> {
  const missing = EXPECTED_FIXTURES.filter((name) => !existsSync(resolve(FIXTURES_DIR, name)));

  if (missing.length > 0) {
    console.log(`\n[test-setup] ${missing.length} zarr fixture(s) missing — generating...`);
    runPythonGenerator(
      'hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py',
      'fixtures'
    );
  }

  // Verify fixture generation succeeded before generating expectations from them.
  const stillMissing = EXPECTED_FIXTURES.filter((name) => !existsSync(resolve(FIXTURES_DIR, name)));
  if (stillMissing.length > 0) {
    throw new Error(
      `Fixture generation ran but these are still missing: ${stillMissing.join(', ')}`
    );
  }

  if (isExpectationsStale(EXPECTED_FIXTURES)) {
    console.log('[test-setup] Round-trip expectations missing/stale — generating...');
    runPythonGenerator(
      'hatch run python packages/luxar-viewer/tests/fixtures/generate_expectations.py',
      'round-trip expectations'
    );
  }
}
