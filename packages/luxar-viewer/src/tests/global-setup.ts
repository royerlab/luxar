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
 * Reads the declarative `FIXTURE_NAMES: list[str] = [...]` constant at the
 * top of the Python generator (audit C1 viewer-integration-fixtures fix).
 * Previously this regex matched scattered `FIXTURES_DIR / "..."` usages,
 * which silently broke if a generator function switched to single quotes,
 * f-strings, or path concatenation. Targeting a single canonical
 * declaration is robust to those variations.
 *
 * The Python script asserts at the end of main() that every name in
 * FIXTURE_NAMES was actually produced — keeping the manifest and the
 * generators in sync.
 */
function parseGeneratedFixtureNames(): string[] {
  const source = readFileSync(GENERATOR_PATH, 'utf-8');
  // Match the FIXTURE_NAMES list declaration. The body captures
  // everything between the [ and ] including newlines; we then pull out
  // each "..." or '...' literal ending in .zarr.
  const listMatch = /FIXTURE_NAMES\s*(?::[^=]*)?=\s*\[([^\]]+)\]/.exec(source);
  if (!listMatch) {
    throw new Error(
      `[test-setup] FIXTURE_NAMES declaration not found in ${GENERATOR_PATH}. ` +
        'Expected a top-level `FIXTURE_NAMES: list[str] = [...]` block. ' +
        'Update generate_test_data.py to declare the manifest, or update this parser.'
    );
  }
  const body = listMatch[1];
  const re = /['"]([^'"]+\.zarr)['"]/g;
  const names = new Set<string>();
  let m: RegExpExecArray | null;
  while ((m = re.exec(body)) !== null) {
    names.add(m[1]);
  }
  if (names.size === 0) {
    throw new Error(
      `[test-setup] FIXTURE_NAMES list in ${GENERATOR_PATH} is empty or unparseable. ` +
        'Expected `.zarr`-terminated string literals.'
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
