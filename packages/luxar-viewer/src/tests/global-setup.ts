/**
 * Vitest global setup — runs once before the entire test suite.
 *
 * Ensures Python-generated zarr test fixtures exist.
 * If any are missing, runs the generator script automatically.
 */

import { existsSync, readFileSync } from 'fs';
import { execSync } from 'child_process';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

// Use import.meta.url for reliable path resolution in vitest global setup
const THIS_DIR = resolve(fileURLToPath(import.meta.url), '..');
const VIEWER_ROOT = resolve(THIS_DIR, '../..');
const FIXTURES_DIR = resolve(VIEWER_ROOT, 'tests/fixtures');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');
const GENERATOR_PATH = resolve(VIEWER_ROOT, 'tests/fixtures/generate_test_data.py');

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

export async function setup(): Promise<void> {
  const missing = EXPECTED_FIXTURES.filter((name) => !existsSync(resolve(FIXTURES_DIR, name)));

  if (missing.length === 0) {
    return;
  }

  console.log(`\n[test-setup] ${missing.length} zarr fixture(s) missing — generating...`);

  try {
    execSync('hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py', {
      cwd: PROJECT_ROOT,
      stdio: 'pipe',
      timeout: 120_000,
    });
    console.log('[test-setup] Fixtures generated successfully.\n');
  } catch (err: unknown) {
    // execSync errors include stderr/stdout as Buffers
    const stderr =
      err && typeof err === 'object' && 'stderr' in err && err.stderr ? String(err.stderr) : '';
    const message = err instanceof Error ? err.message : String(err);
    console.error(`[test-setup] Failed to generate fixtures: ${message}`);
    if (stderr) console.error(stderr);
    console.error(
      '[test-setup] Run manually: hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py'
    );
    throw new Error('Test fixture generation failed. See above for details.');
  }

  // Verify generation succeeded
  const stillMissing = EXPECTED_FIXTURES.filter((name) => !existsSync(resolve(FIXTURES_DIR, name)));
  if (stillMissing.length > 0) {
    throw new Error(
      `Fixture generation ran but these are still missing: ${stillMissing.join(', ')}`
    );
  }
}
