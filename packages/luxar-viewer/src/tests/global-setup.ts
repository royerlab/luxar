/**
 * Vitest global setup — runs once before the entire test suite.
 *
 * Ensures Python-generated zarr test fixtures exist.
 * If any are missing, runs the generator script automatically.
 */

import { existsSync } from 'fs';
import { execSync } from 'child_process';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

/** All zarr fixtures that generate_test_data.py produces. */
const EXPECTED_FIXTURES = [
  'test_broadcasting.zarr',
  'test_lut.zarr',
  'test_quantization.zarr',
  'test_array_refs.zarr',
  'test_mixed.zarr',
  'test_4d.zarr',
  'test_hierarchical_transforms.zarr',
  'test_hdr_colors.zarr',
  'test_integer_colors.zarr',
  'test_log_scalar.zarr',
  'test_4d_scalar_lut.zarr',
  'test_uint16_quantization.zarr',
  'test_sharpness_range.zarr',
  'test_nd_transforms.zarr',
  'test_lines.zarr',
  'test_gsplats.zarr',
];

// Use import.meta.url for reliable path resolution in vitest global setup
const THIS_DIR = resolve(fileURLToPath(import.meta.url), '..');
const VIEWER_ROOT = resolve(THIS_DIR, '../..');
const FIXTURES_DIR = resolve(VIEWER_ROOT, 'tests/fixtures');
const PROJECT_ROOT = resolve(VIEWER_ROOT, '../..');

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
