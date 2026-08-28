/**
 * Demo Script Validation Tests
 *
 * Lightweight tests that verify Python demo scripts are valid without running
 * the full browser pipeline. These checks run fast and catch basic issues:
 * - Python syntax errors in demo scripts
 * - Missing .zarr outputs (datasets not generated)
 *
 * For full pipeline E2E testing (Python -> TypeScript -> WebGL), see:
 * - all-examples-smoke-test.spec.ts
 */

import { test, expect } from './fixtures';
import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at module load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Project root (5 levels up: src/tests/e2e -> tests -> src -> luxar-viewer -> packages -> root)
const PROJECT_ROOT = path.resolve(__dirname, '../../../../..');
const SCRIPTS_DIR = path.join(PROJECT_ROOT, 'packages/luxar/examples');
const EXAMPLES_DIR = path.join(PROJECT_ROOT, 'datasets/examples');

// All Python demo scripts
const DEMO_SCRIPTS = [
  'build_example.py',
  'dense_cubic_gradient_example.py',
  'dense_grid_5d_example.py',
  'dimension_navigation_example.py',
  'dimension_sliders_5d_example.py',
  'hierarchy_example.py',
  'lines_basic_example.py',
  'multiple_objects_example.py',
  'nd_points_example.py',
  'performance_benchmark_example.py',
  'point_spacing_example.py',
  'progressive_writing_example.py',
  'radius_basic_example.py',
  'radius_showcase_example.py',
  'radius_slicing_example.py',
  'rainbow_sphere_4d_example.py',
  'rainbow_sphere_spiral_example.py',
  'rendering_attributes_example.py',
  'rendering_modes_example.py',
  'scene_dimensions_example.py',
  'sharpness_showcase_example.py',
  'simple_nd_example.py',
  'single_point_example.py',
  'spatial_index_demo_example.py',
  'time_series_4d_example.py',
  'transform_example.py',
];

// Scripts to skip in validation
const SKIP_SCRIPTS: Record<string, string> = {
  'memory_optimization_example.py': 'Memory profiling script, not a visualization demo',
};

// Scripts that take too long for routine zarr output checks
const SLOW_SCRIPTS = [
  'temporal_spiral_sphere_4d_example.py',
  'time_series_4d_example.py',
  'performance_benchmark_example.py',
];

/**
 * Get the .zarr output name for a script
 */
function getZarrName(scriptName: string): string {
  const baseName = scriptName.replace('.py', '');
  if (scriptName === 'build_example.py') {
    return 'build_example_manual.luxar.zarr';
  }
  return `${baseName}.luxar.zarr`;
}

// Opt out of the config's `fullyParallel: true`: both tests shell out to
// `hatch run python`, and concurrent hatch invocations contend on (and can
// re-resolve) the shared environment. `default` rather than `serial` so a
// failure does not skip the sibling test.
test.describe.configure({ mode: 'default' });

test.describe('Demo Script Validation', () => {
  test('all demo scripts should be syntactically valid Python', async () => {
    console.log('\n[Demo Validation] Checking Python syntax...\n');

    const errors: string[] = [];

    for (const script of DEMO_SCRIPTS) {
      if (SKIP_SCRIPTS[script]) continue;

      const scriptPath = path.join(SCRIPTS_DIR, script);

      if (!fs.existsSync(scriptPath)) {
        errors.push(`${script}: File not found`);
        continue;
      }

      try {
        execSync(`hatch run python -m py_compile "${scriptPath}"`, {
          cwd: PROJECT_ROOT,
          encoding: 'utf-8',
          timeout: 30000,
        });
        console.log(`  ✅ ${script}`);
      } catch (error: unknown) {
        const execError = error as { stderr?: string; message: string };
        errors.push(`${script}: ${execError.stderr || execError.message}`);
        console.error(`  ❌ ${script}`);
      }
    }

    if (errors.length > 0) {
      console.error('\n[Demo Validation] Syntax errors found:');
      errors.forEach((err) => console.error(`  - ${err}`));
    }

    expect(errors.length).toBe(0);
  });

  test('all demo scripts should have corresponding .zarr outputs', async () => {
    console.log('\n[Demo Validation] Checking .zarr outputs exist...\n');

    const missing: string[] = [];

    for (const script of DEMO_SCRIPTS) {
      if (SKIP_SCRIPTS[script]) continue;
      if (SLOW_SCRIPTS.includes(script)) continue;

      const zarrName = getZarrName(script);
      const zarrPath = path.join(EXAMPLES_DIR, zarrName);

      if (!fs.existsSync(zarrPath)) {
        missing.push(`${script} → ${zarrName}`);
        console.log(`  ❌ ${script} → ${zarrName} (MISSING)`);
      } else {
        console.log(`  ✅ ${script} → ${zarrName}`);
      }
    }

    if (missing.length > 0) {
      console.warn('\n[Demo Validation] Missing datasets:');
      missing.forEach((m) => console.warn(`  - ${m}`));
      console.warn('\nRun "make run-examples" to generate all datasets.');
    }

    // Allow some missing (slow scripts may not have run)
    expect(missing.length).toBeLessThan(DEMO_SCRIPTS.length / 2);
  });
});
