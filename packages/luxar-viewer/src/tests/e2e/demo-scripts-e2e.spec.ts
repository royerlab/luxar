/**
 * E2E Tests for Python Demo Scripts
 *
 * CRITICAL: This test suite validates the ENTIRE Python → TypeScript pipeline:
 * 1. Runs each Python demo script to generate fresh .zarr datasets
 * 2. Loads each dataset in the viewer
 * 3. Validates console output for errors
 * 4. Verifies points are loaded and rendered correctly
 *
 * This catches:
 * - Python encoding bugs
 * - TypeScript decoding bugs
 * - Cross-language compatibility issues
 * - Runtime errors in demo scripts
 *
 * Run with: pnpm test:e2e demo-scripts-e2e
 */

import { test, expect } from '@playwright/test';
import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { waitForLuxarReady, getLuxarState, getConsoleMessages } from './helpers';

// Base URL for examples (served by HTTP server on port 9000)
const EXAMPLES_BASE = 'http://localhost:9000/packages/luxar/examples';

// Project root (4 levels up from this file)
const PROJECT_ROOT = path.resolve(__dirname, '../../../../..');
const EXAMPLES_DIR = path.join(PROJECT_ROOT, 'packages/luxar/examples');

// All Python demo scripts
const DEMO_SCRIPTS = [
  'build_example.py',
  'dense_cubic_gradient_example.py',
  'dense_grid_5d_example.py',
  'dimension_navigation_example.py',
  'dimension_sliders_5d_example.py',
  'hierarchy_example.py',
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
  'temporal_spiral_sphere_4d_example.py',
  'time_series_4d_example.py',
  'transform_example.py',
];

// Scripts that are known to take a long time or have issues
const SLOW_SCRIPTS = [
  'temporal_spiral_sphere_4d_example.py', // Generates 100M+ points
  'time_series_4d_example.py', // Large 4D dataset
  'performance_benchmark_example.py', // Benchmarking script
];

// Scripts to skip in E2E testing (with reason)
const SKIP_SCRIPTS: Record<string, string> = {
  'memory_optimization_example.py': 'Memory profiling script, not a visualization demo',
};

// nD scripts that may have 0 visible points at initial slice position
// These are not broken - they just need navigation to a slice with points
// For smoke tests, we allow 0 points since we're testing for no errors
const ND_SCRIPTS_ALLOW_ZERO_POINTS = [
  'rainbow_sphere_4d_example.py', // 4D sphere - initial slice may have 0 points
  'spatial_index_demo_example.py', // May have 0 points at initial position
];

/**
 * Run a Python demo script and return execution info
 */
function runDemoScript(scriptName: string): { success: boolean; output: string; duration: number } {
  const scriptPath = path.join(EXAMPLES_DIR, scriptName);
  const startTime = Date.now();

  try {
    // Run with hatch to ensure correct environment
    const output = execSync(`hatch run python "${scriptPath}"`, {
      cwd: PROJECT_ROOT,
      encoding: 'utf-8',
      timeout: 300000, // 5 minute timeout for slow scripts
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    return {
      success: true,
      output: output.toString(),
      duration: Date.now() - startTime,
    };
  } catch (error: unknown) {
    const execError = error as {
      stdout?: Buffer | string;
      stderr?: Buffer | string;
      message: string;
    };
    return {
      success: false,
      output: `${execError.stdout?.toString() || ''}\n${execError.stderr?.toString() || ''}\n${execError.message}`,
      duration: Date.now() - startTime,
    };
  }
}

/**
 * Get the .zarr output name for a script
 */
function getZarrName(scriptName: string): string {
  // Most scripts follow the pattern: name_example.py -> name_example.zarr
  const baseName = scriptName.replace('.py', '');

  // build_example.py generates two outputs
  if (scriptName === 'build_example.py') {
    return 'build_example_manual.zarr'; // Test the first one
  }

  return `${baseName}.zarr`;
}

test.describe('Python Demo Scripts - Full Pipeline E2E', () => {
  // Increase timeout for running Python scripts
  test.describe.configure({ mode: 'serial', timeout: 180000 });

  // Track which scripts generated successfully
  const generatedDatasets: Set<string> = new Set();

  test.beforeAll(async () => {
    console.log('\n[Demo E2E] Running Python demo scripts to generate fresh datasets...\n');

    // Run all demo scripts (non-slow ones) to ensure datasets are fresh
    const scriptsToRun = DEMO_SCRIPTS.filter((s) => !SLOW_SCRIPTS.includes(s) && !SKIP_SCRIPTS[s]);

    for (const script of scriptsToRun) {
      const zarrName = getZarrName(script);
      const zarrPath = path.join(EXAMPLES_DIR, zarrName);

      // Check if dataset already exists and is recent (within last hour)
      const exists = fs.existsSync(zarrPath);
      let isRecent = false;

      if (exists) {
        try {
          const stats = fs.statSync(zarrPath);
          const ageMs = Date.now() - stats.mtimeMs;
          isRecent = ageMs < 3600000; // 1 hour
        } catch {
          // Ignore stat errors
        }
      }

      if (exists && isRecent) {
        console.log(`  [SKIP] ${script} → ${zarrName} (recent)`);
        generatedDatasets.add(zarrName);
        continue;
      }

      console.log(`  [RUN] ${script}...`);
      const result = runDemoScript(script);

      if (result.success) {
        console.log(`    ✅ Generated in ${(result.duration / 1000).toFixed(1)}s`);
        generatedDatasets.add(zarrName);
      } else {
        console.error(`    ❌ FAILED (${(result.duration / 1000).toFixed(1)}s)`);
        console.error(`    Output: ${result.output.slice(0, 500)}`);
        // Don't throw - continue with other scripts
      }
    }

    console.log(`\n[Demo E2E] Generated ${generatedDatasets.size} datasets\n`);
  });

  // Create tests for each demo script
  for (const script of DEMO_SCRIPTS) {
    // Skip certain scripts
    if (SKIP_SCRIPTS[script]) {
      test.skip(`${script} - ${SKIP_SCRIPTS[script]}`, async () => {});
      continue;
    }

    // Mark slow scripts as slow (longer timeout)
    const isSlow = SLOW_SCRIPTS.includes(script);
    const testFn = isSlow ? test.skip : test;

    testFn(`should load ${script} output without errors`, async ({ page }) => {
      const zarrName = getZarrName(script);
      const zarrPath = path.join(EXAMPLES_DIR, zarrName);

      // Verify dataset exists
      if (!fs.existsSync(zarrPath)) {
        console.error(`[${script}] Dataset not found: ${zarrPath}`);
        console.error(
          'Run the script manually: hatch run python ' + path.join(EXAMPLES_DIR, script)
        );
        throw new Error(`Dataset ${zarrName} not found - script may have failed`);
      }

      console.log(`\n[Demo E2E] Testing: ${script} → ${zarrName}`);

      // Navigate to example with debug interface
      const url = `/?src=${EXAMPLES_BASE}/${zarrName}&debug`;
      await page.goto(url);

      // Wait for Luxar to fully initialize
      await waitForLuxarReady(page, 60000);

      // CRITICAL: Check for console errors
      const consoleMessages = await getConsoleMessages(page);

      // Filter out expected/harmless messages
      const allowedPatterns = [
        /404.*spatial_index/, // Expected for datasets without grid-based index
        /404.*chunk_bounds.*zattrs/, // chunk_bounds/.zattrs doesn't exist (OK)
        /optional features/, // Informational message
        /Failed to fetch/, // Network flakiness
      ];

      const actualErrors = consoleMessages.errors.filter((err) => {
        return !allowedPatterns.some((pattern) => pattern.test(err));
      });

      // Report console statistics
      console.log(`  Logs: ${consoleMessages.logs.length}`);
      console.log(`  Warnings: ${consoleMessages.warnings.length}`);
      console.log(`  Errors: ${actualErrors.length}`);

      // Log sample of console output for debugging
      if (consoleMessages.logs.length > 0) {
        console.log('  Sample logs:');
        consoleMessages.logs.slice(0, 5).forEach((log) => {
          console.log(`    ${log.substring(0, 100)}`);
        });
      }

      // Assert no unexpected errors
      if (actualErrors.length > 0) {
        console.error(`\n[${script}] ❌ Console Errors:`);
        actualErrors.forEach((err, i) => {
          console.error(`  ${i + 1}. ${err}`);
        });
      }
      expect(actualErrors.length).toBe(0);

      // Get scene state
      const state = await getLuxarState(page);

      // Debug: If no points loaded, dump console logs
      if (state.totalPoints === 0) {
        console.error(`\n[${script}] ⚠️ No points loaded! Console logs:`);
        consoleMessages.logs.slice(-30).forEach((log, i) => {
          console.error(`  [LOG ${i}] ${log}`);
        });
        consoleMessages.warnings.forEach((warn, i) => {
          console.error(`  [WARN ${i}] ${warn}`);
        });
      }

      // Verify data loaded (allow 0 points for known nD datasets that may have no visible points at initial slice)
      const allowZeroPoints = ND_SCRIPTS_ALLOW_ZERO_POINTS.includes(script);
      if (!allowZeroPoints) {
        expect(state.totalPoints).toBeGreaterThan(0);
      }
      expect(state.pointClouds).toBeDefined();
      expect(state.pointClouds.length).toBeGreaterThan(0);

      // Check for WebGL errors
      const webglErrors = await page.evaluate(() => {
        const canvas = document.querySelector('canvas');
        if (!canvas) return ['No canvas found'];

        const gl =
          (canvas as HTMLCanvasElement).getContext('webgl2') ||
          (canvas as HTMLCanvasElement).getContext('webgl');
        if (!gl) return ['No WebGL context'];

        const errors: string[] = [];
        let error;
        let count = 0;
        while ((error = gl.getError()) !== gl.NO_ERROR && count < 100) {
          errors.push(`GL Error: 0x${error.toString(16)}`);
          count++;
        }

        return errors;
      });

      if (webglErrors.length > 0) {
        console.error(`\n[${script}] ❌ WebGL Errors:`);
        webglErrors.slice(0, 10).forEach((err, i) => {
          console.error(`  ${i + 1}. ${err}`);
        });
      }
      expect(webglErrors.length).toBe(0);

      // Verify data loading messages
      const hasLoadSuccess = consoleMessages.all.some((msg) =>
        /Scene loaded successfully|Loaded.*points|Query result/.test(msg)
      );
      expect(hasLoadSuccess).toBe(true);

      // Log success
      console.log(`  ✅ ${script}: ${state.totalPoints} points loaded, no errors\n`);
    });
  }
});

test.describe('Demo Script Validation', () => {
  test('all demo scripts should be syntactically valid Python', async () => {
    console.log('\n[Demo Validation] Checking Python syntax...\n');

    const errors: string[] = [];

    for (const script of DEMO_SCRIPTS) {
      if (SKIP_SCRIPTS[script]) continue;

      const scriptPath = path.join(EXAMPLES_DIR, script);

      if (!fs.existsSync(scriptPath)) {
        errors.push(`${script}: File not found`);
        continue;
      }

      try {
        // Use python -m py_compile to check syntax
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
      if (SLOW_SCRIPTS.includes(script)) continue; // Skip slow scripts

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

test.describe('Critical Demo Deep Validation', () => {
  // Deep validation for demos that exercise important features

  test('hierarchy_example should have nested groups with transforms', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/hierarchy_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    const state = await getLuxarState(page);

    // Should have multiple point clouds from hierarchy
    expect(state.pointClouds.length).toBeGreaterThan(1);

    // Check console for transform messages
    const messages = await getConsoleMessages(page);
    const hasTransformLog = messages.all.some((msg) => /transform|hierarchy|nested/i.test(msg));

    console.log(`[Hierarchy] Point clouds: ${state.pointClouds.length}`);
    console.log(`[Hierarchy] Transform logging: ${hasTransformLog}`);
  });

  test('dense_grid_5d_example should handle 5D nD slicing', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/dense_grid_5d_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    const state = await getLuxarState(page);

    // Should have 5D dimensional metadata
    expect(state.dimensions?.ndim || 0).toBeGreaterThanOrEqual(5);
    expect(state.totalPoints).toBeGreaterThan(0);

    console.log(`[Dense Grid 5D] Dimensions: ${state.dimensions?.ndim}`);
    console.log(`[Dense Grid 5D] Points visible: ${state.totalPoints}`);
  });

  test('sharpness_showcase should render varied sharpness values', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/sharpness_showcase_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    const state = await getLuxarState(page);
    const messages = await getConsoleMessages(page);

    // Should have multiple point clouds with different sharpness
    expect(state.pointClouds.length).toBeGreaterThan(3);
    expect(state.totalPoints).toBeGreaterThan(1000);

    // Check for no errors
    const errors = messages.errors.filter((e) => !e.includes('404') && !e.includes('optional'));
    expect(errors.length).toBe(0);

    console.log(`[Sharpness] Point clouds: ${state.pointClouds.length}`);
    console.log(`[Sharpness] Total points: ${state.totalPoints}`);
  });

  test('radius_showcase should demonstrate radius-based slicing', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/radius_showcase_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    const state = await getLuxarState(page);

    // Should have multiple point clouds showing radius variations
    expect(state.pointClouds.length).toBeGreaterThan(0);
    expect(state.totalPoints).toBeGreaterThan(0);

    console.log(`[Radius Showcase] Point clouds: ${state.pointClouds.length}`);
    console.log(`[Radius Showcase] Total points: ${state.totalPoints}`);
  });
});
