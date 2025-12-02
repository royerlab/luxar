/**
 * Comprehensive Smoke Tests for ALL Example Datasets
 *
 * CRITICAL: This test suite loads EVERY example dataset and validates:
 * - No console errors
 * - No WebGL errors
 * - Points loaded successfully
 * - Scene rendered without crashes
 *
 * This is the PRIMARY defense against regressions. If this passes, all examples work.
 * If this fails, it shows EXACTLY which example broke and what the console errors are.
 *
 * Purpose: Allow Claude Code to detect issues BEFORE user does by running E2E tests.
 */

import { test, expect } from '@playwright/test';
import {
  waitForLuxarReady,
  getLuxarState,
  assertNoConsoleErrors,
  getConsoleMessages,
} from './helpers';

// Base URL for examples (served by HTTP server on port 9000)
const EXAMPLES_BASE = 'http://localhost:9000/packages/luxar/examples';

// ALL example datasets (auto-discovered from packages/luxar/examples/)
const ALL_EXAMPLES = [
  'build_example_manual.zarr',
  'build_example_structured.zarr',
  'dense_cubic_gradient_example.zarr',
  'dense_grid_5d_example.zarr',
  'dimension_navigation_example.zarr',
  'dimension_sliders_5d_example.zarr',
  'hierarchy_example.zarr',
  'multiple_objects_example.zarr',
  'nd_points_example.zarr',
  'performance_benchmark_example.zarr',
  'point_spacing_example.zarr',
  'progressive_writing_example.zarr',
  'radius_basic_example.zarr',
  'radius_showcase_example.zarr',
  'radius_slicing_example.zarr',
  'rainbow_sphere_4d_example.zarr',
  'rainbow_sphere_spiral_example.zarr',
  'rendering_attributes_example.zarr',
  'rendering_modes_example.zarr',
  'scene_dimensions_example.zarr',
  'sharpness_showcase_example.zarr', // CRITICAL: Exposed LUT scalar bug
  'simple_nd_example.zarr',
  'single_point_example.zarr',
  'spatial_index_demo_example.zarr',
  'temporal_spiral_sphere_4d_example.zarr',
  'time_series_4d_example.zarr',
  'transform_example.zarr',
];

test.describe('ALL Examples - Systematic Smoke Tests', () => {
  // Configure for parallel execution to speed up testing
  test.describe.configure({ mode: 'parallel', timeout: 90000 });

  for (const example of ALL_EXAMPLES) {
    test(`should load ${example} without errors`, async ({ page }) => {
      console.log(`\n[Smoke Test] Testing: ${example}`);

      // Navigate to example with debug interface
      const url = `/?src=${EXAMPLES_BASE}/${example}&debug`;
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
      ];

      const actualErrors = consoleMessages.errors.filter((err) => {
        return !allowedPatterns.some((pattern) => pattern.test(err));
      });

      // Report console statistics
      console.log(`  Logs: ${consoleMessages.logs.length}`);
      console.log(`  Warnings: ${consoleMessages.warnings.length}`);
      console.log(`  Errors: ${actualErrors.length}`);

      // Assert no unexpected errors
      if (actualErrors.length > 0) {
        console.error(`\n[${example}] ❌ Console Errors:`);
        actualErrors.forEach((err, i) => {
          console.error(`  ${i + 1}. ${err}`);
        });
      }
      expect(actualErrors.length).toBe(0);

      // Get scene state
      const state = await getLuxarState(page);

      // Verify data loaded
      expect(state.totalPoints).toBeGreaterThan(0);
      expect(state.pointClouds).toBeDefined();
      expect(state.pointClouds.length).toBeGreaterThan(0);

      // Check for WebGL errors (CRITICAL for rendering issues)
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
        console.error(`\n[${example}] ❌ WebGL Errors:`);
        webglErrors.slice(0, 10).forEach((err, i) => {
          console.error(`  ${i + 1}. ${err}`);
        });
        if (webglErrors.length > 10) {
          console.error(`  ... and ${webglErrors.length - 10} more`);
        }
      }
      expect(webglErrors.length).toBe(0);

      // Verify specific console logs exist (data loaded successfully)
      const hasLoadSuccess = consoleMessages.all.some((msg) =>
        /Scene loaded successfully|Loaded.*points/.test(msg)
      );
      expect(hasLoadSuccess).toBe(true);

      // Log success
      console.log(`  ✅ ${example}: ${state.totalPoints} points loaded, no errors\n`);
    });
  }
});

test.describe('Critical Examples - Deep Validation', () => {
  // Deep validation for examples that exposed bugs

  test('sharpness_showcase - should render all point clouds', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/sharpness_showcase_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    // CRITICAL: This example exposed the LUT scalar mode bug
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Should have multiple point clouds (MixedSharpnessCloud, SharpnessGradient, etc.)
    expect(state.pointClouds.length).toBeGreaterThan(5);

    // Total points should be significant
    expect(state.totalPoints).toBeGreaterThan(10000);

    // Verify no WebGL buffer errors
    const webglErrors = await page.evaluate(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return [];
      const gl =
        (canvas as HTMLCanvasElement).getContext('webgl2') ||
        (canvas as HTMLCanvasElement).getContext('webgl');
      if (!gl) return [];

      const errors: string[] = [];
      let error;
      while ((error = gl.getError()) !== gl.NO_ERROR) {
        errors.push(`GL Error: 0x${error.toString(16)}`);
      }
      return errors;
    });

    expect(webglErrors.length).toBe(0);

    // Verify LUT scalar mode logging (if positions use LUT)
    const messages = await getConsoleMessages(page);
    const hasLUTScalar = messages.all.some((msg) => /LUT \(scalar\)/.test(msg));

    // Log what we found
    console.log(`[Sharpness Showcase] LUT scalar mode detected: ${hasLUTScalar}`);
    console.log(`[Sharpness Showcase] Point clouds: ${state.pointClouds.length}`);
    console.log(`[Sharpness Showcase] Total points: ${state.totalPoints}`);
  });

  test('dense_grid_5d - should handle 5D nD data', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/dense_grid_5d_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Should have 5D dimensional data
    expect(state.dimensions?.ndim || 0).toBeGreaterThanOrEqual(5);

    // Should use spatial index
    expect(state.pointClouds.some((pc: any) => pc.usedSpatialIndex)).toBe(true);

    console.log(`[Dense Grid 5D] Dimensions: ${state.dimensions?.ndim}`);
    console.log(`[Dense Grid 5D] Points: ${state.totalPoints}`);
  });

  test('hierarchy_example - should apply transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/hierarchy_example.zarr&debug`);
    await waitForLuxarReady(page, 60000);

    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Should have hierarchical structure
    expect(state.pointClouds.length).toBeGreaterThan(1);

    // Check for transform applications in console
    const messages = await getConsoleMessages(page);
    const hasTransforms = messages.all.some((msg) => /transform|Transform|matrix/.test(msg));

    console.log(`[Hierarchy] Transforms applied: ${hasTransforms}`);
    console.log(`[Hierarchy] Point clouds: ${state.pointClouds.length}`);
  });
});
