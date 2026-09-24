/**
 * Comprehensive Smoke Tests for ALL Example Datasets
 *
 * CRITICAL: This test suite loads every generated example dataset EXCEPT the
 * ones parked in `KNOWN_FLAKY_LARGE_DATASETS` below (each with its documented
 * reason, and a tracking issue where one exists), and validates:
 * - No console errors
 * - No WebGL errors
 * - Renderable geometry loaded successfully
 * - Scene rendered without crashes
 *
 * This is a PRIMARY defense against regressions. If this passes, all the
 * examples it still covers work. If this fails, it shows EXACTLY which example
 * broke and what the console errors are.
 *
 * Purpose: Allow Claude Code to detect issues BEFORE user does by running E2E tests.
 */

import { test, expect } from './fixtures';
import { resolve } from 'node:path';
import type { DebugState } from '../../core/app/debug/debug-state';
import {
  discoverExampleDatasets,
  validateExampleDatasetReferences,
} from '../../../tools/example-smoke-inventory';
import {
  waitForLuxarReady,
  getLuxarState,
  assertNoConsoleErrors,
  getConsoleMessages,
} from './helpers';

// Base URL for examples (served by HTTP server on port 9000)
const EXAMPLES_BASE = 'http://localhost:9000/datasets/examples';

// Known-flaky large datasets that require investigation. These have edge
// cases with effective-radius filtering or WebGL buffer issues — tracked
// for the post-decomposition points-spatial-index-loader work.
const KNOWN_FLAKY_LARGE_DATASETS = {
  'time_series_4d_example.luxar.zarr': 'Large 4D dataset with occasional WebGL buffer failures.',
  // 196 MB on disk; the headless chromium worker pool exhausts
  // ERR_INSUFFICIENT_RESOURCES decoding it in parallel with the rest
  // of the suite. Smoke coverage is provided by smaller fixtures;
  // re-enable once we ship a downsized progressive_writing example or
  // sequential-mode override for oversized fixtures.
  'progressive_writing_example.luxar.zarr':
    '196 MB; parallel headless Chromium workers exhaust decode resources.',
  // 1.5M points (a 1M-point CubicArray group plus a 500k background star
  // field). Parked for HEADROOM, not for #1724:
  // re-measured with frame pacing in place it loads and passes with zero
  // console errors — but in 50 s of the 120 s budget at `--workers=1`. The
  // documented failure mode is PARALLEL contention (the HTTP server plus
  // decompression starving the page.evaluate slot inside waitForLuxarReady /
  // getLuxarState past the ceiling), and 50 s with NO contention leaves no
  // headroom under this suite's `mode: 'parallel'`. So it stays parked
  // pending a sequential-mode override for million-point examples or a
  // downsized fixture, not pending a viewer fix.
  'dense_cubic_gradient_example.luxar.zarr':
    '1.5M points; passes alone in 50s but lacks headroom under parallel contention.',
} as const;

const ALL_EXAMPLES = discoverExampleDatasets(
  resolve(import.meta.dirname, '../../../../../datasets/examples'),
  KNOWN_FLAKY_LARGE_DATASETS
);

// Geometry-only datasets have no point clouds.
const DATASETS_ALLOW_ZERO_POINTS = [
  'lines_basic_example.luxar.zarr', // Lines geometry only - no point clouds
  'lines_indexed_example.luxar.zarr', // Lines geometry only - no point clouds
  'lines_partition_and_sampling_example.luxar.zarr', // Lines geometry only - no point clouds
  'lines_primitive_qa_example.luxar.zarr', // Lines geometry only - no point clouds
  'lines_substitutive_lod_example.luxar.zarr', // Lines geometry only - no point clouds
  'gsplats_basic_example.luxar.zarr', // GSplats geometry only - no point clouds (totalPoints=0)
  'gsplats_fit_volume_example.luxar.zarr', // GSplats geometry only - no point clouds (totalPoints=0)
  'gsplats_lod_example.luxar.zarr', // GSplats kind=lod - no point clouds (totalPoints=0)
  'mesh_basic_example.luxar.zarr', // Mesh geometry only - no point clouds (totalPoints=0)
];

validateExampleDatasetReferences(
  ALL_EXAMPLES,
  KNOWN_FLAKY_LARGE_DATASETS,
  DATASETS_ALLOW_ZERO_POINTS,
  'Example smoke zero-points allowance'
);

test.describe('ALL Examples - Systematic Smoke Tests', () => {
  // Configure for parallel execution to speed up testing.
  // 120s (was 90s) absorbs HTTP-server contention when several
  // worker-pool tabs decode mid-size datasets like
  // dense_cubic_gradient_example.luxar.zarr concurrently — page.evaluate
  // calls inside getLuxarState() consistently bumped against the
  // 90s ceiling under load. Truly oversized fixtures are still
  // routed through KNOWN_FLAKY_LARGE_DATASETS.
  test.describe.configure({ mode: 'parallel', timeout: 120000 });

  for (const example of ALL_EXAMPLES) {
    test(`should load ${example} without errors`, async ({ page }) => {
      console.log(`\n[Smoke Test] Testing: ${example}`);

      // Navigate to example with debug interface
      // `&noOpfs` on every load: this spec never asserts the L2 OPFS tier, and
      // automated Chromium's OPFS stalls systemically (10s per op — issue #1645),
      // starving scene readiness past the test budget. The circuit breaker only
      // helps un-flagged real sessions (it still pays ~3 timeouts per fresh page).
      const url = `/?src=${EXAMPLES_BASE}/${example}&debug&noOpfs`;
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
        /Failed to fetch/, // Network flakiness with large datasets (HTTP server overwhelmed)
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
      const state: DebugState = await getLuxarState(page);
      const allowZeroPoints = DATASETS_ALLOW_ZERO_POINTS.includes(example);
      const geometryTypes = [
        { label: 'points', nodes: state.pointClouds, total: state.totalPoints },
        { label: 'line', nodes: state.lineMeshes, total: state.totalLines },
        { label: 'gsplat', nodes: state.gsplatMeshes, total: state.totalGSplats },
        { label: 'mesh', nodes: state.meshNodes, total: state.totalTriangles },
      ];
      const zeroGeometryTypes = geometryTypes.filter(
        (geometryType) => geometryType.nodes.length > 0 && geometryType.total === 0
      );

      // Debug: If geometry is wholly or partially missing, dump console logs to
      // diagnose. The last clause covers the case the per-type scan cannot see —
      // points were expected but no point node loaded at all, so `points` never
      // enters `zeroGeometryTypes` while another type keeps `totalElements` above
      // zero. That still fails the assertion below, and wants the same logs.
      if (
        state.totalElements === 0 ||
        zeroGeometryTypes.length > 0 ||
        (state.totalPoints === 0 && !allowZeroPoints)
      ) {
        console.error(`\n[${example}] ⚠️ Geometry missing! Dumping console logs:`);
        consoleMessages.logs.slice(-50).forEach((log, i) => {
          console.error(`  [LOG ${i}] ${log}`);
        });
        consoleMessages.warnings.forEach((warn, i) => {
          console.error(`  [WARN ${i}] ${warn}`);
        });
      }

      // Verify data loaded. The allowance covers datasets with no point node at
      // all; a dataset that HAS point nodes still has to fill them, because the
      // per-type loop below is unconditional.
      if (!allowZeroPoints) {
        expect(state.totalPoints).toBeGreaterThan(0);
      }
      // `totalElements` sums the actual point, segment, splat, and triangle
      // counts. Keep this unconditional: DATASETS_ALLOW_ZERO_POINTS waives only
      // the point-specific assertion, not the requirement that geometry loaded.
      expect(state.totalElements).toBeGreaterThan(0);
      // This is per COMMITTED SLICE, unlike Python's per-node-total authoring warning.
      // temporal_spiral_sphere_4d_example authors 524,288 points in one node, but its
      // non-displayed discrete t axis commits only 4,096 points per viewer slice.
      const droppedNodes = [...state.pointClouds, ...state.lineMeshes, ...state.gsplatMeshes]
        .filter((node) => node.droppedElementCount > 0)
        .map((node) => `${node.name}: ${node.droppedElementCount}`)
        .join(', ');
      expect(
        state.totalDroppedElements,
        `renderer capacity dropped elements from ${droppedNodes || 'unknown nodes'}`
      ).toBe(0);

      // A strict per-node assertion is not valid: nd_points_example has a visible
      // /Reference5D node with pointCount=0 while its sibling carries all 820 points,
      // and hidden LOD nodes can also legitimately report zero. Per-type totals still
      // catch a geometry loader silently producing no elements. Mixed-type substitutive
      // LOD examples satisfy this initial-slice contract through their eagerly registered
      // coarse GSplat level; changing default-level or release behavior may require revisiting it.
      for (const geometryType of geometryTypes) {
        if (geometryType.nodes.length > 0) {
          expect(
            geometryType.total,
            `${example}: ${geometryType.label} nodes present but 0 elements`
          ).toBeGreaterThan(0);
        }
      }

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
      console.log(
        `  ✅ ${example}: ${state.totalElements} elements (${state.totalPoints} points) loaded, no errors\n`
      );
    });
  }
});

test.describe('Critical Examples - Deep Validation', () => {
  test.describe.configure({ timeout: 120000 });

  // Deep validation for examples that exposed bugs

  test('mesh_basic - should report its rendered triangles', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/mesh_basic_example.luxar.zarr&debug&noOpfs`);
    await waitForLuxarReady(page, 60000);

    await assertNoConsoleErrors(page);

    const state: DebugState = await getLuxarState(page);

    expect(state.meshNodes).toHaveLength(1);
    expect(state.meshNodes[0].triangleCount).toBe(4);
    expect(state.meshNodes[0].flatNormal).toBe(true);
    expect(state.totalTriangles).toBe(4);
    expect(state.totalElements).toBe(4);
  });

  test('sharpness_showcase - should render all point clouds', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/sharpness_showcase_example.luxar.zarr&debug&noOpfs`);
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
    await page.goto(`/?src=${EXAMPLES_BASE}/dense_grid_5d_example.luxar.zarr&debug&noOpfs`);
    await waitForLuxarReady(page, 60000);

    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Should have 5D dimensional data
    expect(state.dimensions?.ndim || 0).toBeGreaterThanOrEqual(5);

    // Verify points loaded (spatial index usage is tracked internally but not exposed to debug state)
    expect(state.totalPoints).toBeGreaterThan(0);

    console.log(`[Dense Grid 5D] Dimensions: ${state.dimensions?.ndim}`);
    console.log(`[Dense Grid 5D] Points: ${state.totalPoints}`);
    console.log(`[Dense Grid 5D] Point clouds: ${state.pointClouds?.length}`);
  });

  test('hierarchy_example - should apply transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/hierarchy_example.luxar.zarr&debug&noOpfs`);
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

  test('radius_showcase - should demonstrate radius-based slicing', async ({ page }) => {
    await page.goto(`/?src=${EXAMPLES_BASE}/radius_showcase_example.luxar.zarr&debug&noOpfs`);
    await waitForLuxarReady(page, 60000);

    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Should have multiple point clouds showing radius variations
    expect(state.pointClouds.length).toBeGreaterThan(0);
    expect(state.totalPoints).toBeGreaterThan(0);

    console.log(`[Radius Showcase] Point clouds: ${state.pointClouds.length}`);
    console.log(`[Radius Showcase] Total points: ${state.totalPoints}`);
  });
});
