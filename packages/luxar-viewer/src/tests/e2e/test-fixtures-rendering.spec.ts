/**
 * Test Fixture Rendering E2E Tests
 *
 * These tests verify that our unit test fixtures render correctly in the browser.
 * This provides visual verification of the full pipeline:
 * - Python encoding → ZARR storage → TypeScript decoding → WebGL rendering
 *
 * Tests use fixtures from packages/luxar-viewer/tests/fixtures/
 */

import { test, expect } from '@playwright/test';
import {
  waitForLuxarReady,
  getLuxarState,
  assertNoConsoleErrors,
  getConsoleMessages,
  getWebGLErrors,
} from './helpers';

// HTTP server (configured in playwright.config.ts) serves from project root
// Fixtures are at: packages/luxar-viewer/tests/fixtures/
const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

const FIXTURES = {
  sharpness: `${FIXTURES_BASE}/test_sharpness_range.zarr`,
  hdr: `${FIXTURES_BASE}/test_hdr_colors.zarr`,
  hierarchy: `${FIXTURES_BASE}/test_hierarchical_transforms.zarr`,
  nd4d: `${FIXTURES_BASE}/test_4d.zarr`,
  broadcasting: `${FIXTURES_BASE}/test_broadcasting.zarr`,
  lut: `${FIXTURES_BASE}/test_lut.zarr`,
};

test.describe('Test Fixture Rendering', () => {
  test('should render sharpness range fixture correctly', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.sharpness}&debug`);
    await waitForLuxarReady(page);

    // CRITICAL: Check for console errors immediately after loading
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify points loaded
    expect(state.totalPoints).toBe(31); // 31 points with sharpness [1, 31]
    expect(state.pointClouds.length).toBe(1);

    // Verify sharpness attribute exists and has correct range
    const sharpnessData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const points = debug.scene.children.find((obj: any) => obj.type === 'Points');

      if (!points || !points.geometry.attributes.sharpness) {
        return null;
      }

      const sharpnessAttr = points.geometry.attributes.sharpness;
      const array = Array.from(sharpnessAttr.array) as number[];
      return {
        count: sharpnessAttr.count,
        min: Math.min(...array),
        max: Math.max(...array),
        isNormalized: sharpnessAttr.normalized,
      };
    });

    expect(sharpnessData).not.toBeNull();
    expect(sharpnessData?.count).toBe(31);

    // CRITICAL: Verify sharpness reaches high values (not clamped to 15)
    // With bug: max would be ~15
    // With fix: max should be close to 31
    expect(sharpnessData?.max).toBeGreaterThan(200); // uint8: 31 * 255 / 31 ≈ 255

    // Check for WebGL errors (CRITICAL)
    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);

    // Log console messages for debugging
    const consoleMessages = await getConsoleMessages(page);
    console.log(`[Sharpness Test] Console logs: ${consoleMessages.logs.length}`);
    console.log(`[Sharpness Test] Console warnings: ${consoleMessages.warnings.length}`);
    console.log(`[Sharpness Test] WebGL errors: ${webglErrors.length}`);

    // Take screenshot for visual verification
    await page.screenshot({ path: 'test-results/sharpness-range-rendering.png' });
  });

  test('should render HDR colors without clamping', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.hdr}&debug`);
    await waitForLuxarReady(page);

    // CRITICAL: Check for console errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify points loaded
    expect(state.totalPoints).toBe(20); // 20 points with HDR colors

    // Verify colors are Float32Array (HDR)
    const colorData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const points = debug.scene.children.find((obj: any) => obj.type === 'Points');

      if (!points || !points.geometry.attributes.color) {
        return null;
      }

      const colorAttr = points.geometry.attributes.color;
      const array = Array.from(colorAttr.array) as number[];

      // Extract red channel values (every 3rd value starting at 0)
      const redChannels: number[] = [];
      for (let i = 0; i < array.length; i += 3) {
        redChannels.push(array[i]);
      }

      return {
        count: colorAttr.count,
        arrayType: colorAttr.array.constructor.name,
        redMin: Math.min(...redChannels),
        redMax: Math.max(...redChannels),
        redChannels: redChannels,
      };
    });

    expect(colorData).not.toBeNull();
    expect(colorData?.count).toBe(20);

    // CRITICAL: Verify Float32Array (not Uint8Array)
    expect(colorData?.arrayType).toBe('Float32Array');

    // CRITICAL: Verify HDR values preserved (max should be ~10.0, not clamped to 1.0)
    expect(colorData?.redMax).toBeGreaterThan(5.0);
    expect(colorData?.redMax).toBeLessThanOrEqual(10.5);

    // Verify monotonic increase (linspace property)
    const reds = (colorData?.redChannels as number[]) || [];
    for (let i = 1; i < reds.length; i++) {
      expect(reds[i]).toBeGreaterThanOrEqual(reds[i - 1] - 0.01); // Allow tiny float errors
    }

    // Log and check console
    const hdrConsole = await getConsoleMessages(page);
    console.log(`[HDR Test] Logs: ${hdrConsole.logs.length}, Warnings: ${hdrConsole.warnings.length}`);

    // Take screenshot
    await page.screenshot({ path: 'test-results/hdr-colors-rendering.png' });
  });

  test('should render hierarchical transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.hierarchy}&debug`);
    await waitForLuxarReady(page);

    // CRITICAL: Check for console errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify scene loaded
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify transform hierarchy
    const hierarchyData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      // Find parent group
      const parentGroup = debug.scene.getObjectByName('parent_group');
      if (!parentGroup) return null;

      // Find child points
      const childPoints = parentGroup.children.find((obj: any) => obj.name === 'child_points');
      if (!childPoints) return null;

      return {
        parentPosition: {
          x: parentGroup.position.x,
          y: parentGroup.position.y,
          z: parentGroup.position.z,
        },
        childLocalPosition: {
          x: childPoints.position.x,
          y: childPoints.position.y,
          z: childPoints.position.z,
        },
        // Get world position of child (parent + child transforms applied)
        childWorldPosition: (() => {
          const worldPos = new (window as any).THREE.Vector3();
          childPoints.getWorldPosition(worldPos);
          return { x: worldPos.x, y: worldPos.y, z: worldPos.z };
        })(),
      };
    });

    expect(hierarchyData).not.toBeNull();

    // Parent should be at [10, 0, 0]
    expect(hierarchyData?.parentPosition.x).toBeCloseTo(10.0, 1);
    expect(hierarchyData?.parentPosition.y).toBeCloseTo(0.0, 1);
    expect(hierarchyData?.parentPosition.z).toBeCloseTo(0.0, 1);

    // Child local position should be [0, 5, 0]
    expect(hierarchyData?.childLocalPosition.x).toBeCloseTo(0.0, 1);
    expect(hierarchyData?.childLocalPosition.y).toBeCloseTo(5.0, 1);
    expect(hierarchyData?.childLocalPosition.z).toBeCloseTo(0.0, 1);

    // CRITICAL: Child world position should be [10, 5, 0] (transforms composed)
    expect(hierarchyData?.childWorldPosition.x).toBeCloseTo(10.0, 1);
    expect(hierarchyData?.childWorldPosition.y).toBeCloseTo(5.0, 1);
    expect(hierarchyData?.childWorldPosition.z).toBeCloseTo(0.0, 1);

    // Verify no errors during transform application
    const transformConsole = await getConsoleMessages(page);
    expect(transformConsole.errors.length).toBe(0);

    // Take screenshot
    await page.screenshot({ path: 'test-results/hierarchical-transforms-rendering.png' });
  });

  test('should handle 4D nD slicing dataset', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.nd4d}&debug`);
    await waitForLuxarReady(page);

    // CRITICAL: No errors in nD data loading
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 4D data loaded
    // Total: 500 points × 10 time steps = 5000
    // But only points at current time slice are visible
    expect(state.totalPoints).toBe(5000);

    // Verify dimension information
    const dimensionData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const app = debug.app;

      if (!app || !app.sceneDimsManager) return null;

      return {
        numDimensions: app.sceneDimsManager.ndim,
        displayedDimensions: app.sceneDimsManager.displayedDimensions,
        currentSlice: app.sceneDimsManager.currentStep,
      };
    });

    expect(dimensionData).not.toBeNull();
    expect(dimensionData?.numDimensions).toBe(4); // time, x, y, z
    expect(dimensionData?.displayedDimensions.length).toBe(3); // x, y, z displayed

    // Check console for dimension-related messages
    const ndConsole = await getConsoleMessages(page);
    console.log(`[4D Test] Console output: ${ndConsole.all.length} messages`);

    // Take screenshot
    await page.screenshot({ path: 'test-results/4d-slicing-rendering.png' });
  });

  test('should render broadcasting encoded data', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.broadcasting}&debug`);
    await waitForLuxarReady(page);

    // CRITICAL: Verify broadcasting decoding has no errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 1000 points loaded (broadcasted from 1 value)
    expect(state.totalPoints).toBe(1000);

    // Verify all points have same color (broadcasted)
    const colorUniformity = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const points = debug.scene.children.find((obj: any) => obj.type === 'Points');

      if (!points || !points.geometry.attributes.color) {
        return null;
      }

      const colors = points.geometry.attributes.color.array;
      const firstColor = [colors[0], colors[1], colors[2]];

      // Check if all colors match the first color
      let allSame = true;
      for (let i = 0; i < colors.length; i += 3) {
        if (
          Math.abs(colors[i] - firstColor[0]) > 0.01 ||
          Math.abs(colors[i + 1] - firstColor[1]) > 0.01 ||
          Math.abs(colors[i + 2] - firstColor[2]) > 0.01
        ) {
          allSame = false;
          break;
        }
      }

      return { allSame, firstColor };
    });

    expect(colorUniformity).not.toBeNull();
    expect(colorUniformity?.allSame).toBe(true);

    // Verify broadcast decoding in console logs
    const broadcastConsole = await getConsoleMessages(page);
    const broadcastLog = broadcastConsole.all.find(msg => msg.includes('Broadcasting'));
    expect(broadcastLog).toBeDefined(); // Should log broadcasting operation

    // Take screenshot
    await page.screenshot({ path: 'test-results/broadcasting-rendering.png' });
  });

  test('should render LUT encoded data', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.lut}&debug`);
    await waitForLuxarReady(page);

    // CRITICAL: Verify LUT decoding has no errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 1000 points loaded
    expect(state.totalPoints).toBe(1000);

    // Verify LUT decoding: should have exactly 10 unique colors
    const colorStats = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const points = debug.scene.children.find((obj: any) => obj.type === 'Points');

      if (!points || !points.geometry.attributes.color) {
        return null;
      }

      const colors = points.geometry.attributes.color.array;
      const uniqueColors = new Set<string>();

      for (let i = 0; i < colors.length; i += 3) {
        const colorKey = `${colors[i].toFixed(2)},${colors[i + 1].toFixed(2)},${colors[i + 2].toFixed(2)}`;
        uniqueColors.add(colorKey);
      }

      return {
        totalPoints: colors.length / 3,
        uniqueColors: uniqueColors.size,
      };
    });

    expect(colorStats).not.toBeNull();
    expect(colorStats?.totalPoints).toBe(1000);

    // CRITICAL: LUT encoding should preserve 10 unique colors
    expect(colorStats?.uniqueColors).toBe(10);

    // Verify LUT decoding logged in console
    const lutConsole = await getConsoleMessages(page);
    const lutLog = lutConsole.all.find(msg => msg.includes('LUT'));
    expect(lutLog).toBeDefined(); // Should log LUT decoding operation

    // Take screenshot
    await page.screenshot({ path: 'test-results/lut-encoding-rendering.png' });
  });
});

test.describe('Console Error Detection', () => {
  test('should detect and report console errors', async ({ page }) => {
    // Test that our error detection actually works by loading a non-existent dataset
    await page.goto('/?src=http://localhost:9000/nonexistent.zarr&debug');

    // Wait a bit for error to occur
    await page.waitForTimeout(2000);

    // Should have console errors
    const consoleMessages = await getConsoleMessages(page);
    console.log('[Error Detection Test] Errors:', consoleMessages.errors.length);
    console.log('[Error Detection Test] Warnings:', consoleMessages.warnings.length);

    // Verify we captured error messages
    expect(consoleMessages.errors.length + consoleMessages.warnings.length).toBeGreaterThan(0);
  });
});
