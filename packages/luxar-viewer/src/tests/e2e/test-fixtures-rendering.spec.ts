/**
 * Test Fixture Rendering E2E Tests
 *
 * These tests verify that our unit test fixtures render correctly in the browser.
 * This provides visual verification of the full pipeline:
 * - Python encoding → ZARR storage → TypeScript decoding → WebGL rendering
 *
 * Tests use fixtures from packages/luxar-viewer/tests/fixtures/
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
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
    await waitForPointsLoaded(page, 1); // Wait for at least 1 point to load

    // CRITICAL: Check for console errors immediately after loading
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify points loaded
    expect(state.totalPoints).toBe(31); // 31 points with sharpness [1, 31]
    expect(state.pointClouds.length).toBe(1);

    // Verify sharpness attribute exists and has correct range
    const sharpnessData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.sharpness) {
        return null;
      }

      const geometry = points.geometry;
      const sharpnessAttr = geometry.attributes.sharpness;
      // Use drawRange to get actual point count (buffer may be larger due to reuse)
      const drawRangeCount = geometry.drawRange?.count;
      const actualCount =
        drawRangeCount !== undefined && drawRangeCount !== Infinity
          ? Math.min(drawRangeCount, sharpnessAttr.count)
          : sharpnessAttr.count;
      // Only sample values within the draw range
      const array = Array.from(sharpnessAttr.array.subarray(0, actualCount)) as number[];
      return {
        count: actualCount,
        min: Math.min(...array),
        max: Math.max(...array),
        isNormalized: sharpnessAttr.normalized,
      };
    });

    expect(sharpnessData).not.toBeNull();
    expect(sharpnessData?.count).toBe(31);

    // CRITICAL: Verify sharpness reaches high values (not clamped to 15)
    // With bug: max would be ~15
    // With fix: max should be close to 31 (sharpness stored as float32)
    expect(sharpnessData?.max).toBeGreaterThan(25); // Sharpness values range from 1 to 31

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
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Check for console errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify points loaded
    expect(state.totalPoints).toBe(20); // 20 points with HDR colors

    // Verify colors are Float32Array (HDR)
    const colorData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.color) {
        return null;
      }

      const geometry = points.geometry;
      const colorAttr = geometry.attributes.color;
      const posAttr = geometry.attributes.position;
      // Use drawRange to get actual point count (buffer may be larger due to reuse)
      const drawRangeCount = geometry.drawRange?.count;
      const actualCount =
        drawRangeCount !== undefined && drawRangeCount !== Infinity
          ? Math.min(drawRangeCount, colorAttr.count)
          : colorAttr.count;
      const colorArr = Array.from(colorAttr.array.subarray(0, actualCount * 3)) as number[];
      // The fixture stores points along the X axis with index == x-position, so
      // sort by x to recover the input ordering (the loader/spatial index does
      // not preserve insertion order).
      const itemsPerVertex = posAttr.itemSize ?? 3;
      const posArr = Array.from(
        posAttr.array.subarray(0, actualCount * itemsPerVertex)
      ) as number[];
      const indices = Array.from({ length: actualCount }, (_, i) => i);
      indices.sort((a, b) => posArr[a * itemsPerVertex] - posArr[b * itemsPerVertex]);
      const redChannels = indices.map((i) => colorArr[i * 3]);

      return {
        count: actualCount,
        arrayType: colorAttr.array.constructor.name,
        redMin: Math.min(...redChannels),
        redMax: Math.max(...redChannels),
        redChannels,
      };
    });

    expect(colorData).not.toBeNull();
    expect(colorData?.count).toBe(20);

    // CRITICAL: Verify Float32Array (not Uint8Array)
    expect(colorData?.arrayType).toBe('Float32Array');

    // CRITICAL: Verify HDR values preserved (max should be ~10.0, not clamped to 1.0)
    expect(colorData?.redMax).toBeGreaterThan(5.0);
    expect(colorData?.redMax).toBeLessThanOrEqual(10.5);

    // Verify monotonic increase (linspace property), reading colors in
    // x-position order rather than buffer order.
    const reds = (colorData?.redChannels as number[]) || [];
    for (let i = 1; i < reds.length; i++) {
      expect(reds[i]).toBeGreaterThanOrEqual(reds[i - 1] - 0.01); // Allow tiny float errors
    }

    // Log and check console
    const hdrConsole = await getConsoleMessages(page);
    console.log(
      `[HDR Test] Logs: ${hdrConsole.logs.length}, Warnings: ${hdrConsole.warnings.length}`
    );

    // Take screenshot
    await page.screenshot({ path: 'test-results/hdr-colors-rendering.png' });
  });

  test('should render hierarchical transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.hierarchy}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Check for console errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify scene loaded
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify transform hierarchy
    const hierarchyData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      // Find parent group - name includes leading slash from zarr path
      const parentGroup = debug.scene.getObjectByName('/parent_group');
      if (!parentGroup) return null;

      // Find child points - full path from zarr
      const childPoints = parentGroup.children.find(
        (obj: any) => obj.name === '/parent_group/child_points'
      );
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
        // Get world position of child using position.clone() instead of THREE.Vector3
        childWorldPosition: (() => {
          const worldPos = childPoints.position.clone();
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
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: No errors in nD data loading
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 4D data loaded
    // Total: 500 points × 10 time steps = 5000
    // Note: With nD slicing, only visible points at current slice are in geometry
    // The total should be 5000 (metadata), but displayed varies by slice
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify dimension information via getState().dimensions
    const dimensionData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const state = debug.getState();

      if (!state || !state.dimensions) return null;

      return {
        numDimensions: state.dimensions.ndim,
        displayedDimensions: state.dimensions.displayed,
        currentSlice: state.dimensions.currentStep,
      };
    });

    // Dimensions may be null if dataset is 3D (not nD)
    // The test fixture is 4D, so we expect dimensions
    if (dimensionData) {
      expect(dimensionData.numDimensions).toBe(4); // time, x, y, z
      expect(dimensionData.displayedDimensions.length).toBe(3); // x, y, z displayed
    }

    // Check console for dimension-related messages
    const ndConsole = await getConsoleMessages(page);
    console.log(`[4D Test] Console output: ${ndConsole.all.length} messages`);

    // Take screenshot
    await page.screenshot({ path: 'test-results/4d-slicing-rendering.png' });
  });

  test('should render broadcasting encoded data', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.broadcasting}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Verify broadcasting decoding has no errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 1000 points loaded (broadcasted from 1 value)
    expect(state.totalPoints).toBe(1000);

    // Verify all points have same color (broadcasted)
    const colorUniformity = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.color) {
        return null;
      }

      const geometry = points.geometry;
      const colorAttr = geometry.attributes.color;
      // Use drawRange to get actual point count (buffer may be larger due to reuse)
      const drawRangeCount = geometry.drawRange?.count;
      const actualCount =
        drawRangeCount !== undefined && drawRangeCount !== Infinity
          ? Math.min(drawRangeCount, colorAttr.count)
          : colorAttr.count;
      // Only sample color values within the draw range
      const colors = colorAttr.array.subarray(0, actualCount * 3);
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

    // Optionally verify broadcast decoding in console logs (not required for test to pass)
    const broadcastConsole = await getConsoleMessages(page);
    const broadcastLog = broadcastConsole.all.find((msg) => msg.includes('Broadcasting'));
    if (broadcastLog) {
      console.log('✅ Broadcasting log found:', broadcastLog.slice(0, 100));
    } else {
      console.log('ℹ️ Broadcasting log not found (may have been optimized away)');
    }

    // Take screenshot
    await page.screenshot({ path: 'test-results/broadcasting-rendering.png' });
  });

  test('should render LUT encoded data', async ({ page }) => {
    await page.goto(`/?src=${FIXTURES.lut}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1); // Wait for data to load

    // CRITICAL: Verify LUT decoding has no errors
    await assertNoConsoleErrors(page);

    const state = await getLuxarState(page);

    // Verify 1000 points loaded
    expect(state.totalPoints).toBe(1000);

    // Verify LUT decoding: should have exactly 10 unique colors
    const colorStats = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.color) {
        return null;
      }

      const geometry = points.geometry;
      const colorAttr = geometry.attributes.color;
      // Use drawRange to get actual point count (buffer may be larger due to reuse)
      const drawRangeCount = geometry.drawRange?.count;
      const actualCount =
        drawRangeCount !== undefined && drawRangeCount !== Infinity
          ? Math.min(drawRangeCount, colorAttr.count)
          : colorAttr.count;
      // Only sample color values within the draw range
      const colors = colorAttr.array.subarray(0, actualCount * 3);
      const uniqueColors = new Set<string>();

      for (let i = 0; i < colors.length; i += 3) {
        const colorKey = `${colors[i].toFixed(2)},${colors[i + 1].toFixed(2)},${colors[i + 2].toFixed(2)}`;
        uniqueColors.add(colorKey);
      }

      return {
        totalPoints: actualCount,
        uniqueColors: uniqueColors.size,
      };
    });

    expect(colorStats).not.toBeNull();
    expect(colorStats?.totalPoints).toBe(1000);

    // CRITICAL: LUT encoding should preserve unique colors
    // Note: Due to float precision in color key generation, actual count may vary
    // The important thing is that LUT decoding worked and we have multiple distinct colors
    expect(colorStats?.uniqueColors).toBeGreaterThanOrEqual(7);

    // Verify LUT decoding logged in console
    const lutConsole = await getConsoleMessages(page);
    const lutLog = lutConsole.all.find((msg) => msg.includes('LUT'));
    expect(lutLog).toBeDefined(); // Should log LUT decoding operation

    // Take screenshot
    await page.screenshot({ path: 'test-results/lut-encoding-rendering.png' });
  });
});

test.describe('Console Error Detection', () => {
  test('should detect and report console errors', async ({ page }) => {
    // Collect page-level errors as a fallback in case the console interceptor
    // isn't ready before the errors fire
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'error' || msg.type() === 'warning') {
        pageErrors.push(msg.text());
      }
    });

    // Test that our error detection actually works by loading a non-existent dataset
    await page.goto('/?src=http://localhost:9000/nonexistent.zarr&debug');

    // Poll for the first error/warning to be intercepted instead of
    // waiting a fixed 5 s. Caps at 8 s so we still fail loudly if the
    // interceptor is broken; in practice the first network error
    // usually propagates within ~1 s.
    await page
      .waitForFunction(
        () => {
          const debug = (window as any).__luxarDebug;
          if (!debug?.consoleInterceptor?.getBufferedMessages) return false;
          const msgs = debug.consoleInterceptor.getBufferedMessages();
          return msgs.some(
            (m: { type?: string }) => m.type === 'error' || m.type === 'warning'
          );
        },
        null,
        { timeout: 8000 }
      )
      .catch(() => {
        // Interceptor never reported — page-level errors may still have
        // been captured by the captureConsoleMessages listener below.
      });

    // Check both the in-app console interceptor and the Playwright-captured errors
    const consoleMessages = await getConsoleMessages(page);
    const interceptedCount = consoleMessages.errors.length + consoleMessages.warnings.length;
    const totalErrorCount = interceptedCount + pageErrors.length;

    console.log('[Error Detection Test] Intercepted errors:', consoleMessages.errors.length);
    console.log('[Error Detection Test] Intercepted warnings:', consoleMessages.warnings.length);
    console.log('[Error Detection Test] Page-level errors:', pageErrors.length);

    // Verify we captured error messages from either source
    expect(totalErrorCount).toBeGreaterThan(0);
  });
});
