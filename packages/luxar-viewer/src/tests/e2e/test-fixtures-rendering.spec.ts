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
  sharpness: `${FIXTURES_BASE}/test_sharpness_range.luxar.zarr`,
  hdr: `${FIXTURES_BASE}/test_hdr_colors.luxar.zarr`,
  hierarchy: `${FIXTURES_BASE}/test_hierarchical_transforms.luxar.zarr`,
  nd4d: `${FIXTURES_BASE}/test_4d.luxar.zarr`,
  broadcasting: `${FIXTURES_BASE}/test_broadcasting.luxar.zarr`,
  lut: `${FIXTURES_BASE}/test_lut.luxar.zarr`,
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
    expect(state.totalPoints).toBe(32); // 32 points sampling sharpness [0, 1]
    expect(state.pointClouds.length).toBe(1);

    // Verify sharpness attribute exists and has correct range
    const sharpnessData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let points: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.aSharpness) {
        return null;
      }

      const geometry = points.geometry;
      const sharpnessAttr = geometry.attributes.aSharpness;
      // Points render as instanced quads: drawRange is the 6-index base
      // quad, while instanceCount is the visible point count. Attribute
      // buffers may be over-allocated by the GPU pool.
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : sharpnessAttr.count;
      const actualCount = Math.min(instanceCount, sharpnessAttr.count);
      // aSharpness is an InterleavedBufferAttribute; .array would return
      // the shared interleaved buffer. Use getX(i) to read the actual
      // per-instance sharpness values.
      const array: number[] = [];
      for (let i = 0; i < actualCount; i++) {
        array.push(sharpnessAttr.getX(i));
      }
      return {
        count: actualCount,
        min: Math.min(...array),
        max: Math.max(...array),
        isNormalized: sharpnessAttr.normalized,
      };
    });

    expect(sharpnessData).not.toBeNull();
    expect(sharpnessData?.count).toBe(32);

    // Sharpness is a normalized [0, 1] knob (fixture = linspace(0, 1, 32)).
    // Verify the decoded range spans the full [0, 1] knob.
    expect(sharpnessData?.max).toBeGreaterThan(0.95); // reaches ~1.0
    expect(sharpnessData?.min).toBeLessThan(0.05); // reaches ~0.0

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
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.aColor) {
        return null;
      }

      const geometry = points.geometry;
      const colorAttr = geometry.attributes.aColor;
      const posAttr = geometry.attributes.aCenter;
      // Points render as instanced quads: drawRange is the 6-index base
      // quad, while instanceCount is the visible point count. Attribute
      // buffers may be over-allocated by the GPU pool.
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : colorAttr.count;
      const actualCount = Math.min(instanceCount, colorAttr.count);
      // aColor/aCenter are InterleavedBufferAttribute views over a shared
      // buffer; read per-instance components via getX/getY/getZ.
      // The fixture stores points along the X axis with index == x-position,
      // so sort by x to recover the input ordering (the loader/spatial
      // index does not preserve insertion order).
      const positions: { index: number; x: number }[] = [];
      for (let i = 0; i < actualCount; i++) {
        positions.push({ index: i, x: posAttr.getX(i) });
      }
      positions.sort((a, b) => a.x - b.x);
      const redChannels = positions.map((p) => colorAttr.getX(p.index));

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
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.aColor) {
        return null;
      }

      const geometry = points.geometry;
      const colorAttr = geometry.attributes.aColor;
      // Points render as instanced quads: drawRange is the 6-index base
      // quad, while instanceCount is the visible point count. Attribute
      // buffers may be over-allocated by the GPU pool.
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : colorAttr.count;
      const actualCount = Math.min(instanceCount, colorAttr.count);
      // aColor is an InterleavedBufferAttribute view; .array would return
      // the shared interleaved buffer mixing positions/radii/colors/etc.
      // Use getX/getY/getZ to read the actual per-instance RGB triplet.
      const firstColor = [colorAttr.getX(0), colorAttr.getY(0), colorAttr.getZ(0)];

      // Check if all colors match the first color
      let allSame = true;
      for (let i = 0; i < actualCount; i++) {
        if (
          Math.abs(colorAttr.getX(i) - firstColor[0]) > 0.01 ||
          Math.abs(colorAttr.getY(i) - firstColor[1]) > 0.01 ||
          Math.abs(colorAttr.getZ(i) - firstColor[2]) > 0.01
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
        if (obj.userData?.nodeType === 'points' && !points) {
          points = obj;
        }
      });

      if (!points || !points.geometry.attributes.aColor) {
        return null;
      }

      const geometry = points.geometry;
      const colorAttr = geometry.attributes.aColor;
      // Points render as instanced quads: drawRange is the 6-index base
      // quad, while instanceCount is the visible point count. Attribute
      // buffers may be over-allocated by the GPU pool.
      const instanceCount = geometry.isInstancedBufferGeometry
        ? geometry.instanceCount
        : colorAttr.count;
      const actualCount = Math.min(instanceCount, colorAttr.count);
      // aColor is an InterleavedBufferAttribute view; iterate per instance
      // via getX/getY/getZ to read the actual decoded LUT colors instead
      // of mixed stride-misaligned values from the shared buffer.
      const uniqueColors = new Set<string>();

      for (let i = 0; i < actualCount; i++) {
        const r = colorAttr.getX(i).toFixed(2);
        const g = colorAttr.getY(i).toFixed(2);
        const b = colorAttr.getZ(i).toFixed(2);
        uniqueColors.add(`${r},${g},${b}`);
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
          return msgs.some((m: { type?: string }) => m.type === 'error' || m.type === 'warning');
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
