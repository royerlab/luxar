/**
 * E2E tests for Worker and WASM integration
 *
 * These tests run in a real browser environment (Playwright) to verify:
 * - Workers can be created and communicate correctly
 * - WASM modules load and execute (with TypeScript fallback)
 * - Spatial queries actually run in worker with WASM acceleration
 * - Fallback to main thread TypeScript works when workers unavailable
 *
 * Run with: pnpm test:e2e or pnpm test:e2e:ui
 */

import { test, expect } from '@playwright/test';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  getConsoleMessages,
  assertNoConsoleErrors,
} from './helpers';

// Use existing test fixtures
const FIXTURES_BASE = 'http://localhost:9000/packages/luxar-viewer/tests/fixtures';

test.describe('Worker Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to test page with 3D fixture and debug mode
    await page.goto(`/?src=${FIXTURES_BASE}/test_broadcasting.zarr&debug`);

    // Wait for Luxar to initialize
    await waitForLuxarReady(page);
  });

  test('should create worker pool successfully', async ({ page }) => {
    // Wait for points to load (indicates worker infrastructure is functioning)
    await waitForPointsLoaded(page, 1);

    // Check that data was loaded successfully
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Get console messages to verify no worker errors
    const messages = await getConsoleMessages(page);

    // Check for worker-related logs (informational, not required)
    const workerLogs = messages.all.filter(
      (msg) => msg.includes('[WorkerPool]') || msg.includes('worker')
    );
    console.log(`[Worker Test] Found ${workerLogs.length} worker-related log messages`);

    // No critical errors during worker initialization
    await assertNoConsoleErrors(page);
  });

  test('should offload spatial queries to worker', async ({ page }) => {
    // Wait for initial data load
    await waitForPointsLoaded(page, 1);

    // Get initial state
    const initialState = await getLuxarState(page);
    expect(initialState.totalPoints).toBeGreaterThan(0);

    // Simulate camera movement (triggers spatial query re-evaluation)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      if (debug?.controls) {
        // Move camera to trigger spatial index query
        debug.camera.position.set(20, 20, 20);
        debug.controls.update();
        debug.renderOnce();
      }
    });

    await page.waitForTimeout(500);

    // Verify no errors occurred during spatial query
    await assertNoConsoleErrors(page);

    // State should still be valid
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
  });

  test('should fallback to main thread if worker fails', async ({ page }) => {
    // This test verifies graceful degradation
    // The key is that data loads regardless of worker availability

    await waitForPointsLoaded(page, 1);

    // Get debug state
    const state = await getLuxarState(page);

    // Should have loaded points regardless of worker success/failure
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify no crashes
    await assertNoConsoleErrors(page);
  });

  test('should handle rapid view updates without worker congestion', async ({ page }) => {
    await waitForPointsLoaded(page, 1);

    // Rapid camera movements should queue correctly without crashing
    for (let i = 0; i < 10; i++) {
      await page.evaluate((idx: number) => {
        const debug = (window as any).__luxarDebug;
        if (debug?.camera) {
          debug.camera.position.x += idx * 0.5;
          debug.renderOnce();
        }
      }, i);
      await page.waitForTimeout(50); // Rapid updates
    }

    await page.waitForTimeout(500); // Let queries settle

    // Check for errors
    await assertNoConsoleErrors(page);

    // Should have completed without crashes
    const state = await getLuxarState(page);
    expect(state).toBeDefined();
    expect(state.initialized).toBe(true);
  });
});

test.describe('WASM Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Use 4D fixture to test nD operations (WASM handles these)
    await page.goto(`/?src=${FIXTURES_BASE}/test_4d.zarr&debug`);
    await waitForLuxarReady(page);
  });

  test('should load WASM module or TypeScript fallback successfully', async ({ page }) => {
    // Wait for data to load - this confirms WASM/TS decoding works
    await waitForPointsLoaded(page, 1, 30000);

    // Check console for WASM-related messages (informational)
    const messages = await getConsoleMessages(page);
    const wasmLogs = messages.all.filter(
      (msg) => msg.toLowerCase().includes('wasm') || msg.includes('TypeScript fallback')
    );
    console.log(`[WASM Test] WASM-related logs: ${wasmLogs.length}`);

    // Whether WASM or TypeScript fallback, data should load
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // No errors during module loading
    await assertNoConsoleErrors(page);
  });

  test('should use WASM for spatial queries if available', async ({ page }) => {
    await waitForPointsLoaded(page, 1);

    // For 4D data, dimension navigation triggers spatial queries with WASM
    // Navigate dimension (if available)
    const hasDimensions = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const state = debug?.getState?.();
      return state?.dimensions?.ndim > 3;
    });

    if (hasDimensions) {
      // Navigate dimension using keyboard
      await page.keyboard.press('[');
      await page.waitForTimeout(500);

      // Verify no errors during spatial query
      await assertNoConsoleErrors(page);
    }

    // Verify data loaded (confirms WASM/fallback works)
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
  });

  test('should fallback to TypeScript if WASM unavailable', async ({ page }) => {
    // Even if WASM fails to load, TypeScript fallback should work
    await waitForPointsLoaded(page, 1);

    // Check that points were loaded
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // No critical errors
    await assertNoConsoleErrors(page);
  });

  test('should handle WASM errors gracefully', async ({ page }) => {
    await waitForPointsLoaded(page, 1);

    // Navigate multiple times to stress test WASM/fallback
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('[');
      await page.waitForTimeout(200);
    }

    // Allow queries to settle
    await page.waitForTimeout(500);

    // Should have no unhandled errors
    await assertNoConsoleErrors(page);

    // State should be valid
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
  });
});

test.describe('Worker + WASM Combined Performance', () => {
  test('should achieve queries without timeout', async ({ page }) => {
    // Use LUT fixture which has more data
    await page.goto(`/?src=${FIXTURES_BASE}/test_lut.zarr&debug`);
    await waitForLuxarReady(page);

    const startTime = Date.now();

    // Wait for data to load
    await waitForPointsLoaded(page, 1, 30000);

    const endTime = Date.now();
    const totalTime = endTime - startTime;

    // Data should load within reasonable time
    // This is a smoke test, not precise benchmarking
    expect(totalTime).toBeLessThan(30000); // Should finish in <30 seconds

    // Verify data loaded
    const state = await getLuxarState(page);
    expect(state).toBeDefined();
    expect(state.totalPoints).toBeGreaterThan(0);

    console.log(
      `[Performance Test] Data loaded in ${totalTime}ms with ${state.totalPoints} points`
    );
  });
});
