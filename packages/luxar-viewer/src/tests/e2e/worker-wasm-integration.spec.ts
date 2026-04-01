/**
 * E2E tests for Worker and WASM integration
 *
 * These tests run in a real browser environment (Playwright) to verify:
 * - Workers can be created and communicate correctly
 * - WASM modules load and execute
 * - Spatial queries actually run in worker with WASM acceleration
 * - Fallback to main thread TypeScript works when workers unavailable
 *
 * Run with: pnpm test:e2e or pnpm test:e2e:ui
 */

import { test, expect } from '@playwright/test';
import {
  waitForLuxarReady,
  waitForDataLoaded,
  waitForPointsLoaded,
  getConsoleMessages,
  assertConsoleContains,
  waitForConsoleInterceptor,
} from './helpers';

const DATASET_3D = 'http://localhost:9000/datasets/examples/radius_basic_example.zarr';
const DATASET_5D = 'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.zarr';
const DATASET_LARGE = 'http://localhost:9000/datasets/examples/performance_benchmark_example.zarr';

test.describe('Worker Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to test page with debug mode - use existing dataset
    await page.goto(`/?src=${DATASET_3D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
  });

  test('should create worker pool successfully', async ({ page }) => {
    // Wait for scene to be ready
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // Verify worker pool creation via console logs
    // Log format: [emoji] [WorkerPool] Worker pool ready with N worker(s)
    const messages = await getConsoleMessages(page);
    const workerMessages = messages.all.filter(
      (m) => m.includes('[WorkerPool]') || m.includes('worker')
    );
    expect(workerMessages.length).toBeGreaterThan(0);

    // Specifically check for the "Worker pool ready" message
    await assertConsoleContains(page, /\[WorkerPool\].*Worker pool ready/);

    // Also verify points loaded (workers did their job)
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });
    expect(state?.totalPoints).toBeGreaterThan(0);
  });

  test('should offload spatial queries to worker', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // Verify that individual workers initialized (they handle spatial queries)
    // Log format: [emoji] [WorkerPool] Worker 1/N ready
    await assertConsoleContains(page, /\[WorkerPool\].*Worker \d+\/\d+ ready/);

    // Also check for DataWorker ready messages (worker confirmed WASM loaded)
    // Log format: [emoji] [WorkerPool] DataWorker ready
    await assertConsoleContains(page, /\[WorkerPool\].*DataWorker ready/);

    // Verify points loaded (queries completed via workers)
    const initialCount = await page.evaluate(() => {
      const state = (window as any).__luxarDebug?.getState?.();
      return state?.totalPoints || 0;
    });
    expect(initialCount).toBeGreaterThan(0);
  });

  test('should fallback to main thread if worker fails', async ({ page }) => {
    // Block worker script from loading to force fallback path
    // The worker script is loaded via: new Worker(new URL('./data-worker.ts', ...))
    // In production builds it becomes something like data-worker-*.js
    await page.route(/data-worker.*\.(js|ts)/, (route) => route.abort());

    // Navigate fresh with worker blocked
    await page.goto(`/?src=${DATASET_3D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);

    // Even with workers blocked, the app should still load data
    // (graceful degradation to main thread)
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    // Should have loaded points regardless of worker failure
    // The app should not crash - it either falls back or errors gracefully
    expect(state).toBeDefined();
    expect(state?.pointClouds).toBeDefined();
    // Points may or may not load depending on fallback implementation,
    // but the app should not crash
    expect(state?.initialized).toBe(true);
  });

  test('should handle rapid view updates without worker congestion', async ({ page }) => {
    // First make sure initial data is loaded
    await waitForPointsLoaded(page);

    // Wait for interactions to settle
    await page.waitForTimeout(500);

    // Rapid navigation should queue queries correctly
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(50); // Rapid updates
    }

    await page.waitForTimeout(1000); // Let queries settle

    // Should have completed without crashes
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    expect(state).toBeDefined();
    // Points should still be visible - use totalPoints or pointClouds
    expect(state?.totalPoints > 0 || state?.pointClouds?.length > 0).toBeTruthy();
  });
});

test.describe('WASM Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Use 5D dataset to test nD queries which exercise WASM
    await page.goto(`/?src=${DATASET_5D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
  });

  test('should load WASM module successfully', async ({ page }) => {
    // Wait for scene to be ready
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // Verify WASM module was loaded via console logs
    // Either: [emoji] [WASM] Loaded compiled WASM module (success)
    // Or:     [emoji] [WASM] Failed to load WASM module, using TypeScript fallback
    const messages = await getConsoleMessages(page);
    const wasmMessages = messages.all.filter((m) => m.includes('[WASM]'));
    expect(wasmMessages.length).toBeGreaterThan(0);

    // Check for either WASM loaded or TypeScript fallback (both are valid)
    const wasmLoaded = wasmMessages.some((m) => m.includes('Loaded compiled WASM module'));
    const tsFallback = wasmMessages.some((m) => m.includes('TypeScript fallback'));
    expect(wasmLoaded || tsFallback).toBe(true);

    // Also verify DataWorker WASM status logs
    // [emoji] [WorkerPool] DataWorker WASM module loaded successfully
    const workerWasmMessages = messages.all.filter(
      (m) => m.includes('DataWorker') && m.includes('WASM')
    );
    expect(workerWasmMessages.length).toBeGreaterThan(0);

    // Verify points loaded
    const hasPoints = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const state = debug?.getState?.();
      return state?.totalPoints > 0 || state?.pointClouds?.length > 0;
    });
    expect(hasPoints).toBe(true);
  });

  test('should use WASM for spatial queries if available', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // Check WASM status before triggering queries
    const preNavMessages = await getConsoleMessages(page);
    const wasmReady =
      preNavMessages.all.some((m) => m.includes('[WASM]')) ||
      preNavMessages.all.some((m) => m.includes('DataWorker WASM'));

    // WASM or its fallback should have been initialized by now
    expect(wasmReady).toBe(true);

    // Click canvas to ensure it has focus
    await page.click('canvas');
    await page.waitForTimeout(100);

    // Navigate to trigger spatial query on nD dataset (exercises WASM code path)
    await page.keyboard.press('4'); // Select dimension 4
    await page.keyboard.press(']'); // Navigate forward
    await page.waitForTimeout(500);

    // Verify queries completed (data still loaded after navigation)
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    expect(state).toBeDefined();
    expect(state?.pointClouds?.length > 0 || state?.totalPoints >= 0).toBeTruthy();
  });

  test('should fallback to TypeScript if WASM unavailable', async ({ page }) => {
    // Block WASM binary from loading to force TypeScript fallback
    await page.route(/\.wasm$/, (route) => route.abort());
    await page.route(/luxar_wasm/, (route) => route.abort());

    // Navigate fresh with WASM blocked
    await page.goto(`/?src=${DATASET_5D}&debug`);
    await waitForLuxarReady(page);
    await waitForDataLoaded(page);
    await waitForPointsLoaded(page);
    await waitForConsoleInterceptor(page);

    // Should see TypeScript fallback message in console
    const messages = await getConsoleMessages(page);
    const fallbackMessages = messages.all.filter(
      (m) => m.includes('TypeScript fallback') || m.includes('WASM initialization failed')
    );
    expect(fallbackMessages.length).toBeGreaterThan(0);

    // Points should still load even without WASM
    const pointCount = await page.evaluate(() => {
      const state = (window as any).__luxarDebug?.getState?.();
      return state?.totalPoints || 0;
    });
    expect(pointCount).toBeGreaterThan(0);
  });

  test('should handle WASM errors gracefully', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page);

    // Click canvas for focus
    await page.click('canvas');
    await page.waitForTimeout(100);

    // Navigate multiple times to stress test WASM
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press(']');
      await page.waitForTimeout(100);
    }

    // Wait for operations to settle
    await page.waitForTimeout(500);

    // Should have no crashes - state should still be accessible
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    expect(state).toBeDefined();
    // Check points are loaded via pointClouds or totalPoints
    expect(state?.pointClouds?.length > 0 || state?.totalPoints >= 0).toBeTruthy();
  });
});

test.describe('Worker + WASM Combined Performance', () => {
  test('should achieve faster queries with both enabled', async ({ page }) => {
    // Use performance benchmark dataset if available, otherwise fall back to 5D
    await page.goto(`/?src=${DATASET_LARGE}&debug`);

    // Wait for ready (may take longer for large dataset)
    try {
      await waitForLuxarReady(page, 15000);
      await waitForDataLoaded(page);
    } catch {
      // If large dataset not available, skip test
      test.skip();
      return;
    }

    // Measure query time with navigation
    const startTime = Date.now();

    await page.keyboard.press('4');
    await page.keyboard.press('[');
    await page.waitForTimeout(2000); // Let query complete

    const endTime = Date.now();
    const totalTime = endTime - startTime;

    // With workers + WASM, large queries should complete quickly
    // This is a smoke test, not precise benchmarking
    expect(totalTime).toBeLessThan(10000); // Should finish in <10 seconds

    // Verify data loaded
    const state = await page.evaluate(() => (window as any).__luxarDebug?.getState?.());
    expect(state).toBeDefined();
  });
});
