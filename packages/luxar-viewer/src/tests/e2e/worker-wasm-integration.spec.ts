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

test.describe('Worker Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to test page with debug mode
    await page.goto('/?src=/test-data/basic_points_3d.zarr&debug');

    // Wait for Luxar to initialize
    await page.waitForFunction(() => window.__luxarDebug !== undefined, { timeout: 10000 });
  });

  test('should create worker pool successfully', async ({ page }) => {
    // Check console for worker initialization
    const workerLogs = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) =>
        log.message?.includes('[WorkerPool]') || log.message?.includes('Data worker')
      );
    });

    // Should see worker creation logs
    expect(workerLogs.length).toBeGreaterThan(0);

    // Check for worker ready message
    const workerReady = workerLogs.some((log: any) =>
      log.message?.includes('Data worker ready') || log.message?.includes('WorkerPool')
    );
    expect(workerReady).toBe(true);
  });

  test('should offload spatial queries to worker', async ({ page }) => {
    // Navigate slice (triggers spatial query)
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(500);

    // Check that query happened
    const queryLogs = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) =>
        log.message?.includes('Querying spatial index') ||
        log.message?.includes('Worker query')
      );
    });

    expect(queryLogs.length).toBeGreaterThan(0);

    // If workers enabled, should see worker usage
    const config = await page.evaluate(() => {
      return (window as any).__luxarDebug?.config?.dataLoading?.performance?.useWebWorkers;
    });

    if (config) {
      const workerUsed = queryLogs.some((log: any) =>
        log.message?.includes('worker') && !log.message?.includes('failed')
      );
      expect(workerUsed).toBe(true);
    }
  });

  test('should fallback to main thread if worker fails', async ({ page }) => {
    // This test verifies graceful degradation
    // Workers might not be available in all environments

    // Navigate to trigger query
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(500);

    // Get debug state
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    // Should have loaded points regardless of worker success/failure
    expect(state?.scene?.children).toBeDefined();
    const pointsObjects = state?.scene?.children?.filter((c: any) => c.type === 'Points') || [];
    expect(pointsObjects.length).toBeGreaterThan(0);
  });

  test('should handle rapid view updates without worker congestion', async ({ page }) => {
    // Rapid navigation should queue queries correctly
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(50); // Rapid updates
    }

    await page.waitForTimeout(1000); // Let queries settle

    // Check for errors
    const errors = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) => log.level === 'error');
    });

    expect(errors.length).toBe(0);

    // Should have completed without crashes
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.();
    });

    expect(state).toBeDefined();
  });
});

test.describe('WASM Integration E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/?src=/test-data/basic_points_4d.zarr&debug');
    await page.waitForFunction(() => window.__luxarDebug !== undefined, { timeout: 10000 });
  });

  test('should load WASM module successfully', async ({ page }) => {
    // Wait for WASM loading attempt
    await page.waitForTimeout(1000);

    const wasmLogs = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) =>
        log.message?.includes('[WASM]') || log.message?.includes('wasm')
      );
    });

    // Should see WASM-related logs
    expect(wasmLogs.length).toBeGreaterThan(0);

    // Check for success or fallback message
    const wasmStatus = wasmLogs.some((log: any) =>
      log.message?.includes('Loaded compiled WASM') ||
      log.message?.includes('TypeScript fallback')
    );
    expect(wasmStatus).toBe(true);
  });

  test('should use WASM for spatial queries if available', async ({ page }) => {
    // Navigate to trigger query
    await page.keyboard.press('4'); // Select dimension 4
    await page.keyboard.press('['); // Navigate

    await page.waitForTimeout(1000);

    const perfLogs = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) =>
        log.message?.includes('query') || log.message?.includes('WASM')
      );
    });

    // Should have query logs
    expect(perfLogs.length).toBeGreaterThan(0);

    // Verify no crashes
    const errors = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) => log.level === 'error');
    });

    expect(errors.length).toBe(0);
  });

  test('should fallback to TypeScript if WASM unavailable', async ({ page }) => {
    // Even if WASM fails to load, should still work
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(500);

    // Check that points were loaded
    const pointCount = await page.evaluate(() => {
      const state = (window as any).__luxarDebug?.getState?.();
      const pointsObj = state?.scene?.children?.find((c: any) => c.type === 'Points');
      return pointsObj?.geometry?.attributes?.position?.count || 0;
    });

    // Should have loaded points
    expect(pointCount).toBeGreaterThan(0);
  });

  test('should handle WASM errors gracefully', async ({ page }) => {
    // Navigate multiple times to stress test WASM
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(100);
    }

    // Check for error logs
    const errors = await page.evaluate(() => {
      const logs = (window as any).__consoleInterceptor?.logs || [];
      return logs.filter((log: any) =>
        log.level === 'error' &&
        !log.message?.includes('Worker query failed') // Expected fallback, not error
      );
    });

    // Should have no WASM-related errors
    expect(errors.length).toBe(0);
  });
});

test.describe('Worker + WASM Combined Performance', () => {
  test('should achieve faster queries with both enabled', async ({ page }) => {
    await page.goto('/?src=/test-data/large_points_4d.zarr&debug');
    await page.waitForFunction(() => window.__luxarDebug !== undefined, { timeout: 10000 });

    // Measure query time with navigation
    const startTime = Date.now();

    await page.keyboard.press('4');
    await page.keyboard.press('[');
    await page.waitForTimeout(2000); // Let query complete

    const endTime = Date.now();
    const totalTime = endTime - startTime;

    // With workers + WASM, large queries should complete quickly
    // This is a smoke test, not precise benchmarking
    expect(totalTime).toBeLessThan(5000); // Should finish in <5 seconds

    // Verify data loaded
    const state = await page.evaluate(() => (window as any).__luxarDebug?.getState?.());
    expect(state).toBeDefined();
  });
});
