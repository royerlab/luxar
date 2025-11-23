/**
 * AI Debugging Demonstration Tests
 *
 * These tests demonstrate how Claude Code (or any AI agent) can use
 * Playwright to debug the Luxar viewer autonomously.
 *
 * Each test simulates a realistic debugging scenario.
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState } from './helpers';

test.describe('AI Debugging Capabilities', () => {
  test('AI can inspect complete scene state', async ({ page }) => {
    // Scenario: AI needs to understand what's currently loaded
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // AI runs this to get complete state
    const completeState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      return {
        // From main.ts (base interface)
        hasApp: !!debug.app,
        version: debug.version,
        hasConsoleInterceptor: !!debug.consoleInterceptor,

        // From app.ts (runtime interface)
        hasScene: !!debug.scene,
        hasCamera: !!debug.camera,
        hasRenderer: !!debug.renderer,
        runtimeReady: debug.runtimeReady,

        // State snapshot
        state: debug.getState(),

        // Renderer capabilities
        renderer: debug.renderer
          ? {
            pixelRatio: debug.renderer.getPixelRatio(),
            size: debug.renderer.getSize({ width: 0, height: 0 }),
            maxTextureSize: debug.renderer.capabilities.maxTextureSize,
          }
          : null,
      };
    });

    // AI can verify all components are present
    expect(completeState.hasApp).toBe(true);
    expect(completeState.hasScene).toBe(true);
    expect(completeState.hasCamera).toBe(true);
    expect(completeState.runtimeReady).toBe(true);

    // AI can inspect state
    expect(completeState.state.initialized).toBe(true);

    // AI can check renderer capabilities
    if (completeState.renderer) {
      expect(completeState.renderer.pixelRatio).toBeGreaterThan(0);
      expect(completeState.renderer.size.width).toBeGreaterThan(0);
      expect(completeState.renderer.maxTextureSize).toBeGreaterThan(0);
    }
  });

  test('AI can monitor console messages in real-time', async ({ page }) => {
    // Scenario: AI needs to see what's being logged

    const consoleLogs: Array<{ type: string; message: string }> = [];

    // Capture ALL console messages
    page.on('console', (msg) => {
      consoleLogs.push({
        type: msg.type(),
        message: msg.text(),
      });
    });

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // AI can analyze the logs
    const logMessages = consoleLogs.map((log) => log.message);

    // Should see initialization messages
    const hasStartupLog = logMessages.some(
      (msg) => msg.includes('Application starting') || msg.includes('Luxar')
    );
    expect(hasStartupLog).toBe(true);

    // Should see debug interface message
    const hasDebugLog = logMessages.some(
      (msg) => msg.includes('Debug interface') || msg.includes('__luxarDebug')
    );
    expect(hasDebugLog).toBe(true);

    // AI can count errors
    const errorCount = consoleLogs.filter((log) => log.type === 'error').length;
    // Should have no unexpected errors (some 404s for optional features are OK)
    expect(errorCount).toBeLessThan(10); // Allow some expected 404s
  });

  test('AI can verify WebGL context is valid', async ({ page }) => {
    // Scenario: AI debugging "black screen" issue
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const webglInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const canvas = document.querySelector('canvas') as HTMLCanvasElement;

      if (!canvas) return { error: 'No canvas element' };

      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');

      return {
        hasCanvas: true,
        canvasSize: { width: canvas.width, height: canvas.height },
        hasWebGL: !!gl,
        webglVersion: gl instanceof WebGL2RenderingContext ? 2 : 1,
        rendererReady: !!debug.renderer,
        contextLost: canvas.getContext('webgl2')?.isContextLost() ?? null,
      };
    });

    // AI verifies WebGL is working
    expect(webglInfo.hasCanvas).toBe(true);
    expect(webglInfo.hasWebGL).toBe(true);
    expect(webglInfo.rendererReady).toBe(true);
    expect(webglInfo.contextLost).toBe(false);
    expect(webglInfo.canvasSize?.width).toBeGreaterThan(0);
    expect(webglInfo.canvasSize?.height).toBeGreaterThan(0);
  });

  test('AI can diagnose memory usage', async ({ page }) => {
    // Scenario: AI investigating potential memory leaks
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const memoryInfo = await page.evaluate(() => {
      const perf = performance as any;

      return {
        // Browser memory API (Chrome only)
        jsHeapSize: perf.memory?.usedJSHeapSize || null,
        jsHeapSizeLimit: perf.memory?.jsHeapSizeLimit || null,

        // WebGL memory (approximate)
        rendererInfo: (window as any).__luxarDebug.renderer?.info?.memory || null,
      };
    });

    // AI can check if memory is in reasonable range
    if (memoryInfo.jsHeapSize) {
      const heapMB = memoryInfo.jsHeapSize / 1024 / 1024;
      expect(heapMB).toBeLessThan(1000); // Less than 1GB (reasonable)
    }

    // AI can verify WebGL resources are tracked
    if (memoryInfo.rendererInfo) {
      expect(memoryInfo.rendererInfo.geometries).toBeDefined();
      expect(memoryInfo.rendererInfo.textures).toBeDefined();
    }
  });

  test('AI can detect and report initialization failures', async ({ page }) => {
    // Scenario: AI detects that initialization failed

    const errors: string[] = [];
    const warnings: string[] = [];

    page.on('pageerror', (err) => errors.push(err.message));
    page.on('console', (msg) => {
      if (msg.type() === 'warning') warnings.push(msg.text());
    });

    await page.goto('/?debug');

    // Try to wait for initialization (with timeout)
    const initialized = await page
      .waitForFunction(
        () => {
          const debug = (window as any).__luxarDebug;
          return debug?.getState?.()?.initialized === true;
        },
        { timeout: 15000 }
      )
      .then(() => true)
      .catch(() => false);

    // AI can report the state
    if (!initialized) {
      // Get whatever state is available
      const partialState = await page
        .evaluate(() => {
          const debug = (window as any).__luxarDebug;
          return {
            debugExists: !!debug,
            hasGetState: !!debug?.getState,
            rawState: debug?.getState?.() || null,
          };
        })
        .catch(() => null);

      // AI reports: "Initialization failed, state: ..."
      console.log('Initialization state:', partialState);
      console.log('Errors:', errors);
      console.log('Warnings:', warnings);
    }

    // For this test, we expect initialization to succeed
    expect(initialized).toBe(true);
  });

  test('AI can verify all required components are initialized', async ({ page }) => {
    // Scenario: Systematic component check
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const components = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      return {
        // Base components (from main.ts)
        base: {
          app: !!debug.app,
          consoleInterceptor: !!debug.consoleInterceptor,
          version: !!debug.version,
        },

        // Runtime components (from app.ts)
        runtime: {
          scene: !!debug.scene,
          camera: !!debug.camera,
          renderer: !!debug.renderer,
          controls: !!debug.controls,
          postProcessing: !!debug.postProcessing,
          animationController: !!debug.animationController,
          inputHandler: !!debug.inputHandler,
          renderingControls: !!debug.renderingControls,
        },

        // Helper functions
        helpers: {
          getState: typeof debug.getState === 'function',
          renderOnce: typeof debug.renderOnce === 'function',
          getSceneLoader: typeof debug.getSceneLoader === 'function',
        },

        // Flags
        flags: {
          runtimeReady: debug.runtimeReady === true,
        },
      };
    });

    // AI verifies all base components
    expect(components.base.app).toBe(true);
    expect(components.base.consoleInterceptor).toBe(true);
    expect(components.base.version).toBe(true);

    // AI verifies all runtime components
    expect(components.runtime.scene).toBe(true);
    expect(components.runtime.camera).toBe(true);
    expect(components.runtime.renderer).toBe(true);
    expect(components.runtime.controls).toBe(true);

    // AI verifies helper functions
    expect(components.helpers.getState).toBe(true);
    expect(components.helpers.renderOnce).toBe(true);
    expect(components.helpers.getSceneLoader).toBe(true);

    // AI verifies runtime ready flag
    expect(components.flags.runtimeReady).toBe(true);
  });

  test('AI can capture and analyze network requests', async ({ page }) => {
    // Scenario: AI debugging "data won't load" issue

    const requests: Array<{ url: string; status: number; resourceType: string }> = [];
    const failures: Array<{ url: string; error: string }> = [];

    // Capture all requests
    page.on('response', async (response) => {
      requests.push({
        url: response.url(),
        status: response.status(),
        resourceType: response.request().resourceType(),
      });
    });

    // Capture failures
    page.on('requestfailed', (request) => {
      failures.push({
        url: request.url(),
        error: request.failure()?.errorText || 'unknown',
      });
    });

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // AI can analyze network activity
    const htmlRequests = requests.filter((r) => r.resourceType === 'document');
    const scriptRequests = requests.filter((r) => r.resourceType === 'script');

    // Should have loaded the HTML
    expect(htmlRequests.length).toBeGreaterThan(0);

    // Should have loaded scripts
    expect(scriptRequests.length).toBeGreaterThan(0);

    // AI can report: "Found X failed requests"
    if (failures.length > 0) {
      console.log('Failed requests:', failures);
    }

    // Some 404s are expected (optional features like spatial index)
    // But shouldn't have too many failures
    expect(failures.length).toBeLessThan(20);
  });

  test('AI can take diagnostic screenshots at different stages', async ({ page }) => {
    // Scenario: AI documenting the rendering pipeline

    await page.goto('/?debug');

    // Stage 1: After page load (before initialization)
    await page.screenshot({ path: 'test-results/stage1-loaded.png' });

    // Stage 2: After initialization
    await waitForLuxarReady(page);
    await page.screenshot({ path: 'test-results/stage2-initialized.png' });

    // Stage 3: After first render
    await page.evaluate(() => {
      (window as any).__luxarDebug.renderOnce();
    });
    await page.waitForTimeout(200);
    await page.screenshot({ path: 'test-results/stage3-rendered.png' });

    // AI can now compare these screenshots to diagnose issues
    // E.g., if stage3 is black, rendering is broken
    // E.g., if stage2 shows error, initialization is broken

    // Verify we got through all stages
    const finalState = await getLuxarState(page);
    expect(finalState.initialized).toBe(true);
  });
});
