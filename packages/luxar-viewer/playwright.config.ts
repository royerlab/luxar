/**
 * Playwright Configuration for Luxar Viewer
 *
 * This configuration is specifically optimized for testing Three.js/WebGL applications.
 * Key features:
 * - GPU acceleration for realistic rendering
 * - Higher pixel diff tolerance for WebGL variability
 * - Trace capture for debugging
 * - Automatic dev server startup
 */

import { defineConfig, devices } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { createE2EServerMetadata, ensureCheckoutIdentity } from './tools/e2e-server-identity';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at config load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');
const viewerBaseURL = 'http://127.0.0.1:5173';
const dataBaseURL = 'http://127.0.0.1:9000';
const checkoutIdentity = ensureCheckoutIdentity(projectRoot, __dirname);
const serverMetadata = createE2EServerMetadata(checkoutIdentity, viewerBaseURL, dataBaseURL);

/**
 * See https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
  metadata: { luxarE2E: serverMetadata },
  // Global setup - runs before any tests
  globalSetup: path.join(__dirname, 'src/tests/e2e/global-setup.ts'),

  // Test directory
  testDir: './src/tests/e2e',

  // Perf benchmarks (*perf-bench.spec.ts) run only under the dedicated
  // `playwright.perf.config.ts` so their sampling cost — and the
  // generated `perf-results/<sha>/` JSON they write — stay out of the
  // default `pnpm test:e2e` run.
  testIgnore: /.*perf-bench\.spec\.ts$/,

  // Run tests in files in parallel
  fullyParallel: false, // WebGL tests can be GPU-intensive, run serially

  // Fail the build on CI if you accidentally left test.only
  forbidOnly: !!process.env.CI,

  // Retry flaky tests
  // WebGL tests can be flaky due to GPU timing, driver variability, and resource contention.
  // - Local: 0 retries so developers see flaky failures immediately.
  // - CI: 2 retries because CI environments have more variability.
  retries: process.env.CI ? 2 : 0,

  // Local: 2 workers for ~2x speedup (most GPUs handle 2 concurrent WebGL contexts)
  // CI: 1 worker (software rendering is slower and less stable with concurrency)
  workers: process.env.CI ? 1 : 2,

  // Reporter to use
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['list'],
    ...(process.env.CI ? [['github' as const]] : []),
  ],

  // Shared settings for all projects
  use: {
    // Base URL for tests
    baseURL: viewerBaseURL,

    // Collect trace on failure for debugging
    trace: 'retain-on-failure',

    // Screenshot settings - capture visual state for inspection
    // All artifacts (screenshots, videos, traces) saved to test-results/
    screenshot: 'on', // Always take screenshots for visual debugging

    // Video on failure (useful but large files)
    video: 'retain-on-failure',

    // Maximum time for each action (click, fill, etc.)
    actionTimeout: 10000,

    // Navigation timeout
    navigationTimeout: 60000,

    // ========================================================================
    // CRITICAL: GPU ACCELERATION FLAGS FOR WEBGL/THREE.JS
    // ========================================================================
    // These flags force Chromium to use hardware acceleration even in headless mode.
    // Without these, WebGL falls back to software rendering (SwiftShader), which is:
    // - 10-100x slower
    // - Produces different pixels (visual regression tests fail)
    // - May cause timeouts or crashes
    launchOptions: {
      args: [
        '--use-gl=egl', // Force GPU acceleration
        '--ignore-gpu-blocklist', // Unblock older/CI GPUs
        '--enable-webgl-developer-extensions', // Enable WebGL extensions
        '--enable-webgl-draft-extensions', // Enable draft extensions
        '--disable-web-security', // Allow CORS for local testing
        '--no-sandbox', // Often needed in CI environments
        '--disable-setuid-sandbox',
      ],
    },
  },

  // Configure projects for different browsers
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
        // Use Playwright's bundled Chromium for consistency
        // If you want to use installed Chrome, uncomment the line below:
        // channel: 'chrome',
      },
    },

    // Uncomment to test on other browsers (note: WebGL support varies)
    // {
    //   name: 'firefox',
    //   use: { ...devices['Desktop Firefox'] },
    // },
    // {
    //   name: 'webkit',
    //   use: { ...devices['Desktop Safari'] },
    // },
  ],

  // Run local dev servers before starting tests
  // Start BOTH the viewer dev server AND a server for examples/
  webServer: [
    {
      // TypeScript viewer dev server.
      // VITE_LUXAR_USE_WEBGPU is forwarded so per-test runs
      // (e.g. `VITE_LUXAR_USE_WEBGPU=1 pnpm playwright test`) can opt
      // into the WebGPU path on the dev server. Without forwarding,
      // Playwright spawns (or reuses) the server in its own env and
      // the flag is lost. `VITE_LUXAR_USE_LEGACY_WEBGL` (a no-op because
      // WebGL is the default) and `VITE_LUXAR_USE_WEBGPU_RENDERER` (an
      // alias for the WebGPU opt-in) are forwarded too so existing CI
      // invocations keep working harmlessly.
      command: 'pnpm dev --host 127.0.0.1 --strictPort',
      // A sibling checkout has a different marker path and cannot satisfy this
      // readiness probe. If its Vite owns 5173, strictPort fails loudly.
      url: serverMetadata.viewerIdentityURL,
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
      stdout: 'pipe',
      stderr: 'pipe',
      env: {
        VITE_LUXAR_USE_WEBGPU: process.env.VITE_LUXAR_USE_WEBGPU ?? '',
        VITE_LUXAR_USE_LEGACY_WEBGL: process.env.VITE_LUXAR_USE_LEGACY_WEBGL ?? '',
        VITE_LUXAR_USE_WEBGPU_RENDERER: process.env.VITE_LUXAR_USE_WEBGPU_RENDERER ?? '',
      },
    },
    {
      // Python HTTP server to serve repository datasets/examples for E2E tests.
      // E2E global setup checks expected datasets and reports any missing fixtures.
      // Using port 9000 (ports 8000-8001 are used by luxar serve)
      command: 'python3 -m http.server 9000 --bind 127.0.0.1',
      // The marker is an ignored file unique to this checkout. A static server
      // rooted in a sibling worktree returns 404 instead of being reused.
      url: serverMetadata.dataIdentityURL,
      // Use cwd to set working directory to project root (2 levels up from this file)
      cwd: projectRoot,
      reuseExistingServer: !process.env.CI,
      timeout: 15000, // Increased timeout for reliability
      stdout: 'ignore', // Reduce noise in test output
      stderr: 'pipe', // Still capture errors
    },
  ],

  // Output directory for test artifacts
  outputDir: 'test-results/',

  // Test timeout (individual test)
  timeout: 60000, // 60 seconds per test (WebGL init can be slow)

  // Expect timeout (for assertions)
  expect: {
    // ========================================================================
    // CRITICAL: RELAXED PIXEL MATCHING FOR WEBGL
    // ========================================================================
    // WebGL rendering is non-deterministic across GPUs. A scene rendered on:
    // - NVIDIA GPU vs AMD GPU
    // - Mac M1 vs Intel integrated graphics
    // - CI server vs local machine
    // ...will have slightly different anti-aliasing, color precision, etc.
    //
    // Standard screenshot tests (0% tolerance) will fail 100% of the time.
    // We relax the threshold to allow minor pixel differences while still
    // catching real visual regressions.
    toHaveScreenshot: {
      // Allow 5% of pixels to differ (vs 0% for standard DOM apps)
      maxDiffPixelRatio: 0.05,

      // Allow 0.2 color difference per channel (0-1 scale)
      // This tolerates minor anti-aliasing differences
      threshold: 0.2,

      // Disable CSS animation detection (doesn't work with WebGL)
      animations: 'disabled' as const,

      // Take multiple screenshots to ensure scene is stable
      // (Three.js render loop might still be animating)
      timeout: 10000,
    },

    // Timeout for expect() assertions
    // Increased for E2E tests with real dataset loading
    timeout: 60000, // 60 seconds for dataset loading + rendering
  },
});
