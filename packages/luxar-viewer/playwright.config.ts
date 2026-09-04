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
import { e2eWorkerPlan } from './tools/e2e-workers';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at config load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');
// The VIEWER port is overridable per run: a sibling checkout (another agent
// or session) sometimes holds 5173 with its own Vite, which the identity
// guard below rightly refuses to reuse — `LUXAR_E2E_VIEWER_PORT=5199` runs
// the suite alongside it instead of fighting over the port. The DATA port is
// deliberately NOT overridable: spec files hardcode `http://localhost:9000/`
// dataset URLs (they are page-side absolute URLs, not baseURL-relative).
const viewerPort = process.env.LUXAR_E2E_VIEWER_PORT ?? '5173';
const viewerBaseURL = `http://127.0.0.1:${viewerPort}`;
const dataBaseURL = 'http://127.0.0.1:9000';
const checkoutIdentity = ensureCheckoutIdentity(projectRoot, __dirname);
const serverMetadata = createE2EServerMetadata(checkoutIdentity, viewerBaseURL, dataBaseURL);

// Parallelism is decided once, here (see the `workers:` comment below). This module stays
// side-effect-free: the run's parallelism line is printed by the E2E global setup, which — unlike
// this file — is handed the RESOLVED config and can therefore report the count Playwright will
// actually use after `--workers=N` / `--debug` have had their say.
const workerPlan = e2eWorkerPlan();

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

  // Tests within a file run in parallel too.
  //
  // The suite is state-independent by construction: every test takes the
  // per-test `page` fixture, no file creates a shared page in `beforeAll`, and
  // there is no `test.describe.serial` anywhere. What genuinely cannot share is
  // a handful of RESOURCE-bound files — a spawned `luxar serve`, concurrent
  // `hatch` invocations, FPS benchmarks, and files that are simultaneously GPU-
  // context- and dataset-server-bound (webgl-errors.spec.ts: ~14 scene loads
  // over the port-9000 server, each with its own WebGL context and OPFS
  // write-through) — and those opt OUT locally with
  // `test.describe.configure({ mode: 'default' })`, which pins the file to one
  // worker without `'serial'`'s skip-the-rest-after-a-failure behaviour. The
  // perf benches, which do want that, keep `'serial'`.
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only
  forbidOnly: !!process.env.CI,

  // The checked-in visual corpus is recorded on Linux. Other platforms still
  // run the tests, but must not create platform-specific snapshots in source.
  ignoreSnapshots: process.platform !== 'linux',

  // Retry flaky tests
  // WebGL tests can be flaky due to GPU timing, driver variability, and resource contention.
  // - Local: 0 retries so developers see flaky failures immediately.
  // - CI: 2 retries because CI environments have more variability.
  retries: process.env.CI ? 2 : 0,

  // Local: up to 4 workers, scaled DOWN by how loaded the box is. CI: 1
  // (software rendering is slower and less stable with concurrency).
  //
  // The CEILING of 4 is NOT the GPU — it is the Python dataset server (see
  // webServer below) streaming thousands of small zarr chunks to every worker
  // at once. The evidence is already in the tree: all-examples-smoke-test.spec.ts
  // raised its own timeout to 120 s to "absorb HTTP-server contention when
  // several worker-pool tabs decode mid-size datasets concurrently". Raise it
  // past 4 only together with a measurement, and if it saturates, replace that
  // server rather than adding workers.
  //
  // Running AT the ceiling on a busy box invents failures. On a shared 16-core
  // workstation at a 1-minute load of 12-24, dimension-animation.spec.ts failed
  // 15 of 21 tests at 4 workers and passed 21 of 21 at `--workers=1` — every
  // failure a bare wall-clock timeout (`page.click: Timeout 10000ms exceeded`
  // with the element already visible/enabled/stable), no product cause, and two
  // issues filed as viewer regressions off exactly that. So the ceiling is now
  // scaled by the box's free fraction:
  // `clamp(round(4 * (cpus - load1) / cpus), 1, 4)`. The bands are fractions of
  // the box — 4 while at least 7/8 of it is free, 3 down to 5/8, 2 down to 3/8,
  // 1 below that, which on 16 cores is 4 up to load 2, 3 to load 6, 2 to load
  // 10, then 1. An IDLE box of any size still runs at 4 — this only ever backs
  // off under load, giving a loaded box a slower run instead of a red one. Only
  // the two ends described above were measured; the counts between them are
  // interpolation. See tools/e2e-workers.ts, which also records what the
  // load average does and does not tell us. Pin it with `LUXAR_E2E_WORKERS=N`
  // (clamped to `[1, cpus]`), or with Playwright's own `--workers=N`, which
  // overrides this config entirely.
  workers: workerPlan.workers,

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

    // Artifact capture is FAILURE-ONLY.
    //
    // `retain-on-failure` still RECORDS for every test and deletes on pass, and
    // `screenshot: 'on'` wrote a PNG for all 567 tests — roughly 330-430 s of
    // worker time per run spent producing artifacts nobody looks at.
    //
    // `on-first-retry` keeps debuggability exactly where it is needed: CI sets
    // `retries: 2` (below) and `test:e2e:ci` passes `--retries 2`, so a failure
    // is retried and THAT run is fully traced and recorded. Locally `retries: 0`
    // — re-run the failing spec with `--trace on --video on` to get the same
    // artifacts on demand.
    //
    // Specs that write their own screenshots via `page.screenshot({ path })`
    // are unaffected, and the readme/doc-image/gallery generators use their own
    // configs (which already set all three to 'off').
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'on-first-retry',

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

    // Firefox and WebKit are OPT-IN, behind LUXAR_E2E_BROWSERS=all.
    //
    // They used to be commented out, which meant the README's four-browser
    // support table had never been executed even once -- there was no way to
    // run them. Real projects make the claim re-verifiable; gating them keeps
    // the default suite at one engine, because Playwright runs every declared
    // project and three engines would triple a ~17 min suite for a matrix the
    // GPU-backed daemon is the right home for.
    //
    //   LUXAR_E2E_BROWSERS=all pnpm exec playwright test --project=firefox
    //   LUXAR_E2E_BROWSERS=all pnpm test:e2e:browsers   (the smoke subset)
    //
    // Requires `npx playwright install firefox webkit` -- neither ships with
    // the default install.
    ...(process.env.LUXAR_E2E_BROWSERS === 'all'
      ? [
          {
            name: 'firefox',
            use: { ...devices['Desktop Firefox'] },
          },
          {
            name: 'webkit',
            use: { ...devices['Desktop Safari'] },
          },
        ]
      : []),
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
      command: `pnpm dev --host 127.0.0.1 --port ${viewerPort} --strictPort`,
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
      // Range-capable Python HTTP server for repository datasets/examples.
      // Directory stores need only plain GETs, but `.zarr.zip` reads require
      // strict 206 responses for byte windows inside the archive.
      // E2E global setup checks expected datasets and reports any missing fixtures.
      // Using port 9000 (ports 8000-8001 are used by luxar serve)
      command: 'python3 packages/luxar-viewer/tools/range-http-server.py 9000 --bind 127.0.0.1',
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
