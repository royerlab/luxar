/**
 * Performance-benchmark Playwright config.
 *
 * Separate from `playwright.config.ts` so the main E2E run doesn't pay
 * the cost of perf sampling and so this config can opt into the
 * "as real as possible browser" mode: the system's installed Chrome
 * via `channel: 'chrome'`, with WebGPU developer features enabled so
 * the bench can use `timestamp-query` when the GPU supports it.
 *
 * Invoke via:
 *   pnpm test:perf:e2e
 *
 * Defaults to headed mode so the user can observe what's being
 * measured. Set `LUXAR_PERF_HEADLESS=1` to flip to headless (e.g. CI).
 *
 * The bench writes one JSON file per commit SHA into
 * `perf-results/{sha}/results.json`. `scripts/perf-diff.mjs` compares
 * two such files and prints a Markdown delta table.
 *
 * @module playwright.perf.config
 */

import { defineConfig, devices } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const headless = process.env.LUXAR_PERF_HEADLESS === '1';

export default defineConfig({
  globalSetup: path.join(__dirname, 'src/tests/e2e/global-setup.ts'),
  testDir: './src/tests/e2e',
  // Only run perf specs (the line-perf-bench file + future siblings).
  testMatch: /.*perf-bench\.spec\.ts$/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  // Perf runs must be serial — concurrent GPU contention destroys signal.
  workers: 1,
  reporter: [['list']],

  use: {
    baseURL: 'http://localhost:5173',
    trace: 'off',
    screenshot: 'off',
    video: 'off',
    actionTimeout: 30000,
    navigationTimeout: 120000,
    headless,
    launchOptions: {
      args: [
        // GPU-related flags — keep parity with the main config so the
        // perf result is comparable to standard E2E rendering.
        '--use-gl=egl',
        '--ignore-gpu-blocklist',
        '--enable-webgl-developer-extensions',
        '--enable-webgl-draft-extensions',
        '--disable-web-security',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        // WebGPU + timestamp-query support. `--enable-unsafe-webgpu`
        // exposes the device on platforms where Chrome would otherwise
        // require a flag; `--enable-webgpu-developer-features` exposes
        // the `timestamp-query` feature for GPU-side timing in the
        // bench. No-op when not on a supported GPU/driver — the bench
        // feature-detects.
        '--enable-unsafe-webgpu',
        '--enable-webgpu-developer-features',
        '--enable-features=Vulkan,WebGPU',
      ],
    },
  },

  projects: [
    {
      name: 'chrome-real-gpu',
      use: {
        ...devices['Desktop Chrome'],
        // Use the system-installed Chrome rather than Playwright's
        // bundled Chromium so the GPU driver path matches what an
        // end-user would see. The user explicitly asked for "as real
        // as possible browser".
        channel: 'chrome',
      },
    },
  ],

  // Reuse the standard viewer + dataset dev servers.
  webServer: [
    {
      command: 'pnpm dev',
      url: 'http://localhost:5173',
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
      command: 'python3 -m http.server 9000',
      url: 'http://localhost:9000',
      cwd: path.resolve(__dirname, '../..'),
      reuseExistingServer: !process.env.CI,
      timeout: 15000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
  ],

  outputDir: 'test-results-perf/',
  // Perf samples are long-running — generous per-test timeout.
  timeout: 900_000,
});
