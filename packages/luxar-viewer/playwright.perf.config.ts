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
import { createE2EServerMetadata, ensureCheckoutIdentity } from './tools/e2e-server-identity';
import { resolvePerfDataPort } from './src/tests/e2e/perf-data-base';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');

const headless = process.env.LUXAR_PERF_HEADLESS === '1';
// Port overrides so perf runs can avoid ports squatted by other checkouts'
// dev servers (`reuseExistingServer` would silently reuse a foreign server —
// the stale-bundle pitfall). Defaults match the standard dev ports.
const viewerPort = Number(process.env.LUXAR_PERF_PORT ?? 5173);
// Shared with the perf-bench specs (src/tests/e2e/perf-data-base.ts) so the
// port the server boots on and the origin the specs fetch from can't drift.
const dataPort = resolvePerfDataPort();
const viewerBaseURL = `http://127.0.0.1:${viewerPort}`;
const dataBaseURL = `http://127.0.0.1:${dataPort}`;
const checkoutIdentity = ensureCheckoutIdentity(projectRoot, __dirname);
const serverMetadata = createE2EServerMetadata(checkoutIdentity, viewerBaseURL, dataBaseURL);
// Extra Chromium switches, comma-separated. Example: on a Linux/NVIDIA box
// where `--use-gl=egl` yields NO GL context and default headless falls back
// to SwiftShader, pass LUXAR_PERF_CHROME_ARGS=--use-angle=vulkan to get the
// real GPU (verified on the obsidian bench box: ANGLE/Vulkan → RTX 3070).
const extraChromeArgs = (process.env.LUXAR_PERF_CHROME_ARGS ?? '')
  .split(',')
  .map((a) => a.trim())
  .filter(Boolean);

export default defineConfig({
  metadata: { luxarE2E: serverMetadata },
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
    baseURL: viewerBaseURL,
    trace: 'off',
    screenshot: 'off',
    video: 'off',
    actionTimeout: 30000,
    navigationTimeout: 120000,
    headless,
    launchOptions: {
      args: [
        // NOTE: deliberately NOT passing the main config's `--use-gl=egl`.
        // In headless mode that flag yields NO GL context on both macOS
        // (ANGLE Metal works fine without it — verified: "Apple M4 Max")
        // and Linux/NVIDIA (needs `--use-angle=vulkan` instead, via
        // LUXAR_PERF_CHROME_ARGS), so Chrome silently falls back to the
        // SwiftShader software rasterizer — the exact trap the benches'
        // renderer-string probe exists to catch.
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
        ...extraChromeArgs,
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
        //
        // LUXAR_PERF_BROWSER=chromium falls back to the bundled Chromium. Headless
        // system Chrome on macOS was observed (2026-09, Chrome 152) to pick a 30 Hz
        // BeginFrame cadence on some launches — every forced-continuous-render
        // frame metric then reads 33.3 ms with a 96 %-idle main thread — and
        // occasionally to stop firing rAF in a hidden headless window, which
        // hangs a frame-cadence measurement until the test timeout. The bundled
        // Chromium ran the same pages at the display's 120 Hz throughout.
        channel: process.env.LUXAR_PERF_BROWSER === 'chromium' ? undefined : 'chrome',
      },
    },
  ],

  // Reuse the standard viewer + dataset dev servers.
  webServer: [
    {
      // LUXAR_PERF_PREVIEW=1 serves the PRODUCTION bundle (`pnpm build` first):
      // dev-mode ESM inflates time-to-first-paint and request counts, so the
      // viewer-audit bench must run against `vite preview`. The checkout-identity
      // endpoint is served by both servers (tools/e2e-server-identity.ts).
      command:
        process.env.LUXAR_PERF_PREVIEW === '1'
          ? `pnpm exec vite preview --host 127.0.0.1 --port ${viewerPort} --strictPort`
          : `pnpm dev --host 127.0.0.1 --port ${viewerPort} --strictPort`,
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
      command: `python3 -m http.server ${dataPort} --bind 127.0.0.1`,
      url: serverMetadata.dataIdentityURL,
      cwd: projectRoot,
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
