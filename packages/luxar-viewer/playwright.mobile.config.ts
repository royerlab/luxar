/**
 * Mobile / touch Playwright config.
 *
 * Separate from `playwright.config.ts` so the default suite keeps one desktop
 * engine and the phone/tablet specs (`src/tests/e2e/mobile/`) run under real
 * device emulation: viewport, device scale factor, `isMobile`, `hasTouch` and
 * the device user agent. `hasTouch` + `isMobile` are what make the
 * `(pointer: coarse)` / `(hover: none)` media queries match in Chromium, which
 * is what the coarse-pointer stylesheet and `utils/input-capabilities` key on —
 * the first spec asserts exactly that before anything else is trusted.
 *
 * Every project runs on **Chromium**, not the device descriptors' default
 * WebKit: both CI E2E environments standardize on Chromium, and the gestures
 * are driven through the Chrome DevTools Protocol (`Input.dispatchTouchEvent`,
 * see `src/tests/e2e/mobile/touch-helpers.ts`), which WebKit does not expose.
 * Real iOS Safari behaviour is covered by the manual device checklist, not here.
 * Chromium reports `navigator.maxTouchPoints === 1` for every emulated device,
 * so this harness does not cover behaviour gated on multiple touch points.
 *
 * Invoke via:
 *   pnpm test:e2e:mobile
 *   make test-e2e-mobile
 *
 * @module playwright.mobile.config
 */

import { defineConfig, devices } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { createE2EServerMetadata, ensureCheckoutIdentity } from './tools/e2e-server-identity';
import { e2eWorkerPlan } from './tools/e2e-workers';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');
// Same override as the main config: a sibling checkout may own 5173.
const viewerPort = process.env.LUXAR_E2E_VIEWER_PORT ?? '5173';
const viewerBaseURL = `http://127.0.0.1:${viewerPort}`;
const dataBaseURL = 'http://127.0.0.1:9000';
const checkoutIdentity = ensureCheckoutIdentity(projectRoot, __dirname);
const serverMetadata = createE2EServerMetadata(checkoutIdentity, viewerBaseURL, dataBaseURL);

/** Chromium flags shared with the main config (hardware GL where available). */
const CHROMIUM_ARGS = [
  '--use-gl=egl',
  '--ignore-gpu-blocklist',
  '--enable-webgl-developer-extensions',
  '--enable-webgl-draft-extensions',
  '--disable-web-security',
  '--no-sandbox',
  '--disable-setuid-sandbox',
];

/** Force a device descriptor onto Chromium (descriptors default to WebKit). */
function onChromium(device: (typeof devices)[string]) {
  return {
    ...device,
    browserName: 'chromium' as const,
    defaultBrowserType: 'chromium' as const,
    launchOptions: { args: CHROMIUM_ARGS },
  };
}

export default defineConfig({
  metadata: { luxarE2E: serverMetadata },
  globalSetup: path.join(__dirname, 'src/tests/e2e/global-setup.ts'),
  testDir: './src/tests/e2e',
  // Only the mobile specs; the main config's `testIgnore` excludes this folder.
  testMatch: /src\/tests\/e2e\/mobile\/.*\.spec\.ts$/,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // The desktop plan's load-sized count (tools/e2e-workers.ts), capped at 2:
  // the phone viewports are cheap, but the gestures are timing-bound (a
  // 300 ms double-tap window; camera motion applied per animation frame), and
  // a loaded box at two workers ran a 16-step drag in 5 s and read the camera
  // before any frame had applied it. The sizing is what backs that off.
  workers: process.env.CI ? 1 : Math.min(2, e2eWorkerPlan().workers),
  reporter: [['list']],
  // No visual baselines here: the desktop corpus is Linux-only and the mobile
  // specs assert geometry and behaviour, not pixels.
  ignoreSnapshots: true,

  use: {
    baseURL: viewerBaseURL,
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'off',
    actionTimeout: 15000,
    navigationTimeout: 60000,
  },

  projects: [
    { name: 'iphone-portrait', use: onChromium(devices['iPhone 14']) },
    { name: 'iphone-landscape', use: onChromium(devices['iPhone 14 landscape']) },
    { name: 'ipad', use: onChromium(devices['iPad Pro 11']) },
    { name: 'pixel', use: onChromium(devices['Pixel 7']) },
  ],

  webServer: [
    {
      command: `pnpm dev --host 127.0.0.1 --port ${viewerPort} --strictPort`,
      url: serverMetadata.viewerIdentityURL,
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      command: 'python3 packages/luxar-viewer/tools/range-http-server.py 9000 --bind 127.0.0.1',
      url: serverMetadata.dataIdentityURL,
      cwd: projectRoot,
      reuseExistingServer: !process.env.CI,
      timeout: 15000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
  ],

  outputDir: 'test-results-mobile/',
  timeout: 90000,
  expect: { timeout: 30000 },
});
