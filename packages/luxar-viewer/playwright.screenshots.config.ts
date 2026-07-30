/**
 * Playwright Configuration for README Screenshot Generation
 *
 * This configuration is optimized for generating high-quality screenshots
 * of Luxar demos for use in documentation.
 *
 * Usage:
 *   pnpm readme-images
 *   # or: npx playwright test --config playwright.screenshots.config.ts
 *
 * Prerequisites:
 *   - Generate demo datasets: make run-demos
 *   - Servers will be started automatically by this config
 */

import { defineConfig, devices } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';
import { ensureCheckoutIdentity } from './tools/e2e-server-identity';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '../..');
const viewerBaseURL = 'http://127.0.0.1:5173';
const checkoutIdentity = ensureCheckoutIdentity(projectRoot, __dirname);
const viewerIdentityURL = new URL(checkoutIdentity.viewerPath, viewerBaseURL).toString();

export default defineConfig({
  // Test directory - only screenshot tests
  testDir: './src/tests/screenshots',

  // Run tests serially for predictable GPU behavior
  fullyParallel: false,

  // No retries for screenshot generation
  retries: 0,

  // Single worker for GPU stability
  workers: 1,

  // Simple reporter
  reporter: [['list']],

  // Shared settings
  use: {
    // Base URL for viewer
    baseURL: viewerBaseURL,

    // No trace/video for screenshot generation
    trace: 'off',
    screenshot: 'off',
    video: 'off',

    // Longer timeouts for data loading
    actionTimeout: 30000,
    navigationTimeout: 120000,

    // GPU acceleration for realistic rendering
    launchOptions: {
      args: [
        '--use-gl=egl',
        '--ignore-gpu-blocklist',
        '--enable-webgl-developer-extensions',
        '--enable-webgl-draft-extensions',
        '--disable-web-security',
        '--no-sandbox',
        '--disable-setuid-sandbox',
      ],
    },

    // Higher resolution viewport for crisp images
    viewport: { width: 1280, height: 720 },
  },

  // Browser configuration
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
      },
    },
  ],

  // Web servers needed for screenshot generation
  webServer: [
    {
      // TypeScript viewer dev server
      command: 'pnpm dev --host 127.0.0.1 --strictPort',
      url: viewerIdentityURL,
      reuseExistingServer: true,
      timeout: 120000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // Luxar serve for demo datasets with proper CORS headers
      command: 'hatch run luxar serve . -p 9876',
      // `luxar serve` does not guarantee an HTTP success response at `/`, so
      // use a TCP readiness probe rather than polling the root URL.
      port: 9876,
      cwd: projectRoot,
      // `luxar serve` has no checkout identity endpoint, so never adopt an
      // existing process whose serving root cannot be verified.
      reuseExistingServer: false,
      timeout: 60000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
  ],

  // Output directory for test artifacts
  outputDir: 'test-results/screenshots/',

  // Longer timeout for screenshot generation
  timeout: 120000,

  // Expect timeout
  expect: {
    timeout: 30000,
  },
});
