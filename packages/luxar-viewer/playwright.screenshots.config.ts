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
    baseURL: 'http://localhost:5173',

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
      command: 'pnpm dev',
      url: 'http://localhost:5173',
      reuseExistingServer: true,
      timeout: 120000,
      stdout: 'pipe',
      stderr: 'pipe',
    },
    {
      // Luxar serve for demo datasets with proper CORS headers
      command: 'hatch run luxar serve . -p 9000',
      url: 'http://localhost:9000',
      cwd: path.resolve(__dirname, '../..'),
      reuseExistingServer: true,
      timeout: 30000,
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
