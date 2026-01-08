/**
 * Playwright configuration for README video generation.
 *
 * Usage: pnpm readme-videos
 */

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './src/tests/screenshots',
  testMatch: 'generate-readme-videos.spec.ts',
  fullyParallel: false, // Run sequentially for consistent video recording
  forbidOnly: true,
  retries: 0,
  workers: 1, // Single worker for video recording
  reporter: 'list',
  timeout: 120000, // 2 minutes per test for video recording

  use: {
    baseURL: 'http://localhost:5173',
    trace: 'off',
    video: {
      mode: 'on',
      size: { width: 1280, height: 720 },
    },
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  // Start both viewer and data server
  webServer: [
    {
      command: 'pnpm dev',
      port: 5173,
      reuseExistingServer: true,
      timeout: 60000,
    },
    {
      // Use luxar serve for proper CORS headers
      command: 'cd ../.. && hatch run luxar serve . -p 9000',
      port: 9000,
      reuseExistingServer: true,
      timeout: 30000,
    },
  ],
});
