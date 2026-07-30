/**
 * Playwright configuration for README video generation.
 *
 * Usage: pnpm readme-videos
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
  testDir: './src/tests/screenshots',
  testMatch: 'generate-readme-videos.spec.ts',
  fullyParallel: false, // Run sequentially for consistent video recording
  forbidOnly: true,
  retries: 0,
  workers: 1, // Single worker for video recording
  reporter: 'list',
  timeout: 120000, // 2 minutes per test for video recording

  use: {
    baseURL: viewerBaseURL,
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
      command: 'pnpm dev --host 127.0.0.1 --strictPort',
      url: viewerIdentityURL,
      reuseExistingServer: true,
      timeout: 60000,
    },
    {
      // Use luxar serve for proper CORS headers
      command: 'hatch run luxar serve . -p 9876',
      port: 9876,
      cwd: projectRoot,
      // `luxar serve` has no checkout identity endpoint, so never adopt an
      // existing process whose serving root cannot be verified.
      reuseExistingServer: false,
      timeout: 60000,
    },
  ],
});
