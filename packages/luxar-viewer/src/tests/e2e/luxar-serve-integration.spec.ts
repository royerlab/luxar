/**
 * Luxar Serve Integration E2E Tests
 *
 * Tests the actual `luxar serve` CLI command that real users use,
 * verifying end-to-end data serving and viewer loading.
 *
 * These tests start a `luxar serve` subprocess, load data through it,
 * and verify the viewer works correctly. This catches CORS, Content-Type,
 * and path handling bugs that raw HTTP server tests miss.
 *
 * Skipped if `hatch` is not available (e.g., in CI environments without Python).
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForPointsLoaded, assertNoConsoleErrors } from './helpers';
import { spawn, execSync, type ChildProcess } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at module load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const SERVE_PORT = 8005;
const PROJECT_ROOT = path.resolve(__dirname, '../../../../..');
const DATASET_PATH = path.join(
  PROJECT_ROOT,
  'datasets/examples/build_example_structured.luxar.zarr'
);

/** Check if hatch CLI is available */
function isHatchAvailable(): boolean {
  try {
    execSync('hatch --version', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

test.describe('Luxar Serve Integration', () => {
  let serverProcess: ChildProcess | null = null;

  test.beforeAll(async () => {
    if (!isHatchAvailable()) {
      test.skip();
      return;
    }

    // Start luxar serve
    serverProcess = spawn(
      'hatch',
      ['run', 'luxar', 'serve', DATASET_PATH, '--port', String(SERVE_PORT)],
      { cwd: PROJECT_ROOT, stdio: 'pipe' }
    );

    // Wait for server to be ready (listen for uvicorn startup message)
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        // Server may have started without logging — try a health check
        resolve();
      }, 15000);

      const onData = (data: Buffer) => {
        const text = data.toString();
        if (text.includes('Uvicorn running') || text.includes('Application startup complete')) {
          clearTimeout(timeout);
          resolve();
        }
      };

      serverProcess!.stdout?.on('data', onData);
      serverProcess!.stderr?.on('data', onData);
      serverProcess!.on('error', (err) => {
        clearTimeout(timeout);
        reject(err);
      });
    });

    // Additional settle time for the server
    await new Promise((r) => setTimeout(r, 1000));
  });

  test.afterAll(async () => {
    if (serverProcess) {
      serverProcess.kill('SIGTERM');
      await new Promise((r) => setTimeout(r, 500));
      if (!serverProcess.killed) {
        serverProcess.kill('SIGKILL');
      }
      serverProcess = null;
    }
  });

  test('should load dataset served by luxar serve CLI', async ({ page }) => {
    if (!serverProcess) {
      test.skip();
      return;
    }

    await page.goto(`/?src=http://localhost:${SERVE_PORT}&debug`);
    await waitForLuxarReady(page, 30000);
    await waitForPointsLoaded(page, 1, 30000);

    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState();
    });

    expect(state).toBeDefined();
    expect(state.totalPoints).toBeGreaterThan(0);
    expect(state.initialized).toBe(true);

    await assertNoConsoleErrors(page, [
      /404.*spatial_index/,
      /404.*chunk_bounds/,
      /optional features/,
    ]);
  });

  test('should serve zarr metadata with correct response', async ({ page }) => {
    if (!serverProcess) {
      test.skip();
      return;
    }

    // Verify the server responds to .zattrs requests
    const response = await page.request.get(`http://localhost:${SERVE_PORT}/.zattrs`);
    expect(response.ok()).toBe(true);

    // Response should be valid JSON
    const body = await response.text();
    const parsed = JSON.parse(body);
    expect(parsed).toBeDefined();
    expect(parsed.luxar_version).toBeDefined();
  });
});
