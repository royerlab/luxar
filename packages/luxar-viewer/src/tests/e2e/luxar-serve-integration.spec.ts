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
import * as net from 'net';
import * as path from 'path';
import { fileURLToPath } from 'url';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at module load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Ask the OS for a free ephemeral port. A fixed port (previously 8005)
 * collides with any `luxar serve` / `luxar gsplat view` a developer has
 * running, and the tests then silently talk to the WRONG server.
 */
function getFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const address = srv.address();
      if (address === null || typeof address === 'string') {
        srv.close(() => reject(new Error('Could not determine free port')));
        return;
      }
      const port = address.port;
      srv.close(() => resolve(port));
    });
  });
}

let servePort = 0;
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

    // Start luxar serve on a free ephemeral port
    servePort = await getFreePort();
    serverProcess = spawn(
      'hatch',
      ['run', 'luxar', 'serve', DATASET_PATH, '--port', String(servePort)],
      { cwd: PROJECT_ROOT, stdio: 'pipe' }
    );

    // Wait for server to be ready (listen for uvicorn startup message)
    await new Promise<void>((resolve, reject) => {
      let settled = false;
      let output = '';
      const timeout = setTimeout(() => {
        // Fallback: give up waiting for the uvicorn log line and proceed.
        // A legitimately slow-but-alive server must still be allowed to run;
        // the tests' own readiness waits (waitForLuxarReady) will surface a
        // truly-dead server. Do NOT reject here.
        if (!settled) {
          settled = true;
          resolve();
        }
      }, 15000);

      const onData = (data: Buffer) => {
        const text = data.toString();
        output += text;
        if (text.includes('Uvicorn running') || text.includes('Application startup complete')) {
          settled = true;
          clearTimeout(timeout);
          resolve();
        }
      };

      serverProcess!.stdout?.on('data', onData);
      serverProcess!.stderr?.on('data', onData);
      serverProcess!.on('error', (err) => {
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          reject(err);
        }
      });
      serverProcess!.on('exit', (code, signal) => {
        // The process exited before it became ready — fail fast with a
        // debuggable message instead of stalling for the full timeout.
        // (Once settled, this fires for the afterAll SIGTERM and is ignored.)
        if (!settled) {
          settled = true;
          clearTimeout(timeout);
          reject(
            new Error(
              `luxar serve exited before becoming ready (code=${code}, signal=${signal}).\n` +
                `Output:\n${output.trim()}`
            )
          );
        }
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

    await page.goto(`/?src=http://127.0.0.1:${servePort}&debug`);
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
    const response = await page.request.get(`http://127.0.0.1:${servePort}/.zattrs`);
    expect(response.ok()).toBe(true);

    // Response should be valid JSON
    const body = await response.text();
    const parsed = JSON.parse(body);
    expect(parsed).toBeDefined();
    expect(parsed.luxar_version).toBeDefined();
  });
});
