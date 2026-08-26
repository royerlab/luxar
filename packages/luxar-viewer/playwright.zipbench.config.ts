/**
 * Playwright config for the zipped-store open benchmark.
 *
 * Separate from `playwright.config.ts` and `playwright.perf.config.ts` for one
 * reason: BOTH of those serve data with `python3 -m http.server`, which has no
 * HTTP `Range` support at all. A zipped store read over a range-less server
 * gets `200`-with-the-whole-body for every window, so a benchmark run against
 * it would measure nonsense (and, before the 206 guard landed, would have done
 * so silently). This config boots `tools/range-http-server.py` instead.
 *
 * Invoke via:
 *   pnpm bench:zip
 *
 * Fixtures must exist first — see `tools/make-zip-bench-fixtures.py` and the
 * `bench:zip:fixtures` script.
 *
 * @module playwright.zipbench.config
 */

import { defineConfig, devices } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import {
  benchDataOrigin,
  benchDataPort,
  benchFixtureDir,
  benchViewerPort,
} from './src/tests/bench/zip-bench-base';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const headless = process.env.LUXAR_BENCH_HEADLESS !== '0';
const viewerPort = benchViewerPort;
const dataPort = benchDataPort;

fs.mkdirSync(benchFixtureDir, { recursive: true });

export default defineConfig({
  testDir: './src/tests/bench',
  testMatch: /.*\.bench\.ts/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [['list']],
  // Cold-open sampling over several variants and repeats.
  timeout: 600_000,

  use: {
    ...devices['Desktop Chrome'],
    baseURL: `http://127.0.0.1:${viewerPort}`,
    headless,
    launchOptions: {
      args: [
        // `--use-gl=egl` is a Linux/ANGLE-on-X11 flag. On macOS it selects a
        // backend that never finishes initializing, so the viewer hangs before
        // `initialized` — measured while bringing this bench up. Keep the GPU
        // flags where they help and off where they hurt.
        ...(process.platform === 'linux' ? ['--use-gl=egl', '--ignore-gpu-blocklist'] : []),
        // Same rationale as the E2E config: the viewer and the data live on
        // different ports, and the fixture server is not worth a CORS dance.
        '--disable-web-security',
        '--no-sandbox',
        '--disable-setuid-sandbox',
      ],
    },
  },

  webServer: [
    {
      command: `pnpm dev --host 127.0.0.1 --port ${viewerPort} --strictPort`,
      url: `http://127.0.0.1:${viewerPort}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
    {
      command: `python3 tools/range-http-server.py ${dataPort} --bind 127.0.0.1 --directory ${benchFixtureDir}`,
      url: benchDataOrigin,
      cwd: __dirname,
      reuseExistingServer: !process.env.CI,
      timeout: 20_000,
      stdout: 'ignore',
      stderr: 'pipe',
    },
  ],

  outputDir: 'test-results-zipbench/',
});
