/**
 * Playwright configuration for the Luxar gallery harness.
 *
 * Captures a still PNG + orbit video (WebP/GIF) for every demo in
 * scripts/gallery/manifest.json that has a dataset on disk.
 *
 * Usage:
 *   pnpm gallery
 *   GALLERY_ONLY=lorenz,desi_galaxies pnpm gallery
 *
 * Prerequisites:
 *   - Datasets: hatch run python scripts/gallery/generate_gallery_datasets.py
 *   - ffmpeg installed (video conversion)
 */

import { defineConfig, devices } from '@playwright/test';

// Per-worker ports so multiple gallery loops can run in PARALLEL over disjoint
// demo shards without colliding. Each parallel worker exports its own
// GALLERY_VITE_PORT / GALLERY_DATA_PORT; defaults keep single-run behavior.
const VITE_PORT = Number(process.env.GALLERY_VITE_PORT ?? 5199);
const DATA_PORT = Number(process.env.GALLERY_DATA_PORT ?? 9899);
// Data server launcher. Default uses the project's own hatch env. In an isolated
// git worktree (no hatch env of its own) set GALLERY_LUXAR_BIN to a ready luxar
// binary from another checkout to avoid a slow per-worktree env-create; `luxar
// serve` only streams static zarr files by path, so any working install serves.
const SERVE = process.env.GALLERY_LUXAR_BIN ?? 'hatch run luxar';

export default defineConfig({
  testDir: './src/tests/screenshots',
  testMatch: 'generate-gallery.spec.ts',

  // Serial + single worker for deterministic GPU rendering and stable video.
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: [['list']],

  // Generous per-test budget: a heavy scene does a data load + framing/exposure
  // metering screenshots + ORBIT_FRAMES per-angle screenshots + ffmpeg, all in
  // software GL. Each screenshot of a 1080² gsplat scene is a few seconds.
  timeout: 3600000,
  expect: { timeout: 30000 },

  use: {
    baseURL: 'http://localhost:5173',
    trace: 'off',
    screenshot: 'off',
    // SQUARE aspect: gallery subjects (galaxies, volumes, blobs) are roughly
    // round, so a 16:9 frame wasted the sides. A square frame lets the subject
    // fill both dimensions — no horizontal letterboxing.
    actionTimeout: 300000,
    navigationTimeout: 120000,
    viewport: { width: 1080, height: 1080 },
    launchOptions: {
      args: [
        '--use-gl=egl',
        '--ignore-gpu-blocklist',
        '--enable-webgl-developer-extensions',
        '--enable-webgl-draft-extensions',
        '--disable-web-security',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        // Keep requestAnimationFrame running at full rate in headless — Chromium
        // throttles rAF/timers in backgrounded/occluded pages, which made the
        // orbit render loop tick slowly and the recorded video choppy/frozen.
        '--disable-background-timer-throttling',
        '--disable-renderer-backgrounding',
        '--disable-backgrounding-occluded-windows',
        '--disable-features=CalculateNativeWinOcclusion',
      ],
    },
  },

  projects: [
    {
      name: 'chromium',
      // NOTE: spread devices FIRST, then re-assert the square viewport —
      // devices['Desktop Chrome'] sets viewport 1280×720, and project-level
      // `use` wins over the top-level one, so without this the STILL screenshot
      // stayed 16:9 while the video was square.
      use: { ...devices['Desktop Chrome'], viewport: { width: 1080, height: 1080 } },
    },
  ],

  webServer: [
    {
      // Dedicated port 5199 (+ reuseExistingServer:false) so this run always
      // gets its OWN vite, isolated from any concurrent agent's dev server on
      // the default 5173 (whose file-change reloads were wiping __luxarDebug
      // mid-capture). --strictPort fails fast if 5199 is somehow busy.
      command: `pnpm exec vite --port ${VITE_PORT} --strictPort`,
      port: VITE_PORT,
      reuseExistingServer: false,
      timeout: 120000,
      // Drain to /dev/null: with 'pipe' and no consumer, vite's log output can
      // fill the OS pipe buffer over a long/heavy sweep and stall the server.
      stdout: 'ignore',
      stderr: 'ignore',
    },
    {
      command: `${SERVE} serve . -p ${DATA_PORT}`,
      // A TCP port check (not an HTTP url) — `luxar serve` does not return a
      // <400 response at `/`, so a `url` health check never resolves.
      port: DATA_PORT,
      // Serve datasets from GALLERY_DATA_ROOT if set (an isolated worktree has
      // only a `datasets` SYMLINK, which `luxar serve` refuses to follow out of
      // the served root → 403; point it at the real checkout instead). Falls
      // back to ../.. (the repo root) for a normal single-checkout run.
      cwd: process.env.GALLERY_DATA_ROOT ?? '../..',
      reuseExistingServer: false,
      timeout: 90000,
      stdout: 'ignore',
      stderr: 'ignore',
    },
  ],

  outputDir: 'test-results/gallery/',
});
