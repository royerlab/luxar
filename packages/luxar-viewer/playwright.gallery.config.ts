/**
 * Playwright configuration for the Luxar gallery harness.
 *
 * Captures a still PNG + orbit video (WebP/WebM) for every demo in
 * scripts/gallery/manifest.json that has a dataset on disk.
 *
 * Usage:
 *   pnpm gallery
 *   GALLERY_ONLY=lorenz,desi_galaxies pnpm gallery
 *   GALLERY_DEBUG=1 pnpm gallery    # forward both servers' output to the reporter
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
// Both webServers are silent by default (see the note on the vite entry). Set
// GALLERY_DEBUG to forward their output to the reporter instead — the only way
// to see a `luxar serve` "port busy, using N instead" warning, which otherwise
// turns into a mute `Timed out waiting 90000ms from config.webServer`.
const SERVER_STDIO: 'pipe' | 'ignore' = process.env.GALLERY_DEBUG ? 'pipe' : 'ignore';

/**
 * Chromium flags that actually reach the GPU, per platform.
 *
 * This matters more than it looks: capture cost is dominated by per-frame
 * `page.screenshot()`, so falling back to SwiftShader is the difference
 * between a 55-second run and an hour that never finishes. A single orbit
 * frame of a heavy additive scene took >10 min in software.
 *
 * The previous value here was a flat `--use-gl=egl`, which reached the GPU on
 * NEITHER platform. Measured `WEBGL_debug_renderer_info.UNMASKED_RENDERER_WEBGL`:
 *
 *   flags                          macOS (M4 Max)   headless Linux + NVIDIA
 *   --use-gl=egl                   SwiftShader      SwiftShader
 *   --use-angle=vulkan             SwiftShader      NVIDIA RTX 3070
 *   --use-angle=metal              Apple Metal      n/a
 *   --use-angle=default / none     Apple Metal      SwiftShader
 *
 * No single value works on both, hence the switch. Chromium degrades to
 * SwiftShader by itself when the requested backend is unavailable (that is
 * exactly what `--use-angle=vulkan` does on macOS, which has no Vulkan), so a
 * GPU-less machine still runs — just slowly, as it did before.
 *
 * Override with `GALLERY_GL_ARGS` (space-separated, e.g. `GALLERY_GL_ARGS=""`
 * to let Chromium choose) when a machine disagrees.
 */
function gpuBackendArgs(): string[] {
  const override = process.env.GALLERY_GL_ARGS;
  if (override !== undefined) return override.split(' ').filter(Boolean);
  if (process.platform === 'darwin') return ['--use-angle=metal'];
  if (process.platform === 'linux') return ['--use-angle=vulkan', '--enable-features=Vulkan'];
  return [];
}

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
        ...gpuBackendArgs(),
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
      // Quiet by default: vite's per-request log is pure noise over a long
      // sweep. ('pipe' is safe — Playwright attaches a `data` listener to both
      // pipes either way and only forwards conditionally, so nothing can fill
      // the OS pipe buffer.) GALLERY_DEBUG=1 turns the forwarding on.
      stdout: SERVER_STDIO,
      stderr: SERVER_STDIO,
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
      // Unconditional, not gated on GALLERY_DEBUG: arbol's `aprint` does not
      // flush, and Python block-buffers stdout once Playwright hands the child a
      // pipe (it always does — the `stdout` option only gates FORWARDING). On the
      // timeout path Playwright SIGKILLs the process group, so the port-shift
      // warning would be written and then destroyed with the buffer.
      env: { PYTHONUNBUFFERED: '1' },
      stdout: SERVER_STDIO,
      stderr: SERVER_STDIO,
    },
  ],

  outputDir: 'test-results/gallery/',
});
